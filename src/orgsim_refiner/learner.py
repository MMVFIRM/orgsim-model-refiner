"""Counterexample-refined finite-domain policy reconstruction.

Learns raw integer boundaries and categorical contexts; no private oracle
thresholds/flags/formulas are an input. Non-monotone intervals and Boolean
interactions are permitted. Cost can grow exponentially with relevant fields.

High-fidelity mode observes every *discovered region*, not every raw input.
A new raw holdout can share a learned logical region. This is intentionally
reported as interpolation, never as an unobserved-combination theorem.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import random
from .types import Scope,Schema,Observation
from .evidence import EvidenceSet,TrainingOracle,EvidenceError,QueryBudgetExceeded
from .artifact import Partition,PolicyArtifact
from .candidates import CandidateBank,replay,compile_partition
from .util import digest

@dataclass(frozen=True)
class LearnConfig:
    seed:int=17391
    max_cells:int=4096
    max_refinements:int=24
    audit_samples:int=128
    representatives_per_cell:int=2
    min_validation:int=32
    min_validation_per_class:int=4
    min_accuracy:float=.99
    min_coverage:float=.95
    maximum_false_accepts:int=0
    candidate_count:int=8
    def __post_init__(self):
        if not 1<=self.max_cells<=65536 or not 1<=self.max_refinements<=128:raise ValueError('Invalid resource limit')
        if not 1<=self.audit_samples<=10000 or not 1<=self.representatives_per_cell<=8:raise ValueError('Invalid audit budget')
        if not 0<self.min_accuracy<=1 or not 0<self.min_coverage<=1:raise ValueError('Invalid threshold')
        if self.min_validation<1 or self.min_validation_per_class<1 or self.maximum_false_accepts<0:raise ValueError('Invalid calibration settings')
        if not 1<=self.candidate_count<=64:raise ValueError('Invalid candidate count')
    def to_dict(self):return asdict(self)

class ActivePolicyLearner:
    def __init__(self,schema:Schema,scope:Scope,config:LearnConfig|None=None):
        self.schema=schema;self.scope=scope;self.config=config or LearnConfig()
        self.evidence=EvidenceSet(schema,scope);self.partition=Partition(schema)
        self.rng=random.Random(self.config.seed);self.trace=[];self.bank=None
        self.stop_reason='not_fitted';self.audit_passed=False;self.quarantined=False
    def _groups(self):
        groups={}
        for r in self.evidence.rows:
            groups.setdefault(self.partition.key(r.sample),[]).append(r)
        return groups
    def _conflict(self):
        for rows in self._groups().values():
            a=rows[0]
            for b in rows[1:]:
                if a.label!=b.label:return a,b
        return None
    def _get(self,x,oracle,purpose='active'):
        old=self.evidence.get(x)
        if old:return old
        row=oracle.query(x,purpose=purpose)
        return self.evidence.add(row)
    def _split_witness(self,a,b,oracle):
        """Walk between conflicting points, then exhaustively scan the one field
        that changes the outcome. Exhaustive 1D scan does NOT assume monotonicity.
        """
        cur=a
        for f in self.schema.fields:
            if cur.sample[f.name]==b.sample[f.name]:continue
            nxt={**cur.sample,f.name:b.sample[f.name]}
            row=self._get(nxt,oracle)
            if row.label!=cur.label:
                if f.kind!='int':raise EvidenceError('Categorical partition inconsistency')
                cuts=[];last=None
                for value in f.domain:
                    observed=self._get({**cur.sample,f.name:value},oracle)
                    if last is not None and observed.label!=last:cuts.append(value)
                    last=observed.label
                old=list(self.partition.cuts[f.name]);self.partition.add_cuts(f.name,cuts)
                if old==self.partition.cuts[f.name]:raise EvidenceError('Cannot separate conflicting evidence')
                self.trace.append({'action':'refine_boundary','field':f.name,'cuts':cuts,
                                   'witnesses':[a.sample_hash,b.sample_hash]})
                return
            cur=row
        raise EvidenceError('No separable witness; hidden state or contradictory oracle')
    def _resolve(self,oracle):
        n=0
        while conflict:=self._conflict():
            if self.partition.size>self.config.max_cells:raise QueryBudgetExceeded('Learned partition exceeds cell budget')
            self._split_witness(*conflict,oracle);n+=1
            if n>512:raise QueryBudgetExceeded('Boundary-search limit reached')
    def _table(self):
        table={}
        for k,rows in self._groups().items():
            ys={r.label for r in rows}
            if len(ys)==1:table[Partition.text_key(k)]=rows[0].label
        return table
    def fit(self,training:list[Observation],oracle:TrainingOracle,*,extra_candidates=()):
        if oracle.scope!=self.scope or oracle.schema!=self.schema:raise EvidenceError('Oracle binding mismatch')
        for r in training:self.evidence.add(r)
        if not training:raise ValueError('Seed training observations required')
        for r in training:
            if r.sample_hash in oracle.forbidden:raise EvidenceError('Training overlaps reserved evaluation input')
        try:
            for cycle in range(self.config.max_refinements):
                self._resolve(oracle)
                if self.partition.size>self.config.max_cells:raise QueryBudgetExceeded('Learned partition exceeds cell budget')
                # Candidate disagreement ranks the proposed, unlabeled region representatives.
                self.bank=CandidateBank(self.schema,seed=self.config.seed,count=self.config.candidate_count).fit(self.evidence.rows,extra_candidates)
                proposals=[]
                for key in self.partition.cells():
                    for j in range(self.config.representatives_per_cell):
                        x=self.partition.representative(key,None if j==0 else self.rng)
                        # Only hashes, never heldout inputs/labels, are available here.
                        for _ in range(64):
                            if digest(x) not in oracle.forbidden:break
                            x=self.partition.representative(key,self.rng)
                        if digest(x) in oracle.forbidden:continue
                        if self.evidence.get(x) is None:proposals.append(x)
                probs=self.bank.probabilities(proposals)
                order=sorted(range(len(proposals)),key=lambda i:(abs(float(probs[i])-.5),digest(proposals[i])))
                for i in order:self._get(proposals[i],oracle)
                if self._conflict():continue
                table=self._table();mismatches=0;audited=0
                # Independent random TRAINING audit queries expose false equivalence.
                for _ in range(self.config.audit_samples*5):
                    if audited>=self.config.audit_samples:break
                    x=self.schema.sample(self.rng)
                    if digest(x) in oracle.forbidden or self.evidence.get(x):continue
                    guessed=table.get(Partition.text_key(self.partition.key(x)))
                    r=self._get(x,oracle,'audit');audited+=1
                    mismatches+=r.label!=guessed
                self.trace.append({'action':'random_training_audit','cycle':cycle,'samples':audited,'mismatches':mismatches,'regions':self.partition.size})
                if mismatches==0 and audited==self.config.audit_samples:
                    self.audit_passed=True;self.stop_reason='training_audit_complete';break
            else:self.stop_reason='refinement_limit'
        except QueryBudgetExceeded as e:self.stop_reason=str(e)
        except EvidenceError as e:self.stop_reason=str(e);self.quarantined=True
        # Never let subsequent evaluation obtain new oracle labels.
        oracle.close()
        return self._artifact(oracle.calls)
    def _artifact(self,calls):
        table={} if self.quarantined or self.evidence.conflicted or self.partition.size>65536 else self._table()
        ast=None
        if table and len(table)==self.partition.size and not self._conflict():
            # Compact the observed region table. Training replay is mandatory.
            candidate=compile_partition(self.partition,table,self.config.seed)
            if candidate is not None and not replay(candidate,self.schema,self.evidence.rows):ast=candidate
        return PolicyArtifact.create(scope=self.scope.to_dict(),schema=self.schema.to_dict(),cuts=self.partition.cuts,
            table=table,ast=ast,ready=False,reason=self.stop_reason,calibration={},
            evidence={'observations':len(self.evidence.rows),'oracle_attempts':calls,
                      'training_sha256':self.evidence.fingerprint,'training_replay_mismatches':len(replay(ast,self.schema,self.evidence.rows)) if ast else None,
                      'candidate_rejections':self.bank.rejections if self.bank else [],
                      'random_audit_passed':self.audit_passed,'scope_quarantined':self.quarantined,
                      'covered_regions':len(table),'total_regions':self.partition.size,'trace':self.trace,
                      'config':self.config.to_dict(),'provenance':sorted({r.provenance for r in self.evidence.rows})},
            assumptions=['Stationary deterministic labels within tenant + operation + configuration.',
                         'Declared bounded integer/enum/bool schema captures every relevant feature.',
                         'Axis-aligned finite partition is accurate inside its empirically checked regions.',
                         'Region agreement and finite audit are not a universal correctness or safety proof.',
                         'Logical combinations queried during training are NOT unseen compositional holdouts.',
                         'No live ERP writes or production authorization.'])
    def calibrate(self,artifact:PolicyArtifact,validation:list[Observation]):
        """Read validation once for promotion only, never train/repair from it."""
        if artifact.scope!=self.scope or artifact.schema!=self.schema:raise EvidenceError('Calibration binding mismatch')
        if len({r.sample_hash for r in validation})!=len(validation):raise EvidenceError('Repeated validation record')
        for r in validation:
            if r.scope!=self.scope or r.split!='validation' or not r.usable():raise EvidenceError('Invalid calibration observation')
            if self.evidence.get(r.sample):raise EvidenceError('Calibration overlaps learning evidence')
        preds=[artifact.predict(r.sample,self.scope,advisory=True)['decision'] for r in validation]
        correct=sum(p==r.label for p,r in zip(preds,validation));n=len(validation)
        fa=sum(p=='APPROVE' and r.label=='REJECT' for p,r in zip(preds,validation))
        cov=sum(p!='UNKNOWN' for p in preds)/max(1,n)
        counts={y:sum(r.label==y for r in validation) for y in ('APPROVE','REJECT')}
        ready=bool(artifact.data['evidence']['random_audit_passed'] and not artifact.data['evidence']['scope_quarantined'] and
            n>=self.config.min_validation and min(counts.values())>=self.config.min_validation_per_class and
            correct/max(1,n)>=self.config.min_accuracy and cov>=self.config.min_coverage and fa<=self.config.maximum_false_accepts and
            artifact.data['ast'] is not None)
        metrics={'cases':n,'class_counts':counts,'correct':correct,'accuracy':correct/max(1,n),'coverage':cov,
                 'false_accepts':fa,'validation_sha256':digest([r.to_dict() for r in validation]),
                 'used_for':'promotion_only_not_learning','status':'empirically_calibrated' if ready else 'not_promoted'}
        return artifact.with_calibration(metrics,ready,'empirical_calibration_pass' if ready else 'calibration_or_training_incomplete')
