"""One preregistered synthetic study. Oracle rules stay in the evaluator.

Main benchmark: novel raw inputs, NOT a claim of unqueried Boolean-combination
transfer. An additional 'unseen_signature' split is reported when it exists.
"""
from __future__ import annotations
import random, math, platform,sys
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .types import Scope,Field,Schema,Observation
from .util import digest,write_json,canonical,file_digest
from .evidence import TrainingOracle,EvidenceLedger
from .learner import ActivePolicyLearner,LearnConfig
from .candidates import CandidateBank
from .retrieval import TenantMemory
from .effects import EffectModel

FAMILIES=('conjunction','disjunction','nested','xor','conditional','counting','exception_interval','signed_mix')
SCHEMA=Schema((Field('credit','int',0,31),Field('overdue','int',0,31),Field('stock','int',0,31),Field('margin','int',0,31),
               Field('segment','enum',values=('core','growth','strategic','public')),
               Field('channel','enum',values=('direct','partner','reseller','marketplace'))))

class SyntheticTenant:
    """Benchmark oracle, never serialized into learner inputs."""
    def __init__(self,seed,family):
        rng=random.Random(seed);self.seed=seed;self.family=family
        self.thresholds=[rng.randint(8,23) for _ in range(4)]
        self.segment=set(rng.sample(SCHEMA.fields[4].domain,2));self.channel=set(rng.sample(SCHEMA.fields[5].domain,2))
        self.hole=rng.randint(6,25);self.scope=Scope(f'SYNTH-{seed}','approval',f'cfg-{digest([seed,family])[:16]}')
        self.generation_calls=0
    def signature(self,x):
        a,b,c,d=(x['credit']<=self.thresholds[0],x['overdue']<=self.thresholds[1],x['stock']>=self.thresholds[2],x['margin']>=self.thresholds[3])
        e=x['segment'] in self.segment;f=x['channel'] in self.channel
        bits=(a,b,c,d,e,f)
        if self.family=='exception_interval':bits=bits+(self.hole<=x['margin']<=self.hole+1,)
        return ''.join('1' if v else '0' for v in bits)
    def truth(self,x):
        a,b,c,d,e,f=[v=='1' for v in self.signature(x)[:6]]
        family=self.family
        if family=='conjunction':v=a and b and c
        elif family=='disjunction':v=(a and b) or (c and d)
        elif family=='nested':v=(a or b) and c and (d or e or f)
        elif family=='xor':v=bool(a)^bool(b)^bool(c and d)^bool(e and f)
        elif family=='conditional':v=(a and c) if e else (b or (d and f))
        elif family=='counting':v=sum((a,b,c,d,e,f))>=4
        elif family=='exception_interval':v=(a or e or f) and not (self.hole<=x['margin']<=self.hole+1)
        elif family=='signed_mix':v=(a and not b) or (c and (d!=e)) or (f and not a)
        elif family=='diagonal':v=(x['credit']+x['overdue']<=sum(self.thresholds[:2])) and (c or e)
        else:raise ValueError('Unknown synthetic family')
        return 'APPROVE' if v else 'REJECT'
    def observe(self,x,split='active'):
        y=self.truth(x)
        # Explicitly synthetic additional state, fully observed. Independent affine effects.
        hv=int(digest(x)[:8],16)
        before={'inventory':{'units':50+hv%49},'ledger':{'processed':hv%17},'workflow':{'status':'draft'}}
        action={'quantity':1+hv%7}
        after=deepcopy(before)
        if y=='APPROVE':
            after['inventory']['units']-=action['quantity'];after['ledger']['processed']+=1;after['workflow']['status']='processed'
        return Observation(self.scope,dict(x),y,split,'SIMULATOR_OBSERVED',digest([split,x]),before,after,action)
    def public_spec(self):return {'scope':self.scope.to_dict(),'schema':SCHEMA.to_dict()}
    def sealed_spec(self):return {**self.public_spec(),'family':self.family,'seed':self.seed,'thresholds':self.thresholds,
                                  'segments':sorted(self.segment),'channels':sorted(self.channel),'hole':self.hole}

def dataset(tenant,seed,counts):
    rng=random.Random(seed);seen=set();splits={}
    for split,n in counts.items():
        if n%2:raise ValueError('Balanced sample counts must be even')
        rows=[];c=Counter()
        for _ in range(n*500):
            if len(rows)>=n:break
            x=SCHEMA.sample(rng)
            if split=='stress':
                # Evaluator-only boundary sampling; exact cutoffs are never passed
                # to a learner. Other coordinates remain independently sampled.
                name=rng.choice(('credit','overdue','stock','margin'))
                j=('credit','overdue','stock','margin').index(name)
                centers=[tenant.thresholds[j]]
                if name=='margin' and tenant.family=='exception_interval':centers += [tenant.hole,tenant.hole+1]
                x[name]=max(0,min(31,rng.choice(centers)+rng.choice((-1,0,1))))
            h=digest(x)
            if h in seen:continue
            tenant.generation_calls+=1;r=tenant.observe(x,split)
            if c[r.label]>=n//2:continue
            c[r.label]+=1;rows.append(r);seen.add(h)
        if len(rows)!=n:raise RuntimeError('Cannot balance synthetic dataset')
        rng.shuffle(rows);splits[split]=rows
    return splits

def metrics(rows,predictions,effects=None):
    if len(rows)!=len(predictions):raise ValueError('Prediction count mismatch')
    if any(p not in ('APPROVE','REJECT','UNKNOWN') for p in predictions):raise ValueError('Unknown prediction label')
    n=len(rows);correct=sum(r.label==p for r,p in zip(rows,predictions));answered=sum(p!='UNKNOWN' for p in predictions)
    fa=sum(r.label=='REJECT' and p=='APPROVE' for r,p in zip(rows,predictions))
    fr=sum(r.label=='APPROVE' and p=='REJECT' for r,p in zip(rows,predictions))
    pos=sum(r.label=='APPROVE' for r in rows);neg=n-pos
    ta=sum(r.label==p=='APPROVE' for r,p in zip(rows,predictions));tr=sum(r.label==p=='REJECT' for r,p in zip(rows,predictions))
    exact=0;eligible=0;full=tr
    if effects:
        from .util import flatten
        for r,p in zip(rows,predictions):
            if r.label==p=='APPROVE':
                eligible+=1;v=effects.predict(r.before,r.action,r.scope)
                exact+=v==flatten(r.after);full+=v==flatten(r.after)
    return {'cases':n,'correct':correct,'accuracy_all_cases':correct/max(1,n),'accuracy_on_answered':correct/answered if answered else None,
            'coverage':answered/max(1,n),'abstentions':n-answered,'false_accepts':fa,'false_rejects':fr,
            'true_approvals':pos,'true_rejections':neg,'false_accept_rate':fa/neg if neg else None,
            'balanced_accuracy':.5*(ta/pos+tr/neg) if pos and neg else None,
            'effect_cases':eligible,'effect_exact':exact,'effect_exact_rate':exact/eligible if eligible else None,
            'end_to_end_accuracy':full/n if effects and n else None}

def paired_bootstrap(left,right,seed=72,n_boot=5000):
    dif=np.asarray(right)-np.asarray(left)
    rng=np.random.default_rng(seed)
    if not len(dif):return None
    vals=dif[rng.integers(0,len(dif),(n_boot,len(dif)))].mean(axis=1)
    return {'delta':float(dif.mean()),'ci95':[float(x) for x in np.quantile(vals,[.025,.975])],
            'resampling_unit':'tenant','tenant_blocks':len(dif),'resamples':n_boot,'descriptive_only':True}

def run_study(out:Path,*,seed=771902,tenants=24,train=32,validation=128,test=256,stress=256,budget=8192,max_cells=4096,audit_samples=128,model=None,backend=None,model_cases=8):
    if model is not None and (backend is None or model_cases<2 or model_cases%2 or model_cases>test):raise ValueError('Model comparison needs a bounded backend and an even subset size')
    out=Path(out)
    if out.exists():raise FileExistsError('Study output exists: never overwrite a disclosed run')
    if not 1<=tenants<=200 or any(x<4 or x%2 for x in (train,validation,test,stress)):raise ValueError('Invalid study dimensions')
    out.mkdir(parents=True)
    config=LearnConfig(seed=seed,max_cells=max_cells,audit_samples=audit_samples)
    spec={'format':'orgsim-refiner-one-study-v1','seed':seed,'tenants':tenants,'families':list(FAMILIES),
          'counts':{'train':train,'validation':validation,'holdout':test,'stress':stress},'oracle_query_budget_per_tenant':budget,
          'learner':asdict(config),'primary':'raw_input_generalization_with_explicitly_reported_region_coverage',
          'no_claim_of_unqueried_condition_combination_generalization':True,
          'stress_sampling':'One numeric feature sampled within one integer of its evaluator-only true threshold/interval boundary; balanced labels.',
          'tree_partition_reference':'Shares the learned feature partition and compiler; representation control, not an independent discovery method.',
          'source_sha256':{p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')},
          'environment':{'python':sys.version,'platform':platform.platform()},
          'model_comparison':{'enabled':model is not None,'model':model,'cases_per_tenant':model_cases if model else 0,'arms':3,'max_output_tokens':backend.cache_identity().get('max_output_tokens') if backend else None},
          'dolibarr_calls':0,'externally_timestamped':False,'resource_controls':{'tenants':2,'family':'diagonal','query_budget':64,'max_cells':64}}
    write_json(out/'STUDY_FREEZE.json',spec)
    allresults=[]
    for index in range(tenants):
        tenant=SyntheticTenant(seed+index*7919,FAMILIES[index%len(FAMILIES)])
        splits=dataset(tenant,seed+index*12539,spec['counts'])
        env=out/f'tenant_{index+1:03d}';env.mkdir()
        sealed_sha=write_json(env/'SEALED_EVALUATOR.json',{'oracle':tenant.sealed_spec(),'splits':{k:[r.to_dict() for r in v] for k,v in splits.items()}})
        public_sha=write_json(env/'PUBLIC_SCHEMA.json',tenant.public_spec())
        ledger=EvidenceLedger(env/'EVIDENCE.sqlite')
        ledger.append('tenant_data_sealed',{'sealed_sha256':sealed_sha,'public_schema_sha256':public_sha,
            'study_sha256':digest(spec)})
        forbidden={r.sample_hash for k,v in splits.items() if k!='train' for r in v}
        oracle=TrainingOracle(SCHEMA,tenant.scope,tenant.observe,budget=budget,forbidden_hashes=forbidden,ledger=ledger)
        learner=ActivePolicyLearner(SCHEMA,tenant.scope,LearnConfig(**{**asdict(config),'seed':seed+index}))
        policy=learner.fit(splits['train'],oracle)
        policy=learner.calibrate(policy,splits['validation']);policy.save(env/'POLICY.json')
        learned=learner.evidence.rows
        with (env/'TRAINING_EVIDENCE.jsonl').open('xb') as f:
            for r in learned:f.write((canonical(r.to_dict())+'\n').encode())
        effects=EffectModel(tenant.scope).fit(learned,splits['validation']);effects_sha=write_json(env/'EFFECTS.json',effects.to_dict())
        ledger.append('learned_artifacts_frozen',{'policy_sha256':policy.fingerprint,'effects_sha256':effects_sha,
              'training_file_sha256':file_digest(env/'TRAINING_EVIDENCE.jsonl')})
        passive=CandidateBank(SCHEMA,seed=seed+index).fit(splits['train'])
        matched=CandidateBank(SCHEMA,seed=seed+index).fit(learned)
        memory=TenantMemory(SCHEMA,learned)
        from .dsl import label
        candidates={'tree_passive':passive.smallest(),'tree_matched_active':matched.smallest(),'tree_partition_reference':policy.data['ast']}
        seen_signatures={tenant.signature(r.sample) for r in learned}
        preds={};truth=[];splitmetrics={}
        for split in ('holdout','stress'):
            rows=splits[split]
            pp={'active_executable':[policy.predict(r.sample,r.scope)['decision'] for r in rows],
                'always_reject':['REJECT']*len(rows)}
            for name,ast in candidates.items():pp[name]=[label(ast,r.sample) if ast else 'UNKNOWN' for r in rows]
            # Matched evidence, hard tenant filtering, normalized raw-feature distance.
            pp['knn_matched_active']=[memory.vote(r.scope,r.sample,5) for r in rows]
            preds[split]={name:values for name,values in pp.items()}
            splitmetrics[split]={'novelty':{'new_raw_inputs':sum(r.sample_hash not in {x.sample_hash for x in learned} for r in rows),
                'cases':len(rows),'oracle_signature_unseen_in_learning':sum(tenant.signature(r.sample) not in seen_signatures for r in rows),
                'privileged_evaluator_only':True,'warning':'Logical regions/condition signatures can have been queried even when raw inputs are new.'}}
            truth +=[{'split':split,'sample_sha256':r.sample_hash,'truth':r.label} for r in rows]
        model_result=None;model_rows=[]
        if model is not None:
            from .model_bridge import compare_models
            for y in ('APPROVE','REJECT'):
                model_rows += [r for r in splits['holdout'] if r.label==y][:model_cases//2]
            if policy.ready:
                model_result=compare_models(SCHEMA,tenant.scope,policy,learned,[r.sample for r in model_rows],backend,model=model)
            else:
                model_result={'predictions':{a:['UNKNOWN']*len(model_rows) for a in ('llm_zero_shot','llm_tenant_retrieval','llm_executable_context','program_verified')},
                              'status':'not_executed_policy_not_promoted','receipts':[]}
            write_json(env/'MODEL_PREDICTIONS.json',model_result)
            ledger.append('model_predictions_frozen',{'sha256':digest(model_result)})
        # Prediction commitment precedes result-file publication. Evaluation labels are
        # owned by this evaluator, not sent through learner or model interfaces.
        ph=write_json(env/'PREDICTIONS.json',preds)
        ledger.append('predictions_frozen',{'policy_sha256':policy.fingerprint,'predictions_sha256':ph})
        for split in ('holdout','stress'):
            splitmetrics[split].update({name:metrics(splits[split],v,effects) for name,v in preds[split].items()})
        model_metrics={a:metrics(model_rows,ps) for a,ps in model_result['predictions'].items()} if model_result else {}
        ledger.append('scoring_completed',{'scored_splits':['holdout','stress']})
        result={'scope':tenant.scope.to_dict(),'family':tenant.family,'ready':policy.ready,'stop_reason':policy.data['reason'],
                'training_queries':oracle.calls,'initial_training_examples':train,'dataset_generation_oracle_calls':tenant.generation_calls,
                'regions':policy.partition.size,'covered_regions':len(policy.data['table']),
                'policy_sha256':policy.fingerprint,'sealed_evaluator_sha256':sealed_sha,'effects_sha256':effects_sha,'metrics':splitmetrics,'effect_certificate':effects.certificate,
                'model_metrics':model_metrics,'model_subset_sample_hashes':[r.sample_hash for r in model_rows],'evidence':ledger.verify()}
        write_json(env/'RESULT.json',result);allresults.append(result)
        print(f"tenant {index+1}/{tenants}: {tenant.family}; promoted={policy.ready}; calls={oracle.calls}; regions={policy.partition.size}; test={splitmetrics['holdout']['active_executable']['accuracy_all_cases']:.4f}",flush=True)
    arms=list(allresults[0]['metrics']['holdout'].keys());arms.remove('novelty')
    aggregate={}
    for split in ('holdout','stress'):
        aggregate[split]={}
        for arm in arms:
            mm=[r['metrics'][split][arm] for r in allresults]
            totals={k:sum(m[k] for m in mm) for k in ('cases','correct','abstentions','false_accepts','false_rejects','effect_cases','effect_exact','true_rejections','true_approvals')}
            n=totals['cases'];totals.update(accuracy_all_cases=totals['correct']/n,coverage=1-totals['abstentions']/n,
                false_accept_rate=totals['false_accepts']/totals['true_rejections'],
                effect_exact_rate=totals['effect_exact']/totals['effect_cases'] if totals['effect_cases'] else None,
                per_tenant_accuracy=[m['accuracy_all_cases'] for m in mm])
            aggregate[split][arm]=totals
    model_aggregate={}
    if model is not None:
        for arm in ('llm_zero_shot','llm_tenant_retrieval','llm_executable_context','program_verified'):
            ms=[r['model_metrics'][arm] for r in allresults]
            totals={k:sum(m[k] for m in ms) for k in ('cases','correct','abstentions','false_accepts','false_rejects')}
            n=totals['cases']
            totals.update(accuracy_all_cases=totals['correct']/n,coverage=1-totals['abstentions']/n,
                          per_tenant_accuracy=[m['accuracy_all_cases'] for m in ms],
                          effect_prediction_claim=False)
            model_aggregate[arm]=totals
        write_json(out/'PROVIDER_RECEIPTS.json',{'usage':backend.usage(),'receipts':backend.receipts,
            'scope':'Requests actually attempted in this study; no price assumptions.'})
    comparisons={arm:paired_bootstrap(aggregate['holdout'][arm]['per_tenant_accuracy'],aggregate['holdout']['active_executable']['per_tenant_accuracy']) for arm in arms if arm!='active_executable'}
    from .controls import run_controls
    control_result=run_controls(out/'controls',seed+999983)
    report={'format':'orgsim-refiner-integrated-results-v1','study_sha256':digest(spec),'tenants':tenants,
            'promoted_tenants':sum(r['ready'] for r in allresults),'active_queries':sum(r['training_queries'] for r in allresults),
            'scores':aggregate,'paired_comparisons':comparisons,'model_scores':model_aggregate,'provider_usage':backend.usage() if backend else {'provider_attempts':0},'new_llm_calls':backend.calls if backend else 0,'dolibarr_calls':0,
            'unseen_oracle_signature_cases':sum(r['metrics'][s]['novelty']['oracle_signature_unseen_in_learning'] for r in allresults for s in ('holdout','stress')),
            'scope':'New synthetic raw inputs; empirical axis-partition interpolation, NOT autonomous ERP reconstruction. LLM efficacy is measured only if explicit provider arm executed.',
            'model_tenants_not_executed_due_to_policy':sum(not r['ready'] for r in allresults) if model else 0,
            'tenants_detail':allresults,'resource_controls':control_result}
    rh=write_json(out/'REPORT.json',report)
    artifact_files={str(p.relative_to(out)).replace('\\','/'):file_digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.sqlite',) and not p.name.endswith(('-wal','-shm'))} 
    write_json(out/'FINAL_FREEZE.json',{'study_sha256':digest(spec),'report_sha256':rh,'artifact_file_sha256':artifact_files,'externally_timestamped':False,'new_llm_calls':backend.calls if backend else 0,'dolibarr_calls':0})
    return report
