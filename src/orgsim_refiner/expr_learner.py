"""Counterexample-guided learner with oblique integer decision trees (v0.8).

v0.7 reconstructs *axis-aligned* regions only. Real ERP rules compare fields with
each other (stock >= qty), sum lines (qty1 + qty2 <= stock) and multiply
(outstanding + price*qty <= limit). This learner:

1. fits an exact small-integer oblique decision tree (see oblique.py) whose numeric
   splits are *learned from the data* (local linear fit rounded to integers,
   validated by exact separation) as well as drawn from a small column bank;
2. probes the exact integer boundary of every split it believes in (one step on
   each side), plus committee-disagreement and random audit points;
3. adds every counterexample to the evidence and refits, until several consecutive
   rounds of boundary probes and audits find none.

The oracle only ever receives training/audit queries; reserved evaluation inputs
(by hash) are never queried. The result is an executable AST, not text.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from . import dsl
from .oblique import ObliqueBuilder, eval_form, form_fields
from .types import Scope, Schema, Observation
from .evidence import EvidenceSet, TrainingOracle, EvidenceError, QueryBudgetExceeded
from .artifact import PolicyArtifact
from .util import digest


@dataclass(frozen=True)
class ExprConfig:
    seed: int = 20260
    seed_queries: int = 160
    max_rounds: int = 40
    probes_per_split: int = 6
    committee: int = 16
    disagreement_pool: int = 1500
    disagreement_queries: int = 24
    audit_per_round: int = 40
    clean_rounds: int = 3
    use_products: bool = True
    workers: int = 1
    min_validation: int = 32
    min_validation_per_class: int = 4
    min_accuracy: float = .99
    min_coverage: float = .95
    maximum_false_accepts: int = 0

    def to_dict(self):
        return asdict(self)


class FormBank:
    """Small bank of derived columns used (a) as extra split candidates and
    (b) as features of the sklearn committee that ranks uncertain queries."""

    def __init__(self, schema: Schema, config: ExprConfig):
        self.schema = schema
        ints = [i for i, f in enumerate(schema.fields) if f.kind == 'int']
        forms = []
        for i, j in combinations(ints, 2):
            forms.append((('f', i, 1), ('f', j, 1)))
            forms.append((('f', i, 1), ('f', j, -1)))
            forms.append((('f', i, 1), ('f', j, -2)))
            forms.append((('f', j, 1), ('f', i, -2)))
            if config.use_products:
                forms.append((('m', i, j, 1),))
        if len(ints) <= 8:
            for i, j, k in combinations(ints, 3):
                for sj, sk in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    forms.append((('f', i, 1), ('f', j, sj), ('f', k, sk)))
        self.forms = forms

    def matrix(self, samples):
        enc = np.asarray([self.schema.encode(s) for s in samples], dtype=float)
        cols = [enc]
        for form in self.forms:
            col = np.zeros(len(enc))
            for t in form:
                col += t[2] * enc[:, t[1]] if t[0] == 'f' else t[3] * enc[:, t[1]] * enc[:, t[2]]
            cols.append(col[:, None])
        return np.hstack(cols) if len(samples) else np.zeros((0, enc.shape[1] + len(self.forms)))


class ExprPolicyLearner:
    def __init__(self, schema: Schema, scope: Scope, config: ExprConfig | None = None):
        self.schema = schema
        self.scope = scope
        self.config = config or ExprConfig()
        self.bank = FormBank(schema, self.config)
        self.builder = ObliqueBuilder(schema, bank_forms=self.bank.forms, seed=self.config.seed)
        self.evidence = EvidenceSet(schema, scope)
        self.rng = random.Random(self.config.seed)
        self.trace = []
        self.stop_reason = 'not_fitted'
        self.converged = False
        self.quarantined = False
        self.model = None
        self.committee = []

    # ---- oracle plumbing --------------------------------------------------
    def _forbidden(self, x, oracle):
        return digest(x) in oracle.forbidden

    # ---- model fitting ----------------------------------------------------
    def _fit_models(self):
        rows = self.evidence.rows
        y = np.asarray([r.label == 'APPROVE' for r in rows], dtype=int)
        if len(set(y.tolist())) < 2:
            self.model = None
            self.committee = []
            return
        E = np.asarray([self.schema.encode(r.sample) for r in rows], dtype=float)
        tree = self.builder.build_best(E, y)
        if tree is None:
            raise EvidenceError('Tree cannot reproduce observations (conflict or hidden state)')
        pred = np.array([tree.predict(e) for e in E], dtype=int)
        if not np.array_equal(pred, y):
            raise EvidenceError('Tree cannot reproduce observations (conflict or hidden state)')
        self.model = tree
        X = self.bank.matrix([r.sample for r in rows])
        models = []
        n_feat = X.shape[1]
        for i in range(self.config.committee):
            kw = dict(criterion='entropy' if i % 2 else 'gini', random_state=self.config.seed + i)
            if i >= 2:
                kw['max_features'] = max(2, int(n_feat ** (.5 if i % 3 else .75)))
            m = DecisionTreeClassifier(**kw).fit(X, y)
            if np.array_equal(m.predict(X), y):
                models.append(m)
        self.committee = models

    def _predict_one(self, x):
        if self.model is None:
            return None
        return 'APPROVE' if self.model.predict(self.schema.encode(x)) else 'REJECT'

    # ---- query generation -------------------------------------------------
    def _random_sample(self):
        return self.schema.sample(self.rng)

    def _boundary_probes(self):
        if self.model is None:
            return []
        pool = [r.sample for r in self.evidence.rows] + [self._random_sample() for _ in range(300)]
        E = np.asarray([self.schema.encode(x) for x in pool], dtype=float)
        reach = {}

        def descend(node, idx):
            reach[id(node)] = idx
            if node.kind == 'leaf' or len(idx) == 0:
                return
            sub = E[idx]
            if node.kind == 'cat':
                mask = np.isin(sub[:, node.field], node.values)
            else:
                col = np.zeros(len(sub))
                for t in node.form:
                    col += t[2] * sub[:, t[1]] if t[0] == 'f' else t[3] * sub[:, t[1]] * sub[:, t[2]]
                mask = col <= node.cut
            descend(node.left, idx[mask])
            descend(node.right, idx[~mask])
        descend(self.model, np.arange(len(pool)))
        probes = []
        for node in self.model.walk():
            if node.kind == 'leaf':
                continue
            idx = reach.get(id(node), [])
            if len(idx) == 0:
                continue
            chosen = self.rng.sample(list(idx), min(self.config.probes_per_split, len(idx)))
            if node.kind == 'cat':
                f = self.schema.fields[node.field]
                for k in chosen:
                    for val in f.domain:
                        probes.append({**pool[k], f.name: val})
                continue
            fields_j = form_fields(node.form)
            for k in chosen:
                base = pool[k]
                enc = list(E[k])
                for fi in self.rng.sample(fields_j, len(fields_j)):
                    f = self.schema.fields[fi]
                    lo_best = hi_best = None
                    for val in f.domain:
                        enc[fi] = val
                        fv = eval_form(node.form, enc)
                        if fv <= node.cut and (lo_best is None or fv > lo_best[0]):
                            lo_best = (fv, val)
                        if fv > node.cut and (hi_best is None or fv < hi_best[0]):
                            hi_best = (fv, val)
                    for b in (lo_best, hi_best):
                        if b is not None:
                            probes.append({**base, f.name: b[1]})
                    break
        return probes

    def _disagreement_probes(self):
        if len(self.committee) < 2:
            return []
        pool = [self._random_sample() for _ in range(self.config.disagreement_pool)]
        obs = [r.sample for r in self.evidence.rows]
        for _ in range(self.config.disagreement_pool // 2):
            b = dict(self.rng.choice(obs))
            f = self.rng.choice(self.schema.fields)
            if f.kind == 'int':
                b[f.name] = min(f.high, max(f.low, b[f.name] + self.rng.choice([-3, -2, -1, 1, 2, 3])))
            else:
                b[f.name] = self.rng.choice(f.domain)
            pool.append(b)
        X = self.bank.matrix(pool)
        votes = np.mean([m.predict(X) for m in self.committee], axis=0)
        order = np.argsort(np.abs(votes - .5))
        return [pool[i] for i in order[:self.config.disagreement_queries] if 0 < votes[i] < 1]

    def _seed_exploration(self, oracle):
        """Stratified over categorical/bool combinations, uniform over integers."""
        cats = [f for f in self.schema.fields if f.kind != 'int']
        combos = 1
        for f in cats:
            combos *= len(f.domain)
        xs, taken = [], set()
        for k in range(self.config.seed_queries):
            for _ in range(50):          # resample around reserved / already-known inputs
                x = self._random_sample()
                if cats and combos <= 4 * self.config.seed_queries:
                    idx = k
                    for f in cats:
                        x[f.name] = f.domain[idx % len(f.domain)]
                        idx //= len(f.domain)
                h = digest(x)
                if not self._forbidden(x, oracle) and self.evidence.get(x) is None and h not in taken:
                    taken.add(h)
                    xs.append(x)
                    break
        if not xs:
            raise EvidenceError('No queryable seed inputs (everything reserved or already known)')
        rows, exhausted = oracle.query_many(xs, ['active'] * len(xs), workers=self.config.workers)
        for row in rows:
            self.evidence.add(row)
        if exhausted:
            raise QueryBudgetExceeded('Hard oracle request budget exhausted')

    # ---- main loop --------------------------------------------------------
    def fit(self, training: list[Observation], oracle: TrainingOracle):
        if oracle.scope != self.scope or oracle.schema != self.schema:
            raise EvidenceError('Oracle binding mismatch')
        for r in training:
            if r.sample_hash in oracle.forbidden:
                raise EvidenceError('Training overlaps reserved evaluation input')
            self.evidence.add(r)
        clean = 0
        try:
            self._seed_exploration(oracle)
            for rnd in range(self.config.max_rounds):
                self._fit_models()
                prev = self.model
                probes = self._boundary_probes() + self._disagreement_probes()
                probes += [self._random_sample() for _ in range(self.config.audit_per_round)]
                audit_start = len(probes) - self.config.audit_per_round
                seen, todo, purposes = set(), [], []
                for pi, x in enumerate(probes):
                    h = digest(x)
                    if h in seen or self._forbidden(x, oracle) or self.evidence.get(x) is not None:
                        continue
                    seen.add(h)
                    todo.append(x)
                    purposes.append('audit' if pi >= audit_start else 'active')
                rows, exhausted = oracle.query_many(todo, purposes, workers=self.config.workers)
                mism = 0
                for x, row in zip(todo, rows):
                    guess = self._predict_one(x) if prev is not None else None
                    self.evidence.add(row)
                    if guess is not None and row.label != guess:
                        mism += 1
                self.trace.append({'round': rnd, 'new_queries': len(rows), 'mismatches': mism,
                                   'tree_nodes': prev.size() if prev is not None else 0,
                                   'evidence': len(self.evidence.rows)})
                if exhausted:
                    raise QueryBudgetExceeded('Hard oracle request budget exhausted')
                clean = clean + 1 if (prev is not None and mism == 0) else 0
                if clean >= self.config.clean_rounds:
                    self.converged = True
                    self.stop_reason = 'counterexample_free_rounds'
                    break
            else:
                self.stop_reason = 'round_limit'
            self._fit_models()
        except QueryBudgetExceeded as e:
            self.stop_reason = str(e)
            try:
                self._fit_models()
            except EvidenceError:
                pass
        except EvidenceError as e:
            self.stop_reason = str(e)
            self.quarantined = True
        oracle.close()
        return self._artifact(oracle.calls)

    def _artifact(self, calls):
        ast = None
        if self.model is not None and not self.quarantined:
            ast = self.builder.to_ast(self.model)
            dsl.validate(ast, {f.name for f in self.schema.fields})
            if [r for r in self.evidence.rows if dsl.label(ast, r.sample) != r.label]:
                ast = None
        return PolicyArtifact.create(
            scope=self.scope.to_dict(), schema=self.schema.to_dict(), cuts={f.name: [] for f in self.schema.fields if f.kind == 'int'},
            table={}, ast=ast, ready=False, reason=self.stop_reason, calibration={},
            evidence={'observations': len(self.evidence.rows), 'oracle_attempts': calls,
                      'training_sha256': self.evidence.fingerprint,
                      'training_replay_mismatches': 0 if ast else None,
                      'candidate_rejections': [], 'random_audit_passed': self.converged,
                      'scope_quarantined': self.quarantined, 'covered_regions': 0, 'total_regions': 0,
                      'trace': self.trace, 'config': self.config.to_dict(),
                      'learner': 'oblique_tree_v0.8',
                      'provenance': sorted({r.provenance for r in self.evidence.rows})},
            assumptions=['Stationary deterministic labels within tenant + operation + configuration.',
                         'Declared bounded integer/enum/bool schema captures every relevant feature.',
                         'Rule is expressible as thresholds over small-integer linear/pairwise-product forms.',
                         'Boundary probes and counterexample-free rounds are a finite audit, not a proof.',
                         'No live ERP writes beyond the disposable-tenant test oracle; no production authorization.'])

    def calibrate(self, artifact: PolicyArtifact, validation: list[Observation]):
        """Promotion only: never trains or repairs from these labels."""
        if artifact.scope != self.scope or artifact.schema != self.schema:
            raise EvidenceError('Calibration binding mismatch')
        if len({r.sample_hash for r in validation}) != len(validation):
            raise EvidenceError('Repeated validation record')
        for r in validation:
            if r.scope != self.scope or r.split != 'validation' or not r.usable():
                raise EvidenceError('Invalid calibration observation')
            if self.evidence.get(r.sample):
                raise EvidenceError('Calibration overlaps learning evidence')
        preds = [artifact.predict(r.sample, self.scope, advisory=True)['decision'] for r in validation]
        n = len(validation)
        correct = sum(p == r.label for p, r in zip(preds, validation))
        fa = sum(p == 'APPROVE' and r.label == 'REJECT' for p, r in zip(preds, validation))
        cov = sum(p != 'UNKNOWN' for p in preds) / max(1, n)
        counts = {y: sum(r.label == y for r in validation) for y in ('APPROVE', 'REJECT')}
        c = self.config
        ready = bool(artifact.data['evidence']['random_audit_passed'] and not artifact.data['evidence']['scope_quarantined']
                     and n >= c.min_validation and min(counts.values()) >= c.min_validation_per_class
                     and correct / max(1, n) >= c.min_accuracy and cov >= c.min_coverage
                     and fa <= c.maximum_false_accepts and artifact.data['ast'] is not None)
        metrics = {'cases': n, 'class_counts': counts, 'correct': correct, 'accuracy': correct / max(1, n), 'coverage': cov,
                   'false_accepts': fa, 'validation_sha256': digest([r.to_dict() for r in validation]),
                   'used_for': 'promotion_only_not_learning', 'status': 'empirically_calibrated' if ready else 'not_promoted'}
        return artifact.with_calibration(metrics, ready, 'empirical_calibration_pass' if ready else 'calibration_or_training_incomplete')
