"""Exact, small-integer oblique decision trees with an executable AST export.

A split is one of
  * cat : field-index membership (bool / enum values) - exported as `in`
  * num : integer form  sum(c*field) + sum(c*field_i*field_j)  <=  cut

Numeric forms come from (a) a small bank of axis/pairwise columns and (b) a local
linear separator fitted on the node's own data over raw + pairwise-product features,
with weights rounded to small integers and *validated by exact separation of the
node data* (never trusted as fitted). Everything is deterministic given the seed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from . import dsl
from .types import Schema


# A form is a tuple of terms: ('f', i, c)  -> c * v[i]      ('m', i, j, c) -> c * v[i] * v[j]
def eval_form(form, v):
    s = 0
    for t in form:
        s += t[2] * v[t[1]] if t[0] == 'f' else t[3] * v[t[1]] * v[t[2]]
    return s


def form_fields(form):
    out = []
    for t in form:
        out += [t[1]] if t[0] == 'f' else [t[1], t[2]]
    return sorted(set(out))


def form_ast(form, schema: Schema):
    node = None
    for t in form:
        if t[0] == 'f':
            base = dsl.field(schema.fields[t[1]].name)
            c = t[2]
        else:
            base = dsl.binary('mul', dsl.field(schema.fields[t[1]].name), dsl.field(schema.fields[t[2]].name))
            c = t[3]
        term = base if abs(c) == 1 else dsl.binary('mul', dsl.constant(abs(c)), base)
        if node is None:
            node = term if c > 0 else dsl.binary('mul', dsl.constant(-1), term)
        else:
            node = dsl.binary('add' if c > 0 else 'sub', node, term)
    return node


@dataclass
class Node:
    kind: str = 'leaf'            # leaf | cat | num
    label: bool = False
    field: int = -1               # cat: field index
    values: tuple = ()            # cat: encoded values going LEFT
    form: tuple = ()              # num
    cut: int = 0                  # num: form <= cut goes LEFT
    left: 'Node | None' = None
    right: 'Node | None' = None
    n: int = 0

    def route(self, enc):
        node = self
        while node.kind != 'leaf':
            if node.kind == 'cat':
                go = enc[node.field] in node.values
            else:
                go = eval_form(node.form, enc) <= node.cut
            node = node.left if go else node.right
        return node

    def predict(self, enc):
        return self.route(enc).label

    def walk(self):
        yield self
        if self.kind != 'leaf':
            yield from self.left.walk()
            yield from self.right.walk()

    def size(self):
        return sum(1 for _ in self.walk())


def _gini_cost(y_left_pos, n_left, y_right_pos, n_right):
    def g(pos, n):
        if n == 0:
            return 0.0
        p = pos / n
        return n * 2 * p * (1 - p)
    return g(y_left_pos, n_left) + g(y_right_pos, n_right)


def _best_threshold(col, y):
    """Best <= threshold on a numeric column. Returns (cost, cut) or None."""
    order = np.argsort(col, kind='stable')
    c = col[order]
    ys = y[order]
    n = len(c)
    pos_cum = np.cumsum(ys)
    total_pos = pos_cum[-1]
    idx = np.nonzero(c[1:] != c[:-1])[0]          # split after position i
    if len(idx) == 0:
        return None
    nl = idx + 1
    pl = pos_cum[idx]
    nr = n - nl
    pr = total_pos - pl
    with np.errstate(divide='ignore', invalid='ignore'):
        gl = np.where(nl > 0, nl * 2 * (pl / nl) * (1 - pl / nl), 0.0)
        gr = np.where(nr > 0, nr * 2 * (pr / nr) * (1 - pr / nr), 0.0)
    cost = gl + gr
    k = int(np.argmin(cost))
    lo, hi = c[idx[k]], c[idx[k] + 1]
    cut = int(np.floor((lo + hi) / 2.0))
    if cut >= hi:
        cut = int(lo)
    return float(cost[k]), cut


def _best_pure_threshold(col, y):
    """Cut giving the largest *pure* child (all one class). Returns (covered, cut) or None."""
    order = np.argsort(col, kind='stable')
    c = col[order]
    ys = y[order]
    n = len(c)
    best = None
    # pure prefix
    diff = np.nonzero(ys != ys[0])[0]
    end = int(diff[0]) if len(diff) else n          # ys[:end] pure
    k = end
    while k > 0 and k < n and c[k - 1] == c[k]:
        k -= 1                                       # do not split equal values
    if 0 < k < n:
        best = (k, int(np.floor((c[k - 1] + c[k]) / 2.0)) if np.floor((c[k - 1] + c[k]) / 2.0) < c[k] else int(c[k - 1]))
    # pure suffix
    diff = np.nonzero(ys != ys[-1])[0]
    start = int(diff[-1]) + 1 if len(diff) else 0   # ys[start:] pure
    k = start
    while 0 < k < n and c[k - 1] == c[k]:
        k += 1
    if 0 < k < n and (best is None or n - k > best[0]):
        cut = int(np.floor((c[k - 1] + c[k]) / 2.0))
        if cut >= c[k]:
            cut = int(c[k - 1])
        best = (n - k, cut)
    return best


class ObliqueBuilder:
    def __init__(self, schema: Schema, *, bank_forms=(), max_coef_scale=100, seed=0, max_depth=24, svm_c=200.0):
        self.schema = schema
        self.bank_forms = list(bank_forms)
        self.max_coef_scale = max_coef_scale
        self.seed = seed
        self.max_depth = max_depth
        self.svm_c = svm_c
        self.int_idx = [i for i, f in enumerate(schema.fields) if f.kind == 'int']
        self.cat_idx = [i for i, f in enumerate(schema.fields) if f.kind != 'int']
        self.pairs = list(combinations(self.int_idx, 2))

    # -- feature construction -------------------------------------------------
    def _linear_features(self, E):
        cols, terms = [], []
        for i in self.int_idx:
            cols.append(E[:, i])
            terms.append(('f', i))
        for i, j in self.pairs:
            cols.append(E[:, i] * E[:, j])
            terms.append(('m', i, j))
        return (np.stack(cols, axis=1) if cols else np.zeros((len(E), 0))), terms

    def _oblique_candidates(self, E, y):
        """Rounded-integer separators from a local linear fit."""
        from sklearn.svm import LinearSVC
        F, terms = self._linear_features(E)
        if F.shape[1] == 0 or len(set(y.tolist())) < 2 or len(y) < 4:
            return []
        keep = F.std(axis=0) > 0
        if keep.sum() == 0:
            return []
        mu = F[:, keep].mean(axis=0)
        sd = F[:, keep].std(axis=0)
        Z = (F[:, keep] - mu) / sd
        out, seen = [], set()
        try:
            clf = LinearSVC(C=self.svm_c, dual='auto', max_iter=20000, random_state=self.seed)
            clf.fit(Z, y)
        except Exception:
            return []
        w_std = clf.coef_[0]
        w = np.zeros(F.shape[1])
        w[np.nonzero(keep)[0]] = w_std / sd
        if np.max(np.abs(w)) == 0:
            return []
        w = w / np.max(np.abs(w))
        # sparsify: drop tiny weights before rounding (threshold ladder)
        for m in (1, 2, 3, 4, 6, 8, 12, 25, 50, 100):
            if m > self.max_coef_scale:
                break
            for drop in (0.0, 0.004, 0.08, 0.2):
                wi = np.where(np.abs(w) < drop, 0.0, w)
                r = np.round(wi * m).astype(int)
                if not r.any():
                    continue
                form = []
                for ti, c in enumerate(r):
                    if c == 0:
                        continue
                    t = terms[ti]
                    form.append(('f', t[1], int(c)) if t[0] == 'f' else ('m', t[1], t[2], int(c)))
                form = tuple(form)
                if form not in seen:
                    seen.add(form)
                    out.append(form)
        return out

    # -- tree building --------------------------------------------------------
    def build(self, E, y, depth=0, tol=0.0, mode='gini'):
        n = len(y)
        pos = int(y.sum())
        node = Node(n=n)
        if pos == 0 or pos == n or depth >= self.max_depth:
            node.label = bool(pos * 2 >= n) if pos not in (0, n) else bool(pos == n)
            return node
        parent_cost = n * 2 * (pos / n) * (1 - pos / n)
        cands = []   # (cost, complexity, kind, payload)

        # categorical / bool membership splits
        for i in self.cat_idx:
            vals = sorted(set(E[:, i].astype(int).tolist()))
            if len(vals) < 2:
                continue
            cand = [(v,) for v in vals] if len(vals) > 2 else [(vals[0],)]
            for left in cand:
                mask = np.isin(E[:, i], left)
                nl, nr = int(mask.sum()), n - int(mask.sum())
                if nl == 0 or nr == 0:
                    continue
                cost = _gini_cost(int(y[mask].sum()), nl, int(y[~mask].sum()), nr)
                pure = max(nl if int(y[mask].sum()) in (0, nl) else 0, nr if int(y[~mask].sum()) in (0, nr) else 0)
                cands.append((cost, 1, 'cat', (i, left), pure))

        # numeric: raw columns, bank forms, local oblique forms
        forms = [((('f', i, 1),)) for i in self.int_idx] + self.bank_forms
        forms = list(forms) + self._oblique_candidates(E, y)
        for form in forms:
            col = np.zeros(n)
            for t in form:
                col += t[2] * E[:, t[1]] if t[0] == 'f' else t[3] * E[:, t[1]] * E[:, t[2]]
            comp = sum(abs(t[2] if t[0] == 'f' else t[3]) + (0 if t[0] == 'f' else 1) for t in form) + len(form)
            if mode == 'pure':
                rp = _best_pure_threshold(col, y)
                if rp is not None:
                    covered, pcut = rp
                    m = col <= pcut
                    nl = int(m.sum())
                    cost = _gini_cost(int(y[m].sum()), nl, int(y[~m].sum()), n - nl)
                    cands.append((cost, comp, 'num', (form, pcut), covered))
                continue
            r = _best_threshold(col, y)
            if r is None:
                continue
            cost, cut = r
            cands.append((cost, comp, 'num', (form, cut), 0))

        if not cands:
            node.label = bool(pos * 2 >= n)
            return node
        if mode == 'pure':
            top = max(c[4] for c in cands)
            if top > 0:
                admissible = [c for c in cands if c[4] >= top * (1.0 - tol)]
                best = min(admissible, key=lambda c: (c[1], -c[4], c[0]))
            else:   # no rule-like split here: fall back to impurity
                floor = min(c[0] for c in cands)
                best = min([c for c in cands if c[0] <= floor + 1e-9], key=lambda c: (c[1], c[0]))
        else:
            floor = min(c[0] for c in cands)
            admissible = [c for c in cands if c[0] <= floor + tol * parent_cost + 1e-9]
            best = min(admissible, key=lambda c: (c[1], c[0]))
        _, _, kind, payload = best[:4]
        if kind == 'cat':
            i, left = payload
            mask = np.isin(E[:, i], left)
            node.kind, node.field, node.values = 'cat', i, tuple(int(v) for v in left)
        else:
            form, cut = payload
            col = np.zeros(n)
            for t in form:
                col += t[2] * E[:, t[1]] if t[0] == 'f' else t[3] * E[:, t[1]] * E[:, t[2]]
            mask = col <= cut
            node.kind, node.form, node.cut = 'num', form, int(cut)
        node.left = self.build(E[mask], y[mask], depth + 1, tol, mode)
        node.right = self.build(E[~mask], y[~mask], depth + 1, tol, mode)
        return node

    @staticmethod
    def description_length(node):
        total = 0
        for nd in node.walk():
            total += 1
            if nd.kind == 'num':
                total += sum(abs(t[2] if t[0] == 'f' else t[3]) + (0 if t[0] == 'f' else 1) + 1 for t in nd.form)
        return total

    def build_best(self, E, y, tolerances=(0.0, 0.1, 0.3), pure_tolerances=(0.0, 0.1)):
        """Occam selection: smallest description length among consistent trees."""
        best = None
        variants = [('gini', tol) for tol in tolerances] + [('pure', tol) for tol in pure_tolerances]
        for mode, tol in variants:
            tree = self.build(E, y, 0, tol, mode)
            if not all(bool(tree.predict(e)) == bool(l) for e, l in zip(E, y)):
                continue
            score = (self.description_length(tree), 0 if mode == 'pure' else 1, tol)
            if best is None or score < best[0]:
                best = (score, tree)
        return None if best is None else best[1]

    # -- export ---------------------------------------------------------------
    def to_ast(self, node: Node):
        if node.kind == 'leaf':
            return dsl.constant(bool(node.label))
        if node.kind == 'cat':
            f = self.schema.fields[node.field]
            cond = {'op': 'in', 'arg': dsl.field(f.name), 'values': [f.domain[v] for v in node.values]}
        else:
            cond = dsl.binary('le', form_ast(node.form, self.schema), dsl.constant(int(node.cut)))
        return {'op': 'if', 'condition': cond, 'then': self.to_ast(node.left), 'else': self.to_ast(node.right)}
