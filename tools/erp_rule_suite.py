"""Stress-test the learner on additional ERP-style rule families using the fast twin:  python tools/erp_rule_suite.py [task,task]"""
import sys,random,time
from pathlib import Path;ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from synthetic_erp_rules import TASKS
from orgsim_refiner.expr_learner import ExprPolicyLearner,ExprConfig
from orgsim_refiner.evidence import TrainingOracle
from orgsim_refiner.types import Scope,Observation
from orgsim_refiner.util import digest
names=sys.argv[1].split(',') if len(sys.argv)>1 else list(TASKS)
for name in names:
    for seed in (1,2,3):
        S,truth=TASKS[name];sc=Scope('t',name,'c')
        cb=lambda x:Observation(sc,x,'APPROVE' if truth(x) else 'REJECT',provenance='SIMULATOR_OBSERVED')
        rng=random.Random(seed*31)
        hold=[S.sample(rng) for _ in range(1500)]
        # boundary-near holdout: perturb random points by +-1 in one field
        for _ in range(1500):
            x=S.sample(rng);f=rng.choice([f for f in S.fields if f.kind=='int']);x[f.name]=min(f.high,max(f.low,x[f.name]+rng.choice([-1,0,1])));hold.append(x)
        calset=[S.sample(rng) for _ in range(120)]
        res={digest(x) for x in hold+calset}
        L=ExprPolicyLearner(S,sc,ExprConfig(seed=seed));o=TrainingOracle(S,sc,cb,budget=6000,forbidden_hashes=res)
        t=time.time();a=L.fit([cb(S.sample(rng))],o)
        cal=[Observation.from_dict({**cb(x).to_dict(),'split':'validation'}) for x in calset if digest(x) not in {r.sample_hash for r in L.evidence.rows}]
        a=L.calibrate(a,cal)
        pr=[a.predict(x,sc,advisory=True)['decision'] for x in hold]
        ok=sum(p==('APPROVE' if truth(x) else 'REJECT') for p,x in zip(pr,hold));fa=sum(p=='APPROVE' and not truth(x) for p,x in zip(pr,hold))
        print(f'{name:16s} seed{seed} q={o.calls:5d} conv={L.converged!s:5} promoted={a.ready!s:5} acc={ok/len(hold):.4f} FA={fa:3d} {time.time()-t:.0f}s stop={L.stop_reason[:40]}',flush=True)
