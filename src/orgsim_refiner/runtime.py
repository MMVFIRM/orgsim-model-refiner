"""Pure in-process predicted simulator. Never an authoritative training oracle."""
from __future__ import annotations
from .artifact import PolicyArtifact
from .effects import EffectModel
from .types import Scope
from .util import canonical,loads,flatten

def unflatten(values):
    out={}
    for key,value in values.items():
        parts=key.split('.');cur=out
        for p in parts[:-1]:
            if p in cur and not isinstance(cur[p],dict):raise ValueError('Ambiguous state path')
            cur=cur.setdefault(p,{})
        cur[parts[-1]]=value
    return out

class PolicySimulator:
    """Use an explicit feature map (e.g. credit -> state.credit) owned by host.
    Generated trajectories have MODEL_PSEUDOLABEL provenance, so EvidenceSet
    refuses them as new truth for learning the same unknown real environment.
    """
    def __init__(self,policy:PolicyArtifact,effects:EffectModel,state:dict,feature_map:dict):
        if policy.scope!=effects.scope:raise ValueError('Policy/effect binding mismatch')
        if set(feature_map)!={f.name for f in policy.schema.fields}:raise ValueError('Feature-map mismatch')
        if any(not isinstance(p,str) or not p.startswith(('state.','action.')) or '__' in p for p in feature_map.values()):raise ValueError('Invalid feature path')
        self.policy=policy;self.effects=effects;self.state=loads(canonical(state));self.feature_map=feature_map
    def step(self,action,scope:Scope):
        sources=flatten({'state':self.state,'action':action})
        features={k:sources.get(path) for k,path in self.feature_map.items()}
        decision=self.policy.predict(features,scope)
        receipt={'decision':decision['decision'],'committed':False,'provenance':'MODEL_PSEUDOLABEL',
                 'real_erp_write':False,'before':loads(canonical(self.state)),'action':action}
        if decision['decision']=='REJECT':return {**receipt,'after':loads(canonical(self.state))}
        if decision['decision']!='APPROVE':return {**receipt,'reason':decision['reason']}
        after=self.effects.predict(self.state,action,scope)
        if after is None:return {**receipt,'decision':'UNKNOWN','reason':'unresolved_state_effects'}
        # Atomic local copy-swap: no partial changes on validation errors.
        new=unflatten(after);canonical(new);self.state=new
        return {**receipt,'after':loads(canonical(self.state)),'committed':True}
