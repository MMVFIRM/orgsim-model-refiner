"""Certificate-gated export of bounded hypotheses to the synthetic OrgSim DSL.

The exporter never creates a Dolibarr endpoint, grants ERP rights, or sets
reviewed:true. A disputed or incomplete hypothesis is refused outright.
"""
from __future__ import annotations
from copy import deepcopy
from .gate2a_export import model_to_manifest, _expr
from .gate2a_lab import Operation
from .spec import validate_spec


def certified_synthetic_manifest(op:Operation, model:dict)->dict:
    if not model.get('audit',{}).get('ready_for_offline_execution',False):
        raise ValueError('Unverified critical dimensions; execution withheld')
    if model.get('reviewed') is True:
        raise ValueError('Reconstruction may not self-promote to reviewed')
    compact=[];multi=[]
    for atom in model.get('rule',[]):
        if atom['op']=='in':
            choices=atom.get('value',[])
            if len(choices)==1:
                compact.append({'field':atom['field'],'op':'eq','value':choices[0]})
            elif len(choices)>1:
                multi.append(atom)
            else:
                raise ValueError('Empty membership set')
        elif atom['op'] in {'eq','ge','le','gt','lt','ne'}:
            compact.append(atom)
        else:
            raise ValueError('Unsupported guard relation')
    base=model_to_manifest(op,{'rule':compact or [{'field':'record_id','op':'ne','value':'UNSAT'}],
                               'effects':model.get('effects',{})})
    workflow=base['systems']['lab']['workflows']['run_action']
    # If every field is unrestricted, its domain was fully explored and a
    # minimal, non-vacuous guards list still needs a safely supported clause.
    if not compact:
        workflow['preconditions']=[]
    for atom in multi:
        workflow['preconditions'].append({'any':[
            {'compare':{'left':_expr(op,atom['field']),'op':'eq',
                        'right':{'literal':v}}} for v in atom['value']]})
    base['review_notes']+=['Counterexample audit executed in a synthetic oracle only.',
                            'Certificate is NOT formal ERP policy assurance.',
                            'Never enable mutating live Dolibarr calls from this manifest.']
    assert base['reviewed'] is False
    validate_spec(base,allow_draft=True)
    return base
