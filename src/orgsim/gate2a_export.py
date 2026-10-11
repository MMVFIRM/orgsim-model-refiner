"""Compile Gate 2A provisional learned guards/effects to the real OrgSim DSL.

This translation is generic over typed operation definitions. UNKNOWN effects
cause export refusal rather than silent omission. No live ERP I/O.
"""
from __future__ import annotations

from typing import Any
from .gate2a_lab import Operation, features_for
from .spec import validate_spec


def _ref(field: str) -> dict:
    return {'lookup':{'system':'lab','resource':'cases',
                      'id':{'path':'input.id'},'field':field}}


def _expr(op: Operation, field: str) -> dict:
    if field in op.derived:
        sources=op.derived[field]
        return {'add':[_ref(sources[0]),*({'negate':_ref(k)} for k in sources[1:])]}
    if field in op.fields and field!='record_id':return _ref(field)
    raise ValueError('Unsupported inferred field path')


def model_to_manifest(op:Operation, model:dict) -> dict:
    """Export only a complete learned transition hypothesis, not an ERP fact."""
    rules=model.get('rule')
    if not isinstance(rules,list) or not rules:
        raise ValueError('Model has no identifiable precondition hypothesis')
    effects=model.get('effects',{})
    unknown=[f for f in op.outcome_fields if f not in effects]
    if unknown:raise ValueError('Effects are unidentifiable: '+','.join(unknown))
    properties={'id':{'type':'string'}}
    for name,values in op.fields.items():
        value=values[0]
        properties[name]={'type':'boolean' if type(value) is bool else
                          'integer' if type(value) is int else
                          'number' if type(value) is float else 'string'}
    preconditions=[]
    for pred in rules:
        preconditions.append({'compare':{'left':_expr(op,pred['field']),
                                         'op':pred['op'],'right':{'literal':pred['value']}}})
    steps=[]
    for field in op.outcome_fields:
        effect=effects[field];kind=effect['kind']
        if kind=='copy':continue
        if kind=='literal':
            steps.append({'kind':'update','system':'lab','resource':'cases',
                          'id':{'path':'input.id'},'values':{field:{'literal':effect['value']}}})
        elif kind in {'subtract','add','offset'}:
            if kind=='offset':by={'literal':effect['delta']}
            elif kind=='subtract':by={'negate':_ref(effect['source'])}
            else:by=_ref(effect['source'])
            steps.append({'kind':'adjust','system':'lab','resource':'cases',
                          'id':{'path':'input.id'},'field':field,'by':by})
        else:raise ValueError('Unsupported learned expression')
    if not steps:raise ValueError('No learned state changes to execute')
    spec={
        'version':'0.1','organization':'SYNTHETIC_ORACLE_UNREVIEWED',
        'reviewed':False,
        'systems':{'lab':{
            'resources':{'cases':{'id_field':'id',
                'schema':{'type':'object','properties':properties,
                          'required':['id'],'additionalProperties':False},
                'allow':{'create':['test_agent'],'get':['test_agent'],'list':['test_agent']},
                'guards':{},'foreign_keys':[]}},
            'workflows':{'run_action':{
                'description':'Model-inferred synthetic behavior; NOT reviewed and NOT a real ERP rule',
                'roles':['test_agent'],'evidence':'hypothesis',
                'input_schema':{'type':'object','properties':{'id':{'type':'string'}},
                                'required':['id'],'additionalProperties':False},
                'preconditions':preconditions,'steps':steps,
                'result':_ref(op.outcome_fields[0]),
            }}
        }}, 'integrations':[],'scenarios':[],
        'review_notes':['Synthetic discovery hypothesis only. Cannot be presented as verified Dolibarr behavior.',
                        'All preconditions must be validated on new independent examples.']}
    validate_spec(spec,allow_draft=True)
    return spec
