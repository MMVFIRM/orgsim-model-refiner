"""Typed, bounded JSON expression interpreter. None means UNKNOWN.

Supported: comparisons, Boolean AND/OR/NOT/XOR, k-of-n, if, arithmetic
addition/subtraction and multiplication. No strings are evaluated as code.
"""
from __future__ import annotations
import math
from .util import canonical

MAX_NODES=20000
MAX_DEPTH=64

def validate(node, fields:set[str], expected='bool'):
    if len(canonical(node))>2_000_000: raise ValueError('AST too large')
    count=0
    def walk(n,depth=0):
        nonlocal count
        count+=1
        if count>MAX_NODES or depth>MAX_DEPTH or not isinstance(n,dict): raise ValueError('AST structure limit')
        op=n.get('op')
        if op=='const':
            if set(n)!={'op','value'}: raise ValueError('Malformed literal')
            v=n['value']
            if type(v) not in (bool,int,float,str,type(None)): raise ValueError('Invalid literal')
            if type(v) in (int,float) and (not math.isfinite(v) or abs(v)>1e15): raise ValueError('Nonfinite literal')
            if isinstance(v,str) and len(v)>1024:raise ValueError('Oversized literal')
            return 'bool' if type(v)==bool or v is None else ('number' if type(v) in (int,float) else 'string')
        if op=='field':
            if set(n)!={'op','name'} or n['name'] not in fields: raise ValueError('Unknown field')
            return 'dynamic'
        if op=='if':
            if set(n)!={'op','condition','then','else'}: raise ValueError('Malformed if')
            if walk(n['condition'],depth+1) not in ('bool','dynamic'): raise ValueError('Nonboolean if condition')
            a=walk(n['then'],depth+1);b=walk(n['else'],depth+1)
            if a!=b and 'dynamic' not in (a,b): raise ValueError('Branch types differ')
            return a
        if op=='in':
            if set(n)!={'op','arg','values'} or not isinstance(n['values'],list) or len(n['values'])>128: raise ValueError('Malformed membership')
            if any(type(v) not in (str,bool,int,float) for v in n['values']): raise ValueError('Invalid membership')
            walk(n['arg'],depth+1);return 'bool'
        if op not in ('and','or','not','xor','atleast','eq','ne','lt','le','gt','ge','add','sub','mul'):
            raise ValueError('Unrecognized AST operator')
        keys={'op','args','k'} if op=='atleast' else {'op','args'}
        if set(n)!=keys or not isinstance(n['args'],list) or not 1<=len(n['args'])<=32: raise ValueError('Malformed arguments')
        arity=len(n['args'])
        if op=='not' and arity!=1 or op in ('eq','ne','lt','le','gt','ge','sub','mul') and arity!=2: raise ValueError('Arity mismatch')
        types=[walk(x,depth+1) for x in n['args']]
        if op in ('and','or','not','xor','atleast'):
            if any(t not in ('bool','dynamic') for t in types): raise ValueError('Nonboolean operand')
            if op=='atleast' and (type(n['k'])!=int or not 0<=n['k']<=arity): raise ValueError('Bad k')
            return 'bool'
        if op in ('add','sub','mul'):
            if any(t not in ('number','dynamic') for t in types): raise ValueError('Nonnumeric arithmetic')
            return 'number'
        return 'bool'
    typ=walk(node)
    if expected!='any' and typ not in (expected,'dynamic'): raise ValueError(f'Expected {expected}, got {typ}')
    return count

def evaluate(node,sample):
    """Interpret a validated AST with strict types; missing input -> UNKNOWN."""
    op=node['op']
    if op=='const': return node['value']
    if op=='field': return sample.get(node['name'])
    if op=='if':
        c=evaluate(node['condition'],sample)
        if type(c)!=bool: return None
        return evaluate(node['then'] if c else node['else'],sample)
    if op=='in':
        x=evaluate(node['arg'],sample)
        if x is None:return None
        return any(type(x)==type(v) and x==v for v in node['values'])
    args=[evaluate(x,sample) for x in node['args']]
    if op in ('and','or','not','xor','atleast'):
        args=[x if type(x)==bool else None for x in args]
        if op=='and':return False if False in args else (True if all(x is True for x in args) else None)
        if op=='or':return True if True in args else (False if all(x is False for x in args) else None)
        if op=='atleast':
            yes=sum(x is True for x in args);unk=sum(x is None for x in args)
            return True if yes>=node['k'] else (False if yes+unk<node['k'] else None)
        if None in args:return None
        if op=='not':return not args[0]
        return sum(args)%2==1
    if None in args:return None
    if op in ('eq','ne'):
        equal=type(args[0])==type(args[1]) and args[0]==args[1]
        return equal if op=='eq' else not equal
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in args):return None
    a=args[0];b=args[1] if len(args)>1 else None
    if op=='lt':return a<b
    if op=='le':return a<=b
    if op=='gt':return a>b
    if op=='ge':return a>=b
    v=sum(args) if op=='add' else (a-b if op=='sub' else a*b)
    return v if math.isfinite(v) and abs(v)<1e18 else None

def constant(v):return {'op':'const','value':v}
def field(name):return {'op':'field','name':name}
def binary(op,a,b):return {'op':op,'args':[a,b]}
def label(node,sample):
    v=evaluate(node,sample)
    return 'APPROVE' if v is True else ('REJECT' if v is False else 'UNKNOWN')
