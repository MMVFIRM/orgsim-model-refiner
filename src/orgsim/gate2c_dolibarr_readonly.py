"""Gate 2C: carefully bounded, read-only Dolibarr differential discovery.

Only approved GET routes, loopback hosts, no redirects, no arbitrary paths,
no state mutations, no auth values or raw response values written to reports.
The experiment deliberately varies one list parameter per request.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from hashlib import sha256

from .dolibarr_gate import DolibarrClient,_local_host,_shape
from .gate2b_readonly import build_readonly_plan


_VARIANTS=[
    ('thirdparties_limit1','/thirdparties?limit=1','thirdparties_list'),
    ('products_limit1','/products?limit=1','products_list'),
    ('orders_limit1','/orders?limit=1','orders_list'),
    ('invoices_limit1','/invoices?limit=1','invoices_list'),
    ('thirdparties_page1','/thirdparties?limit=1&page=1','thirdparties_limit1'),
    ('products_page1','/products?limit=1&page=1','products_limit1'),
    ('thirdparties_properties','/thirdparties?limit=1&properties=id,name,client','thirdparties_limit1'),
    ('products_properties','/products?limit=1&properties=id,ref,label','products_limit1'),
    ('products_ids_only','/products?limit=1&ids_only=1','products_limit1'),
    ('orders_page1','/orders?limit=1&page=1','orders_limit1'),
    ('invoices_page1','/invoices?limit=1&page=1','invoices_limit1'),
]


def bounded_differential_plan(fixtures:dict|None=None,*,budget:int=16)->list[dict]:
    if type(budget) is not int or not 4<=budget<=20:raise ValueError('Budget must be 4..20')
    base=build_readonly_plan(fixtures,limit=3)
    plan=[dict(p,paired_with=None) for p in base[:4]]
    # Interleave one factor interventions and fixture details while keeping
    # list anchor observations available. Fixed, deterministic priority order.
    for name,path,anchor in _VARIANTS:
        plan.append({'id':name,'method':'GET','path':path,'category':'discovery_variant',
                     'paired_with':anchor})
    plan.extend(dict(p,paired_with=None) for p in base[4:])
    return plan[:budget]


def differential_readonly(client:DolibarrClient,plan:list[dict],*,execute:bool=False,
                          outdir:str|Path='build/gate2c_dolibarr')->dict:
    if not client.loopback:raise ValueError('Only loopback Dolibarr is supported')
    approved=bounded_differential_plan(budget=20)
    # Caller can use approved fixed probes or bounded fixture details, never
    # invent arbitrary endpoints, query strings, or mutating actions.
    allowed_paths={p['path'] for p in approved}
    for p in plan:
        path=p.get('path','')
        if (p.get('method')!='GET' or not isinstance(path,str)
            or not path.startswith('/') or '\\' in path or '//' in path
            or '?' in path and ('DOLAPIKEY' in path.upper() or 'KEY=' in path.upper())):
            raise ValueError('Only bounded GET probes are allowed')
        if path not in allowed_paths:
            # Fixture detail paths are permitted only if the exact path is
            # reproduced by the existing validated fixture-plan constructor.
            from .gate2b_readonly import _ID, _REF
            pieces=path.split('/')
            good=(len(pieces)==3 and pieces[1] in {'thirdparties','products','orders','invoices'}
                  and _ID.fullmatch(pieces[2]) is not None)
            good=good or (len(pieces)==4 and pieces[1:3]==['products','ref']
                             and _REF.fullmatch(pieces[3]) is not None)
            if not good:raise ValueError('Unapproved fixture route')
    if len(plan)>20:raise ValueError('Probe budget exceeded')
    report={'format':'orgsim-gate2c-dolibarr-readonly-v1','phase':'discovery',
            'evidence':'LIVE_DOLIBARR' if execute else 'DRY_RUN',
            'operations_configured':len(plan),'operations_executed':0,
            'writes_executed':0,'raw_values_saved':False,'credentials_saved':False,
            'probes':[],'comparisons':[]}
    memory={}
    for item in plan:
        row={'id':item['id'],'method':'GET','category':item['category'],
             'comparison_anchor':item.get('paired_with')}
        if execute:
            status,result=client.request('GET',item['path'])
            shape=_shape(result)
            row.update({'status':status,'shape':shape})
            memory[item['id']]=(status,shape)
            report['operations_executed']+=1
            anchor=item.get('paired_with')
            if anchor in memory:
                ref_status,ref_shape=memory[anchor]
                report['comparisons'].append({'probe':item['id'],'anchor':anchor,
                    'same_http_status':status==ref_status,
                    'same_top_level_kind':shape.get('kind')==ref_shape.get('kind'),
                    'list_length_changed':shape.get('length')!=ref_shape.get('length')
                                           if shape.get('kind')=='array' and ref_shape.get('kind')=='array' else None,
                    'first_record_key_overlap':len(set(shape.get('first_keys',[])) &
                                                    set(ref_shape.get('first_keys',[])))})
        else:
            row['status']='NOT_EXECUTED'
        report['probes'].append(row)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(report,indent=2,sort_keys=True)+'\n'
    (out/'discovery_report.json').write_text(blob)
    (out/'FREEZE.json').write_text(json.dumps({'phase':'discovery',
       'sha256':sha256(blob.encode()).hexdigest(),'live_writes':0,
       'independent_holdout':False},indent=2)+'\n')
    return report


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description='Dolibarr read-only one-variable interventions; dry run default')
    ap.add_argument('--base-url',default='http://127.0.0.1:8088')
    ap.add_argument('--fixtures',help='JSON with only synthetic fixture IDs')
    ap.add_argument('--budget',type=int,default=16)
    ap.add_argument('--execute-readonly',action='store_true')
    ap.add_argument('--out',default='build/gate2c_dolibarr')
    a=ap.parse_args(argv)
    if not _local_host(urlparse(a.base_url).hostname or ''):
        ap.error('Loopback host required')
    fixtures=json.loads(Path(a.fixtures).read_text()) if a.fixtures else {}
    plan=bounded_differential_plan(fixtures,budget=a.budget)
    key=os.getenv('DOLIBARR_API_KEY') if a.execute_readonly else None
    if a.execute_readonly and not key:ap.error('DOLIBARR_API_KEY required for executed read-only probes')
    c=DolibarrClient(a.base_url,key)
    result=differential_readonly(c,plan,execute=a.execute_readonly,outdir=a.out)
    print(json.dumps({'read_calls':result['operations_executed'],'writes':0,
                      'report':str(Path(a.out)/'discovery_report.json')},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
