"""Bounded local Dolibarr read-only discovery for Gate 2B.

Default is a dry-run plan. Never issues POST/PUT/PATCH/DELETE. Only talks to
loopback Dolibarr, uses sandbox key solely from process environment, rejects
redirects via DolibarrClient, and never persists response values or credentials.
These observations are DEVELOPMENT evidence, not a blind workflow test.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from .dolibarr_gate import DolibarrClient, _shape, _local_host
from urllib.parse import urlparse


_ID = re.compile(r'^[0-9]{1,12}$')
_REF = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def build_readonly_plan(fixtures:dict|None=None, *, limit:int=3)->list[dict]:
    if type(limit) is not int or not 1<=limit<=10:
        raise ValueError('List limit must be 1..10')
    fixtures=fixtures or {}
    if set(fixtures)-{'thirdparty_id','product_id','order_id','invoice_id','product_ref'}:
        raise ValueError('Unrecognized fixture identifier; no arbitrary API paths accepted')
    endpoints=[('thirdparties','/thirdparties'),('products','/products'),
               ('orders','/orders'),('invoices','/invoices')]
    plan=[{'id':name+'_list','method':'GET','path':path+'?limit='+str(limit),
           'category':'collection'} for name,path in endpoints]
    for key,family in [('thirdparty_id','thirdparties'),('product_id','products'),
                        ('order_id','orders'),('invoice_id','invoices')]:
        if key in fixtures and fixtures[key] is not None:
            value=str(fixtures[key])
            if not _ID.fullmatch(value):raise ValueError('Fixture ID must be numeric')
            plan.append({'id':key+'_detail','method':'GET','path':f'/{family}/{value}',
                         'category':'individual'})
    if fixtures.get('product_ref') is not None:
        value=str(fixtures['product_ref'])
        if not _REF.fullmatch(value):raise ValueError('Product reference may only contain alphanumeric, _ and -')
        plan.append({'id':'product_ref_detail','method':'GET',
                     'path':'/products/ref/'+quote(value,safe=''),
                     'category':'alternate_key'})
    return plan


def run_readonly(client:DolibarrClient, plan:list[dict], *,
                 execute:bool=False, outdir:str|Path='build/gate2b_readonly')->dict:
    if not client.loopback:raise ValueError('Gate 2B restricted to a loopback disposable Dolibarr tenant')
    if len(plan)>10:raise ValueError('At most ten read-only operations per run')
    if any(p['method']!='GET' or p.get('category') not in {'collection','individual','alternate_key'}
           for p in plan):raise ValueError('All probes must be bounded GET operations')
    outdir=Path(outdir)
    report={'format':'orgsim-gate2b-readonly-discovery-v1','phase':'discovery',
            'evidence':'LIVE_DOLIBARR' if execute else 'DRY_RUN',
            'scope':'loopback_disposable_sandbox_only',
            'operations_configured':len(plan),'operations_executed':0,
            'writes_executed':0,'record_values_saved':False,'api_key_saved':False,
            'probes':[]}
    for p in plan:
        safe_route=p['path'].split('?')[0]
        if p['category']=='individual':safe_route=re.sub(r'/[0-9]+$','/{id}',safe_route)
        if p['category']=='alternate_key':safe_route='/products/ref/{ref}'
        row={'id':p['id'],'method':'GET','category':p['category'],
             'path_template':safe_route}
        if execute:
            status,payload=client.request('GET',p['path'])
            row.update({'status':status,'shape':_shape(payload)})
            report['operations_executed']+=1
        else:row['status']='NOT_EXECUTED'
        report['probes'].append(row)
    outdir.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(report,indent=2,sort_keys=True)+'\n'
    (outdir/'discovery_report.json').write_text(blob,encoding='utf-8')
    (outdir/'FREEZE.json').write_text(json.dumps({
        'phase':'discovery','status':'requires_new_blind_holdout_after_candidate_freeze',
        'report_sha256':sha256(blob.encode()).hexdigest(),
        'no_live_writes':True,'executed':execute},indent=2)+'\n',encoding='utf-8')
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='LOOPBACK ONLY, read-only discovery; dry run by default')
    p.add_argument('--base-url',default='http://127.0.0.1:8088')
    p.add_argument('--fixtures',help='JSON file of synthetic fixture IDs; never credentials')
    p.add_argument('--out',default='build/dolibarr/gate2b_readonly')
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--execute-readonly',action='store_true')
    args=p.parse_args(argv)
    if not _local_host(urlparse(args.base_url).hostname or ''):
        p.error('Only localhost / 127.0.0.1 is allowed')
    fixtures=json.loads(Path(args.fixtures).read_text()) if args.fixtures else {}
    plan=build_readonly_plan(fixtures,limit=args.limit)
    key=os.environ.get('DOLIBARR_API_KEY') if args.execute_readonly else None
    if args.execute_readonly and not key:
        p.error('DOLIBARR_API_KEY environment variable required for an executed read-only run')
    client=DolibarrClient(args.base_url,key)
    report=run_readonly(client,plan,execute=args.execute_readonly,outdir=args.out)
    print(json.dumps({'phase':report['phase'],'evidence':report['evidence'],
         'read_calls':report['operations_executed'],'writes':0,'report':str(Path(args.out)/'discovery_report.json')},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
