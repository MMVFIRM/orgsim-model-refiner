"""Offline Gate 2D comparison against two *synthetic* hidden ERP tenants.

This is a runner-integration and information-efficiency test, NOT proof of
model learning or vendor-ERP fidelity. Compare fixed v0.2.2 discovery against
v0.2.3 adaptive discovery under the same read ceilings. No network access.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from hashlib import sha256

from .gate2c_dolibarr_readonly import bounded_differential_plan
from .gate2d_adaptive import Investigator, safe_shape


class SyntheticERP:
    loopback = True

    def __init__(self, mode: str):
        if mode not in {'empty', 'sparse', 'populated'}:
            raise ValueError('Unknown synthetic oracle')
        self.calls = []
        self.db = {'thirdparties': [], 'products': [], 'orders': [], 'invoices': []}
        if mode != 'empty':
            self.db['thirdparties'] = [
                {'id': str(i), 'name': f'InternalDemoAccount{i}', 'client': '1',
                 'status': '1', **{f'x_field_{j}': None for j in range(20)}}
                for i in (800001, 800002, 800003)]
            self.db['products'] = [
                {'id': str(i), 'ref': f'SAMPLE-{i}', 'label': f'DemoProduct{i}',
                 'price': '22.00', **{f'y_field_{j}': None for j in range(16)}}
                for i in ((900001, 900002) if mode == 'populated' else (900001,))]
        if mode == 'populated':
            self.db['orders'] = [
                {'id': str(i), 'socid': '800001', 'status': '1',
                 'ref': f'ORD-{i}', 'total_ht': '20.00'}
                for i in (700001,700002)]
            self.db['invoices'] = [
                {'id': '600001', 'socid': '800001', 'status': '0', 'ref':'INV-1'}]

    def request(self, method: str, path: str):
        if method != 'GET': raise RuntimeError('Only GET supported')
        self.calls.append((method, path))
        url = urlsplit(path)
        parts = url.path.lstrip('/').split('/')
        family = parts[0]
        if family not in self.db: raise ValueError('Unregistered synthetic path')
        qs = parse_qs(url.query)
        if len(parts)==1:
            limit = int(qs.get('limit', ['3'])[0]); page = int(qs.get('page',['0'])[0])
            records = self.db[family][page*limit:(page+1)*limit]
            if qs.get('ids_only')==['1']:
                return 200, [r['id'] for r in records]
            if 'properties' in qs:
                properties=qs['properties'][0].split(',')
                return 200, [{k:r[k] for k in properties if k in r} for r in records]
            return 200, records
        if family=='products' and parts[1]=='ref':
            match=next((r for r in self.db['products'] if r['ref']==parts[2]),None)
            return (200,match) if match else (404,{'error':'not found'})
        rec=next((r for r in self.db[family] if r['id']==parts[1]),None)
        if rec is None: return 404, {'error':'not found'}
        if len(parts)==2: return 200,rec
        ending=parts[-1]
        if ending == 'stock':return 200, {'stock_reel':'13', 'stock_theorique':'16'}
        if ending == 'categories':return 200,[]
        if ending in {'outstandingorders','outstandinginvoices'}:return 200, {'total':'0'}
        if ending == 'lines':return 200,[{'id':'500001','qty':'2','fk_product':'900001'}]
        if ending in {'shipment','payments'}:return 200,[]
        raise ValueError('Unexpected synthetic route')


def _snapshot_metrics(results: list[dict], *, arm: str):
    signatures=set()
    wasted=0
    probes=[]
    for r in results:
        sh=r['shape']
        kind=sh['kind']
        is_empty=kind=='array' and sh.get('length',0)==0
        if r['category'] not in ('anchor','collection') and is_empty:
            wasted+=1
        if r['status']==200 and not is_empty:
            fields=sh.get('first_keys',sh.get('keys',[]))
            sig=(r['family'],kind,sh.get('first_item_kind'),tuple(fields))
            signatures.add(sig)
        probes.append(r['id'])
    return {'arm':arm,'reads':len(results),
            'novel_nonempty_response_signatures':len(signatures),
            'nonanchor_empty_reads':wasted,
            'related_or_detail_reads':sum(r['category'] in {'individual','alternate_key','related'} for r in results),
            'probe_sequence':probes}


def evaluate(*,budget:int=15, outdir:str|Path='build/gate2d_offline')->dict:
    if type(budget) is not int or not 4<=budget<=20:
        raise ValueError('Budget must be 4..20 for matched fixed-vs-adaptive offline comparison')
    out=Path(outdir);out.mkdir(parents=True, exist_ok=True)
    cases=[]
    for mode in ('empty','sparse','populated'):
        fixed_oracle=SyntheticERP(mode)
        fixed_plan=bounded_differential_plan(budget=budget)
        fixed_rows=[]
        for p in fixed_plan:
            status,response=fixed_oracle.request('GET',p['path'])
            family=p['path'].split('/')[1].split('?')[0]
            fixed_rows.append({'id':p['id'],'family':family,'category':p['category'],
                               'status':status, 'shape':safe_shape(response)})
        fixed=_snapshot_metrics(fixed_rows,arm='v0.2.2_fixed')
        for auto_ids in (False, True):
            suffix='adaptive_with_ids' if auto_ids else 'adaptive_no_ids'
            oracle=SyntheticERP(mode)
            runout=out/f'{mode}_{suffix}'
            inv=Investigator(oracle,budget=budget,auto_ids=auto_ids)
            report=inv.run(execute=True,outdir=runout)
            adaptive=_snapshot_metrics(report['probes'],arm=suffix)
            assert adaptive['reads'] == report['operations_executed']
            cases.append({'oracle_family':mode,'budget_ceiling':budget,
                          'fixed':fixed,'adaptive':adaptive,
                          'executed_model_calls':0,'live_dolibarr_calls':0})
    report={'format':'orgsim-gate2d-synthetic-comparison-v1',
            'environment':'SYNTHETIC_ERP_ORACLES_ONLY',
            'same_max_read_budget':budget,
            'external_model_calls':0,'real_dolibarr_calls':0,
            'reconstruction_fidelity_measured':False,
            'cases':cases,
            'warnings':['Additional useful response shapes do not imply correct business-rule reconstruction.',
                        'Adaptive arm with IDs has an additional in-memory auto-ID capability; ablation provided.',
                        'This lab uses known synthetic oracle behavior; no blind live Dolibarr holdout was tested.']}
    file=out/'comparison_report.json'
    file.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (out/'FREEZE.json').write_text(json.dumps({'phase':'synthetic_offline_benchmark',
        'report_sha256':sha256(file.read_bytes()).hexdigest(),
        'live_dolibarr_calls':0,'external_model_calls':0},indent=2)+'\n')
    return report


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description='Gate 2D fixed-vs-adaptive comparison (synthetic only)')
    ap.add_argument('--budget',type=int,default=15)
    ap.add_argument('--out',default='build/gate2d_offline')
    args=ap.parse_args(argv)
    report=evaluate(budget=args.budget,outdir=args.out)
    for r in report['cases']:
        print(f"{r['oracle_family']:9} {r['adaptive']['arm']:18} "
              f"fixed unique={r['fixed']['novel_nonempty_response_signatures']} "
              f"adaptive unique={r['adaptive']['novel_nonempty_response_signatures']} "
              f"fixed empty={r['fixed']['nonanchor_empty_reads']} "
              f"adaptive empty={r['adaptive']['nonanchor_empty_reads']}")
    return 0

if __name__=='__main__': raise SystemExit(main())
