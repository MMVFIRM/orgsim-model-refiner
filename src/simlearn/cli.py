"""Single integrated simulation/adaptation experiment. All commands offline by default."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from .dataset import generate_environment
from .benchmark import run_environment,summarize
from .refiner_runner import run_refiner


def _dump(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    data=(json.dumps(content,sort_keys=True,indent=2)+'\n').encode('utf-8')
    path.write_bytes(data)
    return sha256(data).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(prog='simlearn',description='One scoped OrgSim x Model Refiner simulation-learning experiment')
    sub=p.add_subparsers(dest='command',required=True)
    def dims(q):
        q.add_argument('--train',type=int,default=400,help='per operation; balanced')
        q.add_argument('--router',type=int,default=100,help='per operation; balanced')
        q.add_argument('--validation',type=int,default=100,help='per operation; balanced')
        q.add_argument('--test',type=int,default=200,help='per operation; balanced')
        q.add_argument('--stress',type=int,default=200,help='per operation; balanced')
    full=sub.add_parser('full',help='Generate frozen curricula, fit conventional comparators, score once')
    dims(full);full.add_argument('--seed',type=int,default=20261011)
    full.add_argument('--environments',type=int,default=3)
    full.add_argument('--out',default='build/simlearn_confirmatory')
    gen=sub.add_parser('generate',help='Generate a single OrgSim simulated environment dataset')
    dims(gen);gen.add_argument('--seed',type=int,default=20261011);gen.add_argument('--out',required=True)
    bench=sub.add_parser('benchmark',help='Fit and score pre-generated environment')
    bench.add_argument('--data',required=True);bench.add_argument('--out',required=True)
    bench.add_argument('--seed',type=int,default=42)
    rf=sub.add_parser('refiner',help='Model Refiner native SDK dry run / optional actual model fit')
    rf.add_argument('--data',required=True)
    rf.add_argument('--out',default='build/refiner_model_run')
    rf.add_argument('--task',choices=['classification','effects'],default='classification')
    rf.add_argument('--model',default='gpt-6')
    rf.add_argument('--optimizer-model',default=None)
    rf.add_argument('--execute-model',action='store_true',help='EXPLICITLY authorize OpenAI provider calls')
    rf.add_argument('--mock',action='store_true',help='offline integration smoke only; not an LLM')
    rf.add_argument('--train-limit',type=int,default=32)
    rf.add_argument('--router-limit',type=int,default=8)
    rf.add_argument('--validation-limit',type=int,default=8)
    rf.add_argument('--test-limit',type=int,default=32)
    rf.add_argument('--max-attempts',type=int,default=2500)
    rf.add_argument('--max-reserved-tokens',type=int,default=40_000_000)
    rf.add_argument('--max-output-tokens',type=int,default=8192)
    rf.add_argument('--seed',type=int,default=42)
    verify=sub.add_parser('verify',help='Verify the report and dataset content hashes; does not prove external truth')
    verify.add_argument('--root',required=True)
    args=p.parse_args(argv)
    if args.command=='generate':
        result=generate_environment(seed=args.seed,outdir=args.out,train=args.train,
                router=args.router,validation=args.validation,test=args.test,stress=args.stress)
        print(json.dumps({'output':args.out,'oracle_queries':result['oracle_queries_total'],'manifest_sha256':result['manifest_sha256']}))
    elif args.command=='benchmark':
        r=run_environment(args.data,args.out,seed=args.seed)
        print(json.dumps({'output':args.out,'results':{s:{n:{'accuracy':v['accuracy'],
                     'end_to_end_exact_accuracy':v['end_to_end_exact_accuracy']}
                     for n,v in r['scores'][s].items()} for s in ('test','stress')}}))
    elif args.command=='full':
        if not 1<=args.environments<=16:raise ValueError('Environment count must be 1..16')
        root=Path(args.out)
        if root.exists():raise FileExistsError(f'Frozen experiment output exists: {root}')
        root.mkdir(parents=True)
        results=[]
        for n in range(args.environments):
            seed=args.seed+7919*n
            data=root/f'env{n+1}_data';bench=root/f'env{n+1}_benchmark'
            manifest=generate_environment(seed=seed,outdir=data,train=args.train,
                 router=args.router,validation=args.validation,test=args.test,stress=args.stress)
            print(f'[simlearn] env={n+1} generated {manifest["oracle_queries_total"]} OrgSim calls',flush=True)
            r=run_environment(data,bench,seed=seed)
            results.append(r)
            s=r['scores']['test']['tabular_simtrained']
            print(f'[simlearn] env={n+1} held-out: decision={s["accuracy"]:.4f}, end2end={s["end_to_end_exact_accuracy"]:.4f}, false_accepts={s["false_accepts"]}',flush=True)
        summary=summarize(results)
        summary.update({'seeds':[args.seed+7919*n for n in range(args.environments)],
               'real_model_fitted':False,'real_ERP_fidelity_verified':False,
               'oracle_calls_total':sum(x['training_oracle_calls_including_discarded'] for x in results),
               'criterion':{'decision_accuracy_min':.95,'end2end_min':.95,'false_accept_rate_max':.02},
               'scope':'heldout states in seen synthetic policy environments'})
        agg=summary['scores']['test']['tabular_simtrained']
        true_neg=sum(r['scores']['test']['tabular_simtrained']['per_operation'][op]['n']//2 for r in results for op in ('validate_order','pay_invoice','reserve_stock','close_order'))
        summary['performance_target_met']=(agg['accuracy']>=.95 and agg['end_to_end_exact_accuracy']>=.95 and
               agg['false_accepts']/max(1,true_neg)<=.02)
        digest=_dump(root/'SUMMARY.json',summary)
        _dump(root/'RUN_FREEZE.json',{'format':'simlearn-full-run-freeze-v1','summary_sha256':digest,
                 'live_erp_writes':0,'live_erp_reads':0,'model_provider_calls':0,
                 'independent_erp_holdout':False,'performance_target_met':summary['performance_target_met']})
        print(json.dumps({'summary':str(root/'SUMMARY.json'),'performance_target_met':summary['performance_target_met'],
             'results':summary['scores']},indent=2))
    elif args.command=='refiner':
        limits={'train':args.train_limit,'router':args.router_limit,
                'validation':args.validation_limit,'test':args.test_limit}
        report=run_refiner(data_dir=args.data,workspace=args.out,task_name=args.task,
             model=args.model,optimizer_model=args.optimizer_model,execute_model=args.execute_model,
             mock=args.mock,limits=limits,max_attempts=args.max_attempts,
             max_reserved_tokens=args.max_reserved_tokens,max_output_tokens=args.max_output_tokens,seed=args.seed)
        print(json.dumps(report,indent=2,sort_keys=True))
    elif args.command=='verify':
        root=Path(args.root);f=root/'RUN_FREEZE.json';sumfile=root/'SUMMARY.json'
        freeze=json.loads(f.read_text());actual=sha256(sumfile.read_bytes()).hexdigest()
        if actual!=freeze['summary_sha256']:raise ValueError('Summary hash mismatch')
        for n in range(1,len(json.loads(sumfile.read_text())['seeds'])+1):
            folder=(root/f'env{n}_benchmark')
            if not folder.exists():folder=root/f'env{n}'  # compact bundled confirmation layout
            path=folder/'BENCH_FREEZE.json'
            record=json.loads(path.read_text());content=(folder/'benchmark_report.json').read_bytes()
            if sha256(content).hexdigest()!=record['report_sha256']:raise ValueError(f'Benchmark hash mismatch in env {n}')
        print(json.dumps({'integrity':'OK','external_attestation':False,'summary_sha256':actual}))
    return 0

if __name__=='__main__':raise SystemExit(main())
