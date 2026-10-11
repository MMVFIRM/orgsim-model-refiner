"""Standalone one-cycle synthetic black-box program-memory experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from hashlib import sha256
from .dataset import generate_environment,load_outcomes
from .structured import Program,ModelRefinerPolicyAdapter
from .structured_study import run_environment,run_ood,summarize,payload_write,FORMAT
from .oracle import digest


def verify(root:Path)->dict:
    manifest=json.loads((root/'FINAL_FREEZE.json').read_text('utf-8'))
    summary=(root/'STUDY_SUMMARY.json').read_bytes()
    if sha256(summary).hexdigest()!=manifest['summary_sha256']:
        raise ValueError('Summary integrity mismatch')
    head=json.loads(summary)
    for index,seed in enumerate(head['seeds'],1):
        folder=root/f'env{index}_policy'
        frozen=json.loads((folder/'EXPERIMENT_FREEZE.json').read_text())
        report=(folder/'STRUCTURED_REPORT.json').read_bytes()
        programs=(folder/'MODEL_REFINER_PROGRAM_MEMORY.json').read_bytes()
        if sha256(report).hexdigest()!=frozen['report_sha256']:
            raise ValueError(f'Environment {index} report integrity mismatch')
        if sha256(programs).hexdigest()!=frozen['program_sha256']:
            raise ValueError(f'Environment {index} program integrity mismatch')
        parsed=json.loads(programs)
        if digest(parsed['policies'])!=parsed['policies_sha256']:
            raise ValueError(f'Environment {index} program-content integrity mismatch')
        report_obj=json.loads(report)
        if parsed.get('tenant_fingerprint')!=report_obj['source_manifest_sha256']:
            raise ValueError('Tenant binding mismatch')
        for name,p in parsed['policies'].items():
            if Program.from_dict(p).operation!=name:
                raise ValueError('Operation binding mismatch')
        d=root/f'env{index}_data'/'DATASET_FREEZE.json'
        if sha256(d.read_bytes()).hexdigest()!=frozen['dataset_freeze_sha256']:
            raise ValueError('Dataset freeze mismatch')
        if json.loads(report)['seed']!=seed:
            raise ValueError('Seed binding mismatch')
    return {'format':'local-integrity-verification-v1','status':'OK',
            'summary_sha256':sha256(summary).hexdigest(),
            'environments_verified':len(head['seeds']),
            'externally_timestamped':False,'oracle_truth_independently_verified':False}


def main(argv=None):
    p=argparse.ArgumentParser(prog='simlearn.structured_cli',description='Black-box executable policy learning (synthetic OrgSim only)')
    sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('full',help='Create fresh synthetic organizations, fit and freeze, evaluate once')
    run.add_argument('--out',default='build/structured_confirmatory')
    run.add_argument('--environments',type=int,default=5)
    run.add_argument('--seed',type=int,default=20261123)
    for key,default in [('train',128),('router',32),('validation',32),('test',128),('stress',128)]:
        run.add_argument('--'+key,type=int,default=default,help='Per operation; must be even')
    run.add_argument('--probe-budget',type=int,default=160,help='Maximum active queries per operation')
    run.add_argument('--audit-samples',type=int,default=32)
    run.add_argument('--ood-seeds',type=int,default=2)
    llm=sub.add_parser('llm',help='Optional Model Refiner Predictor: baseline vs program recipe; dry-run by default')
    llm.add_argument('--data',required=True)
    llm.add_argument('--memory',required=True)
    llm.add_argument('--model',default='gpt-6-luna')
    llm.add_argument('--limit',type=int,default=64)
    llm.add_argument('--max-calls',type=int,default=None)
    llm.add_argument('--out',default='build/structured_model_assist')
    llm.add_argument('--execute-model',action='store_true')
    chk=sub.add_parser('verify',help='Verify local SHA-256 bindings, not external truth')
    chk.add_argument('--root',required=True)
    pred=sub.add_parser('predict',help='Predict one synthetic sample from a frozen policy memory')
    pred.add_argument('--memory',required=True)
    pred.add_argument('--sample',required=True)
    pred.add_argument('--model-proposal',choices=['ACCEPT','REJECT','ABSTAIN'])
    pred.add_argument('--tenant-fingerprint',required=True,help='Synthetic tenant manifest SHA-256')
    args=p.parse_args(argv)
    if args.command=='llm':
        from .model_assisted import run as run_model_assisted
        response=run_model_assisted(data_dir=Path(args.data),memory_path=Path(args.memory),
                 outdir=Path(args.out),model=args.model,limit=args.limit,
                 max_calls=args.max_calls,execute_model=args.execute_model)
        if args.execute_model:
            print(json.dumps({'out':args.out,'provider_calls':response['model_provider_attempts'],
                               'metrics':response['metrics']},indent=2))
        else:print(json.dumps(response,indent=2))
        return 0
    if args.command=='verify':
        print(json.dumps(verify(Path(args.root)),indent=2));return 0
    if args.command=='predict':
        art=json.loads(Path(args.memory).read_text('utf-8'))
        if art.get('format')!='model-refiner-external-program-memory-v1' or digest(art.get('policies'))!=art.get('policies_sha256'):
            raise ValueError('Program memory integrity mismatch')
        programs={op:Program.from_dict(v) for op,v in art['policies'].items()}
        sample=json.loads(Path(args.sample).read_text('utf-8'))
        verdict=ModelRefinerPolicyAdapter(programs,tenant_fingerprint=art.get('tenant_fingerprint')).decide(
            sample,args.model_proposal,tenant_fingerprint=args.tenant_fingerprint)
        print(json.dumps(verdict,indent=2));return 0
    if not 1<=args.environments<=16 or not 0<=args.ood_seeds<=8 or not 8<=args.audit_samples<=96:
        raise ValueError('Invalid experiment size')
    if any(v<8 or v%2 for v in (args.train,args.router,args.validation,args.test,args.stress)):
        raise ValueError('All split sizes must be even and >=8')
    if not 20<=args.probe_budget<=1000:
        raise ValueError('Invalid active query budget')
    root=Path(args.out)
    if root.exists():raise FileExistsError(f'Refusing to overwrite existing freeze: {root}')
    root.mkdir(parents=True)
    reports=[]
    for i in range(args.environments):
        seed=args.seed+7919*i
        data=root/f'env{i+1}_data'
        prov=generate_environment(seed=seed,outdir=data,train=args.train,router=args.router,
                                  validation=args.validation,test=args.test,stress=args.stress)
        print(f'[structured] organization={i+1} simulator observations={prov["oracle_queries_total"]}',flush=True)
        report=run_environment(data,root/f'env{i+1}_policy',seed=seed,
                               max_probes=args.probe_budget,audit_count=args.audit_samples)
        reports.append(report)
        scores=report['scores']['test']
        print(f'[structured] organization={i+1} program={scores["structured_program"]["accuracy_all_cases"]:.4f} '
              f'active_tabular={scores["tabular_matched_active"]["accuracy_all_cases"]:.4f} '
              f'false_accepts={scores["structured_program"]["false_accepts"]} '
              f'audited={all(x["ready"] for x in report["audits_by_operation"].values())}',flush=True)
    ood=[]
    for index in range(args.ood_seeds):
        seed=args.seed+100000+7877*index
        stress=run_ood(seed,max_probes=args.probe_budget,audit_count=args.audit_samples,eval_per_op=128)
        ood.append(stress)
        print(f'[structured] out-of-class seed={seed} '+str({k:x['coverage'] for k,x in stress['per_operation'].items()}),flush=True)
    summary=summarize(reports,ood)
    if len(reports)>=2:
        transfer=[]
        for target_i in range(2,len(reports)+1):
            source_payload=json.loads((root/f'env{target_i-1}_policy'/'MODEL_REFINER_PROGRAM_MEMORY.json').read_text())
            source_policies={op:Program.from_dict(v) for op,v in source_payload['policies'].items()}
            source_adapter=ModelRefinerPolicyAdapter(source_policies,tenant_fingerprint=source_payload['tenant_fingerprint'])
            test_rows=load_outcomes(root/f'env{target_i}_data'/'outcome.jsonl')['test']
            target_tenant=reports[target_i-1]['source_manifest_sha256']
            bound=[source_adapter.decide(r['sample'],tenant_fingerprint=target_tenant)['decision'] for r in test_rows]
            unbound=[source_policies[r['operation']].classify(r['sample']) for r in test_rows]
            transfer.append({'source_org':target_i-1,'target_org':target_i,
                'cases':len(test_rows),
                'tenant_bound_abstentions':sum(x=='ABSTAIN' for x in bound),
                'hypothetical_unbound_accuracy':round(sum(x==r['decision'] for x,r in zip(unbound,test_rows))/len(test_rows),6),
                'hypothetical_unbound_false_accepts':sum(x=='ACCEPT' and r['decision']=='REJECT' for x,r in zip(unbound,test_rows)),
                'warning':'Unbound simulation is a diagnostic only; program memory refuses cross-tenant authorization.'})
        summary['cross_tenant_without_local_learning']=transfer
    summary.update({'safety_gate':{
       'min_accuracy_all_cases':0.99,'min_coverage':0.95,
       'maximum_false_accepts':0,'exact_state_effects_required':True},
       'scope':'Fresh synthetic tenants with a specified public-feature/monotone-upper-interaction grammar; not Dolibarr',
       'evidence_budget':'Training corpus plus explicitly counted adaptive black-box oracle queries; matched-active comparator gets the same probes',
       'quality_gate_passed':False})
    win=summary['arms']['test']['structured_program'];stress=summary['arms']['stress']['structured_program']
    summary['quality_gate_passed']=all(
        x['accuracy_all_cases']>=.99 and x['coverage']>=.95 and x['incorrect_accepts']==0
        and x['end_to_end_exact_all_cases']>=.99
        for x in (win,stress)) and summary['all_audits_ready']
    sum_sha=payload_write(root/'STUDY_SUMMARY.json',summary)
    payload_write(root/'FINAL_FREEZE.json',{'format':FORMAT+'-final-freeze',
        'summary_sha256':sum_sha,'provider_calls':0,'dolibarr_reads':0,'dolibarr_writes':0,
        'independent_tenant_holdout':False,'externally_timestamped':False})
    print(json.dumps({'summary':str(root/'STUDY_SUMMARY.json'),'quality_gate_passed':summary['quality_gate_passed'],
                      'scores':summary['arms']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
