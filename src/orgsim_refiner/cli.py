"""One-study CLI plus scoped inference and an offline evidence inbox."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from .util import read_json,write_json
from .types import Schema,Scope,Observation
from .workspace import read_rows

def main(argv=None):
    p=argparse.ArgumentParser(description='OrgSim × Model Refiner v0.7 — active executable policies, no automated ERP writes')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','study'):
        q=sub.add_parser(name)
        q.add_argument('--seed',type=int,default=771902);q.add_argument('--tenants',type=int,default=24)
        q.add_argument('--train',type=int,default=32);q.add_argument('--validation',type=int,default=128)
        q.add_argument('--test',type=int,default=256);q.add_argument('--stress',type=int,default=256)
        q.add_argument('--query-budget',type=int,default=8192);q.add_argument('--max-cells',type=int,default=4096)
        q.add_argument('--audit-samples',type=int,default=128);q.add_argument('--execute-model',action='store_true')
        q.add_argument('--model',default='gpt-6-luna');q.add_argument('--model-cases',type=int,default=8)
        q.add_argument('--max-output-tokens',type=int,default=4096);q.add_argument('--max-provider-calls',type=int,default=576)
        if name=='study':q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('verify');q.add_argument('--root',type=Path,required=True)
    q=sub.add_parser('predict');q.add_argument('--policy',type=Path,required=True);q.add_argument('--input',type=Path,required=True)
    q=sub.add_parser('init');q.add_argument('--spec',type=Path,required=True);q.add_argument('--training',type=Path,required=True);q.add_argument('--validation',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--query-budget',type=int,default=8192)
    q=sub.add_parser('learn');q.add_argument('--root',type=Path,required=True)
    q=sub.add_parser('submit');q.add_argument('--root',type=Path,required=True);q.add_argument('--request',required=True);q.add_argument('--observation',type=Path,required=True);q.add_argument('--receipt',type=Path);q.add_argument('--diagnostic',type=Path)
    q=sub.add_parser('propose');q.add_argument('--spec',type=Path,required=True);q.add_argument('--training',type=Path,required=True);q.add_argument('--model',default='gpt-6-luna');q.add_argument('--execute-model',action='store_true');q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('dolibarr-check');q.add_argument('--receipt',type=Path,required=True);q.add_argument('--diagnostic',type=Path,required=True);q.add_argument('--reason',choices=('product_not_on_sale','insufficient_stock'),required=True)
    q=sub.add_parser('dolibarr-status');q.add_argument('--root',type=Path,required=True)
    q=sub.add_parser('audit-history');q.add_argument('--bundle',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command in ('plan','study'):
            needed=3*a.tenants*a.model_cases if a.execute_model else 0
            plan={'format':'orgsim-refiner-integrated-plan-v1','tenants':a.tenants,'seed':a.seed,
                  'heldout_and_stress_cases':a.tenants*(a.test+a.stress),
                  'maximum_simulator_query_attempts':a.tenants*a.query_budget,
                  'model_comparison_requested':a.execute_model,'model':a.model if a.execute_model else None,
                  'maximum_provider_attempts':needed,'provider_calls_made_by_preview':0,
                  'max_output_tokens':a.max_output_tokens,'no_dollar_price_assumed':True,
                  'actual_erp_calls':0,'scope':'Synthetic raw-input study with explicit region-coverage accounting.'}
            if a.command=='plan':result=plan
            else:
                backend=None
                if a.execute_model:
                    if not 16<=a.max_output_tokens<=32000:raise ValueError('Output cap 16..32000')
                    if a.max_provider_calls<needed:raise ValueError(f'Model plan requires at most {needed} attempts; raise your explicit ceiling or reduce study dimensions')
                    if not os.getenv('OPENAI_API_KEY'):raise ValueError('Set OPENAI_API_KEY privately; no key is saved')
                    from model_refiner.backend import OpenAIResponsesBackend
                    from .model_bridge import BudgetBackend
                    backend=BudgetBackend(OpenAIResponsesBackend(max_output_tokens=a.max_output_tokens),a.max_provider_calls)
                from .benchmark import run_study
                result=run_study(a.out,seed=a.seed,tenants=a.tenants,train=a.train,validation=a.validation,test=a.test,stress=a.stress,
                    budget=a.query_budget,max_cells=a.max_cells,audit_samples=a.audit_samples,
                    model=a.model if a.execute_model else None,backend=backend,model_cases=a.model_cases)
                result={k:v for k,v in result.items() if k!='tenants_detail'}
        elif a.command=='verify':
            from .verification import verify_study
            result=verify_study(a.root)
        elif a.command=='predict':
            from .artifact import PolicyArtifact
            inp=read_json(a.input);result=PolicyArtifact.load(a.policy).predict(inp['features'],Scope(**inp['scope']))
        elif a.command=='init':
            from .workspace import initialize
            spec=read_json(a.spec);result=initialize(a.root,Schema.from_dict(spec['schema']),Scope(**spec['scope']),read_rows(a.training),read_rows(a.validation),budget=a.query_budget)
        elif a.command=='learn':
            from .workspace import learn
            result=learn(a.root)
        elif a.command=='submit':
            from .workspace import submit
            result=submit(a.root,a.request,Observation.from_dict(read_json(a.observation)),receipt=a.receipt,diagnostic=a.diagnostic)
        elif a.command=='propose':
            if not a.execute_model:result={'provider_calls':0,'max_attempts':1,'max_output_tokens':4096,'status':'preview_only'}
            else:
                from model_refiner.backend import OpenAIResponsesBackend
                from .model_bridge import BudgetBackend,propose_ast
                if not os.getenv('OPENAI_API_KEY'):raise ValueError('Set OPENAI_API_KEY privately')
                spec=read_json(a.spec);backend=BudgetBackend(OpenAIResponsesBackend(max_output_tokens=4096),1)
                result=propose_ast(Schema.from_dict(spec['schema']),Scope(**spec['scope']),read_rows(a.training),backend,model=a.model)
                result['provider_usage']=backend.usage();write_json(a.out,result)
        elif a.command=='dolibarr-check':
            from .dolibarr_bridge import check_instrumentation
            result=check_instrumentation(a.receipt,a.diagnostic,a.reason)
        elif a.command=='dolibarr-status':
            from .dolibarr_bridge import inspect_workspace
            result=inspect_workspace(a.root)
        elif a.command=='audit-history':
            from .historical_audit import audit_archive
            result=audit_archive(a.bundle,a.out)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,FileExistsError,FileNotFoundError,RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
