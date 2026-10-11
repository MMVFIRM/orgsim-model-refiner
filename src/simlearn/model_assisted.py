"""Optional real Model Refiner *inference* experiment with compiled policy context.

Uses Model Refiner 0.5rc1 Recipe + Predictor, not the native suite.fit
optimizer. Provider calls are opt-in and strictly capped. Structured learner
holds final authority *within the synthetic environment*, not on a real ERP.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from hashlib import sha256
from model_refiner.backend import ModelBackend,OpenAIResponsesBackend
from model_refiner.suite.artifact import Recipe
from model_refiner.suite.runtime import Predictor,Memory
from .dataset import CLASS_TASK,load_outcomes
from .structured import Program,ModelRefinerPolicyAdapter
from .structured_study import evaluate,payload_write
from .learning import EffectLearner
from .oracle import canonical,digest


class QuotaBackend(ModelBackend):
    def __init__(self,backend:ModelBackend,max_calls:int):
        self.backend=backend;self.max_calls=max_calls;self.calls=0
    def generate(self,*,model,messages):
        if self.calls>=self.max_calls:raise RuntimeError('Model provider hard request ceiling reached')
        self.calls+=1
        return self.backend.generate(model=model,messages=messages)
    def cache_identity(self):return self.backend.cache_identity()


def _read(data_dir:Path,memory_path:Path):
    manifest=json.loads((data_dir/'DATASET_FREEZE.json').read_text('utf-8'))
    art=json.loads(memory_path.read_text('utf-8'))
    if art.get('format')!='model-refiner-external-program-memory-v1':raise ValueError('Wrong policy memory format')
    if digest(art.get('policies'))!=art.get('policies_sha256'):raise ValueError('Policy memory integrity mismatch')
    if art['tenant_fingerprint']!=manifest['manifest_sha256']:raise ValueError('Policy belongs to a different synthetic tenant')
    return manifest,ModelRefinerPolicyAdapter({k:Program.from_dict(v) for k,v in art['policies'].items()},tenant_fingerprint=manifest['manifest_sha256'])


def plan(data_dir:Path,memory_path:Path,*,limit:int=64,model:str='gpt-6-luna'):
    if limit<1 or limit>512:raise ValueError('Limit 1..512')
    manifest,adapter=_read(Path(data_dir),Path(memory_path))
    test=load_outcomes(Path(data_dir)/'outcome.jsonl')['test']
    if len(test)<limit:raise ValueError('Insufficient held-out examples')
    return {'format':'simlearn-native-model-refiner-policy-assist-plan-v1',
            'model':model,'scored_examples':limit,'max_provider_attempts_without_repair':2*limit,
            'baseline_prompts':limit,'compiled_program_prompts':limit,
            'source_tenant_sha256':manifest['manifest_sha256'],
            'operations':'synthetic_only','live_Dolibarr_reads':0,'live_Dolibarr_writes':0,
            'provider_calls':0,'execution_authorized':False,
            'note':'Native Model Refiner Predictor/Recipe inference, not suite.fit and not neural weight updates.'}


def run(*,data_dir:Path,memory_path:Path,outdir:Path,model:str='gpt-6-luna',
        limit:int=64,max_calls:int|None=None,execute_model:bool=False,backend:ModelBackend|None=None)->dict:
    data_dir=Path(data_dir);memory_path=Path(memory_path);outdir=Path(outdir)
    preview=plan(data_dir,memory_path,limit=limit,model=model)
    if not execute_model and backend is None:return preview
    if outdir.exists():raise FileExistsError('Refusing to overwrite a native inference result')
    if execute_model and backend is None:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('Set OPENAI_API_KEY securely before explicitly executing model calls')
        backend=OpenAIResponsesBackend(max_output_tokens=80)
    max_calls=2*limit if max_calls is None else max_calls
    if max_calls<2*limit or max_calls>1024:
        raise ValueError('Insufficient/excessive provider call limit')
    guarded=QuotaBackend(backend,max_calls)
    manifest,adapter=_read(data_dir,memory_path)
    source=load_outcomes(data_dir/'outcome.jsonl')
    rows=source['test'][:limit]
    frozen_programs=adapter.frozen_payload()['policies']
    # Expose the *public* baseline conditions as a literal, human-readable
    # contract. The learned private thresholds come exclusively from the
    # program artifact; none are hand-encoded here.
    public_contract={
      'validate_order':'order.status == draft AND product.stock >= order.qty AND customer.credit >= order.amount',
      'pay_invoice':'invoice.status == issued AND invoice.balance >= action.payment',
      'reserve_stock':'order.status == draft AND product.stock >= action.reserve_qty',
      'close_order':'order.status == validated AND order.shipped >= order.qty AND order.invoiced == true',
    }
    rule_text=('STRUCTURED POLICY MEMORY (synthetic OrgSim experiments only). '
       'Read input as JSON, select its operation, apply the PUBLIC PRECONDITIONS below and then its learned guards. '
       'ACCEPT if and only if all public preconditions pass and none of the vetoes applies. '
       'For an upper or sum_upper guard, reject when the expression is GREATER than max. '
       'For a pair_upper guard, reject when BOTH numerical expressions are at least their ge thresholds. '
       'A dot expression reads state.entity.field, except action.field which reads action; '
       'product.reserved+action.reserve_qty is an arithmetic sum. '
       'Never invent additional conditions and do not authorize real ERP transactions. '
       'Return only ACCEPT or REJECT.\nPUBLIC_PRECONDITIONS_JSON\n'
       +canonical(public_contract)+'\nCOMPILED_PRIVATE_GUARDS_JSON\n'+canonical(frozen_programs))
    if len(rule_text)>40_000:raise ValueError('Rule material exceeds Model Refiner Recipe contract')
    baseline=Recipe(id='baseline',rules='',source='baseline')
    compiled=Recipe(id='structured-executable',rules=rule_text,source='hybrid')
    predictor=Predictor(guarded,CLASS_TASK,model,Memory([]),repair_attempts=0,
                        max_prompt_chars=50_000)
    raw={'baseline':[],'compiled_model':[],'policy_verified':[]}
    receipts=[]
    for row in rows:
        sample=row['sample'];record={'id':row['id'],'input_sha256':digest(sample),'target':row['decision']}
        for name,recipe in [('baseline',baseline),('compiled_model',compiled)]:
            try:
                response=predictor.predict(canonical(sample),recipe)
                proposed=response.text.strip().upper() if response.status=='accepted' else 'ABSTAIN'
                if proposed not in ('ACCEPT','REJECT'):proposed='ABSTAIN'
                record[name]={'prediction':proposed,'status':response.status,
                        'logical_calls':response.logical_calls,
                        'input_tokens':response.input_tokens,'output_tokens':response.output_tokens}
            except Exception as exc:
                # Fail closed. Preserve a partial report for audit below.
                record[name]={'prediction':'ABSTAIN','status':'error','error_type':type(exc).__name__}
            raw[name].append(record[name]['prediction'])
        verdict=adapter.decide(sample,raw['compiled_model'][-1],tenant_fingerprint=manifest['manifest_sha256'])
        raw['policy_verified'].append(verdict['decision'])
        record['policy_verified']={'prediction':verdict['decision'],
                       'tenant_binding_matched':verdict['tenant_binding_matched'],
                       'model_overridden':verdict['decision']!=raw['compiled_model'][-1]}
        receipts.append(record)
    # Effect induction sees only historical train/validation trajectories.
    effects=EffectLearner().fit(source['train'],source['validation'])
    metrics={k:evaluate(rows,v,effects) for k,v in raw.items()}
    output={'format':'simlearn-native-refiner-policy-assist-result-v1',
      'model':model,'backend_identity':guarded.cache_identity(),
      'inference_uses_native_model_refiner_Predictor_and_Recipe':True,
      'native_model_refiner_fit_performed':False,
      'same_test_samples_for_baseline_and_policy_prompts':True,
      'model_provider_attempts':guarded.calls,'max_provider_attempts':max_calls,
      'evaluated_examples':len(rows),'source_tenant_sha256':manifest['manifest_sha256'],
      'simulator_labels_only':True,'real_ERP_read_calls':0,'real_ERP_write_calls':0,
      'metrics':metrics,'inference_receipts':receipts,
      'caution':'The program sidecar may override an LLM; verified accuracy is policy program accuracy, not an increase in the LLM base weights.'}
    rb=payload_write(outdir/'MODEL_ASSIST_RESULT.json',output)
    payload_write(outdir/'MODEL_ASSIST_FREEZE.json',{'format':'simlearn-native-refiner-policy-assist-freeze-v1',
        'report_sha256':rb,'provider_attempts':guarded.calls,'externally_timestamped':False,
        'real_ERP_write_calls':0})
    return output
