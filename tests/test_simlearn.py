"""Integration and scientific-integrity regressions for SimLearn."""
import json
import random
from pathlib import Path

import pytest
from model_refiner.suite.dataset import Dataset, Record
from model_refiner.suite.artifact import SuiteArtifact
from simlearn.oracle import OPS, Oracle, sample_input, policy_for, make_manifest, digest
from simlearn.dataset import CLASS_TASK, EFFECT_TASK, generate_environment, load_outcomes
from simlearn.learning import EffectLearner, feature, predict_generic
from simlearn.benchmark import _score, run_environment
from simlearn.refiner_runner import prepare_run, run_refiner
from simlearn.cli import main


@pytest.fixture(scope='module')
def oracle():
    obj=Oracle(1357)
    yield obj
    obj.close()


@pytest.fixture
def data(tmp_path):
    directory=tmp_path/'source'
    freeze=generate_environment(seed=137,outdir=directory,train=20,router=8,
                                validation=8,test=8,stress=8)
    return directory,freeze


@pytest.mark.parametrize('operation',OPS)
def test_real_sim_workflow_has_positive_and_negative(oracle,operation):
    rng=random.Random(272)
    outcomes=[]
    for _ in range(300):
        row=oracle.execute(sample_input(rng,operation))
        outcomes.append(row)
        if {x['decision'] for x in outcomes}=={'ACCEPT','REJECT'}:break
    assert {x['decision'] for x in outcomes}=={'ACCEPT','REJECT'}
    for result in outcomes:
        assert result['provenance']=='SYNTHETIC_ORGSIM_ORACLE'
        assert (result['before']==result['after']) == (result['decision']=='REJECT')
        if result['decision']=='REJECT':assert result['changed']=={}
        else:assert result['changed']


def test_policy_randomization_is_real():
    assert policy_for(1)!=policy_for(2)
    assert digest(make_manifest(policy_for(1)))!=digest(make_manifest(policy_for(2)))


def test_data_balanced_and_splits_sealed(data):
    directory,freeze=data
    split=load_outcomes(directory/'outcome.jsonl')
    keys=set()
    for name,records in split.items():
        assert len(records)==len(OPS)*freeze['splits'][name]
        for op in OPS:
            sub=[x for x in records if x['operation']==op]
            assert sum(x['decision']=='ACCEPT' for x in sub)==len(sub)//2
            assert sum(x['decision']=='REJECT' for x in sub)==len(sub)//2
        assert not keys.intersection(x['group'] for x in records)
        keys.update(x['group'] for x in records)
    assert freeze['oracle_queries_total']>=sum(len(s) for s in split.values())
    assert freeze['groups_disjoint'] is True


def test_model_refiner_accepts_both_datasets(data):
    directory,_=data
    for name,task in [('classification',CLASS_TASK),('effects',EFFECT_TASK)]:
        rows=[Record.from_dict(json.loads(line)) for line in
              (directory/f'{name}_refiner.jsonl').read_text().splitlines()]
        ds=Dataset.prepare(rows,task)
        assert len(ds.train)==80
        assert len(ds.router)==32
        assert len(ds.validation)==32
        assert len(ds.test)==32


def test_private_thresholds_absent_from_prompts(data):
    directory,freeze=data
    lines=(directory/'classification_refiner.jsonl').read_text()
    for key in vars(policy_for(freeze['seed'])):
        assert key not in lines
    assert 'SYNTHETIC_ORGSIM_ORACLE' in lines
    assert 'manifest_sha256' not in lines


def test_data_seed_determinism(tmp_path):
    a=generate_environment(seed=207,outdir=tmp_path/'a',train=8,router=4,validation=4,test=8,stress=8)
    b=generate_environment(seed=207,outdir=tmp_path/'b',train=8,router=4,validation=4,test=8,stress=8)
    assert a==b
    assert (tmp_path/'a'/'outcome.jsonl').read_bytes()==(tmp_path/'b'/'outcome.jsonl').read_bytes()


def test_effect_learner_uses_trajectories_only(data):
    directory,_=data
    sets=load_outcomes(directory/'outcome.jsonl')
    effects=EffectLearner().fit(sets['train'],sets['validation'])
    assert all(effects.report[op]['audited'] for op in OPS)
    for r in sets['validation']:
        assert effects.predict(r['sample'],r['decision'])==r['changed']


def test_public_guard_is_imperfect_on_hidden_thresholds(data):
    sets=load_outcomes(data[0]/'outcome.jsonl')
    truth=sets['test']
    score=_score(truth,[predict_generic(x) for x in truth])
    assert score['accuracy']<1
    assert score['false_accepts']>0


def test_accuracy_distinguishes_false_accepts():
    rows=[{'operation':'validate_order','decision':x,'sample':{'operation':'validate_order'},'changed':{}}
          for x in ['ACCEPT','REJECT','ACCEPT','REJECT']]
    score=_score(rows,['ACCEPT']*4)
    assert score['false_accepts']==2 and score['false_rejects']==0
    assert score['accuracy']==.5 and score['false_accept_rate']==1.


def test_refiner_dryrun_does_not_load_provider(data,monkeypatch):
    import simlearn.refiner_runner as rr
    monkeypatch.setattr(rr,'OpenAIResponsesBackend',lambda **kw: (_ for _ in ()).throw(AssertionError('remote model called')))
    plan=run_refiner(data_dir=data[0],workspace='never_created',limits={'train':8,'router':4,'validation':4,'test':8})
    assert plan['provider']=='not_executed'
    assert plan['plan']['split_counts']['test']==32
    assert plan.get('model_provider_calls') is None


def test_refiner_missing_api_key_fails_closed(data,tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):
        run_refiner(data_dir=data[0],workspace=tmp_path/'model_fit',execute_model=True,
                    limits={'train':8,'router':4,'validation':4,'test':8})
    assert not (tmp_path/'model_fit').exists()


def test_native_refiner_mock_fit_is_real_artifact(data,tmp_path):
    workspace=tmp_path/'mock_fit'
    result=run_refiner(data_dir=data[0],workspace=workspace,mock=True,
                  limits={'train':8,'router':4,'validation':4,'test':8})
    art=SuiteArtifact.load(workspace/'artifact.json')
    assert result['native_refiner_artifact_id']==art.artifact_id
    assert result['run_completed']
    assert result['provider']=='mock'
    assert result['native_refiner_gate']['passed'] is False
    assert result['native_refiner_gate']['checks']['live_provider'] is False
    assert result['model_provider_calls']>0
    assert art.to_dict()['provenance']['base_weights_modified'] is False


def test_benchmark_has_disjoint_source_splits(data,tmp_path):
    report=run_environment(data[0],tmp_path/'bench',seed=137)
    assert report['actual_llm_calls']==0
    assert report['simulator_ground_truth_is_not_dolibarr']
    assert report['selected_models']
    assert report['effect_induction']['validate_order']['audited']
    assert report['scores']['test']['always_reject']['accuracy']==.5
    assert report['scores']['test']['public_guards']['false_accepts']>0


def test_benchmark_refuses_overwrite(data,tmp_path):
    root=tmp_path/'bench'
    run_environment(data[0],root,seed=137)
    with pytest.raises(FileExistsError):run_environment(data[0],root,seed=137)


def test_invalid_unbalanced_dataset_rejected(tmp_path):
    with pytest.raises(ValueError,match='balanced'):
        generate_environment(seed=1,outdir=tmp_path/'bad',train=9,router=4,validation=4,test=8,stress=8)


def test_full_run_hash_tamper_detection(tmp_path):
    root=tmp_path/'full'
    assert main(['full','--environments','1','--train','8','--router','4',
                 '--validation','4','--test','8','--stress','8','--out',str(root)])==0
    assert main(['verify','--root',str(root)])==0
    with (root/'SUMMARY.json').open('a') as f:f.write('tampered')
    with pytest.raises(ValueError,match='Summary hash mismatch'):
        main(['verify','--root',str(root)])
