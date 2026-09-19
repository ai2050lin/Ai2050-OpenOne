"""Supplemental zero-update precision baseline for every training validation.

Added while formal training is underway; never changes a run or checkpoint.
"""
import argparse
from rdc_question_common import *
from rdc_question_learning import prepare_gradients, parameter_fingerprints
from rdc_question_native import QuestionNative
from phase2748_rdc_training import validate
from phase2748_rdc_fit_analysis import bootstrap
import rdc_question_data as data

FOLDER = OUT/'training_bridge_baseline'
NATIVE_RUN = 'untrained_original_BF16_reference'
BRIDGE_RUN = 'untrained_FP32_bridge_reference'
METRICS = ['answer_mean_NLL', 'first_NLL', 'later_mean_NLL']


def metrics(packet):
    ids = packet['teacher_ids']; loss = packet['NLL']
    assert ids.ndim == loss.ndim == 1 and len(ids) == len(loss) and len(ids) > 1
    assert np.isfinite(loss).all()
    return np.array([loss.mean(), loss[0], loss[1:].mean()], dtype=np.float64)


def state_mse(left, right):
    assert np.array_equal(left['teacher_ids'], right['teacher_ids'])
    assert left['postnorm_BF16'].shape == right['postnorm_BF16'].shape
    return float(np.mean((unbits(left['postnorm_BF16']).astype(float)-unbits(right['postnorm_BF16']).astype(float))**2))


def unit():
    packet = {'teacher_ids': np.arange(3), 'NLL': np.array([.5, 2., 4.]),
              'postnorm_BF16': np.full((3,2), 0x3f80, np.uint16)}
    np.testing.assert_array_equal(metrics(packet), [6.5/3, .5, 3.])
    assert state_mse(packet, packet) == 0
    other = {**packet, 'postnorm_BF16': np.full((3,2), 0x4000, np.uint16)}
    assert state_mse(packet, other) == 1.
    bad = {**other, 'teacher_ids': np.array([0,1,4])}
    try: state_mse(packet, bad)
    except AssertionError: pass
    else: raise AssertionError('Teacher mismatch accepted')
    value = {'timestamp':stamp(), 'all_passed':True, 'source_sha256':sha(__file__), 'checks':4,
             'scope':'In-memory CPU metric, full-coordinate squared change and alignment checks only.'}
    immutable(OUT/'unit'/('training_bridge_'+str(time.time_ns())+'.json'),value)
    save(OUT/'unit/training_bridge_current.json',value)
    print('NATURAL_TRAINING_BRIDGE_UNIT_PASS',4,flush=True)


def freeze():
    path=FOLDER/'execution.json'
    revision={'source':snapshot(__file__), 'training':snapshot(Path(__file__).with_name('phase2748_rdc_training.py')),
        'learning':snapshot(Path(__file__).with_name('rdc_question_learning.py')),
        'bridge':snapshot(Path(__file__).with_name('phase2747_rdc_training.py')),
        'native':snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'history':snapshot(Path(__file__).with_name('rdc_question_history.py')),
        'bootstrap':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py')),
        'training_material_sha256':sha(OUT/'training/material_manifest.json'),
        'training_execution_sha256':sha(OUT/'training/execution_contract.json')}
    if path.exists():
        value=read(path); assert value['execution']==revision; return value
    u=read(OUT/'unit/training_bridge_current.json');assert u['all_passed'] and u['source_sha256']==sha(__file__)
    value={'timestamp':stamp(),'execution':revision,'unit_sha256':sha(OUT/'unit/training_bridge_current.json'),
        'design_status':'Supplement added after4question bridge pilot and while formal96step training underway, before these192question zero-update precision results. No main run, selected rule or checkpoint changes.',
        'population':'All192original validation questions, 48wholecontexts; true given-teacher answers including EOS. Not new validation data or confirmation.',
        'execution_shape':'Reuse unchanged formal validate() and native segmentedB1 sequential teacher scoring. Fresh original BF16 endpoint must reproduce all original saved teacher arrays exactly; then same zero-update FP32 block16 bridge used by training.',
        'parameters':'No gradient calculation or optimizer update. All74711040target words copied from original; unchanged other parameter fingerprints and target FP32words verified after evaluation.',
        'comparisons':'All6fixedruns x all4fixedcheckpoints 1/8/32/96 against both untrained bridge and original native under identical true teacher histories. Mean-answer NLL, first, later separately; full teacher postnorm MSE.',
        'uncertainty':'2000paired wholecontext bootstrap resamples, same established seed2748005, cohorts separate/equal. Post-outcome-added diagnostic, descriptive intervals without multiplicity adjustment.',
        'scope':'Separates zero-update arithmetic bridge shift from later training shift. Does not remove BF16 rounding in training, prove a derivative of quantization, establish behavioral success or identify pretraining formation.'}
    immutable(path,value);return value


def packet(ref):
    assert sha(ROOT/ref['path']) == ref['sha256']
    with np.load(ROOT/ref['path']) as z: return {k:z[k].copy() for k in z.files}


def acquire():
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION)|{'phase2748_rdc_acquire.py','phase2748_rdc_training.py',
        'phase2748_rdc_learning_pilot.py','phase2748_rdc_prospective.py','phase2748_rdc_learned.py',
        'phase2748_rdc_readout.py',Path(__file__).name})
    freeze(); final=FOLDER/'acquisition.json'
    if final.exists():
        value=read(final); assert value['all_passed'] and value['execution_sha256']==sha(FOLDER/'execution.json');return
    start=time.monotonic();model=engine=None;handles=[]
    try:
        contract,_,rows,groups=material('qwen4')
        rows=[r for r in rows if r['split']=='validation'];groups=[g for g in groups if g['split']=='validation']
        _,_,original_questions=data.index('qwen4',{'validation'})
        model,tok=load('qwen4',FOLDER/'loader'/str(time.time_ns()))
        assert all(p.device.type=='cuda' for p in model.parameters())
        unchanged=parameter_fingerprints(model,exclude_target=True)
        engine=QuestionNative(model,[])
        native=validate(model,engine,rows,groups,contract,NATIVE_RUN,0)
        for item in native['records']:
            old=packet(original_questions[item['question_id']]['teacher']['field']); actual=packet(item['field'])
            assert set(old)==set(actual) and all(np.array_equal(old[k],actual[k]) for k in old),item['question_id']
        target,original,handles=prepare_gradients(model)
        with torch.no_grad():
            bridge=validate(model,engine,rows,groups,contract,BRIDGE_RUN,0)
        for name,param in target.named_parameters(): assert torch.equal(param.detach().cpu(),original[name])
        assert parameter_fingerprints(model,exclude_target=True)==unchanged
        immutable(FOLDER/'unchanged_parameters.json',unchanged)
        value={'timestamp':stamp(),'all_passed':True,'execution_sha256':sha(FOLDER/'execution.json'),
            'questions':192,'contexts':48,'actual_optimizer_steps':0,'original_teacher_arrays_exact':True,
            'target_and_other_parameter_words_unchanged':True,
            'native_validation_sha256':sha(OUT/'training'/NATIVE_RUN/'validation/step0.json'),
            'bridge_validation_sha256':sha(OUT/'training'/BRIDGE_RUN/'validation/step0.json'),
            'unchanged_parameters_sha256':sha(FOLDER/'unchanged_parameters.json'),
            'teacher_tokens':native['teacher_tokens'],'seconds':time.monotonic()-start,
            'scope':'Actual independent zero-update numerical endpoints, not a new training run or heldout experiment.'}
        immutable(final,value);print('NATURAL_TRAINING_BRIDGE_ACQUIRED',round(value['seconds'],1),flush=True)
    except Exception as exc: failure(FOLDER,start,exc);raise
    finally:
        if engine is not None:engine.close()
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def analyze():
    spec=freeze();final=FOLDER/'result.json';start=time.monotonic()
    if final.exists():
        assert read(final)['execution_sha256']==sha(FOLDER/'execution.json');return
    acq=read(FOLDER/'acquisition.json'); assert acq['all_passed']
    completed=read(OUT/'training/result.json');assert completed['all_passed']
    rows,_,_=data.index('qwen4',{'validation'});assert len(rows)==192
    def get_records(run,step):
        path=OUT/'training'/run/'validation'/('step'+str(step)+'.json')
        value=read(path);assert value['questions']==192 and value['step']==step
        return {r['question_id']:r for r in value['records']}, {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
    nr,nref=get_records(NATIVE_RUN,0);br,bref=get_records(BRIDGE_RUN,0)
    assert nref['sha256']==acq['native_validation_sha256'] and bref['sha256']==acq['bridge_validation_sha256']
    native=[packet(nr[r['question_id']]['field']) for r in rows]
    bridge=[packet(br[r['question_id']]['field']) for r in rows]
    n=np.stack([metrics(p) for p in native]);b=np.stack([metrics(p) for p in bridge])
    arrays={'native_NLL_metrics':n,'untrained_bridge_NLL_metrics':b,
        'bridge_minus_native_teacher_state_MSE':np.array([state_mse(x,y) for x,y in zip(native,bridge)])}
    refs=[nref,bref];records=[]
    def summaries(values):
        result={c:values[np.array([r['cohort']==c for r in rows])].mean(0).tolist() for c in ['drop','quoref']}
        result['equal_cohort']=((np.array(result['drop'])+np.array(result['quoref']))/2).tolist();return result
    for run in [r['run'] for r in completed['runs']]:
        for step in [1,8,32,96]:
            rr,ref=get_records(run,step);refs.append(ref)
            packets=[packet(rr[r['question_id']]['field']) for r in rows]
            m=np.stack([metrics(p) for p in packets]);prefix=run+'__step'+str(step)
            arrays[prefix+'__NLL_metrics']=m
            for label,baseline in [('native',native),('untrained_bridge',bridge)]:
                arrays[prefix+'__teacher_MSE_vs_'+label]=np.array([state_mse(x,y) for x,y in zip(packets,baseline)])
            records.append({'run':run,'step':step,'summary':summaries(m),
                'teacher_state_MSE_vs_native':summaries(arrays[prefix+'__teacher_MSE_vs_native']),
                'teacher_state_MSE_vs_untrained_bridge':summaries(arrays[prefix+'__teacher_MSE_vs_untrained_bridge']),
                'paired_vs_untrained_bridge':{name:bootstrap(rows,m[:,i],b[:,i],2748005) for i,name in enumerate(METRICS)},
                'paired_vs_native':{name:bootstrap(rows,m[:,i],n[:,i],2748005) for i,name in enumerate(METRICS)}})
    field=commit_arrays(Path('training_bridge_baseline'),'all_validation_checkpoint_scores',arrays)
    value={'timestamp':stamp(),'all_passed':True,'execution_sha256':sha(FOLDER/'execution.json'),
        'acquisition_sha256':sha(FOLDER/'acquisition.json'),'training_result_sha256':sha(OUT/'training/result.json'),
        'questions':192,'contexts':48,'metrics':METRICS,'identities':[{k:r[k] for k in ['question_id','group_id','cohort','split']} for r in rows],
        'baseline_summary':{'native':summaries(n),'untrained_bridge':summaries(b),
            'teacher_state_MSE':summaries(arrays['bridge_minus_native_teacher_state_MSE']),
            'paired_bridge_minus_native':{name:bootstrap(rows,b[:,i],n[:,i],2748005) for i,name in enumerate(METRICS)}},
        'checkpoints':records,'validation_sources':refs,'field':field,'seconds':time.monotonic()-start,
        'limits':spec['scope']+' All comparisons use original validation material, not unseen new contexts. Later nativeBF16 behavior must be checked separately.'}
    immutable(final,value);print('NATURAL_TRAINING_BRIDGE_ANALYZED',len(records),round(value['seconds'],1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['unit','freeze','acquire','analyze']);a=p.parse_args()
    {'unit':unit,'freeze':freeze,'acquire':acquire,'analyze':analyze}[a.mode]()
