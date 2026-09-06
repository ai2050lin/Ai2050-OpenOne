"""Frozen second-group actual single-weight confirmation; delivery finalizes this Phase."""
from phase2633_shared_weight_forward_tests import run, summarize, load_model, SOURCE
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2635_expanded_native_path_confirmation'

def main():
    frames=[f for f in read(SOURCE/'material/frames.json') if f['index']==32 and f['step']==0]
    assert len(frames)==32 and len({(f['family'],f['language']) for f in frames})==16
    if (OUT/'analysis/interventions.json').exists():raise RuntimeError('confirmation already exists; do not replace')
    records={r['frame_id']:r for r in read(SOURCE/'analysis/records.json')}
    save(OUT/'protocol/frozen.json',{'frame_ids':[f['frame_id'] for f in frames],
        'selection':'index32, both variants, step0, eight families x two languages; doubles 2633 frame count',
        'same_algorithm_no_retuning':True,'sites':28,'selectors':['per-frame diagnostic maximum','matched fixed index'],
        'rms_scales':[.2,1.0],'signs':[-1,1],'no_donor':True,'nonquantized':'local Qwen3-4B CUDA BF16',
        'limits':'New numerical audit contexts, not an unseen semantic lockbox; items/templates shared with previous form0, gradients already mapped in2632.',
        'script_sha256':{name:sha(TESTS/name) for name in ['phase2633_shared_weight_forward_tests.py','phase2635_expanded_native_path_confirmation.py']}})
    save(OUT/'material/frames.json',frames)
    model,tok=load_model('qwen4');outputs=run(model,frames,records,OUT)
    summary=summarize(outputs)
    checks={'all32_frames':len(frames)==32,'all7200_condition_forwards':len(outputs)==7200,
        'noops_identical':all(r['margin16_change']==0 and r['margin32_change']==0 and r['state_l2_change']==0 for r in outputs if r['kind']=='noop'),
        'all28_weight_hashes_restored':read(OUT/'analysis/weight_restoration.json')['before']==read(OUT/'analysis/weight_restoration.json')['after'],
        'baseline_same_as_adjoint':max(r['baseline32_vs_adjoint_error'] for r in outputs if r['kind']=='shared_weight')==0}
    save(OUT/'analysis/confirmation.json',{'summary':summary,'checks':checks,'all_checks_passed':all(checks.values()),
        'note':'This is completed computation, not final Phase delivery. Client, cleanup and verification remain.'})
    print(json.dumps({'confirmation_checks':checks,'key_results':{k:summary[k] for k in ['all','L0/v_proj','L35/v_proj','L35/down_proj']}},ensure_ascii=True),flush=True)
    assert all(checks.values())

if __name__=='__main__':main()
