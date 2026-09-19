"""Seal bounded campaign delivery without changing frozen scientific results."""
import re
from rdc_feature_common import *
from phase2693_rdc_delivery_audit import get

def main():
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';data=memo.read_bytes()
    prefix=hashlib.sha256(data[:9286774]).hexdigest()
    assert prefix=='437eee1c4a1475e53887cd79ae838fbd726c40cf73e3f81c4bba18f06cfc9969'
    text=data.decode('utf-8-sig')
    for phase in (2691,2692,2693,2694):assert len(re.findall(rf'^## Phase {phase}:',text,re.M))==1
    s0=read(CAMPAIGN/'s0/result.json');s1=read(CAMPAIGN/'s1/result.json');s2=read(CAMPAIGN/'s2pilot/result.json')
    a1=read(CAMPAIGN/'s1/delivery_audit.json');a2=read(CAMPAIGN/'s2pilot/delivery_audit.json');review=read(CAMPAIGN/'s2pilot/confirmation_review.json')
    ui=read(CAMPAIGN/'client/verification.json')
    assert a1['cases']==a2['cases']==512 and review['checks']['frozen_model_sha_unchanged']==40
    assert ui['status']=='verified_with_real_local_artifacts' and not ui['checks']['browser_page_errors']
    selected=get('/runs/s2pilot/field',sample='s2-fruit-0-0-0-en',field='h',layer=3,layers=1,token=97,tokens=1,coordinate=104,width=1)
    assert selected['values']==[[[.119140625]]]
    page=get('/runs/s2pilot/field',sample='s2-fruit-0-0-0-en',field='h',layer=0,layers=1,token=82,tokens=1,coordinate=128,width=1)
    assert page['values'][0][0][0]==ui['checks']['coordinate_page_start128_actual_first_value']
    core=[memo,ROOT/'server/rdc_feature_service.py',ROOT/'frontend/src/components/app/RdcFeatureAtlas.jsx',ROOT/'frontend/src/components/app/RdcFeatureAtlas.css',
        ROOT/'frontend/src/main.jsx',CAMPAIGN/'s1/protocol.json',CAMPAIGN/'s1/benchmark_protocol.json',CAMPAIGN/'s1/result.json',
        CAMPAIGN/'s2pilot/protocol.json',CAMPAIGN/'s2pilot/result.json',CAMPAIGN/'s2pilot/confirmation_review.json',
        CAMPAIGN/'s1/delivery_audit.json',CAMPAIGN/'s2pilot/delivery_audit.json',CAMPAIGN/'client/verification.json']
    result={'timestamp':stamp(),'status':'bounded_approved_campaign_complete','phases':[2691,2692,2693,2694],
        'old_phase2691_original_campaign_complete':False,'s0_synthetic_cases':1536,'s1_actual_cases':512,'s2_pilot_actual_cases':512,
        's1_comparisons':len(s1['results']),'s1_random_label_controls':len(s1['random_word_controls']),
        's1_future_predictions_and_baselines':len(s1['future_field_prediction']),'s2_frozen_comparisons':len(s2['results']),
        's2_confirmed_readability_directions':len(s2['accepted_directions']),
        'memo_prior_prefix_unchanged':True,'prior_prefix_bytes':9286774,'prior_prefix_sha':prefix,
        'raw_field_bytes_retained':a1['field_bytes_retained']+a2['field_bytes_retained'],
        'native_hidden_scalars_audited':a1['native_hidden_scalars']+a2['native_hidden_scalars'],
        'deleted_files':0,'language_encoding_mechanism_solved':False,
        'client_url':'http://localhost:5173/rdc','core_hashes':{str(p.relative_to(ROOT)):sha(p) for p in core},
        'scope_limits':['S0 cubic all-block calibration failure retained with separate restricted-block diagnostic.',
            'S1 joint answer target degeneracy corrected in appended review, not silently overwritten.',
            'S2 is a512-input bounded prospective confirmation; full3072/16384/49152 stages and Q14 cross-model confirmation not executed.',
            'No ongoing model computation or unconditional background campaign is asserted. Read-only client server remains available.']}
    save(CAMPAIGN/'delivery.json',result)
    save(CAMPAIGN/'handoff.json',{'timestamp':stamp(),'current_phase':2694,'state':'bounded_campaign_complete',
        'read_first':['delivery.json','s1/delivery_audit.json','s2pilot/confirmation_review.json','s2pilot/result.json'],
        'deprecated_entrypoints':['phase2691_serial_tail.py','phase2693_campaign_terminal.py'],
        'do_not_resume_old_queue':True,'old_scope':'original four-protocol2691 incomplete, explicitly archived/deferred by user',
        'completed':'S0, S1, live native-coordinate client,512 fresh balanced prompts with frozen readers and whole-coordinate contributions',
        'next_question':'Separate lexical-class readability from relation/role computation and native next-token readout. Reuse stored full fields and frozen predictors before collecting large new corpora.',
        'next_phase':2695,'next_phase_started':False,
        'next_plan':['On retained data, compare U/V/C and norm/lexical contributions without coordinate selection; keep all-coordinate ledgers.',
            'Freeze role-swap, meaning-preserving rewrite and negation-scope prediction contrasts with independent label balance.',
            'Link candidate coordinates to actual native weights and natural first-divergence probabilities; no delta transport as core method.',
            'Pilot cost before expanded cross-family or nonquantized Qwen14 test; one CUDA model and8GiB disk floor.'],
        'client':{'url':'http://localhost:5173/rdc','backend_port':5001,'frontend_port':5173,
            'backend_launch':'AI2050_SKIP_MODEL_LOAD=1, CUDA_VISIBLE_DEVICES=-1; .venv/Scripts/python.exe server/server.py',
            'note':'Keep native arrays: all are accessible by stable-ID queries in client; no data deleted.'}})
    for run in ('s1','s2pilot'):
        status(run,state='complete',completed=512,total=512,source_mode='recorded_model_samples',delivery='../delivery.json')
        event(run,'delivery_complete',phase=2693 if run=='s1' else 2694,language_mechanism_closed=False)
    print('DELIVERY_DONE',result['native_hidden_scalars_audited'],result['raw_field_bytes_retained'],flush=True)

if __name__=='__main__':main()
