"""Finite campaign handoff, after scientific, browser and append-only MEMO verification."""
import hashlib
import re
import urllib.request
from rdc_operator_common import *


def main():
    start=time.monotonic();guard(3*1024**2)
    required={
        2724:['review.json','material_audit.json','qa_extension.json','capture/main/result.json','observation/result.json','qa/qwen4/main/result.json','qa_atlas/result.json'],
        2725:['operators/result.json','operators/frozen.json','calculus/result.json','structure/result.json','confirmation/result.json'],
        2726:['capture/confirmation/result.json','qa/qwen4/confirmation/result.json','compiled/result.json','compiled/paired_audit.json',
            'scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json','qa/qwen14/main/result.json','qa/glm4/main/result.json',
            'operations/result.json','operations/response_agreement_audit.json','precision/result.json','identity_audit/correction_result.json','identity_audit/corrected_cue_composition.json','behavior/result.json','scale/paired_audit.json'],
        2727:['metric_followup/protocol.json','metric_followup/result.json','metric_followup/paired_audit.json','metric_followup/math_check_adaptive.json','metric_followup/math_check_readout.json','metric_followup/population_geometry/result.json','metric_followup/population_geometry/math_check.json','theory_snapshot.json',
            'verification/scientific_integrity.json','verification/client_contract.json','verification/browser.json',
            'verification/model_checkpoint_fingerprints.json','verification/frontend_build.json']}
    for group in required.values():
        for name in group:assert (BASE/name).is_file(),name
    for name in ('scientific_integrity','client_contract','browser','frontend_build'):
        assert read(BASE/'verification'/f'{name}.json')['passed'],name
    build=read(BASE/'verification/frontend_build.json')
    for name,digest in {**build['source_sha256'],**build['compiled_python_files']}.items():
        assert sha(ROOT/name)==digest,('Source changed after final build/compilation',name)
    contract=read(BASE/'verification/client_contract.json');science=read(BASE/'verification/scientific_integrity.json')
    assert contract['all_stored_npz_paths_registered']
    assert contract['registered_array_files']==science['all_npz_files']
    for key in ('server_source','frontend_source'):
        assert sha(ROOT/contract[key]['path'])==contract[key]['sha256']
    prefix=read(BASE/'memo_prefix.json');memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2724]
    assert phases==[2724,2725,2726,2727],phases
    figs=read(BASE/'figures/index.json')['figures'];assert len(figs)>=16
    for r in figs:assert sha(BASE/'figures'/r['path'])==r['sha256']
    live={}
    for name in ('rdc-operator','rdc-joint','rdc-relation','rdc-prefix'):
        with urllib.request.urlopen(f'http://127.0.0.1:5001/api/{name}/overview',timeout=30) as r:
            assert r.status==200;live[name]=r.status
    import psutil
    active=[]
    gpu_scripts=['rdc_operator_qa.py','phase2726_rdc_operator_scale.py','phase2726_rdc_operator_operations.py','phase2727_rdc_operator_metric.py','phase2727_rdc_operator_population_geometry.py']
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if any(any(Path(part).name==name for part in cmd) for name in gpu_scripts):active.append({'pid':p.info['pid'],'cmdline':cmd})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    source_paths=sorted({*list((ROOT/'tests/glm5').glob('phase272[4-7]_rdc*.py')),*list((ROOT/'tests/glm5').glob('rdc_operator*.py')),
        ROOT/'tests/glm5/rdc_native_conditional_operator.py',ROOT/'server/rdc_operator_service.py',
        ROOT/'frontend/src/components/app/RdcOperatorAtlas.jsx',ROOT/'frontend/src/components/app/RdcOperatorAtlas.css'})
    snapshots=[snapshot(p) for p in source_paths]
    ledger('final_operator_campaign_delivery',time.monotonic()-start)
    computation=read(BASE/'compute_ledger.json');resources=read(BASE/'resources.json');actual=usage()
    assert actual<resources['result_ceiling_bytes'] and sum(r['seconds'] for r in computation)<resources['compute_ceiling_seconds']
    manifest={'timestamp':stamp(),'status':'bounded_operator_campaign_completed','phases':[
        {'phase':phase,'status':'executed_and_recorded','artifacts':files} for phase,files in required.items()],
        'automatic_same_goal_followup_executed':[2727],
        'scope':'Audited both attachments; completed integrated ordinary-language/operator/compiled-behavior plan and a substantive automatic output-metric/own-history followup. Finite research campaign completed, not AGI or general language mechanism closure.',
        'original_plan_is_preserved_historical_design':'plan.json',
        'scientific_limits':['Lexical cues/token pieces are not a complete semantic or syntactic graph.',
            'Question tasks are with-context reading comprehension, not closed-book parameter knowledge; nonmatching strings require interpretation.',
            'Local operators receive actual current normalized x; other network modules remain native, so this is not an extracted full language algorithm.',
            'Natural confirmation in2727 is re-analysis; QA task-transfer boundary was new to operator testing, but its baseline behavior was observed.',
            'Known SwiGLU/Jacobian/softmax identities and empirical fits are not new mathematics or a universal gear theorem.',
            'Own-history approximation and full-language generalization remain unclosed; no demonstrated pulse-reset or hallucination cure.'],
        'next_big_question':'Extract transferable relation-conditioned update rules from available source/history information, with capacity-matched full-coordinate fits, independent family validation and full-output behavior. Do not expand rare-event counting as the main route.',
        'retention':{'all2048sources':'Every token/every layer/full native coordinates processed;2anchors per source plus all-layer energies and full-coordinate moments retained.',
            'all_token_raw_fixtures':16,'MLP':'All9728 units at3primary4Bblocks, complete K/J representatives, native scalar access and matched model factor fields retained.',
            'QA':'512original native question runs across3models,64question-order counterparts, actual inputs/outputs/all-layer query fields retained.',
            'autonomous':'64sources x3original branches plus64hybrid branches; hybrid and separate same-chosen-history native complete query fields at4boundaries.',
            'output_geometry':'Two predetermined query matrices plus six complete fitted population matrices; all640query full-coordinate errors/readout means, article split and scalar variance predictions retained.',
            'corrections':'Original data unchanged;4partial-byte cue corrections and full BF16 cancellation vectors retained.',
            'all_material_result_arrays_client_registered':True,'deleted_user_or_original_field_files':[]},
        'resources':{'bytes_before_manifest':actual,'result_ceiling_bytes':resources['result_ceiling_bytes'],'disk_free_bytes':shutil.disk_usage(BASE).free,
            'booked_script_seconds_including_allowances':sum(r['seconds'] for r in computation),'compute_ceiling_seconds':resources['compute_ceiling_seconds'],
            'timing_scope':'Script elapsed timers plus explicitly marked conservative allowances for interrupted tails; not total human/UI elapsed time or GPU kernel-only time.',
            'no_task_CUDA_job_running':True,'budget_exhaustion_claimed':False}}
    save(BASE/'delivery_manifest.json',manifest)
    final={'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),'phases':phases,'original_memo_bytes_unchanged':prefix['bytes'],
        'live_full_service_routes':live,'indexed_figures':len(figs),'all_array_checks':'verification/scientific_integrity.json',
        'browser_checks':'verification/browser.json','source_snapshots':snapshots,'delivery_manifest_sha256':sha(BASE/'delivery_manifest.json'),
        'result_bytes_before_final':usage(),'finite_delivery_complete':True,'general_language_mechanism_closed':False}
    save(BASE/'verification/final.json',final);guard()
    print('FINAL_OPERATOR_DELIVERY_PASS',phases,'bytes',usage(),'of',resources['result_ceiling_bytes'],flush=True)


if __name__=='__main__':main()
