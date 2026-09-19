"""Verify completed finite work, append-only narrative and bounded continuation."""
import re
import urllib.request
from rdc_update_common import *


def main():
    start=time.monotonic();guard(8*1024**2)
    required={
      2736:['contract.json','plan.json','material_frozen.json','graph/frozen.json','graph/pilot.json',
        'capture/result.json','graph/confirmation.json','analysis/phase2736.json'],
      2737:['learning/frozen.json','learning/finite_result.json','learning/forecast_audit.json',
        'learning/autograd_audit.json','middle_training/result.json'],
      2738:['language_material_frozen.json','language_capture/result.json','language_analysis/result.json',
        'language_analysis/logic_audit.json','language_identity/result.json','language_prediction/result.json','causal_anchor/result.json','causal_replay/result.json',
        'native_paths/result.json','own_history/result.json','same_history/result.json',
        'scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json',
        'scale_batch/qwen4/result.json','scale_analysis/result.json'],
      2739:['fresh_graph/material_audit.json','fresh_graph/frozen.json','fresh_graph/result.json',
        'moment_boundary/result.json','predictive_state/result.json','long_answers/result.json',
        'behavior_analysis/result.json','terminal_format_audit/result.json','manual_terminal_audit/result.json','theory_snapshot.json','figures/index.json','next_stage_admission.json',
        'verification/model_checkpoint_fingerprints.json','verification/scientific_integrity.json',
        'client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json']}
    for paths in required.values():
        for path in paths:assert (BASE/path).is_file(),path
    for path in ('verification/model_checkpoint_fingerprints.json','verification/scientific_integrity.json',
      'client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json','terminal_format_audit/result.json','manual_terminal_audit/result.json'):
        assert read(BASE/path)['all_passed'],path
    science=read(BASE/'verification/scientific_integrity.json');checks=read(BASE/'client/code_checks.json')
    for r in checks['source_versions']:assert sha(Path(r['path']))==r['sha256'],('Changed since checks',r['path'])
    contract=read(BASE/'contract.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:contract['memo_prefix_bytes']]).hexdigest()==contract['memo_prefix_sha256']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2736]
    assert phases==[2736,2737,2738,2739],phases
    figures=read(BASE/'figures/index.json')['figures'];review=read(BASE/'client/visual_review.json')
    seen={r['file']:r['sha256'] for r in review['scientific_figures']}
    assert set(seen)=={f['path'] for f in figures}
    for f in figures:assert sha(BASE/'figures'/f['path'])==f['sha256']==seen[f['path']]
    for f in review['browser_screenshots']:assert sha(BASE/'client'/f['file'])==f['sha256']
    theory=read(BASE/'theory_snapshot.json');assert len(theory['puzzles'])==38 and len(theory['formulas'])==31
    assert not theory['global_closed_theorem_added'] and not theory['new_mathematics_claimed']
    admission=read(BASE/'next_stage_admission.json')
    assert admission['same_long_term_goal'] and not admission['full_next_stage_admitted'],'If a complete information-bearing continuation fits, reassess and execute it instead of claiming a resource boundary'
    assert read(BASE/'behavior_analysis/result.json')['trajectories']==1464
    import psutil
    active=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            names=[Path(v).name for v in process.info['cmdline'] or []]
            if any(v.endswith('.py') and 'rdc_update' in v and v not in (Path(__file__).name,'rdc_update_api.py') for v in names):active.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    live={}
    for name in ('rdc-update','rdc-binding','rdc-law','rdc-operator','rdc-joint','rdc-relation','rdc-prefix'):
        port=5002 if name=='rdc-update' else 5001
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/{name}/overview',timeout=60) as response:
            assert response.status==200;live[name]=response.status
    files=sorted((ROOT/'tests/glm5').glob('*rdc_update*.py'))+[ROOT/'server/rdc_update_service.py',
      ROOT/'frontend/src/components/app/RdcUpdateAtlas.jsx',ROOT/'frontend/src/components/app/RdcUpdateAtlas.css',
      ROOT/'frontend/src/main.jsx',ROOT/'server/server.py',ROOT/'tests/glm5/rdc_operator_model.py',
      ROOT/'tests/glm5_temp/rdc_update_api.py']
    snapshots=[snapshot(p) for p in files]
    required_hashes={p:sha(BASE/p) for paths in required.values() for p in paths}
    ledger('update_final_delivery_verification',time.monotonic()-start)
    resources=read(BASE/'resources.json');seconds=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'))
    assert seconds<resources['compute_ceiling_seconds']
    manifest={'timestamp':stamp(),'status':'finite_integrated_delivery_and_one_information_bearing_same_goal_followup_complete',
      'phases':[{'phase':p,'status':'executed_and_appended_to_MEMO','artifacts':v} for p,v in required.items()],
      'automatic_same_goal_followup_executed':[2739],'required_artifact_sha256':required_hashes,
      'scope':'Attachment corrections; whole-coordinate source/response rules; complete native parameter calculus; actual32-step continuation training; bilingual multi-family atlas; own and same histories; sequential unquantized native models; matched fresh confirmation; finite moment/information boundaries and conservative long answers.',
      'not_claimed':['Universal language decoding or AGI','New fundamental mathematics or global closed RDC theorem',
        'A unique semantic concept or causal gear per coordinate/parameter','Reconstruction of original pretraining',
        'Unquantized BF16 batch1 and padded batch8 numerical equality','A pure model-size effect independent of tokenizer/architecture/training/execution shape',
        'Semantic cross-language invariance independent of lexical identity','A later style instruction causally changes an earlier identical-prefix body state','Every saved source attribution is a causal path',
        'Every all-token/all-layer field is retained outside declared fixtures','Synthetic moment collision is a reachable natural-state collision',
        'Unparsed censored generations are complete wrong answers','Unblinded post-outcome terminal adjudication is independent confirmation','Repeated diagnostic rollouts are independent confirmation'],
      'retention':{'client_material_ids':2176,'new_Q4_materials':1664,'reused_strict_natural':512,
        'new_natural_windows':256,'mixed_program_expressions':768,'bilingual_family_expressions':640,
        'full_layer_all_token_fixtures':len(science['full_layer_all_token_fixtures']),
        'full_native_fields':'All anchor coordinates and early causal sources, all selected MLP units; actual full-layer/full-token scans carry per-layer hashes and frozen recomputation inputs.',
        'native_scale':'128 matched frozen materials per model; separate Q4 batch8 shadow plus six nativeB1 prefill controls per model. Q4 original128+36B1 unchanged; larger-model generation batched with own row histories.',
        'trajectory_records':1464,'all_saved_npz_client_registered':True,'all_saved_arrays_finite_and_hashed':True,
        'deleted_original_or_user_files':[],'reason_for_retention':'All saved arrays are queryable by the research client and support mechanism evidence or follow-up; the conditional cleanup request therefore does not apply.'},
      'resources':{'bytes_before_manifest':usage(),'result_ceiling_bytes':resources['result_ceiling_bytes'],
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'booked_script_seconds':seconds,
        'compute_ceiling_seconds':resources['compute_ceiling_seconds'],'no_task_model_job_running':True,
        'timing_scope':'Measured nonnested script timers including measured failed/recovered attempts; not total wall-clock, exactGPU utilization, implementation or browser duration. Recovery receipts preserve attempts without invented timing.',
        'budget_exhaustion_claimed':False},
      'client_scope':'Live read-only API and isolated headless Edge verified. User-browser CUA initialization failed; headless checks do not claim control of the user browser.',
      'next_big_question':theory['next_big_question'],'next_stage_not_executed':theory['next_stage_not_executed'],
      'same_goal_continuation_resource_assessment':admission,
      'continuation_boundary':'2739 executed the planned same-goal follow-up. Current finite campaign complete; remaining directed natural-event identifiability, sufficient ordered state and useful learned effects are open research. Further collection must pass the remaining resource/information-value gate.'}
    save(BASE/'delivery_manifest.json',manifest)
    save(BASE/'verification/final.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
      'finite_delivery_complete':True,'general_language_mechanism_closed':False,'phases':phases,
      'original_memo_prefix_unchanged_bytes':contract['memo_prefix_bytes'],'material_rows':2176,
      'full_array_count':science['all_arrays'],'registered_npz_files':science['all_npz_files'],
      'indexed_scientific_figures':len(figures),'theory_puzzles':38,'theory_formulas':31,
      'live_routes':live,'source_snapshots':snapshots,'manifest_sha256':sha(BASE/'delivery_manifest.json'),
      'result_bytes_before_final':usage()});guard();print('UPDATE_FINAL_DELIVERY_PASS',phases,'bytes',usage(),flush=True)


if __name__=='__main__':main()
