"""Verify the bounded integrated campaign and its actual automatic follow-up."""
import re
import urllib.request
from rdc_binding_common import *

def main():
    start=time.monotonic();guard(4*1024**2)
    required={
      2732:['contract.json','plan.json','prior_material_audit.json','material_frozen.json',
        'prediction/frozen.json','capture/result.json','confirmation/result.json','analysis/result.json','verification/kernel_math.json'],
      2733:['native_bilinear/result.json','middle_training/pilot.json','middle_training/result.json',
        'alpha_natural/result.json','gradient_span/result.json','beta_updates/result.json'],
      2734:['binding_live/result.json','scale/suite_result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/qwen14/recovery_verification.json','scale/glm4/result.json','atlas/result.json'],
      2735:['format_content/protocol.json','format_content/native_capture_result.json','format_content/decomposition_result.json',
        'format_content/numerical_recovery/failed_attempt.json','format_content/numerical_recovery/precision_diagnostic.json',
        'format_content/projection_condition_audit.json',
        'format_content/full_factor_export.json',
        'format_content/autonomous/result.json','format_content/suite_result.json','analysis/behavior.json',
        'verification/content_format_math.json','verification/source_moment_collision/result.json',
        'signed_source/protocol.json','signed_source/frozen.json','signed_source/capture_result.json','signed_source/math.json','signed_source/result.json',
        'signed_source/identity_recovery/result.json',
        'theory_snapshot.json','figures/index.json','verification/scientific_integrity.json',
        'verification/model_checkpoint_fingerprints.json','verification/api.json','verification/browser.json','verification/build.json']}
    for paths in required.values():
        for path in paths:assert (BASE/path).is_file(),path
    for name in ('scientific_integrity','model_checkpoint_fingerprints','api','browser','build'):
        assert read(BASE/'verification'/f'{name}.json')['all_passed'],name
    science=read(BASE/'verification/scientific_integrity.json');api=read(BASE/'verification/api.json')
    assert science['all_npz_files']==api['registered_npz_files']
    assert api['every_saved_npz_registered'] and api['samples']==api['unique_sample_ids']==science['material_rows']==1664
    build=read(BASE/'verification/build.json')
    for path,digest in build['source_sha256'].items():assert sha(ROOT/path)==digest,('Changed after build',path)
    contract=read(BASE/'contract.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:contract['memo_prefix_bytes']]).hexdigest()==contract['memo_prefix_sha']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2732]
    assert phases==[2732,2733,2734,2735],phases
    figures=read(BASE/'figures/index.json')['figures'];browser=read(BASE/'verification/browser.json')
    assert len(figures)==17
    reviewed={r['file']:r['sha256'] for r in browser['reviewed_scientific_figures']}
    for f in figures:
        digest=sha(BASE/'figures'/f['path']);assert digest==f['sha256']==reviewed[f['path']]
    theory=read(BASE/'theory_snapshot.json');assert len(theory['puzzles'])==34 and len(theory['formulas'])==23
    assert not theory['new_mathematics_claimed'] and not theory['global_closed_theorem_added']
    live={}
    for name in ('rdc-binding','rdc-law','rdc-operator','rdc-joint','rdc-relation','rdc-prefix'):
        with urllib.request.urlopen(f'http://127.0.0.1:5001/api/{name}/overview',timeout=30) as response:
            assert response.status==200;live[name]=response.status
    import psutil
    active=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            names=[Path(part).name for part in process.info['cmdline'] or []]
            if any(name.endswith('.py') and 'rdc_binding' in name and name!=Path(__file__).name for name in names):active.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    files=sorted((ROOT/'tests/glm5').glob('*rdc_binding*.py'))+[ROOT/'server/rdc_binding_service.py',
      ROOT/'frontend/src/components/app/RdcBindingAtlas.jsx',ROOT/'frontend/src/components/app/RdcBindingAtlas.css']
    snapshots=[snapshot(p) for p in files]
    ledger('binding_final_delivery_verification',time.monotonic()-start)
    resources=read(BASE/'resources.json');seconds=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'))
    assert seconds<resources['compute_ceiling_seconds']
    manifest={'timestamp':stamp(),'status':'finite_integrated_delivery_and_same_goal_followup_complete',
      'phases':[{'phase':phase,'status':'executed_and_appended_to_MEMO','artifacts':paths} for phase,paths in required.items()],
      'automatic_same_goal_followup_executed':[2735],
      'scope':'Attachment audit; strict natural source binding; full native scalar/unit relations; actual middle-layer training; Alpha/Beta/Gamma; sequential nonquantized three-model observations; own-history deployment; automatic content/format and signed-source diagnostics.',
      'not_claimed':['Universal language decoding or AGI','Original pretraining-history reconstruction','A unique semantic gear per coordinate or parameter',
        'Native reachable-state failure established by a synthetic moment collision','Unsupervised or zero-shot status for gold oracle updates',
        'Semantic isomorphism from cross-model relation correlations','Complete language performance from a lower first-digit loss'],
      'retention':{'all_saved_npz_client_registered':True,'all_saved_arrays_numerically_audited':True,
        'linked_old_natural':{'rows':512,'tokens':29961,'all_layer_anchors':1536},
        'new_Q4_material':{'source_occurrences':1152,'tokens_counting_occurrences':142745,'all_layer_anchors_counting_occurrences':1408,
          'actual_new_capture_forwards':1151,'signed_identical_input_field_aliases':1,
          'scope':'128 signed natural occurrences contain127unique full native inputs; the exact duplicate is explicitly indexed and reused, not counted as a new forward.'},
        'full_layer_all_token_fixtures':16,'old_natural_fields_copied':False,
        'new_natural_windows':256,'program_expressions':896,
        'native_scale':'128same source rows per model, independent native tokenizers/widths; three models loaded serially.',
        'own_history':'80natural branch trajectories;192long native program trajectories;336parameter-update/native program trajectories.',
        'deleted_original_or_user_files':[],
        'not_falsely_retained':'Everylayer/everytoken raw fields outside16declared fixtures are scanned/summarized, not all archived. Complete H12sources and allanchorcoordinates remain available.'},
      'resources':{'bytes_before_manifest':usage(),'result_ceiling_bytes':resources['result_ceiling_bytes'],
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'booked_script_seconds':seconds,'compute_ceiling_seconds':resources['compute_ceiling_seconds'],
        'timing_scope':'Measured stage timers, not unique GPU time or total wall-clock/UI/implementation time. The identity-repair timer includes its separately booked evaluator. Rejected CPU material selection and first identity-repair attempts have no precise timers; see signed_source/recovery.json and signed_source/identity_recovery/recovery_note.json.',
        'no_task_model_job_running':True,'budget_exhaustion_claimed':False},
      'next_big_question':'Identify which source/role/position information is sufficient for downstream conditional computation, not merely distinguishable; test available-prefix edge-directed signed/nonlinear updates, useful parameter formation, unseen operations and own histories together.',
      'continuation_boundary':'The frozen finite integrated campaign and one same-goal diagnostic extension are complete. Further hypothesis families and model training remain open research, not a claimed closed mechanism.'}
    save(BASE/'delivery_manifest.json',manifest)
    save(BASE/'verification/final.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'finite_delivery_complete':True,'general_language_mechanism_closed':False,'phases':phases,
      'original_memo_prefix_unchanged_bytes':contract['memo_prefix_bytes'],'material_rows':1664,
      'full_array_count':science['all_arrays'],'registered_npz_files':science['all_npz_files'],
      'indexed_scientific_figures':len(figures),'theory_puzzles':34,'theory_formulas':23,
      'live_routes':live,'source_snapshots':snapshots,'manifest_sha256':sha(BASE/'delivery_manifest.json'),
      'result_bytes_before_final':usage()});guard()
    print('BINDING_FINAL_DELIVERY_PASS',phases,'bytes',usage(),flush=True)

if __name__=='__main__':main()
