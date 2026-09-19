"""Final append-only, full-array retention, code-version and finite-scope receipt."""
import re,urllib.request,urllib.parse
from rdc_query_common import *


def http(path,params=None):
    url='http://127.0.0.1:5003/api/rdc-query'+path
    if params:url+='?'+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url,timeout=120) as response:
        assert response.status==200
        return json.load(response)


def main():
    start=time.monotonic();guard(8*1024**2)
    assert read(BASE/'queue/status.json')['all_passed'] and read(BASE/'science_queue/status.json')['all_passed']
    identity=(BASE/'identifiability/analysis/result.json').exists()
    required={
      2740:['contract.json','plan.json','material/result.json','algebra/result.json','atlas/result.json','events/result.json','analysis/phase2740.json'],
      2741:['rules/capture_result.json','rules/fit_result.json','rules/vocabulary_result.json','pairs/result.json','transfer/capture_result.json','transfer/fit_result.json','transfer/vocabulary_result.json','analysis/phase2741.json'],
      2742:['formation/result.json','transfer/injection/result.json','late/native/result.json','late/fixed128_digit/result.json','late/entropy_digit/result.json','late/entropy_letter/result.json','late/terminal_marker_digit/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json','analysis/phase2742.json'],
      2743:['followup/protocol.json','followup/result.json','followup/oracle/result.json','theory_snapshot.json','figures/index.json','next_stage_admission.json','verification/model_checkpoint_fingerprints.json','verification/scientific_integrity.json','client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json']}
    if identity:
        required[2744]=['identifiability/protocol.json','identifiability/relations/result.json','identifiability/calibration/result.json','identifiability/behavior/result.json','identifiability/analysis/result.json','identifiability/analysis/pair_change_control.json','identifiability/verification.json','continuation_after_2744.json']
        assert read(BASE/'identifiability/queue/status.json')['all_passed']
        admission=read(BASE/'continuation_after_2744.json')
        assert admission['same_authorized_goal'] and not admission['complete_stage_admitted'],'A whole valuable same-goal stage that fits cannot be left unexecuted without a new authority boundary'
    else:
        admission=read(BASE/'next_stage_admission.json')
        assert admission['same_goal'] and not admission['same_goal_complete_stage_admitted']
    required[2742].append('analysis/query_identity_control.json')
    required[2743].append('analysis/query_language_control.json')
    for paths in required.values():
        for path in paths:assert (BASE/path).is_file(),path
    for path in ['verification/model_checkpoint_fingerprints.json','verification/scientific_integrity.json','client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json']:
        assert read(BASE/path)['all_passed'],path
    codes=read(BASE/'client/code_checks.json')
    for r in codes['source_versions']:assert sha(Path(r['path']))==r['sha256'],('Source changed after scoped checks',r['path'])
    contract=read(BASE/'contract.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:contract['memo_prefix_bytes']]).hexdigest()==contract['memo_prefix_sha256']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2740]
    assert phases==list(range(2740,2745 if identity else 2744)),phases
    theory=read(BASE/'theory_snapshot.json');puzzles=43 if identity else 42;formulas=42 if identity else 40
    assert len(theory['puzzles'])==puzzles and len(theory['formulas'])==formulas
    assert not theory['RDC_primary_formula_changed'] and not theory['global_closed_theorem_added'] and not theory['new_mathematics_claimed']
    if identity:
        control=read(BASE/'identifiability/analysis/pair_change_control.json');assert control['all_passed']
        assert theory['identifiability_pair_change_control_sha256']==sha(BASE/'identifiability/analysis/pair_change_control.json')
        for key,path in {'protocol':'identifiability/protocol.json','main_analysis':'identifiability/analysis/result.json','frozen_pair_metrics':'identifiability/relations/frozen_relation_change_metrics.json.gz','decoder':'rules/decoder.npz'}.items():
            assert control['required_evidence_sha256'][key]==sha(BASE/path)
    figures=read(BASE/'figures/index.json')['figures'];review=read(BASE/'client/visual_review.json')
    seen={r['file']:r['sha256'] for r in review['scientific_figures']}
    assert set(seen)=={r['path'] for r in figures}
    for r in figures:assert sha(BASE/'figures'/r['path'])==r['sha256']==seen[r['path']]
    for r in review['browser_screenshots']:assert sha(BASE/'client'/r['file'])==r['sha256']
    assert review['human_readable_visual_inspection_performed_by_agent']
    # Verify every retained numerical archive is registered, not only the first UI page.
    science=read(BASE/'verification/scientific_integrity.json');archives=gzread(BASE/'verification/all_numerical_archives.json.gz')
    assert sha(BASE/'verification/all_numerical_archives.json.gz')==science['numerical_archive_manifest_sha256']
    expected={r['path']:r['bytes'] for r in archives};actual={};offset=0
    while True:
        page=http('/archives',{'offset':offset,'limit':1000})
        actual.update({r['path']:r['bytes'] for r in page['rows']});offset+=len(page['rows'])
        if offset>=page['total']:break
        assert page['rows']
    assert actual==expected,('Registered archives differ from full finite-array audit',len(actual),len(expected))
    current={p.relative_to(BASE).as_posix() for p in BASE.rglob('*.npz')}
    assert current==set(expected)
    for r in archives:
        p=BASE/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    import psutil
    active=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            names=[Path(v).name for v in process.info['cmdline'] or []]
            if any(n.endswith('.py') and 'rdc_query' in n and n not in (Path(__file__).name,'rdc_query_api.py') for n in names):active.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    live={}
    for name,port in [('rdc-query',5003),('rdc-update',5002),('rdc-binding',5001),('rdc-law',5001),('rdc-operator',5001),('rdc-joint',5001),('rdc-relation',5001),('rdc-prefix',5001)]:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/{name}/overview',timeout=60) as response:
            assert response.status==200;live[name]=response.status
    fingerprints=read(BASE/'verification/model_checkpoint_fingerprints.json')
    assert len(fingerprints['previous_required_artifacts_unchanged'])==47
    ledger('query_final_delivery_verification',time.monotonic()-start)
    limits=read(BASE/'resources.json');seconds=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'));guard()
    required_hashes={p:sha(BASE/p) for paths in required.values() for p in paths}
    manifest={'timestamp':stamp(),'status':'finite_integrated_query_research_and_resource_admitted_same_goal_continuation_complete',
      'phases':[{'phase':phase,'status':'executed_and_appended_to_MEMO','required_artifacts':paths} for phase,paths in required.items()],
      'required_artifact_sha256':required_hashes,'automatically_executed_same_goal_phases':[2743,2744] if identity else [2743],
      'scope':'Attachment claim correction; million full-coordinate fixed-query endpoints; ordered native source/unit/parameter accounting; frozen heldout rules and full-vocabulary controls; original native parameter learning; own-history branches; three sequential unquantized native models; reserved-document confirmation; resource-admitted strict-token/calibration continuation.',
      'hypothesis_outcomes':{'main_coordinate_and_vocabulary':theory['main_coordinate_and_vocabulary_qualification'],
        'semantic_transfer_qualification':read(BASE/'transfer/injection/result.json')['qualification'],
        'execution_checks_do_not_mean_all_hypotheses_passed':True,'global_language_mechanism_closed':False,'new_universal_mathematics_claimed':False},
      'not_claimed':['Universal language decoding or AGI','Unique semantic concept or gear per coordinate, unit or scalar parameter','A universally sufficient finite-query state','Historical pretraining reconstruction from restricted continuation learning','Source-pair allocation is unique causal semantics','Cross-model raw coordinate equality or an isolated model-size effect','A greater NLL gain identifies semantic learning','Strict token multiset control excludes all sequence or position heuristics','Parsed correct terminal answer proves intermediate reasoning or textual faithfulness','Unparsed or capped output is a complete wrong answer','All full-token/all-layer fields are retained outside the declared fixtures','User browser was controlled by successful CUA'],
      'retention':{'all_1000000_main_endpoint_full_coordinates_retained':True,'main_prefixes':10000,'main_documents':2777,'main_all_token_all_layer_fixtures':9,
        'detail_main_prefixes':576,'actual_native_generation_anchors':352,'ordered_source_paths':44,'paired_transfer_expressions':768,'reserved_followup_documents':96,
        'strict_controlled_expressions':320 if identity else 0,'formal_own_history_trajectories':2176 if identity else 576,
        'numerical_archives':len(archives),'all_saved_npz_client_registered':True,'all_saved_arrays_scanned_finite_and_hashed':True,
        'deleted_original_or_user_files':[],'reason':'All retained arrays remain queryable by original axes in the research client and support current evidence or follow-up. Conditional HiddenState cleanup is therefore not triggered.'},
      'resources':{'result_bytes_before_manifest':usage(),'result_ceiling_bytes':limits['result_ceiling_bytes'],'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'booked_script_seconds':seconds,'compute_ceiling_seconds':limits['compute_ceiling_seconds'],'no_task_model_job_running':True,'budget_exhaustion_claimed':False,
        'timing_scope':'Measured script timers including recorded failed/recovered attempts, not total wall-clock, exact GPU-busy time, code-writing or browser QA duration.'},
      'theory':{'puzzles':puzzles,'formulas':formulas,'RDC_primary_formula_changed':False,'global_closed_theorem_added':False},
      'same_goal_next_complete_stage_assessment':admission,
      'client_scope':'Live read-only API, actual-data all-axis paging and isolated headless Edge checks. Failed user-browser CUA is separately retained; no claim of controlling that browser.',
      'continuation_boundary':'The finite integrated bundle and admitted same-goal follow-ups are complete. The next full information-bearing stage is not admitted by the recorded resource gate; this is not scientific closure or a proof that all smaller pilots are impossible.'}
    save(BASE/'delivery_manifest.json',manifest)
    save(BASE/'verification/final.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'finite_delivery_complete':True,
      'general_language_mechanism_closed':False,'phases':phases,'original_memo_prefix_unchanged_bytes':contract['memo_prefix_bytes'],
      'registered_numerical_archives':len(archives),'native_original_models_preserved':len(fingerprints['models']),'prior_required_artifacts_preserved':47,
      'indexed_scientific_figures':len(figures),'theory_puzzles':puzzles,'theory_formulas':formulas,'live_routes':live,
      'manifest_sha256':sha(BASE/'delivery_manifest.json'),'result_bytes_before_final':usage()})
    guard();print('QUERY_FINAL_DELIVERY_PASS',phases,len(archives),usage(),flush=True)


if __name__=='__main__':main()
