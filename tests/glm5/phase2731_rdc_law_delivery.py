"""Finite integrated research campaign handoff after actual scientific/client checks."""
import hashlib
import re
import urllib.request
from rdc_law_common import *


def main():
    start=time.monotonic();guard(3*1024**2)
    required={
        2728:['review.json','material_audit.json','material_eligibility.json','capture/main/result.json','atlas/result.json',
              'formation/initial_stable/result.json','formation/gradient_controls/result.json','verification/math.json'],
        2729:['prediction/protocol.json','prediction/frozen.json','prediction/result.json','formation/controlled_capture/result.json',
              'formation/trajectories/result.json','analysis/result.json','verification/prediction_math.json'],
        2730:['capture/confirmation/result.json','confirmation/result.json','confirmation/combination_visibility/result.json','deployment/protocol.json','deployment/first_gold_scoring_protocol.json',
              'deployment/result.json','deployment/replay/result.json','deployment/scope/result.json','verification/cache_math.json',
              'deployment/scope_precision/result.json','deployment/scope_precision/readout_precision.json','deployment/resume_manifest.json',
              'scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json'],
        2731:['own_history/protocol.json','own_history/result.json','deployment/paired_analysis.json','theory_snapshot.json','figures/index.json',
              'verification/scientific_integrity.json','verification/client_api.json','verification/browser.json',
              'verification/model_checkpoint_fingerprints.json','verification/frontend_build.json']}
    for names in required.values():
        for name in names:assert (BASE/name).is_file(),name
    for name in ('scientific_integrity','client_api','browser','frontend_build'):
        assert read(BASE/'verification'/f'{name}.json')['passed'],name
    build=read(BASE/'verification/frontend_build.json')
    for path,digest in {**build['source_sha256'],**build['compiled_python_files']}.items():
        assert sha(ROOT/path)==digest,('Sourcechangedafterfinalbuild',path)
    science=read(BASE/'verification/scientific_integrity.json');api=read(BASE/'verification/client_api.json')
    assert api['all_stored_npz_paths_registered'] and api['registered_array_files']==science['all_npz_files']
    assert sha(ROOT/api['server_source']['path'])==api['server_source']['sha256']
    prefix=read(BASE/'memo_prefix.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2728]
    assert phases==[2728,2729,2730,2731],phases
    figures=read(BASE/'figures/index.json')['figures'];assert len(figures)>=15
    for f in figures:assert sha(BASE/'figures'/f['path'])==f['sha256'],f['path']
    theory=read(BASE/'theory_snapshot.json');assert len(theory['puzzles'])==30 and len(theory['formulas'])==16
    assert not theory['new_mathematics_claimed'] and not theory['global_closed_theorem_added']
    live={}
    for name in ('rdc-law','rdc-operator','rdc-joint','rdc-relation','rdc-prefix'):
        with urllib.request.urlopen(f'http://127.0.0.1:5001/api/{name}/overview',timeout=30) as response:
            assert response.status==200;live[name]=response.status
    import psutil
    active=[]
    scripts={'phase2728_rdc_law_capture.py','phase2728_rdc_law_control_capture.py','phase2728_rdc_law_formation.py',
        'phase2729_rdc_law_training.py','phase2729_rdc_law_prediction.py','phase2730_rdc_law_scale.py','phase2730_rdc_law_deployment.py','phase2730_rdc_law_scope_precision.py','phase2731_rdc_law_own_history.py'}
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if any(Path(part).name in scripts for part in process.info['cmdline'] or []):active.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    sourcefiles=sorted(set((ROOT/'tests/glm5').glob('*rdc_law*.py'))|{ROOT/'server/rdc_law_service.py',
        ROOT/'frontend/src/components/app/RdcLawAtlas.jsx',ROOT/'frontend/src/components/app/RdcLawAtlas.css'})
    snapshots=[snapshot(p) for p in sourcefiles]
    ledger('final_law_campaign_delivery',time.monotonic()-start)
    resources=read(BASE/'resources.json');computation=read(BASE/'compute_ledger.json');actual=usage()
    assert actual<resources['result_ceiling_bytes'] and sum(r['seconds'] for r in computation)<resources['compute_ceiling_seconds']
    manifest={'timestamp':stamp(),'status':'bounded_formation_operation_composition_campaign_completed',
        'phases':[{'phase':phase,'status':'executed_and_appended_to_MEMO','artifacts':paths} for phase,paths in required.items()],
        'automatic_same_goal_followup_executed':[2731],
        'scope':'Threeattachment audit and integrated natural-allcoordinate/nativeparameter/controlledformation/newcombination/deployment plan, followed automatically by same-ownhistory nativecache/output diagnosis. Finite deliverables completed; general language mechanism and AGI remain open.',
        'historical_design_preserved':'plan.json; material_audit.json records before-capture eligibility refinements, includingEWT replacement and actualavailableHotpotcontexts.',
        'scientific_limits':['Retrospective dependencycooccurrence and tasklabels are not established compositional semanticprograms.',
            'Completecoordinates/allunits do not make conditionalaverages a sufficient state, nor turn correlation into computation.',
            'Training is restrictedactual lastMLP continuation withtwo minibatch-orderseeds, not originalpretraining formation.',
            'PartialMLPprediction retains othernative modules; localMSE, fullvocabKL, fullanswers and ownhistory are separate outcomes.',
            'Known kernel/SGD/chainrule/softmax/cache dependency identities are not newlydiscovered universalmathematics.',
            'Matchedthree-model replication is finite and conflates architecture/tokenizer/training withsize; no causal scale claim.'],
        'next_big_question':'Extract source/role-resolved crosslayer update laws from availableprefixinputs and test their nativeparameter formation, independently heldnewrelations/depths, fulloutput and autonomousbehavior; prefer those tests over expanding rare-event counts.',
        'retention':{'all_saved_result_arrays_client_registered':True,'all_saved_result_arrays_checked':True,
            'main_and_confirmation':'1152originalmaterials/204097actualtokens/3072completeall-layeranchors; allH12sourcevectors and completefinalMLPsourceinputs retained.',
            'full_alltoken_alllayer_raw_fixtures':12,'native_parameter_formation':'Complete74711040parameter gradients represented exactly by outerfactors, all288query Gram relations,36actualsingle-stepupdates and4full64stepparameterdeltas retained.',
            'native_deployment':'672ownhistorytrajectories plus96independent samehistorycompanion trajectories and24prefill/queryscope cases.',
            'scale':'Three sequentialnonquantizedlocalmodels,144matchednaturalwindows and24humanQA each, allnativewidths/units retained atdeclaredanchors.',
            'streamed_not_archived':'Alltoken/alllayer fields outside12fixtures were processed and summarized, not claimed continuouslyretainedraw. Model-specificsource means includeallpositions; scale-only fullsource arrays can be recomputed.',
            'deleted_user_or_original_field_files':[]},
        'resources':{'bytes_before_manifest':actual,'result_ceiling_bytes':resources['result_ceiling_bytes'],'disk_free_bytes':shutil.disk_usage(BASE).free,
            'booked_script_seconds_including_declared_allowances':sum(r['seconds'] for r in computation),'compute_ceiling_seconds':resources['compute_ceiling_seconds'],
            'timing_scope':'Measuredscript timers plus explicitlymarkedrecovery allowances, not total human/UIwallclock or GPUkernel time.',
            'no_task_CUDA_job_running':True,'budget_exhaustion_claimed':False}}
    save(BASE/'delivery_manifest.json',manifest)
    final={'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),'phases':phases,'original_memo_prefix_bytes_unchanged':prefix['bytes'],
        'live_full_service_routes':live,'indexed_figures':len(figures),'theory_puzzles':len(theory['puzzles']),'theory_formulas':len(theory['formulas']),
        'source_snapshots':snapshots,'delivery_manifest_sha256':sha(BASE/'delivery_manifest.json'),
        'full_array_checks':'verification/scientific_integrity.json','browser_checks':'verification/browser.json',
        'result_bytes_before_final':usage(),'finite_delivery_complete':True,'general_language_mechanism_closed':False}
    save(BASE/'verification/final.json',final);guard()
    print('FINAL_LAW_DELIVERY_PASS',phases,'bytes',usage(),'of',resources['result_ceiling_bytes'],flush=True)


if __name__=='__main__':main()
