"""Queryable evidence index, bounded continuation decision and non-circular terminal verification."""
import argparse
import hashlib
import psutil
from rdc_prefix_common import *


def prepare():
    out=CAMPAIGN/'full_source_history';assert read(out/'fresh_result.json')['no_fresh_fit']
    model_identity=[]
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        directory=ROOT/'models/hf'/name
        small=[p for p in directory.iterdir() if p.is_file() and (p.suffix=='.json' or p.name.startswith('tokenizer')) and p.stat().st_size<100*1024**2]
        model_identity.append({'model':name,'path':str(directory),
          'configuration_tokenizer_index_sha256':{p.name:sha(p) for p in small},
          'weight_files_metadata_only':[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(directory.glob('*.safetensors'))],
          'identity_limit':'Checkpoint shard size/mtime recorded, not a new full-weight cryptographic hash. Saved embedding rows are checked directly against actual checkpoint tensors.'})
    review=read(CAMPAIGN/'review.json')
    for item in review['original_sources']:assert sha(ROOT/item['path'])==item['sha256']
    save(CAMPAIGN/'run_manifest.json',{'timestamp':stamp(),'run_id':CAMPAIGN.name,'completed_phase_scopes':[
      {'phase':2711,'scope':'Attachment evidence review, shared natural prefix material, all-coordinate atlas and controls'},
      {'phase':2712,'scope':'Common layer/time prediction rules, native scalar paths, complete-vocabulary evaluation'},
      {'phase':2713,'scope':'Frozen128 source confirmation, serial BF16 scale fits, client and integrity'},
      {'phase':2714,'scope':'Automatic full-source extension, frozen64 source test, exploratory relation profiles and final delivery'}],
      'models':model_identity,'runtime_files':{r:sha(CAMPAIGN/r/'runtime.json') for r in ('qwen4','qwen4_confirmation','qwen14','glm4')},
      'material_sha256':{r:sha(CAMPAIGN/r) for r in ('material_stratified.json','confirmation_material.json','full_source_history/fresh_material.json')},
      'prior_reviewed_results_unchanged':True,'review_sha':sha(CAMPAIGN/'review.json'),
      'selection_boundaries':['shared_rules/frozen_models.json','layer_operators/frozen.json','full_source_history/frozen.json'],
      'source_code_identity':'verification/integrity_audit.json source_hashes after final stable audit',
      'corrections':['UTF8 decoder-prefix metadata and random-hash control: prefix_causality and causal_hash_control artifacts.','Numeric-template overlap and frozen sensitivity: full_source_history/template_sensitivity.json.','Relation bootstrap percentiles are conditional stability diagnostics: full_source_history/relations/uncertainty_audit.json.'],
      'not_completed_or_claimed':['All possible language families or complete knowledge/reasoning/grammar coverage','Prospective role-bound prefix parsing and unseen construction composition','New long free-generation experiment in this natural-prefix campaign','Native causal identification of UD relationships','Universal language closure, model isomorphism, brain mechanism or AGI']})
    legacy=[]
    mapping={x:x for x in ('i_factorial','k_long','m_order','o_generalization','p_token_conditioned')}
    mapping.update({'l_aligned_'+m:'l_aligned/'+m for m in ('qwen4','qwen14','glm4')})
    for run,rel in mapping.items():
        p=PRIOR/rel/'result.json';assert p.exists()
        legacy.append({'run_id':run,'root':str(p.parent.relative_to(ROOT)),'result_sha':sha(p),'status':'previous campaign reference, not new natural observations'})
    save(CAMPAIGN/'legacy_index.json',{'timestamp':stamp(),'purpose':'Correct actual nested aligned-model paths; all registered historical result files verified. Prior campaign artifacts remain untouched.',
      'revision':'Initial new-campaign index incorrectly treated aligned model directories as flat paths; this index fixes only the links, not historical results.','runs':legacy})
    runs=[]
    for run,n in [('qwen4',512),('qwen4_confirmation',128),('qwen14',64),('glm4',64)]:
        rt=read(CAMPAIGN/run/'runtime.json');runs.append({'run':run,'units':n,'layers':rt['depth']+1,'width':rt['width'],
          'saved_positions_per_unit':6,'all_observed_positions_statistics':True,'complete_all_layer_token_panels':16 if run=='qwen4' else 0,
          'sample_endpoint':f'/api/rdc-prefix/runs/{run}/samples','field_endpoint':f'/api/rdc-prefix/runs/{run}/field/{{sample}}',
          'coverage_limit':'Individual nonpanel/nonanchor H not retained except main H12 later extended; all token sums/squares retained.'})
    save(CAMPAIGN/'client_index.json',{'timestamp':stamp(),'client':'http://127.0.0.1:5173/rdc-prefix','prior_client':'http://127.0.0.1:5173/rdc',
      'read_only':True,'runs':runs,'full_source_H12':{'main_units':512,'main_tokens':18480,'fresh_units':64,'fresh_tokens':2222,
        'main_source_reuse':16,'recaptured_main_sources':496,'fresh_saved_checkpoints':'All-token H12, H0/H36/postnorm at two query anchors only; not all-layer six-position fields.',
        'sample_endpoint':'/api/rdc-prefix/history/samples?scope=main|fresh','field_endpoint':'/api/rdc-prefix/history/field?scope={scope}&sample={sample}',
        'prediction_endpoint':'/api/rdc-prefix/history/prediction?scope={scope}&sample={sample}&rule={rule}',
        'relation_endpoint':'/api/rdc-prefix/history/relations?scope=main|fresh'},
      'typed_relations':{'types':14,'pairs':11227,'evidence':'Retrospective exploratory, exact signed-distance control, diagonal all-coordinate products, not causal prefix labels.'},
      'coordinates':'Every native coordinate addressable; whole saved NPZ download. Display signed-log or linear, raw or frozen training z-score. No Top-K/PCA core.',
      'evidence_levels':['observed natural state','retrospective annotation','held-out fitted prediction','frozen-before-fresh source prediction','software-corrected control','native arithmetic oracle','exploratory relation statistic'],
      'retention':{'delete_paths':[],'decision':'Retain all saved fields; they are client-queryable and support frozen checks / prospective role-organization work. No hidden-state files removed.'},
      'legacy_links':'legacy_index.json','scientific_figures':read(CAMPAIGN/'figures/index.json')})
    pilot=read(out/'pilot_audit.json');projected_capture=pilot['bytes_per_source_token']*18480*3
    reserve=80*1024**2;required=projected_capture+reserve;remaining=CEILING-usage()
    assert required>remaining
    save(CAMPAIGN/'continuation_decision.json',{'timestamp':stamp(),'completed_phases':[2711,2712,2713,2714],
      'automatic_same_goal_extension_completed':2714,'scientific_goal_solved':False,'campaign_state':'integrated_science_complete_final_delivery_verification',
      'next_goal_same':True,'next_major_program':[
        {'stage':'Prospective relation/role-bound external prefix graph','tasks':['Build incremental prefix-only role/unfinished-constraint descriptors; keep future/gold syntax separate.','Freeze broad lexical/role/order/distance/context crossings with same family IDs; target at least512 new source units after pilot.','Control word identity/POS and signed distance jointly; include knowledge/reasoning/grammar families, not only one construction.']},
        {'stage':'All-coordinate relation-conditioned transfer rules','tasks':['Compare role-conditioned source kernels against current-state, same-sentence distance, lexical identity and absolute/relative baselines with matched validation/capacity.','Capture complete H12/H23/H36 source coordinates and test cross-layer/held-out combinations without target input.','Use newly frozen source material; do not recycle current fresh64 as an independent confirmation for the new relation method.']},
        {'stage':'Native parameter and generation connection','tasks':['Trace retained structural candidates to actual layer-specific QKV/MLP scalar-weight computations while keeping native arithmetic versus learned input separation.','Check all-vocabulary probabilities and actual multi-step generation content/format/stop separately.','Sequential nonquantized scale-model confirmation only after same-shape capture and resource pilot pass.']}],
      'resource_boundary':{'frozen_new_campaign_ceiling':CEILING,'current_campaign_bytes':usage(),'remaining_bytes':remaining,
        'next_capture_estimate_bytes':projected_capture,'next_analysis_reserve_bytes':reserve,'next_total_estimate_bytes':required,
        'basis':'Current pilot bytes per all-source H12 token *18480 observed-token-sized units *3 checkpoints, plus80MiB fit/output/audit reserve. Provisional estimate, new pilot required for longer/changed materials.',
        'physical_free_bytes':shutil.disk_usage(ROOT).free,'physical_floor_bytes':FLOOR,
        'no_silent_budget_increase':True,'not_physical_disk_full':True},
      'decision':'Complete finite campaign and preserve restartable plan. Next comparably scoped multi-layer/role study exceeds remaining frozen allocation; requires new finite allocation or explicit archiving choice, not an unbounded loop.',
      'data_cleanup':{'removed':[],'reason':'Saved fields are queryable evidence and inputs to prospective follow-up; no dispensable hidden-state collection identified.'}})
    print('PREFIX_DELIVERY_PREPARED',usage(),CEILING-usage(),int(required),flush=True)


def final():
    audit=read(CAMPAIGN/'verification/integrity_audit.json');assert audit['passed'] and audit['source_history_extension']['passed']
    assert read(CAMPAIGN/'verification/client_verification.json')['passed']
    for rel,digest in audit['source_hashes'].items():assert sha(ROOT/rel)==digest,('Source changed since final audit',rel)
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';b=memo.read_bytes();p=read(CAMPAIGN/'memo_prefix.json')
    assert hashlib.sha256(b[:p['bytes']]).hexdigest()==p['sha256']
    phases=list(map(int,re.findall(r'^## Phase (\d+):',b[p['bytes']:].decode('utf-8'),re.M)));assert phases==[2711,2712,2713,2714]
    jobs=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(proc.info['cmdline'] or [])
            if 'python' in (proc.info['name'] or '').lower() and any(x in cmd for x in ('phase2711_rdc_prefix_capture.py','phase2714_rdc_full_source_history.py','phase2714_rdc_source_kernels.py','phase2714_rdc_source_probability.py','rdc_source_serial_tail.py','rdc_prefix_serial_tail.py')):jobs.append({'pid':proc.pid,'command':cmd})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not jobs,jobs
    guard()
    decision=read(CAMPAIGN/'continuation_decision.json');decision['campaign_state']='bounded_campaign_complete';decision['finalized_at']=stamp()
    decision['resource_boundary'].update(current_campaign_bytes=usage(),remaining_bytes=CEILING-usage(),physical_free_bytes=shutil.disk_usage(ROOT).free)
    save(CAMPAIGN/'continuation_decision.json',decision)
    save(CAMPAIGN/'terminal.json',{'timestamp':stamp(),'state':'bounded_campaign_complete','completed_phases':phases,'scientific_goal_solved':False,
      'memo_original_prefix_intact':True,'memo_bytes':len(b),'memo_sha256':hashlib.sha256(b).hexdigest(),
      'source_unchanged_since_final_integrity':True,'integrity_report_sha':sha(CAMPAIGN/'verification/integrity_audit.json'),
      'client_verification_sha':sha(CAMPAIGN/'verification/client_verification.json'),'client':'http://127.0.0.1:5173/rdc-prefix',
      'active_capture_fit_probability_jobs':jobs,'campaign_bytes_before_terminal':usage(),'free_disk_bytes':shutil.disk_usage(ROOT).free,
      'ceiling_bytes':CEILING,'remaining_bytes_before_terminal':CEILING-usage(),'data_deleted':[],
      'next_program':'continuation_decision.json','post_audit_status_updates':['continuation_decision.json','terminal.json'],
      'limits':'Final finite delivery, not language mechanism closure or AGI. Source-history negative results and observational UD profiles retain separate evidence status.'})
    print('PREFIX_TERMINAL_PASS',len(b),usage(),CEILING-usage(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();final() if a.final else prepare()
