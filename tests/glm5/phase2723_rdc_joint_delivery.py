"""Final bounded-campaign manifest. Does not relabel historical plans or claim AGI closure."""
import hashlib,re,urllib.request
from rdc_joint_common import *

def main():
    guard(3*1024**2);start=time.monotonic()
    frozen=read(BASE/'frozen.json')
    for name,digest in frozen['files'].items():assert sha(BASE/name)==digest
    for e in read(BASE/'review.json')['evidence']:assert sha(ROOT/e['path'])==e['sha256']
    prefix=read(BASE/'memo_prefix.json');memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',memo,re.M) if int(x)>=2719]
    assert phases==[2719,2720,2721,2722,2723],phases
    required={
      2719:['review.json','material_audit.json','layer_atlas/result.json','prior_confirmation/result.json','relation_atlas/result.json'],
      2720:['frozen.json','probability_training/result.json','verification/phase2720_integrity.json'],
      2721:['confirmation/result.json','native_factors/result.json','generation/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json','verification/client_api.json'],
      2722:['extension/amplification.json','extension/event_trace/result.json','extension/temperature/result.json','extension/tail_confirmation/result.json','verification/phase2722_delivery.json'],
      2723:['extension/native_regimes/result.json','extension/native_regimes/normalization_audit.json','extension/native_regimes/structure_summary.json','verification/phase2723_delivery.json','verification/browser.json','verification/model_checkpoint_fingerprints.json']}
    for pp in required.values():
        for name in pp:assert (BASE/name).is_file(),name
    assert read(BASE/'verification/phase2722_delivery.json')['passed'] and read(BASE/'verification/phase2723_delivery.json')['passed']
    assert read(BASE/'verification/browser.json')['passed']
    figures=read(BASE/'figures/index.json')['figures'];assert len(figures)==10,len(figures)
    for fig in figures:
        path=BASE/'figures'/fig['path'];assert path.exists()
        # Earlier figure index uses sha rather than sha256.
        digest=fig.get('sha256',fig.get('sha'));assert digest is not None,fig
        assert sha(path)==digest,fig['path']
    live={}
    for name in ('rdc-joint','rdc-relation','rdc-prefix'):
        with urllib.request.urlopen(f'http://127.0.0.1:5001/api/{name}/overview',timeout=20) as r:live[name]=r.status;assert r.status==200
    entry=json.loads(urllib.request.urlopen('http://127.0.0.1:5001/api/rdc-joint/extension-index').read());assert len({e['kind'] for e in entry})==8
    summaries=json.loads(urllib.request.urlopen('http://127.0.0.1:5001/api/rdc-joint/analysis-index').read())
    native=read(BASE/'extension/native_regimes/result.json');assert native['active_signatures']=='corrected_training_signatures.npz'
    computation=read(BASE/'compute_ledger.json');allocated=read(BASE/'resource_allocation.json')
    manifest={'timestamp':stamp(),'status':'bounded_joint_campaign_completed','phases':[
        {'phase':p,'status':'executed_and_recorded','artifacts':files} for p,files in required.items()],
        'automatic_same_goal_continuations_executed':[2722,2723],'original_plan_preserved_as_historical':'plan.json',
        'scope':'Attachment evidence review plus initial three integrated phases and two substantive automatic continuations, all natural-coordinate evidence and client delivery. This is a finite delivered research campaign, not completion of the long-term AGI/language-encoding goal.',
        'unclosed':['Knowledge correctness, multi-step reasoning and syntax are not unified by a proven extracted mechanism.',
            'Autonomous32-step fitted generation degenerates; no successful general language closure.',
            'Rare-event ranking is useful but default0.5 recall is2/19.',
            'New block16/34 event organization has not been replicated on14B/GLM.',
            'Next major question is ordinary-position relation composition after identity, norm/event and temperature controls; new behavioral material/protocol not executed.'],
        'retention':{'all_original_768_fields':'retained and client-queryable','scale_fields':'3 models ×224 fields retained, each own native width',
            'generation':'All64 trajectories and3 own-prefix branches retained and queryable','native_factors':'All256 confirmation fields+4 training fixtures+full-coordinate/unit moments and all320 training identities',
            'tail':'All19 event rows+8 complete three-layer fixtures+all45113 noninitial token results+all original array identities; nonfixture RAM fields released after checks',
            'native_regimes':'All44 selected full37-layer/factor archives and all90 probe summaries retained and queryable','deleted_old_user_files':[]},
        'resources':{'result_bytes_before_manifest':usage(),'result_ceiling_bytes':allocated['result_ceiling_bytes'],
            'result_volume_free_bytes':shutil.disk_usage(BASE).free,'model_analysis_ledger_entries':len(computation),
            'model_analysis_compute_seconds':sum(x['seconds'] for x in computation),'compute_ceiling_seconds':allocated['maximum_model_and_analysis_compute_seconds'],
            'not_exhausted':True,'notes':'Engineering bounds, not user-specified numeric budget. Model ledger is not total wall-clock; UI/editing and independent delivery audits are separate. No further CUDA job left running.'}}
    save(BASE/'delivery_manifest.json',manifest)
    source_paths=sorted({*list((ROOT/'tests/glm5').glob('phase2719_rdc*.py')),*list((ROOT/'tests/glm5').glob('phase272[0-3]_rdc_joint*.py')),
        *list((ROOT/'tests/glm5').glob('rdc_joint*.py')),ROOT/'server/rdc_joint_service.py',ROOT/'frontend/src/components/app/RdcJointAtlas.jsx',ROOT/'frontend/src/components/app/RdcJointAtlas.css'})
    report={'timestamp':stamp(),'passed':True,'phases':phases,'original_frozen_artifacts':len(frozen['files']),
        'old_evidence_files':len(read(BASE/'review.json')['evidence']),'original_memo_bytes_unchanged':prefix['bytes'],
        'scientific_and_API_audits':{'main_client':'verification/client_api.json','extended_all_fields':'verification/phase2722_delivery.json','corrected_native_regimes':'verification/phase2723_delivery.json','browser':'verification/browser.json','full_checkpoint_files':'verification/model_checkpoint_fingerprints.json'},
        'live_routes':live,'registered_extension_arrays':len(entry),'registered_summary_arrays':len(summaries),'indexed_scientific_figures':len(figures),
        'source_snapshots':[snapshot(p) for p in source_paths],'delivery_manifest_sha':sha(BASE/'delivery_manifest.json'),
        'resource_usage_bytes_before_final':usage(),'seconds':time.monotonic()-start,
        'finite_delivery_complete':True,'universal_language_mechanism_closed':False}
    save(BASE/'verification/final.json',report);guard();print('FINAL_JOINT_DELIVERY_PASS',phases,'usage',usage(),'of',allocated['result_ceiling_bytes'],'seconds',report['seconds'],flush=True)

if __name__=='__main__':main()
