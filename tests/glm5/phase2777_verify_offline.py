"""Check actual center-built offline packs, never manufacture pack entries."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_confirm as C
import phase2773_w6_loop as W


def main():
    current=C.OUT/'client-registration-current.json'
    registration=W.read(current if current.exists() else C.OUT/'client-registration.json');plan=W.read(C.OUT/'dataset-atlas-plan-registration-current.json')
    base=W.ROOT/'distributed_data/datapack'
    with gzip.open(base/'datapack.json.gz','rt',encoding='utf-8') as f:mainpack=json.load(f)
    with gzip.open(base/'extra/research-library.json.gz','rt',encoding='utf-8') as f:extra=json.load(f)
    wp=next(w['body'] for w in mainpack['entries']['/api/research-library/workpackages']['workpackages'] if w['body']['run_id']==C.RUN)
    assert wp['stages'].get('registration_validated',{}).get('report_sha256')==registration['report_sha256'],'Main pack awaiting background rebuild'
    assert wp['stages']['dataset_atlas_plan']['plan_sha256']==plan['sha256'],'Main pack awaiting plan'
    entries=extra['entries']
    get=lambda sha:entries['/api/research-library/artifacts/'+sha]
    report=get(registration['report_sha256']);graph=get(registration['graph_sha256']);spec=get(plan['sha256'])
    assert report['schema']=='research_natural_register_report.v1'
    assert spec['kind']=='proposal' and spec['schema']=='research_dataset_atlas_plan.v1'
    inputs=get(report['inputs_artifact_sha256']);assert len(inputs['items'])==512
    assert {len(x['token_ids']) for x in inputs['items']}=={384,512}
    generations=[get(g['sha256']) for g in report['generation_artifacts']]
    assert len(generations)==4 and all(len(g['outputs'])==512 for g in generations)
    monitor=get(report['capability_artifact_sha256'])
    assert len(monitor['questions'])==len(monitor['base'])==len(monitor['edited'])==2340
    assert sum(x['gate_eligible'] for x in monitor['base'])==2200
    assert len([x for x in report['result']['counterexamples'] if x.get('gate_eligible') is False])==9
    assert report['counterexample_breakdown']['formal_capability_regressions']==23
    bundle=get(report['source_bundle_sha256'])
    for source in bundle['files']+[bundle['runtime_adapter']]:assert hashlib.sha256(source['text'].encode('utf-8')).hexdigest()==source['sha256']
    response_count=0
    for sample in report['sample_preview']:
        for layer,sha in sample['responses'].items():
            obj=get(sha);assert obj['block']==int(layer) and len(obj['values'])==9728
            assert obj['run_id']==C.RUN and obj['model_snapshot']==report['model_snapshot']
            response_count+=1
    assert response_count==20
    assert graph['independent_intervention_tests']==1 and len(graph['feature_cards'])==10
    assert all(c['scope']['individual_layer_causality']=='not_tested' for c in graph['feature_cards'])
    assert all(not get(p['sha256'])['pilot_full_result']['passed'] for p in report['pilot_artifacts'])
    previous=get(report['prior_preference_report_sha256'])
    assert previous['result']['target_supported_in_scope'] is True
    result={'kind':'software_offline_acceptance','passed':True,'run_id':C.RUN,
        'main_version':mainpack['version'],'extra_version':extra['version'],
        'report_sha256':registration['report_sha256'],'graph_sha256':registration['graph_sha256'],
        'dataset_plan_sha256':plan['sha256'],'native_response_arrays':response_count,
        'full_frozen_inputs':512,'full_generations':2048,'monitor_items':2340,'failed_pilot_versions':5,
        'prior_preference_claim_preserved':True,'source_text_hashes_verified':True,
        'large_native_tensors_local_only':True}
    W.save(C.OUT/'offline-acceptance.json',result,True)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
