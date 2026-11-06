"""Verify actual background-built packs, including blind AI labels and full arrays."""
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P


def main():
    W=P.W;current=W.read(P.OUT/'client-registration-current.json');base=W.ROOT/'distributed_data/datapack'
    main=json.load(gzip.open(base/'datapack.json.gz','rt',encoding='utf-8'))
    extra=json.load(gzip.open(base/'extra/research-library.json.gz','rt',encoding='utf-8'));entries=extra['entries']
    get=lambda sha:entries['/api/research-library/artifacts/'+sha]
    wp=next(x['body'] for x in main['entries']['/api/research-library/workpackages']['workpackages'] if x['body']['run_id']==P.RUN)
    assert wp['stages']['registration_validated']['report_sha256']==current['report_sha256']
    report=get(current['report_sha256']);graph=get(current['graph_sha256']);catalog=entries['/api/research-library/catalog']
    assert report['schema']=='research_external_ai_register_report.v1' and len(graph['feature_cards'])==10
    assert len(catalog['datasets'])>=11 and len(catalog['claims'])>=3
    assert len(get(report['inputs_sha256'])['items'])==96 and len(get(report['generations_sha256'])['outputs'])==384
    assert get(report['native_audit_sha256'])['verified_files']==96
    bundle=get(report['source_bundle_sha256'])
    assert bundle['frozen_corpus']['model_snapshot']==report['model_snapshot']
    import hashlib
    for source in bundle['files']:assert hashlib.sha256(source['text'].encode('utf-8')).hexdigest()==source['sha256']
    assert bundle['runner_sha256']==report['freeze']['runner_sha256']
    plan=W.read(P.OUT/'dataset-plan-registration.json')
    assert get(plan['sha256'])['implementation_status']=='partial'
    assert wp['stages']['dataset_atlas_plan']['plan_sha256']==plan['sha256']
    assert len(get(report['supersedes_sha256'])['effects'])==3
    n=0
    for row in report['sample_preview']:
        for layer,records in row['trace_responses'].items():
            for item in records:
                response=get(item['sha256']);assert len(response['values'])==9728 and response['block']==int(layer)
                assert response['generation_step']==item['step'] and response['readout']=='generation' and response['model_snapshot']==report['model_snapshot']
                assert len(response['prefix_token_ids'])==response['tokens'];n+=1
    assert n==160
    ai=[r for r in catalog['reviews'] if r['id'].startswith('P2790-AI-')]
    assert len(ai)==3 and all(r['completed_raters']==2 and not r['evidence_promoted'] for r in ai)
    assert sorted(r['items'] for r in ai)==[32,84,307]
    assert all(len(get(a['sha256'])['ratings'])==r['items'] for r in ai for a in r['rating_artifacts'])
    artifact_text=json.dumps(extra,ensure_ascii=False)
    assert 'admin_token' not in artifact_text and 'node_token' not in artifact_text
    audit={'schema':'software_offline_acceptance.v1','kind':'real','run_id':P.RUN,'passed':True,
        'main_version':main['version'],'extra_version':extra['version'],'report_sha256':current['report_sha256'],
        'graph_sha256':current['graph_sha256'],'full_response_arrays':n,'ai_batches':3,'ai_rating_files':6,
        'inputs':96,'outputs':384,'large_native_tensors_local_only':True,'provenance':report['provenance']}
    audit['source_text_hashes_verified']=True;audit['plan_sha256']=plan['sha256']
    W.save(P.OUT/'offline-acceptance-current.json',audit);print(audit)


if __name__=='__main__':main()
