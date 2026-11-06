"""Verify actual server-produced main/extra packs preserve the W6 closed loop."""
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2773_w6_paired as P


def main():
    reg=W.read(P.OUT/'client-registration.json')
    base=W.ROOT/'distributed_data/datapack'
    with gzip.open(base/'datapack.json.gz','rt',encoding='utf-8') as f:mainpack=json.load(f)
    with gzip.open(base/'extra/research-library.json.gz','rt',encoding='utf-8') as f:extra=json.load(f)
    packages=mainpack['entries']['/api/research-library/workpackages']['workpackages']
    wp=next(w['body'] for w in packages if w['body']['run_id']==P.RUN)
    assert wp['stages']['registration_validated']['report_sha256']==reg['report_sha256'],'Main pack is pending its background rebuild'
    entries=extra['entries']
    report=entries['/api/research-library/artifacts/'+reg['report_sha256']]
    graph=entries['/api/research-library/artifacts/'+reg['graph_sha256']]
    assert report['result']['capability']['status']=='noninferior'
    assert len(report['result']['counterexamples'])==49
    assert len(report['dose_curve']['rows'])==6
    assert report['counterexample_inputs']
    responses=0
    for preview in report['sample_preview']:
        for l,sha in preview['responses'].items():
            obj=entries['/api/research-library/artifacts/'+sha]
            assert len(obj['values'])==9728 and obj['block']==int(l)
            assert obj['model_snapshot']==report['model_snapshot']
            responses+=1
    assert responses==20
    assert all(c['scope']['individual_layer_causality']=='not_tested' for c in graph['feature_cards'])
    result={'kind':'software_offline_acceptance','passed':True,'run_id':P.RUN,
        'main_version':mainpack['version'],'extra_version':extra['version'],
        'latest_report_sha256':reg['report_sha256'],'native_response_arrays':responses,
        'counterexamples':49,'dose_points':6,'model_snapshot':report['model_snapshot'],
        'includes_prior_negative_versions':any(e.get('run_id')==W.RUN and e.get('schema')=='research_intervention_report.v1'
            for key,e in entries.items() if key.startswith('/api/research-library/artifacts/') and isinstance(e,dict))}
    assert result['includes_prior_negative_versions']
    W.save(P.OUT/'offline-acceptance.json',result,True)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
