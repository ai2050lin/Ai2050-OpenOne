"""CPU verification and versioned client closeout. No inference or selection."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2773_w6_paired as P


def main():
    from deploy.research_library import Library
    from deploy.parameter_reader import model_snapshot
    from tools import provenance as prov
    root=W.ROOT;lib=Library(root/'distributed_data/research_library')
    # Existing immutable result is the source, not a recomputed narrative.
    r=W.read(P.OUT/'result.json');registration=W.read(P.OUT/'registration.json')
    report=copy.deepcopy(lib.artifact(registration['report_sha256']))
    checked=[];total_bytes=0
    for folder in [P.OLD,P.OUT]:
        f=W.read(folder/'freeze.json')
        assert W.sha(folder/'corpus.json')==f['corpus_sha256']
        assert W.sha(folder/'native/manifest.json')
        manifest=W.read(folder/'native/manifest.json');ids=[]
        for obj in manifest['files']:
            path=folder/'native'/obj['path'];assert W.sha(path)==obj['sha256']
            z=np.load(path);ids.extend(z['ids'].tolist());total_bytes+=path.stat().st_size
            assert z['hidden_bits'].dtype==np.uint16 and z['hidden_bits'].shape[1:]==(37,2,2560)
            assert z['mlp_bits'].dtype==np.uint16 and z['mlp_bits'].shape[1:]==(10,2,9728)
        assert len(ids)==len(set(ids))==len(W.read(folder/'corpus.json'))
        checked.append({'run_id':f['run_id'],'samples':len(ids),'batches':len(manifest['files'])})
    before=W.read(P.OLD/'freeze.json')['model_snapshot']
    after=model_snapshot(W.MODEL)
    assert after['sha256']==before['sha256'],'Original model files must remain identical'
    bc=W.read(P.OLD/'capability-base.json');ec=W.read(P.OUT/'capability-edited.json')
    assert [q['id'] for q in bc]==[q['id'] for q in ec]
    recomputed=W.paired_interval([q['correct'] for q in bc],[q['correct'] for q in ec],np.array([q['gate_eligible'] for q in bc]))
    assert recomputed==r['capability']
    b=P.gap(W.read(P.OUT/'base.json')['scores'])
    selected=P.gap(W.read(P.OUT/'dose'/f'a{r["selected"]["alpha"]}.json')['scores'])
    assert W.bootstrap_mean(b-selected)==r['lockbox']['target']['shift']
    assert set(q['family_id'] for q in W.read(P.OLD/'corpus.json')).isdisjoint(q['family_id'] for q in W.read(P.OUT/'corpus.json'))
    gen={}
    for mode in ['base','edited']:
        rows=W.read(P.OUT/f'generation-{mode}.json')
        gen[mode]={'n':len(rows),'literal_both':sum(q['literal_content_ok'] for q in rows),
            'quantity_mentioned':sum(str(q['expected']['quantity']) in q['text'] for q in rows),
            'item_mentioned':sum(q['expected']['item'] in q['text'] for q in rows),
            'colloquial_markers':sum(q['colloquial_markers'] for q in rows),
            'academic_markers':sum(q['academic_markers'] for q in rows),'hit_limit':sum(q['hit_limit'] for q in rows)}
    lengths={}
    for register in ['acad','colloq']:
        a=[q['tokens'] for q in W.read(P.OUT/'corpus.json') if q['register']==register]
        lengths[register]={'min':min(a),'max':max(a),'mean':sum(a)/len(a)}
    audit={'kind':'software_and_research_recalculation','passed':True,'run_id':P.RUN,'checked_native':checked,
        'native_bytes':total_bytes,'original_weights_unchanged':True,'full_snapshot':after['sha256'],
        'paired_capability_recalculated':True,'primary_effect_recalculated':True,
        'new_families_disjoint':True,'generations':gen,'lengths':lengths,
        'warning':'Register labels co-vary with two fixed constructions and token length; no isolation of abstract register.',
        'limits':'joint ten-layer edit, not individual layer/unit causality; same frozen monitoring battery reused'}
    W.save(P.OUT/'verification.json',audit,True)
    report.update(revision='verified-client-closeout-v2',source_report_sha256=registration['report_sha256'],verification=audit)
    counter_ids={e['sample_id'] for e in r['counterexamples']}
    report['counterexample_inputs']={q['id']:q for q in W.read(P.OLD/'capability-inputs.json') if q['id'] in counter_ids}
    report['limitations'] += ['两个语域水平仍与固定句式及长度共同变化，尚未隔离抽象语域',
        '10层联合编辑，逐层和逐单元的独立因果作用未测；定位单元仅代表编辑列',
        'α=0.5 等剂量在新材料上只作描述，不以其结果重新选择主编辑或声明最小值']
    rp=P.PUBLIC/'verified-report-v2.research.json';W.save(rp,report,True)
    ref=lib.ingest(report,str(rp.relative_to(root)),kind='local_report')['sha256']
    graph=copy.deepcopy(lib.artifact(registration['graph_sha256']));graph['revision']='verified-client-closeout'
    for obj in graph['feature_cards']+graph['graph_edges']:obj['evidence']['artifact_sha256']=ref
    for card in graph['feature_cards']:
        card['label']=f'W6 v3 · 联合编辑 B{card["layer"]} 参数片段'
        card['scope'].update(intervention_group=P.RUN,individual_layer_causality='not_tested')
        card['summary']['limitations']=report['limitations']
    gp=P.PUBLIC/'verified-graph-v2.research.json';W.save(gp,graph,True)
    gr=lib.ingest(graph,str(gp.relative_to(root)),kind='registered')['sha256']
    final={**registration,'report_sha256':ref,'graph_sha256':gr,'source_report_sha256':registration['report_sha256'],
        'source_graph_sha256':registration['graph_sha256'],'verification_sha256':W.sha(P.OUT/'verification.json')}
    W.save(P.OUT/'client-registration.json',final,True)
    P.status('registration_validated',**final)
    # Explicit adapter: legacy provenance helper uses config-only IDs and a fixed
    # output root. Bind its write to this run's full snapshot and required root.
    prov.RESULTS=P.OUT/'provenance_exports'
    prov.model_revision=lambda model_path=None:after['sha256']
    prov._git=lambda *args:subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',*args],cwd=root,text=True).strip()
    p=prov.write_result('phase2773_w6_verified_closeout',{'result':r,'audit':audit,'registration':final},
        model_path=str(W.MODEL),seed=W.SEED,tool='phase2773_closeout.py (full-snapshot compatibility adapter)')
    assert not prov.check_provenance(W.read(p))
    print(json.dumps({'verification':audit,'registration':final,'provenance_export':str(p)},ensure_ascii=False))


if __name__=='__main__':main()
