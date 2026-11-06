"""Independent CPU audit of frozen source, native arrays, publication and descriptive frame effects."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L
from deploy.research_library import Library,canonical


def main():
    s,native,manifest,ids,items,A=L.data();r=L.read(L.OUT/'validation.json');j=r['unit']
    freeze=L.read(L.OUT/'freeze.json');assert freeze['script_sha256']==L.sha(L.__file__)
    assert L.sha(native/'manifest.json')==s['native_field']['manifest_sha256']
    for entry in manifest['files']:
        assert L.sha(native/entry['path'])==entry['sha256']
        assert np.isfinite(np.load(native/entry['path'])['X']).all()
    entities=sorted({i['filler_id'] for i in items.values()});frames=['catalogue','review','heldout']
    table=np.zeros((len(entities),len(frames)))
    for k,key in enumerate(ids):
        i=items[key];table[entities.index(i['filler_id']),frames.index(i['frame_id'])]=A[k,j]
    mu=table.mean();E=table.mean(1,keepdims=True)-mu;F=table.mean(0,keepdims=True)-mu;I=table-mu-E-F
    total=float(np.mean((table-mu)**2))
    parts={'entity':float(np.mean(E**2)/total),'frame':float(np.mean(F**2)/total),'entity_frame_interaction':float(np.mean(I**2)/total)}
    behavior=L.read(L.OUT/'collection/behavior.json')['items']
    posthoc={'schema':'posthoc_description.v1','kind':'real','run_id':L.RUN,'model_snapshot':r['model_snapshot'],'posthoc':True,'no_reselection':True,
        'frame_mean_A':dict(zip(frames,table.mean(0).tolist())),'balanced_mean_square_shares':parts,
        'counterexample_prompts':len(r['counterexamples']),'counterexample_entities':len({x['entity'] for x in r['counterexamples']}),
        'model_behavior_correct':sum(x['correct'] for x in behavior),'model_behavior_n':len(behavior),
        'boundary':'Balanced descriptive variance decomposition of one selected unit on this panel; not causal or information shares.'}
    L.save(L.OUT/'posthoc-frame-audit.json',posthoc,True)
    L.save(L.PUBLIC/'posthoc-frame-audit.research.json',posthoc,True)
    lib=Library(ROOT/'distributed_data/research_library');lib.ingest(posthoc,str((L.PUBLIC/'posthoc-frame-audit.research.json').relative_to(ROOT)),kind='local_report')
    registration=L.read(L.OUT/'response-registration.json');report=lib.artifact(registration['report_sha256'])
    for row in report['sample_preview']:
        field=lib.artifact(row['response_sha256'])
        assert np.array_equal(field['values'],A[ids.index(row['id'])])
        assert field['sample_id']==row['id'] and field['model_snapshot']==r['model_snapshot']
    latest=lib.fragments()['features'];f=next(f for f in latest if f['id']==registration['feature_id'])
    assert f['sha']==registration['graph_sha256'] and f['body']['evidence']['level']=='negative'
    for path,digest in report['raw_artifacts'].items():assert L.sha(ROOT/path)==digest
    audit={'schema':'research_loop_audit.v1','kind':'software_integrity_audit','run_id':L.RUN,'passed':True,
        'native_layers_verified':len(manifest['files']),'samples':len(ids),'complete_response_arrays_verified':len(report['sample_preview']),
        'latest_feature_level':f['body']['evidence']['level'],'feature_revisions':len(f['history']),
        'source_unchanged_after_freeze':True,'raw_evidence_hashes_verified':True,'graph_sha256':f['sha'],
        'scope':'Integrity and numerical recomputation; not independent-node replication or scientific promotion.'}
    L.save(L.OUT/'execution-audit.json',audit,True)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
