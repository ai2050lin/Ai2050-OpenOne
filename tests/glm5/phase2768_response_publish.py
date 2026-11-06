"""Publish complete selected-sample MLP arrays from frozen native fields. CPU only."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L
from deploy.research_library import Library


def main():
    source=copy.deepcopy(L.read(L.PUBLIC/'client-evidence.research.json'))
    summary,native,manifest,ids,items,A=L.data();lookup={key:i for i,key in enumerate(ids)}
    splits=L.GR.assign_splits(L.spec());lib=Library(ROOT/'distributed_data/research_library')
    selected=[];seen=set()
    for row in source['samples']:
        row['entity_split']=splits[items[row['id']]['filler_id']]
        key=(row['split'],row['frame'],row['category'],row['entity_split'])
        if key not in seen:selected.append(row);seen.add(key)
    error_ids={r['id'] for r in source['result']['counterexamples'][:6]}
    selected += [r for r in source['samples'] if r['id'] in error_ids and r not in selected]
    for row in selected:
        field={'schema':'research_mlp_response.v1','kind':'real','run_id':L.RUN,
            'model_snapshot':source['model_snapshot'],'block':L.BLOCK,'boundary':'mlp.down_proj.input',
            'readout':'final','sample_id':row['id'],'entity_split':row['entity_split'],'text':row['text'],
            'tokens':row['tokens'],'selected_unit':source['result']['unit'],'values':A[lookup[row['id']]].tolist(),
            'reconstruction':'A=SiLU(G)*V, float64 recomputed from protocol float16 G/V exported from bf16 forward; not a fresh run or exact bf16 down_proj input.',
            'native':summary['mlp_units']['native'],'source_report_sha256':L.read(L.OUT/'validated-registration.json')['report_sha256']}
        path=L.PUBLIC/'responses'/(row['id'].replace(':','_')+'.research.json');L.save(path,field,True)
        row['response_sha256']=lib.ingest(field,str(path.relative_to(ROOT)),kind='local_report')['sha256']
    source.update(revision='client-response-evidence',sample_preview=selected,
        preview_rule='First item per split/frame/category/entity-family split, plus first six errors; all 192 sample records and native full arrays retained.')
    path=L.PUBLIC/'client-response-evidence.research.json';L.save(path,source,True)
    ref=lib.ingest(source,str(path.relative_to(ROOT)),kind='local_report')['sha256']
    graph=copy.deepcopy(L.read(L.PUBLIC/'client-graph.research.json'));graph['revision']='client-response-evidence'
    for obj in graph['feature_cards']+graph['graph_edges']:obj['evidence']['artifact_sha256']=ref
    path=L.PUBLIC/'client-response-graph.research.json';L.save(path,graph,True)
    reg=lib.ingest(graph,str(path.relative_to(ROOT)),kind='registered')
    L.save(L.OUT/'response-registration.json',{'report_sha256':ref,'graph_sha256':reg['sha256'],'feature_id':graph['feature_cards'][0]['id'],'response_count':len(selected)},True)
    print(json.dumps({'report':ref,'graph':reg['sha256'],'complete_mlp_arrays':len(selected),'units_per_array':A.shape[1]}))


if __name__=='__main__':main()
