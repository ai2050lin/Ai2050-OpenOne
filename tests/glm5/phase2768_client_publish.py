"""Attach inspectable examples to an existing frozen real run; no new inference or selection."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L
from deploy.research_library import Library


def main():
    source=L.read(L.PUBLIC/'validated-evidence.research.json')
    result=L.read(L.OUT/'validation.json')
    summary,native,manifest,ids,items,A=L.data()
    tokens=L.read(L.OUT/'tokens.json')
    rows=[]
    for i,key in enumerate(ids):
        it=items[key]
        rows.append({'id':key,'entity':it['target'],'split':it['split'],'frame':it['frame_id'],
            'category':it['cond']['category'],'text':it['text'],'tokens':len(tokens[key]),
            'activation':float(A[i,result['unit']]),'selected_from_discovery':it['split']=='discovery'})
    # Deterministic first example per split/category/frame, plus first six errors.
    preview=[];seen=set()
    for r in rows:
        k=(r['split'],r['category'],r['frame'])
        if k not in seen:preview.append(r);seen.add(k)
    errors={r['id'] for r in result['counterexamples'][:6]}
    preview += [r for r in rows if r['id'] in errors and r not in preview]
    source=copy.deepcopy(source);source.update(revision='client-evidence',samples=rows,sample_preview=preview,
        preview_rule='First item per split/category/frame in frozen corpus order, plus first six recorded errors; all samples retained.',
        source_report_sha256=L.read(L.OUT/'validated-registration.json')['report_sha256'])
    cells={}
    for r in L.read(L.OUT/'collection/behavior.json')['items']:
        cells.setdefault(r['split']+'|'+r['cell'],[]).append(r['correct'])
    source['behavior_by_split_cell']={k:{'n':len(v),'accuracy':sum(v)/len(v)} for k,v in cells.items()}
    path=L.PUBLIC/'client-evidence.research.json';L.save(path,source,True)
    lib=Library(ROOT/'distributed_data/research_library')
    evidence=lib.ingest(source,str(path.relative_to(ROOT)),kind='local_report')['sha256']
    graph=copy.deepcopy(L.read(L.PUBLIC/'validated-graph.research.json'));graph['revision']='client-evidence'
    for obj in graph['feature_cards']+graph['graph_edges']:obj['evidence']['artifact_sha256']=evidence
    graph['feature_cards'][0]['label']='类别响应候选 · B18 / U9031（跨句式未复现）'
    graphpath=L.PUBLIC/'client-graph.research.json';L.save(graphpath,graph,True)
    reg=lib.ingest(graph,str(graphpath.relative_to(ROOT)),kind='registered')
    L.save(L.OUT/'client-registration.json',{'report_sha256':evidence,'graph_sha256':reg['sha256'],'feature_id':graph['feature_cards'][0]['id']},True)
    print(json.dumps({'report':evidence,'graph':reg['sha256'],'samples':len(rows),'preview':len(preview)},ensure_ascii=False))


if __name__=='__main__':main()
