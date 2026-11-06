"""Append self-contained frozen runner/configuration sources for offline reproduction."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry


def main():
    W=P.W;lib=Library(W.ROOT/'distributed_data/research_library');reg=Registry(lib)
    c=W.read(P.OUT/'corpus.json');index=W.read(P.OUT/'client-registration-current.json')
    freeze=W.read(P.OUT/'freeze.json');runner=Path(P.N.__file__);assert W.sha(runner)==freeze['runner_sha256']
    source_files=[runner,W.ROOT/'tests/glm5/phase2790_materials.py',W.ROOT/'tests/glm5/phase2777_materials_v3.py',
        W.ROOT/'tests/glm5/phase2790_generation_blind.py',W.ROOT/'tests/glm5/phase2790_native_audit.py',W.ROOT/'tests/glm5/phase2790_closeout.py']
    bundle={'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'frozen_corpus':c,'corpus_file_sha256':W.sha(P.OUT/'corpus.json'),'runner_sha256':freeze['runner_sha256'],
        'contract':W.read(next((P.OUT/'node-tasks').glob('*/bundle/contract.json'))),
        'files':[{'path':str(p.relative_to(W.ROOT)),'sha256':W.sha(p),'text':p.read_bytes().decode('utf-8')} for p in source_files],
        'execution':'data for manual review/reproduction; never auto-execute downloaded source; local pinned model and recorded CUDA environment required',
        'recompute':'Frozen runner is standalone. Place frozen_corpus as corpus.json, contract as contract.json, runner source as runner.py. Set AI2050_RUNNER_MODEL, AI2050_NATIVE_DIR, AI2050_SHARD=i/8; python runner.py --out outputs. Big native fields stay on each node.',
        'provenance':c['provenance']}
    p=P.PUBLIC/'sources.research.json';W.save(p,bundle,True);source_ref=lib.ingest(bundle,str(p.relative_to(W.ROOT)),kind='local_report')['sha256']
    report=copy.deepcopy(lib.artifact(index['report_sha256']));report['supersedes_sha256']=index['report_sha256'];report['source_bundle_sha256']=source_ref
    p=P.PUBLIC/'report-v3.research.json';W.save(p,report,True);rr=lib.ingest(report,str(p.relative_to(W.ROOT)),kind='local_report')['sha256']
    graph=copy.deepcopy(lib.artifact(index['graph_sha256']));graph['supersedes_sha256']=index['graph_sha256']
    for row in graph['feature_cards']+graph['graph_edges']:row['evidence']['artifact_sha256']=rr
    for row in graph['feature_cards']:row['counterexamples_full_report_sha256']=rr
    p=P.PUBLIC/'graph-v3.research.json';W.save(p,graph,True);gr=lib.ingest(graph,str(p.relative_to(W.ROOT)),kind='registered')['sha256']
    claim=next(x['body'] for x in reg.catalog()['claims'] if x['body']['run_id']==P.RUN);reg.register({**claim,'artifact_sha256':rr})
    for card in graph['feature_cards']:
        reg.register({'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+P.N.digest({'graph':gr,'card':card['id']}),
            'canonical_feature_id':claim['canonical_feature_id'],'model_snapshot':report['model_snapshot'],'legacy_feature_id':card['id'],
            'artifact_sha256':gr,'relation':'member_locator','run_id':P.RUN,'layer':card['layer'],'boundary':card['boundary'],'provenance':c['provenance']})
    latest={**index,'report_sha256':rr,'graph_sha256':gr,'source_bundle_sha256':source_ref,
        'parameter_url':index['parameter_url'].replace(index['graph_sha256'],gr),'supersedes':index}
    W.save(P.OUT/'client-registration-current.json',latest);P.status('registration_validated',**{k:v for k,v in latest.items() if k!='supersedes'})
    print({'report_sha256':rr,'graph_sha256':gr,'source_bundle_sha256':source_ref})


if __name__=='__main__':main()
