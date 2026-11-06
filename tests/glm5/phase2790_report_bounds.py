"""Append a presentation correction: zero-event bootstrap is not zero uncertainty."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry


def main():
    W=P.W;lib=Library(W.ROOT/'distributed_data/research_library');reg=Registry(lib)
    previous=W.read(P.OUT/'client-registration.json');report=copy.deepcopy(lib.artifact(previous['report_sha256']))
    report['supersedes_sha256']=previous['report_sha256']
    report['presentation_erratum']='Zero-event nonparametric bootstrap degenerates at0; show an exact one-sided family-any-event bound instead. Raw observations and preregistered thresholds unchanged.'
    for e in report['effects']:
        v=e['qualified_formal_to_spoken']
        if v['mean']==0:
            v['bootstrap_ci95_degenerate']=v.pop('ci95')
            v['zero_event_family_bound']={'n':v['n_families'],'events':0,'one_sided_upper95':1-.05**(1/v['n_families']),
                'estimand':'Probability of at least one qualified flip in a six-pair content family under exchangeability; not per-sentence probability.'}
    report['next_decision']='Old edit not supported for free-register control in this exploratory panel. Improve per-register semantic reliability and extract new candidates from natural-generation fields; freeze independent families/frames and a natural-generation dose design before confirmation.'
    report['limitations'].append('原模型口语保义仅32/48，通过的是事前 pooled>=70%门；不能说每个语域保义均达标。下一版须事前分别设置保义门，当前阈值不变。')
    path=P.PUBLIC/'report-v2.research.json';W.save(path,report,True);rr=lib.ingest(report,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
    graph=copy.deepcopy(lib.artifact(previous['graph_sha256']))
    for x in graph['feature_cards']+graph['graph_edges']:x['evidence']['artifact_sha256']=rr
    for x in graph['feature_cards']:x['counterexamples_full_report_sha256']=rr
    graph['supersedes_sha256']=previous['graph_sha256'];path=P.PUBLIC/'graph-v2.research.json';W.save(path,graph,True)
    gr=lib.ingest(graph,str(path.relative_to(W.ROOT)),kind='registered')['sha256']
    cat=reg.catalog();claim=next(x['body'] for x in cat['claims'] if x['body']['run_id']==P.RUN)
    reg.register({**claim,'artifact_sha256':rr,'supersedes_sha256':previous['report_sha256']})
    for card in graph['feature_cards']:
        reg.register({'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+P.N.digest({'graph':gr,'card':card['id']}),
            'canonical_feature_id':claim['canonical_feature_id'],'model_snapshot':report['model_snapshot'],'legacy_feature_id':card['id'],
            'artifact_sha256':gr,'relation':'member_locator','run_id':P.RUN,'layer':card['layer'],'boundary':card['boundary'],
            'provenance':report['provenance']})
    latest={**previous,'report_sha256':rr,'graph_sha256':gr,'parameter_url':previous['parameter_url'].replace(previous['graph_sha256'],gr),
        'supersedes':previous}
    W.save(P.OUT/'client-registration-current.json',latest,True);P.status('registration_validated',**{k:v for k,v in latest.items() if k!='supersedes'})
    print({'report_sha256':rr,'graph_sha256':gr,'family_any_event_upper95':1-.05**(1/8)})


if __name__=='__main__':main()
