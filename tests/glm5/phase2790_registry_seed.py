"""Append canonical data/claim mappings to existing research; retain old CAS."""
from pathlib import Path
import collections
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2790_research as P
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry
from shared.dist_protocol.natural_register_runner import digest


def main():
    lib=Library(W.ROOT/'distributed_data/research_library');reg=Registry(lib);registered=[];datasets={}
    for key,path in [('W6-paired-v3',W.ROOT/'tests/glm5/result/phase2773_w6_v3/corpus.json'),
        ('W6-natural-v1',W.ROOT/'tests/glm5/result/phase2777_confirm/corpus.json'),('multidomain-long-v1',P.OUT/'corpus.json')]:
        source=W.read(path);rows=source['items'] if isinstance(source,dict) else source
        groups=sorted({digest(r.get('facts',r.get('fact',{'family':r['family_id']}))) for r in rows})
        did='dataset:'+key+':'+W.sha(path)[:16];datasets[key]=did
        doc={'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':key,'version':'1','content_sha256':W.sha(path),
            'source':str(path.relative_to(W.ROOT)),'license':'project-authored controlled templates; not an upstream natural corpus',
            'label_status':'input condition annotations, not internal mechanism facts','role':'controlled_mechanism',
            'source_groups':groups,'sample_count':len(rows),'lineage_rule':'same core facts group together across styles, lengths and frames',
            'facets':['Chinese','register','controlled_long_documents'],'provenance':W.read(P.OUT/'corpus.json')['provenance']}
        registered.append(reg.register(doc))
    qs=W.read(P.OLD/'capability-inputs.json')
    for source in sorted({q['source'] for q in qs}):
        rows=[q for q in qs if q['source']==source];content=digest(rows);did='dataset:monitor:'+content[:24]
        doc={'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':source,'version':'frozen-P2773',
            'content_sha256':content,'source':source,'license':'upstream terms require source-specific verification before redistribution',
            'label_status':'provided_dataset_key' if rows[0]['gate_eligible'] else 'unverified_mirror_reference',
            'role':'capability_monitor','source_groups':sorted({digest({'question':q['question'],'choices':q['choices']}) for q in rows}),
            'sample_count':len(rows),'formal_gate_eligible':rows[0]['gate_eligible'],'newly_blind':False,
            'provenance':W.read(P.OUT/'corpus.json')['provenance']}
        registered.append(reg.register(doc))
    c=W.read(P.OUT/'corpus.json');operation={k:c['edit'][k] for k in ['alpha','directions','columns']}
    canonical='operation:'+digest({'model_snapshot':c['model_snapshot'],**operation})
    versions=[('fixed-candidate',W.read(W.ROOT/'tests/glm5/result/phase2773_w6_v3/client-registration.json'),
        'W6-paired-v3','paired_fixed_completion_log_odds_attenuation','supported_in_scope',
        'Two fixed continuations; paired context gap attenuation; B18..27 k512 alpha6 joint edit; original candidate scorer.'),
        ('free-generation',W.read(W.ROOT/'tests/glm5/result/phase2777_confirm/client-registration-current.json'),
        'W6-natural-v1','natural_style_sufficiency_one_way','inconclusive',
        'Free Chinese factual restatement;384/512 tokens;new records and frames;original rubric+same-model judge failed measurement gate;B18..27 k512 alpha6.')]
    for name,index,data,metric,state,scope in versions:
        report=lib.artifact(index['report_sha256']);run=report['run_id']
        doc={'schema':'research_claim.v1','kind':'real','id':'claim:'+digest({'op':canonical,'metric':metric,'scope':scope}),
            'canonical_feature_id':canonical,'model_snapshot':c['model_snapshot'],'external_pattern_id':'language:register',
            'outcome_metric':metric,'scope':scope,'status':state,'dataset_versions':[datasets[data]],
            'artifact_sha256':index['report_sha256'],'run_id':run,'human_validated':False,
            'note':'A different outcome metric or rater changes the scoped claim; one joint operation is not ten independent mechanisms.'}
        registered.append(reg.register(doc))
        graph=lib.artifact(index['graph_sha256'])
        for card in graph['feature_cards']:
            alias={'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+digest({'graph':index['graph_sha256'],'card':card['id']}),
                'model_snapshot':c['model_snapshot'],'canonical_feature_id':canonical,'legacy_feature_id':card['id'],
                'artifact_sha256':index['graph_sha256'],'relation':'member_locator','run_id':run,'layer':card['layer'],
                'boundary':card['boundary'],'index':card['index'],'note':'Representative location in one joint operation, not equivalence of neuron and semantic feature.'}
            registered.append(reg.register(alias))
    for prefix,input_name in [('calibration','calibration-blind'),('semantic-controls','semantic-controls-blind')]:
        blind=W.read(P.OUT/(input_name+'.json'));ref=lib.ingest(blind,'AI-blind:'+prefix,kind='local_report')['sha256']
        cid='P2790-AI-'+prefix+'-v1'
        reg.create_ai_review({'schema':'external_ai_review_campaign.v1','kind':'real','id':cid,
            'rater_models':['gpt-6-astra','gpt-6.1-sol'],'input_sha256':W.sha(P.OUT/(input_name+'.json')),
            'blind_artifact_sha256':ref,'n_items':len(blind['records']),'human_validation':False})
        for suffix in ['astra','sol']:
            p=P.OUT/f'{prefix}-{suffix}.json'
            if p.exists():
                doc=W.read(p);doc.update(kind='real',provenance={**c['provenance'],'model_path':'collaboration:'+doc['rater_model'],
                    'model_revision':'provider_alias:'+doc['rater_model'],'seed':None},
                    rated_model_snapshot=c['model_snapshot'],rater_limit='provider alias pinned by tool; exact model weights and decoding seed not exposed; AI scoring, not human gold',
                    source_rating_file_sha256=W.sha(p))
                reg.submit_ai_review(cid,doc)
    W.save(P.OUT/'registry-registration.json',{'schema':'registry_migration.v1','kind':'real','run_id':P.RUN,
        'registered':registered,'canonical_operation_id':canonical,'old_cas_unchanged':True,'provenance':c['provenance']},True)
    P.status('registry',datasets=len(reg.catalog()['datasets']),claims=len(reg.catalog()['claims']),aliases=len(reg.catalog()['aliases']))
    print({'datasets':len(reg.catalog()['datasets']),'claims':len(reg.catalog()['claims']),'canonical_operation_id':canonical})


if __name__=='__main__':main()
