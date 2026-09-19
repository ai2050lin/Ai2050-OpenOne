"""Large natural-query panel, stable document splits and detailed native subpanel."""
from collections import Counter, defaultdict
from rdc_query_common import *

def prior_inventory():
    from rdc_update_common import prior_natural
    paths=[LAW/'material.json.gz',LAW/'confirmation_material.json.gz',PRIOR/'natural_material.json.gz',PRIOR/'fresh_graph/material.json.gz']
    rows=prior_natural()
    for p in paths:
        if p.exists():rows.extend(gzread(p))
    return rows,paths

def split_for(group):
    v=int(rank('document_split/'+group)[:12],16)%10
    return 'train' if v<5 else 'validation' if v<7 else 'test' if v<9 else 'followup_reserved'

def external_events(row):
    words=row.get('retrospective_ud',[]);byid={w['id']:w for w in words};events=[];relations=[]
    for w in words:
        if w['upos'] not in ('VERB','AUX'):continue
        participants=[];negation=[];auxiliary=[]
        for child in words:
            if child['head']!=w['id']:continue
            rel=child['relation']
            record={'word_id':child['id'],'relation':rel,'text':child['form'],'char_span':child.get('char_span')}
            if rel.split(':')[0] in ('nsubj','obj','iobj','csubj','obl'):participants.append(record)
            if child['lemma'].casefold() in ('not','never','no','neither','without') or 'Polarity=Neg' in child.get('features',''):negation.append(record)
            if rel.split(':')[0] in ('aux','mark','advcl','ccomp','xcomp'):auxiliary.append(record)
        events.append({'event_id':f"{row['source_id']}/verb/{w['id']}",'head_word_id':w['id'],'head_text':w['form'],'head_span':w.get('char_span'),
          'participants':participants,'explicit_negation_cues':negation,'clausal_links':auxiliary,
          'status':'retrospective_UD_event_candidate_not_internal_semantic_fact'})
    for w in words:
        if w['head'] in byid:
            relations.append({'type':'UD/'+w['relation'],'source_span':w.get('char_span'),'target_span':byid[w['head']].get('char_span'),
              'source_word_id':w['id'],'target_word_id':w['head'],'source':'published_treebank','inference_feature':False})
    # Natural Chinese encyclopedia carries real extractive QA annotations, not invented UD or internal facts.
    for q in row.get('annotations',[]):
        visible=[a for a in q['answers'] if a['answer_start']+len(a['text'])<=len(row['text'])]
        if visible:relations.append({'type':'provided_context_question/'+q['question_type'],'question':q['question'],'question_id':q['question_id'],
          'visible_answer_spans':[[a['answer_start'],a['answer_start']+len(a['text'])] for a in visible],
          'source':'original_dataset_annotation','inference_feature':False})
    return events,relations

def round_robin(pool,count,limit=48):
    grouped=defaultdict(list)
    for r in pool:grouped[r['source_group']].append(r)
    for v in grouped.values():v.sort(key=lambda r:rank(r['source_id']))
    groups=sorted(grouped,key=rank);selected=[]
    for k in range(limit):
      for g in groups:
        if len(grouped[g])>k:selected.append(grouped[g][k])
        if len(selected)==count:return selected
    return selected

def main():
    out=BASE/'material';target=out/'result.json'
    if target.exists():print('QUERY_MATERIAL_ALREADY_FROZEN');return
    assert read(BASE/'pilot/result.json')['all_passed']
    from transformers import AutoTokenizer
    import phase2728_rdc_law_material as old
    from rdc_query_probes import freeze
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);probes=freeze(tok)
    previous,oldpaths=prior_inventory();used={tuple(r['prompt_ids']) for r in previous};components={c for r in previous for c in r.get('component_ids',[])}
    exposed_groups={r['source_group'] for r in previous};manifest=[];pools={};inventory=[]
    for bank in ('gum','ewt'):
        candidates=[]
        for part in ('train','dev','test'):
            sourcepath=(old.JOINT/'sources'/f'gum_{part}.conllu.gz') if bank=='gum' else (old.BASE/'sources'/f'ewt_{part}.conllu.gz')
            assert sourcepath.is_file(),('Read-only prior source missing',sourcepath)
            candidates.extend(old.natural_candidates(tok,bank,part,manifest))
        pools[bank]=candidates
    pools['cmrc']=old.encyclopedia_pool(tok,'cmrc_train',manifest)+old.encyclopedia_pool(tok,'cmrc_dev',manifest)
    reserved={};eligible={}
    for bank,pool in pools.items():
        seen=set();keep=[]
        for r in pool:
            seq=tuple(r['prompt_ids'])
            if seq in seen or seq in used or set(r.get('component_ids',[]))&components:continue
            seen.add(seq);r=dict(r,cohort=bank,split=split_for(r['source_group']),historical_document_exposure=r['source_group'] in exposed_groups)
            keep.append(r)
        reserved[bank]=[r for r in keep if r['split']=='followup_reserved']
        eligible[bank]=[r for r in keep if r['split']!='followup_reserved']
        inventory.append({'cohort':bank,'raw_candidates':len(pool),'eligible_exact_and_recent_component_new':len(keep),
          'main_eligible':len(eligible[bank]),'main_groups':len({r['source_group'] for r in eligible[bank]}),'reserved':len(reserved[bank])})
    save(out/'availability.json',{'timestamp':stamp(),'rows':inventory,'prior_rows_in_inventory':len(previous),'source_files':manifest})
    total_capacity=sum(len(round_robin(v,10000)) for v in eligible.values());assert total_capacity>=10000,('Insufficient natural coverage',inventory,total_capacity)
    desired={'gum':3000,'ewt':5000,'cmrc':2000};selected={b:round_robin(eligible[b],n) for b,n in desired.items()}
    deficit=10000-sum(map(len,selected.values()))
    for bank in ('ewt','gum','cmrc'):
        if deficit<=0:break
        expanded=round_robin(eligible[bank],len(selected[bank])+deficit);deficit-=len(expanded)-len(selected[bank]);selected[bank]=expanded
    assert deficit==0
    rows=[];group_splits={}
    for bank,rr in selected.items():
      for r in rr:
        sid='q2740_'+rank(bank+'/'+r['source_id'])[:20];events,relations=external_events(r)
        # Keep actual source text. Full context/question metadata remain references, never predictor input.
        x={k:v for k,v in r.items() if k not in ('full_context','tokens')}
        x.update(sample_id=sid,anchors=[len(r['prompt_ids'])-1],events=events,typed_relations=relations,
          input_kind='raw_natural_prefix_no_chat_wrapper',kind='natural',capture_mode='query',
          actual_prefix_last_token=r['prompt_ids'][-1],actual_prefix_length=len(r['prompt_ids']),
          novelty='New exact token sequence and no component in explicitly inventoried recent windows. Within-split overlapping natural windows are allowed, clustered by real source document; historical document exposure is explicitly flagged, not erased.')
        group_splits.setdefault(x['source_group'],x['split']);assert group_splits[x['source_group']]==x['split'];rows.append(x)
    assert len(rows)==10000 and len({tuple(r['prompt_ids']) for r in rows})==10000
    rows.sort(key=lambda r:rank(r['sample_id']));detail=[]
    for bank in ('gum','ewt','cmrc'):
      for split,n in [('train',96),('validation',32),('test',64)]:
        pool=[r for r in rows if r['cohort']==bank and r['split']==split]
        rr=round_robin(pool,n,limit=8);assert len(rr)==n,(bank,split,len(rr));detail.extend(r['sample_id'] for r in rr)
    assert len(detail)==576
    # Reserve independent documents before any main outcomes. Actual continuation selection is later frozen.
    reserve=[]
    for bank,rr in reserved.items():
        for r in round_robin(rr,min(400,len(rr))):reserve.append(r)
    assert not {r['source_group'] for r in reserve}&set(group_splits)
    compressed(out/'natural.json.gz',rows);compressed(out/'followup_candidates.json.gz',reserve)
    compressed(out/'external_event_index.json.gz',[{'sample_id':r['sample_id'],'events':r['events'],'typed_relations':r['typed_relations'],'graph':r.get('graph',[])} for r in rows])
    fullfixtures=[]
    for bank in ('gum','ewt','cmrc'):
        fullfixtures.extend(r['sample_id'] for r in rows if r['sample_id'] in detail and r['cohort']==bank)
        # Restrict by frozen IDs, not amplitude or successful behavior.
    fullfixtures=[next(r['sample_id'] for r in rows if r['cohort']==b and r['split']==s and r['sample_id'] in detail) for b in ('gum','ewt','cmrc') for s in ('train','validation','test')]
    lengths=[len(r['prompt_ids']) for r in rows];pilot=read(BASE/'pilot/result.json')
    p={'timestamp':stamp(),'source':snapshot(__file__),'natural_prefixes':10000,'probes':100,'prospective_endpoint_count':1000000,
      'detailed_prefix_ids':detail,'full_layer_all_token_fixture_ids':fullfixtures,'query_split_counts':{'train':60,'validation':20,'unseen':20},
      'primary_fit':'Only288detailed train prefixes x60train queries;96validation prefixes and20validation queries choose regularization. Source-heldout192test prefixes and20unseen queries are not used for fitting/selection.',
      'large_atlas_scope':'10000document-assigned natural windows x100known diagnostic suffixes. Full-vocabulary divergences streamed, full native postnorm for every endpoint saved losslessly in BF16 bit representation. Other full-layer/full-token retention is explicitly limited to registered fixtures; detailed576prefixes retain early sources/cache and multiple later-layer targets.',
      'whole_coordinate_policy':'No PCA/TopK pruning, no coordinate order changes. Storing actual native BF16 bits is lossless persistence, not model quantization. Full vocabulary may be reconstructed from the exact stored postnorm, original readout and recorded batch shape.',
      'overlap':'Nearby windows may overlap only within one document split.10000windows are not10000independent documents. Source group uncertainty, repeated-query dependence and historical exposure must be reported.',
      'resource_gate':{'pilot_projected_native_seconds':10000*pilot['mean_seconds_per_prefix100'],
        'raw_postnorm_BF16_bytes':10000*100*2560*2,'raw_all_layer_prefix_anchor_bytes':10000*37*2560*2,
        'chunk_prefixes':2000,'note':'Chunked commits protect7200s process ceiling. Pilot estimate excludes persistence, data-dependent length, extra fields, fitting and later phases. Monitor actual growth every100prefixes.'}}
    immutable(out/'protocol.json',p)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'natural_prefixes':len(rows),'source_documents':len(group_splits),
      'by_cohort':dict(Counter(r['cohort'] for r in rows)),'by_split':dict(Counter(r['split'] for r in rows)),
      'by_genre':dict(Counter(r['genre'] for r in rows)),'by_language':dict(Counter(r['language'] for r in rows)),
      'explicit_UD_events':sum(len(r['events']) for r in rows),'typed_relations':sum(len(r['typed_relations']) for r in rows),
      'historically_exposed_documents':len({r['source_group'] for r in rows if r['historical_document_exposure']}),
      'detailed_prefixes':576,'reserved_candidates':len(reserve),'reserved_documents':len({r['source_group'] for r in reserve}),
      'token_lengths':{'min':min(lengths),'max':max(lengths),'mean':float(np.mean(lengths))},'tokens':sum(lengths),
      'main_material_sha256':sha(out/'natural.json.gz'),'source_files':manifest,'seconds':time.monotonic()-start,
      'science_scope':'UD event candidates are observational labels. Chinese encyclopedia questions describe supplied-context retrieval, not identified pretraining knowledge mechanisms. Query strings are experimental controls, not guaranteed natural continuations.'}
    save(target,result);ledger('natural_query_material_freeze',result['seconds']);print('QUERY_MATERIAL_FROZEN',result['by_cohort'],result['source_documents'],'events',result['explicit_UD_events'],flush=True)

if __name__=='__main__':main()
