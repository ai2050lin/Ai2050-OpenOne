"""Prospective natural documents/sentences, explicit family exclusions and attachment evidence audit."""
from collections import Counter,defaultdict
import hashlib
from rdc_relation_common import *
from phase2711_rdc_prefix_material import aligned_words

ATTACHMENTS=[r'C:/Users/Admin/.codex/attachments/9606128c-5211-4921-8f07-960bdfe7d0a8/pasted-text.txt',
             r'C:/Users/Admin/.codex/attachments/2aea851a-eaa8-4db4-8f0a-acc2b1670205/pasted-text.txt']


def audit():
    if (BASE/'review.json').exists():return
    source_paths=['atlas/nuisance_audit.json','shared_rules/result.json','shared_rules/native_path/result.json',
      'layer_operators/result.json','confirmation/result.json','scale_analysis/result.json','full_source_history/result.json',
      'full_source_history/fresh_result.json','full_source_history/probability_result.json','full_source_history/relations/result.json',
      'full_source_history/template_sensitivity.json','full_source_history/relations/uncertainty_audit.json','verification/integrity_audit.json','terminal.json']
    evidence=[]
    for rel in source_paths:
        p=PREVIOUS/rel;assert p.exists(),p;evidence.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    r=read(PREVIOUS/'shared_rules/result.json');early=next(x for x in r['reports'] if x['model']=='early_linear')
    # Defined numeric assertions, not an invented rerun of the language model.
    native=read(PREVIOUS/'shared_rules/native_path/result.json');assert native['MLP']['units']==9728
    q=read(PREVIOUS/'full_source_history/fresh_result.json');assert abs(q['reports'][0]['mse']-59.99047307548)<1e-7
    rel=read(PREVIOUS/'full_source_history/relations/result.json');assert rel['pairs']==11227 and rel['relations']==14
    layer=read(PREVIOUS/'layer_operators/result.json');assert abs(layer['rollout_reports'][0]['H36']['mse']-885.03426)<1e-4
    prior_final=read(PREVIOUS/'terminal.json');assert prior_final['completed_phases']==[2711,2712,2713,2714]
    corrections=[
      ['A majority numerical summaries','Retain with original scopes; reported covariance/paired product is observational, not a recovered parameter network.'],
      ['A all-source necessity','Full source retention is useful for this audit, not proved necessary for every sufficient extracted state.'],
      ['A historical positive claims','2235 and2619 have documented scoped positives; this review does not rerun their interventions or claim all old phases were independently audited.'],
      ['B one-dimensional probe termination','False: early linear is the strong current predictor; nuisance removal does not erase every first-order signal.'],
      ['B only negation/contrast survive','False: pronoun, conjunction, conditional and modal residual profiles also remain; no pure semantic interpretation follows.'],
      ['B complete implicit historical compression','Not established: H12 contains processed context but is not shown sufficient, and native future computation still uses all-layer history.'],
      ['B explicit graphs harmful','Only these crude cue kernels failed to beat the baseline; graphs beat some random controls and richer prefix relation rules were not tested.'],
      ['B statistical/physical isomorphism','Matching three method rankings does not prove statistical isomorphism or prove every possible native alignment absent.'],
      ['B second-order physical gears','Phase2714 measured same coordinate j at two positions, not different coordinates i and j; a paired moment is not a physical gear or a pure relation.'],
      ['B covariance/exterior product claim','Standardized paired means were not within-relation centered covariances; off-diagonal full outer products were not measured in the11227-pair study.'],
      ['B unrelated controls and complete confound removal','Distance-matched controls can be truly related and do not eliminate lexical, POS, absolute-position or full-context effects.'],
      ['B rank decay, high-frequency details and multi-hop collapse','Not measured. Native coordinate IDs have no defined spatial frequency; no tensor-rank causal explanation was tested.'],
      ['B rolling failure of second-order mechanism','2712 rolled a coordinatewise affine state model, not the2714 relation product. Layer rollout was not free generation.'],
      ['B Apple, passive sign flip, knowledge contractions','Illustrative or unsupported mechanisms, not actual tests; do not add them to historical evidence.'],
      ['B new global formula','Implicit_Compress is undefined and omits MLP, norms, current K/V and finite-precision accounting; retain real architecture equations and explicit extracted-state interfaces.'],
      ['B 50 universal UD relations','UD v2 lists37 universal relations plus language-specific subtypes; inspect actual observed inventory instead of asserting50 universal types.'],
      ['B low-rank tensor basis and guaranteed fixes','Not authorized scientific conclusions. Exact blockwise all-coordinate analysis first; no low-rank basis is assumed to be the mechanism.'],
      ['B code/math cross modality and AGI proof','Textual code/math comparisons are task/domain comparisons here, not demonstrated sensory cross-modality; no hallucination cure or AGI theorem follows.']]
    # Quantify infeasibility of per-example all-token outer products, independently of model outcomes.
    estimated=10000*36*36*2560*2560*4
    save(BASE/'review.json',{'timestamp':stamp(),'attachments':[{'path':p,'sha256':sha(p),'lines':len(Path(p).read_text(encoding='utf-8').splitlines())} for p in ATTACHMENTS],
      'original_sources':evidence,'numeric_checks_passed':True,'corrections':corrections,'reviewed_early_record':early,
      'naive_10000_units_36tokens_one_layer_pair_float32_bytes':estimated,
      'primary_references':[{'url':'https://universaldependencies.org/u/dep/','verified_fact':'UDv2 has37 universal relation types plus optional subtypes.'},
        {'url':'https://arxiv.org/abs/1812.08718','scope':'RNN tensor-product decomposition candidate; natural sentence representations also admitted strong bag-of-words approximation, not a Qwen/GLM mechanism proof.'},
        {'url':'https://arxiv.org/abs/2104.09864','scope':'Q/K rotary positional matching, not arbitrary whole-HiddenState inverse rotation.'}],
      'plan_integration':'A natural-prefix/relationship/update/native-generation program is primary. B complete cross-coordinate observation retained, but low-rank-mechanism, rank-decay repair and universal-isomorphism endpoints rejected as unsupported.'})


def parse_all(path,language):
    out=[]
    for block in path.read_text(encoding='utf-8').strip().split('\n\n'):
        meta={};words=[]
        for line in block.splitlines():
            if line.startswith('# ') and ' = ' in line:k,v=line[2:].split(' = ',1);meta[k]=v
            elif not line.startswith('#'):
                c=line.split('\t')
                if len(c)==10 and c[0].isdigit():words.append({'id':int(c[0]),'form':c[1],'lemma':c[2],'upos':c[3],'head':int(c[6]),'relation':c[7]})
        if not meta.get('text') or not words:continue
        sid=meta.get('sent_id','');group=sid.rsplit('-',1)[0] if language=='en' else 'content-'+normalized(meta['text'])[:60]
        out.append({'source_sentence_id':sid,'source_group':language+'/'+group,'text':meta['text'],'language':language,
          'genre':sid.split('-')[0] if language=='en' else 'wikipedia','source_has_document_id':language=='en',
          'retrospective_ud':aligned_words(meta['text'],words),'component_ids':[sid]})
    return out


def combine(a,b):
    shift=max(w['id'] for w in a['retrospective_ud']);offset=len(a['text'])+1
    words=list(a['retrospective_ud'])
    for w in b['retrospective_ud']:
        q=dict(w,id=w['id']+shift,head=w['head']+shift if w['head'] else 0)
        q['char_span']=[v+offset for v in w['char_span']] if w['char_span'] else None;words.append(q)
    return dict(a,text=a['text']+' '+b['text'],retrospective_ud=words,component_ids=a['component_ids']+b['component_ids'],
      source_sentence_id=a['source_sentence_id']+'..'+b['source_sentence_id'])


def select(tok,part,per_language,excluded):
    group_seen,num_seen,con_seen,text_seen,component_seen=excluded;chosen=[];sources=[];diagnostics={}
    for language in ('en','zh'):
        path=PREVIOUS/f'sources/{language}_{part}.conllu';raw=parse_all(path,language)
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'part':part,'version':'UDr2.15','reused_without_copy':True})
        eligible=[];dedup=set();rejected=Counter()
        for i,a in enumerate(raw):
            # Two actual adjacent corpus sentences from the same English document; never synthesize a Chinese paragraph.
            alternatives=[a]
            if language=='en' and i+1<len(raw) and raw[i+1]['source_group']==a['source_group']:
                pair=combine(a,raw[i+1]);alternatives=([pair,a] if int(hashlib.sha256(a['source_group'].encode()).hexdigest()[:8],16)%2 else [a,pair])
            for r in alternatives:
                components=[a] if len(r['component_ids'])==1 else [a,raw[i+1]]
                if r['source_group'] in group_seen or any(x['source_sentence_id'] in component_seen[language] for x in components):rejected['old_source']+=1;continue
                num={canonical(x['text']) for x in components}|{canonical(r['text'])}
                con={construction(x) for x in components}|{construction(r)}
                nt={normalized(x['text']) for x in components}|{normalized(r['text'])}
                if num&num_seen or con&con_seen or nt&text_seen:rejected['old_numeric_or_skeleton_or_text']+=1;continue
                e=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
                if not 24<=len(e['input_ids'])<=112:rejected['length']+=1;continue
                if r['source_group'] in dedup:continue
                dedup.add(r['source_group']);r=dict(r,prompt_ids=e['input_ids'],token_offsets=e['offset_mapping'],tokens=tok.convert_ids_to_tokens(e['input_ids']),
                  numeric_families=sorted(num),construction_families=sorted(con),normalized_texts=sorted(nt),source_path=str(path.relative_to(ROOT)))
                eligible.append(r);break
        buckets=defaultdict(list)
        for r in eligible:buckets[r['genre']].append(r)
        for bucket in buckets.values():bucket.sort(key=lambda r:hashlib.sha256(('2715:'+r['source_sentence_id']).encode()).hexdigest())
        selected=[]
        for i in range(max(map(len,buckets.values()),default=0)):
            for genre in sorted(buckets):
                if i>=len(buckets[genre]):continue
                r=buckets[genre][i]
                if r['source_group'] in group_seen or set(r['numeric_families'])&num_seen or set(r['construction_families'])&con_seen or set(r['normalized_texts'])&text_seen:continue
                group_seen.add(r['source_group']);num_seen.update(r['numeric_families']);con_seen.update(r['construction_families']);text_seen.update(r['normalized_texts']);component_seen[language].update(r['component_ids']);selected.append(r)
                if len(selected)==per_language:break
            if len(selected)==per_language:break
        assert len(selected)==per_language,(language,part,len(selected),len(eligible),dict(rejected))
        for i,r in enumerate(selected):
            n=len(r['prompt_ids']);anchors=[n//3,2*n//3]
            r.update(sample_id=f'{part}-{language}-r{i:04d}',anchors=anchors,positions=[p+d for p in anchors for d in (0,1,2)],
              split='confirmation' if part=='test' else 'unset',
              material_kind='adjacent_document_sentences' if len(r['component_ids'])==2 else 'natural_sentence',
              retrospective_scope='Full sentence/treebank labels for discovery and evaluation only, never forecast inputs.',
              online_scope='Actual token ID prefix only; learned token-piece tag and candidate relation probabilities may be wrong or unknown.')
        chosen.extend(selected);diagnostics[language]={'eligible_documents_or_content_groups':len(eligible),'rejected':dict(rejected),'selected_tokens':sum(len(r['prompt_ids']) for r in selected),'multi_sentence_units':sum(len(r['component_ids'])>1 for r in selected)}
    if part=='train':
        # Exact320/96/96 source splits, stratified independently within language and genre.
        for language in ('en','zh'):
            rr=[r for r in chosen if r['language']==language];buckets={g:[r for r in rr if r['genre']==g] for g in sorted({r['genre'] for r in rr})}
            def allot(n,cap):
                ideal={k:n*v/sum(cap.values()) for k,v in cap.items()};a={k:int(v) for k,v in ideal.items()}
                for k in sorted(a,key=lambda k:(-(ideal[k]-a[k]),k))[:n-sum(a.values())]:a[k]+=1
                return a
            tr=allot(160,{k:len(v) for k,v in buckets.items()});va=allot(48,{k:len(v)-tr[k] for k,v in buckets.items()})
            for k,rs in buckets.items():
                for i,r in enumerate(rs):r['split']='train' if i<tr[k] else 'validation' if i<tr[k]+va[k] else 'test'
    chosen.sort(key=lambda r:(r['sample_id'][-4:],r['language']))
    return chosen,sources,diagnostics


def main():
    freeze();audit()
    if (BASE/'material.json').exists():print('RELATION_MATERIAL_ALREADY_FROZEN',flush=True);return
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    old=old_material();components={lang:{r['source_sentence_id'] for r in old if r['language']==lang} for lang in ('en','zh')}
    excluded=({r['source_group'] for r in old},{canonical(r['text']) for r in old},{construction(r) for r in old},{normalized(r['text']) for r in old},components)
    material,sources,diag=select(tok,'train',256,excluded);fresh,more,fd=select(tok,'test',64,excluded)
    immutable(BASE/'material.json',material);immutable(BASE/'fresh_material.json',fresh)
    nums=[v for r in material+fresh for v in r['numeric_families']];groups={s:{r['source_group'] for r in material+fresh if r['split']==s} for s in ('train','validation','test','confirmation')}
    for a in groups:
        for b in groups:
            if a!=b:assert not groups[a]&groups[b]
    counts=Counter((r['language'],r['genre'],r['split'],r['material_kind']) for r in material+fresh)
    save(BASE/'material_audit.json',{'passed':True,'timestamp':stamp(),'main':diag,'fresh':fd,'sources':sources+more,
      'main_units':len(material),'fresh_units':len(fresh),'main_tokens':sum(len(r['prompt_ids']) for r in material),
      'fresh_tokens':sum(len(r['prompt_ids']) for r in fresh),'strata':{'/'.join(k):v for k,v in counts.items()},
      'split_counts':dict(Counter(r['split'] for r in material)),'exact_sources_numeric_templates_delex_whole_skeletons_excluded':True,
      'limits':['Chinese has no verified source document IDs.','Whole-sentence delexicalized UD skeleton identity is not every semantic construction or paraphrase.','English adjacency is known corpus sentence order within document; Chinese remains natural single sentences.','No knowledge of original LLM pretraining overlap.'],
      'main_sha':sha(BASE/'material.json'),'fresh_sha':sha(BASE/'fresh_material.json'),'code':snapshot(Path(__file__))})
    guard(16*1024**2);print('RELATION_MATERIAL_FROZEN',len(material),len(fresh),diag,fd,usage(),flush=True)


if __name__=='__main__':main()
