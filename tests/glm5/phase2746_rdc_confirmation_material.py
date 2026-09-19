"""Prospective new-document and new-combination confirmation; no outcome selection."""
from collections import Counter,defaultdict
from rdc_construction_common import *
from phase2746_rdc_history_prediction_contract import OUT as PRED

OUT=PRED/'confirmation'
FAMILIES=['attribute_binding','negation_scope','word_sense','long_distance_role','knowledge_chain']


def recipe(family,case,lang,world):
    a,b=(f'Neral-{case:02d}',f'Vesko-{case:02d}') if lang=='en' else (f'诺岚{case:02d}',f'维斯{case:02d}')
    first,second=(a,b) if world==0 else (b,a);truth=world==0;variant=case%4
    if family=='attribute_binding':
        obj=(['lantern','ribbon','telescope','helmet'] if lang=='en' else ['灯笼','缎带','望远镜','头盔'])[variant]
        c,d=(['purple','amber'] if lang=='en' else ['紫色','琥珀色'])
        body=(f'Inventory A: the {obj} assigned to {first} is {c}. Inventory B: the {obj} assigned to {second} is {d}.' if lang=='en'
              else f'清单甲：分配给 {first} 的{obj}是{c}。清单乙：分配给 {second} 的{obj}是{d}。')
        question=f'Is the {obj} assigned to {a} {c}?' if lang=='en' else f'分配给 {a} 的{obj}是{c}吗？'
        edges=[(first,'assigned_'+obj,c),(second,'assigned_'+obj,d)]
    elif family=='negation_scope':
        verb,past,obj=([('repair','repaired','radio'),('water','watered','orchid'),('unlock','unlocked','cabinet'),('polish','polished','mirror')][variant]
                       if lang=='en' else [('修理','修理了','收音机'),('浇灌','浇灌了','兰花'),('解锁','解锁了','橱柜'),('擦亮','擦亮了','镜子')][variant])
        body=(f'Log A: {first} {past} the {obj}. Log B: {second} did not {verb} the {obj}. Both people inspected the clock.' if lang=='en'
              else f'日志甲： {first} {past}{obj}。日志乙： {second} 没有{verb}{obj}。两个人都检查了时钟。')
        negative=case%2==1;truth=(world==0)!=negative
        statement=(f'{a} did not {verb} the {obj}' if negative else f'{a} {past} the {obj}') if lang=='en' else (f'{a} 没有{verb}{obj}' if negative else f'{a} {past}{obj}')
        question=f'Is the claim "{statement}" supported by the log?' if lang=='en' else f'日志支持“{statement}”这一说法吗？'
        edges=[(first,verb,obj),(second,'not_'+verb,obj)]
    elif family=='word_sense':
        island=(['sailed along the coast of Java','visited the volcanoes of Java','mapped the harbours of Java','crossed the rice fields of Java'][variant] if lang=='en'
                else ['沿着Java岛的海岸航行','游览了Java岛的火山','绘制了Java岛的港口地图','穿过了Java岛的稻田'][variant])
        code=(['compiled a program written in Java','debugged a Java class in the editor','implemented a Java interface for the program','studied the Java compiler documentation'][variant] if lang=='en'
              else ['编译了用Java编写的程序','在编辑器里调试Java类','为程序实现了Java接口','研究了Java编译器文档'][variant])
        x,y=(island,code) if world==0 else (code,island)
        body=f'Passage A: {a} {x}. Passage B: {a} {y}.' if lang=='en' else f'段落甲： {a} {x}。段落乙： {a} {y}。'
        ask=case%2==0;truth=(world==0)==ask
        question=(('Does Java in Passage A name an island?' if ask else 'Does Java in Passage A name a programming language?') if lang=='en'
                  else ('段落甲中的Java指一座岛屿吗？' if ask else '段落甲中的Java指一种编程语言吗？'))
        edges=[('A','contextual_sense','island' if world==0 else 'programming_language'),('B','contextual_sense','programming_language' if world==0 else 'island')]
    elif family=='long_distance_role':
        obj=(['compass','manuscript','medal','portrait'] if lang=='en' else ['指南针','手稿','奖章','肖像'])[variant]
        body=(f'Transfer record: a {obj} was delivered to {first} by {second}. '+' '.join(f'Background note {i+1}: shelf {i+3} held {i+4} folders.' for i in range(8))+f' Afterwards {a} lent a scarf to {b}.' if lang=='en'
              else f'转交记录：{obj}由 {second} 交到了 {first} 手中。'+''.join(f'背景记录{i+1}：第{i+3}层架子上有{i+4}个文件夹。' for i in range(8))+f'随后 {a} 把围巾借给了 {b}。')
        giver=case%2==1;truth=(world==0)!=giver
        question=f'Was {a} the '+('sender' if giver else 'recipient')+f' of the {obj}?' if lang=='en' else f'{a} 是{obj}的'+('交出者' if giver else '接收者')+'吗？'
        edges=[(second,'giver_of_'+obj,first),(a,'scarf_lender',b)]
    else:
        c0,c1,c2,c3=[f'Zal{i}-{case:02d}' for i in range(4)] if lang=='en' else [f'泽类{i}-{case:02d}' for i in range(4)]
        x,y=(c0,c1) if world==0 else (c1,c0)
        body=(f'In this invented registry only explicit rules apply. Clause A: every {x} is {y}. Clause B: every {c1} is {c2}. Clause C: every {c2} is {c3}. Object {a} is {c0}. No means not necessarily entailed; outside knowledge is excluded.' if lang=='en'
              else f'这份虚构登记册只采用明确规则。条款甲：每个 {x} 都属于 {y}。条款乙：每个 {c1} 都属于 {c2}。条款丙：每个 {c2} 都属于 {c3}。对象 {a} 属于 {c0}。否表示不一定能推出，不采用外部知识。')
        question=f'Do the clauses entail that {a} is {c3}?' if lang=='en' else f'能从条款推出 {a} 属于 {c3} 吗？'
        edges=[(a,'instance_of',c0),(x,'subclass_of',y),(c1,'subclass_of',c2),(c2,'subclass_of',c3)]
        reached={c0}
        for _ in range(4):
            for s,r,t in edges:
                if r=='subclass_of' and s in reached:reached.add(t)
        assert (c3 in reached)==truth
    return body,question,truth,[dict(source=s,relation=r,target=t) for s,r,t in edges],[a,b]


def freeze():
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json'),gzread(OUT/'material.json.gz')
    from transformers import AutoTokenizer
    import phase2740_rdc_query_material as prior
    from phase2744_rdc_query_identifiability import language_checks
    start=time.monotonic();frozen=read(PRED/'frozen.json');tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    old,paths=prior.prior_inventory();old+=gzread(OLD/'material/natural.json.gz')+gzread(OLD/'followup/material.json.gz')
    used={tuple(r['prompt_ids']) for r in old};components={c for r in old for c in r.get('component_ids',[])};groups={r['source_group'] for r in old}
    inventory=[{k:r.get(k) for k in ['sample_id','source_id','source_group','prompt_ids','component_ids']} for r in old]
    compressed(OUT/'excluded_inventory.json.gz',inventory)
    pool=gzread(OLD/'material/followup_candidates.json.gz');remaining=defaultdict(dict)
    for r in pool:
        bank=r['cohort'];g=r['source_group']
        if g in groups or tuple(r['prompt_ids']) in used or set(r.get('component_ids',[]))&components:continue
        if g not in remaining[bank] or rank(r['source_id'])<rank(remaining[bank][g]['source_id']):remaining[bank][g]=r
    availability={b:len(remaining[b]) for b in ['gum','ewt','cmrc']};save(OUT/'availability.json',{'timestamp':stamp(),'new_eligible_groups':availability})
    assert availability['ewt']>=25 and availability['cmrc']>=167,availability
    rows=[]
    for bank,count in [('ewt',25),('cmrc',167)]:
      for group in sorted(remaining[bank],key=lambda g:rank('confirmation/'+g))[:count]:
        r=dict(remaining[bank][group]);events,relations=prior.external_events(r)
        r.update(sample_id='q2746new_'+rank(bank+'/'+r['source_id'])[:20],kind='natural',family='natural_'+bank,
            split='independent_confirmation',original_text=r['text'],events=events,typed_relations=relations,
            max_new_tokens=32,input_kind='raw_natural_prefix_no_chat_wrapper',
            novelty='Excluded every source_group, exact input and component in the persisted explicit prior inventory. One original corpus window per new document; not a claim of unseen pretraining data or every historical Phase.')
        rows.append(r)
    fit=gzread(BASE/'phase2746/runtime/material.json.gz');fittexts='\n'.join(r['original_text'] for r in fit if r['split']=='train')
    oldcontrols=gzread(BASE/'material.json.gz')['models']['qwen4']['rows'];usedctl={tuple(r['prompt_ids']) for r in oldcontrols};audits=[]
    for family in FAMILIES:
      for case in range(16):
       group=f'confirmation2746/{family}/{case:02d}'
       for lang in ['en','zh']:
        pair=[]
        for world in [0,1]:
            body,question,truth,edges,entities=recipe(family,case,lang,world)
            choices=['Yes','No'] if lang=='en' else ['是','否'];target=choices[0 if truth else 1]
            text=body+'\n'+question+'\n'+('Answer only Yes or No.' if lang=='en' else '只回答是或否。')
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
            assert tuple(enc['input_ids']) not in usedctl and all(e not in fittexts for e in entities)
            r={'sample_id':'q2746new_'+rank(group+'/'+lang+'/'+str(world))[:20],'source_group':group,'pair_id':group+'/'+lang,
                'family':family,'cohort':family,'kind':'controlled','language':lang,'world':world,'case':case,'truth':truth,'target':target,
                'body':body,'question':question,'original_text':text,'text':prompt,'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],
                'relations':edges,'entity_spans':entities,'split':'independent_confirmation','max_new_tokens':128,
                'novelty':'New full entity strings and wording/combinations relative to fitted material; all subtoken identities need not be unseen. Java sense substitution and three-edge fictional chains are joint changes, not isolated causal depth tests.'}
            pair.append(r);rows.append(r)
        assert Counter(pair[0]['prompt_ids'])==Counter(pair[1]['prompt_ids']),(family,case,lang)
        assert pair[0]['target']!=pair[1]['target'] and pair[0]['question']==pair[1]['question']
        audits.append({'pair_id':pair[0]['pair_id'],'full_prompt_token_multiset_matched':True,'opposite_answers':True,'same_question':True})
    assert len(rows)==512 and len({tuple(r['prompt_ids']) for r in rows})==512
    compressed(OUT/'material.json.gz',rows);compressed(OUT/'pair_audits.json.gz',audits)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'frozen_predictor_sha256':sha(PRED/'frozen.json'),
        'rows':512,'natural_documents':192,'controlled_expressions':320,'controlled_groups':80,'token_matched_pairs':160,
        'family_counts':dict(Counter(r['family'] for r in rows)),'excluded_rows':len(old),'excluded_source_groups':len(groups),
        'excluded_inventory_sha256':sha(OUT/'excluded_inventory.json.gz'),'candidate_source_sha256':sha(OLD/'material/followup_candidates.json.gz'),
        'prior_inventory_implementation':snapshot(prior.__file__),'remaining_source_groups_before_selection':availability,
        'corpus_boundary':'No genuinely new GUM documents in this reserved source bank. EWT25/CMRC167 document confirmation; no fabricated GUM new-window confirmation.',
        'novelty_audit':{'all_controlled_entity_full_spans_absent_from_train':True,'Java_present_in_training_text':'Java' in fittexts,
            'not_claimed':'Every lexical subtoken is new; natural corpora lack pretraining exposure; all historical Phases have been enumerated.'},
        'scoring_checks':language_checks(),'native_capture':{'batch':8,'all_coordinate_field_steps':8,'natural_cap':32,'controlled_cap':128},
        'frozen_routes':frozen['routes'],'autonomous_routes':[frozen['autonomous_direct_baseline_route'],frozen['autonomous_primary_native_constrained_route']],
        'selection':'No refitting or route/lambda reselection on this material. All native behavior, first3available-step scoring and chosen self-fed routes must include failures; early EOS yields missing later states, never zero imputation.',
        'material_sha256':sha(OUT/'material.json.gz'),'seconds':time.monotonic()-start}
    immutable(OUT/'protocol.json',protocol);ledger('phase2746_confirmation_material',protocol['seconds'])
    return protocol,rows


if __name__=='__main__':
    p,r=freeze();print('CONFIRMATION_FROZEN',p['family_counts'],p['excluded_source_groups'],flush=True)
