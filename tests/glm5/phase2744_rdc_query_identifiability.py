"""Same-goal continuation: identity-matched relations and calibration alternatives.

Protocol creation is gated on completed2743, not on anticipated favorable outcomes.
All local model jobs must run after the main serial science queue has finished.
"""
import argparse
from collections import Counter, defaultdict
from rdc_query_common import *

OUT=BASE/'identifiability'
FAMILIES=['attribute_binding','negation_scope','word_sense','long_distance_role','knowledge_chain']
VARIANTS=['native','natural_target_2742','within_cohort_permuted_target_2742','natural_target_2743','within_cohort_permuted_target_2743']
QUERIES=[0,20,40,50,70,99]
TEMPERATURES=[.5,.75,1.,1.25,1.5,2.,3.]
MIXTURES=[0.,.01,.05,.1,.2,.5]


def language_score(row,text,ids,stop,cap):
    """Prospective whole-wrapper grammar; never infer a label from the target."""
    import re
    from rdc_query_scoring import score
    from rdc_update_terminal_audit import unwrap
    previous=score(row,text,ids,stop,cap);result=dict(previous);result['prior_identity_grammar']=previous
    if previous['EOS'] and not previous['censored'] and previous['conservative_final_answer'] is None:
        value=unwrap(text)
        if re.fullmatch(r'(?:yes|no|是|否)',value,re.I):
            correct=value.casefold()==str(row['target']).casefold()
            result.update(conservative_final_answer=value,conservative_final_correct=correct,parsed_and_stopped_correct=correct,
              format_audit_method='prospectively_frozen_complete_yes_no_wrapper')
    result['identity_scope']='Pre-outcome whole yes/no Markdown/LaTeX wrapper extension only; old grammar and strict-format fields retained. No arbitrary leading answer, no unfinished text, no synonym adjudication, no post-outcome scoring change.'
    return result


def language_checks():
    cases=[('**Yes**','Yes'),('`No`','No'),('\\boxed{是}','是'),('$$否$$','否'),('Yes, but continue',None),('**Yes or No**',None),('不是',None)]
    for text,expected in cases:
        row={'kind':'controlled_language','target':'No'}
        assert language_score(row,text,[99],{99},128)['conservative_final_answer']==expected
    assert not language_score({'kind':'controlled_language','target':'Yes'},'**Yes**',[42],{99},1)['parsed_and_stopped_correct']
    return {'all_passed':True,'cases':len(cases)+1,'frozen_before_native_outputs':True}


def recipe(family,case,lang,world):
    from phase2738_rdc_update_language_material import NAMES_EN,NAMES_ZH,FRUIT_CONTEXTS,COMPANY_CONTEXTS
    a,b=(NAMES_EN if lang=='en' else NAMES_ZH)[case],(NAMES_EN if lang=='en' else NAMES_ZH)[case+16]
    first,second=(a,b) if world==0 else (b,a)
    truth=world==0;edges=[]
    if family=='attribute_binding':
        body=f'Record A: {first} owns red apples. Record B: {second} owns green apples.' if lang=='en' else f'记录甲： {first} 有红苹果。记录乙： {second} 有青苹果。'
        question=f'Are the apples owned by {a} red?' if lang=='en' else f'{a} 的苹果是红色的吗？'
        edges=[(first,'owns','red_apples'),(second,'owns','green_apples')]
    elif family=='negation_scope':
        body=f'Record A: {first} packed the apples. Record B: {second} did not pack the apples. Both people packed books.' if lang=='en' else f'记录甲： {first} 装了苹果。记录乙： {second} 没有装苹果。两个人都装了书。'
        negative=case%2==1;truth=(world==0)!=negative
        statement=(f'{a} did not pack the apples' if negative else f'{a} packed the apples') if lang=='en' else (f'{a} 没有装苹果' if negative else f'{a} 装了苹果')
        question=f'Is the statement "{statement}" true according to these records?' if lang=='en' else f'陈述“{statement}”与记录一致吗？'
        edges=[(first,'packed','apples'),(second,'did_not_pack','apples')]
    elif family=='word_sense':
        fruit=FRUIT_CONTEXTS[case][0 if lang=='en' else 1];company=COMPANY_CONTEXTS[case][0 if lang=='en' else 1]
        x,y=(fruit,company) if world==0 else (company,fruit)
        body=f'Context A: {a} {x}. Context B: {a} {y}.' if lang=='en' else f'语境甲： {a} {x}。语境乙： {a} {y}。'
        ask_fruit=case%2==0;truth=(world==0)==ask_fruit
        question=('Does apple in Context A refer to edible fruit?' if ask_fruit else 'Does apple in Context A refer to a technology company?') if lang=='en' else ('语境甲中的苹果指可食用的水果吗？' if ask_fruit else '语境甲中的苹果指科技公司吗？')
        edges=[('A','contextual_sense','fruit' if world==0 else 'company'),('B','contextual_sense','company' if world==0 else 'fruit')]
    elif family=='long_distance_role':
        body=(f'Record A: {second} handed a sealed packet to {first}. '+ ' '.join(f'Unrelated note {i+1}: there were {i+2} chairs.' for i in range(4+case%5))+f' Later {a} handed a cup to {b}.') if lang=='en' else (f'记录甲： {second} 把密封包裹递给 {first}。'+''.join(f'无关记录{i+1}：房间里有{i+2}把椅子。' for i in range(4+case%5))+f'后来 {a} 把杯子递给 {b}。')
        question=f'Was {a} the recipient of the sealed packet?' if lang=='en' else f'{a} 是密封包裹的接收者吗？'
        edges=[(second,'packet_giver',first),(a,'cup_giver',b)]
    else:
        item=('apple','pear','banana','orange')[case%4] if lang=='en' else ('苹果','梨','香蕉','橙子')[case%4]
        middle='fruit' if lang=='en' else '水果';end='food' if lang=='en' else '食物'
        x,y=(item,middle) if world==0 else (middle,item)
        if lang=='en':
            body=f'This fictional catalogue uses only its stated rules. Rule A: every {x} is also {y}. Rule B: every {middle} is also {end}. Item X{case} belongs to {item}. No means not necessarily entailed; do not use outside facts.'
            question=f'Do these rules imply that X{case} is {end}?'
        else:
            body=f'这份虚构分类表只使用明示规则。规则甲：每个 {x} 都属于 {y}。规则乙：每个 {middle} 都属于 {end}。对象X{case}属于 {item}。否表示不一定能推出，不使用外部常识。'
            question=f'能仅从规则推出X{case}属于 {end} 吗？'
        edges=[(f'X{case}','instance_of',item),(x,'subclass_of',y),(middle,'subclass_of',end)]
        reached={item}
        for _ in range(3):
            for s,r,t in edges:
                if r=='subclass_of' and s in reached:reached.add(t)
        assert (end in reached)==truth
    return body,question,truth,[dict(source=s,relation=r,target=t) for s,r,t in edges]


def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    assert read(BASE/'science_queue/status.json')['all_passed']
    assert read(BASE/'followup/result.json')['all_passed']
    start=time.monotonic();guard(650*1024**2)
    used=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'));remaining=read(BASE/'resources.json')['compute_ceiling_seconds']-used
    assert remaining>2000,('Whole continuation plus final audit needs2000recorded seconds',remaining)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);rows=[];pair_audit=[]
    for family in FAMILIES:
      for case in range(16):
       group=f'identity2744/{family}/{case:02d}'
       for lang in ['en','zh']:
        pair=[]
        for world in [0,1]:
            body,question,truth,edges=recipe(family,case,lang,world)
            choices=['Yes','No'] if lang=='en' else ['是','否'];target=choices[0 if truth else 1]
            rule='Answer only Yes or No.' if lang=='en' else '只回答是或否。'
            text=body+'\n'+question+'\n'+rule
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=[tok(c,add_special_tokens=False)['input_ids'] for c in choices]
            assert all(len(v)==1 for v in ids)
            r={'sample_id':'q2744_'+rank(group+'/'+lang+'/'+str(world))[:20],'source_group':group,'pair_id':group+'/'+lang,
               'family':family,'cohort':family,'language':lang,'representation':lang,'world':world,'case':case,'kind':'controlled_language',
               'truth':truth,'target':target,'candidate_texts':choices,'candidate_ids':[v[0] for v in ids],'target_ids':ids[0 if truth else 1],
               'body':body,'question':question,'original_text':text,'text':prompt,'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],
               'relations':edges,'split':'prospective_identity_confirmation','capture_mode':'identifiability',
               'novelty':'New paired relation/role assignment using historically exposed vocabulary/context ingredients; not an unseen-vocabulary claim and not authentic natural corpus.'}
            pair.append(r);rows.append(r)
        matched=Counter(pair[0]['prompt_ids'])==Counter(pair[1]['prompt_ids'])
        assert matched,('Strict full actual token multiset mismatch; no silent fallback',family,case,lang)
        assert pair[0]['target']!=pair[1]['target'] and pair[0]['question']==pair[1]['question']
        pair_audit.append({'pair_id':pair[0]['pair_id'],'same_full_prompt_token_multiset':matched,'same_question':True,'opposite_recipe_answers':True,
          'length':len(pair[0]['prompt_ids']),'source_group':group,'sample_ids':[r['sample_id'] for r in pair]})
    assert len(rows)==320 and len({tuple(r['prompt_ids']) for r in rows})==320
    # Natural calibration is separate from the recipe-derived labels above.
    detailed=set(read(BASE/'material/protocol.json')['detailed_prefix_ids']);natural=[]
    main=[r for r in gzread(BASE/'material/natural.json.gz') if r['sample_id'] in detailed and r['split']=='validation']
    for r in main+gzread(BASE/'followup/material.json.gz'):
        for j,pos in enumerate([len(r['prompt_ids'])//2,2*len(r['prompt_ids'])//3]):
            natural.append({'sample_id':r['sample_id']+f'_cal{j}','source_group':r['source_group'],'cohort':r['cohort'],
              'ids':r['prompt_ids'][:pos+1],'target':r['prompt_ids'][pos+1],'split':'calibration' if r['split']=='validation' else 'prospective_natural',
              'kind':'natural_content','position':pos,'parent_id':r['sample_id']})
    assert len(natural)==384
    train=gzread(BASE/'formation/material.json.gz')['train'];counts=Counter(r['target'] for r in train)
    required=['analysis/phase2741.json','analysis/phase2742.json','formation/result.json','followup/result.json','rules/decoder.npz']
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'phase':2744,'same_authorized_goal':True,'actual_run_admitted':True,
      'recorded_remaining_seconds':remaining,'reserved_expected_bytes':650*1024**2,'whole_stage_reference_seconds':1800,'audit_seconds_reserved':200,
      'common_question':'Do query relationships and continued-training gains contain relational information beyond matched token identity and simple output calibration?',
      'reason':'Main ordered MSE/KL dissociation, mostly unsuccessful semantic-pair transfer controls, and stronger permuted-label NLL gains motivate an identifiability test, not another unqualified closure claim.',
      'controlled_rows':320,'semantic_groups':80,'strict_token_multiset_pairs':160,'families':FAMILIES,'variants':VARIANTS,
      'query_scope':'Native100queries with unchanged five prefix-rule decoders; four actually trained BF16parameter variants evaluated on six fixed queries only. Every selected response keeps all2560coordinates.',
      'variant_queries':QUERIES,'free_generation':'All320paired expressions pervariant; independent B8 own histories,128token cap, exact same row grouping, no logit bias or gold input.',
      'generation_cap':128,'batch_rows':8,'temperatures':TEMPERATURES,'mixture_alpha':MIXTURES,
      'calibration_scope':'192held-validation content positions select scalar temperature and train-target-prior mixture. Evaluate192new content positions from96followup-reserved documents. No target or output of those positions enters calibrator selection.',
      'frequency_prior':'Add-one smoothed151936vocabulary counts from576originaltrainingpositions, same fixed prior for allparameter variants; no gold labels from confirmation.',
      'native_parameter_formation':'Deploy each of four actual74711040-scalar deltas from2742 after reconstructing original+FP32delta and castingtoBF16. No new training run or historical pretraining claim.',
      'native_parameter_structure':'All block16/35gate/up/activation units at every controlled original prompt; native full37layeranchors, trainedvariantH16/H17/H36anchors. All original coordinates, no TopK/PCA.',
      'data_retention':'Permanent full native100querypostnorm, trained6querypostnorm, specified full-layer/unit fields and outputs; transient full native sourceKV can be recomputed from frozen input and original weights. No prior evidence deleted.',
      'inference':'Fixed natural loss/permutation result is discovery evidence, not fresh confirmation. Controlled pair group—not each variant/language/coordinate—is bootstrap unit. Recipe fields are externally known relations, not identified native semantics.',
      'required_evidence':[{'path':r,'sha256':sha(BASE/r)} for r in required]}
    payload={'controlled':rows,'natural':natural,'pairs':pair_audit,'train_token_counts':dict(counts)}
    compressed(OUT/'material.json.gz',payload);immutable(p,protocol);ledger('identity_calibration_protocol',time.monotonic()-start)
    return protocol,payload


def deployed_parameters(model,variant,original):
    """In-memory actual parameter deployment; original on-disk checkpoint is read-only."""
    import torch
    target=model.model.layers[16].mlp
    arrays=None
    if variant!='native':
        folder=BASE/'formation'/variant;r=read(folder/'result.json');assert sha(folder/'parameter_delta_FP32.npz')==r['delta_sha256']
        arrays=np.load(folder/'parameter_delta_FP32.npz')
    with torch.no_grad():
      for name,p in target.named_parameters():
        value=original[name] if arrays is None else original[name]+torch.from_numpy(arrays[name])
        p.copy_(value.to(device=p.device,dtype=torch.bfloat16))
    if arrays is not None:arrays.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze']);args=p.parse_args();print(freeze()[0])
