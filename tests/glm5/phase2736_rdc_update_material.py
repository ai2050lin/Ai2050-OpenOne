"""New source occurrences, content boundaries, and executable mixed operation graphs."""
from collections import Counter,defaultdict
import random
from rdc_update_common import *

OPS=('alias','mapping','conditional','addition')

def content_anchors(row,count=3):
    # Retrospective labels select measurement locations only; never predictor inputs.
    offsets=row['token_offsets'];choices=[]
    for word in row['retrospective_ud']:
        span=word.get('char_span');upos=word['upos']
        if not span or upos not in ('NOUN','PROPN','VERB','ADJ','ADV'):continue
        match=[i for i,(s,e) in enumerate(offsets) if e>s and s<span[1] and e>span[0]]
        if match and match[0]>max(8,len(offsets)//3):choices.append((match[0]-1,word['form'],upos,word['relation']))
    choices=list({x[0]:x for x in choices}.values());choices.sort()
    if len(choices)<count:return None
    ii=np.unique(np.linspace(0,len(choices)-1,count,dtype=int))
    return [choices[i] for i in ii]

def mixed_programs(tok):
    result=[]
    # Hold out ordered operation bigrams conditional→mapping and addition→alias
    # from all fitting programs; complete case families share their split.
    held={('conditional','mapping'),('addition','alias')}
    for case in range(192):
        split='train' if case<96 else 'validation' if case<128 else 'test' if case<160 else 'mixed_holdout'
        rng=random.Random('rdc2736-mixed/'+str(case));depth=2+(case%2) if case<160 else 4+(case%2)*2
        while True:
            seq=[rng.choice(OPS) for _ in range(depth)]
            has=any(tuple(seq[j:j+2]) in held for j in range(len(seq)-1))
            if len(set(seq))>=2 and has==(split=='mixed_holdout'):break
        names=[f'v_{case}_{j}' for j in range(depth+1)]
        value=rng.randrange(1,9);initial=value;en=[f'{names[0]} has value {value}.'];zh=[f'{names[0]}的值为{value}。'];code=[f'{names[0]} = {value}'];edges=[]
        for j,op in enumerate(seq,1):
            a,b=names[j-1:j+1];before=value;params={}
            if op=='alias':
                en.append(f'{b} copies the value of {a}.');zh.append(f'{b}复制{a}的值。');code.append(f'{b} = {a}')
            elif op=='mapping':
                table=list(range(1,9));rng.shuffle(table);value=table[value-1];params={'mapping':table}
                s=', '.join(f'{i+1}: {v}' for i,v in enumerate(table))
                en.append(f'Set {b} by looking up {a} in the table {{{s}}}.');zh.append(f'用表{{{s}}}查询{a}，结果赋给{b}。');code.append(f'{b} = {{{s}}}[{a}]')
            elif op=='conditional':
                t=rng.randrange(2,8);value=9-value if value>t else value;params={'threshold':t}
                en.append(f'{b} equals 9 minus {a} if {a} is greater than {t}, and equals {a} otherwise.')
                zh.append(f'若{a}大于{t}，{b}等于9减{a}；否则{b}等于{a}。');code.append(f'{b} = 9 - {a} if {a} > {t} else {a}')
            else:
                t=rng.randrange(1,5);value=(value+t-1)%8+1;params={'increment':t}
                en.append(f'{b} is {a} plus {t}, wrapping from 8 to 1.');zh.append(f'{b}等于{a}加{t}，超过8后从1循环。');code.append(f'{b} = ({a} + {t} - 1) % 8 + 1')
            edges.append({'source':a,'target':b,'operation':op,'step':j,'input_value':before,'output_value':value,**params})
        namespace={};exec('\n'.join(code),{'__builtins__':{}},namespace);assert namespace[names[-1]]==value
        variants={'en':' '.join(en)+f' What is {names[-1]}? Output only its digit.',
          'zh':''.join(zh)+f'{names[-1]}是多少？只输出一个数字。',
          'python':'What digit is printed? Output the digit only.\n```python\n'+'\n'.join(code)+f'\nprint({names[-1]})\n```',
          'en_reordered':' '.join(reversed(en))+f' These are definitions, not sequential assignments. What is {names[-1]}? Output only its digit.'}
        for rep,text in variants.items():
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);group=f'mixed2736/{case:03d}'
            result.append({'sample_id':'u2736_'+ranked(group+'/'+rep)[:20],'source_group':group,'kind':'controlled_program','cohort':'mixed_'+rep,
              'split':split,'family':'mixed_operations','operation_sequence':seq,'depth':depth,'representation':rep,'language':'zh' if rep=='zh' else 'en',
              'original_text':text,'text':prompt,'program':'\n'.join(code),'initial_value':initial,'target':str(value),'relations':edges,
              'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],'anchors':[len(enc['input_ids'])-1],
              'target_ids':tok(str(value),add_special_tokens=False)['input_ids'],'capture_mode':'update',
              'novelty':'New deterministic mixed-operation family. No fitting family contains conditional→mapping or addition→alias; test also changes depth, which is separately labeled.'})
    assert len(result)==768 and len({r['sample_id'] for r in result})==768
    assert all(len(r['target_ids'])==1 for r in result)
    return result

def main():
    from transformers import AutoTokenizer
    from phase2728_rdc_law_material import natural_candidates
    from phase2732_rdc_binding_material import connected
    if (BASE/'material_frozen.json').exists():return
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    old=prior_natural()+gzread(LAW/'material.json.gz')+gzread(LAW/'confirmation_material.json.gz')
    used={c for r in old for c in r.get('component_ids',[])};texts={tuple(r['prompt_ids']) for r in old}
    train_groups={r['source_group'] for r in old if r.get('split')=='train'}
    manifest=[];natural=[];availability=[]
    for bank in ('gum','ewt'):
        candidates=natural_candidates(tok,bank,'test',manifest)
        for r in candidates:r['connected_held']=connected(r)
        for want in (True,False):
            pool=[r for r in candidates if bool(r['connected_held'])==want and not set(r['component_ids'])&used and r['source_group'] not in train_groups and tuple(r['prompt_ids']) not in texts and content_anchors(r)]
            grouped=defaultdict(list)
            for r in sorted(pool,key=lambda r:ranked(r['source_id'])):grouped[r['source_group']].append(r)
            selected=[];groups=sorted(grouped,key=ranked)
            for round_index in range(32):
              for group in groups:
                eligible=[r for r in grouped[group] if not set(r['component_ids'])&used and tuple(r['prompt_ids']) not in texts]
                if not eligible:continue
                r=eligible[0];anchors=content_anchors(r);sid='u2736_'+ranked(bank+'/'+'/'.join(r['component_ids']))[:20]
                rr=dict(r,sample_id=sid,cohort=bank,split='new_connected' if want else 'new_matched',capture_mode='update',
                  anchors=[x[0] for x in anchors],target_positions=[x[0]+1 for x in anchors],content_boundary_annotations=[{'position':p,'next_word':w,'upos':u,'relation':rel} for p,w,u,rel in anchors],
                  novelty='No component or exact token sequence in the explicitly inventoried law/binding materials; shared public corpus, not a guarantee against all historical exposure.')
                selected.append(rr);used.update(r['component_ids']);texts.add(tuple(r['prompt_ids']))
                if len(selected)==32:break
              if len(selected)==32:break
            availability.append({'cohort':bank,'connected':want,'eligible_initial':len(pool),'selected':len(selected),'documents':len({r['source_group'] for r in selected})})
            assert len(selected)==32,availability
            natural.extend(selected)
    program=mixed_programs(tok)
    for name,rows in [('natural',natural),('program',program)]:
        assert len({r['sample_id'] for r in rows})==len(rows)
        assert len({tuple(r['prompt_ids']) for r in rows})==len(rows)
        compressed(BASE/f'{name}_material.json.gz',rows)
    immutable(BASE/'material_protocol.json',{'natural_rows':128,'natural_boundaries_per_row':3,'natural_target':'UPOS noun/propernoun/verb/adjective/adverb next-word starts, measurement-location annotations not predictor inputs',
      'program_rows':768,'semantic_families':192,'representations':['en','zh','python','en_reordered'],'train_families':96,'validation_families':32,'test_families':32,'mixed_holdout_families':32,
      'holdout_ordered_bigrams':[['conditional','mapping'],['addition','alias']],
      'secondary_confounds':'Holdout bigram and depth change jointly. No claim of new vocabulary or entire natural-language semantics; shared syntax tokens/templates remain.',
      'native_capture':'Freeze predictors before new confirmation inference. Full token coordinate scan, full sources, full layer anchors and all block6/16/35 units; selected full-field fixtures declared before inference.'})
    result={'timestamp':stamp(),'source':snapshot(__file__),'natural_availability':availability,'raw_sources':manifest,
      'natural_sha256':sha(BASE/'natural_material.json.gz'),'program_sha256':sha(BASE/'program_material.json.gz'),
      'counts':{'natural':128,'program':768},'token_lengths':{'natural':[min(map(lambda r:len(r['prompt_ids']),natural)),max(map(lambda r:len(r['prompt_ids']),natural))],
        'program':[min(map(lambda r:len(r['prompt_ids']),program)),max(map(lambda r:len(r['prompt_ids']),program))]},'seconds':time.monotonic()-start}
    save(BASE/'material_frozen.json',result);ledger('freeze_mixed_and_new_natural',result['seconds']);print('UPDATE_MATERIAL',result,flush=True)

if __name__=='__main__':main()
