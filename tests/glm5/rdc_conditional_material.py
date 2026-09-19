"""Independently crossed surface factors, controlled typed-record semantics."""
from itertools import product
from rdc_conditional_common import *
FAMILIES=('taxonomy_chain','attribute_binding','agent_patient','comparison','negation_scope','reference_transfer','punctuation','conjunction')
NAMES=[('Adrian','Beatrice','阿德里安','比阿特丽丝'),('Calvin','Daphne','卡尔文','达芙妮'),('Edwin','Flora','埃德温','弗洛拉'),('Gavin','Hazel','加文','海泽尔'),('Ivan','Jocelyn','伊凡','乔斯琳'),('Keaton','Leona','基顿','莉奥娜'),('Morgan','Nadia','摩根','娜迪娅'),('Oscar','Petra','奥斯卡','佩特拉'),('Ronan','Selma','罗南','塞尔玛'),('Tristan','Ursula','特里斯坦','乌尔苏拉'),('Victor','Wendy','维克托','温迪'),('Xander','Yvette','赞德','伊薇特'),('Alden','Bianca','奥尔登','比安卡'),('Cedric','Diana','塞德里克','戴安娜'),('Emmett','Freya','埃米特','芙蕾雅'),('Gordon','Helena','戈登','海伦娜')]
OBJECTS=list(zip(('lantern','suitcase','teapot','compass','wallet','vase','key','parcel','brush','mirror','map','towel','cup','violin','folder','clock'),('灯笼','手提箱','茶壶','指南针','钱包','花瓶','钥匙','包裹','刷子','镜子','地图','毛巾','杯子','小提琴','文件夹','时钟')))


def scenario(f,i,t,form,lang):
    a,b,az,bz=NAMES[i];obj,oz=OBJECTS[i];zh=lang=='zh'
    if zh:a,b,obj=az,bz,oz
    if f=='taxonomy_chain':
        # Equal three-edge positive/negative proofs; synthetic category names are task facts.
        c,d,e=(('甲类','乙类','丙类') if zh else ('category amber','category cobalt','category ivory'))
        clauses=([f'{a}的{obj}属于{c}。',f'{c}的成员都属于{d}。',f'{d}的成员都'+('属于' if t else '不属于')+f'{e}。'] if zh else [f'The {obj} owned by {a} is in {c}.',f'Every member of {c} is in {d}.',('Every member' if t else 'No member')+f' of {d} is in {e}.'])
        if form:clauses=clauses[::-1]
        return ' '.join(clauses),(f'{a}的{obj}属于{e}。' if zh else f'The {obj} owned by {a} is in {e}.'),a,obj,3
    if f=='attribute_binding':
        x,y=('透明','不透明') if zh else ('transparent','opaque');ca,cb=(x,y) if t else (y,x)
        clauses=([f'{a}的{obj}是{ca}的。',f'{b}的{obj}是{cb}的。'] if zh else [f"{a}'s {obj} is {ca}.",f"{b}'s {obj} is {cb}."])
        if form:clauses.reverse()
        return ' '.join(clauses),(f'{a}的{obj}是{x}的。' if zh else f"{a}'s {obj} is {x}."),a,b,1
    if f=='agent_patient':
        actor,patient=(a,b) if t else (b,a)
        facts=(f'{patient}被{actor}感谢了。' if form else f'{actor}感谢了{patient}。') if zh else (f'{patient} was thanked by {actor}.' if form else f'{actor} thanked {patient}.')
        return facts,(f'{a}感谢了{b}。' if zh else f'{a} thanked {b}.'),a,b,1
    if f=='comparison':
        lo,hi=41+6*i,44+6*i;na,nb=(hi,lo) if t else (lo,hi)
        clauses=([f'{a}有{na}颗珠子。',f'{b}有{nb}颗珠子。'] if zh else [f'{a} has {na} beads.',f'{b} has {nb} beads.'])
        if form:clauses.reverse()
        return ' '.join(clauses),(f'{a}的珠子比{b}多。' if zh else f'{a} has more beads than {b}.'),a,b,1
    if f=='negation_scope':
        if zh:
            facts=(f'{a}说{b}没有离开。' if t else f'{a}没有说{b}离开。') if not form else (f'据{a}所说，{b}并未离开。' if t else f'关于{b}离开一事，{a}并未说过。')
            claim=f'{a}说了{b}未离开的消息。'
        else:
            facts=(f'{a} stated that {b} had not left.' if t else f'{a} did not state that {b} had left.') if not form else (f'According to {a}, {b} had not left.' if t else f'As for {b} having left, {a} made no such statement.')
            claim=f'{a} stated that {b} had not left.'
        return facts,claim,a,b,1
    if f=='reference_transfer':
        tail=(f'接收者把它放进口袋。' if t else f'接收者把它退给了{a}。') if zh else ('The receiver put it in a pocket.' if t else f'The receiver returned it to {a}.')
        facts=((f'{a}把{obj}交给{b}。' if not form else f'{b}从{a}手里接过{obj}。') if zh else (f'{a} handed the {obj} to {b}. ' if not form else f'{b} received the {obj} from {a}. '))+tail
        return facts,(f'事件结束后，{b}持有{obj}。' if zh else f'After these events, {b} holds the {obj}.'),a,b,2
    if f=='punctuation':
        facts=(f'{a}找到了{obj}' if not form else f'{obj}被{a}找到了') if zh else (f'{a} found the {obj}' if not form else f'The {obj} was found by {a}')
        facts+=(('？' if t else '！') if zh else ('?' if t else '!'))
        return facts,('仅记录文本自身的末尾标点是问号。' if zh else 'The final punctuation mark of the record text itself is a question mark.'),a,obj,1
    clauses=([f'{a}'+('已' if t else '未')+'登记。',f'{b}已登记。'] if zh else [f'{a} has '+('' if t else 'not ')+'registered.',f'{b} has registered.'])
    if form:clauses.reverse()
    return ' '.join(clauses),(f'{a}和{b}都已登记。' if zh else f'Both {a} and {b} have registered.'),a,b,2


def build(tok):
    rows=[]
    for fi,f in enumerate(FAMILIES):
      for i in range(16):
       for t,q,form,style,lang in product((False,True),(False,True),(0,1),(0,1),('en','zh')):
        facts,claim,u,v,depth=scenario(f,i,t,form,lang)
        if lang=='en':
            sys='Judge support only from the supplied record, accepting it as given even if fictional. Missing support counts as not supported. Answer Yes or No only.'
            question=('Is the claim supported by the record?' if not q else 'Is support for the claim absent from the record?')
            intro='Evidence review follows.' if style else 'Please check this.'
            body=f'{intro}\n<Record>\n{facts}\n</Record>\nClaim: {claim}\n{question}'
        else:
            sys='只依据所给记录判断支持情况，即使记录是假设也接受。缺少支持计为不支持。只回答是或否。'
            question='记录是否支持断言？' if not q else '记录是否缺少对断言的支持？'
            intro='以下为证据审查。' if style else '请帮忙看一下。'
            body=f'{intro}\n<记录>\n{facts}\n</记录>\n断言：{claim}\n{question}'
        prompt=tok.apply_chat_template([{'role':'system','content':sys},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);spans={}
        for role,term in [('u',u),('v',v)]:
            start=prompt.index(facts)+facts.index(term);end=start+len(term)
            pos=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<end and b>a];assert pos
            spans[role]={'term':term,'chars':[start,end],'positions':pos}
        yes=t!=q;sid=f'i-{fi}-{i}-{int(t)}{int(q)}{form}{style}-{lang}'
        rows.append(dict(sample_id=sid,base_id=f'i-{fi}-{i}',entity_group=i,family=f,family_index=fi,unit=i,
          form=form,style=style,language=lang,fact_truth=t,negative_query=q,expected_yes=yes,
          word_split='train' if i<8 else 'validation' if i<12 else 'test',
          full_token_panel=i in (0,12),depth=depth,record=facts,positive_statement=claim,question=question,u=u,v=v,
          target=(('Yes' if yes else 'No') if lang=='en' else ('是' if yes else '否')),
          system=sys,user=body,prompt=prompt,prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),spans=spans,
          relation_edges=[{'type':f,'source':u,'target':v,'support':t}],source_mode='live_model',
          material_limit='Explicit-record constructed tasks; two forms/styles are not all paraphrases. Form operation differs by family. Shared categories/function words are intentional. No claim of unseen pretraining data.'))
    assert len(rows)==4096 and len({r['prompt'] for r in rows})==4096
    assert sum(r['full_token_panel'] for r in rows)==512
    assert all(len({r['word_split'] for r in rows if r['entity_group']==i})==1 for i in range(16))
    return rows
