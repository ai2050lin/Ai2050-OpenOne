"""New explicit-record benchmark; generated labels from typed relations, not model output."""
from rdc_continuity_common import *
FAMILIES=('taxonomy_chain','attribute_binding','agent_patient','comparison','negation_scope','reference_transfer','punctuation','conjunction')
NAMES=[('Quinn','Rhea','奎因','瑞娅'),('Soren','Tessa','索伦','泰莎'),('Ulric','Vera','乌尔里克','薇拉'),('Willa','Xavi','薇拉娜','哈维'),('Yara','Zane','雅拉','赞恩'),('Alba','Bram','阿尔巴','布拉姆'),('Cora','Dax','科拉','达克斯'),('Elio','Faye','埃利奥','费伊'),('Greta','Heath','格蕾塔','希思'),('Ida','Jett','伊达','杰特'),('Kaia','Lars','凯娅','拉尔斯'),('Milo','Nell','米洛','内尔'),('Orla','Pax','奥拉','帕克斯'),('Remy','Sage','雷米','塞奇'),('Troy','Uma','特洛伊','乌玛'),('Vito','Wren','维托','雷恩')]
OBJECTS=list(zip(('ticket','ribbon','bottle','spoon','basket','scarf','coin','drum','badge','notebook','umbrella','rope','pencil','helmet','tray','envelope'),('票','丝带','瓶子','勺子','篮子','围巾','硬币','鼓','徽章','笔记本','雨伞','绳子','铅笔','头盔','托盘','信封')))
WORDS=list(zip(('quail','pike','cedar','moth','beet','flute','tulip','ferry','peach','otter','elm','swan','radish','cello','lily','scooter'),('鹌鹑','狗鱼','雪松','飞蛾','甜菜','长笛','郁金香','渡轮','桃子','水獭','榆树','天鹅','萝卜','大提琴','百合','滑板车')))
CATS=[('bird','animal','organism','鸟','动物','生物'),('fish','animal','organism','鱼','动物','生物'),('tree','plant','organism','树','植物','生物'),('insect','animal','organism','昆虫','动物','生物'),('vegetable','food','object','蔬菜','食物','物体'),('instrument','object','item','乐器','物体','物品'),('flower','plant','organism','花','植物','生物'),('vehicle','object','item','交通工具','物体','物品')]

def scenario(f,i,t,lang):
    a,b,az,bz=NAMES[i];obj,oz=OBJECTS[i];word,wz=WORDS[i];c,d,e,cz,dz,ez=CATS[i%8]
    zh=lang=='zh'
    if zh:a,b,obj,word,c,d,e=az,bz,oz,wz,cz,dz,ez
    if f=='taxonomy_chain':
        depth=2+i%2;top=d if depth==2 else e
        facts=(f'{word}属于{c}。{c}都属于{d}。'+(f'{d}都属于{e}。' if depth==3 else '')) if zh else (f'The {word} belongs to category {c}. Everything in category {c} belongs to category {d}. '+(f'Everything in category {d} belongs to category {e}.' if depth==3 else ''))
        if not t:facts=facts.replace(f'{c}都属于{d}',f'{c}都不属于{d}') if zh else facts.replace(f'Everything in category {c} belongs to category {d}',f'Nothing in category {c} belongs to category {d}')
        # Negative chain needs explicit denial of the target, avoiding invalid non-transitivity.
        if not t and depth==3:facts+=f'该{word}不属于{e}。' if zh else f' This {word} does not belong to category {e}.'
        statement=f'{word}属于{top}。' if zh else f'The {word} belongs to category {top}.'
        return facts,statement,word,c,depth
    if f=='attribute_binding':
        x,y=('光滑','粗糙') if zh else ('smooth','rough');ca,cb=(x,y) if t else (y,x)
        return (f'{a}的{obj}是{ca}的；{b}的{obj}是{cb}的。' if zh else f'The {obj} owned by {a} is {ca}, while the one owned by {b} is {cb}.'), (f'{a}的{obj}是{x}的。' if zh else f"{a}'s {obj} is {x}."),a,b,1
    if f=='agent_patient':
        actor,patient=(a,b) if t else (b,a)
        facts=(f'{patient}被{actor}邀请了。' if i%2 else f'{actor}邀请了{patient}。') if zh else (f'{patient} was invited by {actor}.' if i%2 else f'{actor} invited {patient}.')
        return facts,(f'{a}是邀请{b}的人。' if zh else f'The person inviting {b} was {a}.'),a,b,1
    if f=='comparison':
        lo,hi=103+7*i,108+7*i;na,nb=(hi,lo) if t else (lo,hi)
        return (f'{a}走了{na}米；{b}走了{nb}米。' if zh else f'{a} walked {na} metres; {b} walked {nb} metres.'),(f'{a}走得比{b}远。' if zh else f'{a} travelled farther than {b}.'),a,b,1
    if f=='negation_scope':
        facts=(f'{a}报告{b}没有到达。' if t else f'{a}没有报告{b}到达。') if zh else (f'{a} reported that {b} had not arrived.' if t else f'{a} did not report that {b} had arrived.')
        return facts,(f'{a}报告了{b}未到达的消息。' if zh else f'{a} reported the absence of {b}.'),a,b,1
    if f=='reference_transfer':
        tail=(f'接收者将它收好。' if t else f'接收者将它退还给{a}。') if zh else ('The receiver stored it.' if t else f'The receiver handed it back to {a}.')
        return (f'{a}将{obj}递给{b}。' if zh else f'{a} passed the {obj} to {b}. ')+tail,(f'事件结束时{b}持有{obj}。' if zh else f'After these events, {b} possesses the {obj}.'),a,b,2
    if f=='punctuation':
        return (f'{a}拿到了{obj}' if zh else f'{a} obtained the {obj}')+(('？' if t else '！') if zh else ('?' if t else '!')),('最后一个标点是问号。' if zh else 'The final punctuation is a question mark.'),a,obj,1
    # Truth of conjunction needs both facts. Order and failing conjunct alternate with unit.
    p,q=(True,True) if t else ((False,True) if i%2 else (True,False))
    facts=(f'{a}'+('已' if p else '未')+f'到达。{b}'+('已' if q else '未')+'到达。') if zh else f'{a} has '+('' if p else 'not ')+f'arrived. {b} has '+('' if q else 'not ')+'arrived.'
    return facts,(f'{a}和{b}都已到达。' if zh else f'Both {a} and {b} have arrived.'),a,b,2

def build(tok):
    rows=[]
    for fi,f in enumerate(FAMILIES):
      for i in range(16):
       for t in (False,True):
        for q in (False,True):
         for lang in ('en','zh'):
          facts,statement,u,v,depth=scenario(f,i,t,lang)
          if lang=='en':
            system='Use only the supplied record to assess support for the claim. Accept the record even when it is fictional. Absence of support counts as not supported. Respond with Yes or No.'
            questions=[('Can the claim be justified using this record?','Is the claim unjustified using this record?'),('Is there support here for the claim?','Is support for the claim missing here?')]
            question=questions[i%2][int(q)];body=f'Evidence: {facts}\nClaim: {statement}\n{question}'
          else:
            system='请仅依据所给记录判断断言的支持情况。记录即使是假设也应接受。缺少支持就计为不支持。回答是或否。'
            question=[('这条断言有记录作为依据吗？','这条断言缺少记录依据吗？'),('记录能为断言提供支持吗？','记录是否没有为断言提供支持？')][i%2][int(q)]
            body=f'记录内容：{facts}\n待判断言：{statement}\n{question}'
          prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
          enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);spans={}
          for k,term in [('u',u),('v',v)]:
            s=prompt.index(facts)+facts.index(term);e=s+len(term)
            ix=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>s and a<e and b>a];assert ix
            spans[k]={'term':term,'chars':[s,e],'positions':ix}
          sid=f'e-{f}-{i}-{int(t)}-{int(q)}-{lang}';yes=t!=q
          rows.append(dict(sample_id=sid,base_id=f'{f}-{i}',family=f,family_index=fi,unit=i,form=i%2,language=lang,record=facts,positive_statement=statement,question=question,u=u,v=v,relation=f,depth=depth,fact_truth=t,negative_query=q,expected_yes=yes,target=(('Yes' if yes else 'No') if lang=='en' else ('是' if yes else '否')),prompt=prompt,prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),spans=spans,word_split='train' if i<8 else 'validation' if i<12 else 'test',source_mode='live_model',role='Typed record anchors U,V; C=last real prompt token',material_limit='Explicit-record benchmark. Shared forms and function tokens; not universal natural-language competence.'))
    assert len(rows)==1024 and len({r['prompt'] for r in rows})==1024
    old=read(HISTORY/'b_relations/material.json');assert not ({r['prompt'] for r in old}&{r['prompt'] for r in rows})
    return rows
