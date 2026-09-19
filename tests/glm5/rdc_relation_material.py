"""Eight typed language operations, independent fact/query crossing, grouped bilingual cases."""
from rdc_mechanism_common import *

FAMILIES=('taxonomy_chain','attribute_binding','word_sense','agent_patient','comparison','negation_scope','reference_transfer','punctuation')
PAIRS=[('Ada','Ben','艾达','本'),('Cleo','Dorian','克莱奥','多里安'),('Eira','Finn','艾拉','芬恩'),('Gail','Hugo','盖尔','雨果'),('Iris','Juno','艾瑞丝','朱诺'),('Kira','Leon','琪拉','里昂'),('Mara','Nico','玛拉','尼科'),('Owen','Pia','欧文','皮娅')]
WORDS=[('apple','苹果','fruit','水果','food','食物'),('salmon','鲑鱼','fish','鱼','animal','动物'),('oak','橡树','tree','树','plant','植物'),('sparrow','麻雀','bird','鸟','animal','动物'),('carrot','胡萝卜','vegetable','蔬菜','food','食物'),('violin','小提琴','instrument','乐器','object','物品'),('rose','玫瑰','flower','花','plant','植物'),('truck','卡车','vehicle','车辆','object','物品')]
OBJECTS=[('book','书'),('parcel','包裹'),('key','钥匙'),('lamp','灯'),('cup','杯子'),('box','盒子'),('card','卡片'),('hat','帽子')]
COLORS=[('red','红色'),('blue','蓝色'),('green','绿色'),('yellow','黄色'),('black','黑色'),('white','白色'),('purple','紫色'),('orange','橙色')]
VERBS=[('interviewed','采访了'),('helped','帮助了'),('followed','跟随了'),('thanked','感谢了'),('called','呼叫了'),('greeted','问候了'),('praised','表扬了'),('photographed','拍摄了')]
SENSES=[
 ('bank','银行','river','河流','cash','现金','a riverside bank','河岸'),
 ('bat','球拍','cave','洞穴','ball','球','a flying animal','飞行动物'),
 ('bark','树皮','tree','树','dog','狗','the outer covering of a tree','树木外层'),
 ('crane','起重机','wetland','湿地','construction','施工','a bird','鸟'),
 ('seal','海豹','ocean','海洋','envelope','信封','a marine animal','海洋动物'),
 ('spring','弹簧','season','季节','metal','金属','a season','季节'),
 ('match','火柴','flame','火焰','contest','比赛','a small fire-lighting stick','点火小木棍'),
 ('club','俱乐部','members','会员','wooden','木制','an association','组织')]

def scenario(f,i,t,lang):
    a,b,az,bz=PAIRS[i]; obj,objz=OBJECTS[i]; color,colorz=COLORS[i];verb,verbz=VERBS[i]
    if lang=='zh':a,b,obj,color,verb=az,bz,objz,colorz,verbz
    if f=='taxonomy_chain':
        word,wz,middle,mz,top,tz=WORDS[i]
        if lang=='zh':word,middle,top=wz,mz,tz
        # The negative branch explicitly denies the second edge; not an unknown-world inference.
        facts=(f'{word} is a {middle}. Every {middle} is '+('' if t else 'not ')+f'a {top}.') if lang=='en' else f'{word}属于{middle}。所有{middle}'+('属于' if t else '不属于')+f'{top}。'
        statement=f'{word} is a {top}.' if lang=='en' else f'{word}属于{top}。'
        return facts,statement,word,middle,'subclass_composition','Counterfactual task facts may conflict with ordinary knowledge; score given record, not encyclopedic truth.'
    if f=='attribute_binding':
        other,oz=COLORS[(i+1)%8];other=other if lang=='en' else oz
        ca,cb=(color,other) if t else (other,color)
        facts=f"{a}'s {obj} is {ca}; {b}'s {obj} is {cb}." if lang=='en' else f'{a}的{obj}是{ca}，{b}的{obj}是{cb}。'
        statement=f"{a}'s {obj} is {color}." if lang=='en' else f'{a}的{obj}是{color}。'
        return facts,statement,a,b,'owner_attribute','Owner and property binding.'
    if f=='word_sense':
        word,wz,c0,c0z,c1,c1z,meaning,mz=SENSES[i]
        if lang=='en':
            texts=[('They sat on the bank beside the river.','They deposited cash at the bank.'),('The bat flew from the cave.','The bat hit the ball.'),('The bark covered the tree.','The dog gave a loud bark.'),('A crane stood in the wetland.','A crane lifted steel at the construction site.'),('The seal swam in the ocean.','A seal closed the envelope.'),('The spring season follows winter.','The metal spring was compressed.'),('The match produced a flame.','They won the contest in the final match.'),('The club welcomed its members.','He carried a wooden club.')]
            facts=texts[i][0 if t else 1];statement=f'Here, "{word}" refers to {meaning}.';return facts,statement,word,c0 if t else c1,'contextual_word_sense','Actual English ambiguous word; not always same orthographic ambiguity in Chinese.'
        # Chinese-specific polysemy uses genuinely identical written tokens in two contexts.
        z=[('苹果','他吃了一个苹果。','苹果发布了新款手机。','一种水果','吃','手机'),('花','花在花园里开放。','他花了十元钱。','植物的花朵','花园','钱'),('面','碗里有面。','这个盒子有六个面。','面条','碗','盒子'),('开','他开了门。','花开了。','打开','门','花'),('长','这条绳子很长。','孩子长高了。','长度较大','绳子','孩子'),('打','他打了球。','他打了一个电话。','击打','球','电话'),('光','房间里有光。','他把饭吃光了。','光线','房间','饭'),('书','这本书很厚。','请书写姓名。','一本读物','厚','姓名')][i]
        word,s0,s1,meaning,c0,c1=z;return s0 if t else s1,f'这里的“{word}”表示{meaning}。',word,c0 if t else c1,'contextual_word_sense','Chinese-specific senses; not token-aligned translation of English.'
    if f=='agent_patient':
        agent,patient=(a,b) if t else (b,a)
        if lang=='en':facts=f'{agent} {verb} {patient}.' if i%2==0 else f'{patient} was {verb} by {agent}.'
        else:facts=f'{agent}{verb}{patient}。' if i%2==0 else f'{patient}被{agent}{verb}。'
        statement=f'{a} was the one who {verb} {b}.' if lang=='en' else f'是{a}{verb}{b}。'
        return facts,statement,a,b,'agent_patient','Active/passive forms; explicit semantic-role reversal.'
    if f=='comparison':
        low=7+i*3;high=low+5;na,nb=(high,low) if t else (low,high)
        facts=f'{a} collected {na} coins; {b} collected {nb} coins.' if lang=='en' else f'{a}收集了{na}枚硬币，{b}收集了{nb}枚硬币。'
        statement=f'{a} collected more coins than {b}.' if lang=='en' else f'{a}收集的硬币比{b}多。'
        return facts,statement,a,b,'ordered_quantity','Numbers and names are grouped by base case; order changes relation.'
    if f=='negation_scope':
        facts=(f'{a} said that {b} did not leave.' if t else f'{a} did not say that {b} left.') if lang=='en' else (f'{a}说{b}没有离开。' if t else f'{a}没有说{b}离开了。')
        statement=f'{a} said that {b} did not leave.' if lang=='en' else f'{a}说{b}没有离开。'
        return facts,statement,a,b,'scope_of_negation','Negative branch is not entailment of the inner-negation claim; use supported/not-supported policy, not unknown=false factual ontology.'
    if f=='reference_transfer':
        facts=(f'{a} gave the {obj} to {b}. The recipient kept it.' if t else f'{a} gave the {obj} to {b}. The recipient returned it to {a}.') if lang=='en' else (f'{a}把{obj}交给{b}。收件人保留了它。' if t else f'{a}把{obj}交给{b}。收件人把它退回给{a}。')
        statement=f'{b} holds the {obj} at the end.' if lang=='en' else f'最后{obj}在{b}手中。'
        return facts,statement,a,b,'reference_and_transfer','Role-linked recipient and final possession, not only nearest token matching.'
    facts=(f'{a} found the {obj}'+('?' if t else '.')) if lang=='en' else f'{a}找到了{obj}'+('？' if t else '。')
    statement='The record ends with a question mark.' if lang=='en' else '材料以问号结束。'
    return facts,statement,a,obj,'punctuation_function','Punctuation identification control, not a claim about all utterance semantics.'

def build(tok):
    rows=[]
    for fi,f in enumerate(FAMILIES):
        for i in range(8):
            for truth in (False,True):
                for negative in (False,True):
                    for lang in ('en','zh'):
                        facts,statement,u,v,relation,limit=scenario(f,i,truth,lang)
                        if lang=='en':
                            system='Judge only whether the statement is supported by the record. Use the stated facts even if counterfactual. Not supported includes information absent from the record. Answer only Yes or No.'
                            question=('Is the statement NOT supported by the record?' if negative else 'Is the statement supported by the record?') if i%2==0 else ('Does the record fail to establish the statement?' if negative else 'Does the record establish the statement?')
                            body=f'Record: {facts}\nStatement: {statement}\n{question}'
                        else:
                            system='只判断陈述是否得到材料支持。即使材料与常识不同，也按给定事实判断。材料未说明也算不支持。只回答是或否。'
                            question=('陈述是否未得到材料支持？' if negative else '陈述是否得到材料支持？') if i%2==0 else ('是否不能由材料得出这个陈述？' if negative else '是否可以由材料得出这个陈述？')
                            body=f'材料：{facts}\n陈述：{statement}\n{question}'
                        prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        e=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);spans={};start_record=prompt.index(facts)
                        for key,term in [('u',u),('v',v)]:
                            start=start_record+facts.index(term);end=start+len(term)
                            positions=[j for j,(s,t) in enumerate(e['offset_mapping']) if t>start and s<end and t>s]
                            assert positions;spans[key]={'term':term,'chars':[start,end],'positions':positions}
                        qs=prompt.index(question);qpositions=[j for j,(s,t) in enumerate(e['offset_mapping']) if t>qs and s<qs+len(question) and t>s]
                        expected=bool(truth)!=bool(negative);sid=f'b-{f}-{i}-{int(truth)}-{int(negative)}-{lang}'
                        rows.append(dict(sample_id=sid,base_id=f'{f}-{i}',family=f,family_index=fi,unit=i,form=int(negative),
                            language=lang,u=u,v=v,positive_statement=statement,record=facts,relation=relation,material_limit=limit,
                            fact_truth=truth,negative_query=negative,expected_yes=expected,target=('Yes' if expected else 'No') if lang=='en' else ('是' if expected else '否'),
                            prompt=prompt,prompt_ids=e['input_ids'],tokens=tok.convert_ids_to_tokens(e['input_ids']),spans=spans,
                            context_positions=qpositions,word_split='train' if i<4 else 'validation' if i<6 else 'test',
                            source_mode='live_model',role='U primary record anchor; V other participant or disambiguating cue; roles typed per family'))
    assert len(rows)==512 and len({r['prompt'] for r in rows})==512
    for f in FAMILIES:
        for lang in ('en','zh'):
            for split in ('train','validation','test'):
                group=[r for r in rows if (r['family'],r['language'],r['word_split'])==(f,lang,split)]
                assert sum(r['expected_yes'] for r in group)==len(group)//2
    return rows
