"""Eight controlled language families with explicit sample units and strict output checks."""
import re
from phase2620_native_coordinate_contract import *

FAMILIES=('chronology','word_sense','taxonomy','syntax_role','negation','reference','punctuation','long_reorder')
NAMES_EN=['Ada','Boris','Celia','Dario','Esther','Felix','Greta','Hector','Ines','Jules','Katya','Lucian','Marta','Nabil','Opal','Petra','Quentin','Rosalind','Silas','Talia','Ursula','Viktor','Willa','Xavier','Yvette','Zelda','Ansel','Beatriz','Cedric','Daphne','Emil','Flora','Gideon','Helena','Ismael','Juniper']
NAMES_ZH=['安达','博里','塞莉','达里','艾丝','费利','格蕾','赫克','伊奈','朱尔','卡蒂','卢西','玛塔','纳比','欧珀','佩特','昆廷','罗莎','西拉','塔莉','乌苏','维克','薇拉','泽维','伊薇','泽尔','安塞','贝娅','塞德','达芙','埃米','芙洛','吉迪','海莲','伊斯','朱妮']
EVENTS_EN=['rinsing','sifting','baking','freezing','cutting','folding','sorting','binding','drilling','packing','casting','drying','blending','steaming','carving','shaping','grinding','screening','pressing','boiling','slicing','coating','peeling','wrapping','extruding','quenching','annealing','etching','tanning','spinning','weaving','braiding','roasting','curing','crushing','kneading']
EVENTS_ZH=['冲洗','筛分','烘焙','冷冻','切割','折叠','分拣','装订','钻孔','打包','浇铸','晾干','混合','蒸煮','雕刻','塑形','磨削','过筛','压制','煮沸','切片','涂层','去皮','包裹','挤出','淬火','退火','蚀刻','鞣制','纺纱','织布','编织','烘烤','固化','破碎','揉制']
FRUITS_EN=['apple','pear','banana','peach','plum','mango','apricot','cherry','lemon','orange','fig','grape','melon','papaya','guava','lychee','persimmon','nectarine','tangerine','pomegranate','kiwi','pineapple','strawberry','blueberry','raspberry','blackberry','cranberry','coconut','date','olive','quince','kumquat','pomelo','durian','jackfruit','passionfruit']
FRUITS_ZH=['苹果','梨','香蕉','桃','李子','芒果','杏','樱桃','柠檬','橙子','无花果','葡萄','甜瓜','木瓜','番石榴','荔枝','柿子','油桃','橘子','石榴','猕猴桃','菠萝','草莓','蓝莓','覆盆子','黑莓','蔓越莓','椰子','枣','橄榄','榅桲','金桔','柚子','榴莲','菠萝蜜','百香果']

def item(family,lang,i,form,v):
    en=lang=='en'; names=NAMES_EN if en else NAMES_ZH; events=EVENTS_EN if en else EVENTS_ZH
    a,b=names[i],names[(i+13)%36]; ea,eb=events[i],events[(i+11)%36]; anchor=a
    prefix=(f'In record {i+1}, ' if en else f'第{i+1}份记录中，') if form else ''
    if family=='chronology':
        relation=('before','after') if en else ('早于','晚于')
        rel=relation[v]; anchor=rel
        if en:
            body=(f'{ea} happened {rel} {eb}.' if not form else f'{ea} was {"earlier" if v==0 else "later"} than {eb}.')
            if form:anchor='earlier' if v==0 else 'later'
            question=' Which event happened first? Answer with its name only.'
        else:
            body=f'{ea}{rel}{eb}。' if not form else f'{ea}发生在{eb}{"之前" if v==0 else "之后"}。'
            if form:anchor='之前' if v==0 else '之后'
            question='哪个事件先发生？只回答事件名。'
        target,alt=(ea,eb) if v==0 else (eb,ea)
    elif family=='word_sense':
        anchor='Apple' if en else '苹果'
        if en:
            body=(f'{a} ate an Apple with lunch.' if v==0 else f'{a} bought a laptop made by Apple.') if not form else (f'{a} sliced the Apple for a fruit salad.' if v==0 else f'{a} installed software released by Apple.')
            question=' In this sentence, does Apple refer to fruit or a company? Answer fruit or company only.'
            target,alt=('fruit','company') if v==0 else ('company','fruit')
        else:
            body=(f'{a}午餐时吃了一个苹果。' if v==0 else f'{a}购买了苹果制造的笔记本电脑。') if not form else (f'{a}切开苹果做水果沙拉。' if v==0 else f'{a}安装了苹果发布的软件。')
            question='这里的苹果指水果还是公司？只回答水果或公司。';target,alt=('水果','公司') if v==0 else ('公司','水果')
    elif family=='taxonomy':
        anchor=(FRUITS_EN if en else FRUITS_ZH)[i]
        if en:
            body=f'A {anchor} is a fruit. All fruits are foods. All foods are physical objects.'
            question=(f' What is the immediate category explicitly stated for {anchor}?' if v==0 else f' What is the broadest category reached by the complete stated chain for {anchor}?')
            if form:question=question.replace('category','class')
            question+=' Copy only that category phrase.';target,alt=('fruit','physical objects') if v==0 else ('physical objects','fruit')
        else:
            body=f'{anchor}属于水果，所有水果都是食物，所有食物都是物体。'
            question=(f'{anchor}在所述链中直接属于哪一类？' if v==0 else f'沿所述完整分类链，{anchor}最终属于最广的哪一类？')
            if form:question=question.replace('哪一类','什么类别')
            question+='只回答类别名称。';target,alt=('水果','物体') if v==0 else ('物体','水果')
    elif family=='syntax_role':
        agent,patient=(a,b) if v==0 else (b,a)
        if en:
            body=f'{agent} congratulated {patient}.' if not form else f'{patient} was congratulated by {agent}.'
            question=' Who did the congratulating? Give only the name.'
        else:
            body=f'{agent}祝贺了{patient}。' if not form else f'{patient}被{agent}祝贺了。'
            question='谁实施了祝贺？只回答姓名。'
        target,alt=agent,patient
    elif family=='negation':
        target,alt=(a,b) if v==0 else (b,a);anchor='not' if en else '没有'
        if en:
            body=f'{target} did not sign the form; {alt} did sign it.' if not form else f'{alt} signed the form, but {target} did not.'
            question=' Who left the form unsigned? Give only the name.'
        else:
            body=f'{target}没有签署表格；{alt}签署了。' if not form else f'{alt}签署了表格，但{target}没有签。'
            question='谁没有签署表格？只回答姓名。'
    elif family=='reference':
        target,alt=(a,b) if v==0 else (b,a);anchor='it' if en else '它'
        if en:
            body=f'{a} carried a notebook and {b} carried a lantern. {target} set the '+('notebook' if target==a else 'lantern')+' on a bench. Later, '+target+' picked it up.'
            question=' Who picked the object up? Give only the name.' if not form else ' Name the person who retrieved it. Give only the name.'
        else:
            body=f'{a}拿着笔记本，{b}拿着灯笼。{target}将'+('笔记本' if target==a else '灯笼')+f'放在长椅上。随后{target}又把它拿了起来。'
            question='谁拿起了物品？只回答姓名。' if not form else '谁把它取了回来？只回答姓名。'
    elif family=='punctuation':
        anchor='?' if v==0 and en else '？' if v==0 else '! ' if en else '！'
        if en:
            body=f'The line is: "{a} has arrived'+('?' if v==0 else '!')+'"'
            question=' Classify its terminal punctuation as question or exclamation. Give only the category.' if not form else ' Which mark ends the quoted line? Answer question or exclamation only.'
            target,alt=('question','exclamation') if v==0 else ('exclamation','question');anchor='?' if v==0 else '!'
        else:
            body=f'句子为：“{a}已经到达'+('？' if v==0 else '！')+'”'
            question='句末是问号还是叹号？只回答符号名称。' if not form else '判断引号内末尾符号，只回答问号或叹号。'
            target,alt=('问号','叹号') if v==0 else ('叹号','问号')
    elif family=='long_reorder':
        if en:
            sentences=[f'At 08:00, {a} unlocked the cabinet and carefully counted the blue folders.',f'At 10:00, {b} inspected the notebook beside the wooden window.',f'At 12:00, {a} placed the signed receipt inside the green envelope.']
            body=' '.join(sentences[j] for j in ((2,0,1) if not form else (1,2,0)))
            question=f' Reorder these three sentences from {"earliest to latest" if v==0 else "latest to earliest"}. Copy every sentence exactly once, changing only their order. Output one sentence per line, with no numbering.'
        else:
            sentences=[f'08:00，{a}打开柜子，仔细清点了里面的蓝色文件夹。',f'10:00，{b}检查了木窗旁边的笔记本。',f'12:00，{a}把签字收据放进绿色信封。']
            body=''.join(sentences[j] for j in ((2,0,1) if not form else (1,2,0)))
            question=f'将这三句话按时间{"由早到晚" if v==0 else "由晚到早"}重排。每句原文完整复制一次，只改变顺序，不添加编号，每行一句。'
        target='\n'.join(sentences if v==0 else sentences[::-1]);alt='\n'.join(sentences[::-1] if v==0 else sentences);anchor='08:00'
    else: raise KeyError(family)
    text=prefix+body+question
    assert anchor in text,(family,anchor,text)
    return dict(text=text,target=target,alternate=alt,anchor=anchor)

def build(tokenizer,start=0,stop=12,forms=(0,1)):
    rows=[]
    for family in FAMILIES:
        for lang in ('en','zh'):
            for i in range(start,stop):
                for form in forms:
                    for v in (0,1):
                        r=item(family,lang,i,form,v)
                        prompt=tokenizer.apply_chat_template([{'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
                        begin=prompt.index(r['text'])+r['text'].index(r['anchor']);end=begin+len(r['anchor'])
                        positions=[j for j,(s,e) in enumerate(encoded['offset_mapping']) if e>begin and s<end]
                        if not positions: raise ValueError('unmapped anchor')
                        r.update(case_id=f'{family}/{lang}/{i}/f{form}/v{v}',family=family,language=lang,index=i,form=form,variant=v,
                            base_unit=f'{family}/{lang}/{i}',split='discovery' if i<6 else 'heldout' if i<12 else 'expanded',
                            prompt=prompt,prompt_ids=encoded['input_ids'],anchor_positions=positions,token_strings=tokenizer.convert_ids_to_tokens(encoded['input_ids']))
                        rows.append(r)
    return rows

def normalize(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]','',s.casefold())

def evaluate(row,text):
    # Preserve punctuation and content exactly for sentence reordering (only whitespace is normalized).
    clean=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    if row['family']=='long_reorder':
        norm=lambda s:re.sub(r'\s+','',s)
        exact=norm(clean)==norm(row['target'])
        parts=row['target'].splitlines(); content=all(clean.count(p)==1 for p in parts)
        return {'strict_correct':exact,'answer_correct':exact,'content_preserved':content,'empty':not clean}
    first=next((s for s in clean.splitlines() if s.strip()),'')
    ans=normalize(first);t=normalize(row['target']);a=normalize(row['alternate'])
    return {'strict_correct':ans==t,'answer_correct':t in ans and a not in ans,'content_preserved':None,'empty':not clean}

if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok)
    assert len(rows)==768
    assert len({r['prompt'] for r in rows})==768
    save(RESULT/'phase2621_native_language_behavior/material/cases.json',rows)
    print('768 unique prompts, 192 base items; structured templates, not 768 independent events.')
