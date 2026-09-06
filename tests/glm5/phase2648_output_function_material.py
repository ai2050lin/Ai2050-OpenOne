"""Eight matched operations crossed with natural output functions and query entity."""
import re
from phase2641_matched_operation_material import item,FAMILIES,SENSE_EN,SENSE_ZH
from phase2620_native_coordinate_contract import *

EN_NAMES='Adam Brian Clara David Alice Henry Diana George Fiona James Laura Oscar Julia Peter Maria Victor Olivia Simon Sarah Thomas Anna Felix Isabel Kevin Rachel Steven Sophie Arthur Teresa Daniel Monica Edward'.split()
ZH_NAMES='高悦 马宁 罗欣 徐峰 孔雯 韦明 田蕾 萧川 夏琳 余轩 章琪 任涛 戴晴 汤博 赖颖 陶森 文佳 袁凯 钟萱 蒲宇 傅洁 熊泽 易蓉 施航 武芸 严瑞 关瑶 康杰 欧婷 石朗 池曼 路诚'.split()
INITIAL_UNITS=(0,1,2,3);CONFIRM_UNITS=(12,13,14,15)
MODES=('name','cloze','truth_a','truth_b')

def parts(fam,lang,unit,form,v,order):
    old=item(fam,lang,unit,form,v,order);a,b=(EN_NAMES if lang=='en' else ZH_NAMES)[2*unit:2*unit+2]
    text=old['text'].replace(old['entity_a'],'{{ENTITY_A}}').replace(old['entity_b'],'{{ENTITY_B}}').replace('{{ENTITY_A}}',a).replace('{{ENTITY_B}}',b)
    text=text.removesuffix(" Answer with the person's name only." if lang=='en' else '只回答人名。')
    match=re.search(r' (?:Who|Which person|Whose|In whose)\b' if lang=='en' else r'较早到达的是谁|谁',text)
    assert match,(fam,lang,text)
    return a,b,text[:match.start()],text[match.start():]

def relation(fam,lang,unit,form):
    if lang=='en':
        descriptions={'chronology':'arrived first','taxonomy':'had fruit','syntax_role':'did the congratulating' if form==0 else 'gave the praise',
            'negation':'left the document unsigned','comparison':'had the larger count','reference':'folded the map','punctuation':'used a question mark'}
        if fam=='word_sense':
            word,_,_,meaning=SENSE_EN[unit%8]
            return lambda name:f'In the situation involving {name}, "{word}" refers to {meaning}',f'The person whose situation uses "{word}" to mean {meaning} is '
        desc=descriptions[fam];return lambda name:f'{name} {desc}',f'The person who {desc} was '
    descriptions={'chronology':'先到达','taxonomy':'带有水果','syntax_role':'实施祝贺' if form==0 else '表达赞扬',
        'negation':'没有签署文件','comparison':'拥有较多数量','reference':'折起地图','punctuation':'使用问号'}
    if fam=='word_sense':
        word,_,_,meaning=SENSE_ZH[unit%4]
        return lambda name:f'{name}的情境中的“{word}”指{meaning}',f'情境中的“{word}”指{meaning}的人是'
    desc=descriptions[fam];return lambda name:f'{desc}的人是{name}',f'{desc}的人是'

def build(tok):
    assert len(EN_NAMES)==len(ZH_NAMES)==32 and len(set(EN_NAMES))==len(set(ZH_NAMES))==32
    rows=[]
    for fam in FAMILIES:
        for lang in ('en','zh'):
            for unit in range(16):
                for form in (0,1):
                    for v in (0,1):
                        for order in (0,1):
                            a,b,body,question=parts(fam,lang,unit,form,v,order);statement,cloze=relation(fam,lang,unit,form)
                            for mode in MODES:
                                en=lang=='en';prefill='';probe=None;truth=None
                                if mode=='name':
                                    usertext=body+question+(" Answer with the person's name only." if en else '只回答人名。');target=(a,b)[v];alternate=(b,a)[v]
                                    common=[a,b]
                                elif mode=='cloze':
                                    usertext=body+('\nContinue the following factual sentence with the correct person\'s name.' if en else '\n请接着下面的事实句写出正确的人名。')
                                    prefill=cloze;target=(a,b)[v];alternate=(b,a)[v];common=[a,b]
                                else:
                                    probe=0 if mode=='truth_a' else 1;truth=v==probe;name=(a,b)[probe]
                                    usertext=body+('\nStatement: '+statement(name)+'. Is this statement true? Answer only Yes or No.' if en else '\n陈述：'+statement(name)+'。这句话是否正确？只回答是或否。')
                                    common=['Yes','No'] if en else ['是','否'];target=common[0 if truth else 1];alternate=common[1 if truth else 0]
                                prompt=tok.apply_chat_template([{'role':'user','content':usertext}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill
                                encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);startbody=prompt.index(body);spans={}
                                for key,name in [('a',a),('b',b)]:
                                    start=startbody+body.index(name);end=start+len(name)
                                    spans[key]=[i for i,(s,e) in enumerate(encoded['offset_mapping']) if e>start and s<end];assert spans[key]
                                cid=[tok.encode(word,add_special_tokens=False)[0] for word in common]
                                rows.append({'case_index':len(rows),'case_id':f'{fam}/{lang}/u{unit}/f{form}/v{v}/o{order}/{mode}',
                                    'family':fam,'language':lang,'unit':unit,'form':form,'target_index':v,'mention_order':order,'mode':mode,
                                    'output_function':'truth' if mode.startswith('truth') else mode,'probe_index':probe,'truth':truth,
                                    'body':body,'text':usertext,'prefill':prefill,'entity_a':a,'entity_b':b,'target':target,'alternate':alternate,
                                    'prompt':prompt,'prompt_ids':encoded['input_ids'],'token_strings':tok.convert_ids_to_tokens(encoded['input_ids']),
                                    'entity_spans':spans,'common_readout_words':common,'common_readout_ids':cid,'common_readout_available':cid[0]!=cid[1],
                                    'field_set':'initial' if unit in INITIAL_UNITS else 'confirmation' if unit in CONFIRM_UNITS else 'behavior_only',
                                    'published':(unit,form,v,order)==(0,0,0,0),
                                    'response_orientation':-1 if mode=='truth_b' else 1})
    return rows

def evaluate(row,text):
    text=text.strip();strict=text.casefold()==row['target'].casefold();content=text.strip(' .。').casefold()
    if row['output_function']=='truth' and row['language']=='zh':
        content={'是的':'是','不是':'否','不是的':'否'}.get(content,content)
    return {'strict_correct':strict,'content_correct':content==row['target'].casefold(),'empty':not text,
        'boundary':'Cloze is real greedy continuation of an externally supplied assistant prefix, not autonomously planned full-sentence generation.' if row['mode']=='cloze' else 'Complete answer exact after declared punctuation/yes-no normalization.'}
