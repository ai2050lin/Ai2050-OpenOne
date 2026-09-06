"""Prospective native-source material builder, not a completed/frozen campaign.

Reuses the audited fact templates; changes entity identities and adds explicit
name/cloze output functions on the exact same bodies as truth/mapped controls.
The four-function panel is 512 cells, 256 already in the 8192 truth grid plus
256 new prompts. Never report it as 512 additional independent observations.
"""
from phase2620_native_coordinate_contract import RESULT,read
from phase2677_source_role_regions import character_regions,token_regions

EN='Aster Corwin Mavis Evander Nerys Theron Elowen Roderic'.split()
ZH='昝晴 郗原 阮溪 欧岳 解蓉 亓岚 申澈 荀叶'.split()


def encode(tok,row):
    prefix=tok.apply_chat_template([{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt=prefix+row.get('prefill','');e=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
    start=prompt.index(row['body']);end=start+len(row['body'])
    body_end=max(i for i,(s,t) in enumerate(e['offset_mapping']) if t>start and s<end)
    row={**row,'prompt':prompt,'prompt_ids':e['input_ids'],'token_strings':tok.convert_ids_to_tokens(e['input_ids']),
         'body_end_token':body_end,'task_end_token':len(e['input_ids'])-1,'eos_token_id':tok.eos_token_id,
         'canonical_answer_ids':[tok.encode(w,add_special_tokens=False) for w in row['common_readout_words']]}
    chars=character_regions(row);tokens=token_regions(e['offset_mapping'],chars)
    row.update(character_regions=chars,token_regions=tokens)
    return row


def statement_of(row):
    if row['language']=='en':
        line=row['text'][len(row['body']):].splitlines()[1]
        assert line.startswith('Statement: ') and '. Is this statement true?' in line
        return line[len('Statement: '):].split('. Is this statement true?')[0]
    line=row['text'][len(row['body']):].splitlines()[1]
    assert line.startswith('陈述：') and '。判断陈述是否正确。' in line
    return line[len('陈述：'):].split('。判断陈述是否正确。')[0]


def build(tok):
    previous=read(RESULT/'phase2670_native_mlp_contract/material/cases.json');rows=[];extra=[]
    discard={'prompt','prompt_ids','token_strings','entity_spans','common_readout_ids','canonical_answer_ids','eos_token_id','body_end_token'}
    for old in previous:
        r={k:v for k,v in old.items() if k not in discard};names=EN if r['language']=='en' else ZH
        a,b=names[2*r['unit']:2*r['unit']+2]
        for key in ('body','text'):
            r[key]=r[key].replace(old['entity_a'],'{{NEW_A}}').replace(old['entity_b'],'{{NEW_B}}').replace('{{NEW_A}}',a).replace('{{NEW_B}}',b)
        r.update(entity_a=a,entity_b=b,case_id='source/'+r['case_id'],output_function='truth' if r['mapping']==0 else 'mapped_truth',prefill='',fp_selected=False)
        source=(r['unit'] in (2,3) and all(r[k]==0 for k in ('form','mention_order','probe_index','polarity')))
        r['source_selected']=source
        # Four-function panel: both truth maps, and the name/cloze additions.
        r['published']=source and (r['unit'],r['content_instance'],r['target_index'])==(2,0,0)
        r['case_index']=len(rows);rows.append(encode(tok,r))
        if source and r['mapping']==0:
            statement=statement_of(r);marker='<PERSON>' if r['language']=='en' else '〈人名〉'
            assert statement.count(a)==1
            template=statement.replace(a,marker);en=r['language']=='en'
            for mode in ('name','cloze'):
                if en:
                    text=r['body']+'\nQuestion: Which person makes the following statement true? "'+template+'"'
                    text+='\nOutput rule: Replace <PERSON> with the correct person\'s name. Output only that name.'
                    prefill='' if mode=='name' else 'The person making the statement "'+template+'" true is '
                else:
                    text=r['body']+'\n问题：哪一个人使下面的陈述成立？“'+template+'”'
                    text+='\n输出规则：将〈人名〉替换成正确的人名，只输出这个姓名。'
                    prefill='' if mode=='name' else '使陈述“'+template+'”成立的人是'
                if mode=='cloze':
                    text+='\nCompletion instruction: Continue the supplied factual sentence with only the name.' if en else '\n续写要求：接着给定的事实句，只填写人名。'
                nr={**r,'case_id':r['case_id']+'/'+mode,'text':text,'prefill':prefill,'output_function':mode,
                    'expected_yes':None,'statement_truth':None,'question_affirmative':None,
                    'probe_index':None,'polarity':None,'mapping':None,
                    'target':(a,b)[r['target_index']],'alternate':(b,a)[r['target_index']],
                    'common_readout_words':[a,b],'short_answer_words':[a,b]}
                extra.append(nr)
    for r in extra:r['case_index']=len(rows);rows.append(encode(tok,r))
    return rows


def evaluate(row,text):
    s=text.strip();clean=s.strip(' .。').casefold()
    if row['output_function'] in ('truth','mapped_truth') and row['language']=='zh':
        clean={'是的':'是','不是':'否','不是的':'否'}.get(clean,clean)
    return {'strict_correct':s.casefold()==row['target'].casefold(),'content_correct':clean==row['target'].casefold(),'empty':not s}
