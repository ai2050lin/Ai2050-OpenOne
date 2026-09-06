"""Prospective all-family lexical/content confirmation, not completed research.

This builder may be tokenizer-audited while earlier GPU phases run. Its formal
freeze and any model test must wait for review of actual2679/2680 results.
"""
import itertools
from phase2662_symmetric_mapping_contract import compose,FAMILIES
from phase2677_source_role_material import encode,statement_of

EN='Briar Oberon Tamsin Leander Maelle Soren Ione Gareth'.split()
ZH='邬青 任砚 邵栩 蒙霁 计澄 汲兰 莫岚 晏槐'.split()


def substitutions(family,content,language):
    en=language=='en'
    if family=='chronology':return [('arrived',('checked in','left')[content])] if en else [('到达',('报到','入住')[content])]
    if family=='syntax_role':
        if en:return [('congratulating',('mentoring','assisting')[content]),('congratulated',('mentored','assisted')[content]),('praise',('advice','support')[content])]
        return [('祝贺',('指导','协助')[content]),('表达了赞扬',('提出了建议','表达了支持')[content]),('表达赞扬',('提出建议','表达支持')[content]),('赞扬',('建议','支持')[content])]
    if family=='negation':
        return [('unsigned',('unapproved','unsubmitted')[content]),('signed',('approved','submitted')[content]),('sign',('approve','submit')[content])] if en else [('签署',('批准','提交')[content])]
    if family=='reference':return [('map',('blueprint','brochure')[content])] if en else [('地图',('蓝图','宣传册')[content])]
    if family=='punctuation':
        return [('It is ready',('Can we enter','Will it snow')[content]),('Already here',('Where is the parcel','Who has the ticket')[content])] if en else [('已经准备好了',('可以进去吗','明天会下雨吗')[content]),('已经到了',('包裹在哪里','谁拿着票')[content])]
    return []


def name_variant(row,mode):
    r=dict(row);en=r['language']=='en';a,b=r['entity_a'],r['entity_b'];statement=statement_of(r)
    marker='<PERSON>' if en else '〈人名〉';assert statement.count(a)==1
    template=statement.replace(a,marker)
    if en:
        text=r['body']+'\nQuestion: Which person makes the following statement true? "'+template+'"'
        text+='\nOutput rule: Replace <PERSON> with the correct person\'s name. Output only that name.'
        prefill='' if mode=='name' else 'The person making the statement "'+template+'" true is '
        if mode=='cloze':text+='\nCompletion instruction: Continue the supplied factual sentence with only the name.'
    else:
        text=r['body']+'\n问题：哪一个人使下面的陈述成立？“'+template+'”'
        text+='\n输出规则：将〈人名〉替换成正确的人名，只输出这个姓名。'
        prefill='' if mode=='name' else '使陈述“'+template+'”成立的人是'
        if mode=='cloze':text+='\n续写要求：接着给定的事实句，只填写人名。'
    v=r['target_index']
    r.update(output_function=mode,text=text,prefill=prefill,expected_yes=None,statement_truth=None,question_affirmative=None,
        probe_index=None,polarity=None,mapping=None,target=(a,b)[v],alternate=(b,a)[v],common_readout_words=[a,b],short_answer_words=[a,b])
    return r


def build(tok):
    rows=[]
    for fam,lang,e,c,f,o,v in itertools.product(FAMILIES,('en','zh'),range(4),range(2),range(2),range(2),range(2)):
        names=EN if lang=='en' else ZH;a,b=names[2*e:2*e+2]
        for mode in ('truth','mapped_truth','name','cloze'):
            r=compose(fam,lang,c+4,f,v,o,0,0,1 if mode=='mapped_truth' else 0,0 if lang=='en' else 1,1)
            for key in ('body','text'):
                text=r[key].replace(r['entity_a'],'{{A}}').replace(r['entity_b'],'{{B}}').replace('{{A}}',a).replace('{{B}}',b)
                for before,after in substitutions(fam,c,lang):text=text.replace(before,after)
                r[key]=text
            r.update(entity_a=a,entity_b=b,unit=e,content_instance=c,output_function=mode,prefill='',fp_selected=False,
                field_set='fresh_confirmation',source_selected=(e in (2,3) and f==o==0),
                published=(e,c,f,o,v)==(2,0,0,0,0),parameter_published=(e,c,f,o,v)==(2,0,0,0,0) and mode=='truth')
            if mode in ('name','cloze'):r=name_variant(r,mode)
            r.update(case_index=len(rows),case_id=f'fresh_source/{fam}/{lang}/e{e}/c{c}/f{f}/o{o}/v{v}/{mode}')
            rows.append(encode(tok,r))
    return rows
