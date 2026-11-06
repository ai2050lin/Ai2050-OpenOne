"""Pilot-informed NEW version: neutral facts x explicit paraphrased register request.

All factual source text is identical within a style pair. The verified answer
is supplied because this is a style-preserving restatement, not arithmetic.
Old failed pilot and its builder remain byte-for-byte available.
"""
import re
import phase2777_materials as V1

SEED=27772027
INSTRUCTION=('请输出三到五句连贯的话，正文须明确写出物品名称和已核验的结余数量。'
 '任务是转述，不要重新计算，不要编造新事实，不要列提纲，也不要加风格标签。')
COLLOQ=V1.COLLOQ;ACADEMIC=V1.ACADEMIC
rubric=V1.rubric;references=V1.references;chinese_number=V1.chinese_number


def fact(i):return dict(V1.fact(i),verified_result_supplied=True)


def requests(frame,reg):
    frames=[
      (['请用正式、客观的书面语说明。','请采用正式书面语，客观说明这些事实。','请像撰写工作报告一样，严谨、客观地解释。'],
       ['请用自然、随和的日常口语说明。','请像跟熟悉的同事聊天一样，把事情讲清楚。','请用轻松的聊天口吻，别把这段话说成报告。']),
      (['这段文字面向专业读者，请使用规范的书面表达。','假设材料要存入工作档案，请用正式书面语进行说明。','请以报告正文的叙述方式，客观陈述已核验的事实。'],
       ['这段话是说给熟人听的，请自然一点，像平时聊天那样说。','假设你正跟同事当面聊这件事，请用随和的日常说法。','别照着报告念，请像聊天一样，把这件事说得明白好懂。']),
      (['请将记录转述为正式的文字说明，使用严谨的书面语言。','如果把它写进正式材料，应使用客观、规范的书面语。','请按专业说明文字的语体，完整陈述记录中的事实。'],
       ['请把记录讲给身边的同事听，用亲切自然的口语来解释。','如果当面把这件事说给朋友听，请用轻松的聊天语气。','请按平时跟熟人说话的方式，把这些已确定的事讲出来。']),
      (['设想这是工作报告中的一段，请采用正式书面语体。','请以客观论述的书面形式，说明记录中的结果与依据。','这段话将进入正式文件，表达应严谨，并采用书面语言。'],
       ['设想你正坐在同事旁边聊天，请用平常说话的口吻。','请像随口向熟人解释那样说，语气自然，别写成报告。','这段话是当面跟人聊的，请用轻松、亲近的日常口语。']),
    ]
    suffix=['请写成完整段落。','表达完整清楚即可。','请用完整句子表达清楚。','需要形成一段连贯的话。',
            '请注意把事情完整地交代清楚。','前后衔接应当清楚，不要省略必要内容。']
    return [r+s for r in frames[frame][0 if reg=='acad' else 1] for s in suffix]


def source(f,frame):
    name,item,n,a,s,q,place=[f[k] for k in ['actor','item','initial','added','removed','quantity','place']]
    frames=[
        f'本次记录涉及{place}的{item}，负责人是{name}。起初有{n}份，随后接收{a}份，并向另一组移交{s}份。已核验的最终结余是{q}份{item}。期间没有损耗和其他调拨，所有数量采用同一单位。',
        f'{name}负责{place}的{item}核对。最终结余已经核验，确认为{q}份。相关记录的起点是{n}份，后续接收{a}份，再向另一组移交{s}份。除这两项变化之外，没有损耗，也没有其他调拨。',
        f'由{name}负责核验的，是{place}的{item}记录。所接收的{a}份计入增加项，移交另一组的{s}份计入减少项，初始库存为{n}份。核验人员已经确认：最终剩余{q}份{item}，期间不存在额外调拨或损耗。',
        f'对于{place}的{item}，已确认的结余为{q}份，核验负责人为{name}。这一结果对应的记录包括初始{n}份、后来接收{a}份，以及对另一组移交{s}份。所给数字使用相同计数单位，没有其他调拨与损耗。',
    ]
    return frames[frame]


class Builder(V1.Builder):
    def pair(self,i,frame,length,split):
        f=fact(i);body=source(f,frame)
        # Pair semantic register requests with exactly equal token counts in
        # their actual common suffix, not stand-alone text approximations.
        maps={}
        for reg in ['acad','colloq']:
            d={}
            for r in requests(frame,reg):
                text=body+'\n\n'+r+'\n'+INSTRUCTION
                n=self.count(self.wrap(text))
                d.setdefault(n,(text,r))
            maps[reg]=d
        common=sorted(set(maps['acad'])&set(maps['colloq']))
        for n in common:
            prefixes=self.neutral(i%2)
            for k in range(length-n-4,length-n+5):
                if k not in prefixes:continue
                prefix=prefixes[k]
                if prefix.count('。')<4:continue
                texts=[prefix+'\n'+maps[reg][n][0] for reg in ['acad','colloq']]
                prompts=[self.wrap(t) for t in texts]
                ids=[self.tok.encode(p,add_special_tokens=False) for p in prompts]
                if not all(len(x)==length for x in ids):continue
                assert ids[0][-1]==ids[1][-1]
                return [{'id':f'nat2-{i:04d}-F{frame}-L{length}-{reg}','family_id':f'nat2-{i:04d}',
                    'record':i,'frame':frame,'length':length,'split':split,'register':reg,
                    'text':text,'prompt':prompt,'token_ids':toks,'tokens':len(toks),'fact':f,
                    'matching':{'active_token_exact':True,'common_neutral_prefix':prefix,'common_fact_source':body,
                        'register_request':maps[reg][n][1],'core_tokens':n,'no_truncation':True,'no_masked_padding':True,
                        'study':'explicit register instructions; source facts identical, verified result given'}}
                    for reg,text,prompt,toks in zip(['acad','colloq'],texts,prompts,ids)]
        raise ValueError(f'No exact matched instruction pair for F{frame},L{length},record{i}; request token counts={list(maps)}')


def content(text,f):
    # Generic "每一份" is not an extra inventory quantity.
    normalized=re.sub(r'(每|任何|任意|各)一(?=份|本|册|张)',r'\1',text)
    aliases={'记录册':['记录册','登记册'],'资料袋':['资料袋','文件袋']}.get(f['item'],[f['item']])
    result=V1.content(normalized,f)
    result['item_ok']=any(a in text for a in aliases)
    result['pass']=result['item_ok'] and result['quantity_ok'] and not result['unexpected_quantities']
    result.update(allowed_item_aliases=aliases,scope='provided verified result restatement; generic each-one expressions excluded; not a reasoning benchmark')
    return result
