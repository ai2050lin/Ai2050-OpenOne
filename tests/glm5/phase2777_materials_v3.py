"""Third discovery pilot: explicit audience interaction, style request LAST.

The scorer distinguishes factual topic from register; neither literal example
markers nor a candidate answer pool are supplied to the generator.
"""
import re
import phase2777_materials_v2 as V2

SEED=27772028
INSTRUCTION=V2.INSTRUCTION
fact=V2.fact;content=V2.content;chinese_number=V2.chinese_number
COLLOQ=V2.COLLOQ;ACADEMIC=V2.ACADEMIC
JUDGE_DESCRIPTIONS={
 'acad':'正式报告或学术说明中的书面语体，不与读者当面交谈',
 'colloq':'对熟人当面解释的自然口语，包括完整句子，不要求俚语；陈述事实也可以是口语',
 'unknown':'混合、无法辨认或文字不成段'}


def requests(frame,reg):
    pairs=[
      (['这段话要写进正式工作报告，请使用客观、规范的书面语，不要与读者聊天。',
        '请以正式报告正文的语体陈述，措辞严谨，避免面对面聊天的口吻。',
        '输出用于正式书面材料，应客观规范，不采用和人交谈的语气。'],
       ['你正在当面给熟悉的同事解释这件事，请直接对他说，语气亲近自然，别像宣读报告。',
        '请像和熟人聊天一样，直接对他说清这件事，可以自然地反问、感叹，别写成报告。',
        '这段话是当面跟朋友聊的，请让对方听着像平时交谈，表达随和，别用报告腔。']),
      (['请将结果写成专业说明段落，采用正式的书面表达，避免直接称呼读者。',
        '这里需要的是书面报告式说明，应使用客观严谨的语言，不与读者对话。',
        '把这段内容作为正式文件正文，使用规范书面语，不要加入聊天式互动。'],
       ['设想对方正坐在你旁边，你一边聊天一边把这件事讲明白，直接跟他说，不要念文稿。',
        '你要对熟悉的人当面说这件事，请让语言像自然聊天，有亲近的交流感，别像写文件。',
        '请面对面向同事解释，直接和对方交谈，可以顺口提醒或提问，别采用公文语气。']),
      (['请采用正式论述的书面语，客观说明经过，不向读者发起对话。',
        '文字将作为工作记录的正式说明，应使用书面语，避免交谈中的称呼和语气。',
        '这是一段专业材料的正文，请用严谨书面语说明，不要像当面聊天。'],
       ['请把这件事说给身边的熟人听，直接跟对方聊，讲得顺口亲切，不要照报告念。',
        '假设你正在和朋友当面说这件事，直接对他说，语气轻松，可以有自然的提醒和感叹。',
        '这回是面对面聊天，请跟同事亲近地交流，把事情说得顺口好懂，避免书面报告腔。']),
      (['请保持专业报告的书面语体，客观交代记录，不采用与读者交谈的形式。',
        '设想文字将收入正式档案，请用规范书面语言，避免聊天式的直接交流。',
        '请写成客观完整的正式说明，使用书面表达，不向对方说话或提问。'],
       ['设想同事正在听你说这件事，请直接和他交谈，用日常口语说清，可以自然地问一句或感叹一下。',
        '请像跟熟人边聊边解释那样说，直接对着对方，语言随和顺口，不用正式文稿的口吻。',
        '你是在当面跟同事聊这件事，请直接交流，让对方听着亲近自然，别写得像专业报告。'])]
    suffix=['把事实交代完整。','请把情况完整说明。','需要保持完整连贯。','前后的意思要连贯。',
            '请完整表达，不省略必要内容。','表达应当完整连贯，保留必要信息。']
    return [r+s for r in pairs[frame][0 if reg=='acad' else 1] for s in suffix]


class Builder(V2.Builder):
    def pair(self,i,frame,length,split):
        f=fact(i);body=V2.source(f,frame);maps={}
        for reg in ['acad','colloq']:
            d={}
            for request in requests(frame,reg):
                text=body+'\n\n'+INSTRUCTION+'\n'+request
                d.setdefault(self.count(self.wrap(text)),(text,request))
            maps[reg]=d
        for n in sorted(set(maps['acad'])&set(maps['colloq'])):
            prefixes=self.neutral(i%2)
            for k in range(length-n-4,length-n+5):
                if k not in prefixes or prefixes[k].count('。')<4:continue
                prefix=prefixes[k];texts=[prefix+'\n'+maps[r][n][0] for r in ['acad','colloq']]
                prompts=[self.wrap(t) for t in texts];ids=[self.tok.encode(t,add_special_tokens=False) for t in prompts]
                if not all(len(x)==length for x in ids):continue
                return [{'id':f'nat3-{i:04d}-F{frame}-L{length}-{reg}','family_id':f'nat3-{i:04d}',
                    'record':i,'frame':frame,'length':length,'split':split,'register':reg,'text':text,'prompt':prompt,
                    'token_ids':tokens,'tokens':len(tokens),'fact':f,'matching':{'active_token_exact':True,
                        'common_neutral_prefix':prefix,'common_fact_source':body,'register_request':maps[reg][n][1],
                        'core_tokens':n,'style_request_position':'last','no_truncation':True,'no_masked_padding':True}}
                    for reg,text,prompt,tokens in zip(['acad','colloq'],texts,prompts,ids)]
        raise ValueError(f'No matched pair for {i},F{frame},L{length}')


def rubric(text):
    t=re.sub(r'“[^”]*”|「[^」]*」|"[^"]*"','',text)
    groups={
        'interaction':bool(re.search(r'(你|您)(看|想|听|说|知道|别|先|得|要|可以)|咱们|跟你|你瞧',t)),
        'particles':bool(re.search(r'(吧|呢|嘛|呀|啦|呗|哦|喔)[，。！？!?]',t)),
        'spoken_actions':bool(re.search(r'弄|搞|拿|凑|算一算|对一对|别|就行|说白了|这回|本来|手里|送来',t)),
        'question':bool(re.search(r'[？?]',t)),
        'discourse':bool(re.search(r'所以嘛|这么说|说到底|其实吧|你瞧|那就|可不是|就是这么',t))}
    a=[s for s in ACADEMIC if s in t]
    label='colloq' if sum(groups.values())>=2 else 'acad' if len(a)>=2 else 'unknown'
    sentences=[s for s in re.split('[。！？!?]',t) if s.strip()]
    return {'label':label,'colloquial_groups':groups,'academic':a,'chars':len(text),'sentences':len(sentences),
        'adequate':len(text)>=45 and len(sentences)>=2}


def references():
    rows=V2.references()
    rows += [{'id':'mild-factual-C','label':'colloq','text':'你看，这回是林宁在管资料，一开始收到的东西都有记录。后来又拿来一些，也分走了一部分，现在手里剩下多少已经核过了，咱们照着那个数说就行。别把换个位置放，也当成多出一份呀。'},
             {'id':'factual-A','label':'acad','text':'本次资料核验由林宁负责，相关记录完整保存。根据既有登记，接收与移交两项数量已分别核对，最终结余与记录一致。上述结果仅对应所给事实，不涉及对工作效率的评价。'}]
    return rows
