"""Final bounded discovery calibration: register examples, not forced outputs.

Example topics differ from inventory. Paired examples express the same meaning.
All user input outside the equal-token example is identical within a pair.
"""
import phase2777_materials_v3 as V3

SEED=27772030
INSTRUCTION=V3.INSTRUCTION;JUDGE_DESCRIPTIONS=V3.JUDGE_DESCRIPTIONS
fact=V3.fact;content=V3.content;rubric=V3.rubric;references=V3.references
chinese_number=V3.chinese_number;COLLOQ=V3.COLLOQ;ACADEMIC=V3.ACADEMIC
EXAMPLES=[
  ([ '出行前应核实时间和目的地，再准备所需物品。若天气发生变化，应重新确认安排，不宜将尚未核实的信息视为定论。',
     '在安排出行时，首先需要核实出发时间与目的地，随后准备必要物品。如天气出现变化，应再次确认相关安排。对于未经核实的信息，不应直接认定其准确。',
     '关于出行安排，应当先明确时间及目的地，再依据实际需要准备相应物品。天气条件发生变化时，原有安排需要重新核验；尚未得到核实的事项不能被视为已经确定。'],
   ['你看，咱们出门前先把时间和去哪儿对一对吧，再把要带的东西收拾好。要是天气变了，就再问清楚，没核实的可别当真嘛。',
    '咱们先说好几点出门、去哪儿，再收拾要带的东西吧。你想，天气要是变了，安排还得再问问，没弄清楚的别先当成真的呀。',
    '你看，出门这事，咱们先把时间和目的地弄清楚，再准备东西就行。天气要是有变化，就再确认一下呗，没核实的话可不能先信了呀。']),
  (['阅读前应查看目录，再选定章节并记录疑问。对未能理解的部分，应在后续阅读中重新核查，不宜直接略过。',
    '开展阅读时，应当先浏览目录，随后选定一个章节展开阅读，并记录尚未解决的问题。对于暂时不能理解的内容，需要在后续过程中再次查阅。',
    '阅读活动应从了解目录开始，再选择具体章节，并将存在疑问之处记录下来。尚未理解的部分不应直接略去，而应保留到后续阅读时继续核查。'],
   ['你先翻翻目录，咱们再挑一章看吧，哪里没懂就记下来。别急着跳过去嘛，后面再回来对一对就行呀。',
    '你看，咱们先看目录，再找一章读读呗，碰上不懂的就记着。可别一跳了之呀，回头再翻翻，慢慢弄明白就行。',
    '咱们先把目录翻一遍吧，再挑一章往下看。你要是有疑问，就顺手记下来嘛，没懂的别扔一边，后面还得再看看呀。']),
  (['安排聚会应先确认时间和地点，再告知相关人员。若计划发生变更，应及时更新信息；缺少确认依据的事项不应视为已确定。',
    '聚会安排首先需要明确时间及地点，然后通知参与人员。当原定计划发生变化时，应及时传达更新；没有确认的信息不能被视为已经落实。',
    '对于聚会活动，应当先核实时间与场地，再将相关安排告知其他人员。后续计划若有变更，需要更新已有信息，不应将未经确认的事项当作既定安排。'],
   ['你看，咱们聚会先把时间和地点对好吧，再跟大家说。要是改了，就赶紧更新一下嘛，没问清的别当成定好了呀。',
    '咱们先问清楚哪天、在哪儿聚，再告诉大家呗。你想，计划要是变了，可得再说一声，没确认的就别先当真嘛。',
    '你先把聚会时间和地方弄明白吧，咱们再通知大家。改了安排就得及时说呀，还没确认的话，别先当作已经定了呗。']),
  (['检查设备时应先核实连接状态，再查看已有记录。对于原因尚未明确的问题，应进一步查证，不宜在证据不足时作出确定判断。',
    '设备检查应从连接状态的核实开始，再结合已有记录进行判断。若问题原因仍不明确，需要继续查证，避免在缺少证据时直接认定结论。',
    '对设备进行检查时，首先应确认各项连接，再查阅已有的使用记录。对于原因无法确认的情况，后续仍需补充核验；证据不足不能支持确定性判断。'],
   ['你看，咱们先看看设备接好了没有，再翻翻原来的记录吧。原因没弄明白就再查查嘛，可别没证据就把话说死呀。',
    '咱们先把设备连接对一对呗，再看看以前怎么记的。你要是还不知道哪里有问题，就继续查嘛，没依据的别急着下定论呀。',
    '你先看设备有没有接好吧，咱们再找原来的记录。问题还没弄清就接着查呗，没有证据，可别一口咬定就是那个原因呀。'])]
TAIL='请只模仿示例的表达习惯来转述上方已核验的记录，不照抄例句，不讨论示例内容。'


class Builder(V3.Builder):
    def pair(self,i,frame,length,split):
        f=fact(i);body=V3.V2.source(f,frame)+'\n\n'+INSTRUCTION
        maps={}
        extensions=['','表达保持完整连贯。','请把事情完整交代清楚。','请在保持事实的前提下完整表达。']
        for reg in ['acad','colloq']:
            d={}
            examples=EXAMPLES[frame][0 if reg=='acad' else 1]
            for example in examples:
                for extension in extensions:
                    exemplar=example+extension
                    text=body+'\n\n下面是另一件事的表达示例：\n'+exemplar+'\n'+TAIL
                    d.setdefault(self.count(self.wrap(text)),(text,exemplar))
            maps[reg]=d
        for n in sorted(set(maps['acad'])&set(maps['colloq'])):
            prefixes=self.neutral(i%2)
            for k in range(length-n-4,length-n+5):
                if k not in prefixes or prefixes[k].count('。')<4:continue
                prefix=prefixes[k];texts=[prefix+'\n'+maps[r][n][0] for r in ['acad','colloq']]
                prompts=[self.wrap(t) for t in texts];ids=[self.tok.encode(t,add_special_tokens=False) for t in prompts]
                if not all(len(t)==length for t in ids):continue
                return [{'id':f'nat5-{i:04d}-F{frame}-L{length}-{reg}','family_id':f'nat5-{i:04d}',
                    'record':i,'frame':frame,'length':length,'split':split,'register':reg,'text':text,'prompt':prompt,
                    'token_ids':tokens,'tokens':len(tokens),'fact':f,'matching':{'active_token_exact':True,
                        'common_neutral_prefix':prefix,'common_fact_source':V3.V2.source(f,frame),
                        'style_example':maps[reg][n][1],'register_request':maps[reg][n][1],
                        'scope':'few-shot register transfer, exemplar topic independent from inventory',
                        'no_truncation':True,'no_masked_padding':True}}
                    for reg,text,prompt,tokens in zip(['acad','colloq'],texts,prompts,ids)]
        raise ValueError(f'No equal example-token pair for {i},F{frame},L{length}')
