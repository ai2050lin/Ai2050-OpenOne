"""Controlled bilingual relation cases, explicitly distinct from authentic corpora."""
from rdc_update_common import *

FAMILIES=('attribute_binding','negation_scope','word_sense','long_distance_role','knowledge_chain')
NAMES_EN=('Mira','Noel','Tara','Evan','Lina','Omar','Iris','Theo','Aria','Bryn','Cora','Dara','Ella','Finn','Gail','Hugo',
          'Isla','Jade','Kira','Levi','Milo','Nora','Orin','Pia','Quin','Rhea','Seth','Tess','Uma','Vera','Wren','Zane')
NAMES_ZH=('米拉','诺尔','塔拉','伊文','丽娜','奥玛','艾莉','西奥','安雅','柏林','可拉','达拉','艾拉','芬恩','盖尔','雨果',
          '伊莎','洁德','琪拉','利维','米洛','诺拉','欧林','皮娅','奎因','瑞雅','塞斯','苔丝','乌玛','维拉','温恩','赞恩')
FRUIT_CONTEXTS=(
 ('picked an apple from the orchard','在果园摘下一个苹果'),('peeled an apple for a snack','削了一个苹果当点心'),
 ('sliced an apple on a cutting board','在砧板上切开一个苹果'),('baked apple pieces in a pie','把苹果块烤进馅饼'),
 ('pressed an apple to make juice','用苹果榨汁'),('found seeds inside an apple','在苹果里面发现种子'),
 ('washed an apple before eating it','吃苹果之前先把它洗净'),('placed a ripe apple in a fruit bowl','把成熟的苹果放进果盘'),
 ('tasted a sour apple at a farm','在农场尝了一个酸苹果'),('cooked an apple into a sweet sauce','把苹果煮成甜果酱'),
 ('bought an apple from a produce stall','从果蔬摊买了一个苹果'),('packed a fresh apple in a lunch box','把新鲜苹果装进饭盒'),
 ('removed the core from an apple','挖掉一个苹果的果核'),('saw an apple hanging from a branch','看到枝头挂着一个苹果'),
 ('cut an apple into wedges for guests','把苹果切成小瓣招待客人'),('stored a harvested apple in a cellar','把收获的苹果放进地窖'))
COMPANY_CONTEXTS=(
 ('read an Apple laptop manual','阅读苹果笔记本电脑的说明书'),('installed software published by Apple','安装苹果发布的软件'),
 ('visited an Apple electronics shop','参观苹果电子产品商店'),('watched an Apple phone demonstration','观看苹果手机的演示'),
 ('used an Apple developer tool','使用苹果的开发者工具'),('compared Apple computer processors','比较苹果电脑的处理器'),
 ('read an Apple operating-system guide','阅读苹果操作系统指南'),('tested an Apple smartwatch interface','测试苹果智能手表的界面'),
 ('studied an Apple product design','研究苹果产品的设计'),('contacted Apple technical support','联系苹果技术支持'),
 ('opened an Apple software update panel','打开苹果软件更新面板'),('checked an Apple computer warranty','查看苹果电脑的保修条款'),
 ('attended an Apple developer presentation','参加苹果开发者演示'),('read documentation for an Apple tablet','阅读苹果平板电脑的文档'),
 ('configured an Apple device account','配置苹果设备账户'),('repaired an Apple laptop keyboard','维修苹果笔记本电脑的键盘'))

def main():
    from transformers import AutoTokenizer
    if (BASE/'language_material_frozen.json').exists():return
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);rows=[]
    for family in FAMILIES:
      for case in range(32):
        split='language_train' if case<16 else 'language_validation' if case<24 else 'language_test'
        group=f'language2738/{family}/{case:02d}';a0=case;b0=(a0+1+case//8)%32
        for lang in ('en','zh'):
            a,b=(NAMES_EN if lang=='en' else NAMES_ZH)[a0],(NAMES_EN if lang=='en' else NAMES_ZH)[b0]
            qperson=(a,b)[(case//2)%2];truth=bool(case%2);relations=[]
            if family=='attribute_binding':
                red_owner=a if truth else b;green_owner=b if truth else a;truth=qperson==red_owner
                body=f'{red_owner} owns a basket of red apples. {green_owner} owns a basket of green apples.' if lang=='en' else f'{red_owner}有一篮红苹果，{green_owner}有一篮青苹果。'
                ask=f'Are the apples in {qperson}\'s basket red?' if lang=='en' else f'{qperson}篮子里的苹果是红色的吗？'
                relations=[{'source':red_owner,'relation':'owns_attribute','target':'red apples'},{'source':green_owner,'relation':'owns_attribute','target':'green apples'}]
            elif family=='negation_scope':
                actor=a if truth else b;other=b if truth else a;negative=(case//4)%2==1;truth=(qperson==actor)!=negative
                body=f'{other} did not pack the apples, but {actor} did pack them. Both people packed some books.' if lang=='en' else f'{other}没有装苹果，但{actor}装了苹果。两个人都装了一些书。'
                statement=(f'{qperson} did not pack the apples' if negative else f'{qperson} packed the apples') if lang=='en' else (f'{qperson}没有装苹果' if negative else f'{qperson}装了苹果')
                ask=f'Is the statement "{statement}" true according to the record?' if lang=='en' else f'陈述“{statement}”与上述记录一致吗？'
                relations=[{'source':actor,'relation':'agent','target':'pack apples','polarity':1},{'source':other,'relation':'agent','target':'pack apples','polarity':-1}]
            elif family=='word_sense':
                fruit=truth;ask_fruit=(case//2)%2==0;truth=fruit==ask_fruit
                context=(FRUIT_CONTEXTS if fruit else COMPANY_CONTEXTS)[case//2]
                if lang=='en':
                    body=f'{a} {context[0]}.'
                    ask='Does apple in this passage refer to the edible fruit?' if ask_fruit else 'Does apple in this passage refer to a technology company?'
                else:
                    body=f'{a}{context[1]}。'
                    ask='文中的苹果指可以食用的水果吗？' if ask_fruit else '文中的苹果指科技公司吗？'
                relations=[{'source':'apple/苹果','relation':'contextual_sense','target':'fruit' if fruit else 'company'}]
            elif family=='long_distance_role':
                recipient=a if truth else b;giver=b if truth else a;truth=qperson==recipient
                if lang=='en':
                    body=f'{giver} handed a sealed packet to {recipient}. '+' '.join(f'Unrelated note {j+1}: the room had {j+2} chairs.' for j in range(4+case%3))+f' Later {recipient} handed a cup to {giver}.'
                    ask=f'Was {qperson} the recipient of the sealed packet?'
                else:
                    body=f'{giver}把一个密封包裹递给{recipient}。'+''.join(f'无关记录{j+1}：房间里有{j+2}把椅子。' for j in range(4+case%3))+f'后来{recipient}把一个杯子递给{giver}。'
                    ask=f'{qperson}是密封包裹的接收者吗？'
                relations=[{'source':giver,'relation':'agent','target':'packet-transfer'},{'source':recipient,'relation':'recipient','target':'packet-transfer'},
                  {'source':recipient,'relation':'agent','target':'cup-transfer'},{'source':giver,'relation':'recipient','target':'cup-transfer'}]
            else:
                fruit_names=('apples','pears','bananas','oranges');fruit_zh=('苹果','梨','香蕉','橙子');item=(fruit_names if lang=='en' else fruit_zh)[case%4]
                # No means 'not entailed', never the invalid closed-world inference
                # that the target proposition must therefore be false.
                truth=case%2==0;first_forward=truth or (case//2)%2==0;second_forward=truth or not first_forward
                label_a=item[:-1] if lang=='en' and item.endswith('s') else item
                edge1=(label_a,'fruit') if first_forward else ('fruit',label_a)
                edge2=('fruit','food') if second_forward else ('food','fruit')
                if lang=='en':
                    body=f'This is a fictional catalogue. Use only its stated rules, not outside facts: every {edge1[0]} is a {edge1[1]}; every {edge2[0]} is a {edge2[1]}. Item X{case} is an instance of {label_a}. No means that a conclusion does not necessarily follow.'
                    ask=f'Do the stated rules imply that X{case} is food?'
                else:
                    translate={'fruit':'水果','food':'食物',label_a:label_a}
                    body=f'这是一份虚构分类表，只使用明示规则，不使用外部常识：每个{translate[edge1[0]]}都是{translate[edge1[1]]}；每个{translate[edge2[0]]}都是{translate[edge2[1]]}。对象X{case}属于{label_a}。否表示结论不一定能从规则推出。'
                    ask=f'能仅从明示规则推出X{case}属于食物吗？'
                relations=[{'source':f'X{case}','relation':'instance_of','target':label_a},{'source':edge1[0],'relation':'subclass_of','target':edge1[1]},
                  {'source':edge2[0],'relation':'subclass_of','target':edge2[1]}]
            for style in ('direct','explain'):
                choices=['Yes','No'] if lang=='en' else ['是','否'];target=choices[0 if truth else 1]
                if lang=='en':rule='Answer only Yes or No.' if style=='direct' else 'Explain briefly, then finish with exactly Answer: Yes or Answer: No.'
                else:rule='只回答是或否。' if style=='direct' else '简短解释，最后严格以“答案：是”或“答案：否”结束。'
                text=body+'\n'+ask+'\n'+rule
                prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);body_start=prompt.index(body);body_end=body_start+len(body)
                before=max(i for i,(s,e) in enumerate(enc['offset_mapping']) if e>s and e<=body_end)
                ids=[tok(c,add_special_tokens=False)['input_ids'] for c in choices];assert all(len(x)==1 for x in ids)
                rows.append({'sample_id':'u2738_'+ranked(group+'/'+lang+'/'+style)[:20],'source_group':group,'kind':'controlled_language','cohort':family+'_'+lang,
                  'family':family,'language':lang,'representation':lang+'_'+style,'answer_style':style,'split':split,'truth':truth,'target':target,
                  'candidate_texts':choices,'candidate_ids':[x[0] for x in ids],'target_ids':ids[0 if truth else 1],
                  'original_text':text,'body':body,'question':ask,'text':prompt,'relations':relations,'prompt_ids':enc['input_ids'],
                  'token_offsets':enc['offset_mapping'],'anchors':[before,len(enc['input_ids'])-1],'body_end_character':body_end,'capture_mode':'language',
                  'scope':'New controlled bilingual cases, not authentic natural corpus. Meaningful labels are recipe-derived; same semantic family across language/output-style variants. Explanation style has a different requested first-token format.'})
    assert len(rows)==640 and len({r['sample_id'] for r in rows})==640
    assert len({tuple(r['prompt_ids']) for r in rows})==640
    compressed(BASE/'language_material.json.gz',rows)
    result={'timestamp':stamp(),'source':snapshot(__file__),'rows':len(rows),'families':list(FAMILIES),'semantic_groups':160,
      'split_groups':{'train':80,'validation':40,'test':40},'languages':['en','zh'],'styles':['direct','explain'],
      'material_sha256':sha(BASE/'language_material.json.gz'),'minimum_tokens':min(len(r['prompt_ids']) for r in rows),'maximum_tokens':max(len(r['prompt_ids']) for r in rows),
      'scoring':'At initial prefix, binary conditional loss is a declared forced-candidate diagnostic; explanation first-token CE is not final-answer accuracy. Native final-answer scoring is separately evaluated with terminal markers and censoring.',
      'limitations':'No claim of complete natural-language semantics, random family leakage exclusion across all project history, or universal word-sense resolution.',
      'seconds':time.monotonic()-start}
    save(BASE/'language_material_frozen.json',result);ledger('freeze_bilingual_relation_material',result['seconds']);print('LANGUAGE_MATERIAL',result,flush=True)

if __name__=='__main__':main()
