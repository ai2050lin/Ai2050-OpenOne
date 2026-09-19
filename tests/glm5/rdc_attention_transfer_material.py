"""Fresh bilingual entity strings, eight operations and crossed task wording/order."""
import re
from rdc_conditional_common import *

ENTITIES=[
 ('Anika Vellin','Bastian Korven','岑映竹','邵问舟'),('Elodie Maren','Farid Solven','晏知棠','贺闻溪'),
 ('Galen Torvik','Hana Evers','蒲书遥','蔺以宁'),('Isolde Nerin','Jonas Avel','屈望舒','柏念初'),
 ('Keira Dalen','Lucan Sorell','韦听澜','萧语川'),('Maelle Vardin','Nikolai Tesven','尹映禾','温澄野'),
 ('Orla Brenin','Pascal Ulden','辛怀瑾','袁慕白'),('Rhea Corvin','Soren Talvik','季云岫','黎见山'),
 ('Thalia Fenwick','Ulric Marden','商予安','许聆夏'),('Vera Halden','Willem Norvik','喻清晏','池墨衡'),
 ('Xenia Borden','Yusuf Kelmar','顾言芷','唐映川'),('Zara Melvin','Amir Trell','苏望野','林书珩'),
 ('Beatrix Loden','Cillian Merik','楚听雨','闻星辞'),('Delia Rostan','Eamon Veller','陆栖迟','夏弦歌'),
 ('Freya Corlen','Gideon Navel','秦疏桐','何景初'),('Helena Brivik','Idris Welmar','莫予川','沈云枝')]
OBJECTS=[('linen pouch','亚麻袋'),('brass compass','黄铜罗盘'),('ceramic cup','陶瓷杯'),('velvet ribbon','天鹅绒带'),
 ('oak box','橡木盒'),('silver bell','银铃'),('glass marble','玻璃弹珠'),('wool scarf','羊毛围巾'),
 ('wooden whistle','木哨'),('copper key','铜钥匙'),('paper lantern','纸灯笼'),('leather notebook','皮面笔记本'),
 ('blue envelope','蓝信封'),('clay figurine','泥塑像'),('iron clasp','铁扣'),('woven basket','编织篮')]
FAMILIES=('handover','category_chain','quantity_update','role_binding','negation_scope','timeline_reorder','translation','style_rewrite')


def semantic_case(f,unit,language):
    zh=language=='zh';ae,be,az,bz=ENTITIES[unit];a,b=(az,bz) if zh else (ae,be);obj=OBJECTS[unit][int(zh)]
    meta={};depth=4+3*(unit%2)
    if f=='handover':
        third=f'值班员{70+unit}' if zh else f'Custodian {70+unit}'
        owners=([a,b,third,a,b] if depth==4 else [a,third,b,a,b,third,a,b])
        source=(f'最初，{a}拿着{obj}。' if zh else f'At the beginning, {a} carries the {obj}. ')
        source+=' '.join(f'{x}把它递给{y}。' if zh else f'{x} hands it to {y}.' for x,y in zip(owners,owners[1:]))
        instructions=(('请指出最后谁持有物品，再用一句话解释最后一次交接。','先写最终持有人姓名，然后说明最后发生的交接。') if zh else ('Identify the final holder, then explain the last handover in one sentence.','Name whoever now has the object and describe the final transfer.'))
        meta={'expected_holder':owners[-1],'owner_path':owners}
    elif f=='category_chain':
        cats=[f'R{40+unit}{chr(65+j)}' for j in range(depth+1)];positive=unit%2==0
        clauses=[f'{a}属于{cats[0]}。' if zh else f'{a} belongs to {cats[0]}.']
        clauses.extend(f'{cats[i]}的所有成员都属于{cats[i+1]}。' if zh else f'All members of {cats[i]} belong to {cats[i+1]}.' for i in range(depth-1))
        clauses.append((f'{cats[-2]}的成员'+('都属于' if positive else '都不属于')+f'{cats[-1]}。') if zh else (f'All members of {cats[-2]} belong to {cats[-1]}.' if positive else f'No member of {cats[-2]} belongs to {cats[-1]}.'))
        source=' '.join(clauses[::2]+clauses[1::2])
        instructions=((f'判断{a}是否属于{cats[-1]}，给出结论并写出支持它的关系链。',f'根据这些分类规则，说明{a}与{cats[-1]}的关系，并列出中间类别。') if zh else (f'Decide whether {a} belongs to {cats[-1]}. Give the conclusion and the supporting chain.',f'Explain how {a} relates to {cats[-1]} under these rules, listing the intervening categories.'))
        meta={'expected_membership':positive,'categories':cats}
    elif f=='quantity_update':
        av,bv=20+unit,33+unit;transfer=1+unit%3;bonus=4+unit%2
        source=(f'{a}原有{av}张卡片，{b}有{bv}张。{a}交给{b}{transfer}张，然后{b}又得到{bonus}张。此后没有变化。') if zh else (f'{a} starts with {av} cards and {b} with {bv}. {a} gives {transfer} cards to {b}; afterward {b} receives {bonus} more. Nothing else changes.')
        instructions=(('分别列出两人的最终数量和合计，并简要写出计算。','计算最后每个人有多少张卡片及两人的总数，说明加减步骤。') if zh else ('State both final counts and their total, briefly showing the arithmetic.','Work out how many cards each person has now and the combined number; explain the updates.'))
        meta={'expected_counts':[av-transfer,bv+transfer+bonus,av+bv+bonus]}
    elif f=='role_binding':
        source=f'{a}在车站把{obj}递给了{b}。' if zh else f'{a} handed the {obj} to {b} at the station.'
        instructions=(('用被动句改写这句话，保留施事、物品、接收者和地点。','让物品成为句子的主语，重述原事件，不交换两个人的角色。') if zh else ('Rewrite this in the passive voice, preserving the agent, object, recipient and location.','Make the object the subject of a passive sentence without swapping the two people.'))
        meta={'agent':a,'patient':obj,'recipient':b,'location':'station'}
    elif f=='negation_scope':
        source=(f'{a}检查了{obj}，但没有给它贴标签。{b}没有检查{obj}，却给它贴了标签。') if zh else (f'{a} inspected the {obj} but did not label it. {b} did not inspect the {obj} but did label it.')
        ask_inspect=unit%2==0;verb='检查' if ask_inspect else '贴标签'
        instructions=((f'说明谁确实执行了“{verb}”操作，并指出谁没有执行。',f'把“{verb}”的肯定执行者与被否定的执行者分开列出。') if zh else (f'Say who actually {"inspected" if ask_inspect else "labelled"} the object and who did not.',f'List the positive and negated participants for the {"inspection" if ask_inspect else "labelling"} action separately.'))
        meta={'expected_positive':a if ask_inspect else b,'expected_negative':b if ask_inspect else a}
    elif f=='timeline_reorder':
        clauses=[f'08:00，{a}收到了{obj}。',f'11:00，{a}把它交给{b}。',f'14:00，{b}把它放入柜子。'] if zh else [f'08:00, {a} received the {obj}.',f'11:00, {a} handed it to {b}.',f'14:00, {b} placed it in a cabinet.']
        source='\n'.join(clauses[i] for i in (2,0,1))
        instructions=(('按时间先后重新排列这些完整句子，不删改事件。','从最早到最晚列出三件事，保留原有人名、物品和时间。') if zh else ('Reorder these complete sentences chronologically without changing the events.','List the three events from earliest to latest, retaining names, object and times.'))
        meta={'ordered_clauses':clauses}
    elif f=='translation':
        source=(f'{a}原本想把{obj}寄给{b}，但由于大雨，决定明天再寄。') if zh else (f'{a} intended to mail the {obj} to {b}, but heavy rain led to a decision to send it tomorrow.')
        instructions=(('翻译成英文，保留人名、转折、原因和明天这一时间信息。','用英文准确重述这段话，不增加新的行动或原因。') if zh else ('Translate into Chinese, preserving names, contrast, cause and the timing tomorrow.','Restate this accurately in Chinese without adding actions or reasons.'))
        meta={'expected_output_language':'en' if zh else 'zh','semantic_checks':['intention','rain causes delay','tomorrow']}
    else:
        source=(f'喂，{b}，今天下午三点前把{obj}送给{a}，别忘了。') if zh else (f'Hey {b}, get the {obj} to {a} before three this afternoon. Do not forget.')
        instructions=(('改写成礼貌、正式的请求，保留人物、物品和截止时间。','在不改变事实要求的前提下，把这段话写得正式而客气。') if zh else ('Rewrite this as a polite formal request, preserving people, object and deadline.','Make this courteous and professional without changing the factual request.'))
        meta={'recipient_of_request':b,'delivery_to':a,'object':obj,'deadline':'today15:00'}
    return dict(family=f,unit=unit,language=language,u=a,v=b,source=source,instructions=list(instructions),external_constraints=meta)


def build(tok):
    prior='\n'.join(str(r.get('prompt','')) for name in ('i_factorial/material.json','k_long/prefixes.json','m_order/prefixes.json') for r in read(CAMPAIGN/name))
    for names in ENTITIES:
        assert all(name not in prior for name in names),'Entity string is not held out from current-campaign fit materials'
    rows=[]
    for unit in range(16):
     for family in FAMILIES:
      for language in ('en','zh'):
       base=semantic_case(family,unit,language)
       for form in (0,1):
        instruction=base['instructions'][form];source=base['source'];body=instruction+'\n\n'+source if form==0 else source+'\n\n'+instruction
        system='按用户要求完成语言操作，直接给出答案。' if language=='zh' else 'Carry out the requested language operation and give the answer directly.'
        prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);assert prompt.count(source)==1;start=prompt.index(source);end=start+len(source)
        positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<end and b>a];assert positions
        rows.append(dict(base,sample_id=f'o-{family}-{unit}-{language}-f{form}',base_id=f'o-{family}-{unit}-{language}',form=form,
          word_split='test',system=system,user=body,prompt=prompt,prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),
          record_token_positions=positions,target='Frozen attention-forecast domain transfer; not complete-answer scoring'))
    assert len(rows)==512 and len({r['prompt'] for r in rows})==512
    return rows
