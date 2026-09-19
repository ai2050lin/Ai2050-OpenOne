"""64 lexical-use entries x four query forms x two languages. Labels are task conventions."""
from rdc_feature_common import *

LEXICON = [
 ('fruit','水果','apple banana orange pear peach grape mango lemon','苹果 香蕉 橙子 梨 桃子 葡萄 芒果 柠檬'),
 ('whole_plant','完整植物','oak pine birch cedar willow bamboo fern moss','橡树 松树 桦树 雪松 柳树 竹子 蕨类 苔藓'),
 ('animal','动物','cat dog horse rabbit tiger lion whale eagle','猫 狗 马 兔子 老虎 狮子 鲸 鹰'),
 ('tool','工具','hammer saw wrench pliers drill shovel chisel screwdriver','锤子 锯子 扳手 钳子 电钻 铲子 凿子 螺丝刀'),
 ('action','动作','run walk jump swim eat drink write read','跑 走 跳 游泳 吃 喝 写 读'),
 ('property','性质','red blue large small warm cold heavy light','红色 蓝色 大 小 温暖 寒冷 重 轻'),
 ('function_word','功能词','and or but if because although with without','而且 或者 但是 如果 因为 虽然 和 没有'),
 ('punctuation','标点','. , ? ! : ; ( )','。 ， ？ ！ ： ； （ ）'),
]

def build(tokenizer):
    rows=[]
    rules={
      'en':'For this task use eight separate lexical-use groups: fruit names, whole-plant names, animal names, tools, actions (verbs), properties (adjectives), function words, and punctuation. Fruit names and whole-plant names count as different groups. Use the stated lexical usage, not a biological hierarchy. Compare the two entries and output only Yes or No.',
      'zh':'本题采用八个分开的词汇用途组：水果名、完整植物名、动物名、工具、动作（动词）、性质（形容词）、功能词和标点。水果名与完整植物名按不同组处理，按词汇用途而非生物学上下位关系判断。比较两个词，只输出是或否。'}
    questions={
      'en':['Are A and B in the same group?','Would this grouping put A together with B?',
            'Are A and B in different groups?','Would this grouping keep A separate from B?'],
      'zh':['甲词与乙词是否属于同一组？','按上述分组，甲词和乙词是否应放在一起？',
            '甲词与乙词是否属于不同组？','按上述分组，甲词和乙词是否应分开放置？']}
    for f,(family,label,en,zh) in enumerate(LEXICON):
        for unit in range(8):
            for form in range(4):
                same=(unit//2+form)%2==0
                partner_family=f if same else (f+1+(unit+form)%7)%8
                partner_unit=unit^1
                for language in ('en','zh'):
                    langindex=2 if language=='en' else 3
                    a=LEXICON[f][langindex].split()[unit];b=LEXICON[partner_family][langindex].split()[partner_unit]
                    a_prefix='A: ' if language=='en' else '甲词：'; b_prefix='B: ' if language=='en' else '乙词：'
                    lines=[a_prefix+a,b_prefix+b]
                    if (unit+form)%2: lines.reverse()
                    body='\n'.join(lines)+'\n'+questions[language][form]
                    messages=[{'role':'system','content':rules[language]},{'role':'user','content':body}]
                    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
                    spans={}
                    for name,prefix,term in [('u',a_prefix,a),('v',b_prefix,b)]:
                        start=prompt.index(prefix+term)+len(prefix);end=start+len(term)
                        positions=[i for i,(s,e) in enumerate(encoded['offset_mapping']) if e>start and s<end and e>s]
                        assert positions
                        spans[name]={'chars':[start,end],'positions':positions,'term':term}
                    # Separate native template embedding includes no lexical term tokens.
                    question_positions=[]
                    qstart=prompt.index(questions[language][form])
                    for i,(s,e) in enumerate(encoded['offset_mapping']):
                        if e>qstart and s<qstart+len(questions[language][form]):question_positions.append(i)
                    assert question_positions
                    expected= same if form<2 else not same
                    sample_id=f's1-{family}-{unit}-{form}-{language}'
                    rows.append({'sample_id':sample_id,'family':family,'family_index':f,'family_label':label,'unit':unit,'form':form,
                        'base_id':f'{family}-{unit}','lexical_block':unit//2,'language':language,'u':a,'v':b,
                        'partner_family':LEXICON[partner_family][0],'partner_family_index':partner_family,'partner_unit':partner_unit,
                        'same_group':same,'negative_query':form>=2,'expected_yes':expected,
                        'target':('Yes' if expected else 'No') if language=='en' else ('是' if expected else '否'),
                        'prompt':prompt,'prompt_ids':encoded['input_ids'],'tokens':tokenizer.convert_ids_to_tokens(encoded['input_ids']),
                        'spans':spans,'context_positions':question_positions,'input_order':'BA' if (unit+form)%2 else 'AB',
                        'role':'u is the named A/甲 entry regardless of physical order','source_mode':'live_model',
                        'word_split':'train' if unit<4 else 'validation' if unit<6 else 'test'})
    assert len(rows)==512 and len({r['sample_id'] for r in rows})==512
    assert len({r['prompt'] for r in rows})==512
    for split in ('train','validation','test'):
        group=[r for r in rows if r['word_split']==split]
        assert all((r['unit']//2<2 if split=='train' else r['unit']//2==(2 if split=='validation' else 3)) for r in group)
        assert all(r['partner_unit']//2==r['unit']//2 for r in group)
    return rows
