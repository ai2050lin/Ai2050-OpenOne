"""Same named entities across eight operations; answer, mention order and form crossed."""
import re
from phase2620_native_coordinate_contract import *

FAMILIES=('chronology','taxonomy','word_sense','syntax_role','negation','comparison','reference','punctuation')
EN_NAMES='Ava Noah Mia Liam Emma Owen Iris Finn Ruby Theo Lucy Hugo Nora Leon Ella Reed Nina Seth Lena Joel Tess Wade Cora Zane June Troy Rose Eric Gwen Sean Hope Ivan Dawn Alec Faye Blake Grace Cole Hazel Drew Jade Grant Joy Miles Kate Neil Lily Paul May Quinn Paige Ross Ruth Saul Skye Todd Vera Vince Wren Wyatt Zoe Evan Pearl Clark'.split()
ZH_NAMES='林岚 周博 沈琳 何峰 许宁 孙浩 陈悦 吴凯 郑雅 王辰 李晴 赵森 黄萱 张杰 刘芸 杨泽 朱瑶 胡轩 郭洁 梁宇 宋婷 唐越 冯佳 韩涛 曹曼 曾瑞 彭茜 蔡铭 潘涵 袁朗 董芮 苏恒 蒋欣 吕航 丁蕾 魏川 谢蓉 叶诚 杜玲 江毅 汪娴 石睿 廖莹 邓俊 贺蓓 秦晟 邱璐 白哲 侯薇 孟骁 龙婧 尹卓 段怡 雷鸣 邵倩 毛锋 顾珊 陆衡 薛琪 任勋 覃萌 史哲 郝雯 金岩'.split()
FRUIT_EN=['apple','pear','banana','peach','mango','cherry','lemon','grape']
FRUIT_ZH=['苹果','梨','香蕉','桃子','芒果','樱桃','柠檬','葡萄']
TOOLS_EN=['hammer','wrench','screwdriver','pliers','saw','drill','chisel','spanner']
TOOLS_ZH=['锤子','扳手','螺丝刀','钳子','锯子','电钻','凿子','套筒扳手']
SENSE_EN=[('bank','sat on the bank beside the flowing river','visited the bank to deposit money','a financial institution'),
          ('bat','watched a bat flying out of the cave','swung a bat to hit the baseball','sports equipment'),
          ('crane','photographed a crane standing in shallow water','operated a crane to lift a steel beam','a lifting machine'),
          ('seal','watched a seal swimming in the pool','pressed a seal onto the official document','an official stamp'),
          ('Apple','sliced an Apple for a fruit salad','installed software released by Apple','a technology company'),
          ('mouse','fed a mouse in a cage','clicked a mouse attached to a computer','a computer input device'),
          ('python','watched a python coil around a branch','wrote a program in Python','a programming language'),
          ('organ','examined an organ inside the body','played an organ in a church','a musical instrument')]
SENSE_ZH=[('苹果','切开苹果做水果沙拉','安装了苹果发布的软件','科技公司'),
          ('小米','用小米熬了一锅粥','购买了小米推出的手机','科技公司'),
          ('杜鹃','看见杜鹃振翅飞过树林','给花盆里的杜鹃浇水','植物'),
          ('长城','沿着长城的古城墙步行','阅读了长城汽车的产品说明','汽车品牌')]
INITIAL_UNITS=(0,1,30,31)
CONFIRM_UNITS=(12,13,14,15)

def item(family,lang,unit,form,target_index,order):
    en=lang=='en';names=EN_NAMES if en else ZH_NAMES;a,b=names[2*unit:2*unit+2]
    target=(a,b)[target_index];e0,e1=(a,b) if order==0 else (b,a);first_correct=target==e0
    if family=='chronology':
        if en:
            rel=('before' if first_correct else 'after') if form==0 else ('earlier than' if first_correct else 'later than')
            body=f'{e0} arrived {rel} {e1}.';question=' Who arrived first?' if form==0 else ' Which person was the earlier arrival?'
        else:
            body=f'{e0}比{e1}{"先" if first_correct else "后"}到达。' if form==0 else f'{e0}的到达时间{"早于" if first_correct else "晚于"}{e1}。'
            question='谁先到达？' if form==0 else '较早到达的是谁？'
    elif family=='taxonomy':
        fruit=(FRUIT_EN if en else FRUIT_ZH)[unit%8];tool=(TOOLS_EN if en else TOOLS_ZH)[unit%8]
        x,y=(fruit,tool) if first_correct else (tool,fruit)
        if en:
            article=lambda value:('an ' if value[0].lower() in 'aeiou' else 'a ')+value
            body=f'{e0} brought {article(x)}; {e1} brought {article(y)}.' if form==0 else f'The basket belonging to {e0} contained {article(x)}, while the basket belonging to {e1} contained {article(y)}.'
            question=' Who brought a fruit?' if form==0 else ' Whose basket contained fruit?'
        else:
            body=f'{e0}带来了{x}，{e1}带来了{y}。' if form==0 else f'{e0}的篮子里装着{x}，而{e1}的篮子里装着{y}。'
            question='谁带来了水果？' if form==0 else '谁的篮子里装着水果？'
    elif family=='word_sense':
        word,left,right,meaning=(SENSE_EN if en else SENSE_ZH)[unit%(8 if en else 4)]
        x,y=(right,left) if first_correct else (left,right)
        if en:
            body=f'{e0} {x}; {e1} {y}.' if form==0 else f'One report says that {e0} {x}. Another says that {e1} {y}.'
            question=f' In whose situation does "{word}" refer to {meaning}?'
        else:
            body=f'{e0}{x}；{e1}{y}。' if form==0 else f'一份记录说{e0}{x}。另一份记录说{e1}{y}。'
            question=f'谁的情境中的“{word}”指{meaning}？'
    elif family=='syntax_role':
        if en:
            body=(f'{e0} congratulated {e1}.' if first_correct else f'{e0} was congratulated by {e1}.') if form==0 else (f'{e0} gave praise to {e1}.' if first_correct else f'{e0} received praise from {e1}.')
            question=' Who did the congratulating?' if form==0 else ' Who gave the praise?'
        else:
            body=(f'{e0}祝贺了{e1}。' if first_correct else f'{e0}被{e1}祝贺了。') if form==0 else (f'{e0}向{e1}表达了赞扬。' if first_correct else f'{e0}收到了{e1}的赞扬。')
            question='谁实施了祝贺？' if form==0 else '谁表达了赞扬？'
    elif family=='negation':
        if en:
            x,y=('did not sign','signed') if first_correct else ('signed','did not sign')
            body=f'{e0} {x} the form; {e1} {y} the form.' if form==0 else f'The report says {e0} {x} the document, whereas {e1} {y} it.'
            question=' Who left the document unsigned?'
        else:
            x,y=('没有签署','签署了') if first_correct else ('签署了','没有签署')
            body=f'{e0}{x}表格；{e1}{y}表格。' if form==0 else f'记录显示{e0}{x}文件，而{e1}{y}文件。'
            question='谁没有签署文件？'
    elif family=='comparison':
        low=unit+2;high=low+7;x,y=(high,low) if first_correct else (low,high)
        if en:
            body=f'{e0} collected {x} shells; {e1} collected {y} shells.' if form==0 else f'The count for {e0} is {x} stones, compared with {y} stones for {e1}.'
            question=' Who collected more?' if form==0 else ' Whose count is larger?'
        else:
            body=f'{e0}收集了{x}枚贝壳；{e1}收集了{y}枚贝壳。' if form==0 else f'{e0}拥有{x}颗石子，相比之下{e1}拥有{y}颗。'
            question='谁收集得更多？' if form==0 else '谁的数量较多？'
    elif family=='reference':
        if en:
            selector=('former' if first_correct else 'latter') if form==0 else ('first-mentioned person' if first_correct else 'second-mentioned person')
            body=f'{e0} and {e1} examined a map. The {selector} then folded it.';question=' Who folded the map?'
        else:
            selector=('前者' if first_correct else '后者') if form==0 else ('先提到的人' if first_correct else '后提到的人')
            body=f'{e0}和{e1}一起查看地图。随后{selector}把它折了起来。';question='谁把地图折了起来？'
    elif family=='punctuation':
        x,y=('?','!') if first_correct else ('!','?')
        if en:
            body=f'{e0} wrote "It is ready{x}"; {e1} wrote "It is ready{y}".' if form==0 else f'The note from {e0} ended "Already here{x}", while the note from {e1} ended "Already here{y}".'
            question=' Who used a question mark?'
        else:
            x,y=('？','！') if first_correct else ('！','？')
            body=f'{e0}写道“已经准备好了{x}”；{e1}写道“已经准备好了{y}”。' if form==0 else f'{e0}的便条末尾写着“已经到了{x}”，而{e1}的便条末尾写着“已经到了{y}”。'
            question='谁使用了问号？'
    else:raise KeyError(family)
    instruction=' Answer with the person\'s name only.' if en else '只回答人名。'
    return {'text':body+question+instruction,'entity_a':a,'entity_b':b,'target':target,'alternate':b if target==a else a,'first_entity':e0}

def build(tok):
    assert len(EN_NAMES)==len(ZH_NAMES)==64 and len(set(EN_NAMES))==len(set(ZH_NAMES))==64
    cases=[]
    for family in FAMILIES:
        for lang in ('en','zh'):
            for unit in range(32):
                for form in (0,1):
                    for v in (0,1):
                        for order in (0,1):
                            r=item(family,lang,unit,form,v,order)
                            prompt=tok.apply_chat_template([{'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                            encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);begin=prompt.index(r['text'])
                            spans={}
                            for name,key in [(r['entity_a'],'a'),(r['entity_b'],'b')]:
                                start=begin+r['text'].index(name);end=start+len(name)
                                spans[key]=[t for t,(s,e) in enumerate(encoded['offset_mapping']) if e>start and s<end]
                                assert spans[key]
                            aid=tok.encode(r['entity_a'],add_special_tokens=False);bid=tok.encode(r['entity_b'],add_special_tokens=False)
                            assert aid and bid
                            r.update(case_id=f'{family}/{lang}/u{unit}/f{form}/v{v}/o{order}',case_index=len(cases),family=family,language=lang,unit=unit,form=form,target_index=v,mention_order=order,
                                prompt=prompt,prompt_ids=encoded['input_ids'],token_strings=tok.convert_ids_to_tokens(encoded['input_ids']),entity_spans=spans,
                                entity_a_ids=aid,entity_b_ids=bid,common_readout_ids=[aid[0],bid[0]],common_readout_available=aid[0]!=bid[0],
                                field_set='initial' if unit in INITIAL_UNITS else 'confirmation' if unit in CONFIRM_UNITS else 'behavior_only')
                            assert r['text'].find(r['first_entity'])==min(r['text'].find(r['entity_a']),r['text'].find(r['entity_b']))
                            cases.append(r)
    return cases

def evaluate(row,text):
    clean=text.strip();strict=clean.casefold()==row['target'].casefold()
    content=clean.strip(' .。').casefold()==row['target'].casefold()
    return {'strict_correct':strict,'name_content_correct':content,'empty':not clean}
