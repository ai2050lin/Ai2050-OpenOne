"""Roster order, fact order and semantic target are separate explicit factors.

Controlled language fixtures, not an exhaustive atlas or proof of semantics.
"""
import itertools
from phase2677_source_role_material import encode
FAMILIES=('chronology','taxonomy','word_sense','syntax_role','negation','comparison','reference','punctuation')
NAMES={
 'initial':{'en':'Cleo Dorian Eamon Hester Lucan Mirelle Orson Selene'.split(),'zh':'浦棠 缪舟 逄竹 仉禾 逯星 訾枫 麹汀 郜雁'.split()},
 'confirmation':{'en':'Ansel Beatrix Caspian Delia Emery Helena Imogen Jules'.split(),'zh':'邴霞 殳松 糜霖 郏月 竺弦 茹滔 巩榆 殷笙'.split()}}

def situation(family,language,content,form,a,b,target,split):
    en=language=='en';fresh=split=='confirmation';c=content;names=(a,b);winner=names[target];other=names[1-target]
    if family=='chronology':
        early,late=((('07:10','08:50'),('15:25','16:45')) if fresh else (('08:15','09:40'),('13:05','14:20')))[c]
        vals=[early if i==target else late for i in range(2)]
        facts=[(f'{n} registered at {x} on the same day.' if form==0 else f'The same-day registration time recorded for {n} was {x}.') if en else (f'{n}在同一天的{x}登记。' if form==0 else f'登记表中，{n}在当日登记的时间为{x}。') for n,x in zip(names,vals)]
        predicate='registered earlier' if en else '登记得较早'
    elif family=='taxonomy':
        fruit,tool=((('apricot','crowbar'),('fig','clamp')) if fresh else (('plum','rivet'),('kiwi','ladle')))[c] if en else ((('杏子','撬棍'),('无花果','夹具')) if fresh else (('李子','铆钉'),('猕猴桃','汤勺')))[c]
        vals=[fruit if i==target else tool for i in range(2)]
        facts=[(f'{n} carried one {x}.' if form==0 else f'One {x} was inside the bag carried by {n}.') if en else (f'{n}携带了{x}。' if form==0 else f'{n}所带的袋子里面装着{x}。') for n,x in zip(names,vals)]
        predicate='carried fruit' if en else '携带了水果'
    elif family=='word_sense':
        if en:
            catalogs=[('bark','examined bark peeling from a tree trunk','heard a bark from a dog','tree covering'),('spring','replaced a metal spring inside a clock','saw flowers bloom in spring','a coiled metal part')]
            if fresh:catalogs=[('pupil','examined the pupil at the center of an eye','taught a pupil in a classroom','a part of an eye'),('mole','watched a mole digging a tunnel','examined a dark mole on the skin','a burrowing animal')]
            word,yes,no,meaning=catalogs[c];vals=[yes if i==target else no for i in range(2)]
            facts=[f'{n} {x}.' if form==0 else f'According to the observation, {n} {x}.' for n,x in zip(names,vals)]
            predicate=f'encountered "{word}" meaning {meaning}'
        else:
            catalogs=[('喇叭','吹奏了铜制的喇叭乐器','按响了汽车的喇叭','乐器'),('窗口','关闭了电脑程序的窗口','修好了外墙的玻璃窗口','软件界面部件')]
            if fresh:catalogs=[('键','按下了钢琴的黑色键','讨论了分子中的化学键','琴键'),('苗','给菜地里的苗浇水','介绍了苗族的传统服饰','幼小植物')]
            word,yes,no,meaning=catalogs[c];vals=[yes if i==target else no for i in range(2)]
            facts=[f'{n}{x}。' if form==0 else f'观察记录表明，{n}{x}。' for n,x in zip(names,vals)]
            predicate=f'的情境中“{word}”指{meaning}'
    elif family=='syntax_role':
        if en:
            if not fresh:
                facts=([f'{winner} interviewed {other}.',f'{other} answered the questions from {winner}.'] if form==0 else [f'{other} was interviewed by {winner}.',f'The questions from {winner} were answered by {other}.']) if c==0 else ([f'{winner} photographed {other}.',f'{other} posed for {winner}.'] if form==0 else [f'{other} was photographed by {winner}.',f'{winner} was the photographer for whom {other} posed.'])
                predicate=('conducted the interview','took the photograph')[c]
            else:
                verb,noun=(('invited','invitation'),('thanked','thanks'))[c]
                facts=[f'{winner} {verb} {other}.',f'{other} received the {noun} from {winner}.'] if form==0 else [f'{other} was {verb} by {winner}.',f'The {noun} received by {other} came from {winner}.']
                predicate=('issued the invitation','expressed thanks')[c]
        else:
            verb,noun=(('邀请','邀请'),('感谢','谢意'))[c] if fresh else (('采访','提问'),('拍摄','拍摄安排'))[c]
            facts=[f'{winner}{verb}了{other}。',f'{other}接受了来自{winner}的{noun}。'] if form==0 else [f'{other}被{winner}{verb}了。',f'来自{winner}的{noun}由{other}接受。']
            predicate=('发出了邀请','表达了谢意')[c] if fresh else ('实施了采访','实施了拍摄')[c]
    elif family=='negation':
        verb,part=(('archive','archived'),('print','printed'))[c] if fresh else (('upload','uploaded'),('review','reviewed'))[c]
        zhverb=('归档','打印')[c] if fresh else ('上传','审阅')[c]
        if en:
            facts=[(f'{n} did not {verb} the report.' if i==target else f'{n} {part} the report.') if form==0 else (f'The report was not {part} by {n}.' if i==target else f'The report was {part} by {n}.') for i,n in enumerate(names)]
            predicate=f'did not {verb} the report'
        else:
            facts=[(f'{n}没有{zhverb}报告。' if i==target else f'{n}已经{zhverb}报告。') if form==0 else (f'报告尚未由{n}{zhverb}。' if i==target else f'报告已经由{n}{zhverb}。') for i,n in enumerate(names)]
            predicate=f'没有{zhverb}报告'
    elif family=='comparison':
        lo,hi=((31,46),(58,73))[c] if fresh else ((12,29),(37,54))[c]
        thing=('buttons','tickets')[c] if en else ('纽扣','票券')[c];vals=[hi if i==target else lo for i in range(2)]
        facts=[(f'{n} counted {v} {thing}.' if form==0 else f'The number of {thing} counted by {n} was {v}.') if en else (f'{n}清点了{v}枚{thing}。' if form==0 else f'{n}清点的{thing}数量为{v}。') for n,v in zip(names,vals)]
        predicate='counted the larger number' if en else '清点的数量较多'
    elif family=='reference':
        obj=(('envelope','box') if fresh else ('parcel','package'))[c] if en else (('信封','盒子') if fresh else ('包裹','小包'))[c]
        if en:
            facts=[f'{other} handed a {obj} to {winner}.',f'The recipient opened the {obj}.'] if form==0 else [f'{winner} received a {obj} from {other}.',f'The receiver then opened it.']
            predicate=f'opened the {obj}'
        else:
            facts=[f'{other}把一个{obj}交给{winner}。',f'接收者打开了{obj}。'] if form==0 else [f'{winner}从{other}那里收到了一个{obj}。',f'收到它的人随后把它打开。']
            predicate=f'打开了{obj}'
    elif family=='punctuation':
        phrase=(('Is the gate open','Where is the train') if fresh else ('Where are we','Is lunch ready'))[c] if en else (('门开了吗','火车在哪里') if fresh else ('我们在哪里','午饭好了吗'))[c]
        marks=['?' if i==target else '.' for i in range(2)] if en else ['？' if i==target else '。' for i in range(2)]
        facts=[(f'{n} wrote "{phrase}{m}"' if form==0 else f'The note written by {n} ended with "{phrase}{m}"') if en else (f'{n}写道“{phrase}{m}”' if form==0 else f'{n}所写便条的结尾是“{phrase}{m}”') for n,m in zip(names,marks)]
        predicate='used a question mark' if en else '使用了问号'
    else:raise KeyError(family)
    return facts,predicate

def build(tok,split='initial'):
    rows=[];names=NAMES[split]
    for fam,lang,e,c,f,roster,order,v in itertools.product(FAMILIES,('en','zh'),range(4),range(2),range(2),range(2),range(2),range(2)):
        a,b=names[lang][2*e:2*e+2];facts,predicate=situation(fam,lang,c,f,a,b,v,split);en=lang=='en'
        roster_names=(a,b) if roster==0 else (b,a);display_facts=facts if order==0 else list(reversed(facts))
        body=(f'Roster: {roster_names[0]}, {roster_names[1]}.\n' if en else f'名单：{roster_names[0]}、{roster_names[1]}。\n')+'\n'.join(display_facts)
        for mode in ('truth','mapped_truth','name','cloze'):
            mapped=mode=='mapped_truth';prefill='';affirm=v==0
            if mode in ('truth','mapped_truth'):
                words=['Yes','No'] if en else ['是','否'];positive,negative=words[int(mapped)],words[1-int(mapped)]
                text=body+(f'\nStatement: {a} {predicate}. Is this statement true?\nOutput rule: If true, output {positive}; if false, output {negative}. Output only the code.' if en else f'\n陈述：{a}{predicate}。这句话是否正确？\n输出规则：陈述正确时输出{positive}；错误时输出{negative}。只输出编码。')
                expected=affirm!=mapped;target,alternate=words[0 if expected else 1],words[1 if expected else 0]
            else:
                text=body+(f'\nQuestion: Which person {predicate}?\nOutput rule: Output only that person\'s name.' if en else f'\n问题：谁{predicate}？\n输出规则：只输出这个人的姓名。')
                if mode=='cloze':
                    text+='\nCompletion instruction: Continue the supplied sentence with only the name.' if en else '\n续写要求：接着给定句子，只填写人名。'
                    prefill=f'The person who {predicate} is ' if en else f'{predicate.removeprefix("的")}的人是'
                words=[a,b];target,alternate=(a,b)[v],(b,a)[v];expected=None
            row={'case_index':len(rows),'case_id':f'role_qkv/{split}/{fam}/{lang}/e{e}/c{c}/f{f}/r{roster}/o{order}/v{v}/{mode}',
                'family':fam,'language':lang,'unit':e,'content_instance':c,'form':f,'roster_order':roster,'mention_order':order,'target_index':v,'output_function':mode,
                'entity_a':a,'entity_b':b,'body':body,'text':text,'prefill':prefill,'target':target,'alternate':alternate,'common_readout_words':words,
                'short_answer_words':words,'polarity':0 if expected is not None else None,'probe_index':0 if expected is not None else None,
                'expected_yes':expected,'field_set':split,'source_selected':True,
                'published':(e,c,f,roster,order,v)==(0,0,0,0,0,0),'parameter_published':(e,c,f,roster,order,v)==(0,0,0,0,0,0) and mode=='truth',
                'gold_semantic_target':(a,b)[v],'facts_in_canonical_order':facts,'predicate':predicate,
                'operations':'r changes only roster order; o changes only fact order; v swaps relational target; f is specified surface realization; e renames entities. Roster is explicit scaffold, not a discovered model module.'}
            rows.append(encode(tok,row))
    assert len(rows)==8192 and len({r['prompt'] for r in rows})==8192
    return rows
