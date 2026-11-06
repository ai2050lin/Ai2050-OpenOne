"""Tokenizer-exact parallel paraphrases, common neutral prefixes, crossed frames.

No truncation, masked padding, repeated-space fillers, or answer contamination.
Only complete meaning-preserving variants are used to match active token counts.
"""
import hashlib
import re

SEED=27772026
ORDERS=[[0,1,2,3,4,5,6,7],[4,0,3,1,6,2,5,7],[3,5,0,6,1,4,2,7],[6,4,2,0,5,1,3,7]]
INSTRUCTION=('请接着写三到五句话，保持前文的说话方式，把当前的结果和核对依据解释清楚。'
 '正文至少一次明确写出所核对的物品名称和正确的剩余份数，不要仅用“物资”或“东西”代替名称。'
 '不要增加新事实，不要列提纲，也不要输出风格标签。')
NEUTRAL=[
 ['登记表分为不同栏目。','登记表上设有若干不同的栏目。'],
 ['原始资料保存在同一文件夹内。','原始资料统一保存在专门的文件夹内。'],
 ['每一页都标有页码。','每一页资料均标有相应的页码。'],
 ['复核人员可以查看原件。','负责复核的人员能够查看原始文件。'],
 ['记录中的物资属于同一批次。','这份记录涉及的物资均属于同一批次。'],
 ['地点名称用于区分工作区域。','记录的地点名称仅用于区分不同工作区域。'],
 ['资料按日期顺序存放。','全部资料按登记日期的先后顺序存放。'],
 ['原件和副本具有相同的文字。','所保存的原件与副本具有相同的文字内容。'],
 ['不同区域的材料分别装订。','属于不同工作区域的材料分别装订和保存。'],
 ['核对过程中允许查阅附件。','在核对过程中可以查阅相关附件。'],
 ['记录内容与评价意见分开。','已登记的事实内容和评价意见分开保存。'],
 ['文件封面注明保管地点。','文件封面上注明了资料的保管地点。'],
 ['签收人可查验材料清单。','签收材料的工作人员可以查验对应清单。'],
 ['已保存的记录不随意改动。','已经保存的原始记录不进行随意改动。'],
 ['这里没有额外的调拨记录。','除此之外，这里没有记载额外的物资调拨。'],
 ['材料涉及的时间先后是明确的。','各项记录中涉及的事件先后顺序是明确的。'],
 ['资料整理完成后交给复核人员。','相关资料整理完成以后，交由复核人员继续查验。'],
 ['缺少的证据不能依靠猜测补充。','缺失的原始证据不能通过主观猜测予以补充。'],
 ['各个栏目使用相同的计数单位。','在这些不同的栏目中，物资都使用相同的计数单位。'],
 ['这段背景不提供最终的数量。','以上背景材料并未直接提供需要计算的最终数量。'],
 ['复核意见另附在原始记录之后。','负责核查的人员将复核意见另附在原始记录之后。'],
 ['材料名称以登记表中的写法为准。','有关材料的具体名称，采用原始登记表中已经使用的写法。'],
 ['清单中的不同栏目分别保存信息。','物资清单中设置的不同栏目，分别用于保存各项原始信息。'],
 ['日期标签用于帮助查找文件。','文件上的日期标签用于帮助查找对应的记录与附件。'],
 ['复印件不能代替缺失的原始证据。','对资料进行复印，并不能补出原始文件中从未记载的事实。'],
 ['工作区域的名称与数量计算分开。','工作区域使用什么名称，与记录中的数量计算属于不同问题。'],
 ['需要修正的说明另外注明原因。','如需对文字说明作出修正，应另行注明修正所依据的原因。'],
 ['查阅材料时保留其原有排列。','工作人员查阅相关材料时，需要保留这些文件原有的排列顺序。'],
 ['文件资料可以由另一位同事再次核验。','已经整理的文件资料，可以交由另一位工作人员进行再次核验。'],
 ['封面信息不能替代正文记录。','文件封面的信息用于检索，不能替代正文中具体记载的事项。'],
]


def fact(i):
    items=['记录册','文件夹','标签纸','信封','笔记本','资料袋','包装盒','索引卡']
    item=items[i%len(items)]
    n=120+(i*17)%300;add=12+(i*11)%70;sub=11+(i*7)%60
    return {'item':item,'initial':n,'added':add,'removed':sub,'quantity':n+add-sub,
        'actor':['林安','陈宁','周敏','王晨','刘悦','张平','李欣','赵文'][i%8],
        'place':['图书室','资料中心','展览馆','档案室'][i%4]}


def clauses(f,reg):
    item,n,a,s,name,place=[f[k] for k in ['item','initial','added','removed','actor','place']]
    if reg=='acad':return [
      [f'本记录说明{place}的物资核对过程，负责人为{name}。',f'本文记录的是{place}的物资核对情况，其负责人是{name}。',f'本次物资核对在{place}开展，并由{name}负责相关记录。'],
      [f'初始登记的{item}共有{n}份。',f'根据初始记录，{item}的数量为{n}份。',f'原始登记表中明确记载，{item}的初始数量是{n}份。'],
      [f'此后接收补充物资{a}份，并向另一工作组移交{s}份。',f'后续记录显示新增{a}份，同时有{s}份移交另一工作组。',f'在初始登记之后，接收数量为{a}份，向另一工作组移交的数量为{s}份。'],
      ['各项记录采用相同单位，期间没有损耗及其他调拨。','所有数量均采用同一计数单位，且记录期间不存在损耗或其他调拨。','上述物资使用一致的计数单位，期间未发生任何损耗，也没有另外的调拨。'],
      ['核对时应依次纳入增加与减少，并保留各自来源。','复核应按照事件的先后纳入增加项与减少项，同时保留来源依据。','在核对过程中，应当按顺序考虑新增部分与已移交部分，并分别保留其原始依据。'],
      ['移动存放位置不改变数量，移交部分不再计入当前库存。','物资存放位置发生移动时，其数量并不改变；已经移交的部分不属于当前库存。','仅改变物资的存放位置不能视为新增，已移交给其他工作组的部分也不应继续计入当前库存。'],
      ['以上陈述限定于已记录事实，不评价个人动机或工作效率。','上述说明仅依据已经登记的事实，不涉及对个人动机及工作效率的评价。','本段陈述的范围仅限于原始记录明确支持的事实，不据此推断有关人员的个人动机，也不评价其工作效率。'],
      ['据此可核算剩余物资，并说明各项变化与结果的关系。','因此，可依据这些记录计算剩余数量，并明确增加项、减少项与最终结果之间的关系。','由此，当前剩余数量应在原有数量的基础上纳入补充和移交，具体结果须与各项记录对应。']]
    return [
      [f'咱们说说{place}这次清点物资的事，负责的人是{name}。',f'这回要说的是{place}清点物资的事，是{name}在负责。',f'你看，这次是在{place}清点东西，负责记下来的人就是{name}。'],
      [f'一开始有{n}份{item}。',f'刚开始的时候，手里有{n}份{item}。',f'咱们先看看原来的记录，上面写的是一开始有{n}份{item}。'],
      [f'后来又送来{a}份，接着拿走{s}份交给另一个小组。',f'再后来呢，有人又送来{a}份，也有{s}份交给了别的小组。',f'往后看，咱们又收到了{a}份，接着还拿出{s}份，交到了另一个小组的手里。'],
      ['说的都是同一种单位，中间没弄丢，也没有别的调拨。','这里前后用的是同一种单位，东西没弄丢，也没人另外拿来或者拿走。','前面和后面说数量时，用的单位都是一样的；中间没有弄丢，也没有别的小组又拿来或拿走。'],
      ['算的时候先加上送来的，再减去拿走的，记录都留着。','咱们核对的时候，按先后把送来的加上，再把拿走的减掉，原来的记录别扔。','真要核对清楚，就照事情发生的顺序，把后来送来的加上，再把交出去的减掉；每一项的记录都好好留着。'],
      ['换张桌子放不会变多，交给别组的就别算在手里。','东西换个地方放，并不会多出来；已经交给别组的，也就别算在自己手里了。','你想，把东西从这边挪到那边，数量当然不会凭空多出来；已经交给另一个小组的那部分，也不能还算在咱们手里。'],
      ['就照记下来的说，别猜人家怎么想，也别评谁干得快。','咱们只说已经记下来的事就行，别替别人猜心思，也别急着评谁做得快不快。','这事咱们就照已经记下来的情况说，没有证据的先别猜；至于别人心里怎么想，干活快不快，这里都先别评价。'],
      ['这样就能算清还剩多少，再把来龙去脉说明白。','说白了，把这些数字对上，就知道手里还剩多少，也能讲明白是怎么来的。','所以嘛，咱们把原来的、后来送来的和已经拿走的放在一起核对，就能说明白现在手里还剩多少。']]


class Builder:
    def __init__(self,tok):self.tok=tok;self.prefixes={}
    def count(self,s):return len(self.tok.encode(s,add_special_tokens=False))
    def wrap(self,text):
        return self.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
    def variants(self,f,reg,frame):
        states={0:''}
        variants=clauses(f,reg)
        item,n,a,s,name,place=[f[k] for k in ['item','initial','added','removed','actor','place']]
        if frame==1:
            variants[0]=[f'在{place}开展的这次核对由{name}负责。',f'关于{place}的本次物资核对，记录负责人为{name}。',f'就{place}本次物资核对而言，负责进行记录的人员为{name}。'] if reg=='acad' else [f'{place}这次清点，是{name}在负责呢。',f'要说{place}这次清点的事，负责的是{name}。',f'你要问{place}这回谁负责清点，那就是{name}，咱们说的也是这件事。']
            variants[1]=[f'作为核对起点的，是{n}份{item}。',f'核对所依据的初始状态为{n}份{item}。',f'本次核对的出发点并非其他数量，而是记录中的{n}份{item}。'] if reg=='acad' else [f'先别急，原来手里是{n}份{item}。',f'咱们先记住，开始的时候手里有{n}份{item}。',f'这件事从哪儿算起呢？就是原来手里的{n}份{item}，别把起点弄错了。']
        elif frame==2:
            variants[0]=[f'由{name}负责的，是{place}的物资复核。',f'需要说明的核对发生于{place}，该项工作由{name}负责。',f'本段需要交代的是{place}物资核对的情况，其中相关记录与复核工作由{name}负责。'] if reg=='acad' else [f'{name}这回负责的，是{place}清点东西的事。',f'说清楚点，这回是{name}在{place}负责清点东西。',f'咱们这回聊的，就是{name}在{place}负责清点东西的那件事，先把这点说清楚。']
            variants[2]=[f'补充的{a}份计入增加项，交予另一组的{s}份计入减少项。',f'后续数量变动包括接收{a}份，以及向另一工作组移交{s}份。',f'接收补充物资意味着增加{a}份，而向另一工作组移交物资意味着减少{s}份，这两项均发生在初始登记之后。'] if reg=='acad' else [f'送来的{a}份要加上，交给别组的{s}份要减掉。',f'后来送过来的有{a}份，得加进去；给别的小组的有{s}份，要减掉。',f'往后还有两件事嘛：有人又送来{a}份，咱们要加上；又交给别组{s}份，这部分当然要减掉。']
        elif frame==3:
            variants[1]=[f'原始记录所确定的初始库存为{n}份{item}。',f'根据可供核验的原始记录，{item}起初共有{n}份。',f'作为后续核算依据的初始数据已经明确：原始登记的{item}共有{n}份，该数量构成起始库存。'] if reg=='acad' else [f'原来的单子写着，一开始是{n}份{item}。',f'翻翻原来的单子就知道，{item}一开始有{n}份。',f'咱们翻开最早那张单子看看吧，上面明明白白写着：一开始有{n}份{item}，就从这儿算起。']
            variants[7]=['最终结果应与原始库存、后续补充和移交记录共同对应。','计算当前剩余数量时，初始库存与后续接收、移交三项记录应保持一致。','由此得到的剩余数量，应当同时满足原始库存、后续补充和对外移交这三项记录所确定的数量关系。'] if reg=='acad' else ['最后剩多少，得跟原来和后来进出的都对上。','说白了，最后剩下的得跟开始有多少、后来收进多少和拿走多少都对得上。','所以嘛，最后到底还剩多少，不能随口一说，得把原来有的、后来送来的和已经交出去的都对一对才行。']
        for pos in ORDERS[frame]:
            new={}
            for text in states.values():
                for v in variants[pos]:
                    s=text+v+'\n';n=self.count(s)
                    if n not in new:new[n]=s
            states=new
        return states
    def neutral(self,panel):
        if panel in self.prefixes:return self.prefixes[panel]
        states={0:''};seq=NEUTRAL[panel:]+NEUTRAL[:panel]
        for variants in seq:
            new=dict(states)
            for text in states.values():
                for v in variants:
                    s=text+v+'\n';n=self.count(s)
                    if n<=420 and n not in new:new[n]=s
            states=new
        self.prefixes[panel]=states
        return states
    def pair(self,i,frame,length,split):
        f=fact(i);a=self.variants(f,'acad',frame);c=self.variants(f,'colloq',frame)
        common=sorted(set(a)&set(c),key=lambda n:(abs(n-230),n))
        if not common:raise ValueError('No meaning-preserving equal-length core')
        prefixes=self.neutral(i%2)
        for n in common[:18]:
            core_a=a[n]+'\n'+INSTRUCTION;core_c=c[n]+'\n'+INSTRUCTION
            base=self.count(self.wrap(core_a))
            for k in range(length-base-4,length-base+5):
                if k not in prefixes:continue
                prefix=prefixes[k]
                if prefix.count('。')<4:continue
                texts=[prefix+'\n'+core for core in [core_a,core_c]]
                prompts=[self.wrap(t) for t in texts]
                ids=[self.tok.encode(t,add_special_tokens=False) for t in prompts]
                if not all(len(x)==length for x in ids):continue
                assert ids[0][-1]==ids[1][-1]
                return [{'id':f'nat-{i:04d}-F{frame}-L{length}-{reg}','family_id':f'nat-{i:04d}',
                    'record':i,'frame':frame,'length':length,'split':split,'register':reg,
                    'text':text,'prompt':prompt,'token_ids':toks,'tokens':len(toks),'fact':f,
                    'matching':{'active_token_exact':True,'common_neutral_prefix':prefix,'core_tokens':n,
                        'no_truncation':True,'no_masked_padding':True}}
                    for reg,text,prompt,toks in zip(['acad','colloq'],texts,prompts,ids)]
        raise ValueError(f'No natural exact match for record{i} frame{frame} length{length}')


COLLOQ=['咱们','你看','说白了','嘛','啦','呀','呗','甭','别急','这么一来','手里','弄清','对一对','讲清楚','就行','没啥','别把','还得','可别','反正','就是这么','吧','呢']
ACADEMIC=['因此','据此','综上','表明','可知','依据','核算','核对','相应','数量为','剩余数量','上述','应当','该项','予以','进一步','过程','记录显示','一致','由此','应为','最终','验证']


def rubric(text):
    text=re.sub(r'“[^”]*”|「[^」]*」|"[^"]*"','',text)
    c=[s for s in COLLOQ if s in text];a=[s for s in ACADEMIC if s in text]
    # A single opening marker does not make a whole paragraph colloquial.
    tail=text[20:];ct=sum(s in tail for s in COLLOQ)
    label='colloq' if len(c)>=2 and ct>=1 and len(c)>len(a) else 'acad' if len(a)>=2 and len(a)>len(c) else 'unknown'
    sentences=[s for s in re.split('[。！？!?]',text) if s.strip()]
    return {'label':label,'colloquial':c,'academic':a,'sentences':len(sentences),'chars':len(text),
        'adequate':len(text)>=45 and len(sentences)>=2}


def chinese_number(s):
    if s.isdigit():return int(s)
    numbers={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    units={'十':10,'百':100,'千':1000}
    if not any(c in units for c in s):return int(''.join(str(numbers[c]) for c in s))
    total=0;digit=0
    for c in s:
        if c in numbers:digit=numbers[c]
        else:total+=(digit or 1)*units[c];digit=0
    return total+digit


def content(text,f):
    # Numbers immediately associated with quantity words; ordinal discourse is excluded.
    found=[]
    for m in re.finditer(r'([0-9]+|[零〇一二两三四五六七八九十百千]+)\s*(?:份|本|个|册|张)',text):
        try:found.append({'value':chinese_number(m.group(1)),'at':m.start()})
        except (ValueError,KeyError):continue
    allowed={f[k] for k in ['initial','added','removed','quantity']}|{f['initial']+f['added']}
    bad=[x['value'] for x in found if x['value'] not in allowed]
    item=f['item'] in text;quantity=any(x['value']==f['quantity'] for x in found)
    return {'item_ok':item,'quantity_ok':quantity,'unexpected_quantities':bad,'pass':item and quantity and not bad,
        'quantity_mentions':found,'scope':'automatic item/quantity consistency; not full semantic equivalence'}


def references():
    # Authored references use different content and frames than experimental inventory.
    topics=['会议日程','课程安排','阅读计划','路线选择','设备检查','预约流程','资料检索','学习方法']
    rows=[]
    for i,topic in enumerate(topics):
        rows += [{'id':f'ref-{i}-A','label':'acad','text':f'关于{topic}，首先应明确适用条件，并依据现有信息核对相关安排。各项记录应当保持一致，避免将尚未验证的判断作为既定事实。因此，后续工作需要逐项核查，最终形成具有明确依据的说明。'},
          {'id':f'ref-{i}-C','label':'colloq','text':f'说到{topic}，咱们先别急着下结论，先看看手头都有什么信息吧。你看，把前后的事情对一对，哪些能确定、哪些还得再问问，就比较清楚了。说白了，先把事情弄明白，再往下安排就行。'}]
    rows += [{'id':'trap-A','label':'acad','text':'本文以“咱们”“你看”和“说白了”为分析对象，考察这些表达的语用功能。相关词项作为引用材料出现，并不改变本文的论述方式。因此，判断文本语域需要结合完整语境，不能仅依据某个词项是否出现。'},
      {'id':'trap-C','label':'colloq','text':'你看，我刚才说的“因此”和“综上”只是举个例子嘛，又不是让你照着写报告。咱们把事情说明白就行，别一看到那几个词，就觉得整个说话方式都变了呀。'}]
    return rows
