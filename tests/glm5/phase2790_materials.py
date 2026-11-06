"""Controlled multi-domain long documents, exact active lengths within pairs.

These are authored template documents, not an external natural corpus. Both
styles share identical facts and background; lengths add background content.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_materials_v3 as OLD

SEED=27902026
DOMAINS=['community','handoff','observation','software']
ACTORS=['林宁','宋然','苏楠','许泽','陆青','江岚','叶澄','陈禾','唐月','沈知','程安','周晴']
INFO={
 'community':('社区读书会','青禾书屋','周六下午三点','遇到下雨就移到室内，活动照常举行','免费参加，不需要购票'),
 'handoff':('资料交接','南楼资料室','周三下午两点','旧版文件只读，新修订经复核后再发布','交接时保留原件，不删除旧版'),
 'observation':('花园观察课','学校东侧花园','周五上午九点','下雨时改在教室阅读已存的观察记录','记录观察，不把同时发生当作因果证明'),
 'software':('软件更新说明','测试环境','周一晚上八点','只调整日志格式，验证失败就回滚','不改变业务计算规则，并保留原始日志')}
OBJECTS={
 'community':['活动通知','阅读材料','报名说明','场地安排','参与者的问题','志愿者的分工'],
 'handoff':['交接清单','原始文件','版本说明','复核记录','接收人的问题','归档人员的分工'],
 'observation':['课程通知','观察记录','植物照片','教学安排','学生的问题','指导人员的分工'],
 'software':['更新说明','测试记录','日志样本','发布安排','使用者的问题','开发人员的分工']}


def facts(i):
    domain=DOMAINS[i%4];subject,place,when,condition,limit=INFO[domain]
    return {'domain':domain,'actor':ACTORS[(i//4)%len(ACTORS)],'subject':subject,'place':place,
        'time':when,'conditional_rule':condition,'invariant':limit,'record_code':f'R{i}'}


def background(f,frame):
    # Connected sections of the same document, with domain-specific objects.
    sections=['材料准备','事项核对','团队沟通','形成说明','复核保存','后续查阅',
        '意见整理','附件核对','问题汇总','记录校对','说明复核','交接归档']
    patterns=[
      ('{o}由同一团队整理，先核对来源，再供相关人员查阅。','关于{o}，团队先逐项核对材料来源，随后将整理完成的内容提供给相关人员查阅。'),
      ('大家对{o}提出的问题分别登记，确认后补充解释。','对于大家围绕{o}提出的具体问题，负责整理的人员先分别登记，在确认已有依据后再补充解释。'),
      ('{o}中的已知事实和待确认事项分别说明。','在涉及{o}的说明中，已经有依据的事实和仍然有待确认的事项分别表述，避免让读者混淆。'),
      ('复核{o}时保留原有语境，不只摘取其中一句。','团队复核{o}时，会一并查看前后的原有语境，避免只摘取其中一句而改变整段材料原先表达的意思。'),
      ('查阅{o}的人可以把疑问交给负责整理的同事。','需要查阅{o}的人员如果对相关内容存在疑问，可以把具体问题交给负责整理的同事，在原有材料范围内进一步核对。'),
      ('团队根据{o}讨论说明方式，尚未确认的信息不写成结论。','围绕{o}开展讨论时，团队主要核对怎样把已有事项交代清楚，并明确约定，不把尚未确认的信息直接写成既定结论。'),
      ('{o}整理完成后与相关附件一起保存。','在{o}完成本阶段的整理以后，团队会把文字说明与对应的附件放在一起保存，以便后续查阅者核对它们之间的联系。'),
      ('团队在{o}旁另列待复核的问题，避免改动已确认内容。','为了区分已经确认的事项和仍需复核的问题，团队在{o}旁另行记录待处理项目，不把补充意见直接混入已有事实。'),
    ]
    output=[]
    for s,section in enumerate(sections):
        for j,(short,long) in enumerate(patterns):
            obj=OBJECTS[f['domain']][(s+j)%6]
            a,b=short.format(o=obj),long.format(o=obj)
            if frame%2:
                a='一位同事说：'+a;b='讨论到'+section+'时，一位同事解释说：'+b
            elif j==0:
                a=section+'阶段，'+a;b='在'+section+'这个阶段，'+b
            output.append([a,b])
    return output


class Builder:
    def __init__(self,tok):self.tok=tok;self.cache={}
    def wrap(self,text):return self.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def count(self,text):return len(self.tok.encode(text,add_special_tokens=False))
    def prefixes(self,domain,frame):
        key=(domain,frame)
        if key in self.cache:return self.cache[key]
        f={'domain':domain};states={0:''}
        for variants in background(f,frame):
            nxt={}
            choices=[(self.count(v+'\n'),v+'\n') for v in variants]
            for n,prefix in states.items():
                for size,text in choices:
                    m=n+size
                    if m<=2200:nxt.setdefault(m,prefix+text)
            # Allow ending the document section at any whole-sentence boundary.
            states=nxt
            self.cache.setdefault(key,{})
            for n,text in states.items():self.cache[key].setdefault(n,text)
        return self.cache[key]
    def pair(self,i,frame,length,split):
        f=facts(i)
        if frame%2:
            source=(f'本页最后确认的安排如下：{f["actor"]}负责{f["subject"]}，时间为{f["time"]}，地点或对象是{f["place"]}。'
                f'条件安排是：{f["conditional_rule"]}。必须保留的限制是：{f["invariant"]}。')
        else:
            source=(f'关于{f["subject"]}，已经确认由{f["actor"]}负责，在{f["time"]}于{f["place"]}开展。'
                f'{f["conditional_rule"]}；{f["invariant"]}。以上为需要转述的全部核心安排。')
        common='请转述最后确认的核心安排，正文四到六句。交代负责人、地点或对象、时间、条件与限制，不新增事实，不输出语域标签。'
        maps={}
        for reg in ['acad','colloq']:
            d={}
            for req in OLD.requests(frame,reg):
                text='\n'+source+'\n'+common+'\n'+req
                d.setdefault(self.count(self.wrap(text)),(text,req))
            maps[reg]=d
        prefixes=self.prefixes(f['domain'],frame)
        for n in sorted(set(maps['acad'])&set(maps['colloq'])):
            for k in range(length-n-8,length-n+9):
                if k not in prefixes:continue
                body=prefixes[k];texts=[body+maps[r][n][0] for r in ['acad','colloq']]
                prompts=[self.wrap(t) for t in texts];ids=[self.tok.encode(p,add_special_tokens=False) for p in prompts]
                if not all(len(x)==length for x in ids):continue
                return [{'id':f'nat4-{i}-F{frame}-L{length}-{reg}','family_id':f'nat4-{i}',
                    'record':i,'frame':frame,'length':length,'tokens':length,'split':split,'register':reg,
                    'domain':f['domain'],'source_genre':'meeting_transcript' if frame%2 else 'working_document',
                    'facts':f,'text':text,'prompt':prompt,'token_ids':token,
                    'style_request':maps[reg][n][1],'source_group_id':f'authored-family:{i}',
                    'matching':'identical full document and facts outside equal-token style request; whole sentences, no truncation or padding'}
                    for reg,text,prompt,token in zip(['acad','colloq'],texts,prompts,ids)]
        raise ValueError(f'Cannot exactly match {i},frame{frame},length{length}')
