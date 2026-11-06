"""Unique, externally keyed atomic-number attributes; no colour relabelling."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as OLD
from shared.dist_protocol.mechanism_trace import digest,file_sha
ROOT=OLD.ROOT;MODEL=OLD.MODEL;SEED=27942026;RUN='P2794-ATOMIC-ARBITRATION-v4'
OUT=ROOT/'tests/glm5/result/phase2794_arbitration_v4';PUBLIC=ROOT/'research/phase2794';save=OLD.save
SOURCE='https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON'
RULE='作答规则：问题指定文档时，只复述虚构文档记载的数值；问题指定现实时，忽略文档的人为改写，按现实化学周期表回答该元素固定的原子序数。两种范围不可混用。只输出一个阿拉伯整数，不写单位和解释。'

def entities():
    raw=json.loads((OUT/'pubchem-periodic-table.json').read_text(encoding='utf-8'))
    names=raw['Table']['Columns']['Column'];rows=[dict(zip(names,r['Cell'])) for r in raw['Table']['Row']]
    rows=[{'id':r['Symbol'],'entity':r['Name']+' ('+r['Symbol']+')','prior':str(r['AtomicNumber']),'source':SOURCE} for r in rows if int(r['AtomicNumber'])<=96]
    assert len(rows)==96 and {int(r['prior']) for r in rows}==set(range(1,97))
    rows.sort(key=lambda r:digest({'seed':SEED,'entity':r['id']}))
    for i,r in enumerate(rows):r['split']='seen' if i<64 else 'new_entity'
    return rows

def build_items(tok,mode):
    b=OLD.BACK.Builder(tok);es=entities();items=[]
    base={'文档':'本轮只做信息抽取：严格复制材料中目标元素的虚构原子序数，不使用真实周期表。',
          '现实':'本轮只做知识问答：忽略材料中所有虚构数值，使用真实周期表回答目标元素的原子序数。'}
    tails=['','请核对。','请认真核对。','请仔细核对。','请核对范围。','务必遵循任务。','请遵循这个范围。','只完成本轮任务。','请先分清所问的范围。','请注意当前任务的范围。']
    smaps={s:{b.count(t+tail):(t+tail) for tail in tails} for s,t in base.items()};common=set(smaps['文档'])&set(smaps['现实']);assert common
    size=min(common);systems={s:smaps[s][size]+'只输出一个阿拉伯整数，不写单位和解释。' for s in base}
    if mode=='pilot':jobs=[(i,f,1024,'pilot') for i in range(16) for f in [0,1]]
    else:
        jobs=[(i,f,1024,'discovery' if f<2 else 'unseen_combination') for i in range(64) for f in range(4)]
        jobs += [(i,f,1024,'unseen_entity') for i in range(64,96) for f in [2,3]]
        jobs += [(i,0,n,'length_panel') for i in range(16) for n in [512,2048]]
    for i,frame,length,split in jobs:
        e=es[i];pool=es[:64] if i<64 else es[64:];j=(pool.index(e)+[1,7,19,31][frame])%len(pool);other=pool[j]
        # All facts/partners in heldout frames differ from this target's discovery combinations.
        choices=[str(n) for n in range(1,97) if str(n)[0]!=e['prior'][0]]
        doc=choices[(int(e['prior'])*7+frame*13)%len(choices)]
        odoc=str((int(other['prior'])-1+[17,29,43,59][frame])%96+1)
        source=[f'虚构文档记载：{e["entity"]}的原子序数为{doc}。',f'小说里的元素表列出：{e["entity"]}，原子序数{doc}。',
            f'虚构研究笔记给出的设定是：{e["entity"]}对应原子序数{doc}。',f'故事作者把元素记录改写为：原子序数{doc}属于{e["entity"]}。'][frame]
        tail='上述数字仅是虚构文档的内容，不改变真实元素的定义。'
        qs={s:f'请按本轮系统任务回答：{e["entity"]}的原子序数是多少？' for s in ['文档','现实']}
        def wrap(s,text):return tok.apply_chat_template([{'role':'system','content':systems[s]},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        reference=f'现实参考卡（来源：PubChem元素表，经独立核对）：{e["entity"]}的真实原子序数为{e["prior"]}。'
        suffix={s:reference+'\n'+source+tail+'\n'+q for s,q in qs.items()};sizes={s:b.count(wrap(s,t)) for s,t in suffix.items()}
        assert len(set(sizes.values()))==1,(e['id'],frame,sizes)
        prefixes=b.prefixes('handoff',frame);found=None
        for k in range(length-next(iter(sizes.values()))-8,length-next(iter(sizes.values()))+9):
            if k not in prefixes:continue
            texts={s:prefixes[k]+suffix[s] for s in qs};prompts={s:wrap(s,t) for s,t in texts.items()};ids={s:tok.encode(t,add_special_tokens=False) for s,t in prompts.items()}
            if all(len(t)==length for t in ids.values()):found=texts,prompts,ids;break
        if found is None:raise ValueError(f'No whole-sentence exact length {e["id"]} F{frame} L{length}')
        texts,prompts,ids=found
        for scope in qs:
            prompt=prompts[scope];start=prompt.index(source);ep=len(tok.encode(prompt[:start+source.index(e['entity'])+len(e['entity'])],add_special_tokens=False))-1
            items.append({'id':f'{e["id"]}:F{frame}:L{length}:'+('doc' if scope=='文档' else 'world'),'family_id':e['id'],'entity':e['entity'],
                'partner':'','system_instruction':systems[scope],'split':split,'frame':frame,'length':length,'tokens':length,'scope':scope,'prior':e['prior'],'document':doc,
                'expected':doc if scope=='文档' else e['prior'],'text':texts[scope],'prompt':prompt,'token_ids':ids[scope],
                'readout_positions':[ep,length-1],'entity_position':ep,'source_text':source,'source_question':qs[scope],
                'reference_text':reference,'reference_mode':'explicit trusted reference in BOTH scopes; not closed-book world knowledge'})
    return items

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);mode=sys.argv[1] if len(sys.argv)>1 else 'pilot'
    ids={str(i):tok.encode(str(i),add_special_tokens=False) for i in range(1,97)}
    c={'schema':'atomic_arbitration_corpus.v1','kind':'real','run_id':RUN,'seed':SEED,'entities':entities(),'items':build_items(tok,mode),
       'answer_ids':ids,'digit_ids':{str(i):tok.encode(str(i),add_special_tokens=False)[0] for i in range(10)},'model_snapshot':OLD.W.read(OLD.OUT/'execution-freeze.json')['model_snapshot'],
       'reference_source':SOURCE,'reference_file_sha256':file_sha(OUT/'pubchem-periodic-table.json'),'scope':'atomic number uniquely defines the element, not a general attribute atlas'}
    save(OUT/(mode+'-corpus.json'),c,True);print({'mode':mode,'n':len(c['items']),'sha256':digest(c)})
if __name__=='__main__':main()
