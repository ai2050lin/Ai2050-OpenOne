"""Candidate context cues, pole witnesses and the durable whole-tail report."""
from collections import Counter
import gc
import re
import shutil
import time
import numpy as np
import mlp_tail_goodfire as C

STOP=set('the and that this with from have which were there their they what when where into also than these those are was has had for not but can will would should been being you your its our his her she him all any more most some such each other about after before because only then them through between'.split())


def words(text):
    english=[w.lower() for w in re.findall(r'[A-Za-z]{3,24}',text) if w.lower() not in STOP]
    chinese=[]
    for run in re.findall(r'[\u4e00-\u9fff]{2,}',text):
        chinese.extend(run[i:i+2] for i in range(len(run)-1))
    return set(english+chinese)


def main():
    from transformers import AutoTokenizer
    C.OUT.mkdir(exist_ok=True);started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    source=C.read(C.OUT/'documents.json');positions=source['positions'];tokens=np.load(C.TOKENS,mmap_mode='r')
    contexts=[];df=Counter()
    for doc in range(2048):
        for p in positions:
            terms=words(tok.decode(tokens[doc,max(0,p-128):p+1].tolist()))
            contexts.append(terms)
            if doc<1024:df.update(terms)
    shutil.copyfile(C.MODEL/'tokenizer.json',C.OUT/'tokenizer.json')
    for bi,block in enumerate(C.BLOCKS):
        profile=C.read(C.OUT/f'B{block}-profile.json');cues=[]
        poles=np.lib.format.open_memmap(C.OUT/f'B{block}-pole-values.npy',mode='w+',dtype=np.uint16,shape=(3,3,2,C.WIDTH,8,3))
        maps={f:np.load(C.OUT/f'B{block}-{f}.npy',mmap_mode='r') for f in ['G','V','A']}
        for fi,field in enumerate(['G','V','A']):
            for si,split in enumerate(['discovery','confirm','holdout']):
                for pi,pole in enumerate(['positive','negative']):
                    ids=np.asarray(profile['fields'][field][split][pole+'_indices'])
                    units=np.arange(C.WIDTH)[:,None]
                    for component,name in enumerate(['G','V','A']):poles[fi,si,pi,:,:,component]=maps[name][ids,units]
        poles.flush()
        g=profile['fields']['G']
        for unit in range(C.WIDTH):
            positive=g['discovery']['positive_indices'][unit];negative=g['discovery']['negative_indices'][unit]
            pos=Counter();neg=Counter();seen=set()
            for idx in positive:
                if idx//16 not in seen:pos.update(contexts[idx]);seen.add(idx//16)
            seen_negative=set()
            for idx in negative:
                if idx//16 not in seen_negative:neg.update(contexts[idx]);seen_negative.add(idx//16)
            def score(term):return (pos[term]-.5*neg[term])*np.log1p(16384/max(1,df[term]))
            best=sorted((w for w in pos if df[w]>=3 and score(w)>0),key=lambda w:(-score(w),w))[:6]
            negbest=[w for w,_ in neg.most_common() if w not in best and df[w]>=3][:6]
            repeated={}
            for split in ['confirm','holdout']:
                ids=g[split]['positive_indices'][unit]
                repeated[split]={'cue_hit_count':sum(bool(contexts[i]&set(best)) for i in ids),'n_pole_positions':len(ids),
                                 'distinct_documents':len({i//16 for i in ids})}
            cues.append({'unit':unit,'positive_keywords':best,'negative_keywords':negbest,
                         'discovery_positive_documents':len(seen),'discovery_negative_documents':len(seen_negative),
                         'repeat':repeated,'status':'candidate_context_keywords_not_explanation',
                         'description':'候选语境线索：'+' / '.join(best[:3]) if best else '尚无稳定语境关键词；保留原始响应与上下游数据。'})
        C.save_json(C.OUT/f'B{block}-cues.json',{'schema':'mlp_tail_cues.v1','kind':'real','run_id':C.RUN,'block':block,'units':cues,
                  'scope':'仅发现集正负极语境关键词。不是概念定义、单义性、功能解释或因果结论。留出关键词命中只是线索检查。'})
        C.log('context_witnesses',bi+1,4,started,block=block)
        del profile,cues,poles,maps;gc.collect()
    path=C.read(C.OUT/'path-report.json');replay=C.read(C.OUT/'replay.json');prediction=C.read(C.OUT/'prediction.json')
    delta=np.load(C.OUT/'path-target-deltas.npy',mmap_mode='r');assert np.isfinite(delta).all()
    report={'schema':'mlp_tail_atlas_report.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
       'blocks':C.BLOCKS,'d_ff':C.WIDTH,'n_units':4*C.WIDTH,'documents':2048,'context_tokens':1024,'positions_per_layer':32768,
       'coverage':{'native_statistics':38912,'native_parameter_locators':38912,'semantic_explanations_confirmed':0,
                   'causally_sampled_source_units':12,'all_adjacent_pair_matrices':15},
       'replay':replay,'prediction':prediction,'path':path,
       'conclusion':'四层全量原生特性与关联已采集；稀疏关联的跨层闭合预测未成立。关键词是候选；局部干预不是全部神经元语义破解。',
       'limitations':['既有长文本库：留出仅对本估计器未拟合，不是新语义盲测。','输入库为MLP归一化后输入，不能当作未归一化残差方向做steering。',
           '写入/读取权重余弦未包含RMSNorm和注意力；相关矩阵不证明计算使用。',
           '8父单元状态预测缺少完整跨位置残差/注意力状态，失败不证明全部原生表示不可组合。',
           '关键词不等于特征解释；38,912是覆盖，不是已破解数量。',
           '能力保持只有自然下一词代理；翻译、推理和编程的非劣效尚未建立。',
           '抽样位置位于533--1006，不能覆盖每个token与所有长期依赖。'],
       'next_large_task':'冻结独立长文本与可检验语义解释，在完整残差/注意力边界上建立跨位置、跨层预测；扩展当前12源单元干预证据，保留反例和多任务能力门。'}
    C.save_json(C.OUT/'report.json',report);C.save_json(C.PUBLIC/'report.research.json',report)
    metadata={'schema':'mlp_tail_manifest.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
       'state':'complete','blocks':C.BLOCKS,'width':C.WIDTH,'positions':positions,'files':[]}
    for p in sorted(C.OUT.iterdir()):
        if p.is_file() and p.name!='manifest.json':
            metadata['files'].append({'path':p.name,'bytes':p.stat().st_size,'sha256':C.file_sha(p),
              'display_required':not re.match(r'B\d+-(G|V|A)\.npy$',p.name) and not p.name.endswith('scores.npz')})
    assert sum(f['bytes'] for f in metadata['files'])<16*2**30
    C.save_json(C.OUT/'manifest.json',metadata)
    C.log('atlas_finalized',38912,38912,started,bytes=sum(f['bytes'] for f in metadata['files']))


if __name__=='__main__':main()
