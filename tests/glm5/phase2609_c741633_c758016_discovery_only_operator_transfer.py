"""Freeze coordinate directions on discovery; apply without held-out donors."""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/'tests/glm5'; sys.path.insert(0,str(TESTS))
import model_utils
import phase2603_c643329_c659712_unique_natural_lockbox as material
import phase2605_c676097_c692480_singleprompt_source_patch as source
import phase2608_c725249_c741632_autonomous_source_band as greedy
OUT=TESTS/'result/phase2609_c741633_c758016_discovery_only_operator_transfer'
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
WORDS={'en':[['assembly','inspection','packing','delivery','painting','drying','testing','storage'],
 ['calibration','welding','polishing','sterilization','installation','registration','measurement','repair','printing','sealing']],
 'zh':[['装配','检查','包装','运送','涂漆','干燥','测试','入库'],['校准','焊接','抛光','消毒','安装','登记','测量','维修','印刷','密封']]}

def build(tok,split):
    pairs=[]
    for language in ('en','zh'):
        words=WORDS[language][0 if split=='discovery' else 1]
        combinations=list(itertools.combinations(words,2))[:20]
        for form in (0,) if split=='discovery' else (0,1,2):
            for i,(a,b) in enumerate(combinations):
                pair=[]
                for v in (0,1):
                    if language=='en':
                        relation=[f'{a} occurred {"before" if v==0 else "after"} {b}',f'{a} was {"earlier" if v==0 else "later"} than {b}',f'{a} {"preceded" if v==0 else "followed"} {b}'][form]
                        prompt=f'In this process report, {relation}. Which of these two events occurred first? Reply with only the event name.\nAnswer:'
                    else:
                        relation=[f'{a}{"早于" if v==0 else "晚于"}{b}',f'{a}{"先于" if v==0 else "后于"}{b}',f'{a}发生在{b}{"之前" if v==0 else "之后"}'][form]
                        prompt=f'工艺记录说明：{relation}。这两个事件中哪一个先发生？只回答事件名称。\n答案：'
                    pair.append({'pair_id':f'{split}/{language}/f{form}/{i}','case_id':f'{split}/{language}/f{form}/{i}/{v}',
                        'family':'chronology','language':language,'split':split,'form':form,'variant':v,
                        'target':a if v==0 else b,'alternate':b if v==0 else a,'prompt':prompt,
                        'prompt_ids':tok.encode(prompt,add_special_tokens=False),'events':[a,b]})
                spans=material.source_spans(pair[0]['prompt_ids'],pair[1]['prompt_ids'])
                for row,pos in zip(pair,spans):row['source_token_positions']=pos
                pairs.append(pair)
    return pairs

def main():
    model,tok,_=model_utils.load_model('qwen3',dtype=torch.bfloat16,use_8bit=False)
    train=build(tok,'discovery'); test=build(tok,'heldout'); source.save_json(OUT/'material/train.json',train);source.save_json(OUT/'material/test.json',test)
    source.OUT=OUT/'discovery'; p=source.collect_source_means(model,tok,train); field=np.load(p,mmap_mode='r')
    deltas=field[:,1].astype('float32')-field[:,0].astype('float32')
    means={lang:np.mean(deltas[[i for i,pair in enumerate(train) if pair[0]['language']==lang]],axis=0) for lang in ('en','zh')}
    OUT.mkdir(parents=True,exist_ok=True); np.savez(OUT/'frozen_train_directions.npz',**means)
    frozen_hash=source.sha256(OUT/'frozen_train_directions.npz')
    source.save_json(OUT/'protocol/frozen.json',{'time':datetime.now().astimezone().isoformat(),'train_directions_sha256':frozen_hash,'source_layer':5,'no_heldout_donor_in_learned_direction':True})
    source.OUT=OUT/'heldout'; p=source.collect_source_means(model,tok,test); tf=np.load(p,mmap_mode='r')
    oracle=tf[:,1].astype('float32')-tf[:,0].astype('float32')
    learned=np.stack([means[p[0]['language']] for p in test]); records=[]; summaries={}
    # No test normalization, angle alignment, answer identities or test delta used for learned conditions.
    for name,cond,vec in [('baseline','baseline',learned[:,6]),('learned025','source025',learned[:,6]),
        ('learned05','source05',learned[:,6]),('learned1','source1',learned[:,6]),('learned15','source15',learned[:,6]),
        ('learned_roll','source_roll',learned[:,6]),('learned_wrong','source_wrong',learned[:,6]),
        ('cross_language','source1',np.stack([means['zh' if p[0]['language']=='en' else 'en'][6] for p in test])),
        ('oracle_diagnostic','source1',oracle[:,6])]:
        rows=greedy.generate_condition(model,tok,test,cond,vec,{})
        for row,pair in zip(rows,test):row['condition']=name;row['form']=pair[0]['form'];row['language']=pair[0]['language']
        source.write_jsonl(OUT/f'behavior/{name}.jsonl',rows);records+=rows;summaries[name]=greedy.metrics(rows)
        print(name,json.dumps(summaries[name]),flush=True)
    transport={}
    for lang in ('en','zh'):
        for form in range(3):
            idx=[i for i,p in enumerate(test) if p[0]['language']==lang and p[0]['form']==form]
            actual=oracle[idx]; pred=learned[idx]
            cos=np.sum(actual*pred,axis=-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(pred,axis=-1)+1e-12)
            transport[f'{lang}/form{form}']={'mean_cos_by_checkpoint':cos.mean(0).tolist(),
                'median_relative_error_by_checkpoint':np.median(np.linalg.norm(actual-pred,axis=-1)/(np.linalg.norm(actual,axis=-1)+1e-12),axis=0).tolist(),
                'generation':{c:greedy.metrics([r for r in records if r['language']==lang and r['form']==form and r['condition']==c]) for c in summaries}}
    result={'phase':2609,'timestamp':datetime.now().astimezone().isoformat(),'train_pairs':len(train),'test_pairs':len(test),
        'conditions':summaries,'transport':transport,'frozen_sha256':frozen_hash,
        'checks':{'train_test_event_vocab_disjoint':all(set(WORDS[l][0]).isdisjoint(WORDS[l][1]) for l in WORDS),
        'train40_test120':len(train)==40 and len(test)==120,'all1080_greedy':len(records)==1080,
        'frozen_direction_unchanged':source.sha256(OUT/'frozen_train_directions.npz')==frozen_hash},
        'language_mechanism_closed':False}
    result['all_checks_passed']=all(result['checks'].values());source.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    memo=rf'''

## Phase 2609: 发现集全坐标规则对未见事件与三种算子表面的1080次生成测试 [{stamp}]

**测试原理、用例。** 中英各20训练pair；各20未见事件对×3种时间关系表面=120测试pair。训练事件词表与测试完全不交。仅用训练40 pair冻结37×2560原生坐标均值，不归一化到测试差分；测试给定recipient及数据生成器提供的算子位置，不读取其答案或counterfactual构造学习方向。

$$\mu_{{g,l}}=\frac1{{|D_g|}}\sum_{{(x^0,x^1)\in D_g}}[H_l(x^1)_S-H_l(x^0)_S],\quad H_{{5,S}}(x)\leftarrow H_{{5,S}}(x)+\alpha\mu_{{g,6}}.$$

layer5在发现合同固定；9条件包括四剂量、roll、错token、跨语言训练方向及单独标记的oracle诊断。自主生成最多24 token，另全坐标逐层比较预测差分与实际差分；不使用Top-K。

**结果汇总。** {json.dumps(summaries,ensure_ascii=False)}

**相关文件。** `{OUT}`中的train/test材料、冻结方向及哈希、全部1080生成、完整source场、逐表面37层误差/夹角及final；脚本为本Phase同名.py。

**分析、理论进展。** 可以直接区分同词形相对编码与跨同义算子复用。训练方向在新事件有效也最多支持这两种时间关系的局部相对编码，不能推广为六族万能规则。

**问题硬伤。** 同一20事件对被三个表面复用，有效测试不是120个独立语义事件；位置由实验者标出；训练均值仍来自配对数据而非模型自己提出规则；两事件比较比长距离叙事弱；source均值混合多token。oracle诊断与无测试donor条件严格分开。

**结论。** 冻结检查={json.dumps(result['checks'])}。量化可复用边界，不根据学习方向单次失败更换整条路线；机制未闭合。
'''
    if '## Phase 2609:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(memo)

if __name__=='__main__':main()
