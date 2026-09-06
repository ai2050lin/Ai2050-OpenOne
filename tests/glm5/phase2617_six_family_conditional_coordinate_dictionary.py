"""Basic all-coordinate operation-conditioned dictionary; training-only activations."""
import json,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';RESULT=TESTS/'result';sys.path.insert(0,str(TESTS))
import model_utils
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as gen
OUT=RESULT/'phase2617_six_family_conditional_coordinate_dictionary';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'

def group(p):return p[0]['family']+'/'+p[0]['language']
def key(p):
    return group(p)+'/'+json.dumps([[r['prompt_ids'][i] for i in r['source_token_positions']] for r in p])

def main():
    train=gen.pairs_for('discovery');test=gen.pairs_for('external')
    model,tok,_=model_utils.load_model('qwen3',dtype=torch.bfloat16,use_8bit=False)
    io.OUT=OUT;sf=np.load(io.collect_source_means(model,tok,train,storage_dtype=np.float32));delta=sf[:,1]-sf[:,0]
    grouped=defaultdict(list);keyed=defaultdict(list)
    for i,p in enumerate(train):grouped[group(p)].append(i);keyed[key(p)].append(i)
    means={g:delta[idx].mean(0) for g,idx in grouped.items()};dictionary={k:delta[idx].mean(0) for k,idx in keyed.items()}
    index_keys=sorted(dictionary);OUT.mkdir(parents=True,exist_ok=True)
    np.savez(OUT/'frozen_dictionary.npz',keys=np.array(index_keys),directions=np.stack([dictionary[k] for k in index_keys]))
    checksum=io.sha256(OUT/'frozen_dictionary.npz');io.save_json(OUT/'protocol/frozen.json',{'hash':checksum,'fit_pairs':240,'source_layer':5,'query_key':'family/language/source-token replacement specification; external operation supplied, not inferred by model','no_test_donor_activations':True})
    family=np.stack([means[group(p)][6] for p in test]);coverage=[key(p) in dictionary for p in test]
    selected=np.stack([dictionary.get(key(p),means[group(p)])[6] for p in test]);wrong=[];wrong_scope=[]
    for p,v in zip(test,selected):
        candidates=[k for k in index_keys if k!=key(p) and k.startswith(group(p)+'/')]
        wrong_scope.append('same_family_language' if candidates else 'other_family_language')
        other=(candidates or [k for k in index_keys if k!=key(p)])[0]
        w=dictionary[other][6];wrong.append(w*np.linalg.norm(v)/(np.linalg.norm(w)+1e-12))
    records=[];summary={}
    for name,c,vec in [('baseline','baseline',family),('family_mean','source1',family),('conditioned_dictionary','source1',selected),('dictionary_roll','source_roll',selected),('different_operation_matched_norm','source1',np.stack(wrong))]:
        rows=gen.generate_condition(model,tok,test,c,vec,{})
        for r,p,covered in zip(rows,test,coverage):r['condition']=name;r['dictionary_covered']=covered;r['operation_key']=key(p)
        records+=rows;summary[name]=gen.metrics(rows);io.write_jsonl(OUT/f'behavior/{name}.jsonl',rows);print(name,json.dumps(summary[name]),flush=True)
    bygroup={g:{c:gen.metrics([r for r in records if r['group']==g and r['condition']==c]) for c in summary} for g in grouped}
    io.save_json(OUT/'material/test_operations.json',[{'pair_id':p[0]['pair_id'],'operation':key(p),'covered':v} for p,v in zip(test,coverage)])
    result={'phase':2617,'timestamp':datetime.now().astimezone().isoformat(),'conditions':summary,'by_group':bygroup,'dictionary_entries':len(dictionary),'covered_test_pairs':sum(coverage),'wrong_operation_control_scopes':{k:wrong_scope.count(k) for k in set(wrong_scope)},
        'checks':{'all_240_train_pairs':len(train)==240,'all_120_test_pairs':len(test)==120,'all_600_greedy':len(records)==600,'frozen_dictionary_unchanged':io.sha256(OUT/'frozen_dictionary.npz')==checksum},
        'language_mechanism_closed':False,'claim':'conditional reuse under known lexical replacement operations, not an endogenous extractor of semantic operations'}
    result['all_checks_passed']=all(result['checks'].values());io.save_json(OUT/'analysis/final.json',result)
    # Publish complete coordinates for frozen representative operation directions; every group is included.
    asset=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json';payload=io.load_json(asset);rows=[]
    for g in grouped:
        k=next(k for k in index_keys if k.startswith(g+'/'))
        for li in (0,6,18,36):
            for name,arr in [('family_mean',means[g]),('specific_operation',dictionary[k])]:
                rows.append({'label':f'{g}/{name}/checkpoint{li}','source':'phase2617_training_dictionary','phase':2617,'layer':li,'coordinate_kind':'frozen_train_operation_direction','preview':True,'values':arr[li].tolist()})
    panel={'key':'phase2617_conditional_dictionary','model':'Qwen3-4B six-family training-only coordinate dictionary','precision':'BF16 fields, float32 training contrasts','coordinate_count':2560,'coordinate_semantics':'all original physical coordinates; operation identity specified externally; prototype differences can encode lexical answer identity','rows':rows}
    payload['models']=[p for p in payload['models'] if p['key']!=panel['key']]+[panel];payload['phase']=2617
    payload.setdefault('summary',{})['phase2617']={'dictionary_entries':len(dictionary),'covered_test_pairs':sum(coverage),'mechanism_closed':False}
    asset.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2617: 同目标自动续研——六族训练集条件坐标字典与600次生成比较 [{stamp}]

**原理、用例与公式。** 对全部六族×中英的240发现pair采集37×2560完整source坐标，冻结族均值与具体token替换操作条件均值。120外测只提供recipient及外部替换操作说明，不读取自身counterfactual HiddenState构造方向。所有族进入，而非只试一种时间句型。

$$\mu_{{g,l}}=\operatorname{{mean}}_{{i:g_i=g}}D_{{i,l}},\quad \mu_{{g,o,l}}=\operatorname{{mean}}_{{i:g_i=g,o_i=o}}D_{{i,l}},\quad H_{{5,S}}\leftarrow H_{{5,S}}+\mu_{{g,o,6}}.$$

baseline、族均值、条件字典、同位置等范数roll、不同操作的训练范数匹配方向共600次24-token无候选greedy。错操作优先同族同语言；某族只有一个操作key时使用跨族方向，范围逐项记录。操作key未见时退回族均值，逐条记录coverage，不隐瞒缺失。

**结果汇总。** 字典条目={len(dictionary)}；测试key覆盖={sum(coverage)}/120；{json.dumps(summary,ensure_ascii=False)}。

**相关文件。** `{OUT}`保存全部source原坐标、冻结字典哈希、600输出、coverage与12组结果；现有客户端增加各组族均值/具体操作方向96行×2560列。

**理论进展。** 该基础图谱区分“族名”与“给定具体词形操作”两种粒度；任何提升首先归于已知词形操作的跨上下文复用，不包装成自动理解语义。字典可积累坐标规律拼图，但不是最终破解算法。

**问题硬伤、结论。** 外测与发现的核心词形操作重叠，只有上下文前缀不同；key包含两侧替换token，外部操作说明有时直接携带答案身份，因此不是无答案信息的内生算子。词形键未见的退回策略不是泛化证明；同族错操作对照只匹配训练范数；source均值丢次序。保留全部阴性和失败材料，检查={json.dumps(result['checks'])}；语言机制未闭合。
'''
    if '## Phase 2617:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
