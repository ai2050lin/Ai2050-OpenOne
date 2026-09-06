"""Same scientific target, explicit chat/no-thinking interface calibration, not a new lockbox."""
import copy,json,sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';sys.path.insert(0,str(TESTS))
import model_utils
import phase2603_c643329_c659712_unique_natural_lockbox as mat
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as greedy
import phase2609_c741633_c758016_discovery_only_operator_transfer as transfer
OUT=TESTS/'result/phase2615_chat_interface_operator_confirmation';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'

def wrap(tok,pairs):
    pairs=copy.deepcopy(pairs)
    for pair in pairs:
        for row in pair:
            row['prompt']=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            row['prompt_ids']=tok.encode(row['prompt'],add_special_tokens=False)
        for row,pos in zip(pair,mat.source_spans(pair[0]['prompt_ids'],pair[1]['prompt_ids'])):row['source_token_positions']=pos
    return pairs

def main():
    model,tok,_=model_utils.load_model('qwen3',dtype=torch.bfloat16,use_8bit=False)
    train=wrap(tok,transfer.build(tok,'discovery'));test=wrap(tok,transfer.build(tok,'heldout'))
    io.save_json(OUT/'material/train.json',train);io.save_json(OUT/'material/test.json',test)
    io.OUT=OUT/'discovery';f=np.load(io.collect_source_means(model,tok,train,storage_dtype=np.float32));d=f[:,1]-f[:,0]
    means={g:np.mean(d[[i for i,p in enumerate(train) if p[0]['language']==g]],axis=0) for g in ('en','zh')}
    np.savez(OUT/'frozen_train_directions.npz',**means);sha=io.sha256(OUT/'frozen_train_directions.npz')
    io.OUT=OUT/'heldout';f=np.load(io.collect_source_means(model,tok,test,storage_dtype=np.float32));oracle=f[:,1]-f[:,0]
    learned=np.stack([means[p[0]['language']] for p in test]);records=[];summaries={}
    configs=[('baseline','baseline',5,learned[:,6]),('natural_donor','baseline',5,learned[:,6])]
    for l in (0,2,5,11,17):
        configs.extend([(f'learned_l{l}','source1',l,learned[:,l+1]),(f'roll_l{l}','source_roll',l,learned[:,l+1]),(f'oracle_l{l}','source1',l,oracle[:,l+1])])
    # Cross-language norm matching is determined by training means only, not held-out donors.
    cross=[]
    for p in test:
        lang=p[0]['language'];other='zh' if lang=='en' else 'en'
        vec=means[other][6]*np.linalg.norm(means[lang][6])/(np.linalg.norm(means[other][6])+1e-12);cross.append(vec)
    configs.append(('cross_normmatched_l5','source1',5,np.stack(cross)))
    for name,kind,l,vec in configs:
        pairs=[list(reversed(p)) for p in test] if name=='natural_donor' else test
        rows=greedy.generate_condition(model,tok,pairs,kind,vec,{},source_layer=l,max_new_tokens=64)
        for row,pair in zip(rows,test):
            row['condition']=name;row['language']=pair[0]['language'];row['form']=pair[0]['form']
            row['eos_generated']=tok.eos_token_id in row['generated_ids']
        records+=rows;summaries[name]=greedy.metrics(rows);summaries[name]['eos_rate']=float(np.mean([r['eos_generated'] for r in rows]))
        io.write_jsonl(OUT/f'behavior/{name}.jsonl',rows);print(name,json.dumps(summaries[name]),flush=True)
    groups={f'{lang}/form{form}':{c:greedy.metrics([r for r in records if r['condition']==c and r['language']==lang and r['form']==form]) for c in summaries} for lang in ('en','zh') for form in range(3)}
    result={'phase':2615,'timestamp':datetime.now().astimezone().isoformat(),'conditions':summaries,'by_surface':groups,
        'checks':{'all2160_generations':len(records)==2160,'frozen_directions_unchanged':io.sha256(OUT/'frozen_train_directions.npz')==sha},
        'language_mechanism_closed':False,'independent_replication':False}
    result['all_checks_passed']=all(result['checks'].values());io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2615: 同目标自动续研——chat接口下跨事件算子、五层与训练等范数迁移 [{stamp}]

**续研原因。** Phase2609的原始续写常先回答再自我修正；64-token规范chat/no-thinking用于核验“方向不会用”与“接口没有结束答案”的混杂。目标仍是跨事件相对编码，因此直接进入下一轮，不把原始续写失败当路线结束。

**原理、用例与公式。** 保留40训练pair、120测试pair及词表隔离，重用测试材料故不是独立复现。chat模板包裹原文本，重新确定source位置；训练冻结各层全2560坐标均值。baseline与自然donor；层0/2/5/11/17的训练方向、roll、oracle；层5跨语言方向按训练均值范数匹配，共2160次、上限64 token真实生成。浅层0/2在本Phase运行前加入，原因是Phase2609同词形方向高度相似但source层5 oracle也不转向，需要检验信息是否已向其他位置传播，不能只看中层source。

$$\delta^{{cross}}_l=\mu_{{other,l}}\frac{{\|\mu_{{same,l}}\|}}{{\|\mu_{{other,l}}\|}},\qquad \mu\text{{只来自训练集}}.$$

**结果汇总。** {json.dumps(summaries,ensure_ascii=False)}

**相关文件。** `{OUT}`保存chat完整材料、训练与测试全部source坐标、冻结方向哈希、2160输出含EOS信息、逐语言/表面和final。

**低值存储校准。** 本Phase模型仍BF16非量化，但source场改用FP32保存实际BF16数值，避免FP16导出把极小值变零；每份场另记录FP16反事实导出的下溢数、溢出数与最大误差。FP32存储不等于FP32模型推理，旧FP16导出不应称bit-exact。

逐次记录预期位移与BF16加法后的实际位移范数；向量roll仅保证施加前数学范数一致，不保证舍入后的实际位移精确等量，二者不混称。

**理论进展与分析。** 若接口使自然成功恢复而学习方向仍不能稳定转向，则条件化编码不能由单一语言均值充分概括；oracle只用于可干预性诊断。跨语言等范数比较仅排除训练方向幅度一种混杂。

**问题硬伤、结论。** 五层探索和重用测试材料不能包装成新锁箱；位置由实验者给定；每语言只有20基础事件对；跨语言操作仅时间先后。保持全坐标观察与机制验证两套账，检查={json.dumps(result['checks'])}；机制未闭合。
'''
    if '## Phase 2615:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
