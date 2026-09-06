"""Basic coordinate bookkeeping and open-vocabulary changes, without dimensionality reduction."""
import json, sys
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';sys.path.insert(0,str(TESTS))
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as p8
OUT=TESTS/'result/phase2613_fullcoordinate_evidence_atlas';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'

def cosine(a,b):return np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-12)
def logsoftmax(x):
    x=x.astype('float64');x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))

def main():
    field=p8.OUT/'field';pairs=io.load_json(p8.OUT/'material/pairs.json');base=np.load(field/'greedy_prefill_baseline.float16.npy').astype('float32')
    conditions={};raw_centroids={}
    for c in p8.CONDITIONS:
        x=np.load(field/f'greedy_prefill_{c}.float16.npy').astype('float32');delta=x-base
        lp=logsoftmax(np.load(field/f'first_logits_{c}.float16.npy'));lb=logsoftmax(np.load(field/'first_logits_baseline.float16.npy'))
        kl=np.maximum(0,np.sum(np.exp(lb)*(lb-lp),axis=-1));tv=.5*np.sum(np.abs(np.exp(lp)-np.exp(lb)),axis=-1)
        conditions[c]={'rms_by_layer':np.sqrt(np.mean(delta**2,axis=(0,2))).tolist(),'first_vocab_kl_base_to_condition_mean':float(kl.mean()),
            'first_vocab_tv_mean':float(tv.mean()),'first_argmax_changed':float(np.mean(lp.argmax(-1)!=lb.argmax(-1))),
            'family_groups':{}}
        for g in sorted({p[0]['family']+'/'+p[0]['language'] for p in pairs}):
            idx=[i for i,p in enumerate(pairs) if p[0]['family']+'/'+p[0]['language']==g]
            cent=delta[idx].mean(0);raw_centroids[g+'/'+c]=cent
            conditions[c]['family_groups'][g]={'n':len(idx),'late_rms':float(np.sqrt(np.mean(delta[idx,25:]**2))),
                'first_vocab_kl':float(kl[idx].mean()),'first_vocab_tv':float(tv[idx].mean())}
    source=np.load(next(field.glob('source_span_means*')),mmap_mode='r').astype('float32')
    ds=source[:,1]-source[:,0];rms=np.sqrt(np.mean(ds**2,axis=-1))
    group_directions={g:ds[[i for i,p in enumerate(pairs) if p[0]['family']+'/'+p[0]['language']==g]].mean(0) for g in sorted({p[0]['family']+'/'+p[0]['language'] for p in pairs})}
    groups=list(group_directions);stack=np.stack(list(group_directions.values()));common=stack.mean(0)
    # Always retain original D coordinates. Centering is a separate descriptive view.
    reuse=np.stack([[cosine(a,b) for b in stack] for a in stack]);center=stack-common
    residual_reuse=np.stack([[cosine(a,b) for b in center] for a in center])
    fp=OUT/'field';fp.mkdir(parents=True,exist_ok=True)
    np.savez(fp/'all_coordinate_centroids.npz',**raw_centroids)
    np.savez(fp/'source_family_atlas.npz',groups=np.array(groups),raw=stack,centered=center,reuse=reuse,centered_reuse=residual_reuse)
    # Magnitude strata are summaries of all coordinates, not a selection used by the analysis.
    strata=[]
    for layer in range(37):
        cuts=np.quantile(np.abs(source[:,0,layer]),[.25,.5,.75],axis=-1)
        contributions=[]
        for i in range(120):
            labels=np.searchsorted(cuts[:,i],np.abs(source[i,0,layer]),side='right')
            energy=ds[i,layer]**2;total=float(energy.sum())
            contributions.append([float(energy[labels==q].sum())/(total+1e-12) for q in range(4)])
        strata.append(np.mean(contributions,axis=0).tolist())
    result={'phase':2613,'timestamp':datetime.now().astimezone().isoformat(),'conditions':conditions,
        'source_delta_rms_median_by_checkpoint':np.median(rms,axis=0).tolist(),'source_delta_energy_by_baseline_magnitude_quartile':strata,
        'checks':{'all12conditions':len(conditions)==12,'all120pairs':len(pairs)==120,'all37x2560coords':base.shape==(120,37,2560),
        'allfinite':all(np.isfinite(v).all() for v in raw_centroids.values()),'embedding_answer_delta_zero':all(conditions[c]['rms_by_layer'][0]==0 for c in conditions)},
        'language_mechanism_closed':False}
    result['all_checks_passed']=all(result['checks'].values());io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2613: 六族全坐标复用、低幅值能量与开放词表变化的基础图谱 [{stamp}]

**测试原理、用例与公式。** 使用Phase2608全部120 pair×12条件×37层×2560坐标，以及每条件完整151936词表首步logits。逐坐标计算差分与均值，保留原始坐标顺序；族均值和减公共均值分别显示，不将去均值称语义剥离。

$$D_{{i,l,j}}=H^{{patch}}_{{i,l,j}}-H^{{base}}_{{i,l,j}},\quad R_l=\sqrt{{\frac1{{ND}}\sum_{{i,j}}D_{{i,l,j}}^2}},\quad TV(p,q)=\tfrac12\sum_v|p_v-q_v|.$$

额外遍历所有坐标，按baseline幅值四等分描述差分能量分配；所有四组均保留，不以最大组解释系统。词表KL/TV使用全词表，避免把两个候选的变化当全部输出变化。

**结果汇总。** 逐条件RMS、逐族开放词表变化、12×12×37 raw/中心化复用与37×4幅值分层见final与npz；embedding边界差分检查={result['checks']['embedding_answer_delta_zero']}。末检查点低值四分位能量分配={strata[-1]}。这里source最后检查点是final-normalized状态，greedy prefill则是raw block35，二者不混用。

**相关文件。** `{OUT}`中的完整2560列族均值npz与final；原始数据在Phase2608。本脚本不加载模型、不降维，也不按输出效果筛坐标。

**分析、理论进展。** 将外部操作、source物理状态、后续重复层读出、开放词表和生成分别记账；规律候选必须同时标注语言、表述、位置和输出协议，不由一个标量相关值代替。

**问题硬伤、结论。** 层间残差加法与词表softmax只是模型已有结构的恒等记账，不是新语言数学；KL大也可能只是生成受损。分层低值能量不是低值坐标必要性的证明；中心化不是语义纯化。检查={json.dumps(result['checks'])}；不提出超出数据的高级理论。
'''
    if '## Phase 2613:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(result['checks']),flush=True)

if __name__=='__main__':main()
