"""CPU-only correction register and evidence dependency index. Legacy files immutable."""
import contextlib
import io
import itertools
import json
from statistics import NormalDist
from pathlib import Path
import numpy as np
from phase2751_trusted_rebuild import ROOT,OUT,OLD,write,sha
import audit_gpt5_memo_20260923 as audit


def main():
    dest=OUT/'evidence';dest.mkdir(parents=True,exist_ok=True)
    # Re-run corrected rank/block statistics under a new versioned output directory.
    audit.OUT=dest
    with contextlib.redirect_stdout(io.StringIO()):
        audit.continuum();audit.mathematical_checks()
    nd=NormalDist();transforms=[]
    for phase in [3081,3085,3087,3089]:
        candidates=sorted((OLD/f'phase{phase}').glob('*/*.npz'))
        p=next(p for p in candidates if 'arbitration' in str(p)) if phase!=3081 else candidates[0]
        with np.load(p) as z:
            for stat,pair in itertools.product(['T','U'],['AB','AC','BC']):
                r=float(z[f'E3_F2_CTT_{stat}_{pair}']);pv=float(z[f'E3P_F2_CTT_{stat}_{pair}'])
                clipped=max(1e-12,min(1-1e-12,pv))
                transforms.append(dict(phase=phase,test=stat+'_'+pair,r=r,inherited_p=pv,
                    corrected_signed_z=float(np.sign(r)*nd.inv_cdf(1-clipped/2)),
                    original_signed_z=float(np.sign(r)*nd.inv_cdf(1-clipped)),input_sha256=sha(p)))
    write(dest/'p3092_transform_correction.json',dict(rows=transforms,
        aggregate_p=None,aggregate_status='not_identified_without_joint_null',
        note='Corrected z conversion only. Inherited p values and Spearman values are NOT revalidated here. No valid combined significance is asserted.'))
    p=next((OLD/'phase3099').glob('*/*.npz'));jrows=[]
    with np.load(p) as z:
        for side in ['4B','14B']:
            for fa,fb in itertools.combinations('ABC',2):
                vals=[]
                for ci in [1,2,3]:
                    a=set(z[f'TOP64_{side}_{fa}_ci{ci}'].tolist());b=set(z[f'TOP64_{side}_{fb}_ci{ci}'].tolist())
                    val=len(a&b)/len(a|b);saved=float(z[f'J64_{side}_{fa+fb}'][ci-1])
                    assert abs(val-saved)<1e-12
                    vals.append(val)
                jrows.append(dict(side=side,pair=fa+fb,formal=vals[0],shakespeare=vals[1],topic=vals[2],
                    style_minus_formal=vals[1]-vals[0],
                    note='Descriptive paired difference, bounded [-1,1]. Fixed aggregate top sets do not support sample bootstrap.'))
    write(dest/'p3099_gate_replacement.json',dict(rows=jrows,input_sha256=sha(p),
        old_gate='invalid feasibility at observed control; withdrawn',
        replacement='Report matched same-model same-pair J(style)-J(formal), no posthoc binary confirmation.',
        inference_status='Set identity absent/present mechanism remains unresolved. Need independent group-level sets.'))
    specs=[
      ('RDC-M01',[3100],'SwiGLU一阶微分','invalid_legacy_formula','用正确乘积法则、自动微分、有限差分与真实模型工作点校验替代。',['RDC-M02','RDC-T01']),
      ('RDC-M02',[3100],'残差96.5%能量解释','withdrawn','范数份额不能互补；报告交叉项及可为负的投影贡献。',['RDC-T01']),
      ('RDC-M03',[3093,3100],'自然头与因果头解耦','invalid_coordinate_comparison','只比较同一投影前头空间；后投影坐标切片不作头编号。',['RDC-T01']),
      ('RDC-M04',[3075,3076],'Möbius次模等价定理','false_theorem','保留集合函数和真实条件二阶差分；撤回错误等价及直接高阶协作解释。',['RDC-T02']),
      ('RDC-S01',[3086,3088,3091],'跨架构连续统确认','downgraded_descriptive','平均秩与模型块置换，纳入14B；模型总体仍未随机抽样。',['RDC-T01']),
      ('RDC-S02',[3092],'带符号Stouffer确认','aggregate_unidentified','正确双侧p到z转换；缺少联合null时不输出合并显著性。',['RDC-S01']),
      ('RDC-S03',[3099],'方向编码排除集合编码','withdrawn_exclusive_claim','替换不可行三倍门为匹配差值；描述差异不自动构成机制排他性。',['RDC-T01']),
      ('RDC-O01',[2751,2752,2753],'受限训练参数子集','legacy_candidate_not_independently_rerun','保留历史局部实验候选，当前不升级为已独立复现。',[]),
      ('RDC-O02',[2807,2809],'有限类别未见词预测','legacy_candidate_not_independently_rerun','剥离恒等式解释，保留已报告分类任务范围。',[]),
      ('RDC-O03',[2987],'上下文改变导致签名失效','retained_counterexample','这是适用范围约束；不能把未测卡片一并判失效。',[]),
      ('RDC-O04',[3074,3075],'集合边际非加性','recomputed_frozen_observation','正确重算467/1792条件二阶差分>0.02；噪声显著性未确证。',['RDC-T02']),
      ('RDC-O05',[3098,3099],'末块MLP局部作用','legacy_candidate_with_scope','分解与干预可作局部证据；读出桥接和排他性分开验证。',['RDC-T01']),
      ('RDC-T01',[3098,3099,3100],'RDC完整上游至读出机制链','incomplete','受M01/M02/M03/S03影响；不得以某一局部修复宣称整链完成。',[]),
      ('RDC-T02',[3074,3075,3076],'竞争—完备性普适定律','hypothesis_only','跨族参数不稳且数学解释有误；保留条件集合函数作后续输入。',[]),
    ]
    snapshot=ROOT/'tests/glm5/result/gpt5_memo_audit_20260923/memo_snapshot.md'
    source_lines=snapshot.read_text(encoding='utf-8').splitlines()
    records=[]
    for ident,phases,title,status,reason,dependents in specs:
        sources=[]
        for phase in phases:
            for file in sorted((ROOT/'tests/glm5').glob(f'phase{phase}_*.py')):
                sources.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
        # These are explicitly curated dependencies, not every textual reference.
        records.append(dict(id=ident,phases=phases,title=title,status=status,reason=reason,
            downstream_claim_ids=dependents,sources=sources,
            memo_lines=[n for n,line in enumerate(source_lines,1) if any(line.startswith(f'## Phase {p}:') for p in phases)]))
    write(dest/'claims.json',dict(schema='rdc-evidence-register-v1',claims=records,
        snapshot_sha256=sha(snapshot),rules=['Numerical identity is distinct from empirical generalization.',
            'Legacy candidates are not promoted to independently validated by this register.',
            'Any complete-chain claim requires all required dependencies validated within the same scope.'],
        scope='Curated critical dependencies for audited GPT5 phases, not an exhaustive whole-repository dependency graph.'))
    write(OUT/'index.json',dict(run_id='rdc_trusted_rebuild_20260923',status='measurement_repair_and_validation_in_progress',
        material='material.json',design='preregistered_design.json',claims='evidence/claims.json',
        models={s:{'capture':(OUT/s/'capture_done.json').exists(),'measurement':(OUT/s/'measurement_repair.json').exists(),
                    'composition':(OUT/s/'composition.json').exists()} for s in ['4B','14B']},
        trusted_scope='See per-claim status and actual model results. No AGI or universal language theorem claimed.'))
    print(json.dumps(dict(claims=len(records),stouffer_rows=len(transforms),jaccard_pairs=len(jrows)),ensure_ascii=False))


if __name__=='__main__':main()
