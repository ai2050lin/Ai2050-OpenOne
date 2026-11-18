"""Verify frozen archive results and append a GLM5 research phase."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import paradigm_archive_field as A
import paradigm_archive_native as N
import paradigm_path_core as P
from b36_semantic14_memo import append
from deploy.storage_policy import check_write


R,C,M,OUT=A.R,A.C,A.M,A.OUT
TITLE='等长档案的跨层格式路径、残差旁路与二阶融合留出'
MEMO=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'


def stat(values):
    a=np.asarray(values)
    if len(a)==0:
        return dict(n=0,mean=None,ci=[None,None])
    return A.X.V.summary(a)


def fmt(s):
    return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]" if s['n'] else '缺有效对照'


def above(s):
    return s['n'] and s['ci'][0]>0


def archive_effect(cross):
    meta={k:cross[k][::8] for k in ('family','cell','frame','bank')}
    expected=np.asarray(M.ARCV)
    assert (cross['variant'].reshape(-1,8,3)==expected).all()
    margin=cross['margin'][:,1].reshape(-1,2,2,2)
    diffs=[(np.take(margin,1,axis=axis)-np.take(margin,0,axis=axis)).mean((1,2)) for axis in (1,2,3)]
    a,w,p=diffs
    style=margin[:,1]-margin[:,0]
    aw=(style[:,1]-style[:,0]).mean(1)
    ap=(style[:,:,1]-style[:,:,0]).mean(1)
    groups={}
    for cell in M.TARGETS:
        select=meta['cell']==cell
        assert select.sum()==M.CROSS
        result={}
        categories=[('all',select)]
        for key in ('frame','bank'):
            categories.extend((key+str(value),select&(meta[key]==value)) for value in sorted(set(meta[key])))
        for name,keep in categories:
            result[name]={k:stat(value[keep]) for k,value in zip(('format','wording','punctuation','format_x_wording',
                                                              'format_x_punctuation'),(a,w,p,aw,ap),strict=True)}
            result[name]['format_positive']=int((a[keep]>0).sum())
        groups[f'{cell:04b}']=result
    return groups


def predictions(natural,holdout,tok):
    kl=natural['pred']
    assert kl.shape==(128,2,len(A.RULES)) and np.isfinite(kl).all()
    j=A.RULES.index(holdout['conditional_rule'])
    primary=A.RULES.index(holdout['rule'])
    indices=sorted({0,1,j,primary})
    result={}
    # For the aggregate keep the two cells together inside a family.
    for name,a in [('aggregate',kl.mean(1)),*((f'{cell:04b}',kl[:,i]) for i,cell in enumerate(M.TARGETS))]:
        conditional=stat(a[:,0]-a[:,j])
        universal=stat(a[:,1]-a[:,j])
        row=dict(methods={A.RULES[k]:stat(a[:,k]) for k in indices},gain_over_additive=conditional,
                 gain_over_pair=universal,conditional_passed=bool(above(conditional) and above(universal)))
        result[name]=row
    for i,cell in enumerate(M.TARGETS):
        y=kl[:,i]
        name=f'{cell:04b}'
        row=result[name]
        row['top1']={A.RULES[k]:float((natural['pred_top'][:,i,k+1]==natural['pred_top'][:,i,0]).mean())
                     for k in indices}
        row['always_worse']=int((y[:,1]>y[:,0]).sum())
        row['conditional_worse']=int((y[:,j]>y[:,0]).sum())
        row['rule_used']=int(sum(A.policy(cell,a,m)[j] for a,m in
                          zip(natural['variant'][::16,0],natural['pred_margin'][:,i],strict=True)))
        cases=[]
        for k in np.argsort(-(y[:,1]-y[:,0]))[:12]:
            original=16*int(k)+cell
            words=tok.batch_decode([[int(z)] for z in natural['pred_top'][k,i]])
            cases.append(dict(id=str(natural['ids'][original]),bank=int(natural['bank'][original]),
                frame=int(natural['frame'][original]),archive=int(natural['variant'][original,0]),
                additive=float(y[k,0]),pair=float(y[k,1]),conditional=float(y[k,j]),tokens=words))
        row['counterexamples']=cases
    return result


def load_causal():
    index=R.read('causal-files.json')
    binding=N.sha(N.RUNNER)
    assert index['holdout']==binding
    stacks={}
    for name in index['names']:
        receipt=R.read(Path(name).with_suffix('.json').name)
        assert receipt['holdout']==binding and receipt['sha']==N.sha(name)
        for k,v in A.contents(name).items():
            stacks.setdefault(k,[]).append(v)
    return {k:np.concatenate(v) for k,v in stacks.items()}


def causality(raw,holdout,candidate,tok):
    names=holdout['arms']
    assert names==list(N.ARMS)
    assert len(raw['ids'])==128 and len(set(raw['family']))==64
    assert (raw['kl'][:,:,[0,5]]==0).all()
    actual=raw['base']-raw['donor']
    direction=np.sign(actual)
    baseline=raw['base']
    labels=(raw['cell']==15).astype(int)
    fast=candidate['forecast']
    x=raw['features'][:,:,1]-raw['features'][:,:,0]
    means=np.asarray(fast['mean'])[labels]
    prediction=means+((x-np.asarray(fast['feature_mean'])[labels])/fast['scale'])@fast['coefficients']
    result=dict(views=len(raw['ids']),families=len(set(raw['family'])),exact=True,forecast={},routes={})
    for v,cell in enumerate(M.TARGETS):
        subset=labels==v
        assert subset.sum()==64
        b=np.square(means[subset]-actual[subset])
        f=np.square(prediction[subset]-actual[subset])
        result['forecast'][f'{cell:04b}']=dict(baseline_MAE=float(abs(means[subset]-actual[subset]).mean()),
            field_MAE=float(abs(prediction[subset]-actual[subset]).mean()),MSE_gain=stat(b-f),
            natural=stat(actual[subset]))
    early,late=sorted(holdout['routes'],key=lambda r:r['layer'])
    for ri,route in enumerate(holdout['routes']):
        route_name=route['site']
        shifts=direction[:,None]*(baseline[:,None]-raw['margin'][:,ri])
        s={k:shifts[:,i] for i,k in enumerate(names)}
        groups={}
        for cell in M.TARGETS:
            keep=raw['cell']==cell
            src=raw['dose'][:,ri,0]
            rcv=raw['dose'][:,ri,1]
            srows=keep&src[:,5].astype(bool)
            rrows=keep&rcv[:,5].astype(bool)
            total=float(np.abs(actual[keep]).mean())
            effect={k:stat(s[k][keep]) for k in ('source_kv','source_k','source_v','head_donor','skip_only','mlp_only')}
            both=effect['source_kv']
            source_coverage=P.dose_percentiles(src[keep])
            receiver_coverage=P.dose_percentiles(rcv[keep])
            match_source=stat((s['source_matched']-s['source_control_matched'])[srows])
            match_recv=stat((s['receiver_control_matched']-s['receiver_matched'])[rrows])
            signed_recv=stat((s['source_kv']-s['receiver_matched'])[rrows])
            restore_gain=stat((np.abs(s['source_kv'])-np.abs(s['receiver_matched']))[rrows])
            restore_control=stat((np.abs(s['receiver_control_matched'])-np.abs(s['receiver_matched']))[rrows])
            direction_near=rrows&(rcv[:,2:4]/np.maximum(rcv[:,8:10],1e-12)<=3).all(1)
            freezes={label:stat((s['source_kv']-s[label])[keep]) for label in ('freeze_g','freeze_u','freeze_gu','freeze_control')}
            closure=both['mean']/max(total,1e-10)
            natively_passed=bool(closure>=.10 and above(both) and above(match_source) and source_coverage['coverage']>=.98)
            receiver_passed=bool(above(restore_control) and above(restore_gain) and receiver_coverage['coverage']>=.98)
            delta=raw['receiver'][keep,ri,1]-raw['receiver'][keep,ri,0]
            controls=np.flatnonzero(keep&~(srows&rrows))
            ranked=np.flatnonzero(keep)[np.argsort(s['source_kv'][keep])]
            ids=[]
            for k in dict.fromkeys([*ranked[:4],*controls]):
                ids.append(dict(id=str(raw['ids'][k]),natural=float(actual[k]),
                      effects={op:float(s[op][k]) for op in ('source_kv','head_donor','skip_only','mlp_only')},
                      dose=raw['dose'][k,ri].tolist(),
                      words=tok.batch_decode([[int(i)] for i in [*raw['original'][k],raw['top'][k,ri,1]]])))
            relay={}
            if route_name==early['site']:
                relay={op:stat((s['source_kv']-s[op])[keep]) for op in ('block_later_head','block_later_gu')}
            groups[f'{cell:04b}']=dict(n=int(keep.sum()),natural_abs=total,raw=effect,closure=closure,
                source_control=match_source,receiver_directional_control=match_recv,selected_directional_attenuation=signed_recv,
                receiver_control=restore_control,selected_recovery=restore_gain,
                near_receiver_control=stat((np.abs(s['receiver_control_matched'])-
                                            np.abs(s['receiver_matched']))[direction_near]),
                source_dose=source_coverage,receiver_dose=receiver_coverage,proposed_receivers=freezes,
                source_pass=natively_passed,receiver_pass=receiver_passed,relay=relay,
                skip_mlp_interaction=stat((s['source_kv']-s['skip_only']-s['mlp_only'])[keep]),
                receiver_GUA_abs=dict(selected=np.abs(delta[:,:32]).mean((0,1)).tolist(),
                                      control=np.abs(delta[:,32:]).mean((0,1)).tolist()),
                source_flips=int((raw['top'][keep,ri,1]!=raw['original'][keep,1]).sum()),counterexamples=ids)
        result['routes'][route_name]=dict(layer=route['layer'],head=route['head'],control=route['control_head'],groups=groups)
    result['relay_order']=[f"B{r['layer']}/H{r['head']}@{r['site']}" for r in (early,late)]
    return result


def audit_field(discovery,tokenizer):
    stored=discovery['native_field']
    ids=discovery['native_field_ids']
    assert stored.shape[:2]==(2,41) and stored.shape[-1]==5120
    originals={r['id']:r for r in M.material(tokenizer,'discovery')[1]}
    r0,r1=(originals[k] for k in ids)
    assert r0['family']==r1['family'] and r0['cell']==r1['cell']
    i=min(r0['spans']['archive'])
    assert np.array_equal(stored[0,:,:i],stored[1,:,:i])
    trace=discovery['trace']
    assert np.isfinite(trace).all() and np.isfinite(discovery['map']).all()
    e=r0['query'][0]
    return dict(ids=ids.tolist(),native_shape=list(stored.shape),first_archive_position=i,onset=e,
                causal_prefix_identical=True,trace_shape=list(trace.shape),head_map_shape=list(discovery['map'].shape))


def interpretation(result):
    old=result['previous']
    effects=result['archives']
    pred=result['prediction']
    native=result['native']
    plan=result['design']
    holds=result['holdout']
    line=[
        f"- 运行 {M.RUN}，Qwen3-14B 本地原始 BF16，CPU 取权重、CUDA 单模型逐层执行，40 个块，批 8，右填充至 160。设计于 {plan['at']} 冻结，候选于 {result['candidate']['at']} 冻结，独立句式运行方案于 {holds['at']} 冻结。",
        f"- Phase 2972 对同一家族替换整个档案，1011 的要求格式间隔变化 {old['1011']:.4f}，1111 为 {old['1111']:.4f}。本轮把其当成待解释的输出效应。六套档案每个变体都占六个 token，每对共有相同正文、请求及读出位置。代码和论文档案均保留相同两词正文，词变体替换同义名词，前导标点 | / ; 独立互换。银行 4 固定引号、括号和正文，只变 Report/print 这个标记。自然论文与代码的模板仍有内部句法及标点差异。该最小银行检验字面线索是否足够。",
        '- 16 个试跑家族、24 个发现家族、128 个独立留出家族和 128 个只登记哈希的锁箱家族。发现与留出用不同名词和材料，留出有两套未见句式、三套未见档案。留出前 32 家族对 1011 和 1111 做完整的样式×措辞×标点交叉，前 64 家族做原生 K/V 配对干预。所有交叉比较按家族统计，99% 区间。新句式的持出范围仍为这两种人造句式。',
        '- 四项冻结成对行为门的最差格 '+
            '、'.join(f"{k} {v['minimum_stratum']:.0%}" for k,v in result['behavior'].items())+
            '。格式自由读首 token。语言、计数和括号的行为点评依赖固定续写前缀。',
        '- 全场在 40 块逐 token 先比较残差的原始投影及范数，再比较 G、U、A 的逐位置平均绝对变化；记录所有头在两个候选位置的 QK 概率质量及全位置 V/O 投影差。保存一对发现材料所有有效位置、41 个边界和全部 5120 个 BF16 残差坐标。档案之前的位置全层逐位相同。K/V 使用 RoPE 后原生键和原生值，候选读出是固定的 print−平均论文开头 unembedding 投影。全场热图见同一结果目录的 full-field.png。',
        '',
        r'$$H_{l,h,i,S}=W^O_{l,h} \sum_{j\in S} \mathrm{softmax}(Q_iK^\top/\sqrt{d})_j V_j,\quad '
        r'A_u=\mathrm{SiLU}(G_u)U_u,\quad r_{l+1}=r_l+\mathrm{Attn}(r_l)+W^\mathrm{down}A.$$',
        '',
        '- 候选按两个发现档案里的同方向性、两种目标组合的读出效应筛出。接收位点分别为自由起始位置与语法词末位。每个候选取局部写回投影相关的 32 个 G/U 单元，配一个冻结的同块随机组。后续在原生形状只替换命名头的聚合，分别交换档案 K/V、K、V 或整个头；逐一冻结接收端 G、U、G+U，原生整头恢复用相同位置安装原值。等幅对照按实际 BF16 残差范数选择，方向余弦至少 .90，相对幅度差最多 3%。',
        '- 检查残差旁路时，分别让变化只进入残差加法、只进入 MLP 写回。另从早期位置完整推进变动后缀，恢复晚期候选头或 G/U 为原值，观察早期作用是否衰减。局部阻断效应只代表这条截边对当前输入的作用，其他时点、注意力头、GQA 分支和残差读写仍能传递信号。',
        '',
        '| 留出组合与档案银行 | 家族 | 档案样式效应 | 措辞 | 前导标点 |',
        '| --- | ---: | --- | --- | --- |']
    for key,item in effects.items():
        for group in ('all','frame0','frame1','bank2','bank3','bank4'):
            a=item[group]
            line.append(f"| {key} {group} | {a['format']['n']} | {fmt(a['format'])} | {fmt(a['wording'])} | {fmt(a['punctuation'])} |")
    line.extend(['',
        f"- 1011 两句式里的样式效应分别为 {fmt(effects['1011']['frame0']['format'])} 和 "
        f"{fmt(effects['1011']['frame1']['format'])}。同一档案的方向会随请求上下文改变；"
        "第一句式在档案后给语言/格式，第二句式先给语言/格式。两句式还换了具体表达，所以这次不能把差异单独归因于位置。",
        f"- 1111 的最小词面银行交换效应 {fmt(effects['1111']['bank4']['format'])}，证明在这批材料里只换功能词也足以影响格式分数。"
        f"1011 的样式×前导标点效应 {fmt(effects['1011']['all']['format_x_punctuation'])}，"
        f"1111 为 {fmt(effects['1111']['all']['format_x_punctuation'])}。标点虽然短，仍能改变档案线索的有效强度。"])
    line.extend(['','### 二阶融合的冻结规则','',
        '- 在每个家族用未切换、单因子及双因子条件的全词表 logits 计算加法与二阶预测，目标格输出只用于 KL 评分。二阶项为 Iᵢⱼ=zᵢⱼ−zᵢ−zⱼ+z₀。候选规则包括总是加二阶、只对 1011 加、只对同向 1011 加、按预测前即可取得的 |加法格式 margin| ≤4 或 ≤8 加。发现阶段最差档案银行 KL 最低者被选为主规则。另于留出前冻结同一尺度下最好的有条件规则，要求它同时优于加法和总用二阶才过门。',
        f"- 主规则 {holds['rule']}，有条件规则 {holds['conditional_rule']}。两套规则在新材料首次模型前向前冻结。",
        '',
        '| 组合 | 加法 KL | 总用二阶 KL | 有条件规则 KL | 条件规则相对加法 / 总用二阶的改善 |',
        '| --- | ---: | ---: | ---: | --- |'])
    for key,item in pred.items():
        s=item['methods']
        apart=[s[x]['mean'] for x in ('add','pair',holds['conditional_rule'])]
        line.append(f"| {key} | {apart[0]:.5f} | {apart[1]:.5f} | {apart[2]:.5f} | "
                    f"{fmt(item['gain_over_additive'])} / {fmt(item['gain_over_pair'])} |")
        if key!='aggregate':
            c=item['counterexamples'][0]
            line.append(f"| {key} 最坏二阶反例 {c['id']} | {c['additive']:.5f} | {c['pair']:.5f} | {c['conditional']:.5f} | "
                        f"真实/Add/Pair 首词 {c['tokens'][0]!r}/{c['tokens'][1]!r}/{c['tokens'][2]!r} |")
    line.extend(['',f"- 1111 中总用二阶使 {pred['1111']['always_worse']}/128 条恶化，冻结条件规则加入二阶 "
                      f"{pred['1111']['rule_used']}/128 条。旧反例 57c765c3b3fb61d4de66-1111 和本轮全部分数都留档。",
        '', '### 逐边阻断与残差旁路', '',
        f"- 两站点依层序为 {' → '.join(native['relay_order'])}。64 个新家族、128 个代码档案接收样本，另有逐条论文档案原生供体。每条空操作和整头恢复在全词表 logits 上逐位相同。",
        '- 来源与旁路效应沿自然档案交换方向取号。越大越接近供体。闭合比例用平均干预效应除以同格自然档案平均绝对差。来源的等幅对照在 MLP 之前，接收单元的等幅对照在 MLP 之后。G/U 恢复另用距未干预分数的绝对距离计分，用同组的有向衰减作辅助读出。两种接受率和外推比例按格存档。',
        '',
        '| 位点、组合 | 档案 K+V | K 单独 / V 单独 | 整头替换 | K+V 闭合比例 | 残差单路 / MLP 单路 | 过门来源 / G/U |',
        '| --- | --- | --- | --- | ---: | --- | --- |'])
    for route,r in native['routes'].items():
        for cell,v in r['groups'].items():
            x=v['raw']
            line.append(f"| {route} B{r['layer']}/H{r['head']} {cell} | {fmt(x['source_kv'])} | "
                        f"{fmt(x['source_k'])} / {fmt(x['source_v'])} | {fmt(x['head_donor'])} | {v['closure']:+.2%} | "
                        f"{fmt(x['skip_only'])} / {fmt(x['mlp_only'])} | {v['source_pass']} / {v['receiver_pass']} |")
    line.extend(['','| 位点、组合 | 源／接收实际等幅覆盖 | 源等幅优势 | G 与 U 单独恢复 | G/U 等幅优势 | 阻断晚期头 / G/U 后损失 |',
                      '| --- | --- | --- | --- | --- | --- |'])
    for route,r in native['routes'].items():
        for cell,v in r['groups'].items():
            c=v['proposed_receivers']
            relay=v['relay']
            later=' / '.join(fmt(relay[k]) for k in ('block_later_head','block_later_gu')) if relay else '晚端'
            line.append(f"| {route} {cell} | {v['source_dose']['coverage']:.1%} / {v['receiver_dose']['coverage']:.1%} | "
                        f"{fmt(v['source_control'])} | {fmt(c['freeze_g'])} / {fmt(c['freeze_u'])} | "
                        f"{fmt(v['receiver_control'])} | {later} |")
    line.extend(['','### 结构解释与证据边界',''])
    for key,n in native['forecast'].items():
        line.append(f"- {key} 两条候选来源写入预测档案交换的 MAE {n['field_MAE']:.4f}，发现组同目标均值基线 {n['baseline_MAE']:.4f}，"
                    f"MSE 改善 {fmt(n['MSE_gain'])}。预测用的是这两处前向场，属于图谱读出，输入范围不同于只用文本的预测。")
    for route,r in native['routes'].items():
        for cell,v in r['groups'].items():
            c=v['counterexamples'][0]
            line.append(f"- {route} {cell} 的可回查反向端样本 {c['id']}，自然档案差 {c['natural']:+.4f}，"
                        f"K+V 移植沿自然方向的效应 {c['effects']['source_kv']:+.4f}，"
                        f"残差与 MLP 的非加和项 {fmt(v['skip_mlp_interaction'])}。")
    pocket=native['routes']['onset']['groups']
    line.append(f"- 自由起始位 1011 的残差单路 {fmt(pocket['1011']['raw']['skip_only'])}，MLP 单路 "
        f"{fmt(pocket['1011']['raw']['mlp_only'])}，出现相反方向。1111 的 MLP 单路 "
        f"{fmt(pocket['1111']['raw']['mlp_only'])}，原生 K/V 合路 {fmt(pocket['1111']['raw']['source_kv'])}。"
        "同一站点受到不同组合条件约束，局部旁路与 MLP 可以消减彼此的作用。")
    line.append(f"- 1011 自由起始位的等幅 G/U 操作有有向差异 "
        f"{fmt(pocket['1011']['receiver_directional_control'])}，实际回到未改输出的优势 "
        f"{fmt(pocket['1011']['receiver_control'])}。这区分了能拨动输出的单元与已经证实能恢复候选路径的单元。")
    aggregate=pred['aggregate']['conditional_passed']
    passes=[(k,c) for k,r in native['routes'].items() for c,v in r['groups'].items() if v['source_pass']]
    unit=[(k,c) for k,r in native['routes'].items() for c,v in r['groups'].items() if v['receiver_pass']]
    line.extend([
        f"- 条件融合规则跨目标是否同时优于两条基线 {'通过' if aggregate else '未通过'}。原生来源过门 {passes or '零格'}，"
        f"G/U 等幅恢复过门 {unit or '零格'}。完整摘要保存各位置来源质量、G/U 阈值差、匹配失败、负作用与所有 1111 的原预测。",
        '- 外部图谱新增可控制的档案样式、措辞、标点关系。内部图谱新增所有层和位置的响应场与两个可定位接收点。两图谱的关联由留出预测和原生截边逐项打分。现阶段对 RDC 的增量是可检验的条件化读写关系和旁路约束。局部 V/O 投影只是线性代数读出，较强的档案整体换文效应可以同时经多头、多位置和残差传播。完整机制还要在新的自然生成任务里预测效果大小、失效和其他能力副作用。',
        '- 下一大阶段把档案交换后的差分作为多输出目标，用不同数量和位置的固定头集合检验效应是否闭合，同时把原生残差携带的档案信号与后续 G/U 改写的新增信号分开验证。用多档案和多句式冻结一条带置信修正的乘法或分段规则，在新组合上同时比全加法和二阶。继续保留 1111 极端案例及有符号的未解释残差。',
        f"- 原始分片、冻结协议、候选、汇总和运行日志位于 tests/glm5/result/{OUT.name}。只清理本轮临时目录，原始场和必要结果均保留。"])
    return '\n'.join(line)+'\n'


def run(close):
    from transformers import AutoTokenizer

    design=R.read('design.json'); holdout=R.read(N.RUNNER); candidate=R.read('candidate.json')
    assert design['code']==A.code_identity() and design['old']==A.bind() and holdout['runner']==N.source_sha()
    assert holdout['candidate']==N.sha('candidate.json') and holdout['parent']==N.sha('design.json')
    assert (OUT/'candidate.json').stat().st_mtime_ns<(OUT/N.RUNNER).stat().st_mtime_ns
    assert all((OUT/N.RUNNER).stat().st_mtime_ns<(OUT/n).stat().st_mtime_ns
               for n in ('confirm-files.json','confirm-cross-files.json','causal-files.json'))
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    discovery=A.load('discovery-cross')
    old=json.loads((A.X.OUT/'analysis.json').read_bytes())
    history={k:old['causal']['heads']['25'][k]['text_archive_swap']['mean'] for k in ('1011','1111')}
    record=dict(at=C.stamp(),design=design,holdout=holdout,candidate=candidate,previous=history,
        behavior=R.read('behavior.json'),field=audit_field(discovery,tok),
        archives=archive_effect(A.load('confirm-cross')),prediction=predictions(A.load('confirm'),holdout,tok),
        native=causality(load_causal(),holdout,candidate,tok))
    R.write('analysis.json',record)
    body=interpretation(record)
    check_write(OUT/'report.md',len(body.encode())+4096,operation='留出报告写入').write_text(
        '# '+TITLE+'\n\n'+body,encoding='utf-8')
    if not close:
        return
    assert not (OUT/'memo-receipt.json').exists() and TITLE not in MEMO.read_text(encoding='utf-8-sig')
    memo=append(check_write(MEMO,len(body.encode())+4096,operation='追加 GLM5 研究记录'),TITLE,body)
    clean=[]
    if A.TMP.is_dir():
        path=A.TMP.resolve()
        assert path==A.TMP and path.parent==(R.ROOT/'tests/glm5_temp').resolve()
        group=[path,*path.rglob('*')]
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(path) for p in group)
        size=sum(p.stat().st_size for p in group if p.is_file())
        shutil.rmtree(path)
        clean.append(dict(path=str(path),freed_bytes=size))
    R.write('memo-receipt.json',dict(memo=memo,cleanup=clean))
    files={p.name:dict(sha256=A.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir())
           if p.is_file() and p.name!='manifest.json'}
    extras=(Path(__file__).resolve(),Path(N.__file__).resolve(),R.ROOT/'tests/glm5/paradigm_archive_visual.py')
    extra={p.relative_to(R.ROOT).as_posix():A.sha(p) for p in extras}
    R.write('manifest.json',dict(run=M.RUN,files=files,sources=design['code']|extra))
    print('GLM5 Phase',memo['phase'],'appended',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--close',action='store_true')
    args=parser.parse_args()
    A.setup()
    run(args.close)
