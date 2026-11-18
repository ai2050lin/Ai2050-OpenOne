"""Audit frozen outcomes, retain failing cases, and append the GLM5 research memo."""

import argparse
from collections import defaultdict
import shutil

import numpy as np

import paradigm_path_run as X
import paradigm_path_core as P
import paradigm_path_material as M
from b36_semantic14_memo import append
from deploy.storage_policy import check_write


R,C,OUT=X.R,X.C,X.OUT
TITLE='四因子融合复测与 B30 两条头到神经元路径的逐头因果检验'
MEMO=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'


def summary(values):
    values=np.asarray(values)
    if len(values)==0:
        return dict(n=0,mean=None,ci=[None,None])
    return X.V.summary(values)


def positive(stat,minimum=0):
    return stat['n']>0 and stat['mean']>=minimum and stat['ci'][0]>0


def fusion(data,tokenizer):
    scores=data['fusion']
    family=data['family'][::16]
    archive=data['archive'][::16]
    natural=np.r_[True,family[1:]!=family[:-1]]
    paired=set(family[~natural])
    assert natural.sum()==128 and len(paired)==32
    assert all(set(archive[family==key])=={0,1} for key in paired)
    all_paired=np.array([key in paired for key in family])
    output={}
    for j,target in enumerate(M.TARGET):
        z=scores[:,j]
        gain=z[:,1,0]-z[:,2,0]
        result={}
        style=(target>>2)&1
        for name,keep in (('natural',natural),('natural_aligned',natural&(archive==style)),
                          ('natural_conflict',natural&(archive!=style)),
                          ('swap_aligned',all_paired&(archive==style)),
                          ('swap_conflict',all_paired&(archive!=style))):
            g=summary(gain[keep])
            g['relative']=float(g['mean']/max(z[keep,1,0].mean(),1e-10))
            g['passed']=bool(g['relative']>=.10 and positive(g))
            result[name]=dict(gain=g,base=summary(z[keep,0,0]),additive=summary(z[keep,1,0]),
                              pairwise=summary(z[keep,2,0]),worsened=int((gain[keep]<0).sum()),
                              top1=z[keep,:,1].mean(0).tolist(),
                              omit_pair=[summary(z[keep,k,0]-z[keep,2,0]) for k in range(3,z.shape[1])])
        paired_gain={int(a):{key:float(g) for key,g in zip(family[all_paired&(archive==a)],
                      gain[all_paired&(archive==a)],strict=True)} for a in (0,1)}
        result['swap_gain_aligned_minus_conflict']=summary([paired_gain[style][f]-paired_gain[1-style][f] for f in sorted(paired)])
        cases=[]
        for k in np.argsort(gain)[:12]:
            row=16*int(k)+target
            cases.append(dict(id=str(data['ids'][row]),archive=int(archive[k]),additive=float(z[k,1,0]),
                pairwise=float(z[k,2,0]),gain=float(gain[k]),tokens=tokenizer.batch_decode(
                    [[int(i)] for i in data['fusion_top'][k,j,:4]])))
        result['counterexamples']=cases
        output[f'{target:04b}']=result
    return output


def paths(data,tokenizer):
    top=data['edit_top']
    assert np.array_equal(top[:,0],data['top'][:,0]) and np.array_equal(top[:,0],top[:,1])
    assert (data['edit_kl'][:,:2]==0).all()
    margin=data['edit']
    baseline=margin[:,0]
    result=dict(views=len(top),repaired_views=int((top[:,0]==top[:,1]).sum()),heads={})
    for i,head in enumerate(P.HEADS):
        effects={name:margin[:,X.ARMS.index(f'{head}/{name}')]-baseline for name in X.NATIVE}
        decision={}
        for target in M.TARGET:
            style=(target>>2)&1
            sign=2*style-1
            before={}
            for condition,a in (('aligned',style),('conflict',1-style)):
                rows=(data['cell']==target)&(data['archive']==a)
                assert rows.sum()==64 and len(set(data['family'][rows]))==64
                source,restore=[data['dose'][:,i,k] for k in range(2)]
                ok_source=rows&source[:,5].astype(bool)
                ok_restore=rows&restore[:,5].astype(bool)
                source_dose=P.dose_percentiles(source[rows])
                restore_dose=P.dose_percentiles(restore[rows])
                start=effects['archive']
                native_distance={op:summary((np.abs(start)-np.abs(effects[op]))[rows])
                                 for op in ('gate_restore','up_restore','both_restore','random_restore')}
                stats=dict(full_head=summary(sign*effects['zero'][rows]),
                     archive=summary(sign*start[rows]),request_loss=summary(-sign*effects['request'][rows]),
                     joint_nonadd=summary(sign*(effects['joint']-start-effects['request'])[rows]),
                     match_source=summary(sign*(effects['source_matched']-effects['control_matched'])[ok_source]),
                     native_distance_gain=native_distance,
                     matched_selected_distance_gain=summary((np.abs(start)-np.abs(effects['both_matched']))[ok_restore]),
                     matched_selected_over_random=summary((np.abs(effects['random_matched'])-
                                                           np.abs(effects['both_matched']))[ok_restore]),
                     source_dose=source_dose,restore_dose=restore_dose)
                unit_change=data['gua'][rows,2+4*i]-data['gua'][rows,0]
                stats['archive_GUA_abs']={name:np.abs(unit_change[:,block]).mean((0,1)).tolist()
                                           for name,block in (('selected',slice(32)),('random',slice(32,64)))}
                stats['first_unit_archive_change']=unit_change[:,0].mean(0).tolist()
                stats['mean_attention_mass']=dict(zip(X.F.M.REGIONS,data['attention_mass'][rows,i].mean(0).tolist()))
                stats['mean_native_write_norm']=dict(zip(('zero','archive','request','joint'),
                                                         data['attention_norm'][rows,i].mean(0).tolist()))
                ratios=restore[:,2:4]/np.maximum(restore[:,8:10],1e-12)
                near=ok_restore&(ratios<=3).all(1)
                stats['near_native_selected_over_random']=summary((np.abs(effects['random_matched'])-
                    np.abs(effects['both_matched']))[near])
                stats['top1_flips']={op:int((top[rows,X.ARMS.index(f'{head}/{op}')]!=top[rows,0]).sum()) for op in X.NATIVE}
                stats['competition']=bool(condition=='conflict' and positive(stats['archive'],.10) and
                    positive(stats['request_loss'],.10) and source_dose['coverage']>=X.A.TECHNICAL_MIN and
                    positive(stats['match_source']))
                stats['direction_specific_rescue']=bool(restore_dose['coverage']>=X.A.TECHNICAL_MIN and
                    positive(stats['matched_selected_distance_gain']) and positive(stats['matched_selected_over_random']))
                cases=[]
                selections=np.flatnonzero(rows)
                ordered=selections[np.argsort(sign*start[rows])]
                dose_fail=selections[~(ok_source&ok_restore)[selections]]
                for index in dict.fromkeys([*ordered[:4],*dose_fail]):
                    word_indices=[0]+[X.ARMS.index(f'{head}/{op}') for op in ('archive','request','both_restore','random_restore')]
                    cases.append(dict(id=str(data['ids'][index]),effects={op:float(e[index]) for op,e in effects.items()},
                        dose=data['dose'][index,i].tolist(),tokens=tokenizer.batch_decode([[int(top[index,k])] for k in word_indices])))
                stats['counterexamples']=cases
                before[condition]=stats
            aligned=(data['cell']==target)&(data['archive']==style)
            conflict=(data['cell']==target)&(data['archive']!=style)
            assert np.array_equal(data['family'][aligned],data['family'][conflict])
            before['text_archive_swap']=summary(sign*(baseline[aligned]-baseline[conflict]))
            before['source_contrast_conflict_minus_aligned']=summary(sign*(effects['archive'][conflict]-effects['archive'][aligned]))
            decision[f'{target:04b}']=before
        result['heads'][str(head)]=decision
    return result


def show(stat):
    lo,hi=stat['ci']
    return f"{stat['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]" if stat['n'] else '缺有效配对'


def render(plan,cal,behavior,prediction,causality):
    cases=cal['recorded_failures']
    lines=[
        f"- 本地 Qwen3-14B，原生 BF16，全量 40 个块按层串行送入同一张 CUDA 卡。运行 {plan['run']}，冻结 {plan['at']}。确认用新名词和新句式，按名词及初始/修订计数组成的家族切分。16 个试跑家族、128 个确认家族、128 个锁箱家族。锁箱只有文本身份哈希，未执行模型。",
        f"- 旧前向采 8 个家族共 128 条，复现 6 个原失配，首个差异均在 B30。把单查询/批量干预投影换成原序列形状后，每条空操作 logits 逐位相同。单独切短注意力 query 或改变本次读出矩阵形状未复现这 6 条的相差。旧操作在 B30 的 K/V 就出现舍入差异，不能把它归因于某个 CUDA 内核或只归到单一算子。失配原始 token 与差值逐条见 calibration.json。",
        '- 失配家族与旧→修正后的先决问题逐条保留。',
        '',
        '| 旧样本 | 原全序列首 token | 旧空操作 | 修正后 | 原 top1−top2 | B30 K/V 最大差 |',
        '| --- | --- | --- | --- | ---: | --- |']
    for v in cases:
        first,old,new=v['words']
        d=v['qkv_and_sliced_attention']
        lines.append(f"| {v['id']} | {first} | {old} | {new} | {v['onset_native_gap']:.3f} | {d[1]:.5f} / {d[2]:.5f} |")
    lines.extend([
        '',
        f"- 旧恢复位移均值选中/随机为 {cal['previous_native_means'][0]:.3f}/{cal['previous_native_means'][1]:.3f}，相对差 P95={cal['previous_rescue_p95']:.2%}。恢复同样数量的单元会产生不同写入，单元数量不能校准剂量。第一版缩到较弱方向的规则只覆盖 25%–58%，舍入会让弱位移变为零或改变方向。",
        '- 保留原生全幅阻断和恢复，再单独做定幅对照。方向 d 取原生干预在注入点真正写入 BF16 残差的差值。仅在该差值为零时以原模型 O 投影列或 down_proj 列的 FP32 代数写入作为备用方向，并保留两种度量。定义 δ(s)=BF16(r+s·d)−r，其中 r 为注入点原生残差。要求两臂实际 ‖δ‖₂ 相差 ≤3%、均 ≥.003、与各自 d 的余弦均 ≥.90。在较强自然写入的 .5–8 倍中搜可达共同剂量，优先靠近 1 倍。剂量选择只读内部写入。H20 对照是不同 KV 组的同块单个头。匹配时可能外推弱方向，恢复另报告两臂均在 3 倍以内的敏感性检查。',
        '- 融合用实际全词表分布检验。单项响应为 Δᵢ=zᵢ−z₀，二阶交互 Iᵢⱼ=zᵢⱼ−zᵢ−zⱼ+z₀。对多因子目标预测 ẑ=z₀+ΣΔᵢ+ΣIᵢⱼ，用 KL(p原模型‖softmax(ẑ)) 计分。拟合用不到目标格输出，需要同一家族低阶材料的模型响应。竞争用原生阻断验证，格式读出为 m=logit(print)−logsumexp(logits(论文开头候选))，按要求格式取正负号。阻断档案取值保留 QK，只在原序列末位安装该头新的 V 聚合输出；阻断请求同理。部分恢复直接替换选中 G、U 或二者，重新执行下游块。',
        '- 试跑行为门按 8 个固定的其余因子组合分别检验，同一条件每格 16 配对。最低正确配对率 '+
            '、'.join(f"{k} {v['minimum_stratum']:.1%}" for k,v in behavior.items())+'。三处读出依赖规定的续写前缀。',
    ])
    if prediction:
        lines.extend(['','| 目标及参考条件 | 家族 | 加法 KL | 二阶 KL | 改善 99% CI | 变差条数 |',
                      '| --- | ---: | ---: | ---: | --- | ---: |'])
        for target,item in prediction.items():
            for setting in ('natural','natural_aligned','natural_conflict','swap_aligned','swap_conflict'):
                v=item[setting]
                lines.append(f"| {target} {setting} | {v['gain']['n']} | {v['additive']['mean']:.4f} | {v['pairwise']['mean']:.4f} | {show(v['gain'])} | {v['worsened']} |")
        for target,item in prediction.items():
            case=item['counterexamples'][0]
            tokens=case['tokens']
            lines.append(f"- {target} 最大二阶恶化 {case['id']}，加法 {case['additive']:.4f} → 二阶 {case['pairwise']:.4f}。真实/Add/Pair 首 token 分别是 {tokens[0]!r}、{tokens[2]!r}、{tokens[3]!r}。配对参考交换后的同向−冲突增益 {show(item['swap_gain_aligned_minus_conflict'])}。")
            gain=item['natural']['gain']
            split=[setting for setting in ('natural_aligned','natural_conflict','swap_aligned','swap_conflict')
                   if item[setting]['gain']['passed']]
            lines.append(f"- {target} 的自然档案平均相对收益 {gain['relative']:+.1%}，冻结融合门"
                         f"{'通过' if gain['passed'] else '未通过'}；分层通过范围 {', '.join(split) or '无'}。"
                         f"自然档案二阶预测首 token 命中率 {item['natural']['top1'][2]:.1%}。")
    elif not all(v['passed'] for v in behavior.values()):
        lines.append('- 联合行为门未通过，按冻结规则停止新确认材料的融合预测。')
    if causality:
        lines.extend(['',f"- 逐头干预 {causality['views']} 个视图，空操作与整头原生恢复在所有视图的全词表 logits 上逐位一致。每格/档案方向以 64 个构造家族做簇 bootstrap，区间置信度 99%。单一名词库和句式仍限制外推。",
            '',
            '| 头 / 组合 / 档案 | 整头阻断 | 移除档案收益 | 移除请求损失 | 等幅头对照 | 等幅选中恢复优势 | 竞争 / 恢复门 |',
            '| --- | --- | --- | --- | --- | --- | --- |'])
        for head,cells in causality['heads'].items():
            for cell,group in cells.items():
                for setting in ('aligned','conflict'):
                    a=group[setting]
                    columns=[show(a[k]) for k in ('full_head','archive','request_loss','match_source','matched_selected_over_random')]
                    gates=f"{'通过' if a['competition'] else '未通过'} / {'通过' if a['direction_specific_rescue'] else '未通过'}"
                    lines.append('| '+' | '.join([f'H{head} {cell} {setting}',*columns,gates])+' |')
        lines.extend(['','| 头 / 组合 / 档案 | 头 / 恢复等幅覆盖 | G 恢复 | U 恢复 | G+U 恢复 | 随机恢复 |',
                      '| --- | --- | --- | --- | --- | --- |'])
        for h,cells in causality['heads'].items():
            for cell,group in cells.items():
                for setting in ('aligned','conflict'):
                    a=group[setting]
                    coverage=f"{a['source_dose']['coverage']:.1%} / {a['restore_dose']['coverage']:.1%}"
                    rows=[show(a['native_distance_gain'][k]) for k in ('gate_restore','up_restore','both_restore','random_restore')]
                    lines.append('| '+' | '.join([f'H{h} {cell} {setting}',coverage,*rows])+' |')
        lines.append('')
        for h,cells in causality['heads'].items():
            for cell,group in cells.items():
                a=group['conflict']
                case=a['counterexamples'][0]
                lines.append(f"- H{h} 在 {cell} 冲突下的档案 V × 请求 V 交互 {show(a['joint_nonadd'])}，"
                    f"在近剂量范围内选中恢复相对随机 {show(a['near_native_selected_over_random'])} "
                    f"（{a['near_native_selected_over_random']['n']} 家族）。"
                    f"反向端样本 {case['id']} 的原格式 margin 变化 "
                    f"档案阻断 {case['effects']['archive']:+.4f}、请求阻断 {case['effects']['request']:+.4f}。")
    elif not behavior['format']['passed']:
        lines.append('- 格式行为门未通过，按冻结规则停止确认材料的原生路径干预。')
    lines.extend([
        '',
        '- 可复核材料在 tests/glm5/result/paradigm_path_20261117_v1，含冻结协议、行为、逐批原生值/剂量、两个目标的逐家族预测、反向样本和 SHA-256。保留任何分数逆转和不合格剂量。四位依次为中文、Python、修订计数和方括号，各位为 0 时是英文、论文、初始计数和圆括号。',
        '- 理论边界：因子交互对分布有用，只说明这个输出读出在这些条件下需要组合项。局部原生恢复能验证一条可用通路，等幅优势才支持候选方向的相对特异性。保留其他残差通路、相邻头和语法/计数位置作为未解释部分。按 W_O 列读到同块 G/U、再由 W_down 写回的代数路径是可定位的；信号是否在其他家族被复用由样本外预测决定。',
        '- 下一大阶段应覆盖多个独立句式和可自然生成的答案空间，盲测“档案/请求 → 头输出 → G/U → 格式/数字/括号”每条边的方向与效应量。预注册跨层残差旁路、多组匹配控制，与单组 G/U 恢复作对照。冻结一个同时预测正例、拮抗和四阶失败的分层模型，达到行为、预测和逐边干预三方面的门才提升图谱等级。',
    ])
    return '\n'.join(lines)+'\n'


def report(close):
    from transformers import AutoTokenizer

    plan=R.read('plan.json')
    assert plan['code']==X.hashes() and plan['inputs']==X.bind_inputs()
    cal=R.read('calibration.json')
    behavior=R.read('behavior.json')
    assert (OUT/'plan.json').stat().st_mtime_ns<(OUT/'behavior.json').stat().st_mtime_ns
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    pred=fusion(X.load('confirm'),tok) if all(v['passed'] for v in behavior.values()) else None
    cau=paths(X.load('native'),tok) if behavior['format']['passed'] else None
    R.write('analysis.json',dict(plan=X.digest(OUT/'plan.json'),at=C.stamp(),fusion=pred,causal=cau,behavior=behavior))
    body=render(plan,cal,behavior,pred,cau)
    check_write(OUT/'report.md',len(body.encode())+4096,operation='写入逐头路径复核').write_text(
        '# '+TITLE+'\n\n'+body,encoding='utf-8')
    if not close:
        return
    if not (OUT/'memo-receipt.json').exists():
        assert TITLE not in MEMO.read_text(encoding='utf-8-sig')
        verified=append(check_write(MEMO,len(body.encode())+4096,operation='追加 GLM5 研究记录'),TITLE,body)
        cleanup=[]
        if X.A.TEMP.is_dir():
            folder=X.A.TEMP.resolve()
            assert folder==X.A.TEMP and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
            owned=[folder,*folder.rglob('*')]
            assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(folder) for p in owned)
            size=sum(p.stat().st_size for p in owned if p.is_file())
            shutil.rmtree(folder)
            cleanup.append(dict(path=str(folder),freed_bytes=size))
        R.write('memo-receipt.json',dict(memo=verified,cleanup=cleanup))
    files={p.name:dict(bytes=p.stat().st_size,sha256=X.digest(p)) for p in sorted(OUT.iterdir())
           if p.is_file() and p.name!='manifest.json'}
    R.write('manifest.json',dict(run=plan['run'],files=files,sources=plan['code']))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--close',action='store_true')
    args=cli.parse_args()
    X.A.setup()
    report(args.close)
