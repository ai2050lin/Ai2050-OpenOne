"""Report the independent placeholder contrasts and the wider native pathway."""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import paradigm_marker_field as A
import paradigm_marker_native as N
from paradigm_position_report import stat, show, direction
from b36_semantic14_memo import append
from deploy.storage_policy import check_write


R,C,M,S,P,OUT=A.R,A.C,A.M,A.S,A.P,A.OUT
TITLE='1111 两处占位的独立因子、共同后缀联合干预与 B32 分解'
MEMO=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
MARK=('旧档案/旧请求','旧档案/新请求','新档案/旧请求','新档案/新请求')


def contrast(x):
    return dict(archive=stat((x[:,2]+x[:,3]-x[:,0]-x[:,1])/2),
        request=stat((x[:,1]+x[:,3]-x[:,0]-x[:,2])/2),
        interaction=stat(x[:,3]-x[:,2]-x[:,1]+x[:,0]),
        archive_only=stat(x[:,2]-x[:,0]),request_only=stat(x[:,1]-x[:,0]),
        archive_minus_request=stat(x[:,2]-x[:,1]))


def factorial(data,c,tok):
    y=A.increments(data)
    n=len(y)
    assert n==128
    feat=data['project'].reshape(n,4,3,2,4)
    x=feat[:,:,:,1]-feat[:,:,:,0]
    out=dict(layouts={},edges={},marker_states={},counterexamples=[])
    for i,l in enumerate(M.LAYOUTS):
        out['layouts'][str(l)]=dict(means=[stat(y[:,m,i]) for m in range(4)],effects=contrast(y[:,:,i]))
    for marker in range(4):
        a,b,z=y[:,marker].T
        out['marker_states'][str(marker)]=dict(reversal=int(((a>0)&(b<0)).sum()),
            return_positive=int(((b<0)&(z>0)).sum()),both=int(((a>0)&(b<0)&(z>0)).sum()))
    for i,(edge,_,_) in enumerate(M.EDGES):
        actual=y[:,:,i+1]-y[:,:,i]
        read=x[:,:,i+1]-x[:,:,i]
        f=c['predictions'][edge]
        err=f['joint']['baseline']-actual
        scores={};pred={}
        for route,design in (('early',read[:,:,:1]),('joint',read)):
            w=f[route]
            guess=w['baseline']+(design-w['mean'])/w['spread']@w['beta']
            pred[route]=guess
            loss=guess-actual
            scores[route]=dict(MAE=float(np.abs(loss).mean()),MAE_by_marker=np.abs(loss).mean(0).tolist(),
                MSE_gain=stat((err**2-loss**2).mean(1)))
        scores['joint']['gain_over_B0']=stat((((pred['early']-actual)**2)-((pred['joint']-actual)**2)).mean(1))
        out['edges'][edge]=dict(means=[stat(actual[:,m]) for m in range(4)],effects=contrast(actual),
            baseline=float(np.abs(err).mean()),baseline_by_marker=np.abs(err).mean(0).tolist(),prediction=scores)
        chosen=c['markers'][1]
        fails=np.flatnonzero(y[:,chosen,1]<0) if edge=='archive' else np.flatnonzero(y[:,chosen,2]<=0)
        largest=np.argsort(np.abs(pred['joint'][:,chosen]-actual[:,chosen]))[-6:][::-1]
        for f0 in dict.fromkeys([*fails[:8],*largest]):
            base=int(f0)*24
            pos=[base+6*chosen+2*j+a for j in (i,i+1) for a in (0,1)]
            out['counterexamples'].append(dict(edge=edge,family=str(data['family'][base]),marker=chosen,
                ids=[str(data['ids'][j]) for j in pos],natural=y[f0,chosen].tolist(),
                predicted=float(pred['joint'][f0,chosen]),tokens=tok.batch_decode([[int(ii)] for ii in data['top'][pos,0]])))
    return out


def field(d,tokenizer):
    trace=d['trace']
    assert trace.shape==(384,33,160,6)
    assert d['field'].shape[:2]==(2,41) and d['field'].shape[-1]==5120
    rows=M.build(tokenizer,'discovery')[0]
    assert [r['id'] for r in rows]==d['ids'].tolist()
    first=rows[0]['source'][0]
    assert np.array_equal(d['field'][0,:,:first],d['field'][1,:,:first])
    for i,row in enumerate(rows[::2]):
        assert (trace[i,:,:row['source'][0],[0,1,2]]==0).all()
    record={}
    for l in M.LAYOUTS:
        inds=np.flatnonzero(d['layout'][::2]==l)
        source=d['source'][::2,-1][inds]
        x=trace[inds,0,source+1,1]
        assert (source+1<d['query'][::2,0][inds]).all()
        use=[rows[i*2]['token']['ids'][q] for i,q in zip(inds,source+1,strict=True)]
        top=tokenizer.batch_decode([[t] for t in sorted(set(use))])
        markers=np.asarray([rows[2*i]['marker'] for i in inds])
        record[str(l)]=dict(B0_first_mean=float(x.mean()),first_strong=int((x>=A.RMS_LINE).sum()),
                             count=len(x),tokens=top,B0_by_marker=[float(x[markers==m].mean()) for m in range(4)])
    return dict(shape=list(d['field'].shape),trace=list(trace.shape),pre_archive_identical=True,layout=record)


def native(raw,clean,c,tok):
    assert len(raw['ids'])==384 and len(set(raw['family']))==64
    assert (raw['layout'].reshape(-1,2,3)==np.array(M.LAYOUTS)[None,None]).all()
    marker=np.array([int(k[1]) for k in raw['ids']])
    assert (marker.reshape(-1,2,3)==np.array(c['markers'])[None,:,None]).all()
    assert (raw['kl'][:,[N.ARMS.index(k) for k in ('sham','restore')]]==0).all()
    lookup={k:i for i,k in enumerate(clean['ids'])}
    idx=[[lookup[k[:-1]+'0'],lookup[k]] for k in raw['ids']]
    assert np.array_equal(clean['top'][idx,0],raw['original'])
    maxdiff=float(np.abs(clean['margin'][idx,1]-raw['natural']).max())
    assert maxdiff<=1e-5
    geom=raw['geometry']
    assert geom.shape==(384,4,5) and (geom[:,:,1]>0).all() and (geom[:,:,3]>0).all()
    assert (geom[:,0,0]==4).all() and (geom[:,1:3,0]==8).all()
    assert (geom[:,3,0]==raw['query'][:,0]-raw['sites'][:,4]).all()
    source=raw['source'][:,-1]
    for j,stage in enumerate(c['stages']):
        if stage['kind']=='proximal':
            q1,q2=source+1,source+4
        elif stage['kind']=='windows':
            sites=raw['sites'][:,stage['sites']]
            q1,q2=sites.min(1),sites.max(1)+1
        else:
            q1,q2=raw['sites'][:,4],raw['query'][:,0]-1
        assert (geom[:,j,1:]==np.stack((q1-source,q2-source,raw['query'][:,0]-q2,raw['query'][:,0]-q1),axis=1)).all()
    receipt=raw['dose']
    assert receipt.shape==(384,4,13)
    dose=[];geometry=[]
    for i,g in enumerate(c['stages']):
        entry=P.dose_percentiles(receipt[:,i,:12])
        entry['orthogonal_abs_p95']=float(np.quantile(np.abs(receipt[:,i,12]),.95))
        dose.append(entry)
        geometry.append(dict(stage=g,min=geom[:,i].min(0).tolist(),max=geom[:,i].max(0).tolist()))
    original=raw['natural'][:,1]
    moved=original-raw['natural'][:,0]
    arm={k:original-raw['edit'][:,i] for i,k in enumerate(N.ARMS)}
    result=dict(exact_restore=True,alignment=dict(views=len(idx)*2,margin_max=maxdiff),geometry=geometry,
        doses=dose,receivers=P.dose_percentiles(raw['gu_dose']),markers={},counterexamples=[])
    for marker_id in c['markers']:
        m=marker==marker_id
        by_edge={}
        for key,a,b in M.EDGES:
            before=m&(raw['layout']==a);after=m&(raw['layout']==b)
            assert before.sum()==after.sum()==64 and np.array_equal(raw['family'][before],raw['family'][after])
            sign=-1 if key=='archive' else 1
            def edge(values):
                return sign*(values[after]-values[before])
            natural=stat(moved[after]-moved[before])
            denominator=max(abs(natural['mean']),1e-9)
            valid=(receipt[before,:,5]>0).all(1)&(receipt[after,:,5]>0).all(1)
            gv=(raw['gu_dose'][before,5]>0)&(raw['gu_dose'][after,5]>0)
            raw_effect={name:stat(edge(arm[name])) for name in ('proximal','suffix','joint','joint_skip','joint_mlp','joint_at_E')}
            shares={name:statistic['mean']/denominator for name,statistic in raw_effect.items()}
            matched=stat(edge(arm['matched']-arm['scramble'])[valid])
            g={name:stat(edge(arm['joint']-arm['joint_'+name])) for name in ('freeze_g','freeze_u','freeze_gu','freeze_random')}
            absolute=stat((np.abs(edge(arm['joint']))-np.abs(edge(arm['joint_gu_match'])))[gv])
            random=stat((np.abs(edge(arm['joint_random_match']))-np.abs(edge(arm['joint_gu_match'])))[gv])
            reversal=(direction(stat(moved[before]))==-sign and direction(stat(moved[after]))==sign and direction(natural)==sign)
            qualified=bool(reversal and shares['joint']>=.10 and direction(raw_effect['joint'])==1 and
                            valid.mean()>=.98 and direction(matched)==1)
            by_edge[key]=dict(natural=natural,expected_sign_change=reversal,effects=raw_effect,shares=shares,
                matched_vs_control=matched,covered=float(valid.mean()),gu_coverage=float(gv.mean()),gu=g,
                gu_share={k:v['mean']/denominator for k,v in g.items()},gu_recovery=absolute,gu_over_random=random,
                gu_supported=bool(gv.mean()>=.98 and direction(absolute)==direction(random)==1),
                late_extra=stat(edge(arm['joint']-arm['joint_at_E'])),
                joint_over_suffix=stat(edge(arm['joint']-arm['suffix'])),
                E_nonlinear=stat(edge(arm['joint_at_E']-arm['joint_skip']-arm['joint_mlp'])),passes=qualified)
            eligible=np.flatnonzero(after)
            worst=eligible[np.argsort(edge(arm['joint']))[:6]]
            for i in dict.fromkeys([*worst,*eligible[~(valid&gv)]]):
                result['counterexamples'].append(dict(marker=marker_id,edge=key,id=str(raw['ids'][i]),
                    natural=float(moved[i]),arm_change=float(arm['joint'][i]),
                    tokens=tok.batch_decode([[int(v)] for v in [*raw['original'][i],raw['top'][i,N.ARMS.index('joint')]]]),
                    dose=receipt[i].tolist()))
        output={}
        for trial in ('proximal','suffix','joint','matched','scramble','joint_gu_match','joint_random_match'):
            j=N.ARMS.index(trial)
            output[trial]=dict(mean_KL=float(raw['kl'][m,j].mean()),
                               top_changes=int((raw['top'][m,j]!=raw['original'][m,1]).sum()))
        activations=raw['gua'][m,1]-raw['gua'][m,0]
        gua={group:np.abs(activations[:,view]).mean((0,1)).tolist() for group,view
             in (('selected',slice(32)),('random',slice(32,64)))}
        result['markers'][str(marker_id)]=dict(edges=by_edge,output=output,
            gua_change=gua,common_path=bool(all(z['passes'] for z in by_edge.values())))
    return result


def chart(all_):
    os.environ['MPLCONFIGDIR']=str(A.TEMP/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f=all_['factorial'];native=all_['native']
    fig,ax=plt.subplots(1,2,figsize=(11.0,4.5),layout='constrained')
    colors=('#755644','#187061','#365ba3','#b64d57')
    labels=('Old / Old','Old archive / New request','New archive / Old request','New / New')
    for marker in range(4):
        values=[f['layouts'][str(i)]['means'][marker] for i in M.LAYOUTS]
        y=np.asarray([v['mean'] for v in values]);ci=np.asarray([v['ci'] for v in values])
        ax[0].errorbar(np.arange(3)+(marker-1.5)*.026,y,yerr=np.stack((y-ci[:,0],ci[:,1]-y)),
            fmt='o-',ms=5,capsize=3,color=colors[marker],label=labels[marker])
    ax[0].axhline(0,color='.4',lw=1)
    ax[0].set_xticks(range(3),['Early archive','Late archive','Late request'])
    ax[0].set_ylabel('Code minus prose archive score')
    ax[0].set_title('Independent placeholder swap · 128 families')
    ax[0].legend(loc='upper center',fontsize=8,frameon=False)
    rows=[(m,e,values) for m,pack in native['markers'].items() for e,values in pack['edges'].items()]
    k=np.arange(len(rows))
    for j,(name,color) in enumerate((('proximal','#767b82'),('suffix','#3e82ae'),('joint','#df8249'))):
        ax[1].bar(k+(j-1)*.23,[100*v['shares'][name] for _,_,v in rows],width=.22,label=name,color=color)
    ax[1].axhline(0,color='.45',lw=1)
    ax[1].set_xticks(k,[f'{m} · {e}' for m,e,_ in rows],rotation=15)
    ax[1].set_title('Original donor cuts · 64 families')
    ax[1].set_ylabel('% of natural movement')
    ax[1].legend(loc='best',fontsize=8,frameon=False)
    fig.savefig(check_write(OUT/'marker-paths.png',5*2**20,operation='标记定量图'),dpi=165)
    plt.close(fig)


def render(out):
    f=out['factorial'];d=out['design'];c=out['candidate'];n=out['native'];p=out['holdout']
    old=out['prior']
    lines=[
      f"- Phase 2975 的同家族两标记整组互换将早请求晚档案的效应由 {old['old_mid']:+.4f} 改到 "
      f"{old['new_mid']:+.4f}，翻转由 {old['old_flip']}/128 降至 {old['new_flip']}/128。本轮把两个标记独立交叉。",
      f"- 本地 Qwen3-14B 原始 BF16，CPU 取权重、CUDA 顺序执行 40 块，批 8、宽 160。设计冻结 {d['at']}，"
      f"候选 {c['at']}，留出方法 {p['at']}。16 试跑、32 发现、128 独立确认家族、128 锁箱家族。锁箱只核对材料哈希。",
      f"- 档案占位由 {M.OLD[0]} 改为 {M.NEW[0]}，请求占位由 {M.OLD[1]} 改为 {M.NEW[1]}。"
      "活动文本、顺序对照中的槽宽、提示和读出位置均固定。单独换每个标记时严格核对 token 差落在对应的不活动槽。"
      "三布局为 0 档案早／请求早、2 档案晚／请求早、3 档案晚／请求晚。",
      f"- 先设行为门 .75，成对试跑最差格式条件 {min(z for m in out['behavior']['rates'] for y in m for z in y):.0%}，"
      f"四轴最差 {min(v['minimum_stratum'] for v in out['behavior']['factors'].values()):.0%}。"
      "复测 1111 的自由起始格式评分，不以是否翻转作为准入。",
      "- D 等于代码档案时的 print 相对论文候选的 logit 间隔减去论文档案时的间隔。档案后移边是 D₂−D₀，请求后移边是 D₃−D₂。"
      "表中先对同一例中两种档案配对，再按家族 bootstrap，区间 99%。标记主效应在另一个标记的两水平上平均，交互是差上之差。",
      '',
      r'$$C_A=(D_{10}+D_{11}-D_{00}-D_{01})/2,\quad C_R=(D_{01}+D_{11}-D_{00}-D_{10})/2,\quad I=D_{11}-D_{10}-D_{01}+D_{00}.$$',
      '',
      '| 标记组合 | 布局 0 的 D | 布局 2 的 D | 布局 3 的 D | 档案后移反转 / 请求后移转正 |',
      '| --- | --- | --- | --- | --- |',
    ]
    for i,label in enumerate(MARK):
        effect=[show(f['layouts'][str(j)]['means'][i]) for j in M.LAYOUTS]
        counts=f['marker_states'][str(i)]
        lines.append(f"| m{i} {label} | {' | '.join(effect)} | {counts['reversal']}/128 / {counts['return_positive']}/128 |")
    lines.extend(['',
        '| 对比 | 档案标记主效应 | 请求标记主效应 | 两标记交互 | 只换档案 / 只换请求 |',
        '| --- | --- | --- | --- | --- |'])
    for key,rec in [('layout2',f['layouts']['2']),*f['edges'].items()]:
        a=rec['effects']
        lines.append(f"| {key} | {show(a['archive'])} | {show(a['request'])} | {show(a['interaction'])} | "
            f"{show(a['archive_only'])} / {show(a['request_only'])} |")
    b=f['layouts']['2']['effects']
    dominance=direction(b['archive_minus_request'])
    marker_result=('档案占位的独立提升更大' if dominance>0 else
                   '请求占位的独立提升更大' if dominance<0 else '两处占位的独立提升未分出高低')
    lines.extend(['',f"- 晚档案、早请求条件下，只换档案与只换请求的效应之差 {show(b['archive_minus_request'])}。"
        f"{marker_result}。抬高中间布局的档案效应会削弱由正变负。两条移动的实际差值另有独立统计。",
        '',
        '### 共同路径预测与原生联合干预',
        f"- 发现组保存 {out['field']['shape']} 的原生全场和 {out['field']['trace']} 的分层分位置对照，"
        "档案前逐位一致。冻结同一组位置用于两条移动边。层序为 "+'、'.join(f"B{v['block']}/{v['kind']}" for v in c['stages'])+'。',
        '- B0 保留紧跟活动档案的四个 token，两个中间阶段各八个 token，B31 从晚档案槽后一直到生成起始位前。'
        "两组中间位按档案词表分组响应的预测能力排序，直接档案和最终起始位均不移植。"
        f"冻结 {MARK[c['markers'][0]]} 和 {MARK[c['markers'][1]]} 进入原生实验。"
        "其余两组全部参与自然行为和留出预测。预测输入为起始位之前的残差投影，也保留只读 B0 与发现组均值两种基线。",
        '',
        '| 留出边 | 发现均值 MAE | B0 MAE | 四边界 MAE | 四边界相对 B0 MSE 改善 |',
        '| --- | ---: | ---: | ---: | --- |'])
    for edge,v in f['edges'].items():
        r=v['prediction']
        lines.append(f"| {edge} | {v['baseline']:.4f} | {r['early']['MAE']:.4f} | {r['joint']['MAE']:.4f} | "
                     f"{show(r['joint']['gain_over_B0'])} |")
    lines.extend(['',
        '- 同家族同位置的论文档案残差是代码档案端的原生供体。以真实计算形状先注入、再推进整个后缀。'
        "对同一输入，两段距离同时相等就固定了 token 位置。因此主控制用完全相同的层与 token，改变扰动方向并近似正交化。"
        "逐阶段匹配实际 BF16 Frobenius 剂量，方向余弦不低于 .90、范数至少 .003、相对剂量差不超过 3%。"
        "剂量针对每阶段的整组位置，单个 token 可不同；置换方向也可偏离自然状态分布。",
        f"- 64 家族原生切换。原生无操作和复位逐位重现全词表 logits。常规前向对账 {n['alignment']['views']} 个输入，"
        f"格式间隔最大差 {n['alignment']['margin_max']:.3g}。以下正比例表示移去预期方向的自然效应；负值表示把变化推得更远。",
        '',
        '| 标记 / 边 | B0-only | B31 后缀-only | 四段联合 | 等距等幅优势 | B32 旁路 / MLP | 共路分边门 |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |'])
    for m,group in n['markers'].items():
        for e,r in group['edges'].items():
            s=r['shares']
            lines.append(f"| m{m} {e} | {s['proximal']:+.1%} | {s['suffix']:+.1%} | {s['joint']:+.1%} | "
                f"{show(r['matched_vs_control'])} | {s['joint_skip']:+.1%} / {s['joint_mlp']:+.1%} | "
                f"{'通过' if r['passes'] else '未通过'} |")
    lines.append('')
    for m,group in n['markers'].items():
        clean=group['output']['joint'];control=group['output']['scramble']
        lines.append(f"- m{m} 原生联合与置换方向的全词表 KL {clean['mean_KL']:.5f} / {control['mean_KL']:.5f}，"
                     f"首 token 改变 {clean['top_changes']}/192 / {control['top_changes']}/192。")
        gua=group['gua_change']
        lines.append(f"- m{m} B32 候选 G/U/A 的平均绝对变化 "+'/'.join(f'{a:.4f}' for a in gua['selected'])
                     +'，同层随机组 '+ '/'.join(f'{a:.4f}' for a in gua['random'])+'。')
        for e,r in group['edges'].items():
            g=r['gu_share']
            lines.append(f"- m{m}/{e} 联合效应 {show(r['effects']['joint'])}，联合减 B31 后缀 {show(r['joint_over_suffix'])}。"
                f"B32 的 G 与 U 有向恢复占自然边 {g['freeze_g']:+.1%} / {g['freeze_u']:+.1%}，"
                f"同层等幅恢复对随机组优势 {show(r['gu_over_random'])}。两级等幅覆盖 {r['covered']:.0%} / {r['gu_coverage']:.0%}。"
                f"B32 非起始位后续贡献 {show(r['late_extra'])}，起始位两路非加和 {show(r['E_nonlinear'])}。")
    for v,geom,dose in zip(c['stages'],n['geometry'],n['doses'],strict=True):
        scales=dose['extrapolation_median']
        scale_text=f"（原生方向/置换方向的实投剂量中位倍率 {scales[0]:.2f}/{scales[1]:.2f}）" if scales else ''
        lines.append(f"- B{v['block']} 控制两段距档案和输出逐条相同，联合点数 {geom['min'][0]}–{geom['max'][0]}。"
                     f"合格剂量比例 {dose['coverage']:.1%}，正交方向余弦绝对值 P95 {dose['orthogonal_abs_p95']:.4f}"
                     f"{scale_text}。")
    for name in ('archive','request'):
        q=next(r for r in f['counterexamples'] if r['edge']==name)
        z=next(r for r in n['counterexamples'] if r['edge']==name)
        lines.append(f"- {name} 的反例家族 {q['family']}，所选标记布局 0/2/3 的档案效应 {q['natural']}。"
                     f"原生异常 {z['id']} 的档案差 {z['natural']:+.4f}，联合效应 {z['arm_change']:+.4f}。"
                     "完整输入、首 token 与剂量保存在结果清单。")
    accepted=[key for key,x in n['markers'].items() if x['common_path']]
    spreads=[r['shares']['joint'] for v in n['markers'].values() for r in v['edges'].values()]
    if all(x['edges']['request']['passes'] for x in n['markers'].values()):
        lines.append('- 四段共同组在两套受干预标记下都通过了请求后移边，B31 后缀单独的作用更小，新档案占位组甚至反向。'
                     '档案后移的原生切换也减小了自然差，但在相同几何和剂量下未超过方向对照。'
                     'B31 单独切换是原生范围诊断，这一臂没有独立的等幅配对。')
    lines.extend([
        f"- 两条边的四段联合比例范围 {min(spreads):+.1%} 至 {max(spreads):+.1%}，同时通过冻结检验的标记组 {accepted or '零组'}。"
        '原生联合移植可检验这组状态的介入作用，等距等幅特异性与天然不变量还需要各自的证据。',
        '- B32 分解取残差支路或新 MLP 写入并将其余 token 状态装回原值。G/U 单元用原模型门值和上投影，'
        '经 SiLU(G)×U 与 down_proj 重新写回。局部贡献可相抵；这个分解只覆盖 B32 自由生成起始位置，不把局部写入当完整机制。',
        '- 外部图谱新增两个可独立控制的占位因素及其交互。内部图谱新增 B0 到共同后缀的联合边和幅度台账。'
        '两图谱的联系以同一组路径在不同因子上的预测和原生效应来评分。RDC 下一阶段沿有证据的后缀窗口做分段收缩，'
        '冻结同剂量的范围变化和全后缀恢复，并向自然续写验证拟合范围。',
        f"- 运行 {M.RUN}，原始产物存 tests/glm5/result/{OUT.name}。包含四标记对照、宽路径数据、模型与代码哈希、反例与绘图。"
    ])
    return '\n'.join(lines)+'\n'


def run(close):
    from transformers import AutoTokenizer

    design=R.read('design.json');c=R.read('candidate.json');plan=R.read(N.PLAN)
    assert design['code']==A.source_id() and design['previous']==A.previous()
    assert plan['code']==N.code() and plan['design']==N.sha('design.json') and plan['candidate']==N.sha('candidate.json')
    assert (OUT/'candidate.json').stat().st_mtime_ns<(OUT/N.PLAN).stat().st_mtime_ns
    assert all((OUT/N.PLAN).stat().st_mtime_ns<(OUT/(n+'-files.json')).stat().st_mtime_ns for n in ('confirm','native'))
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    data=S.load('confirm',N.sha(N.PLAN))
    raw=S.load('native',N.sha(N.PLAN))
    prev=json.loads((A.PREVIOUS/'analysis.json').read_bytes())
    out=dict(at=C.stamp(),design=design,candidate=c,holdout=plan,behavior=R.read('behavior.json'),
        prior=dict(old_mid=prev['marker']['old']['2']['mean'],new_mid=prev['forecast']['layouts']['2']['mean'],
                   old_flip=prev['marker']['old_edges']['archive']['directions'],
                   new_flip=prev['marker']['new_edges']['archive']['directions']),
        factorial=factorial(data,c,tok),field=field(S.load('discover'),tok),native=native(raw,data,c,tok))
    R.write('analysis.json',out)
    chart(out)
    body=render(out)
    check_write(OUT/'report.md',len(body.encode())+4096,operation='四标记宽路径报告').write_text(
        '# '+TITLE+'\n\n'+body,encoding='utf-8')
    if not close:
        return
    assert not (OUT/'memo-receipt.json').exists() and TITLE not in MEMO.read_text(encoding='utf-8-sig')
    done=append(check_write(MEMO,len(body.encode())+4096,operation='追加 GLM5 Memo'),TITLE,body)
    cleanup=[]
    if A.TEMP.is_dir():
        p=A.TEMP.resolve()
        assert p==A.TEMP and p.parent==(R.ROOT/'tests/glm5_temp').resolve()
        items=[p,*p.rglob('*')]
        assert all(not x.is_symlink() and not x.is_junction() and x.resolve().is_relative_to(p) for x in items)
        size=sum(x.stat().st_size for x in items if x.is_file())
        shutil.rmtree(p)
        cleanup.append(dict(path=str(p),freed_bytes=size))
    R.write('memo-receipt.json',dict(memo=done,cleanup=cleanup))
    manifest={p.name:dict(sha256=A.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    sources={q.relative_to(R.ROOT).as_posix():A.sha(q) for q in (Path(__file__).resolve(),Path(N.__file__).resolve(),
        Path(N.K.__file__).resolve(),R.ROOT/'tests/glm5/paradigm_position_report.py')}
    R.write('manifest.json',dict(run=M.RUN,files=manifest,sources=design['code']|sources))
    print('GLM5 Phase',done['phase'],'appended',flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--close',action='store_true')
    args=cli.parse_args()
    A.setup()
    run(args.close)
