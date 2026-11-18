"""Verify native and predictive evidence for the two archive placement moves."""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import paradigm_order_field as A
import paradigm_order_native as N
import paradigm_order_marker as K
import paradigm_order_legacy as L
from paradigm_position_report import stat, show, direction
from b36_semantic14_memo import append
from deploy.storage_policy import check_write


R,C,M,S,P,OUT=A.R,A.C,A.M,A.S,A.P,A.OUT
TITLE='1111 档案反转和请求转正的交叉路径、等距等幅与 B32 分解'
MEMO=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'


def name(point):
    block,site=point
    return f"B{block}·{'档案后第一个 token' if site==-1 else M.P.ANCHORS[site]}"


def forecasts(data,c,tok):
    deltas,_=A.pairs(data)
    layouts={str(p):stat(deltas[:,i]) for i,p in enumerate(M.LAYOUTS)}
    assert len(deltas)==128
    edges={}
    for k,a,b in M.EDGES:
        i,j=M.LAYOUTS.index(a),M.LAYOUTS.index(b)
        actual=deltas[:,j]-deltas[:,i]
        err0=c['summary'][k]['mean']-actual
        scores={};pred={}
        for key,pts in {'early':[A.EARLY],**c['routes']}.items():
            fit=c['forecast'][k][key]
            x=(A.edge_matrix(data,(k,a,b),pts)-fit['mean'])/fit['scale']
            y=fit['y']+x@np.asarray(fit['beta'])
            err=y-actual
            pred[key]=y
            scores[key]=dict(MAE=float(np.abs(err).mean()),MSE_gain=stat(err0**2-err**2),
                             direction_agreement=float((np.sign(y)==np.sign(actual)).mean()))
        for route in N.KEYS:
            scores[route]['MSE_gain_over_B0']=stat((pred['early']-actual)**2-(pred[route]-actual)**2)
        sign=-1 if k=='archive' else 1
        consistent=(sign*deltas[:,i]<0)&(sign*deltas[:,j]>0)
        cases=[]
        wrong=np.flatnonzero(~consistent)
        outliers=np.argsort(np.abs(pred[k]-actual))[-8:][::-1]
        for f in dict.fromkeys([*wrong[:12],*outliers]):
            rowids=[int(6*f+2*p+v) for p in (i,j) for v in (0,1)]
            cases.append(dict(family=str(data['family'][6*f]),ids=[str(data['ids'][j]) for j in rowids],
                 effects=[float(deltas[f,i]),float(deltas[f,j])],forecast={x:float(z[f]) for x,z in pred.items()},
                 tokens=tok.batch_decode([[int(t)] for t in data['top'][rowids,0]])))
        edge=stat(actual)
        edges[k]=dict(effect=edge,from_layout=a,to_layout=b,baseline_MAE=float(np.abs(err0).mean()),
            scores=scores,observed_sign_pairs=int(consistent.sum()),
            sign_supported=bool(direction(layouts[str(a)])==-sign and direction(layouts[str(b)])==sign
                                and direction(edge)==sign),counterexamples=cases)
    return dict(layouts=layouts,edges=edges,forward_pairs=len(deltas),
                both=int(((deltas[:,0]>0)&(deltas[:,1]<0)&(deltas[:,2]>0)).sum()))


def field(data,c,tok):
    traces=data['trace']
    assert traces.shape==(384,33,160,6)
    raw=data['field']
    assert raw.shape[:2]==(2,41) and raw.shape[-1]==5120
    rows=M.build(tok,'confirm')[0]
    assert [x['id'] for x in rows]==data['ids'].tolist()
    assert np.array_equal(raw[0,:,:rows[0]['source'][0]],raw[1,:,:rows[0]['source'][0]])
    sides={}
    for layout in M.LAYOUTS:
        group=np.flatnonzero(data['layout'][::2]==layout)
        source=data['source'][::2][group,-1]
        onset=data['query'][::2][group,0]
        for i in group:
            assert (traces[i,:,:data['source'][2*i,0],[0,1,2]]==0).all()
        early={}
        for offset in range(1,5):
            site=source+offset
            response=traces[group,0,site,1]
            assert (site<onset).all()
            labels=[tok.decode([rows[2*i]['token']['ids'][p]]) for i,p in zip(group,site,strict=True)]
            early[str(offset)]=dict(attention_rms=float(response.mean()),strong=int((response>=S.RAW_THRESHOLD).sum()),
                          count=len(response),tokens=sorted(set(labels),key=lambda t:(-labels.count(t),t))[:3])
        chosen={}
        for k,route in c['routes'].items():
            chosen[k]=[]
            for b,s in route:
                site=source+1 if s==-1 else data['sites'][::2][group,s]
                above=traces[group,:,site,1]>=S.RAW_THRESHOLD
                first=np.where(above.any(1),above.argmax(1),-1)
                labels=[tok.decode([rows[2*i]['token']['ids'][p]]) for i,p in zip(group,site,strict=True)]
                chosen[k].append(dict(point=[b,s],tokens=sorted(set(labels),key=lambda t:(-labels.count(t),t))[:4],
                   B0_count=int((first==0).sum()),dist_from_archive=[int((site-source).min()),int((site-source).max())],
                   dist_from_onset=[int((onset-site).min()),int((onset-site).max())]))
        sides[str(layout)]=dict(early=early,candidate_geometry=chosen)
    return dict(shape=list(raw.shape),trace_shape=list(traces.shape),prefix_identical=True,layouts=sides)


def audit_alignment(clean,native):
    index={k:i for i,k in enumerate(clean['ids'])}
    pairs=[[index[k[:-1]+'0'],index[k]] for k in native['ids']]
    agree=np.array_equal(clean['top'][pairs,0],native['original'])
    error=float(np.abs(clean['margin'][pairs,1]-native['natural']).max())
    assert agree and error<=1e-5
    return dict(views=2*len(pairs),top1_match=agree,margin_max=error)


def marker_control(clean,bridge):
    assert np.array_equal(clean['ids'],bridge['ids'])
    y,banks=A.pairs(clean)
    z,prior_banks=A.pairs(bridge)
    assert np.array_equal(banks,prior_banks) and len(y)==len(z)==128
    result=dict(old={},paired_new_minus_old={},old_edges={},new_edges={},banks={})
    for i,layout in enumerate(M.LAYOUTS):
        result['old'][str(layout)]=stat(z[:,i])
        result['paired_new_minus_old'][str(layout)]=stat(y[:,i]-z[:,i])
    for key,_,_ in M.EDGES:
        k=0 if key=='archive' else 1
        a,b=k,k+1
        sign=-1 if key=='archive' else 1
        result['old_edges'][key]=dict(difference=stat(z[:,b]-z[:,a]),
            directions=int(((sign*z[:,a]<0)&(sign*z[:,b]>0)).sum()))
        result['new_edges'][key]=dict(difference=stat(y[:,b]-y[:,a]),
            directions=int(((sign*y[:,a]<0)&(sign*y[:,b]>0)).sum()))
    result['old_triplet_count']=int(((z[:,0]>0)&(z[:,1]<0)&(z[:,2]>0)).sum())
    for b in (0,1):
        mask=banks==b
        result['banks'][str(b)]=dict(old=stat(z[mask,1]),new=stat(y[mask,1]),
                                    shift=stat(y[mask,1]-z[mask,1]))
    return result


def full_native(raw,c,tok):
    assert len(raw['ids'])==192 and len(set(raw['family']))==64
    assert (raw['layout'].reshape(-1,3)==M.LAYOUTS).all()
    assert (raw['kl'][:,[N.ARMS.index(k) for k in ('sham','restore')]]==0).all()
    doses=raw['stage_dose'];geometry=raw['geometry']
    assert doses.shape==(192,2,3,13) and geometry.shape==(192,2,3,3)
    for ri,key in enumerate(N.KEYS):
        for bi,(_,site) in enumerate(c['routes'][key]):
            pos=raw['source'][:,-1]+1 if site==-1 else raw['sites'][:,site]
            want=np.stack((pos,pos-raw['source'][:,-1],raw['query'][:,0]-pos),1)
            assert (geometry[:,ri,bi]==want).all()
    adj=raw['prox_geometry']
    assert (adj[:,2]-adj[:,0]==1).all() and (adj[:,1]-adj[:,3]==1).all()
    actual=raw['natural'][:,1]-raw['natural'][:,0]
    arms={v:raw['natural'][:,1]-raw['edit'][:,i] for i,v in enumerate(N.ARMS)}
    result=dict(exact_sham=True,views=len(actual),families=len(set(raw['family'])),
        dose={key:[P.dose_percentiles(doses[:,r,j,:12])|
                  {'orthogonality_abs_p95':float(np.quantile(np.abs(doses[:,r,j,12]),.95))}
                  for j in range(3)] for r,key in enumerate(N.KEYS)},
        proximal_dose=P.dose_percentiles(raw['prox_dose']),geometry_exact=True,edges={},layouts={})
    for k,a,b in M.EDGES:
        aa=raw['layout']==a;bb=raw['layout']==b
        assert np.array_equal(raw['family'][aa],raw['family'][bb])
        sign=-1 if k=='archive' else 1
        native_edge=stat(actual[bb]-actual[aa])
        natural=(direction(stat(actual[aa]))==-sign and direction(stat(actual[bb]))==sign and direction(native_edge)==sign)
        scale=max(abs(native_edge['mean']),1e-8)
        def attenuation(vals):
            return sign*(vals[bb]-vals[aa])
        proximal=stat(attenuation(arms['proximal']))
        valid=(raw['prox_dose'][aa,5]>0)&(raw['prox_dose'][bb,5]>0)
        near=stat(attenuation(arms['prox_match']-arms['nearby_match'])[valid])
        routes={}
        for r,key in enumerate(N.KEYS):
            def arm(tag):
                return arms[f'{key}_{tag}']
            eligible=(doses[aa,r,:,5]>0).all(1)&(doses[bb,r,:,5]>0).all(1)
            receiver=(raw['gu_dose'][aa,r,5]>0)&(raw['gu_dose'][bb,r,5]>0)
            stats={q:stat(attenuation(arm(q))) for q in ('joint','skip','mlp','at_E')}
            share={q:stats[q]['mean']/scale for q in stats}
            matched=stat(attenuation(arm('match')-arm('scramble'))[eligible])
            g_and_u={mode:stat(attenuation(arm('joint')-arm(mode))) for mode in ('freeze_g','freeze_u','freeze_gu','freeze_random')}
            cuts=attenuation(arm('joint'))
            restore=np.abs(cuts)-np.abs(attenuation(arm('gu_match')))
            over_random=np.abs(attenuation(arm('random_match')))-np.abs(attenuation(arm('gu_match')))
            recovery=stat(restore[receiver]);control=stat(over_random[receiver])
            restored=bool(receiver.mean()>=N.PRECISION and direction(recovery)==direction(control)==1)
            used=bool(natural and share['joint']>=.10 and direction(stats['joint'])==1 and
                      direction(matched)==1 and eligible.mean()>=N.PRECISION)
            routes[key]=dict(values=stats,shares=share,targets_equal=float(eligible.mean()),receiver_equal=float(receiver.mean()),
                matched_vs_scramble=matched,joint_minus_proximal=stat(attenuation(arm('joint')-arms['proximal'])),
                suffix_other_than_E=stat(attenuation(arm('joint')-arm('at_E'))),
                split_interaction=stat(attenuation(arm('at_E')-arm('skip')-arm('mlp'))),
                GU_restore=g_and_u,GU_share={mode:v['mean']/scale for mode,v in g_and_u.items()},
                GU_recovery=recovery,GU_over_random=control,GU_pass=restored,path_pass=used)
        other=N.KEYS[1-N.KEYS.index(k)]
        result['edges'][k]=dict(natural=native_edge,sign_supported=bool(natural),proximal=proximal,
            proximal_matched_vs_neighbor=near,proximal_match_rate=float(valid.mean()),routes=routes,
            own_vs_cross_joint=stat(attenuation(arms[k+'_joint']-arms[other+'_joint'])))
    for layout in M.LAYOUTS:
        keep=raw['layout']==layout
        sign=np.sign(actual);group={}
        for ri,key in enumerate(N.KEYS):
            src=arms[key+'_joint']
            dose=raw['gu_dose'][:,ri]
            matched=keep&(dose[:,5]>0)
            selected=np.abs(arms[key+'_gu_match'])
            random=np.abs(arms[key+'_random_match'])
            candidates=np.flatnonzero(keep)
            worst=candidates[np.argsort((sign*src)[keep])[:4]]
            failed=candidates[~((doses[:,ri,:,5]>0).all(1)&(dose[:,5]>0))[keep]]
            cases=[]
            for i in dict.fromkeys([*worst,*failed]):
                cases.append(dict(id=str(raw['ids'][i]),natural=float(actual[i]),joint=float((sign*src)[i]),
                    dose=doses[i,ri].tolist(),tokens=tok.batch_decode([[int(p)] for p in
                        [*raw['original'][i],raw['top'][i,N.ARMS.index(key+'_joint')]]])))
            g=raw['gua'][keep,ri,1]-raw['gua'][keep,ri,0]
            group[key]=dict(natural=stat(actual[keep]),joint=stat((sign*src)[keep]),
                GUA_delta_selected=np.abs(g[:,:32]).mean((0,1)).tolist(),GUA_delta_random=np.abs(g[:,32:]).mean((0,1)).tolist(),
                restore=stat((np.abs(src)-selected)[matched]),over_random=stat((random-selected)[matched]),
                dose=P.dose_percentiles(dose[keep]),counterexamples=cases)
        result['layouts'][str(layout)]=group
    result['shared_routes']=[k for k in N.KEYS if all(result['edges'][e]['routes'][k]['path_pass'] for e in N.KEYS)]
    return result


def marker_native_difference(new,old):
    assert np.array_equal(new['ids'],old['ids']) and np.array_equal(new['geometry'],old['geometry'])
    output={}
    for edge,a,b in M.EDGES:
        first=new['layout']==a
        second=new['layout']==b
        assert np.array_equal(new['family'][first],new['family'][second])
        sign=-1 if edge=='archive' else 1
        by_route={}
        for route in N.KEYS:
            portions={}
            for term in ('joint','skip','mlp','at_E'):
                j=N.ARMS.index(f'{route}_{term}')
                aa=new['natural'][:,1]-new['edit'][:,j]
                bb=old['natural'][:,1]-old['edit'][:,j]
                x=aa-bb
                portions[term]=stat(sign*(x[second]-x[first]))
            by_route[route]=portions
        output[edge]=by_route
    return output


def chart(out):
    os.environ['MPLCONFIGDIR']=str(A.TEMP/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    previous_native=out['old_marker_native']
    fig,axs=plt.subplots(1,3 if previous_native else 2,figsize=(13 if previous_native else 10.3,4.6),
                         layout='constrained')
    stats=[out['forecast']['layouts'][str(k)] for k in M.LAYOUTS]
    y=np.array([n['mean'] for n in stats])
    iv=np.array([n['ci'] for n in stats])
    axs[0].errorbar(range(3),y,yerr=np.stack((y-iv[:,0],iv[:,1]-y)),fmt='o-',capsize=5,color='#14615a',label='New markers')
    last=[out['marker']['old'][str(k)] for k in M.LAYOUTS]
    old=np.array([n['mean'] for n in last])
    ci=np.array([n['ci'] for n in last])
    axs[0].errorbar(range(3),old,yerr=np.stack((old-ci[:,0],ci[:,1]-old)),fmt='s--',capsize=4,color='#a14b32',label='Old markers')
    axs[0].legend(loc='upper center',frameon=False)
    axs[0].axhline(0,color='.45',lw=1)
    axs[0].set_xticks(range(3),['0\narchive early','2\narchive late','3\nrequest also late'])
    axs[0].set_ylabel('Code minus prose archive effect')
    axs[0].set_title('Natural effect on 128 holdout families')
    sources=[('New markers',out['native'])]
    if previous_native:
        sources.append(('Old markers',previous_native))
    matrices=[np.array([[100*source['edges'][e]['routes'][r]['shares']['joint'] for r in N.KEYS]
                       for e in N.KEYS]) for _,source in sources]
    bounds=max(10,np.max(np.abs(matrices)))
    for ax,matrix,(label,_) in zip(axs[1:],matrices,sources,strict=True):
        im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bounds,vcenter=0,vmax=bounds))
        for i,j in np.ndindex(matrix.shape):
            ax.text(j,i,f'{matrix[i,j]:+.1f}%',ha='center',va='center')
        ax.set_xticks([0,1],['Archive route','Request route'])
        ax.set_yticks([0,1],['Archive move','Request move'])
        ax.set_title(label+' · edge attenuation')
    fig.colorbar(im,ax=axs[-1],label='% of natural movement effect')
    fig.savefig(check_write(OUT/'order-paths.png',5*2**20,operation='交叉路径绘图'),dpi=160)
    plt.close(fig)


def render(data):
    c=data['candidate'];f=data['forecast'];n=data['native'];old=data['previous'];bridge=data['marker']
    lines=[
        f"- Phase 2974 在早请求条件下后移档案，1111 有 {old['individual']}/128 家族反转，群体档案效应 "
        f"{old['old_layouts'][0]:+.4f}→{old['old_layouts'][1]:+.4f}；固定晚档案而后移请求时为 {old['old_layouts'][2]:+.4f}。",
        f"- 本轮仅使用本地原始 BF16 Qwen3-14B，40 块 CPU 读权重、CUDA 串行，批 8，宽 160。设计冻结 {data['design']['at']}，"
        f"候选冻结 {c['at']}，方法冻结 {data['holdout']['at']}。16 试跑、32 发现、128 新确认家族和 128 只登记哈希的锁箱家族。",
        f"- 新占位标记为 {M.EMPTY_A} 和 {M.EMPTY_R}。两类槽仍分别为六/五 token。位置 0 为档案早、请求早；2 为档案晚、请求早；3 为档案晚、请求晚。每次只交换一对等长槽，其余位置和读出逐 token 核对。活动档案和请求的词面沿用上一阶段。",
        f"- 试跑格式成对门最低 {min(min(a) for a in data['gate']['rates']):.0%}，四因子门最低 "
        f"{min(v['minimum_stratum'] for v in data['gate']['factors'].values()):.0%}。行为门独立于档案反转方向。"
        "格式读出是自由起始处 print 相对论文候选的 logit 间隔；数值代表模型评分。",
        '- 候选筛选只用发现集，计算档案替换的残差方向投影，再取相邻两个布局间的变化。B0 首邻 token 为共用点；中、晚段选择在两个档案词表中符号一致的候选。用低层活化预测未见家族的格式变化，另比发现均值与 B0 单点。',
    ]
    for k,points in c['routes'].items():
        lines.append(f"- 冻结{'档案移位' if k=='archive' else '请求移位'}候选为 {' → '.join(name(p) for p in points)}。")
    lines.extend(['',
        r'$$D_\ell=m_{\ell,\mathrm{code}}-m_{\ell,\mathrm{prose}},\quad T_{i\to j}=D_j-D_i,\quad '
        r'R_{i\to j}=\mathrm{sgn}(T_\mathrm{expected})[T_{i\to j}-\widetilde T_{i\to j}].$$',
        '',
        '- 自然档案效应 D 先在同布局内配对。每条单移边的对比值 T 再按家族相减。对原生路径，R 衡量受干预时沿预注册方向丢失多少原始位置效应。平均数按家族 bootstrap 给出 99% 区间。',
        '',
        '| 布局与单移 | 格式间隔或边效应 | 家族符合方向 |',
        '| --- | --- | ---: |'])
    for i,s in f['layouts'].items():
        lines.append(f"| 1111 布局 {i} | {show(s)} | — |")
    for tag,r in f['edges'].items():
        lines.append(f"| {tag} {r['from_layout']}→{r['to_layout']} | {show(r['effect'])} | {r['observed_sign_pairs']}/128 |")
    lines.extend(['',f"- 同一家族同时出现正→负→正的有 {f['both']}/128。"
        f"留出 {data['field']['trace_shape'][0]} 个原生档案配对含 B0–B32 完整位置摘要，并保留 "
        f"{data['field']['shape']} 的一对原生全场。档案之前的前缀逐位一致。"])
    lines.extend([
        f"- 为核对上次反转比例的落差，在确认集的新标记结果已知后，于 {data['marker_plan']['at']} "
        "冻结同 128 家族的旧标记复跑。只换两个不活动位置的占位文字，字面变化严格限制在不活动槽。"
        "这是同家族配对控制，两个标记作为一组换回，两个标记各自的作用未拆开。",
        '',
        '| 布局 | 旧标记同家族 | 新减旧的成对差 |',
        '| --- | --- | --- |',
    ])
    for l,statistic in bridge['old'].items():
        lines.append(f"| {l} | {show(statistic)} | {show(bridge['paired_new_minus_old'][l])} |")
    counts='；'.join(f"{edge} 旧 {v['directions']}/128、新 {bridge['new_edges'][edge]['directions']}/128"
                    for edge,v in bridge['old_edges'].items())
    lines.extend(['',f"- 同一家族的正负转换 {counts}。旧标记的正→负→正为 {bridge['old_triplet_count']}/128。"
                  "先以自然行为核对标记效应，再用相同的原生路径检验旧标记。"])
    for bank,values in bridge['banks'].items():
        lines.append(f"- 档案词表 {bank} 的早请求、晚档案效应从旧标记的 {show(values['old'])} 变到新标记的 "
                     f"{show(values['new'])}，同家族标记效应 {show(values['shift'])}。词表分组只作描述。")
    for layout,entry in data['field']['layouts'].items():
        e=entry['early']['1']
        lines.append(f"- 布局 {layout} 的档案后一 token {e['tokens'][0]!r}，B0 注意力后残差 RMS {e['attention_rms']:.5f}，"
                     f"{e['strong']}/{e['count']} 过 .0005 诊断线。距离、接收 token 和后续所有层写入保留在结构摘要。")
    lines.extend(['','| 留出边 | 发现均值 MAE | B0 MAE | 档案路线 MAE | 请求路线 MAE |',
                    '| --- | ---: | ---: | ---: | ---: |'])
    for k,entry in f['edges'].items():
        s=entry['scores']
        lines.append(f"| {k} | {entry['baseline_MAE']:.4f} | {s['early']['MAE']:.4f} | "
            f"{s['archive']['MAE']:.4f} | {s['request']['MAE']:.4f} |")
    lines.append('')
    for k,entry in f['edges'].items():
        lines.append(f"- {k} 留出 MSE 相对发现均值的变化，B0 {show(entry['scores']['early']['MSE_gain'])}，"
                     f"档案路线 {show(entry['scores']['archive']['MSE_gain'])}，请求路线 {show(entry['scores']['request']['MSE_gain'])}。"
                     '正值代表更接近实际边效应。')
        lines.append(f"- {k} 多层路径相对 B0 单点的 MSE 改善，档案路径 "
                     f"{show(entry['scores']['archive']['MSE_gain_over_B0'])}、"
                     f"请求路径 {show(entry['scores']['request']['MSE_gain_over_B0'])}。")
    lines.extend(['',
        '- 原生干预把同家族、同布局的论文档案残差联合移植到代码档案。按各自边界顺序注入，随后推进完整受影响后缀。两个候选均在三个布局执行。原值装回后，空操作、复位及未干预供体的 logits 全词表逐位相同。',
        f"- 原生干预 {n['families']} 家族、{n['views']} 个接收样本。与不干预前向重合的 {data['overlap']['views']} 条输入首 token 全一致，"
        f"格式间隔最大差 {data['overlap']['margin_max']:.3g}。",
        '- 同一例里，距档案和输出的两段距离 u−s、E−u 唯一决定位置 u。主对照用完全相同的位置和层，'
        '将干预方向作固定带符号的坐标重排，并去除与原方向的投影。这是等距离的方向对照。另在 B0 移向相邻一词，用同家族原生供体匹配剂量。'
        '主对照的置换方向可能脱离自然激活分布，邻位对照的两段距离各相差一个 token。',
        '- 剂量用各臂刚要注入时的实际 BF16 残差差值，要求范数相对差≤3%、各自原生方向余弦≥.90、范数≥.003。各层单独匹配后推进活化。B32 下游等幅另在原生残差写回后测量。',
        '',
        '| 待解释边、干预路径 | 原生联合所占比例 | 等距等幅优势 | B32 残差单路比例 | B32 MLP 单路比例 | G/U 等幅恢复优势 |',
        '| --- | ---: | --- | ---: | ---: | --- |'])
    for e,entry in n['edges'].items():
        for route,rr in entry['routes'].items():
            p=rr['shares']
            lines.append(f"| {e} / {route} | {p['joint']:+.1%} | {show(rr['matched_vs_scramble'])} | "
                f"{p['skip']:+.1%} | {p['mlp']:+.1%} | {show(rr['GU_over_random'])} |")
    lines.append('')
    for e,entry in n['edges'].items():
        lines.append(f"- {e} B0 单点作用 {show(entry['proximal'])}，等幅 B0 对相邻原生控制 {show(entry['proximal_matched_vs_neighbor'])}。"
                     f"本边候选相对交叉候选的原生优势 {show(entry['own_vs_cross_joint'])}。")
        for route,rr in entry['routes'].items():
            g=rr['GU_restore']
            lines.append(f"- {e}/{route} 等距等幅覆盖 {rr['targets_equal']:.1%}，G/U 等幅覆盖 {rr['receiver_equal']:.1%}。"
                f"G、U 单独恢复对该边干预量的有向改动 {show(g['freeze_g'])} / {show(g['freeze_u'])}。"
                f"占自然边效应 {rr['GU_share']['freeze_g']:+.1%} / {rr['GU_share']['freeze_u']:+.1%}。"
                f"B32 其他 token 后缀状态的剩余贡献 {show(rr['suffix_other_than_E'])}，"
                f"B32 两路非加和 {show(rr['split_interaction'])}，路径门 {'通过' if rr['path_pass'] else '未通过'}。")
    for route,points in n['dose'].items():
        doses='、'.join(f"第{i+1}段 {d['coverage']:.1%}（实际夹角余弦绝对值 P95 {d['orthogonality_abs_p95']:.3f}）"
                       for i,d in enumerate(points))
        lines.append(f"- {route} 路径剂量覆盖 {doses}。余弦按 BF16 注入后的实际控制方向对同位原生方向算。")
    for e,r in f['edges'].items():
        x=r['counterexamples'][0]
        lines.append(f"- {e} 保留留出反例或最大预测误差 {x['family']}，前后效应 {x['effects'][0]:+.4f}→{x['effects'][1]:+.4f}，"
            f"首 token {x['tokens']!r}。逐条反例、剂量和几何可从原始分片复算。")
    for route,detail in n['layouts']['2'].items():
        x=detail['counterexamples'][0]
        lines.append(f"- 晚档案、早请求布局的 {route} 联合干预反例 {x['id']}，自然差 {x['natural']:+.4f}，"
                     f"朝自然供体方向的干预效应 {x['joint']:+.4f}。")
    legacy=data['old_marker_native']
    if legacy:
        lines.extend(['',
            f"- 旧标记原生复跑于 {data['old_marker_native_plan']['at']} 预先冻结。旧标记 128 家族均值通过正、负、正门后，"
            "复用同 64 家族、路径、候选单元与控制。旧、新标记的干预位置和几何逐条相同。旧标记原生复位与空操作全词表逐位一致。",
            '',
            '| 旧标记待解释边、路径 | 原生联合所占比例 | 等距等幅优势 | B32 残差 / MLP 单路比例 | G/U 等幅优势 | 路径门 |',
            '| --- | ---: | --- | --- | --- | --- |'])
        for e,entry in legacy['edges'].items():
            for route,rr in entry['routes'].items():
                p=rr['shares']
                lines.append(f"| {e} / {route} | {p['joint']:+.1%} | {show(rr['matched_vs_scramble'])} | "
                             f"{p['skip']:+.1%} / {p['mlp']:+.1%} | {show(rr['GU_over_random'])} | "
                             f"{'通过' if rr['path_pass'] else '未通过'} |")
        lines.extend(['',f"- 旧标记中跨两条边的共同路径 {legacy['shared_routes'] or '零条'}。"
                          "原生复跑与换标记行为配对属于同家族敏感性检验。"])
        for edge,modes in data['marker_native_difference'].items():
            for route,values in modes.items():
                lines.append(f"- {edge}/{route} 的新减旧标记联合原生效应 {show(values['joint'])}，"
                    f"旁路差 {show(values['skip'])}、MLP 差 {show(values['mlp'])}。"
                    "配对同一家族和几何，仍受同时换回两个标记的限制。")
    else:
        lines.append('- 旧标记三布局的自然效应未通过正、负、正门，按冻结规则不进入旧标记干预。')
    lines.extend([
        f"- 跨两条边都通过冻结门的共同路径 {n['shared_routes'] or '零条'}。共同的 B0 首邻响应显示档案信息先进入同类 token。"
        '同一条瓶颈需要同一条路径的两次干预都能移走足够的位置效应。',
        '- B32 分解在自由起始词位分别保留编辑侧 attention 残差加基线 MLP，或基线残差加编辑侧 MLP。非起始词位置回到原态。'
        'G/U 的方向性和回归干预前输出的绝对距离分别登记。写入与后续预测、以及真正影响输出分别取证。',
        '- 外部图谱新增等词面的独立位置反转与转正对照。内部图谱新增全位置残差的三个边界、逐级距离与匹配记录。'
        '两图谱之间新增交叉留出预测和原生干预矩阵。下一阶段可冻结更宽的后缀或较早的跨 token 写入组，匹配几何和写入剂量，再检验哪些位置合在一起能减少两条边的效应，并在连续生成中检查格式与内容的副作用。',
        f"- 数据、结构图与运行日志在 tests/glm5/result/{OUT.name}。上一轮 1111 二阶极端样本仍绑定在前序分析，本轮反例侧重位置与路径。"
    ])
    if legacy:
        all_estimates=[r['shares']['joint'] for data in (n,legacy) for group in data['edges'].values()
                       for r in group['routes'].values()]
        lines.append(f"- 新旧标记下八个联合效应比例介于 {min(all_estimates):+.1%} 和 {max(all_estimates):+.1%}。"
            "候选在早层可读、可跟踪，交叉阻断只移走小部分自然效应。旧标记恢复了明确的反转与转正，"
            "因此两条受测路径在该条件下也无法承担整个效应。是否还存在共同的更宽路径仍待检验。")
    return '\n'.join(lines)+'\n'


def run(close):
    from transformers import AutoTokenizer

    d=R.read('design.json');c=R.read('candidate.json');p=R.read(N.PLAN)
    assert d['code']==A.identity() and d['previous']==A.prior_identity()
    assert p['code']==N.source_id() and p['design']==N.sha('design.json') and p['candidate']==N.sha('candidate.json')
    assert (OUT/'candidate.json').stat().st_mtime_ns<(OUT/N.PLAN).stat().st_mtime_ns
    assert all((OUT/N.PLAN).stat().st_mtime_ns<(OUT/(n+'-files.json')).stat().st_mtime_ns for n in ('confirm','native'))
    mark=R.read(K.FILE)
    assert mark['code']==K.source() and mark['holdout']==N.sha(N.PLAN)
    assert mark['already_exposed_confirm']==N.sha('confirm-files.json')
    assert (OUT/K.FILE).stat().st_mtime_ns<(OUT/'marker-files.json').stat().st_mtime_ns
    legacy_plan=R.read(L.PLAN)
    legacy_gate=R.read(L.GATE)
    assert legacy_plan['code']==L.source() and legacy_plan['marker']==N.sha(K.FILE)
    assert legacy_gate['plan']==N.sha(L.PLAN) and legacy_gate['material']==N.sha('marker-files.json')
    assert (OUT/L.PLAN).stat().st_mtime_ns<(OUT/'marker-files.json').stat().st_mtime_ns
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    clean=S.load('confirm',N.sha(N.PLAN));raw=S.load('native',N.sha(N.PLAN))
    old_clean=S.load('marker',N.sha(K.FILE))
    old_native=None
    difference=None
    if legacy_gate['passed']:
        old_raw=S.load('marker-native',N.sha(L.PLAN))
        assert np.array_equal(raw['ids'],old_raw['ids']) and np.array_equal(raw['geometry'],old_raw['geometry'])
        old_native=full_native(old_raw,c,tok)
        old_native['overlap']=audit_alignment(old_clean,old_raw)
        difference=marker_native_difference(raw,old_raw)
    old=json.loads((A.PREVIOUS/'analysis.json').read_bytes())['location']['1111']
    record=dict(at=C.stamp(),design=d,candidate=c,holdout=p,gate=R.read('behavior.json'),
        previous=dict(individual=old['edges']['0-2']['individual_reversal'],old_layouts=[old['layouts'][i]['mean'] for i in M.LAYOUTS]),
        overlap=audit_alignment(clean,raw),field=field(clean,c,tok),forecast=forecasts(clean,c,tok),
        native=full_native(raw,c,tok),marker=marker_control(clean,old_clean),marker_plan=mark,
        old_marker_native=old_native,old_marker_native_plan=legacy_plan,old_marker_native_gate=legacy_gate,
        marker_native_difference=difference)
    R.write('analysis.json',record)
    chart(record)
    text=render(record)
    check_write(OUT/'report.md',len(text.encode())+4096,operation='相同路径总览写入').write_text(
        '# '+TITLE+'\n\n'+text,encoding='utf-8')
    if not close:
        return
    assert not (OUT/'memo-receipt.json').exists() and TITLE not in MEMO.read_text(encoding='utf-8-sig')
    memo=append(check_write(MEMO,len(text.encode())+4096,operation='GLM5 MEMO 追加'),TITLE,text)
    cleanup=[]
    if A.TEMP.is_dir():
        folder=A.TEMP.resolve()
        assert folder==A.TEMP and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
        owned=[folder,*folder.rglob('*')]
        assert all(not x.is_symlink() and not x.is_junction() and x.resolve().is_relative_to(folder) for x in owned)
        freed=sum(x.stat().st_size for x in owned if x.is_file())
        shutil.rmtree(folder)
        cleanup.append(dict(path=str(folder),freed_bytes=freed))
    R.write('memo-receipt.json',dict(memo=memo,cleanup=cleanup))
    manifest={p.name:dict(sha256=A.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    sources={x.relative_to(R.ROOT).as_posix():A.sha(x) for x in (Path(__file__).resolve(),Path(N.__file__).resolve(),Path(K.__file__).resolve(),
               Path(L.__file__).resolve(),
               R.ROOT/'tests/glm5/paradigm_position_report.py')}
    R.write('manifest.json',dict(run=M.RUN,files=manifest,sources=d['code']|sources))
    print('GLM5 Phase',memo['phase'],'appended',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--close',action='store_true')
    args=parser.parse_args()
    A.setup()
    run(args.close)
