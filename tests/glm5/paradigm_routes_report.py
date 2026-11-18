"""Quantitative attention/MLP fusion maps, discovery-frozen forecasts, and append-only review."""

from collections import defaultdict
from pathlib import Path
from itertools import product
from shutil import rmtree
import time

import numpy as np

import paradigm_routes as X
import paradigm_routes_material as M
import record_update_study as S
from b36_semantic14_memo import append
from deploy.storage_policy import check_write
from kv_origin_audit import summarize


R,C,OUT=X.R,X.C,X.OUT
TITLE='Goodfire 原生神经元与语言范式的联合破解、注意力竞争和融合'
AXES=('requested_format','reference_format','answer_language','request_reference','request_language','reference_language','triple')
CONTRAST=np.asarray([[(-1.)**sum(cell[j] for j in axes) for cell in M.LEVELS]
                     for axes in ((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))],np.float32)
CONTRAST[:3]*=-1


def unbits(x):
    import torch

    return torch.as_tensor(np.ascontiguousarray(x)).view(torch.bfloat16).float().numpy()


def shards(split,conditional=False):
    stem=split+('-conditional' if conditional else '')
    for receipt in sorted(OUT.glob(stem+'-???.json')):
        row=R.read(receipt.name)
        assert row['plan']==C.A.digest(OUT/'plan.json') and row['sha']==C.A.digest(OUT/row['file'])
        yield X.read_np(row['file'])


def field(data):
    assert len(data)%8==0
    shape=(len(data)//8,8,*data.shape[1:])
    return np.einsum('ac,nc...->na...',CONTRAST,data.reshape(shape),optimize=True)/4


def family_summary(v):
    return summarize(np.asarray(v,dtype=np.float64),.99)


def pilot(plan,a,b,tok):
    rows=a['rows']
    z=list(shards('pilot'))
    onset=np.concatenate([r['top'] for r in z])
    scripts=np.concatenate([r['script'] for r in z])
    conditional=np.concatenate([r['top'] for r in shards('pilot',True)])
    assert len(rows)==len(onset)==len(conditional)
    chars=tok.batch_decode([[int(v)] for v in conditional])
    is_zh=np.asarray([any('\u3400'<=char<='\u9fff' for char in word) for word in chars])
    cells=[]
    for i,(target,reference,lang) in enumerate(M.LEVELS):
        picked=np.arange(i,len(rows),8)
        print_=(onset[picked]==plan['anchors']['print'])
        language=is_zh[picked]==bool(lang)
        cells.append(dict(regime=[target,reference,lang],n=len(picked),print_rate=float(print_.mean()),
                          style_hit=float((print_==bool(target)).mean()),conditional_language_hit=float(language.mean())))
    passed=all(r['style_hit']>=.75 and r['conditional_language_hit']>=.75 for r in cells)
    zh_odds=scripts[:,0]-scripts[:,1]
    order=(zh_odds.reshape(-1,2)[:,1]>zh_odds.reshape(-1,2)[:,0]).mean()
    passed=passed and order>=.75
    record=dict(at=C.stamp(),passed=bool(passed),cells=cells,language_odds_increase_fraction=float(order),
                first_examples=[dict(id=row['id'],regime=row['regime'],token=int(token),text=tok.decode([int(token)]),
                conditional=tok.decode([int(later)])) for row,token,later in zip(rows,onset,conditional,strict=True)],
                scope='First-token format and language after supplied print prefix; full natural answers unmeasured')
    R.write('pilot-gate.json',record)
    print('Pilot paradigm onset/conditional',passed,cells,flush=True)
    return record


def read_features(tok,split,units,random):
    rows=list(M.material(tok,split))
    targets=[];output=[];features=[];randoms=[];head=[];headnorm=[]
    for z in shards(split):
        targets.append(unbits(z['target']))
        output.append(z['logits'])
        act=unbits(z['A'])
        features.append(np.stack([act[:,l,u] for l,u in units],1))
        randoms.append(np.stack([act[:,l,u] for l,u in random],1))
        head.append(z['head_effect'])
        headnorm.append(z['head_norm'])
    return rows,np.concatenate(targets),np.concatenate(output),np.concatenate(features),np.concatenate(randoms),np.concatenate(head),np.concatenate(headnorm)


def text_features(rows):
    design=[]
    for row in rows:
        a,b,c=row['regime']
        ids=np.asarray(row['token']['ids'],np.int64)
        unigram=np.bincount(ids%64,minlength=64)/len(ids)
        bigram=np.bincount((ids[1:]*39+ids[:-1]*49)%64,minlength=64)/max(len(ids)-1,1)
        design.append(np.r_[1,a,b,c,a*b,a*c,b*c,a*b*c,row['later'],row['gold']/1000,len(ids)/160,unigram,bigram])
    return np.asarray(design,np.float32)


def select_and_maps(split):
    totals=None
    graphs=defaultdict(list)
    fields=[]
    for z in shards(split):
        act=unbits(z['A'])
        contributions=act*z['down'][None]
        contrast=field(contributions)
        score=np.abs(contrast[:,0]).sum(0)
        totals=score if totals is None else totals+score
        graphs['head_effect'].append(field(z['head_effect']))
        graphs['attention_mass'].append(field(z['alpha']))
        graphs['source_score'].append(field(z['scores']))
        graphs['unit_axis_mean'].append(contrast.sum(0))
        fields.append(field(unbits(z['field'])).astype(np.float32))
    count=sum(len(a) for a in graphs['head_effect'])
    # Keep all coordinates in the response map; selection serves the next frozen step only.
    graph={k:np.concatenate(v) for k,v in graphs.items() if k!='unit_axis_mean'}
    graph['unit_axis_mean']=sum(graphs['unit_axis_mean'])/count
    graph['hidden_axis_response']=np.concatenate(fields)
    X.write_np(split+'-maps.npz',graph)
    return totals/count,graph


def freeze_candidates(plan,tok):
    if (OUT/'candidate-freeze.json').exists():
        record=R.read('candidate-freeze.json')
        assert record['fit_sha']==C.A.digest(OUT/'forecast-fit.npz')
        return record
    score,maps=select_and_maps('discovery')
    excluded=np.ones_like(score,dtype=bool)
    excluded[12:32]=False
    score[excluded]=-np.inf
    flat=np.argsort(-score.ravel(),kind='stable')[:32]
    units=[list(map(int,np.unravel_index(v,score.shape))) for v in flat]
    rng=np.random.default_rng(M.SEED)
    occupied={tuple(v) for v in units}
    random=[]
    for layer,_ in units:
        candidates=[u for u in range(score.shape[1]) if (layer,u) not in occupied]
        unit=int(rng.choice(candidates))
        random.append([layer,unit]);occupied.add((layer,unit))
    effect=np.abs(maps['head_effect'][:,0]).mean(0)
    effect[:8]=-np.inf;effect[32:]=-np.inf
    layer,head=map(int,np.unravel_index(effect.argmax(),effect.shape))
    rows,target,_,native,rand,heads,norms=read_features(tok,'discovery',units,random)
    ratio=np.median(np.minimum(norms[:,layer],norms[:,layer,head,None])/np.maximum(np.maximum(norms[:,layer],norms[:,layer,head,None]),1e-6),axis=0)
    pool=sorted((h for h in range(40) if h//5!=head//5),key=lambda h:(-ratio[h],h))[:8]
    control=int(rng.choice(pool))
    baseline=text_features(rows)
    fits={}
    for method,extra in [('text',None),('native',native),('random',rand)]:
        inputs=baseline if extra is None else np.c_[baseline,extra,heads[:,layer,head]]
        fit=S.linear(inputs,target,12.)
        fits.update({f'{method}.{k}':v for k,v in fit.items()})
    fitsha=X.write_np('forecast-fit.npz',fits)
    record=dict(at=C.stamp(),plan=C.A.digest(OUT/'plan.json'),units=units,random_units=random,
                head=[layer,head],control_head=control,control_pool=pool,fit_sha=fitsha,
                maps_sha=C.A.digest(OUT/'discovery-maps.npz'),axes=AXES,
                interpretation='All maps descriptive of original native responses. Discovery-selected unit writes and head responses local to frozen query.')
    R.write('candidate-freeze.json',record)
    return record


def vocab_scores(model,target,estimate):
    torch=model.t
    values=[]
    for offset in range(0,len(target),8):
        x=torch.as_tensor(target[offset:offset+8],device='cuda',dtype=torch.bfloat16)
        y=torch.as_tensor(estimate[offset:offset+8],device='cuda',dtype=torch.bfloat16)
        lp=model.model.lm_head(x).float().log_softmax(-1)
        lq=model.model.lm_head(y).float().log_softmax(-1)
        values.append(torch.stack(((lp.exp()*(lp-lq)).sum(-1),(lp.argmax(-1)==lq.argmax(-1)).float()),-1).cpu().numpy())
    return np.concatenate(values)


def confirm(plan,frozen,tok,model):
    if (OUT/'confirmation.json').exists():return
    score,maps=select_and_maps('confirm')
    rows,target,observed,native,random,heads,_=read_features(tok,'confirm',frozen['units'],frozen['random_units'])
    fits=X.read_np('forecast-fit.npz')
    inputs=text_features(rows)
    metrics={};arrays={}
    layer,head=frozen['head']
    for method,extra in [('text',None),('native',native),('random',random)]:
        x=inputs if extra is None else np.c_[inputs,extra,heads[:,layer,head]]
        fit={k.removeprefix(method+'.'):v for k,v in fits.items() if k.startswith(method+'.')}
        prediction=S.evaluate(fit,x)
        estimates=vocab_scores(model,target,prediction)
        arrays[method]=estimates
        per_family=estimates.reshape(-1,8,2).mean(1)
        metrics[method]=dict(kl=family_summary(per_family[:,0]),top1=family_summary(per_family[:,1]))
    delta=(arrays['text'][:,0]-arrays['native'][:,0]).reshape(-1,8).mean(1)
    gain=family_summary(delta)
    gain['relative']=gain['mean']/max(metrics['text']['kl']['mean'],1e-8)
    gain['passed']=gain['ci'][0]>0 and gain['relative']>=.1 and gain['n']>=64
    X.write_np('confirm-predictions.npz',arrays)
    rows,report=map_summary(tok,plan,frozen,maps,score)
    R.write('confirmation.json',dict(at=C.stamp(),metrics=metrics,gain=gain,response_report=report,
        intervention_admitted=bool(gain['passed'] and R.read('pilot-gate.json')['passed']),
        freeze=C.A.digest(OUT/'candidate-freeze.json')))
    if gain['passed'] and R.read('pilot-gate.json')['passed']:
        intervene(model,plan,frozen)


def map_summary(tok,plan,frozen,maps,score):
    units=[]
    total=0;selected=0;survived=[]
    for z in shards('confirm'):
        act=unbits(z['A'])
        projected=act*z['down'][None]
        effects=field(projected)
        total+=float((effects[:,0]**2).sum())
        for l,u in frozen['units']:
            selected+=float((effects[:,0,l,u]**2).sum())
        survived.append(effects[:,0,[l for l,u in frozen['units']],[u for l,u in frozen['units']]])
    for index,(layer,unit) in enumerate(frozen['units']):
        vals=np.concatenate(survived)[:,index]
        units.append(dict(layer=layer,unit=unit,requested_style_write_effect=family_summary(vals)))
    mass=maps['attention_mass']
    write=maps['head_effect']
    layer,head=frozen['head']
    report=dict(selected_units=units,
        selected_sq_write_effect_fraction=selected/max(total,1e-12),
        attention_head=[layer,head],
        head_main_effects={axis:family_summary(write[:,i,layer,head]) for i,axis in enumerate(AXES)},
        attention_mass_effects={axis:{region:family_summary(mass[:,i,layer,head,k]) for k,region in enumerate(X.REGIONS)}
            for i,axis in enumerate(AXES)},
        nonlinear_interaction_ratio=float(np.mean(write[:,3:]**2)/max(np.mean(write**2),1e-12)))
    return units,report


def intervene(model,plan,frozen):
    """Single-head original output removal with matched projected dose and exact self-restoration."""
    layer_id,head=frozen['head'];control=frozen['control_head']
    torch=model.t
    rows=list(M.material(model.tok,'confirm'))
    result=[]
    started=time.monotonic()
    names=('sham','target-off','matched-target','matched-control','native-restore')
    for part,batch_rows in enumerate(batched(rows,64)):
        panels=[C.packet(model,b) for b in batched(batch_rows,X.BATCH)]
        for index in range(40):
            with model.block(index) as block:
                for p in panels:
                    x,mask=p['x'].cuda(),p['mask'].cuda()
                    angles=[y.cuda() for y in p['angles']]
                    q,k,v=model.qkv(block,x,angles)
                    o=model.attention(block,q,k,v,mask)
                    j=torch.arange(len(x),device='cuda');c=p['c'].cuda();b=len(j);arms=len(names)
                    if index>=layer_id:
                        live=x[j,c].expand(arms,-1,-1).clone() if index==layer_id else p['live'].cuda()
                        aa=[value[j,c,None].repeat(arms,1,1) for value in angles]
                        qq,kk,vv=model.qkv(block,live.reshape(arms*b,1,model.width),aa)
                        branch_k=k.repeat(arms,1,1,1);branch_v=v.repeat(arms,1,1,1)
                        idx=torch.arange(arms*b,device='cuda');cc=c.repeat(arms)
                        branch_k[idx,:,cc]=kk[:,:,0];branch_v[idx,:,cc]=vv[:,:,0]
                        visible=mask[j,:,c].unsqueeze(2).repeat(arms,1,1,1)
                        writes=model.attention(block,qq,branch_k,branch_v,visible).reshape(arms,b,40,128)
                        if index==layer_id:
                            before=writes[0].clone()
                            doses=model.projections(block,(head,control),before[:,[head,control]]).norm(dim=-1)
                            goal=doses.min(-1).values
                            writes[1,:,head]=0
                            measured=[]
                            for arm_id,h,pos in ((2,head,0),(3,control,1)):
                                factor=(goal/doses[:,pos].clamp_min(1e-8))[:,None]
                                writes[arm_id,:,h]=(before[:,h].float()*(1-factor)).to(writes.dtype)
                                delta=writes[arm_id,:,h].float()-before[:,h].float()
                                measured.append(model.projections(block,(h,),delta[:,None]).norm(dim=-1)[:,0])
                            writes[4]=before
                            p['dose']=torch.stack((goal,*measured),-1).cpu().numpy()
                        p['live']=model.residual(block,live,writes).cpu()
                    p['x']=model.residual(block,x,o).cpu()
        for p in panels:
            logits=model.logits(p['live'].cuda()).float()
            direction=logits[:,:,plan['anchors']['print']]-logits[:,:,plan['anchors']['The']]
            assert torch.equal(logits[0],logits[4])
            result.append(dict(ids=np.array([r['id'] for r in p['rows']]),margin=direction.T.cpu().numpy(),
                               top=logits.argmax(-1).T.cpu().numpy(),dose=p['dose']))
        C.N.progress('confirmation-head-edit',min(len(rows),(part+1)*64),len(rows),started)
    joined={k:np.concatenate([r[k] for r in result]) for k in result[0]}
    X.write_np('confirm-intervention.npz',joined)
    sign=np.asarray([1 if r['regime'][0] else -1 for r in rows])
    effect=(joined['margin'][:,3]-joined['margin'][:,2])*sign
    relative=abs(joined['dose'][:,1]-joined['dose'][:,2])/np.maximum(joined['dose'][:,0],.02)
    stat=family_summary(effect.reshape(-1,8).mean(1))
    p95=float(np.quantile(relative,.95))
    R.write('confirm-intervention.json',dict(at=C.stamp(),arms=names,head=frozen['head'],control=control,
        matched_loss=stat,dose_p95=p95,passed=bool(stat['mean']>=.1 and stat['ci'][0]>0 and p95<=.03),
        changed_top={name:int((joined['top'][:,i]!=joined['top'][:,0]).sum()) for i,name in enumerate(names)},
        native_self_restore_exact=True))


def render(plan,record):
    pilot=R.read('pilot-gate.json')
    confirm=R.read('confirmation.json')
    gain=confirm['gain'];maps=confirm['response_report']
    f=R.read('candidate-freeze.json')
    from record_update_report import table,show

    cells=[[r['regime'],r['n'],r['print_rate'],r['style_hit'],r['conditional_language_hit']] for r in pilot['cells']]
    units=[[u['layer'],u['unit'],show(u['requested_style_write_effect'])] for u in maps['selected_units'][:10]]
    head=[[axis,show(value)] for axis,value in maps['head_main_effects'].items()]
    predictions=[[method,show(value['kl']),show(value['top1'])] for method,value in confirm['metrics'].items()]
    intervention=R.read('confirm-intervention.json') if (OUT/'confirm-intervention.json').exists() else None
    return f"""### 双路线主线与当前范围

根据用户最新指令，核心问题是 LLM 内部如何由有限的原生组件提取、保持、融合与竞争语言范式。Goodfire 原生路线沿单元归一输入、门分支 G、值分支 U、实际激活 A 与 down 写出追踪。语言范式路线沿 attention 原生 Q/K、按实验区域的选择比例、V/O 写入及下游 MLP 追踪。二者在同一次前向上对应到具体坐标、单元和参数。外部格式／语言标签定义材料的因素；对应的内部计算由自然响应、留出预测及干预按各自证据登记。

本轮 `{M.RUN}`，模型 Qwen3-14B 原始 BF16 `{plan['snapshot']['snapshot']}`。原创事实计数任务，交叉请求的论文式／Python、档案示例的论文式／Python、输出英文／中文，8 视图同内容、同查询位置和 token 长度。试跑 16、发现 48、确认 64 家族，联合锁箱 128 家族封存。确认改写和数量组合未参加选取。读取真实自由回答的首词分布，另在试跑使用指定 `print("` 前缀的条件语言查询；完整回答与自然语法正确性未测。

### 计算定义与自然结构

对源位置 i，α_i = softmax(q·k_i/√128)，头写入 y_h = W_O^h Σ_i α_i v_i。一次 SwiGLU 的 x 为原生 post-attention RMSNorm 输出，G_j=g_jᵀx，U_j=u_jᵀx，A_j=SiLU(G_j)U_j，单坐标写入为 W_down[k,j]A_j。本轮保存所有40层×40头的分组质量与未归一化组 logsumexp、原生写出的 print−The 投影，以及全部 40层×17408单元的 BF16 G/U/A 和对应固定读出方向系数。原生单元的整段计算与原模型 residual 在试跑三个冻结层逐位对账。每条输入另保留41隐藏边界和全词表 logits。最终归一化、非线性交互和备份路径均需单独解释。

融合测量使用完整 2×2×2 矩阵。因素的主效应和两两、三阶交互均从同家族八视图求正交对比。交互表示联合条件的响应偏离所测加法项，具体机制还需由 QK、归一化和 MLP 门乘积逐项对账。选择候选使用发现集的本地写出投影，在12–31层固定32个原生单元、8–31层固定一个注意力头和同层随机对照。确认同一编号检验，有效维度同时保留全坐标图。

试跑格式及条件语言口径通过 `{pilot['passed']}`，语言条件引起的汉字概率优势增幅比例 `{pilot['language_odds_increase_fraction']:.3f}`。

{table(['请求/示例/语言 0或1','家族','首词print比例','首词格式命中','条件语言命中'],cells)}

确认选中头 `{f['head']}`，原生写出各因素对比为下表。

{table(['因子或交互','print−The 投影对比 [99% CI]'],head)}

固定32单元中前10个的确认写出对比如下。选择范围内的投影只解释这个输出差方向。

{table(['块','MLP单元','请求格式写出对比 [99% CI]'],units)}

这32单元占全部原生单元在请求格式对比上的平方投影响应份额 `{maps['selected_sq_write_effect_fraction']:.4f}`。头写出交互项均方相对全部对比均方的描述性比 `{maps['nonlinear_interaction_ratio']:.4f}`。完整注意力分组／单位写出图在 discovery-maps.npz 与 confirm-maps.npz，可按原生坐标重新分析。

### 留出预测与原生作用

文本基线输入实际请求、示例、输出语言及其交互、数量选择、词序和长度。native 加32个冻结原生激活与冻结头的早期写出摘要，random 加同层32个随机单元与相同头摘要。发现拟合末层归一状态 ĥ，确认经原 lm_head 比较全词表 KL。两个增强方法使用额外内部信息。它们的增量刻画上游组件对后续分布的预测价值。

{table(['方法','全词表KL [99% CI]','真实首选匹配 [99% CI]'],predictions)}

native 相对文本的家族均值 KL 改善为 `{gain['relative']:.2%}`，差值区间 `{gain['ci']}`，冻结预测门 `{gain['passed']}`。试跑行为与预测联合准入 `{confirm['intervention_admitted']}`。

{str(intervention) if intervention else '本轮原生头干预未进入；自然结构图与响应预测按实际范围保留。'}

### 研究判断、问题和下一大阶段

本轮优先以神经元计算和注意力范式选择为联合对象。范式可以是语言、文体、知识关系、角色或语法条件，对应候选对象允许分散到多个位置、层和原生组件。注意力 softmax 提供同查询下的来源竞争，跨头残差相加和 SwiGLU 提供不同形式的组合。把这些恒等式升级为某种语言范式的机制，需要能解释多任务复用、条件切换和未见组合的因果响应。论文式和程序代码含有不同语法与格式，当前只测其入口和条件语言；本轮24种轴对比支持域、角色与数字事实仍有局限。自然信号分类、局部投影、预测保真和原生干预分别登记。

后续一个大阶段围绕同一内容交叉语言、文体、逻辑关系及语法槽位；保留多 token 词、共用标点、切换片段、代码中的中文注释与开放文本。先用全原生场找复用写出，再把范式源 token→QK 选择→V/O 写入→MLP G/U/A→下一层源或输出连起来。两因素对照决定融合位置，冲突源距离、数量和强度决定竞争条件，指定参数与原生恢复追踪真实因果。最终把冻结组件与其接续规律用于未见内容和未见范式组合的完整输出预测，并复核独立节点。RDC 沿用有域条件转移，没有新增统一定律。

### 复核与资源

入口 tests/glm5/paradigm_routes.py freeze、all、close。计划 `{record['plan']}`，权重/分词器与软件版本均绑定，CUDA/共享显存及逐层进度见 progress.jsonl，原始证据保留D盘。正式登记需要独立复算和发布授权。
"""


def close(plan):
    import json

    execution=R.read('execution.json')
    assert execution['plan']==C.A.digest(OUT/'plan.json')
    assert execution['candidate_freeze']==C.A.digest(OUT/'candidate-freeze.json')
    assert not list(OUT.glob('lockbox-*'))
    for split in ('pilot','discovery','confirm'):
        for _ in shards(split):pass
    assert (OUT/'candidate-freeze.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in OUT.glob('confirm-???.json'))
    body=render(plan,execution)
    check_write(OUT/'report.md',len(body.encode())+4096,operation='联合范式研究报告').write_text('# '+TITLE+'\n\n'+body,encoding='utf-8')
    memo=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    assert TITLE not in memo.read_text(encoding='utf-8-sig')
    added=append(check_write(memo,len(body.encode())+4096,operation='追加神经元范式联合研究'),TITLE,body)
    ref=R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    index=append(check_write(ref,4096,operation='追加联合研究索引'),TITLE,f"详见 GLM5 Phase {added['phase']}，运行 `{M.RUN}`。")
    cleaned=[]
    if X.TEMP.is_dir():
        scratch=X.TEMP
        assert scratch.resolve()==scratch and scratch.parent==(R.ROOT/'tests/glm5_temp').resolve()
        children=[scratch,*scratch.rglob('*')]
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(scratch) for p in children)
        size=sum(p.stat().st_size for p in children if p.is_file())
        rmtree(scratch);cleaned.append(dict(path=str(scratch),bytes=size))
    R.write('memo-receipt.json',dict(records=[added,index],cleanup=cleaned))
    files={p.name:dict(bytes=p.stat().st_size,sha256=C.A.digest(p)) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    R.write('manifest.json',dict(files=files,sources=plan['sources']))
    print('Appended phase',added['phase'],flush=True)
