"""Audit the decisive Qwen3-14B test and append the bounded research verdict."""

from hashlib import sha256
from itertools import product
from pathlib import Path
import time

import numpy as np

import paradigm_verdict_run as V
import paradigm_verdict_native as N
from deploy.storage_policy import check_write,staging_file
from paradigm_routes_report import family_summary as ci

R,W,S,C,M,OUT=V.R,V.W,V.S,V.C,V.M,V.OUT
Q=V.RULES


def forecast(raw,candidate):
    y,x=V.predictors(raw)
    _,_,forms=V.movement(raw)
    lex=forms[:,0]//8
    assert (np.unique(lex,return_counts=True)[1]==64).all()
    guess={name:V.predict(v,candidate['fits'][name]) for name,v in x.items()}

    def compare(mask):
        actual=y[mask]
        classified={name:(g[mask]>0)==(actual>0) for name,g in guess.items()}
        fields={name:dict(sign=ci(flag.mean(1)),mae=float(abs(actual-guess[name][mask]).mean()))
                for name,flag in classified.items()}
        lift=classified['native'].astype(float)-classified['text']
        text_lift=classified['text'].astype(float)-classified['geometry']
        fields['native_gain']=ci(lift.mean(1)); fields['text_gain']=ci(text_lift.mean(1))
        fields['native_mse_gain']=ci(((actual-guess['text'][mask])**2-(actual-guess['native'][mask])**2).mean(1))
        return fields

    overall=compare(np.ones(len(y),dtype=bool))
    wording={str(int(k)):compare(lex==k) for k in np.unique(lex)}
    p=overall['native_gain']
    passed=(overall['native']['sign']['mean']>=Q['sign_accuracy'] and p['mean']>=Q['sign_gain'] and p['ci'][0]>0
            and all(row['native_gain']['mean']>=0 for row in wording.values()))
    text=overall['text_gain']
    text_pass=overall['text']['sign']['mean']>=Q['sign_accuracy'] and text['mean']>=Q['sign_gain'] and text['ci'][0]>0
    ids=raw['ids'].reshape(-1,8,4)
    order=np.argsort(-abs(y-guess['native']).ravel(),kind='stable')[:8]
    cases=[dict(id=ids[i,j].tolist(),outcome=float(y[i,j]),prediction=float(guess['native'][i,j]),
                text=float(guess['text'][i,j]),wording=int(lex[i])) for i,j in zip(*np.unravel_index(order,y.shape))]
    return dict(all=overall,by_wording=wording,passed=bool(passed),text_pass=bool(text_pass),counterexamples=cases)


def conditions(clean,raw_bridge,c):
    y,_,forms=V.movement(clean)
    old,_,_=V.movement(raw_bridge,2)
    lex=forms[:,0]//8
    assert all((forms[lex==k]%8==np.arange(8)).all() for k in (0,1))
    effects={};pairs=[]
    for name,bit in zip(('headers','facts','instruction'),(4,2,1),strict=True):
        cells=[(i,i|bit) for i in range(8) if not i&bit]
        average=(y[:,[b for a,b in cells]]-y[:,[a for a,b in cells]]).mean(1)
        effects[name]=dict(pooled=ci(average),by_wording={str(k):ci(average[lex==k]) for k in (0,1)})
        for a,b in cells:
            panels={str(k):[ci(y[lex==k,i]) for i in (a,b)] for k in (0,1)}
            direction=int(np.sign(y[:,b].mean()-y[:,a].mean()))
            if direction>0:
                flip=all(d[0]['ci'][1]<0<d[1]['ci'][0] for d in panels.values())
            else:
                flip=all(d[1]['ci'][1]<0<d[0]['ci'][0] for d in panels.values())
            pairs.append(dict(domain=name,indices=[a,b],direction=direction,
                              by_wording=panels,robust_flip=bool(flip)))
    bridge=[ci(a) for a in old.T]
    top=[ci(y[:,j]) for j in c['corners']]
    top_wording={str(k):[ci(y[lex==k,j]) for j in c['corners']] for k in (0,1)}
    return dict(bridge=bridge,bridge_pass=bool(bridge[0]['ci'][1]<0<bridge[1]['ci'][0]),
        main_effects=effects,pairs=pairs,corners=top,corners_by_wording=top_wording,
        corners_pass=bool(all(vals[0]['ci'][1]<0<vals[1]['ci'][0] for vals in top_wording.values())),
        all_means=[ci(a) for a in y.T])


def causality(raw,clean,c):
    arms=N.ARMS
    j=list(dict.fromkeys(raw['family']))
    lookup={name:i for i,name in enumerate(j)}
    families=np.asarray([lookup[id_] for id_ in raw['family']])
    corners=np.asarray([c['corners'].index(int(k)) for k in raw['corner']])
    layouts=raw['layout']-2
    idx=(families,corners,layouts)
    shape=(len(j),2,2)
    assert len(set(zip(*idx)))==len(j)*4==M.CAUSAL*4
    pool=dict(zip(clean['ids'],zip(clean['margin'][:,1],clean['top'][:,0]),strict=True))
    for key,side,top in [('ids','original','original_top'),('ids','reference','reference_top')]:
        names=raw[key] if side=='original' else [id_.replace('-a1-','-a0-') for id_ in raw[key]]
        assert all((value,token)==pool[id_] for id_,value,token in zip(names,raw[side],raw[top],strict=True))
    unpatched=raw['original']-raw['reference']
    assert np.array_equal(unpatched,raw['positive_control'])
    shifts=raw['original'][:,None]-raw['edit']
    assert (shifts[:,(arms.index('sham'),arms.index('restore'))]==0).all()
    d=np.empty(shape);d[idx]=unpatched
    delta=np.empty((*shape,len(arms)));delta[idx]=shifts
    coverage={};audit={}
    for name in (*arms[4:],):
        good=np.ones(len(raw['original']),dtype=bool)
        for k,values in raw.items():
            if not (k.startswith('dose-') and k.endswith('-'+name)):
                continue
            goal,actual,cos,old,_=values.transpose(2,0,1)
            mask=goal>0
            relative=abs(goal-actual)/np.maximum(np.maximum(goal,actual),1e-12)
            a,b=[np.linalg.norm(x,axis=1) for x in (goal,actual)]
            total_gap=abs(a-b)/np.maximum(np.maximum(a,b),1e-12)
            ok=(~mask|((relative<=Q['dose'])&(cos>=.90))).all(1)&(total_gap<=Q['dose'])
            good &= ok
            audit[k]=dict(row_cover=float(ok.mean()),active_tokens=int(mask.sum()),
                failed_norm=int((mask&(relative>Q['dose'])).sum()),failed_angle=int((mask&(cos<.90)).sum()),
                zero_direction=int((mask&(old==0)).sum()))
        z=np.empty(shape,dtype=bool);z[idx]=good
        coverage[name]=z.all(2)

    findings={}
    for i,(label,sign) in enumerate(zip(c['corners'],c['directions'],strict=True)):
        natural=(d[:,i,1]-d[:,i,0])*sign
        effect=(delta[:,i,1]-delta[:,i,0])*sign
        stats={'natural':ci(natural)}
        for arm in ('full','last20'):
            e=effect[:,arms.index(arm)]
            e_ci=ci(e)
            cover={}
            for ctrl in ('perm','natural'):
                key=arm+'_'+ctrl;matched=coverage[key][:,i]
                diff=e-effect[:,arms.index(key)]
                cover[ctrl]=dict(coverage=float(matched.mean()),advantage=ci(diff[matched]) if matched.sum()>=4 else None,
                                 all_rows=ci(diff))
            commands=dict(natural=stats['natural']['ci'][0]>0,effect=e_ci['ci'][0]>0,
                          closure=bool(natural.mean()>0 and e.mean()>=Q['minimum_closure']*natural.mean()))
            for ctrl,q in cover.items():
                a=q['advantage']
                commands[ctrl+'_cover']=q['coverage']>=Q['cover']
                commands[ctrl+'_advantage']=a is not None and a['mean']>=Q['minimum_margin_advantage'] and a['ci'][0]>0
            k=int(np.argmin(e));indices=np.flatnonzero((families==k)&(corners==i))
            take=corners==i
            excluded=[kind for kind,q in cover.items() if q['coverage']>=Q['cover'] and q['advantage'] is not None
                      and q['advantage']['ci'][1]<Q['minimum_margin_advantage']]
            stats[arm]=dict(effect=e_ci,controls=cover,checks=commands,passed=all(commands.values()),
                            practically_insufficient_controls=excluded,
                            fraction=float(e.mean()/natural.mean()) if abs(natural.mean())>1e-8 else None,
                            changed=int((raw['top'][take,arms.index(arm)]!=raw['original_top'][take]).sum()),
                            counterexample=dict(family=str(j[k]),effect=float(e[k]),ids=raw['ids'][indices].tolist()))
        findings[str(label)]=stats
    shared=[arm for arm in ('last20','full') if all(g[arm]['passed'] for g in findings.values())]
    return dict(corners=findings,shared=shared,doses=audit,positive_control_rows=len(raw['positive_control']),
                ordinary_rows=len(raw['positive_control'])*2)


def fmt(stat):
    return f"{stat['mean']:+.3f} [{stat['ci'][0]:+.3f}, {stat['ci'][1]:+.3f}]"


def put(name,text):
    data=text.encode('utf-8')
    with staging_file(check_write(OUT/name,len(data)+1024)) as pending:
        pending.write_bytes(data)


def write_memo(text):
    path=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    label='## Phase 2979: 等长句式拆解与原生路径路线判定 '
    if label.encode() in path.read_bytes():
        raise FileExistsError('Phase 2979 already appended')
    raw=f"\n\n{label}[{time.strftime('%Y-%m-%d %H:%M')}]\n\n{text.rstrip()}\n".encode('utf-8')
    with check_write(path,len(raw)).open('ab') as handle:
        handle.write(raw)
    assert path.read_bytes().endswith(raw)
    R.write('memo-receipt.json',dict(phase=2979,at=C.stamp(),bytes=len(raw),sha256=sha256(raw).hexdigest()))


def save_manifest():
    files={p.name:dict(bytes=p.stat().st_size,sha256=W.sha(p)) for p in sorted(OUT.iterdir())
           if p.is_file() and p.name not in ('manifest.json','memo-receipt.json')}
    sources=R.read('design.json')['code']
    if (OUT/'candidate.json').exists():
        sources.update({f'tests/glm5/{k}':v for k,v in R.read('candidate.json')['execution'].items()})
    R.write('manifest.json',dict(run=M.RUN,files=files,sources=sources))


def save_cases(stats,d,n,binding):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    rows=M.build(tok,'confirm')
    assert M.W.digest(rows)==d['materials']['confirm']['sha256']
    all_inputs={r['id']:r for panel in rows[:2] for r in panel}
    keys={k for case in stats['counterexamples'] for k in case['id']}
    if n:
        keys.update(k for entry in n['corners'].values() for arm in ('full','last20') for k in entry[arm]['counterexample']['ids'])
    R.write('counterexamples.json',dict(binding=binding,forecast=stats['counterexamples'],
        native={key:{arm:q[arm]['counterexample'] for arm in ('full','last20')} for key,q in n['corners'].items()} if n else {},
        prompts={k:{field:all_inputs[k][field] for field in ('prompt','free_ids','frame','layout','archive','corner','gold')}
                 for k in sorted(keys)}))


def run():
    gate=R.read('behavior.json')
    if not gate['passed']:
        raise RuntimeError('Pilot did not qualify. Retain failure before choosing a new frozen material.')
    d,c,binding=V.checked_plan()
    clean=S.load('confirm',binding)
    forecasts=forecast(clean,c)
    shape=conditions(clean,S.load('bridge',binding),c)
    eligible=R.read('heldout-behavior.json')
    native=causality(S.load('native',binding),clean,c) if eligible['passed'] else None
    passed=(forecasts['passed'] and shape['bridge_pass'] and shape['corners_pass'] and native is not None and bool(native['shared']))
    decision='prospective_gate_pass' if passed else 'not_adopted'
    generated=None
    if passed and (OUT/'generation-files.json').exists():
        index=R.read('generation-files.json'); assert index['holdout']==binding
        views=[]
        for ref in index['files']:
            assert W.sha(OUT/ref['name'])==ref['sha']
            views.extend(R.read(ref['name'])['views'])
        generated={name:{'n':sum(v['arm']==name for v in views),
                          'joint':sum(v['arm']==name and v['score']['joint'] for v in views)}
                   for name in ('prose','code','full','last20')}
    summary=dict(at=C.stamp(),binding=binding,decision=decision,scope=d['scope'],forecasts=forecasts,
                 conditions=shape,heldout_behavior=eligible,causal=native,generation=generated)
    R.write('analysis.json',summary)
    if passed and generated is None:
        print('Prospective margin route passed. Run generate and report again to finish the frozen output test.',flush=True)
        return
    save_cases(forecasts,d,native,binding)
    decision_text=('在这两个受控句式的首词格式偏好上可复现' if passed else
                   '当前固定残差路径不能作为跨句式的共同机制采用')
    text=[f'- 路线判定。{decision_text}。判定限于本地 Qwen3-14B BF16、等长档案／请求任务和预注册效果门。Goodfire 启发的观测、对照与干预方法可继续使用，当前候选路径的结论不能外推到整个模型的可解释性。',
          f"- 样本与冻结。24 试跑，32 发现，128 独立确认，64 原生干预预算，128 锁箱只哈希。两个等长措辞版本各 64 确认家族，其中版本 1 没进入拟合。试跑所有成对格式格最低 {gate['minimum']:.1%}，因果候选确认复核最低 {eligible['minimum']:.1%}。家族为复算及 99% bootstrap 单位。",
          '- 实际请求的任务、档案位置与输出位置固定。独立交换档案／请求标题、两句事实模板、题目语言／计数／括号说明。相同家族每次变化只在已验证、互不重叠的 token 位置集合。最后的通用格式说明固定。前两轮完整句式另作原样桥接。',
          '- 先算格式间隔 `m = logit(print) − logsumexp(论文开头候选)`，求档案式样的差 `D = m(代码档案) − m(论文档案)`，再用 `Y = D(晚请求) − D(早请求)` 给请求后移取号。负值表示请求后移使代码档案更偏向压低该格式分数。',
          f"- 原样桥接在 64 个新家族的两个旧完整句式为 {fmt(shape['bridge'][0])} 和 {fmt(shape['bridge'][1])}，过门 {shape['bridge_pass']}。冻结选中等长角 {c['corners']}，两个措辞版本均保留相反方向的门 {shape['corners_pass']}。"]
    for name,e in shape['main_effects'].items():
        text.append(f"- {name} 换为第二水平的平均主效应 {fmt(e['pooled'])}。两个措辞版本分别为 {fmt(e['by_wording']['0'])}／{fmt(e['by_wording']['1'])}。")
    robust=[(p['domain'],p['indices']) for p in shape['pairs'] if p['robust_flip']]
    text.append(f'- 两种等长措辞下都复现的单区段符号翻转 {robust}。完成 12 个预定义相邻对，区间未做跨对校正，保留为条件结构线索。')
    for model in ('geometry','text','residual','native'):
        p=forecasts['all'][model];text.append(f"- {model} 留出方向命中 {p['sign']['mean']:.1%}，MAE {p['mae']:.3f}。")
    text.append(f"- 文本相对几何的方向准确率增益 {fmt(forecasts['all']['text_gain'])}，通过 {forecasts['text_pass']}。原生状态加文本相对文本的增益 {fmt(forecasts['all']['native_gain'])}，通过 {forecasts['passed']}。")
    if native:
        text.append(f"- 正对照直接把供体的原生末层 E 状态放到接收端，{native['positive_control_rows']} 个视图都还原了供体整个词表的 logits。{native['ordinary_rows']} 个干净接收端与供体的分数和首选 token 对上常规留出结果。真正候选窗只在 E 前修改，空操作和原生恢复也逐位一致。")
        for corner,q in native['corners'].items():
            text.append(f"- 冻结角 {corner} 的自然边沿预先方向为 {fmt(q['natural'])}。")
            for arm in ('full','last20'):
                row=q[arm];p=row['controls']['perm'];b=row['controls']['natural']
                pretty=lambda item:fmt(item['advantage']) if item['advantage'] is not None else '合格对不足'
                text.append(f"  - {arm} 削弱量 {fmt(row['effect'])}。超过置换方向 {pretty(p)}，覆盖 {p['coverage']:.1%}。超过天然方向 {pretty(b)}，覆盖 {b['coverage']:.1%}。过门 {row['passed']}，反例 {row['counterexample']['family']}。")
                if row['practically_insufficient_controls']:
                    text.append(f"    - 这些合格对照下的优势 99% 区间上限低于事前的 0.10 logit，{row['practically_insufficient_controls']}。")
    else:
        text.append('- 新材料的行为门阻止原生干预。候选未取得机制资格，该执行结果分开登记。')
    if generated:
        text.append(f'- 自然生成的完整通过计数 {generated}。只在锁定的首词证据过门后运行。')
    else:
        text.append('- 冻结的后续自然续写仅在新预测和新干预同时通过才启动。Phase 2978 已独立评估旧候选的 256 个续写视图。')
    text.extend(['- 研究边界。这里的原生残差窗是条件受控的计算假说。先独立定位可以预测符号且有实际效应的输入条件，再追 QK 来源、V/O 写入、G/U 和参数级读出。末层正对照验证测量链，不能替代跨位置因果路径。单一快照、固定任务、人工句式仍限制外推。',
                 f'- 原始结果、冻结哈希、剂量和反例存于 {OUT.relative_to(R.ROOT).as_posix()}。所有 GPU 阶段使用同一快照顺序运行，未开封锁箱。'])
    body='\n'.join(text)+'\n'
    put('report.md','# Phase 2979 · 等长句式拆解与路线判定\n\n'+body)
    save_manifest();write_memo(body)
    print(OUT/'report.md',flush=True)
