"""Family-level audit and append-only report for the frozen window campaign."""

from collections import defaultdict
import ast
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

import paradigm_window_retest as retest
import paradigm_window_run as W
from deploy.storage_policy import check_write,staging_file
from paradigm_routes_report import family_summary as ci


def clean_scores(raw,candidate):
    d,y,x=W.edges(raw)
    ids=raw['ids'].reshape(-1,2,3,2)
    frame=raw['frame'][::12]
    states={}
    for position,edge in enumerate(('archive','request')):
        actual=y[:,:,position]
        pred=candidate['forecasts']
        baseline=np.asarray(pred['b0'][edge]['baseline'])
        stats=dict(natural=ci(actual.mean(1)),by_marker=[ci(v) for v in actual.T],
                   reversal=((d[:,:,position]>0)&(d[:,:,position+1]<0) if edge=='archive'
                             else (d[:,:,position]<0)&(d[:,:,position+1]>0)).sum(0).tolist(),
                   layout_means=d.mean(0).tolist(),baseline_mae=float(abs(actual-baseline).mean()),
                   by_frame={str(int(f)):[ci(q) for q in actual[frame==f].T] for f in np.unique(frame)})
        unfavorable=np.argmax(actual,axis=0) if edge=='archive' else np.argmin(actual,axis=0)
        stats['counterexamples']=[dict(marker=i,views=ids[j,i].tolist(),effect=float(actual[j,i]),
                                            layouts=d[j,i].tolist()) for i,j in enumerate(unfavorable)]
        for name,columns in candidate['feature_index'].items():
            p=pred[name][edge]
            xx=x[:,:,position,columns]
            yhat=baseline+((xx-np.asarray(p['mean']))/np.asarray(p['spread']))@np.asarray(p['beta'])
            improve=((actual-baseline)**2-(actual-yhat)**2).mean(1)
            stats[name]=dict(mae=float(abs(actual-yhat).mean()),mse_gain=ci(improve),
                sign_by_frame={str(int(f)):dict(correct=int(((yhat[frame==f]>0)==(actual[frame==f]>0)).sum()),
                    baseline_correct=int(((baseline>0)==(actual[frame==f]>0)).sum()),total=int(actual[frame==f].size))
                    for f in np.unique(frame)})
        states[edge]=stats
    return states


def native_scores(raw,clean,arms):
    refs=dict(zip(clean['ids'],zip(clean['margin'][:,1],clean['top'][:,0])))
    keys=list(dict.fromkeys(raw['family']))
    family={name:i for i,name in enumerate(keys)}
    markers,layouts=len(W.M.MARKERS),len(W.M.LAYOUTS)
    shape=(len(keys),markers,layouts)
    j=np.asarray([family[key] for key in raw['family']])
    m,l=raw['marker'],np.asarray([W.M.LAYOUTS.index(k) for k in raw['layout']])
    family_frames=np.full(len(keys),-1,dtype=int)
    family_frames[j]=raw['frame']
    assert (family_frames[j]==raw['frame']).all()
    assert len(set(zip(j,m,l)))==len(raw['ids'])==np.prod(shape)
    pack=np.empty((*shape,len(arms)))
    pack[j,m,l]=raw['original'][:,None]-raw['edit']
    d=np.empty(shape);d[j,m,l]=raw['original']-raw['reference']
    a=np.array([refs[name][0] for name in raw['ids']]); assert np.array_equal(raw['original'],a)
    assert np.array_equal(raw['original_top'],np.asarray([refs[name][1] for name in raw['ids']]))
    prose=[key.replace('-a1-','-a0-') for key in raw['ids']]
    assert np.array_equal(raw['reference'],np.asarray([refs[name][0] for name in prose]))
    assert np.array_equal(raw['reference_top'],np.asarray([refs[name][1] for name in prose]))
    assert (pack[...,arms.index('sham')]==0).all() and (pack[...,arms.index('restore')]==0).all()
    coverage={};markers_pass={};control_doses={}
    for name in arms:
        parts=[key for key in raw if key.startswith('dose-') and key.endswith('-'+name)]
        if not parts:
            continue
        row=np.ones(len(j),dtype=bool)
        for key in parts:
            a,b,cos,original,align=raw[key].transpose(2,0,1)
            active=a>0
            relative=abs(a-b)/np.maximum(np.maximum(a,b),1e-12)
            stage_a=np.linalg.norm(a,axis=1);stage_b=np.linalg.norm(b,axis=1)
            stage_gap=abs(stage_a-stage_b)/np.maximum(np.maximum(stage_a,stage_b),1e-12)
            good=(~active|((relative<=.03)&(cos>=.90))).all(1)&(stage_gap<=.03)
            row&=good
            control_doses[key]=dict(eligible=float(good.mean()),active_tokens=int(active.sum()),
                token_ok=float(((relative[active]<=.03)&(cos[active]>=.9)).mean()) if active.any() else None,
                failed_norm=int((active&(relative>.03)).sum()),failed_direction=int((active&(cos<.90)).sum()),
                zero_donor_direction=int((active&(original==0)).sum()),
                relative_gap_p95=float(np.quantile(relative[active],.95)) if active.any() else None,
                alignment_p95=float(np.quantile(abs(align[active]),.95)) if active.any() else None,
                natural_raw_multiplier_median=float(np.median((b/np.maximum(original,1e-12))[active])) if active.any() else None)
        coverage[name]=np.zeros(shape,dtype=bool)
        coverage[name][j,m,l]=row

    result={}
    labels=('archive','request')
    for e,edge in enumerate(labels):
        orientation=(-1,1)[e]
        normal=(d[:,:,e+1]-d[:,:,e])*orientation
        effect=(pack[:,:,e+1]-pack[:,:,e])*orientation
        sets={}
        for marker in range(markers):
            def M(name):
                return effect[:,marker,arms.index(name)]
            natural=normal[:,marker]
            denom=float(natural.mean())
            tests={}
            for name in arms:
                rows=(raw['marker']==marker)&np.isin(raw['layout'],[W.M.LAYOUTS[e],W.M.LAYOUTS[e+1]])
                vals=M(name)
                tests[name]=dict(effect=ci(vals),fraction=float(vals.mean()/denom) if abs(denom)>1e-6 else None,
                                 top_changes=int((raw['top'][rows,arms.index(name)]!=raw['original_top'][rows]).sum()),
                                 top_n=int(rows.sum()))
            for name in ('full','last20','suffix'):
                vals=M(name)
                for ctrl in ('perm','natural'):
                    key=f'{name}_{ctrl}'
                    matched=coverage[key][:,marker,e:e+2].all(1)
                    dif=vals-M(key)
                    tests[name][ctrl]=dict(coverage=float(matched.mean()),qualified=int(matched.sum()),
                                           advantage=ci(dif[matched]) if matched.sum()>=4 else None,
                                           all_rows_diagnostic=ci(dif))
                a=tests[name]
                a['checks']=dict(direction=denom>0,closure=a['fraction'] is not None and a['fraction']>=.10,
                                 effect=a['effect']['ci'][0]>0)
                for ctrl in ('perm','natural'):
                    a['checks'][ctrl+'_coverage']=a[ctrl]['coverage']>=.98
                    a['checks'][ctrl+'_advantage']=a[ctrl]['advantage'] is not None and a[ctrl]['advantage']['ci'][0]>0
                a['passed']=all(a['checks'].values())
            comparison=dict(retention_gap=ci(M('last20')-.70*M('full')),
                 drop8=ci(M('full')-M('drop8')),drop20=ci(M('full')-M('drop20')),
                 dropboth=ci(M('full')-M('dropboth')),
                 early_interaction=ci(M('full')-M('drop8')-M('drop20')+M('dropboth')),
                 restore_late20=ci(M('full')-M('late20_restore')))
            bad=int(np.argmin(M('full')))
            tested=keys[bad]
            examples=[dict(id=str(raw['ids'][i]),natural=float(raw['original'][i]-raw['reference'][i]),
                           full_delta=float(raw['original'][i]-raw['edit'][i,arms.index('full')]),
                           original_top=int(raw['original_top'][i]),full_top=int(raw['top'][i,arms.index('full')]))
                      for i in range(len(j)) if j[i]==bad and m[i]==marker and l[i] in (e,e+1)]
            strata={str(int(frame)):dict(natural=ci(natural[family_frames==frame]),
                       full=ci(M('full')[family_frames==frame]),late20=ci(M('last20')[family_frames==frame]))
                    for frame in np.unique(family_frames)}
            sets[str(marker)]=dict(natural=ci(natural),arms=tests,contrasts=comparison,exploratory_by_frame=strata,
                                  counterexample=dict(family=str(tested),effect=float(M('full')[bad]),views=examples))
        markers_pass[edge]=all(row['arms']['full']['passed'] for row in sets.values())
        result[edge]=sets
    result['primary']=dict(full_pass=markers_pass['request'],archive_full_pass=markers_pass['archive'],
        localized=markers_pass['request'] and all(q['arms']['last20']['passed'] and q['contrasts']['retention_gap']['ci'][0]>0
                                                 for q in result['request'].values()))
    result['doses']=control_doses
    result['identity']=dict(natural_rows=2*len(j),exact_margins_and_top=True,restore_and_sham_checked_by_runner=True)
    counts=raw['query'][:,0]-raw['sites'][:,4]
    result['full_suffix_length']=dict(min=int(counts.min()),max=int(counts.max()))
    return result


def generation(binding):
    index=W.R.read('generation-files.json')
    assert index['holdout']==binding
    views=[]
    for item in index['files']:
        assert W.sha(W.OUT/item['file'])==item['sha256']
        receipt=W.R.read(item['file'])
        assert receipt['holdout']==binding and receipt['native']==W.sha(W.OUT/'native-files.json')
        views.extend(receipt['views'])
    assert len({(v['id'],v['arm']) for v in views})==W.M.GENERATE*2*2*4
    panels=defaultdict(dict)
    for v in views:
        panels[(v['marker'],v['layout'],v['arm'])][v['family']]=v
    out={}
    for (marker,layout,arm),group in panels.items():
        assert len(group)==W.M.GENERATE
        base=panels[(marker,layout,'code')]
        out[f'{marker}/{layout}/{arm}']=dict(n=len(group),
            passed={key:int(sum(v['score'][key] for v in group.values())) for key in next(iter(group.values()))['score']},
            differences=int(sum(v['tokens']!=base[k]['tokens'] for k,v in group.items())),
            joint_delta=sum(int(v['score']['joint'])-int(base[k]['score']['joint']) for k,v in group.items()),
            next_match=sum(v['check']['equal'] is True for v in group.values()),
            next_compared=sum(v['check']['equal'] is not None for v in group.values()),
            exploratory_static_print=sum(static_print(v['text']) for v in group.values()))
    failed=[v for v in views if v['check']['equal'] is False]
    total={}
    for arm in ('prose','code','full','last20'):
        groups=[v for k,v in out.items() if k.endswith('/'+arm)]
        total[arm]=dict(n=sum(v['n'] for v in groups),joint=sum(v['passed']['joint'] for v in groups),
                        changes=sum(v['differences'] for v in groups),stopped=sum(v['passed']['stopped'] for v in groups),
                        wider_static_format=sum(v['exploratory_static_print'] for v in groups))
    return dict(panels=out,total_views=len(views),cache_check_mismatches=failed,
                total_by_arm=total,cache_next_checks=sum(v['check']['equal'] is not None for v in views),
                examples=[v for v in views if v['arm'] in ('full','last20') and not v['score']['joint']][:8])


def static_print(text):
    """Exploratory format audit; parse but never execute generated text."""
    try:
        root=ast.parse(text.strip(),mode='eval').body
    except (SyntaxError,ValueError):
        return False
    if not (isinstance(root,ast.Call) and isinstance(root.func,ast.Name) and root.func.id=='print'
            and len(root.args)==1 and not root.keywords):
        return False
    pending=[root.args[0]]
    while pending:
        value=pending.pop()
        if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add):
            pending.extend((value.left,value.right))
        elif not (isinstance(value,ast.Constant) and isinstance(value.value,str)):
            return False
    return True


def display(stat):
    return f"{stat['mean']:+.4f} [{stat['ci'][0]:+.4f}, {stat['ci'][1]:+.4f}]"


def write_md(name,text):
    raw=text.encode('utf-8')
    with staging_file(check_write(W.OUT/name,len(raw)+1024)) as file:
        file.write_bytes(raw)


def manifest(path):
    out={p.name:dict(bytes=p.stat().st_size,sha256=W.sha(p)) for p in sorted(path.iterdir())
         if p.is_file() and p.name not in ('manifest.json','memo-receipt.json')}
    d=W.R.OUT
    try:
        W.R.OUT=path
        sources=W.R.read('design.json')['code']
        if (path/'candidate.json').exists():
            sources.update({f'tests/glm5/{k}':v for k,v in W.R.read('candidate.json')['execution'].items()})
        sources[f'tests/glm5/{Path(__file__).name}']=W.sha(Path(__file__))
        W.R.write('manifest.json',dict(run=W.R.read('design.json')['run'],files=out,sources=sources))
    finally:
        W.R.OUT=d


def append_memo(body):
    memo=W.R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    header='## Phase 2978: 固定晚档案的 B31 残差窗口、分段去除及自然生成 '
    if header.encode() in memo.read_bytes():
        raise FileExistsError('Phase 2978 is already present')
    raw=f"\n\n{header}[{time.strftime('%Y-%m-%d %H:%M')}]\n\n{body.rstrip()}\n".encode('utf-8')
    with check_write(memo,len(raw)).open('ab') as stream:
        stream.write(raw)
    assert memo.read_bytes().endswith(raw)
    W.R.write('memo-receipt.json',dict(at=W.C.stamp(),phase=2978,appended_sha256=sha256(raw).hexdigest(),bytes=len(raw)))


def resources():
    progress=[json.loads(line) for line in (W.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    return {key:max((p.get(key) or 0) for p in progress)
            for key in ('dedicated_gpu_bytes','shared_gpu_bytes','cuda_peak_bytes')}


def save_counterexamples(forecast,causal,free,digest):
    from transformers import AutoTokenizer

    tokenizer=AutoTokenizer.from_pretrained(W.R.MODEL,local_files_only=True)
    material=W.M.material(tokenizer,'confirm')
    assert W.M.digest(material)==W.R.read('design.json')['materials']['confirm']['sha256']
    identify={p['id']:p for p in material[0]}
    suffixes={tuple(p['free_ids'][p['query'][0]-20:p['query'][0]]) for p in material[0]}
    shared=dict(distinct_token_sequences=len(suffixes),
                text=tokenizer.decode(next(iter(suffixes)),skip_special_tokens=False) if len(suffixes)==1 else None)
    keys=set()
    for edge in ('archive','request'):
        keys.update(k for example in forecast[edge]['counterexamples'] for row in example['views'] for k in row)
        for evidence in causal[edge].values():
            for example in evidence['counterexample']['views']:
                code=example['id']
                keys.update(code.replace('-a1-',f'-a{a}-') for a in (0,1,2))
    keys.update(sample['id'] for sample in free['examples'])
    keys.update(sample['id'] for sample in free['cache_check_mismatches'])
    W.R.write('counterexamples.json',dict(holdout=digest,material_sha256=W.M.digest(material),
        inputs={key:{field:identify[key][field] for field in ('prompt','free_ids','marker','layout','archive','frame','gold')}
                for key in sorted(keys)},
        predicted={edge:forecast[edge]['counterexamples'] for edge in ('archive','request')},
        intervention={edge:{mark:v['counterexample'] for mark,v in causal[edge].items()} for edge in ('archive','request')},
        generation=free['examples'],cache_mismatches=free['cache_check_mismatches']))
    return shared


def main():
    old=json.loads((retest.ORIGINAL/'behavior.json').read_bytes())
    cur=W.R.read('behavior.json')
    assert not old['passed']
    first=float(np.min(np.asarray(old['rates'])))
    second=float(np.min(np.asarray(cur['rates'])))
    lines=[f'- v1 在确认材料打开前停门。24 个试跑家族，最小格式成对率 {first:.1%}，失败集中在首试句式的晚档案、晚请求。v2 统一为已验证的显式回答说明，换独立试跑／发现词表，重新冻结，保留 .75 的原门。v2 最小成对率 {second:.1%}。',
           '- 本轮本地 Qwen3-14B 原生 BF16。发现与确认使用不同句式，家族按实体与计数对生成。加速使用原生序列 shape 重跑前缀，配置和模型哈希随冻结保存。128 个联合锁箱家族仅做材料哈希。']
    lines.append('- v1 与 v2 同时更换回答说明及试跑词表。通过 v2 说明新版本足以恢复配对区分，单独归因两项变化仍需同家族对照。')
    if not cur['passed']:
        lines+=['- v2 行为门仍未通过，停止确认前向与干预。失败样本、成对输出、计划与日志均保留。待厘清失误来自晚请求的语言理解或首词读出，再冻结独立材料。']
        W.R.write('analysis.json',dict(status='behavior_gate_failed',previous=first,repeat=second,behavior=cur))
    else:
        binding=W.sha(W.OUT/'candidate.json'); candidate=W.R.read('candidate.json')
        clean=W.S.load('confirm',binding); raw=W.S.load('native',binding)
        stats=clean_scores(clean,candidate); cause=native_scores(raw,clean,list(candidate['arms'])); free=generation(binding)
        gate=W.R.read('confirm-frame-gate.json')
        usage=resources()
        shared=save_counterexamples(stats,cause,free,binding)
        W.R.write('analysis.json',dict(at=W.C.stamp(),holdout=binding,design=W.sha(W.OUT/'design.json'),
                                      forecasts=stats,causal=cause,generation=free,frame_gate=gate,resource_peaks=usage,
                                      last20_surface=shared))
        lines+=['- 格式分数 `m = logit(print) − logsumexp(论文候选)`。先算 `D_l = m(代码档案) − m(论文档案)`，再比较档案后移 `D₂ − D₀` 和请求后移 `D₃ − D₂`。干预削弱量是自然边减去干预后的边，档案采用预注册反方向取号。区间按家族 bootstrap 99%。',
                f"- 128 家族清洁留出。请求后移跨占位平均 {display(stats['request']['natural'])}。新窗的预测比较以发现组标记均值为底线，使用向量在冻结格式方向的投影，无目标 logit 输入。档案／请求的底线 MAE 分别为 {stats['archive']['baseline_mae']:.4f}／{stats['request']['baseline_mae']:.4f}，B31 后20为 {stats['archive']['late20']['mae']:.4f}／{stats['request']['late20']['mae']:.4f}。请求后移 MSE 增益 {display(stats['request']['late20']['mse_gain'])}。",
                f"- 新句式的 32 家族格式对照在任何原生干预前全部过门，逐格最低 {gate['minimum']:.1%}。两标记的自然档案反转 {stats['archive']['reversal']}／128，请求转正 {stats['request']['reversal']}／128。",
                '- 原生干预 64 家族。保持 B0/B8/B20，B31 分别保留整个后缀、前40、后40或后20。另将 B8 与 B20 的去除组成因子实验，给单独 B31 也设置匹配对照。等距对照用同一位置，用实际 BF16 逐词和整段剂量核对，另有保持代码格式但改变档案载荷的天然方向。全量原生空操作和复原一致。',
                '- 判据要求两标记均达到正向衰减至少 10%，两类对照的家族匹配覆盖至少 98%，优势的 99% 区间高于零。后20 定位另要求其相对 70% 宽路径的保留量可信为正。天然方向来自实际同格式前向，缩放后仅保持方向及注入量的口径。',
                '', '| 边／占位 | 宽路径削弱量 | 前40 | 后40 | 后20 | B31 单独 | 宽／后20是否过门 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for edge in ('request','archive'):
            for marker,q in cause[edge].items():
                a=q['arms']
                cells=' | '.join(f"{a[key]['effect']['mean']:+.4f}" for key in ('full','first40','last40','last20','suffix'))
                lines.append(f"| {edge}／m{marker} | {cells} | {a['full']['passed']}／{a['last20']['passed']} |")
        lines.append('')
        for f,vals in stats['request']['by_frame'].items():
            lines.append(f"- 请求后移在句式 {f} 的自然效果，按两套占位分别为 {display(vals[0])}／{display(vals[1])}。")
        sign=stats['request']['late20']['sign_by_frame']
        lines.append('- 后20 冻结读出按句式计算的方向命中为 ' + '、'.join(
            f"{f} 的 {p['correct']}/{p['total']}（均值底线 {p['baseline_correct']}/{p['total']}）" for f,p in sign.items())
            + '。两个标记共用同一家族，方向命中的观察格数不能当独立家族数。')
        if shared['distinct_token_sequences']==1:
            lines.append('- 按实际 tokenizer 核对，所有确认视图的后20是同一串 token，覆盖格式说明结尾和助手角色标记。预测使用的是同一串 token 在不同前史下的残差，词串身份本身无法区分这些视图。')
        lines.append('- 新句式使用成组措辞变化，现有数据可定位到句式条件。哪个词、短语或绝对位置触发换符号，仍需做逐段等长最小对。')
        lines.append('- 表中的削弱量以 logit 间隔为单位，按主判预定方向取号。四组原生样本的自然边区间均跨零或落在反方向。带符号比例留在分析中，接近零或反向的自然效应会使比例无法直观解释。')
        for marker,q in cause['request'].items():
            a=q['arms'];v=q['contrasts'];c=q['counterexample'];full=a['full']
            lines.append(f"- m{marker} 请求后移：宽路径对等剂量置换／天然方向的严格家族覆盖 {full['perm']['coverage']:.1%}／{full['natural']['coverage']:.1%}。去掉 B8／B20 的变化为 {display(v['drop8'])}／{display(v['drop20'])}。B31 后20原值装回引起的宽路径减幅 {display(v['restore_late20'])}。反例 {c['family']} 削弱量为 {c['effect']:+.4f}。其余反例和每个控制的实际剂量见分析及分片。")
            lines.append(f"  - 宽路径首选词改变 {full['top_changes']}/{full['top_n']}。后20保留量相对于 .70×宽路径的差 {display(v['retention_gap'])}。B8 与 B20 非加和 {display(v['early_interaction'])}。")
            lines.append(f"  - 同一原定方向下，宽路径绝对削弱量 {display(full['effect'])}，后20绝对削弱量 {display(a['last20']['effect'])}。")
        lines.append(f"- 请求宽路径总门 {cause['primary']['full_pass']}，后20定位门 {cause['primary']['localized']}，档案移动宽路径总门 {cause['primary']['archive_full_pass']}。正的部分原生作用和足够的匹配覆盖各自登记。")
        dose=cause['doses']['dose-31-full_natural']
        lines.append(f"- B31 宽路径天然方向的实际范数失配 {dose['failed_norm']}/{dose['active_tokens']} 个 token，方向余弦失配 {dose['failed_direction']} 个。逐词覆盖 {dose['token_ok']:.2%}，而逐视图要求全段都合格，覆盖为 {dose['eligible']:.2%}。这一执行瓶颈与置换对照充足但门失败分别登记。")
        lines.append(f"- 自由生成采用 16 新家族、{free['total_views']} 个完整读出臂，按真实首词开始，无预填目标答案。24 token 上限，达到上限计为未终止。程序比较第一步可对齐的全形状与 KV 缓存解码，分歧 {len(free['cache_check_mismatches'])}/{free['cache_next_checks']}。")
        for arm,v in free['total_by_arm'].items():
            lines.append(f"  - {arm} 严格全对 {v['joint']}/{v['n']}，相对同家族 code 回答改变 {v['changes']}/{v['n']}，终止 {v['stopped']}/{v['n']}。放宽语法，仅静态接受字面字符串及字符串拼接时，Python 格式 {v['wider_static_format']}/{v['n']}，此口径是补充诊断。")
        for key,q in free['panels'].items():
            if '/3/' in key:
                v=q['passed'];lines.append(f"  - {key} 格式／中文／计数／方括号／完全正确分别 {v['format']}/{v['language']}/{v['count']}/{v['square']}/{v['joint']}，每格 {q['n']}。")
        lines+=['- 支撑的研究层次限于原生残差与下一词偏好。只有窗口同时过效应、两种匹配对照和留出预测，才适合将后续预算投入该窗内的 QK、V/O 或 G/U 的源与写入权重。连续生成样本量较小，代码、中文和格式评分均保留失败串。整段先前同剂量的结果与本轮逐 token 匹配主判采用不同对照口径。',
                '- 理论上优先检验条件化路径能否跨不同句式维持方向与相对比例。通过才进一步追踪可复用的计算边。在信号散布、控制方向也可造成同等变化时，应先重建多位置贡献和恢复后的剩余效应，再缩减原子单位。']
        if not cause['primary']['localized']:
            lines.append('- 下一阶段先对两类句式做区段最小对，分别替换槽位标题、事实句式和格式说明，等长固定位置并将档案两种载荷独立交叉。要求只用已出现的线索预测换位效应符号，再用新的独立家族确认。B31 后20保留为观察读出，匹配失败只在独立材料上校准，避免用已开封家族调控制。')
        lengths=cause['full_suffix_length']
        lines.append(f"- 本轮 B31 完整后缀为 {lengths['min']}–{lengths['max']} 个位置。次要宽度只作探索。GPU 日志采样 CUDA 峰值 {usage['cuda_peak_bytes']/2**30:.2f} GiB，共享显存最高 {usage['shared_gpu_bytes']/2**20:.0f} MiB。")
    lines.append(f"- 本轮材料 SHA、冻结方案、实际进度与所有分片在 {W.OUT.relative_to(W.R.ROOT).as_posix()}。原始 v1 的停门记录在 {retest.ORIGINAL.relative_to(W.R.ROOT).as_posix()}。")
    lines.append('- 执行入口为 `paradigm_window_run.py freeze|pilot`，随后依次运行 `paradigm_window_retest.py` 的 `freeze`、`pilot`、`smoke`、`candidate`、`confirm`、`native`、`generate`，最后用 `paradigm_window_report.py` 从分片重新计分。模型前向使用本项目 `.venv` 的 Python。')
    body='\n'.join(lines)+'\n'
    write_md('report.md','# Phase 2978 · 请求后移的后缀窗口定位\n\n'+body)
    manifest(retest.ORIGINAL);manifest(W.OUT)
    append_memo(body)
    print(W.OUT/'report.md',flush=True)


if __name__=='__main__':
    with retest.configured():
        main()
