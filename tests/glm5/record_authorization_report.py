"""Recompute paired outcomes and append the completed authorization study."""

from collections import Counter, defaultdict
from itertools import batched
from shutil import rmtree

import record_authorization_material as A
import record_update_report as prior
from deploy.storage_policy import check_write
from kv_origin_audit import summarize


R,C = prior.R,prior.R.C
TITLE = 'Qwen3-14B 单 token 授权对照、自然记录更新与机制准入'
LABELS = prior.LABELS


def category(row,out):
    if not out['format']:
        return 'format'
    names = row['names']
    return ('old','other_initial','earlier_proposal','focal_proposal')[names.index(out['state'])] if out['state'] in names else 'outside'


def readback(plan,stages):
    from transformers import AutoTokenizer

    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    review={}
    for stage,result in stages.items():
        rows=[A.encode(tok,row) for row in A.families(tok,stage)]
        groups=defaultdict(Counter)
        patterns=Counter()
        effects=[]
        cases=[]
        for pair,outputs in zip(batched(rows,2),batched(result['records'],2),strict=True):
            ordered=sorted(zip(pair,outputs,strict=True),key=lambda x:x[0]['update'])
            views=[]
            for row,output in ordered:
                key=(row['frame'],*row['setting'],row['update'])
                group=groups[key]
                status=category(row,output)
                group['n']+=1
                group[status]+=1
                for metric in ('joint','format','stop','state_ok','code_ok'):
                    group[metric]+=int(output[metric])
                label=output['state']
                if label in row['names']:
                    group['known_label']+=1
                    group['code_follows_emitted']+=int(row['codes'][row['names'].index(label)]==output['code'])
                views.append(dict(id=row['id'],status=row['status'],label=label,kind=status,truth=row['truth'][-1],
                                  joint=output['joint'],code=output['code'],text=output['text'],actual=output['actual']))
            ungranted,granted=views
            state_pattern=tuple(x['kind'] for x in views)
            patterns[state_pattern]+=1
            effects.append(int(granted['kind']=='focal_proposal')-int(ungranted['kind']=='focal_proposal'))
            if not all(view['joint'] for view in views):
                cue=pair[0]['token']['cue']
                cases.append(dict(family=pair[0]['family'],frame=pair[0]['frame'],setting=pair[0]['setting'],
                                  names=pair[0]['names'],codes=pair[0]['codes'],views=views,cue=cue,
                                  common_length=len(pair[0]['token']['prompt']),
                                  prompts=[dict(status=row['status'],sections=row['sections'],token_ids=row['token']['prompt']) for row,_ in ordered]))
        totals={status:dict(sum((group for k,group in groups.items() if k[-1]==bool(i)),Counter()))
                for i,status in enumerate(A.STATUS)}
        path=stage+'-behavior.json'
        review[stage]=dict(links={path:C.A.digest(R.OUT/path)},totals=totals,
            cells=[dict(frame=f,earlier_target=target,reverse=reverse,status=A.STATUS[update],**x)
                   for (f,target,reverse,update),x in sorted(groups.items())],
            pair_patterns=[dict(refused=a,granted=b,n=n) for (a,b),n in sorted(patterns.items())],
            proposal_following_effect=summarize(effects,.99),pair_joint=len(result['gate']['complete_families']),
            families=len(effects),gate_passed=result['gate']['passed'],counterexamples=cases)
    return dict(run=plan['run'],at=C.stamp(),plan=C.A.digest(R.OUT/'plan.json'),panels=review)


def render(p,review,decision,edits):
    outcomes=[]
    pairs=[]
    behavior_cells=[]
    failures=[]
    pred_rows=[]
    causal_rows=[]
    for stage,result in review['panels'].items():
        for status,counts in result['totals'].items():
            outcomes.append([LABELS[stage],status,counts['n'],counts.get('old',0),counts.get('focal_proposal',0),
                             counts.get('earlier_proposal',0),counts.get('other_initial',0),counts.get('outside',0),
                             counts.get('format',0),counts.get('stop',0),counts.get('joint',0),
                             f"{counts.get('code_follows_emitted',0)}/{counts.get('known_label',0)}"])
        pairs.append([LABELS[stage],f"{result['pair_joint']}/{result['families']}",
                      prior.show(result['proposal_following_effect']),'通过' if result['gate_passed'] else '未过'])
        for cell in result['cells']:
            behavior_cells.append([LABELS[stage],cell['frame'],'East' if cell['earlier_target'] else 'West',
                                   'Q' if cell['reverse'] else 'P',cell['status'],f"{cell['joint']}/{cell['n']}"])
        seen=set()
        for case in result['counterexamples']:
            no,yes=case['views']
            pattern=(no['kind'],yes['kind'])
            if pattern in seen:continue
            seen.add(pattern)
            show=lambda view:view['text'].replace(chr(10),' ↵ ').replace(chr(96),'′')
            failures.append(f"- {LABELS[stage]} `{case['family']}`，起始 `{case['names'][0]}`，末条拟更新为 `{case['names'][3]}`。拒绝时 `{show(no)}`，批准时 `{show(yes)}`。")
            if len(seen)>=3:break
    if decision:
        for stage,result in decision['panels'].items():
            for key,metric in result.get('metric',{}).items():
                gain=metric.get('improvement')
                pred_rows.append([LABELS[stage],key,metric['read_count'],prior.show(metric['kl']),
                                  prior.show(metric['source_kl']),prior.show(metric['full_vocabulary_match']),
                                  f"{gain['relative']:.2%}" if gain and gain['relative'] is not None else '—',
                                  '通过' if gain and gain['passed'] else '未过' if gain else '基线'])
    for stage,result in edits.items():
        causal_rows.append([LABELS[stage],result['heads'],result['eligible'],result['dose_error_p95'],
                            prior.show(result['loss_vs_matched']),prior.show(result['partial_native_restore']),
                            result['full_native_restore_matches_sham'],result['real_state_switch_by_arm'],
                            '通过' if result['passed'] else '未过'])
    table=prior.table
    pred=(table(['面板','读出与方法','有效视图','全词表 KL','来源 KL','首选匹配','相对改进','判定'],pred_rows)
          if pred_rows else '正式留出预测未执行。')
    causal=(table(['面板','头','有效视图','剂量差 P95','匹配净损失','原生局部恢复','全词表精确恢复','改选候选标签','判定'],causal_rows)
            if causal_rows else '原生阻断与恢复未执行。')
    end_reason=('预测门通过方法 '+ '、'.join(decision['admitted']) if decision and decision['admitted'] else
                '联合预测门未通过' if decision else '行为准入未通过，发现或留出门后的阶段封存')
    gpu=prior.trajectory()
    return f"""### 问题、冻结与读出

运行 `{p['run']}`，Qwen3-14B 原始 BF16 `{p['snapshot']['snapshot']}`，原始 safetensors 逐块送 RTX 5080，固定右填充 {p['width']}，CUDA SDPA。第 1 条请求始终拒绝，第 2 条固定指向 East。每个家族的初始标签、两项提议、通知顺序、句式、码本和输入长度完全一致。第 2 条请求只替换 7 字符、1 token 的 `refused`／`granted`，分歧位置与所有分段位置相同。两种真值用不同码字。旧词/拟改词的无序对跨家族唯一。词池避开前三版的标签 ID 并隔离试跑、新词与联合锁箱。共冻结 {sum(v['families'] for v in p['material'].values())} 个家族。试跑 96，发现 192，新词、新句式、新组合各 384，三因素联合锁箱 384。

模型在既有 `State:` 后自然贪婪生成。每一步用全词表首选，包括发出的实际标签、答码、结束符。正确性联合要求标签、答码、格式和终止。在每个句式×早条收件记录×码本方向×授权格要求 ≥85%，对应每个句式×设置的成对联合正确率要求 ≥80%。均按预注册门槛。行为对照效应定义为 Δ = 平均[I(批准侧实发末提议) − I(拒绝侧实发末提议)]，99% bootstrap 按家族抽样。相关置信区间描述这套家族采样。旧值 g 与末条提议 v 的规则输出为 g′ = (1−a)g + av，其中 a∈{{0,1}} 是授权值，g、v 以互斥标签的 one-hot 向量表示。

### 逐对行为与反例

{table(['面板','授权','视图','旧标签','末提议','早提议','另记录初值','表外标签','合格式','正常停止','联合正确','实发标签的码本命中'],outcomes)}

{table(['面板','双侧联合正确','末提议跟随率变化 [99% CI]','行为门'],pairs)}

{table(['面板','句式索引','早条收件','旧标签码','授权','联合正确'],behavior_cells)}

{chr(10).join(failures) or '已执行材料中没有联合失败的匹配对。'}

### 留出预测与原生干预

比较同一可见前缀下的文本规则/词法位置基线、同预算 32 个原生坐标与 32 个稀疏字典特征。B20 读取上游状态，经发现集拟合末层状态 ĥ，再计算 q̂ = softmax(W_U ĥ)。用实发历史预测 Code，并要求两侧均与原生读回对齐后才进入配对评分。准入要求同一方法在三留出的 State、Code 全词表 KL 都降低至少 10%，且 99% 区间下端为正、至少 365 完整家族。来源质量和原生坐标重建独立记账。{end_reason}。

{pred}

干预针对 State 读出，固定候选层 B25，发现阶段按授权 token Value 的原生投影与候选标签差方向选头。清除一枚授权 token 对目标查询头的 Value 写入，另用等投影幅度、不同 GQA 组的控制头，再在同时阻断授权与提议的分支独立恢复授权 Value。匹配净损失、局部恢复及全恢复按冻结要求分别核对。

{causal}

### 解释、局限与下一大阶段

配对处理只控制授权词在这套提示中的局部作用。词形、语义与训练频率仍同该 token 绑定。一个早条始终拒绝，目标和末通知序号固定，答码只在两字母间变化。结果界定“保留旧标签、跟随新提议、后续查表”三个可分测环节。用已通过的留出预测和原生阻断逐条提升图谱边证据等级。RDC 当前用有条件的状态转移描述组合，内部实现须由能预测未见组合且有因果对照的计算补足。下一大阶段在独立面板交叉授权同义词、授权和提议的相对位置、取消及二次生效。保持每对单因素，再比较写入、保持与重新寻址；围绕未见行为建立可预测的条件转移，并在独立物理节点复算。

### 复算与资源

入口 `tests/glm5/record_authorization_run.py freeze`、`all`、`close`。冻结计划 SHA-256 `{review['plan']}`。结果 `tests/glm5/result/{R.OUT.name}/`。原始 greedy token、所有失败对的提示与位置、分格、冻结产物和日志均保留。CPU 复算真实输出、组门和冻结材料。{('CUDA 分配峰值 '+format(gpu['peak_gib'],'.2f')+' GiB，共享显存峰值 '+format(gpu['shared_mib'],'.0f')+' MiB，进度 '+str(gpu['events'])+' 行。') if gpu else ''}正式证据登记待独立节点复算及发布授权。
"""


def close():
    plan=R.verified()
    stages,decision,edits=prior.audit(plan)
    assert not (R.OUT/'lockbox-behavior.json').exists()
    if decision is None:
        assert not any((R.OUT/(k+'-behavior.json')).exists() for k in R.HOLDOUTS)
        assert not (R.OUT/'prediction-freeze.json').exists()
    review=readback(plan,stages)
    body=render(plan,review,decision,edits)
    R.write('behavior-audit.json',review)
    check_write(R.OUT/'report.md',len(body.encode())+4096,operation='授权研究报告').write_text('# '+TITLE+'\n\n'+body,encoding='utf-8')
    memo=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    claude=R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    for path in (memo,claude):
        assert TITLE not in path.read_text(encoding='utf-8-sig')
    added=prior.append(check_write(memo,len(body.encode())+4096,operation='追加 GLM5 授权记录'),TITLE,body)
    ref=prior.append(check_write(claude,4096,operation='追加授权研究索引'),TITLE,
                     f"详见 GLM5 Phase {added['phase']}。运行 `{plan['run']}`，证据及门槛见本地报告。")
    cleanup=[]
    folder=R.TEMP
    if folder.exists():
        assert folder.resolve()==folder and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
        checked=[folder,*folder.rglob('*')]
        assert all(not x.is_symlink() and not x.is_junction() and x.resolve().is_relative_to(folder) for x in checked)
        size=sum(x.stat().st_size for x in checked if x.is_file())
        rmtree(folder)
        cleanup.append(dict(path=str(folder),bytes=size))
    R.write('memo-receipt.json',dict(records=[added,ref],cleanup=cleanup))
    files={x.name:dict(sha256=C.A.digest(x),bytes=x.stat().st_size) for x in R.OUT.iterdir()
           if x.is_file() and x.name!='manifest.json'}
    R.write('manifest.json',dict(provenance=plan['provenance'],files=files,sources=plan['sources']))
    print('Phase',added['phase'],'appended; audit and manifest verified against frozen plan',flush=True)
