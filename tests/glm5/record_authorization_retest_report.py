"""Independently replay paired generation scores and append the focused replication."""

from collections import Counter
from itertools import batched
from shutil import rmtree
import json

import record_authorization_report as previous
import record_authorization_retest_material as A
from deploy.storage_policy import check_write
from kv_origin_audit import summarize


R,C,base=previous.R,previous.C,previous.prior
TITLE='14B 授权最小对复测、同义词替换与失败条件'
ARMS=dict(original='原授权词',synonym='同义词')
PANELS=dict(pilot='独立复测',discovery='发现',confirm='留出')
OUTCOMES=dict(pilot_behavior_failed='复测行为门未过',discovery_behavior_failed='发现行为门未过',
              confirm_behavior_failed='留出行为门未过',prediction_gate_failed='留出预测门未过',intervention_finished='已执行原生干预')
EVENTS=(('retained_after_grant',True,'old'),('copied_after_refusal',False,'focal_proposal'))


def replay(plan,arm,panel,tok,term):
    path=panel+'-behavior.json'
    raw=R.read(path)
    rows=A.rows(tok,panel,arm)
    assert raw['plan']==C.A.digest(R.OUT/'plan.json')
    assert raw['material']==A.M.digest(rows)==plan['material'][arm][panel]['sha256']
    assert R.gate(rows,raw['records'])==raw['gate']
    pairs={}
    for (a,b),(x,y) in zip(batched(rows,2),batched(raw['records'],2),strict=True):
        key=a['family']
        assert key==b['family']
        pair={}
        for row,record in ((a,x),(b,y)):
            assert R.behavior(tok,row,record['actual'],term,record['draft_matches'],record['draft_divergences'])==record
            pair[row['update']]=dict(kind=previous.category(row,record),out=record,row=row)
        pairs[key]=pair
    return pairs,raw['gate'],C.A.digest(R.OUT/path)


def attach_forecasts(arm,data):
    frozen=R.OUT/'prediction-freeze.json'
    outcome=data['outcome']
    early=outcome in ('pilot_behavior_failed','discovery_behavior_failed')
    if early:
        assert not frozen.exists() and not (R.OUT/'confirm-behavior.json').exists()
        assert not (R.OUT/'confirm-predictions.json').exists() and not (R.OUT/'confirm-intervention.npz').exists()
        return {}
    confirm=R.OUT/'confirm-behavior.json'
    assert frozen.stat().st_mtime_ns < confirm.stat().st_mtime_ns
    base.verify_links(R.read(frozen.name))
    base.verify_links(R.read('discovery-native-binding.json'))
    if outcome=='confirm_behavior_failed':
        assert not R.read(confirm.name)['gate']['passed']
        assert not (R.OUT/'confirm-predictions.json').exists() and not (R.OUT/'confirm-intervention.npz').exists()
        return {}
    assert R.read(confirm.name)['gate']['passed']
    base.verify_links(R.read('confirm-native-binding.json'))
    forecast=R.read('confirm-predictions.json')
    base.verify_links(forecast)
    assert [m for m in ('native','dictionary') if all(forecast['metric'][f'{q}.{m}']['improvement']['passed']
            for q in ('state','code'))]==data['admitted'],arm
    evidence=dict(forecast=forecast)
    if outcome=='intervention_finished':
        intervention=R.read('confirm-intervention.json')
        assert C.A.digest(R.OUT/'confirm-intervention.npz')==intervention['effects_sha']
        assert C.A.digest(R.OUT/'confirm-predictions.json')==intervention['input_predictions_sha']
        evidence['intervention']=intervention
    else:
        assert not (R.OUT/'confirm-intervention.npz').exists()
    return evidence


def outcome_counts(rows):
    counts=Counter(x['kind'] for x in rows)
    counts['n']=len(rows)
    for metric in ('joint','format','stop','state_ok','code_ok'):
        counts[metric]=sum(x['out'][metric] for x in rows)
    recognized=[x for x in rows if x['out']['state'] in x['row']['names']]
    counts['recognized']=len(recognized)
    counts['code_matches_emitted']=sum(x['out']['code']==x['row']['codes'][x['row']['names'].index(x['out']['state'])] for x in recognized)
    return dict(counts)


def paired_changes(pilot):
    original,synonym=(pilot[k] for k in A.LEXICONS)
    assert original.keys()==synonym.keys()
    stats=[]
    errors=[]
    for name,side,kind in EVENTS:
        for setting in A.SETTINGS:
            for key_order in (None,False,True):
                cases=[]
                for family,views in original.items():
                    before,after=views[side],synonym[family][side]
                    row=before['row']
                    if tuple(row['setting'])==setting and (key_order is None or row['proposal_first']==key_order):
                        cases.append((before['kind']==kind,after['kind']==kind))
                bins=Counter(cases)
                stat=summarize([int(b)-int(a) for a,b in cases],.99)
                stats.append(dict(event=name,setting=setting,proposal_first=key_order,n=len(cases),
                    original=sum(a for a,b in cases),synonym=sum(b for a,b in cases),
                    both=bins[True,True],original_only=bins[True,False],synonym_only=bins[False,True],difference=stat))
    for family,views in original.items():
        four={arm:pilot[arm][family] for arm in A.LEXICONS}
        for side in (False,True):
            A.verify_cue_only(four['original'][side]['row'],four['synonym'][side]['row'])
        if all(x['out']['joint'] for arm in four.values() for x in arm.values()):
            continue
        row=views[False]['row']
        errors.append(dict(family=family,names=row['names'],setting=row['setting'],proposal_first=row['proposal_first'],
            arms={arm:[dict(status=sample['row']['status'],truth=sample['row']['truth'][-1],kind=sample['kind'],
                output=sample['out'],sections=sample['row']['sections'],token=sample['row']['token'])
                for _,sample in sorted(samples.items())] for arm,samples in four.items()}))
    return stats,errors


def inspect(plan,scope,decision):
    from transformers import AutoTokenizer

    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    eos=json.loads((R.MODEL/'generation_config.json').read_bytes())['eos_token_id']
    term=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    assert (R.OUT/'original/pilot-behavior.json').stat().st_mtime_ns < (R.OUT/'synonym/pilot-behavior.json').stat().st_mtime_ns
    panels={}
    pilot={}
    evidence={}
    for arm in A.LEXICONS:
        panels[arm]={}
        verdict=decision['arms'][arm]
        with scope(arm):
            for panel in PANELS:
                if not (R.OUT/(panel+'-behavior.json')).exists():
                    continue
                pairs,gate,sha=replay(plan,arm,panel,tok,term)
                if panel=='pilot':
                    assert sha==verdict['behavior_sha'] and verdict['passed']==gate['passed']
                    pilot[arm]=pairs
                cells=[]
                for condition in gate['cells']:
                    setting=(condition['earlier_target'],condition['reverse'])
                    reports=[sample for pair in pairs.values() for side,sample in pair.items()
                             if side==bool(condition['updates']) and tuple(sample['row']['setting'])==setting]
                    cells.append(dict(condition,outcomes=outcome_counts(reports)))
                panels[arm][panel]=dict(gate=gate,sha=sha,cells=cells)
            evidence[arm]=attach_forecasts(arm,verdict)
            assert not (R.OUT/'lockbox-behavior.json').exists()
            assert ('discovery' in panels[arm])==verdict['passed']
            if 'discovery' in panels[arm]:
                assert panels[arm]['discovery']['gate']['passed']==(verdict['outcome']!='discovery_behavior_failed')
    effects,errors=paired_changes(pilot)
    return dict(run=plan['run'],at=C.stamp(),plan=decision['plan'],panels=panels,outcomes=decision['arms'],
                matched_effects=effects,counterexamples=errors,evidence=evidence)


def setting_name(setting):
    earlier,reverse=setting
    return f"早条 {'East' if earlier else 'West'}／旧标签码 {'Q' if reverse else 'P'}"


def render(plan,audit):
    rows=[]
    paired=[]
    forecasts=[]
    interventions=[]
    for arm,panels in audit['panels'].items():
        for panel,result in panels.items():
            for cell in result['cells']:
                outcome=cell['outcomes']
                rows.append([ARMS[arm],PANELS[panel],setting_name((cell['earlier_target'],cell['reverse'])),
                    A.LEXICONS[arm][cell['updates']],outcome['n'],outcome.get('old',0),outcome.get('focal_proposal',0),
                    outcome.get('earlier_proposal',0)+outcome.get('other_initial',0)+outcome.get('outside',0),
                    outcome['format'],outcome['stop'],cell['joint'],'通过' if cell['passed'] else '未过'])
            for cell in result['gate']['pairs']:
                paired.append([ARMS[arm],PANELS[panel],setting_name((cell['earlier_target'],cell['reverse'])),
                    f"{cell['both']}/{cell['n']}",'通过' if cell['passed'] else '未过'])
        evidence=audit['evidence'][arm]
        if 'forecast' in evidence:
            for k,item in evidence['forecast']['metric'].items():
                gain=item.get('improvement')
                forecasts.append([ARMS[arm],k,item['read_count'],base.show(item['kl']),base.show(item['source_kl']),
                    base.show(item['full_vocabulary_match']),f"{gain['relative']:.2%}" if gain and gain['relative'] is not None else '—',
                    '通过' if gain and gain['passed'] else '未过' if gain else '基线'])
        if 'intervention' in evidence:
            item=evidence['intervention']
            interventions.append([ARMS[arm],item['heads'],item['eligible'],item['dose_error_p95'],base.show(item['loss_vs_matched']),
                base.show(item['partial_native_restore']),item['full_native_restore_matches_sham'],item['real_state_switch_by_arm'],
                '通过' if item['passed'] else '未过'])
    effects=[]
    for item in audit['matched_effects']:
        if item['proposal_first'] is not None:continue
        effects.append(['批准仍保留' if item['event']=='retained_after_grant' else '拒绝仍跟随提议',setting_name(item['setting']),item['n'],
            item['original'],item['synonym'],item['both'],item['original_only'],item['synonym_only'],base.show(item['difference'])])
    failures=[]
    seen=set()
    for case in audit['counterexamples']:
        pattern=tuple(view['kind'] for arm in case['arms'].values() for view in arm)
        if pattern in seen:continue
        seen.add(pattern)
        actual=[]
        for arm,samples in case['arms'].items():
            for view in samples:
                label=view['output']['state'] or view['output']['text'][:48].replace('\n',' ↵ ')
                actual.append(f"{view['status']} → `{label}`")
        failures.append(f"- `{case['family']}`，{setting_name(case['setting'])}，旧值 `{case['names'][0]}`，末提议 `{case['names'][3]}`。"+'，'.join(actual)+'。')
        if len(failures)>=5:break
    status='，'.join(f"{ARMS[k]}{OUTCOMES[v['outcome']]}" for k,v in audit['outcomes'].items())
    table=base.table
    predictions=table(['词表','读出与方法','有效视图','全词表KL','来源KL','首选匹配','相对改进','判定'],forecasts) if forecasts else '留出预测未执行。'
    causal=table(['词表','头','有效视图','剂量误差P95','等幅净损失','原生局部恢复','精确全恢复','改选旧值/提议','判定'],interventions) if interventions else '原生阻断与恢复未执行。'
    gpu=base.trajectory()
    return f"""### 复测设计与冻结口径

本轮 `{plan['run']}`，Qwen3-14B 原始 BF16 `{plan['snapshot']['snapshot']}`。复测 Phase 2959 句式 0 的两种失败设置，各 128 个新家族，按码本中末提议是否列于旧值之前均衡。旧值、其他记录、两个提议、通知位置、码本及长度在同一家族四种输入中完全相同。先跑完相同句式的 `refused`／`granted`，再替换末通知同一位置为 `blocked`／`allowed`。四个词各 7 字符、各 1 token。逐项核验任意两种输入仅这一位置变化。前文规则和第一条通知保留原术语，使差异对应末通知用词，同时包含同义词与规则显式用词的匹配关系。复测、发现、留出和锁箱词池互斥，词表排除既往四轮标签 ID。每种词表用同一组 256 个复测家族，统计独立单位为家族。

同一 `State:` 前缀后逐 token 取原模型全词表首选，实际 EOS 或最多 32 token。逐格要求 State、Code、格式和终止均正确 ≥85%，每个设置两侧均正确的家族 ≥80%。两个设置共同通过才允许相应词表进入独立发现和留出，阈值沿用 Phase 2959。两个词表均先完成行为复测，再按各自门控继续。记 g 为旧标签、v 为提议、a 为授权状态，规则目标为 g′=(1−a)g+av。对失败事件用在同一家族的差 Δ=I(同义词失败)−I(原词失败) 的均值估计变化，按家族 bootstrap 给出描述性 99% 区间。码本先后次序的预注册分层、确切转移和逐例数据在 `behavior-audit.json`。

### 实测行为

{table(['授权词','面板','设置','末授权','视图','旧值','末提议','其他选择','格式','停止','联合正确','单侧门'],rows)}

{table(['授权词','面板','设置','双侧联合正确','双侧门'],paired)}

{table(['失败事件','设置','匹配家族','原词','同义词','两边都失败','仅原词','仅同义词','风险差 [99% CI]'],effects)}

{chr(10).join(failures) or '复测家族全部通过自然行为评分。'}

### 留出预测与原生干预

{status}。冻结的预测比较在相同实发前缀与相同留出上评估文本规则/词序/位置、等预算 32 个 B20 原生坐标和 32 个稀疏特征。Code 只使用真正在前面的输出。由估计末状态 ĥ 经 q̂=softmax(W_U ĥ) 得可复算全词表分布。一个方法在 State、Code 都须超过文本基线至少 10% KL 改善，99% 区间下界为正且至少 365 完整家族。通过方进入在 B25 发现选择的原生授权 Value 阻断、匹配同层控制和原生局部／完全恢复，效应、匹配剂量误差、全词表首选和副作用按冻结设计分别核对。

{predictions}

{causal}

### 证据边界与后续

两种设置同时改了早条收件人与旧标签码，设置差不能进一步拆为两者各自的效应。固定为首个英文句式、两通知且第一条拒绝，结果适用于此局部转移。词义、词形熟悉度及与规则用词的显式匹配仍绑定于四个测试词。用授权触发状态转移是可证伪的外部规则。内部图谱必须由未见预测和原生干预分别取证。本轮行为证据可区分未更新、误更新和查表失败。下一大阶段在独立家族上正交叉分早条收件人、码本方向与词序，加入规则／通知同义词一致性与末条位置交叉，再冻结多词迁移和连续更新面板。由通过行为门的材料比较上游状态如何维持、被授权覆盖及送到输出，最后用独立物理节点复算。RDC 理论的局部状态转移假设仍须按上述范围检验。

运行脚本 `tests/glm5/record_authorization_retest_run.py freeze`、`all`、`close`。冻结 SHA-256 `{audit['plan']}`。逐例实发序列、冻结门、失败对的四份文本与 token、分格、运行日志、结果索引在 `tests/glm5/result/{R.OUT.name}/`。{('RTX 5080 CUDA 分配峰值 '+format(gpu['peak_gib'],'.2f')+' GiB，共享显存 '+format(gpu['shared_mib'],'.0f')+' MiB，进度 '+str(gpu['events'])+' 行。') if gpu else ''}执行为本机物理节点，正式证据登记仍需按项目审计流程。
"""


def close(plan,scope):
    decision=R.read('stage-decision.json')
    assert decision['plan']==C.A.digest(R.OUT/'plan.json')
    audit=inspect(plan,scope,decision)
    R.write('behavior-audit.json',audit)
    body=render(plan,audit)
    check_write(R.OUT/'report.md',len(body.encode())+4096,operation='授权复测报告').write_text('# '+TITLE+'\n\n'+body,encoding='utf-8')
    memo=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    index=R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    for path in (memo,index):
        assert TITLE not in path.read_text(encoding='utf-8-sig')
    appended=base.append(check_write(memo,len(body.encode())+4096,operation='追加 GLM5 授权复测'),TITLE,body)
    reference=base.append(check_write(index,4096,operation='追加授权复测索引'),TITLE,
                          f"详见 GLM5 Phase {appended['phase']}。运行 `{plan['run']}`，两词表的复测与晋级按各自冻结门登记。")
    cleanup=[]
    scratch=R.TEMP
    if scratch.exists():
        assert scratch.resolve()==scratch and scratch.parent==(R.ROOT/'tests/glm5_temp').resolve()
        entries=[scratch,*scratch.rglob('*')]
        assert all(not x.is_symlink() and not x.is_junction() and x.resolve().is_relative_to(scratch) for x in entries)
        size=sum(x.stat().st_size for x in entries if x.is_file())
        rmtree(scratch)
        cleanup.append(dict(path=str(scratch),bytes=size))
    R.write('memo-receipt.json',dict(records=[appended,reference],cleanup=cleanup))
    files={x.relative_to(R.OUT).as_posix():dict(sha256=C.A.digest(x),bytes=x.stat().st_size) for x in R.OUT.rglob('*')
           if x.is_file() and x.name!='manifest.json'}
    R.write('manifest.json',dict(files=files,sources=plan['sources']))
    print('Appended Phase',appended['phase'],flush=True)
