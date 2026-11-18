"""Close a finished or gate-stopped record-update run with an append-only report."""

import json
import shutil
import sys

from b36_semantic14_memo import append
from deploy.storage_policy import check_write
import record_update_run as R
import record_update_material as M


TITLE='14B 自然记录更新、跨词句与组合的冻结预测和原生验证'
LABELS=dict(pilot='试跑',discovery='发现',words='新词',frames='新句式',combos='新组合')


def table(headers, rows):
    return '\n'.join(' | '.join(map(str,values)) for values in (headers,['---']*len(headers),*rows))


def show(x):
    if not x or x.get('mean') is None:return '缺失'
    lo,hi=x['ci']
    return f"{x['mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def audit(p):
    from transformers import AutoTokenizer

    run=R.C.A.digest(R.OUT/'plan.json')
    tokenizer=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    eos=json.loads((R.MODEL/'generation_config.json').read_bytes()).get('eos_token_id')
    term=set(eos if isinstance(eos,list) else [eos]) | {tokenizer.eos_token_id}
    stages={}
    for stage in LABELS:
        name=stage+'-behavior.json'
        if not (R.OUT/name).exists():continue
        result=R.read(name)
        assert result['plan']==run and result['material']==p['material'][stage]['sha256']
        rows=[M.encode(tokenizer,r) for r in M.families(tokenizer,stage)]
        assert M.digest(rows)==result['material']
        assert M.digest(R.gate(rows,result['records']))==M.digest(result['gate'])
        for row,out in zip(rows,result['records'],strict=True):
            rebuilt=R.behavior(tokenizer,row,out['actual'],term,out['draft_matches'],out['draft_divergences'])
            assert rebuilt==out
        stages[stage]=result
    if 'pilot' not in stages:raise ValueError('No model behavior recorded')
    decision=R.read('stage-decision.json') if (R.OUT/'stage-decision.json').exists() else None
    stopped=not stages['pilot']['gate']['passed'] or ('discovery' in stages and not stages['discovery']['gate']['passed'])
    if decision is None and not stopped:raise ValueError('Behavior is ongoing. Append after the full study or a frozen gate stops it.')
    edits={}
    if decision:
        frozen=R.OUT/'prediction-freeze.json'
        assert frozen.is_file()
        verify_links(R.read(frozen.name))
        for stage,carrier in decision['panels'].items():
            assert frozen.stat().st_mtime_ns < (R.OUT/(stage+'-behavior.json')).stat().st_mtime_ns
            if carrier['behavior']=='passed':
                verify_links(R.read(stage+'-native-binding.json'))
                proxy=R.read(stage+'-predictions.json')
                verify_links(proxy)
                assert proxy['metric']==carrier['metric']
    if decision and decision['admitted']:
        for k in R.HOLDOUTS:
            record=R.read(k+'-intervention.json')
            assert R.C.A.digest(R.OUT/(k+'-intervention.npz'))==record['effects_sha']
            assert R.C.A.digest(R.OUT/(k+'-predictions.json'))==record['input_predictions_sha']
            edits[k]=record
    return stages,decision,edits


def verify_links(record):
    assert all(R.C.A.digest(R.OUT/name)==expected for name,expected in record['links'].items())


def trajectory():
    path=R.OUT/'progress.jsonl'
    if not path.exists():return None
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows:return None
    peak=max((r.get('cuda_peak_bytes',0) for r in rows),default=0)
    share=max((r.get('shared_gpu_bytes',0) or 0 for r in rows),default=0)
    return dict(peak_gib=peak/2**30,shared_mib=share/2**20,events=len(rows))


def model_cases(stages,decision):
    if not decision:return []
    from transformers import AutoTokenizer
    from record_update_study import caption,load_np

    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    for stage,carrier in decision['panels'].items():
        if carrier['behavior']!='passed':continue
        z=load_np(stage+'-native.npz')
        val=load_np(stage+'-predictions.npz')
        assert z['ids'].tolist()==val['ids'].tolist()
        raw={r['id']:r for r in stages[stage]['records']}
        observations=[caption(tok,r,raw[r['id']])
                      for r in (M.encode(tok,r0) for r0 in M.families(tok,stage))]
        for j,query in enumerate(('state','code')):
            top=z['argmax'][:,j]
            guessed=[val[f'{query}.{m}.metrics'][:,1].astype('int64') for m in ('text','native','dictionary')]
            import numpy as np
            observed=np.array([r['expected_top'][j] for r in observations])
            better=(observed==top)&(guessed[0]==top)&((guessed[1]!=top)|(guessed[2]!=top))
            for i in better.nonzero()[0][:2]:
                name=str(z['ids'][i])
                decode=lambda idx:tok.decode([int(idx)]).replace(chr(10),' ↵ ').replace(chr(96),'′')
                out=raw[name]['text'].replace(chr(10),' ↵ ').replace(chr(96),'′')
                yield (f"- {LABELS[stage]} {query}，§{name}§。实发输出 §{out}§。"
                       f"底层首选 §{decode(top[i])}§，文本／坐标／字典预测 "
                       + '／'.join('§'+decode(y[i])+'§' for y in guessed)+'。').replace('§',chr(96))


def behavior_cross_version():
    if R.OUT.name!='record_update_20261117_v3':return ''
    groups=[]; lines=[]
    for version in range(1,4):
        path=R.OUT.with_name(f'record_update_20261117_v{version}')
        data=json.loads((path/'behavior-audit.json').read_bytes())
        assert all(R.C.A.digest(path/name)==sha for name,sha in data['links'].items())
        for n,x in sorted(data['groups'].items()):
            groups.append([f'V{version}',n,x['views'],x['joint'],x['format'],x['stop'],
                           x.get('wrong_notice1',0)+x.get('wrong_notice2',0),
                           x.get('wrong_other_initial',0),x.get('wrong_outside_labels',0),
                           f"{x.get('code_follows_emitted',0)}/{x.get('recognized_label',0)}"])
        lines.append(f"- V{version} 独立复核 §{path.relative_to(R.ROOT).as_posix()}/behavior-audit.json§。")
    headers=['版次','目标有效更新数','视图','联合正确','格式','停止','错抄提议','错抄其他初始标签','标签表外','可识别实发标签的码本命中']
    return ('### 三版逐例核对与证据范围\n\n'+table(headers,groups)+'\n\n'
            '码本命中以模型实际发出的标签另算，范围限定在能在码本查到的输出。它允许判断模型是先选错标签再正确转换，还是同时错在两处。'
            '三版使用不同试跑词表和提示，成绩差异不估计某项提示改动的因果效应。'
            '两次都有效的组合属于预定留出，试跑未覆盖，不能从单次更新外推。'
            '本轮三图谱增加的是受控更新材料、失败标签与读出反例。没有新增原生因果边，RDC 公式沿用原记录。\n\n'
            +'\n'.join(lines)+'\n\n').replace('§',chr(96))


def render(p,stages,decision,edits):
    behavior=[]; fails=[]
    for key,result in stages.items():
        for c in result['gate']['cells']:
            behavior.append([LABELS[key],c['frame'],c['updates'],c['n'],c['state'],c['code'],
                             c['format'],c['stop'],c['joint'],'通过' if c['passed'] else '未过'])
        for r in [x for x in result['records'] if not x['joint']][:4]:
            fails.append(f"- {LABELS[key]} §{r['id']}§，{r['failure']}。实际输出 §{r['text'].replace(chr(10),' ↵ ').replace(chr(96),'′')}§。")
    predictions=[]; intervention=[]
    if decision:
        for stage,result in decision['panels'].items():
            for key,v in result.get('metric',{}).items():
                pred=v.get('improvement')
                predictions.append([LABELS[stage],key,v['read_count'],show(v['kl']),show(v['source_kl']),show(v['full_vocabulary_match']),
                                    show(pred),'通过' if pred and pred['passed'] else '未过' if pred else '基线'])
    for stage,result in edits.items():
        intervention.append([LABELS[stage],result['eligible'],result['dose_error_p95'],
                             show(result['loss_vs_matched']),show(result['partial_native_restore']),
                             result['full_native_restore_matches_sham'],result['real_code_changes_by_arm'],
                             '通过' if result['passed'] else '未过'])
    power=p['power']['words']; gpu=trajectory()
    cases=list(model_cases(stages,decision))
    outcome=(('通过全部冻结预测门的方法为 '+ '、'.join(decision['admitted'])+'。')
             if decision and decision['admitted'] else
             '三块留出的联合预测门未通过。' if decision else '行为阶段按冻结门槛停止。')
    return f"""### 问题、样本与预测边界

运行 §{p['run']}§，Qwen3-14B 原始 BF16 快照 §{p['snapshot']['snapshot']}§。任务给定两个记录、两次带授权状态的替换和 P/Q 码本。只跟踪指定记录。模拟真值由规则逐步计算。以实际自然生成的 State、Code、固定格式及终止共同评分。输出首标签和字母均用未受限全词表。草案负责对同一前向批的因果位置进行校验，对每个分歧都提交原模型实际首选，而后继续。32 token 达上限保留未终止。

冻结后，用发现表现训练，比较词袋、n-gram、位置基线与同预算的早期原生坐标、稀疏字典增强。State 读出在实际首词之前，Code 读出仅使用模型已发出的历史。预测器共享原 lm_head 作为输出基底，指标为全词表 KL、真实首选和来源分布误差。三留出各 384 家族，初始词对互斥。以配对二值差 {power['expected_delta_pt']:.0f} 个百分点、失谐率 {power['p_d']:.2f}、α=.01 规划，所需 {power['n_required']} 家族，预期功效 {power['achieved_power']:.3f}。小幅连续指标不借用这项功效。联合锁箱保持未开。

令 g_i 表示实际被追踪记录的状态，第 i 条已获批准且针对该记录的更新把 g_i 替换为新标签，否则延续 g_(i−1)。这是外部真值。预测器以同一可见前缀在 B20 输入的候选量估计末层归一状态 ĥ，再由共享词表投影得到 q̂=softmax(W_U ĥ)。逐题计算 KL(p_model ∥ q̂)，按家族平均后比较。当前测试用上游激活作为模型输入，范围内的通过仍需要另做纯文本端到端预测。

### 模型实际行为

{table(['面板','句式','有效更新数','总视图','State对','Code对','格式','终止','全部对','门'],behavior)}

{chr(10).join(fails) or '已执行面板未取得自然行为错例。'}

{behavior_cross_version()}

### 冻结留出预测

{table(['面板','读出与方法','视图','全词表KL','来源KL','预测实际首选','对基线KL改进','判定'],predictions) if predictions else '行为准入停止了后续分析，尚无预测结果。'}

{outcome}

{chr(10).join(cases)}

### 原生阻断与恢复

{table(['面板','有效视图','剂量相对差P95','等幅分差','原生局部恢复','全恢复一致','实发改码','判定'],intervention) if edits else '本轮未进入原生干预。'}

### 边界与后续

只把已执行的自然行为、合格预测及原生干预分别记入对应范围。固定标记、离散标签、已观察的早期状态与真实生成格式限制推广。字典拟合的重建误差与未见预测独立计量。RDC 本体与既有图谱公式保留。下轮用独立材料复查失败条件，进一步区分相对位置、内容与竞争路由。

复核文件在 §{R.OUT.relative_to(R.ROOT).as_posix()}§，入口是 §tests/glm5/record_update_run.py freeze§、§all§ 和 §tests/glm5/record_update_report.py§。{('CUDA 分配峰值 '+format(gpu['peak_gib'],'.2f')+' GiB，共享显存峰值 '+format(gpu['shared_mib'],'.0f')+' MiB。进度日志 '+str(gpu['events'])+' 行。') if gpu else ''}
""".replace('§',chr(96))


def report(close):
    p=R.verified()
    stages,decision,edits=audit(p)
    text=render(p,stages,decision,edits)
    target=check_write(R.OUT/'report.md',len(text.encode())+4096,operation='复核报告')
    target.write_text('# '+TITLE+'\n\n'+text,encoding='utf-8')
    if not close:return
    glm=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    assert TITLE not in glm.read_text(encoding='utf-8-sig')
    memo=append(check_write(glm,len(text.encode()),operation='追加 GLM5 研究记录'),TITLE,text)
    claude=R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    ref=append(check_write(claude,2048),TITLE,f"详见 GLM5 Phase {memo['phase']}。运行 ID {p['run']}。")
    cleanup=[]
    targets=[R.TEMP]
    if R.OUT.name=='record_update_20261117_v3':
        targets.extend(R.TEMP.with_name(f'record_update_20261117_v{k}') for k in (1,2))
    for folder in targets:
        if not folder.is_dir():continue
        assert folder.resolve()==folder and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
        owned=[folder,*folder.rglob('*')]
        assert all(not q.is_symlink() and not q.is_junction() and q.resolve().is_relative_to(folder) for q in owned)
        size=sum(x.stat().st_size for x in owned if x.is_file())
        shutil.rmtree(folder)
        cleanup.append(dict(path=str(folder),bytes=size))
    R.write('memo-receipt.json',dict(provenance=p['provenance'],records=[memo,ref],cleanup=cleanup))
    files={x.name:dict(sha256=R.C.A.digest(x),bytes=x.stat().st_size) for x in sorted(R.OUT.iterdir())
           if x.is_file() and x.name!='manifest.json'}
    supplements=['record_update_report.py','record_update_behavior_audit.py']
    sources=p['sources']|{f'tests/glm5/{name}':R.C.A.digest(R.ROOT/'tests/glm5'/name) for name in supplements}
    R.write('manifest.json',dict(provenance=p['provenance'],files=files,sources=sources))


if __name__=='__main__':
    report('close' in sys.argv)
