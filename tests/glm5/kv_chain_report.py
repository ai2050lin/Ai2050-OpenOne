"""Audit completed 14B chain experiments and append the research record."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import kv_chain_run as C
import kv_chain_native as V
import kv_chain_study as S
from kv_orth_audit import summarize
from b36_semantic14_memo import append


TITLE='14B 从事实到实发 State、竞争性读取与答码预测的双留出'
CLIENT=C.ROOT/'tests/glm5/result/kv_length_client_20261117_v1'
SET_LABEL={'words':'新词、既有句式','frames':'原新句式','frames-repair':'复验新词与新句式'}
ROLE_NAMES=dict(address='移除 H18 对 State 的寻址',value='清零 H18 读取的 State Value',note='清零 Note Value',
                joint='联合清零 State / Note Value',rescue='联合清零后恢复原生 State Value',
                **{'restore-gen':'从联合清零恢复 State','restore-all':'原生全部恢复','matched-address':'等幅目标寻址',
                   'random-address':'等幅随机寻址','matched-value':'等幅目标写入','random-value':'等幅随机写入'})


def table(header,rows):
    return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in (header,['---']*len(header),*rows))


def ci(d):
    return '不可估计' if d.get('ci') is None else f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def pass_state(d):
    return '是' if d.get('passed') else '否'


def cases(key):
    doc=C.A.load(key+'-analysis.json')
    ordinary={r['id']:r for r in C.A.load(key+'-greedy.json')['records']}
    special=C.A.load(key+'-conflict.json') if (C.OUT/(key+'-conflict.json')).exists() else None
    return doc,ordinary,{r['id']:r for r in special['records']} if special else {}


def independent(p,key):
    status=C.A.load(key+'-status.json')
    out=dict(status=status,manifests={})
    if not status['clean']:
        name=key+'-greedy.json'
        if (C.OUT/name).exists():
            raw=C.A.load(name)
            status.update(planned=raw['gate']['n'],state=sum(r['state_verified'] for r in raw['records']),greedy=len(raw['gate']['eligible']))
            out['failed_cells']=[v for v in raw['gate']['cells'] if v['code']/v['n']<.85]
        return out
    tok=C.A.Base.tok(); planned={r['id']:r for r in S.panels(type('TokenOnly',(),{'tok':tok})(),p,key)}
    out['planned']=len(planned)
    doc,ordinary,special=cases(key)
    expected={**ordinary,**special}; z=S.load_np(key+'-native.npz'); preds=S.load_np(key+'-predictions.npz')
    donor={}
    if special:
        d=S.load_np(key+'-donors.npz')
        donor={str(k):int(v) for k,v in zip(d['ids'],d['top'],strict=True)}
        assert d['native'].dtype==np.int16 and d['native'].shape==(len(donor),5120)
    meta=C.A.load(key+'-predictions.json')
    assert meta['sha']==C.B.sha(key+'-predictions.npz')==doc['predictions_sha']
    assert meta['clean']==C.B.sha(key+'-native.npz')==status['source']
    assert meta['frozen']==C.B.sha('frozen-models.json')==doc['study_sha']
    assert (C.OUT/'frozen-models.json').stat().st_mtime_ns < (C.OUT/(key+'-greedy.json')).stat().st_mtime_ns
    assert list(preds['read_ids'])==list(z['ids'])
    choices=Counter(); short_long=defaultdict(dict); shape_failed=0; distances=defaultdict(set); role_switch=0
    for j,id_ in enumerate(z['ids']):
        r=planned[id_.removesuffix('-switch')]; obs=expected[id_]; switched=id_.endswith('-switch')
        rebuilt=C.M.build_row(tok,r,r['fact'],(1-r['fact'] if switched else r['fact']),obs['breaks'])
        s=rebuilt['token']; source=C.M.groups(tok,rebuilt)
        assert s['ids']==obs['prefix'] and s['ids'][s['start']:]==obs['emitted'][:-1]
        assert tok.decode(obs['emitted'])==obs['text'] and obs['emitted'][-1]==obs['top'] and obs['state_verified']
        assert all(i<len(s['ids'])-1 for a in source.values() for i in a)
        distances[f"{r['text']}{r['distance']}"].add(s['gap'][0]); preferred=int(np.argmax(z['masses'][j,-1,1]))
        accurate=abs(np.asarray(obs['pq'])-z['pq'][j]).max()<=1 and obs['top']==z['top'][j]
        if not accurate:shape_failed+=1;continue
        choices[r['id'].split('-')[-1],int(switched),C.M.SITES[preferred]]+=1
        if switched:
            assert r['challenge'] and obs['actual_state']!=ordinary[r['id']]['actual_state']
            assert obs['actual_state']==donor[r['id']]
            role_switch+=1
            follow='State' if obs['top']==s['letters'][rebuilt['code']=='Q'] else 'fact'
            choices['conflict',r['distance'],r['compete'],follow]+=1
        else:
            f=(r['text'],r['distance'],r['compete'],r['map'])
            short_long[r['family']][f]=float(z['masses'][j,-1,1,2])
    assert doc['read']['valid_n']==len(z['ids'])-shape_failed and role_switch<=status['conflict']
    factors={}
    nat=[v for v in short_long.values() if len(v)==16]
    for label,axis in [('中性文本',0),('长减短',1),('竞争槽',2),('码本反转',3)]:
        val=[sum(a*(2*f[axis]-1) for f,a in group.items())/8 for group in nat]
        if val:factors[label]=summarize(np.array(val))
    for text in (0,1):
        val=[sum(a*(2*f[1]-1) for f,a in group.items() if f[0]==text)/4 for group in nat]
        if val:factors[f'文本 {text} 内长减短']=summarize(np.array(val))
    out.update(scored=doc,exact_role_switch=role_switch,clean_mismatches=shape_failed,
               gap={k:sorted(v) for k,v in distances.items()},complete_factor_families=len(nat),factor_changes=factors,
               outcomes=[{'group':list(k),'count':v} for k,v in choices.items() if k[0]=='conflict'])
    if 'impact' in doc:
        shard=key+'-edits.npz'; e=S.load_np(shard); d=S.load_np(key+'-effect-detail.npz')
        assert (C.OUT/(key+'-predictions.json')).stat().st_mtime_ns < (C.OUT/shard).stat().st_mtime_ns
        assert np.array_equal(e['ids'],d['ids']) and np.array_equal(e['ids'],preds['flip_ids'])
        assert doc['impact']['valid']==int(d['eligible'].sum())
        m=e['pq'][:,:,1]-e['pq'][:,:,0]; sign=np.where(e['top'][:,0]==tok.encode(' Q',add_special_tokens=False)[0],1,-1)
        assert np.array_equal(e['top'][:,0],e['top'][:,V.ARMS.index('restore-all')])
        assert np.array_equal(e['pq'][:,0],e['pq'][:,V.ARMS.index('restore-all')])
        ix=V.ARMS.index
        for name,first,second in [('address','random-address','matched-address'),('value','random-value','matched-value'),
                                  ('joint','sham','joint'),('rescue','restore-gen','joint')]:
            assert np.array_equal(d[name],sign*(m[:,ix(first)]-m[:,ix(second)])),name
        actual=e['top'][:,[ix('address'),ix('value')]]
        code_ids=[tok.encode(' '+letter,add_special_tokens=False)[0] for letter in 'PQ']
        changed=(actual!=e['top'][:,0,None])&np.isin(actual,code_ids)
        for name in ('never','text_flip','native_flip'):
            loss=float(np.square(preds[name][d['eligible']]-changed[d['eligible']]).mean())
            assert np.isclose(doc['impact']['brier'][name]['brier'],loss,rtol=1e-7,atol=1e-9),name
        out['artifacts']=[shard,key+'-effect-detail.npz',key+'-conflict.json',key+'-donors.npz']
    return out


def pool(suites):
    data=defaultdict(list); errors=defaultdict(list)
    for key,s in suites.items():
        if 'impact' not in s.get('scored',{}):continue
        d=S.load_np(key+'-effect-detail.npz'); e=S.load_np(key+'-edits.npz'); mask=d['eligible']
        for name in ('address','value','joint','rescue'):
            grouped=defaultdict(list)
            for row,v in zip(d['ids'][mask],d[name][mask],strict=True):grouped[row.split('-')[0]].append(float(v))
            data[name].extend(np.mean(x) for x in grouped.values())
        for j,name in enumerate(('address','value')):
            a,b=e['dose'][mask,j*2],e['dose'][mask,j*2+1]
            errors[name].extend(abs(a[:,2]-b[:,2])/np.maximum(a[:,1],.02))
    tests={}
    for name,x in data.items():
        y=summarize(np.asarray(x),1-.05/8); p95=float(np.quantile(errors[name],.95)) if name in errors else None
        panel_size=all(s.get('scored',{}).get('impact',{}).get('effects',{}).get(name,{}).get('n',0)>=50 for s in suites.values())
        y.update(p95=p95,passed=bool(y['n']>=100 and panel_size
            and y['mean']>=.1 and y['ci'][0]>0 and (p95 is None or p95<=.03)))
        tests[name]=y
    return tests


def report(p,frozen,suites,tests,repair):
    s0=p['snapshot']['snapshot']; calibration=C.A.load('pilot-greedy.json')['gate']; ds=C.A.load('discovery-status.json')
    samples=table(['范围','计划视图','实发 State 核验','真实答码过门视图','进入读取','确认冲突续写','句式准入'],
        [(SET_LABEL[k],v.get('planned',p['panels'].get(k,{}).get('n','—')),v.get('state','—'),v.get('greedy','—'),v['clean'],v.get('conflict',0),v['eligible_frames'])
         for k,s in suites.items() for v in [s['status']]])
    result=[]; factors=[]; flips=[]; special=[]; examples=[]; controls=[]; categories=[]; calibration_rows=[]; readings=[]
    for k,s in suites.items():
        if 'scored' not in s:continue
        key=SET_LABEL[k]; d=s['scored']; a=d['read']['accuracy']; main=d['primary'];
        result.append((key,'L25H18 最大标注来源组',d['read']['valid_n'],f"{a['majority']:.1%} / {a['read_base']:.1%} / {a['read_native']:.1%}",ci(main['read']),pass_state(main['read'])))
        named=d['read']['named_top_counts']
        categories.append((key,*(named[n] for n in C.M.SITES),f"{d['read']['global_top_is_annotated']}/{d['read']['valid_n']}"))
        measured_best=max(a,key=a.get)
        reading_label={'majority':'发现集多数类','read_base':'文本与位置模型','read_native':'加入早层读数的模型'}
        state_line=f"{key}的读取预测最高的模型是{reading_label[measured_best]}（{a[measured_best]:.2%}）"
        for group,h in s['factor_changes'].items():factors.append((key,group,s['complete_factor_families'],ci(h)))
        for r in s['outcomes']:special.append((key,'长' if r['group'][1] else '短','有' if r['group'][2] else '无',r['group'][-1],r['count']))
        if 'impact' in d:
            info=d['impact']; b=info['brier']; stats=' / '.join(f"{b[x]['brier']:.5f}" for x in ('never','text_flip','native_flip'))
            result.append((key,'实际 P↔Q 改码',info['valid'],stats,ci(main['flip']),pass_state(main['flip'])))
            changes=info['real_flips']
            state_line+=f"。原生改码模型的 Brier 为 {b['native_flip']['brier']:.5f}，文本与干净边际基线为 {b['text_flip']['brier']:.5f}。等幅寻址阻断与随机头分别改变 {changes['matched-address']} 和 {changes['random-address']} 条首选，等幅写入阻断与随机头分别改变 {changes['matched-value']} 和 {changes['random-value']} 条。"
            for model,label in (('never','永不改码'),('text_flip','文本与干净边际'),('native_flip','同一信息加原生响应')):
                mm=b[model]
                calibration_rows.append((key,label,mm['true_positive'],mm['false_positive'],mm['false_negative'],
                                         f"{mm['precision']:.1%}",f"{mm['recall']:.1%}"))
            for arm,n in info['real_flips'].items():
                flips.append((key,ROLE_NAMES.get(arm,arm),n,info['valid']))
            for arm,m in info['dose'].items():
                controls.append((key,ROLE_NAMES[arm],f"{m['target_median']:.4f}",f"{m['control_median']:.4f}",f"{m['p95']:.3%}"))
            for r in d['cases'][:8]:
                def show(name):
                    x=r['code'][name]; return 'PQ'[x] if x in (0,1) else str(x)
                examples.append(f"- `{r['id']}`　{r['case']}。原事实 {r['fact']}，实发 State {r['state']}，原选码 {show('sham')}，寻址阻断 {show('address')}，V 阻断 {show('value')}，原生部分恢复 {show('restore-gen')}。")
        readings.append(state_line.rstrip('。')+'。')
    main_lines=table(['留出','终点','有效视图','两种基线 / 原生预测','家族增益及校正区间','过门'],result)
    named_table=table(['留出','事实','码本','实发 State','Note','其他输出词','Receipt','全域最大 token 在标注组内'],categories)
    calibration_table=table(['留出','预测器','实际命中改码','误报','漏报','精确率','召回率'],calibration_rows)
    effect=table(['干预','作用及校正区间','家族','剂量偏差 p95','过门'],[(ROLE_NAMES[k],ci(v),v['n'],
                f"{v['p95']:.3%}" if v['p95'] is not None else '全剂量',pass_state(v)) for k,v in tests.items()])
    per_panel=table(['面板','干预','作用及校正区间','有效家族','该面板单独通过'],[
        (SET_LABEL[key],ROLE_NAMES[k],ci(v),v['n'],pass_state(v)) for key,s in suites.items()
        for k,v in s.get('scored',{}).get('impact',{}).get('effects',{}).items()])
    fact_table=table(['留出','自然同族对照','完整家族','State 读权重差 [探索 95%]'],factors)
    special_table=table(['留出','中性段','Note 竞争','真实答码跟随','视图'],special)
    flip_table=table(['留出','干预','全词表首选改变','有效视图'],flips)
    doses=table(['留出','阻断','目标实际幅度中位数','对照实际幅度中位数','相对差 p95'],controls)
    with (C.OUT/'progress.jsonl').open(encoding='utf-8') as log:resource=[json.loads(x) for x in log]
    peaks=(max(x.get('cuda_peak_bytes',0) for x in resource)/2**30,max(x.get('shared_gpu_bytes') or 0 for x in resource)/2**20)
    verified=json.loads((CLIENT/'client-readback.json').read_text(encoding='utf-8'))
    bw=CLIENT/'client-browser-verification.json'; browser=json.loads(bw.read_text(encoding='utf-8')) if bw.is_file() else dict(checks=[])
    repair_doc=''
    if repair:
        behavior=C.A.load('frame-repair-pilot-greedy.json')['gate']
        base=table(['原新句式','文本/长度/竞争/码本','正确','总计'],[
            (x['key'][0],''.join(map(str,x['key'][1:])),x['code'],x['n']) for x in suites['frames'].get('failed_cells',[])])
        repair_doc=f"""
原新句式面板的 {repair['diagnosis']['failed']} 条失败中，{repair['diagnosis']['at_state']} 条首个错分歧在 State，包含过早给字母和选错词。原行为证据全部保留，原面板未做内部预测和干预。以下是未过门条件。

{base}

根据这一失败单独冻结第二套验证，明确写入 State 名称及独立 Code 栏。16个隔离校准家族完成 {len(behavior['eligible'])}/{behavior['n']} 条严格行为验证。冻结原模型、预测器、原生头、对照、剂量和门槛均沿用原发现期。复验面板另预定80个词表互斥新家族，结果另列。这个复验仅能支持校准后的句式，原先两种新句式的失败仍计入原方案跨句式判断。
"""
    return f"""# {TITLE}

运行 `{C.RUN}`，本地 Qwen3-14B 原始 BF16 `{s0}`。承接 Phase 2952，原模型完成事实选择、实际 State 与答码。预定问题是四种输入因素会怎样影响 State 的读取，以及答码之前保存的冻结预测能否指出随后阻断会在哪些视图造成真实 P↔Q 改码。

### 材料、读出和预测原理

每个家族完整交叉两段原创或既有中性文字、两档长度、Note 是否复制备选词、P/Q 码本方向，共16视图。事实角色在每个句式内按家族平衡。短段两个文本在规范分词下等长。长段两段的 State–Code 间隔相差两个 token。提示指明要原样复写指定 Receipt，因此两档内容和长度同时出现在提示与实发续写中。此设计用于跨文本与跨句式预测，Phase 2952 的逐字同提示插入对照另存。冻结后的新词面板有80个家族和旧两句式，新句式面板有80个家族、两个新句式与不重合词表。发现侧32家族。各视图按前缀稳定性和实际全词表 token 核验。模型输出若先走到其他词，保留当时已经生成的前缀和错误 token。

每家族预定四个挑战。角色供体只反转事实句，Note、码本和提示其余部分相同。用供体 CausalLM 真正的终层残差和原始全词表读出，决定接收分支第一个 State token。其余 token，包括 Receipt、Code 和答码，从接收者实际历史逐位置核对。自然分支没有供体编辑。每个句式内自然16格各至少85%正确才准入。挑战另需句式内至少80%核验后才能作干预。保留模块在 E（State 前）、G（实际 State 位）、C（Code 后的答码 query）的原生边界，一条指定视图保存41个隐藏边界至结果档案。

对于源位置 i，H18 的权重 αᵢ = exp(q·kᵢ/√128) / Σⱼ exp(q·kⱼ/√128)。源组质量是其中各位置的 αᵢ 总和。所预测的位置是六个可回查标注源组中的质量最大者（事实、码本、State、Note、其他输出词、Receipt）。另统计真正全域最大 token 是否落入标注区域。读取预测比较发现多数类、从实验条件与实际可见前缀提取词袋和位置的岭回归、再加 L20H5/L21H12/L24H18 更早层的原生源组质量。读取规则没有输入 L25H18 的目标质量或答码。

令 m=logit(Q)−logit(P)。改码预测分别拟合 Δm；基线使用可见文本与干净前向的 m，原生预测另加干净的 Q/K 得分、物理写入范数、各源读取和直接写出投影。它是给定原模型未编辑运行的干预预测。所有预测系数在发现结束后连同模型、材料和随机头一起冻结。留出逐例预测文件在同一面板运行干预之前写盘。换码标签依据原模型全词表 argmax，Brier 分数越低越好。家族级增益须达到预注册最小变化且校正区间下界大于0。

令 a 为 State 的 α，v 为其 Value，u 为不含该槽的归一化其他源输出。原生寻址掩码的局部变化为 a(u−v)，清零 Value 的局部变化为 −av，实际 BF16 变化经原 O 投影。随机对照从同层发现测量相近头中用固定随机种子抽出，按两臂实际可达较小幅度配平。联合阻断在同一个 H18 中去掉 State 与 Note 两组 Value，分别恢复原生 State 与两组原值，对完整恢复和 sham 做逐位 P/Q 以及全词表首选核对。

### 实际行为、留出预测和作用

行为试跑 {len(calibration['eligible'])}/{calibration['n']}，发现进入读取 {ds['clean']} 条、确认 State 换词 {ds['conflict']} 条。随机头分别为寻址 H{frozen['controls']['address']}、写入 H{frozen['controls']['value']}。

{samples}

{repair_doc}

下表读取行显示多数类、纯文本和原生增强的准确率。改码行显示永不改码、文本加干净边际和原生增强的 Brier 分数。校正区间按家族聚类，完全留出范围仅含通过相应技术和行为检查的视图。

{main_lines}

{named_table}

{calibration_table}

原计划在两块确认面板同时通过行为准入后检验合并干预门。下表仍用原方案判据。单面板和修订句式复验的效果另列。

{effect}

{per_panel}

{doses}

{chr(10).join(readings)}

在只有 P/Q 两名竞争者的近似下，改码要求干净分差 m 与干预后分差 m+Δm 异号。即使某条路径对平均分差有可重复作用，远离决策边界的样本仍保持原选码。主结果一律使用全词表首选，另保存实际落到题设候选之外的情况。平均的局部作用和逐例的预测增量各有检验。

在两块面板保留的完整自然家族上，四因素的有符号主效应如下。两文本各自的长短差供后续定位，属于描述性分析。

{fact_table}

成功发出反角色 State 后，实际答码有以下原事实与实发词归属。样本是每家族固定四视图且通过格式门的子集。

{special_table}

{flip_table}

### 反例、严格审视与下一步

{chr(10).join(examples) or '未取得满足干预技术门的反例集合。'}

预测跨新词及新句式的要求应分别核验。六个源组的最大质量只表示这个预定可见区域内的读出。完整注意力对其他原因不穷尽，源组的质量解释与实际写入效果分开。改码模型使用干净前向分数及部分同层原生量，不能当作纯输入文本到逐例反事实答案的闭合预测。预注册对调度误差和剂量失配的排除、每个分支的首词替换以及行为格式筛选都限制了推广。两种文字、两个长度、两种 P/Q 映射与同一个模型快照只覆盖该任务族，尚缺独立物理主机、开放续写及更强稀疏字典比较。RDC 无新统一公式。

第一性原理线索是把固定的原生写入与后继查询的访问条件分别检验。下一阶段以同一提示下的多个严格等长替换和多档插入做验证，把提前预测限制在早于受测头的合法输入。优先解释哪些真实改码由 State 与 Note 竞争决定，哪些由提示原事实或其他备份保持。用原生坐标、简单基线和同预算近似模型预报未见变体，并由独立节点验证。客户端图谱只将本次通过门的干预登记在对应范围。

### 工程、客户端与复算

原始 GPU 任务串行。RTX 5080 CUDA 分配峰值 {peaks[0]:.2f} GiB，共享 {peaks[1]:.0f} MiB。全部增量在 D 盘。预冻结试跑的 Python 源码解析错误保留日志，正式计划冻结前修复。真实填充、输出 token、文件哈希、原生恢复和匹配差逐项验算。研究结果在 `tests/glm5/result/kv_chain_20261117_v1/`。

执行顺序为 `python -B -u tests/glm5/kv_chain_run.py pilot`、`smoke`、`all`，原新句式行为失败后另运行 `python -B -u tests/glm5/kv_chain_frame_repair.py`。CPU 复算及追加入口是 `python -B -u tests/glm5/kv_chain_report.py` 和同入口的 `close`。实际轨迹、随机头抽样、拟合系数、留出预测的写盘时间、脱靶个例与全恢复数组在同目录按版本绑定。

同时将 Phase 2952 既有有效结果接入「空间 → 关联机制 → 成果库」，包含1报告、3定位卡、2范围干预边及按 SHA 查询的反例。真实档案可通过现有索引读取，登记包 {verified['object_count']} 个对象，本地浏览器实测 {len(browser['checks'])} 项，包含14B定位、桌面和手机。定向 ESLint 和生产构建通过。前端全套62/63，唯一失败为现有 ClientGuide 的语言目录测试预期69、实际72。隔离预览中先补入真实模型骨架再完成3D定位，验收没有模型加载或运行请求。公网依项目手动部署流程。网页包、登记包和验收以 `tests/glm5/result/kv_length_client_20261117_v1/` 为准。以上接入不对 Phase 2952 的证据重新升级。
"""


def run(close):
    p=C.A.load('plan.json'); fixed=S.F.verify_freeze(p)
    suites={k:independent(p,k) for k in ('words','frames')}; tests=pool(suites)
    repair=None
    if (C.OUT/'frame-repair-plan.json').exists():
        from kv_chain_frame_repair import HOLDOUT, use_material
        repair=C.A.load('frame-repair-plan.json')
        assert all(C.B.sha(k)==v for k,v in repair['binding'].items())
        assert repair['adapter_sha']==C.A.digest(C.ROOT/'tests/glm5/kv_chain_frame_repair.py')
        assert (C.OUT/'frame-repair-plan.json').stat().st_mtime_ns < (C.OUT/'frame-repair-pilot-greedy.json').stat().st_mtime_ns
        use_material(C.A.Base.tok(),p)
        combined=dict(p,panels=p['panels']|repair['panels'])
        suites[HOLDOUT]=independent(combined,HOLDOUT)
    title=report(p,fixed,suites,tests,repair); C.P.check_write(C.OUT/'report.md').write_text(title,encoding='utf-8')
    C.A.save('independent-audit.json',dict(panels=suites,pooled=tests,freeze_sha=C.B.sha('frozen-models.json')))
    if not close:return
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'; assert TITLE not in memo.read_text(encoding='utf-8-sig')
    entries={*p['sources'],'tests/glm5/kv_chain_report.py','tests/glm5/kv_length_client.py',
             'tests/glm5/kv_length_client_server.py','tests/glm5/verify_kv_length_client.cjs',
             'frontend/src/components/v8/KvLengthReport.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx'}
    if repair:entries.add('tests/glm5/kv_chain_frame_repair.py')
    with ZipFile(C.P.check_write(C.OUT/'sources.zip'),'w',ZIP_DEFLATED) as z:
        for name in sorted(entries):z.write(C.ROOT/name,name)
    with ZipFile(C.OUT/'sources.zip') as z:assert all(z.read(name)==(C.ROOT/name).read_bytes() for name in entries)
    memo_receipt=append(C.P.check_write(memo),TITLE,title.split('\n\n',1)[1]); phase=memo_receipt['phase']
    index=append(C.P.check_write(C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'),'14B 选事实、自产 State 与答码预测索引',
                 f'详见 GLM5 Phase {phase}。运行 {C.RUN}，预测和干预按新词与新句式面板独立评估。Phase 2952 已接入本机成果库并交付登记包。')
    C.A.save('memo-receipt.json',dict(records=[memo_receipt,index]))
    cleanup=[]
    for q in (C.TMP,C.ROOT/'tests/glm5_temp/kv_length_client_20261117_v1'):
        if not q.exists():continue
        assert q.resolve()==q and q.parent==(C.ROOT/'tests/glm5_temp').resolve()
        own=list(q.rglob('*')); assert all(not path.is_symlink() and not path.is_junction() and path.resolve().is_relative_to(q) for path in [q,*own])
        freed=sum(path.stat().st_size for path in own if path.is_file());shutil.rmtree(q)
        cleanup.append(dict(path=str(q),bytes=freed))
    C.A.save('cleanup.json',dict(paths=cleanup,free=shutil.disk_usage(C.OUT).free))
    C.A.save('manifest.json',dict(files={p.name:C.A.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},
                                 sources={name:C.A.digest(C.ROOT/name) for name in sorted(entries)}))
    print('Memo Phase',phase,flush=True)


if __name__=='__main__':
    import sys
    C.BUDGET=6*2**20
    C.setup(); run('close' in sys.argv)
