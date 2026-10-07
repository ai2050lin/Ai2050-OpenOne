"""Finalize completed artifacts; append findings without rewriting research history."""
import hashlib,json,re,sys,statistics
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/glm5'))
from phase2751_trusted_rebuild import OUT,write,sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    arms={s:read(OUT/s/'summary.json') for s in ['4B','14B']}
    measurements={s:read(OUT/s/'measurement_repair.json') for s in arms}
    second={s:read(OUT/s/'second_order.json') for s in arms}
    control=read(OUT/'length_control/4B/summary.json')
    captures={s:read(OUT/s/'capture_done.json') for s in arms}
    cap_control=read(OUT/'length_control/4B/capture_done.json')
    assert captures['4B']['n']==480 and captures['14B']['n']==288 and cap_control['n']==384
    for folder in [OUT/'4B',OUT/'14B',OUT/'length_control/4B']:
        anchors=read(folder/'anchors.json')
        assert all(r['mlp_replay']==0 and r['block_chain']==0 for r in anchors)
    audit=read(OUT/'material_audit.json')
    assert all(x['strict_named_entity_disjoint'] for x in audit['length_control']['families'].values())
    c4={r['group']:r for r in read(OUT/'4B/composition.json')['groups']}
    c14={r['group']:r for r in read(OUT/'14B/composition.json')['groups']}
    assert set(c14)<=set(c4) and len(c14)==48
    matched={}
    for split in ['development','entity_holdout','wording_holdout','joint_holdout']:
        ids=[k for k,v in c14.items() if v['split']==split]
        matched[split]=dict(groups=ids,summary={s:{method:statistics.median(d[k]['relative_errors'][method][-1] for k in ids)
                for method in ['negation','style','additive','dev_mean_interaction']} for s,d in [('4B',c4),('14B',c14)]})
    write(OUT/'matched_model_comparison.json',dict(comparison=matched,n_matched_wording_groups=48,
        note='Same source IDs/material, within-model normalized prediction errors. No cross-model coordinate alignment or population significance.'))
    now=datetime.now().astimezone()
    repair_table=[]
    for s in arms:
        old=measurements[s]['summary']['legacy']
        repair_table.append(f"| {s}，72个旧条件对 | {old['legacy_wrong']['relative_error']['median']:.4f} | {old['corrected']['relative_error']['median']:.4f} | {second[s]['summary']['legacy']['second']:.4f} | {second[s]['summary']['legacy']['exact_nonlinear']:.4f} |")
    combo_table=[]
    for label,result in [('4B主实验',arms['4B']),('14B主实验',arms['14B']),('4B等长+实体隔离诊断',control)]:
        for split,ch in [('entity_holdout','主实体配置留出'),('wording_holdout','表述留出'),('joint_holdout','配置+表述留出')]:
            r=result['composition'][split];e=r['errors']
            combo_table.append(f"| {label} | {ch} | {r['n_groups']} | {e['negation']['median']:.4f} | {e['additive']['median']:.4f} | {e['dev_mean_interaction']['median']:.4f} | {r['first_token_correct']}/{r['denominator_first_token']} |")
    head_table=[]
    for f,r in arms['14B']['heads'].items():
        head_table.append(f"| {f} | {r['top8_correct_preprojection']} | {r['overlap']}/8 | {r['rho_abs_causal']:.4f} |")
    intervals=[]
    for label,result in [('4B',arms['4B']),('14B',arms['14B']),('4B等长诊断',control)]:
        for split in ['entity_holdout','wording_holdout','joint_holdout']:
            v=result['composition'][split]['paired_calibrated_minus_additive']
            intervals.append(f"- {label}/{split}：校正减加性误差的配对中位数 {v['median']:+.4f}，源组bootstrap 95%区间 [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]；负值表示校正有利。")
    marker='可信测量重建与独立条件组合检验'
    report=rf'''### 目标、状态与证据范围 [{now:%Y-%m-%d %H:%M}]

用户要求先修复测量和证据链，再基于可信部分尝试整合理论。本阶段已完成：数学/统计修复、关键命题依赖登记、4B/14B顺序CUDA复核、分组条件组合检验、4B等长且实体隔离的诊断、全坐标图与只读查询。它完成的是一个有资源边界的研究阶段，不是整个语言理论或AGI问题。

主运行ID：`rdc_trusted_rebuild_20260923`；结果根目录：`tests/glm5/result/rdc_trusted_rebuild_20260923/`。运行前材料与设计已保存。4B正式480条，14B正式288条，诊断384条，共1152条正式前向；另有三个12条采集试运行，其中一次14B在采集后导出offload参数时失败，失败目录保留。模型始终顺序加载，本轮未运行GLM4。

4B在CUDA上原生bf16/eager/batch1；14B显存上限10GiB，框架把部分权重存于CPU并调度计算，非量化，其他采集口径一致。均无padding、截断、KV缓存，不使用chat模板；记录实际token ID、首token预测、整词表logits、末位置所有HiddenState返回边界与全部原生坐标、末块gate/up/act及残差子步。**这是末位置全坐标采集，不是全token位置场**；末个HiddenState返回项为final norm后，原始末块输出另存h2。首token正确不等于完整生成及停止正确。

### C001：可信测量重建，而非沿用错误公式

令 $g=W_gx,u=W_ux,m=W_d[\operatorname{{silu}}(g)\odot u]$。修正后的局部Jacobian为：

$$
J_M(x)=W_d\left[\operatorname{{diag}}(u\odot\phi'(g))W_g+\operatorname{{diag}}(\phi(g))W_u\right],\quad \phi=\operatorname{{silu}}.
$$

二阶增量诊断采用：

$$
\Delta a_2=u\phi'(g)\Delta g+\phi(g)\Delta u+\phi'(g)\Delta g\Delta u+\tfrac12u\phi''(g)(\Delta g)^2.
$$

这里乘法按坐标逐项进行，$W_d$最后映射至残差坐标。这些是**已有架构的微分恒等关系与Taylor近似**，不是新发现的语言定律。预测使用两条件下已观测到的上游g/u及真实W_d；它验证局部计算和近似质量，不是仅由语言关系直接预测隐藏场的提取器。二阶诊断在看到主实验结果后新增，已经明确标为事后诊断，没有宣称独立盲测。

自动微分、合成有限差分、真实模型工作点的fp64局部函数有限差分均通过。最后一项检查针对平滑fp64参考函数，而不是声称bf16舍入函数可微。所有正式采集的原生MLP重放与block残差接续锚误差均为0。新代码复算旧错误统计的最大余弦差：4B={arms['4B']['old_wrong_statistic_replay_max_abs']:.3g}，14B={arms['14B']['old_wrong_statistic_replay_max_abs']:.3g}。因此新旧实验对象已校准，修复带来的差异不是偷偷更换输入。

下表为相对范数误差中位数，越小越好；不把1−cos当误差：

| 模型/材料 | 原错误式 | 正确一阶 | 正确二阶 | 完整局部非线性重建 |
|---|---:|---:|---:|---:|
{chr(10).join(repair_table)}

“完整局部非线性重建”仍有原生bf16计算/保存边界差异，只作数值参照，不能作为机制提取成绩。正确一阶在新材料各拆分上的误差、范数比、余弦及源组区间见各模型`measurement_repair.json`与`summary.json`，不只报告有利中位数。

通道分解同时记录 $\|a\|^2,\|h\|^2,2\langle a,h\rangle$，归一于 $\|a+h\|^2$。另记录有符号投影 $\langle a,a+h\rangle/\|a+h\|^2$；其与h项相加为1，但可为负、可大于1，**不是因果概率或独立能量百分比**。原“96.5%残差输入”和“6%高阶残差”撤回。可以保留某些范围内残差通道较强的候选观察，不能恢复原精确百分比与“归一化透明”的解释。

### C002：修复头空间、统计与证据依赖

14B自然响应改为L37 `o_proj` 输入的真实头通道，保持原先逐样本头L2份额再取中位数的聚合方式，与3093干预编号同基比较：

| 族 | 修复后的自然top8 | 与因果top8重叠 | 与因果幅度绝对值的平均秩相关 |
|---|---|---:|---:|
{chr(10).join(head_table)}

40头随机各取8头时平均重叠为1.6；这里只有三族，不能从重叠低直接推导必要性或机制分离。幅度统计与因果效应、条件差分与特定替换仍不是同一个对象。原投影后切片比较作废，新结果作为范围明确的观察保存。

CPU修复同时完成：3075改用条件二阶差分与Möbius系数求和的正确关系，保留467/1792个差分超过0.02这一冻结数值观察；3088/3091改为平均秩与模型块置换，并纳入已有14B；3092修正双侧p至有符号z转换，因缺少有效联合null而**不发布合并显著性**；3099移除观测条件下不可达的三倍Jaccard门，改报同模型同族对的风格减formal差值，不以新事后门宣布机制成立。

`evidence/claims.json`按稳定ID连接原脚本哈希、源记录位置、结论状态和下游命题。原始程序/结果不覆盖，正确实现作为新版本入口，防止破坏旧seal；这不意味着旧程序已自动改对。旧“完整RDC机制链”保持incomplete；历史候选与本轮独立复核分开。关键依赖是人工审定的有限子图，不是全仓库所有理论依赖已经清理完成。

### C003：条件组合预测与真正的推广边界

新材料包含分类关系、施受角色、左右关系链三类；每条源命题设置风格S与否定N两个条件的四种组合。4B有48个语义源组×2种表述×4条件=384条；14B根据65.57秒的12条试采成本，事前按ID均衡选24个源组×2表述×4条件=192条，另各有96条旧材料。14B选择规则按资源成本而非结果制定，记录于资源修订文件。模型不是随机抽样，不能据两个模型进行总体显著性推断。

组合预测定义为：

$$
\hat h_{{11}}^{{add}}=h_{{10}}+h_{{01}}-h_{{00}},\qquad I=h_{{11}}-h_{{10}}-h_{{01}}+h_{{00}}.
$$

这是定义与候选模型，不是恒等式已证明可加。另在开发集各族学习 $\bar I_{{dev,f}}$，用 $\hat h_{{11}}=\hat h_{{11}}^{{add}}+\bar I_{{dev,f}}$ 原样预测各留出集。预测时只使用三个单/无条件实测状态和开发集参数，**不输入留出双条件目标状态或答案标签**。基线包括零变化、仅否定、仅风格；报告的是相对于真实双条件变化的误差，而不是容易被共同基底抬高的原始状态余弦。

数据审计发现主材料的role/chain中，某些留出主实体名字在开发集中曾作为另一个参与者出现。主实验因此只能称“未见主实体配置”，不能称严格未见词汇。原材料保持不动；新增4B诊断把所有参与者留在同一实体分区，并用ordinary/formal、really/not使每组四条件token长度完全相同。这里“未见实体”指本阶段提取器开发材料未见，不表示LLM预训练从未见过名字。词义/语用仍可能改变，且诊断同时修改了长度与实体分区，不能单独归因于其中一个因素。诊断使用同一批语义源设计，是事后诊断而非新增384个独立语义样本。另保存`matched_model_comparison.json`，仅比较两模型共有的48个表述组；不以不同样本量的中位数直接判断规模优劣，不直接对应跨模型坐标。

下表为final norm后、末位置变化的相对误差中位数；主实验“配置留出”按上述限定解释，诊断才满足参与者名字不重叠：

| 模型/协议 | 留出类型 | 源组数 | 仅否定基线 | 单条件加性 | 开发集交互校正 | 首token对/总数 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(combo_table)}

配对校正收益的源组bootstrap（2000次，区间仅针对固定材料设计，不作大量逐项显著性确认）：

{chr(10).join(intervals)}

这些结果要求区分“共享计算存在”与“共享交互向量稳定可搬运”。固定族均值交互若仅在相同表述下有用、换表述后变差，就否定其作为跨表述统一规则的主张。即使加性超过单条件基线，仍有未解释误差；不能将中等预测能力命名为全机制闭合。行为指标只覆盖首token，不据此声称完整推理与停止能力被解释。

### 三图谱与RDC的实际增量

1. 外部图新增有稳定源组ID的三类关系、两个条件、两种表述及明确拆分；材料标签是实验操作，不当作模型内部模块事实。
2. 内部图新增正确采集的末位置全层全坐标、真实头输入及MLP分解，提供原始值与逐行RMS标准化图，原生坐标不排序，不用Top-K定义主干。原始全坐标数组可查询；图形只是聚合视图，均值可相互抵消。
3. 关联图新增可复算的局部JVP、二阶诊断和有失败边界的组合预测。模型真值前向与已知权重重建明确标为核对对象，没有充当从外部语言提取规律的成绩。

**本轮可以采用的可信核心，是正确的条件化局部计算构件和经复算的有限范围观察；不是一套已经完成的全局语言理论。**

全局上仍以 $p(x_{{t+1}}\mid x_{{\le t}})=\operatorname{{softmax}}(W_U\operatorname{{Norm}}_f(h_{{L,t}})+b)$ 作为架构核对关系。它与局部Jacobian公式均非新数学发现。RDC尚缺：从已知语义关系/角色/语境预测实际工作点与路由、跨层可复用的状态转移、可执行的历史/KV状态定义、未见更深组合及自然生成验证。**本轮没有建立新的RDC统一定理；也没有用“动态路由”一词把推广失败自动变成理论成功。**

从第一性原理看，当前关键是把“固定参数”与“条件改变有效作用”区分清楚。Jacobian中的 $u\phi'(g)$ 与 $\phi(g)$ 说明相同W_g/W_u/W_d在不同工作点会形成不同有效连接；二阶项保留了gate与up的交互。这提供一个可信的计算层骨架。要成为语言理论，还必须用更简单、可推广的关系描述预测这些工作点和交互，不能把全部原模型中间量重新算一遍当作解释。

### 硬伤、失败记录与下一阶段大任务

- 仅英语受控短输入、三类任务、两种表述、末位置；没有长语境、组合深度扩展、全token场或完整自回归生成。有限名字和模板仍限制统计外推。
- 使用单条件真状态预测组合状态，属于局部条件组合检验；距离无需运行原模型的机制提取器还有明显缺口。
- 主材料参与者重叠已追加修正；严格实体隔离与等长只在4B诊断完成，14B此控制未做，不能称两模型都通过严格词汇推广。
- 首次14B导出时Accelerate的meta占位参数不能直接复制，已经改为读取同一检查点的实际权重；失败产物保留。一次CPU统计输入选择误取层扫描数组，已改为明确arbitration数组后重算。没有把失败运行冒充完成。
- 原始脚本曾在校准过程中演进，已恢复并验证与执行时SHA相同的代码快照；这是内容身份核验，不补造事前时间戳。
- 已有139项等历史可追溯检查不能视作本轮全部重新认证。证据清理仍有未覆盖消费者，原候选不自动升级。

下一大阶段围绕一个问题组织：**能否从输入时已知的关系/角色和基态，预测随表述改变的交互与跨层接续？** 先冻结新的、严格实体/模板隔离且匹配位置的材料，再比较原生坐标条件回归、类型化关系图/超图与局部转移三条路线；显式对照词身份、长度、位置和仅单条件状态。材料与选择分开，针对关键效应扩大独立源组而不是复制改写。只有候选在新组合与更深组合上稳定优于简单基线，才追踪相应单元/标量参数路径，并推进到完整下一token分布、首次分叉与自由生成。

资源：4B正式采集 {captures['4B']['elapsed']:.1f}s，14B {captures['14B']['elapsed']:.1f}s，4B诊断 {cap_control['elapsed']:.1f}s（各自采集回执口径，不等同整个任务耗时；另有加载/试运行/分析）。所有旧数据保留，无原场清理。核心代码位于`tests/glm5/rdc_trusted_measurements.py`、`phase2751_trusted_rebuild.py`、`phase2751_evidence_repair.py`、`phase2751_rebuild_summary.py`、`phase2751_length_control.py`、`phase2751_second_order_diagnostic.py`。只读接口 `/api/rdc-construction/trusted-rebuild` 提供索引、命题、模型摘要与完整坐标行；已通过接口契约测试，未声称已重启用户正在运行的服务器。

通俗说明：我们先把量尺校正了，并确认有些局部计算确实能复现和近似预测。新的组合测试说明，“同一批参数会复用”并不意味着“同一条交互修正到处都能搬用”。理论接下来要解释的是为什么语境和表述会改变这些交互，以及能否事先预测这种改变，而不是继续把局部成功拼成已经完成的故事。
'''
    write(OUT/'trusted_kernel.json',dict(scope='bounded repaired measurements and empirical constraints',
        architecture_identities=['SwiGLU product derivative','RMSNorm derivative','energy cross-term identity'],
        validated_measurements=['native MLP/block anchors','real-working-point finite differences','same-basis head comparison'],
        empirical_results={s:f'{s}/summary.json' for s in arms},control='length_control/4B/summary.json',
        not_established=['universal routing rule','text-only mechanism extraction','full autoregressive theory','AGI mechanism'],
        next_question='Predict context-dependent interactions without target-state leakage, then connect layers and generation.'))
    report_path=OUT/'phase_report.md';report_path.write_text(report,encoding='utf-8')
    glm=root/'research/glm5/docs/AGI_GLM5_MEMO.md';before=glm.read_bytes();text=before.decode('utf-8-sig')
    assert marker not in text,'Already appended; do not duplicate.'
    phase=max(map(int,re.findall(r'^## Phase (\d+)',text,re.M)))+1
    title=f'## Phase {phase}: {marker} [{now:%Y-%m-%d %H:%M}]'
    addition=('\n\n'+title+'\n\n'+report+'\n').encode('utf-8')
    assert glm.read_bytes()==before,'Concurrent memo change; retry after review.'
    with glm.open('ab') as f:f.write(addition)
    assert glm.read_bytes()[:len(before)]==before
    gpt=root/'research/gpt5/docs/AGI_GPT5_MEMO.md';gb=gpt.read_bytes();gt=gb.decode('utf-8-sig')
    gphase=max(map(int,re.findall(r'^## Phase (\d+)',gt,re.M)))+1
    pointer=f'''\n\n## Phase {gphase}: 测量与证据链修复索引——旧结论分级、版本化复算与条件组合边界 [{now:%Y-%m-%d %H:%M}]

本次按用户“先修复测量和证据链，再做独立组合推广”的指令完成。完整记录追加于 `research/glm5/docs/AGI_GLM5_MEMO.md` Phase {phase}；详细结果为 `tests/glm5/result/rdc_trusted_rebuild_20260923/phase_report.md`，命题状态见 `evidence/claims.json`（相对同结果根目录）。

修正/撤回：3100的SwiGLU导数、投影前后头编号比较、1−cos当误差与能量补数；3075的次模等价式；3088/3091的并列秩和逐行置换；3092不再声称有效合并显著性；3099不可达的三倍门。原文件保持历史身份，正确实现是新版本入口，不可继续把旧“完整机制链”等判决作为当前可信结论。

实际完成4B/14B顺序CUDA复核与分组组合检验，以及4B等长且参与者隔离诊断。主材料的部分角色/关系链参与者名字跨拆分重叠已经降级说明，不能称严格未见词汇。保留经校验的局部计算与有边界的组合观察；RDC整体理论仍未完成，所有局部核验不得自动外推至完整语言机制或AGI。全部原数据与失败运行保留，新增结果可通过只读查询接口回查。
'''
    assert gpt.read_bytes()==gb,'Concurrent GPT memo change; inspect before append.'
    with gpt.open('ab') as f:f.write(pointer.encode('utf-8'))
    assert gpt.read_bytes()[:len(gb)]==gb
    # Snapshot current source content as well as older execution-hash recovered versions.
    snapshots=OUT/'code_snapshots';snapshots.mkdir(exist_ok=True)
    for p in (root/'tests/glm5').glob('*2751*.py'):
        if p.name.startswith(('phase2751_trusted','phase2751_evidence','phase2751_rebuild','phase2751_length','phase2751_second')):
            (snapshots/(sha(p)+'.py')).write_bytes(p.read_bytes())
    receipt=dict(completed_local=now.isoformat(),completed_utc=datetime.now(timezone.utc).isoformat(),
        glm_phase=phase,gpt_phase=gphase,glm_line=before.count(b'\n')+3,gpt_line=gb.count(b'\n')+3,
        glm_original_prefix_sha256=hashlib.sha256(before).hexdigest(),gpt_original_prefix_sha256=hashlib.sha256(gb).hexdigest(),
        append_only_preserved=True,formal_forwards=1152,pilot_forwards=36,
        results={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion_receipt.json'})
    write(OUT/'completion_receipt.json',receipt)
    print(json.dumps(dict(glm_phase=phase,gpt_phase=gphase,glm_line=receipt['glm_line'],report_characters=len(report),formal_forwards=1152)))


if __name__=='__main__':main()
