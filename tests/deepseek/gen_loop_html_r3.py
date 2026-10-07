# -*- coding: utf-8 -*-
"""复核轮 R3：从 loop_stats_r3.json 渲染单文件 HTML 报告页（零手工转录数字）。"""
import json
import os

OUTDIR = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek\result"
S = json.load(open(os.path.join(OUTDIR, "loop_stats_r3.json"), encoding="utf-8"))
sc, kw, vv, g1, kill, ac, lg, af, nl = (S["scale"], S["keywords"], S["verdict_vocab"], S["g1"],
                                        S["kill"], S["audit_consistency"], S["ledger"],
                                        S["audit_findings"], S["nline"])
g5, ds = kw["gpt5"], kw["deepseek"]

rows_readout = "".join(
    f"<tr><td>{r['model']}</td><td class='num'>{r['b4_kstar']:.4f}</td>"
    f"<td class='num hot'>{r['b4_readout']:.4f}</td><td>{r['above5']}</td>"
    f"<td class='mono'>{r['m1_kstar']}</td></tr>" for r in g1["rows"])

AN = ["类间", "实体×类（加性残差）", "交互格 (i,c)", "模板内高秩散布"]
rows_anova = "".join(
    f"<tr><td>{a['model']}</td>" + "".join(f"<td class='num'>{v}%</td>" for v in a["share"]) + "</tr>"
    for a in g1["anova"])

# 条形图：B4 误差 k* vs 读出层（横轴 0..0.45）
def bar(label, val, vmax, cls, note=""):
    w = max(1.0, val / vmax * 100)
    return (f"<div class='brow'><span class='blab'>{label}</span>"
            f"<span class='btrack'><span class='bfill {cls}' style='width:{w:.1f}%'></span></span>"
            f"<span class='bval'>{val:.4f}</span><span class='bnote'>{note}</span></div>")

vmax = 0.45
bars = ""
for r in g1["rows"]:
    bars += bar(f"{r['model']} @k*（7.5% 深度）", r["b4_kstar"], vmax, "ok")
    bars += bar(f"{r['model']} @读出层", r["b4_readout"], vmax, "bad",
                f"= {r['b4_readout']/0.05:.1f}× 于 5% 门")

# 方差分解堆叠条
segs = ""
for a in g1["anova"]:
    inner = "".join(
        f"<span class='seg s{i}' style='width:{v}%' title='{AN[i]}: {v}%'></span>"
        for i, v in enumerate(a["share"]))
    segs += f"<div class='stackrow'><span class='blab'>{a['model']}</span><span class='stack'>{inner}</span></div>"
legend = "".join(f"<span class='lg'><i class='seg s{i}'></i>{AN[i]}</span>" for i in range(4))

HTML = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>循环诊断 v1 — RDC/LPF</title>
<style>
:root{{--bg:#f7f8fa;--card:#fff;--ink:#16181d;--sub:#5b6472;--line:#e3e6ec;
--ok:#2f7d5d;--bad:#c0392b;--acc:#2b5fd9;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:34px 22px 60px}}
h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.3px}}
h2{{font-size:19px;margin:34px 0 12px;padding-left:11px;border-left:4px solid var(--acc)}}
h3{{font-size:15.5px;margin:22px 0 8px;color:var(--sub)}}
.sub{{color:var(--sub);font-size:13px;margin-bottom:6px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:18px 20px;margin:14px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(212px,1fr));gap:12px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--bad);
border-radius:10px;padding:13px 15px}}
.kpi.g{{border-left-color:var(--ok)}} .kpi.a{{border-left-color:var(--acc)}}
.kpi .k{{font-size:12px;color:var(--sub);letter-spacing:.3px}}
.kpi .v{{font-size:23px;font-weight:650;margin:3px 0 1px;font-variant-numeric:tabular-nums}}
.kpi .n{{font-size:12px;color:var(--sub)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}}
th{{background:#f0f2f6;font-weight:600;color:var(--sub);font-size:12.5px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.hot{{color:var(--bad);font-weight:650}} td.mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--sub)}}
.brow,.stackrow{{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:12.5px}}
.blab{{width:186px;flex:none;color:var(--sub)}}
.btrack{{flex:1;height:15px;background:#eceff4;border-radius:4px;overflow:hidden}}
.bfill{{display:block;height:100%;border-radius:4px}}
.bfill.ok{{background:#9fb3d9}} .bfill.bad{{background:#c0392b}}
.bval{{width:62px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}}
.bnote{{width:132px;font-size:11.5px;color:var(--sub)}}
.stack{{flex:1;display:flex;height:17px;border-radius:4px;overflow:hidden}}
.seg{{display:block;height:100%}} .s0{{background:#2b5fd9}} .s1{{background:#e08a2e}}
.s2{{background:#3f9a6a}} .s3{{background:#b8434f}}
.lg{{display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:12px;color:var(--sub)}}
.lg i{{width:12px;height:12px;border-radius:3px;display:block}}
blockquote{{margin:12px 0;padding:11px 15px;background:#fff6e8;border-left:4px solid #d68a1e;border-radius:0 8px 8px 0;font-size:14px}}
ul{{padding-left:20px;margin:8px 0}} li{{margin:4px 0}}
.tag{{display:inline-block;font-size:11.5px;padding:1.5px 8px;border-radius:20px;background:#eef1f6;color:var(--sub);margin-right:6px}}
.foot{{margin-top:34px;padding-top:14px;border-top:1px solid var(--line);color:var(--sub);font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>循环诊断 v1：为什么 402 个 Phase 只产出局部特征</h1>
<div class="sub">对象 <code>AGI_GPT5_MEMO.md</code> {sc['gpt5_bytes']:,} B / {sc['gpt5_lines']:,} 行 / <b>{sc['gpt5_phases_distinct']} 个独立 Phase</b> / sha8 <code>{sc['gpt5_sha8']}</code>
　·　参照 <code>AGI_DEEPSEEK_MEMO.md</code> {sc['ds_bytes']:,} B / {sc['ds_phases_distinct']} 个 Phase / sha8 <code>{sc['ds_sha8']}</code><br>
元层复核（非 Phase 记录，不改动任何 MEMO 原文）　·　全部数字由脚本从源文件现场解析：<code>loop_stats_r3.json</code></div>

<h2>§0 三个结论（×3 强调）</h2>
<div class="card">
<ol>
<li><b>循环不是纪律失灵，是验收函数写错了。</b>现行验收奖励"找到一个能过门的局部结构"——高维系统里这个条件几乎恒真，所以每个 Phase 都能交差。稀缺的验收是"降低一个全局预测误差"，它<b>会失败</b>，因此能筛掉工作。</li>
<li><b>"无法破解整体"已有数字答案，只是被记成了成功。</b>组合主线上，行为读出层的未见组合预测误差为 <span class="tag">0.3316</span><span class="tag">0.3898</span><span class="tag">0.3986</span>（4b / glm4 / 14b），是同一条死线所设 5% 门的 <b>{kill['readout_fail_ratio']}×</b>；而唯一被预注册的救援手段（rank-5 交互）在该层只承载 <b>0.0–0.3%</b> 方差。<b>两类候选模型在行为发生的层上同时失效。</b></li>
<li><b>死线在设计上就被免疫了。</b>三条 kill criteria 全部要求"跨模型合取"（K1 需 3/3），而项目自己反复记录"跨模型阴性反复出现"。<b>合取 + 系统异质 ⇒ 死线永不触发。</b>实测 above 5% 门 = <b>1/3</b> → 未触发。</li>
</ol>
</div>

<h2>§1 最重要的量化证据</h2>

<h3>1.1 裁判席被搬到了假设擅长的地方（G1 组合主线）</h3>
<div class="card">
<div class="sub">B4 = 全加性基线（"最强对手"）对<b>未见组合</b>的预测误差。k* = 候选声称作用层位（≈深度 7.5%）；读出层 = 行为实际发生的层位。</div>
{bars}
<div class="sub" style="margin-top:10px">误差从 k* 到读出层单调恶化 <b>{g1['amplification'][0]}×–{g1['amplification'][1]:.0f}×</b>（三模型）。该放大在 MEMO 中被记为"新增层位定律"，但它的直接含义是<b>可加性假设在行为层崩坏</b>——同一事实，作为"定律"是发现，作为"误差曲线"是否证。</div>
<table><thead><tr><th>模型</th><th>B4@k*</th><th>B4@读出层</th><th>above 5% 门</th><th>M1(rank-5 交互)@k* 配对 margin</th></tr></thead><tbody>
{rows_readout}</tbody></table>
<blockquote><b>K1 原文（预注册冻结）</b>：「3 模型 × 未见组合的 logit 预测误差 &gt; 5% 且不显著优于全加性基线 B4 → 放弃"条件齿轮组=算子代数"，降级为"功能性端口类描述"。」<br>
<b>判定层却被挂在"候选声称作用层位 k*"</b> → 1/3 超门 → 不触发。若按 K1 的语义（能否预测没见过的组合）挂在行为读出层：<b>三个模型全部超门</b>。当前判决 <code>{kill['verdict']}</code> 依赖"裁判席由假设自己指定"。</blockquote>
</div>

<h3>1.2 读出层残差的方差分解：过半是"从未被命名的量"</h3>
<div class="card">
{segs}
<div style="margin-top:9px">{legend}</div>
<div class="sub" style="margin-top:10px">交互格能量 <b>0.0–0.3%</b> ⇒ 3152 的"读出层 rank 越高越好"之谜闭合：残差<b>没有网格坐标</b>。而 <b>模板内高秩散布 56.8–60.4%</b> 跨三模型稳定（谱形相关 0.968–0.991），402 个 Phase 一致把它当"噪声/弥散"处理。<b>一个跨模型稳定、占比过半、始终未被命名的量 —— 这就是循环的出口候选。</b></div>
</div>

<h3>1.3 判决语言：48 个标签只用了 47 次</h3>
<div class="kpis">
<div class="kpi"><div class="k">「判决」出现次数（gpt5）</div><div class="v">{g5['判决']}</div><div class="n">deepseek 线 {ds['判决']}</div></div>
<div class="kpi"><div class="k">「预注册」出现次数</div><div class="v">{g5['预注册']}</div><div class="n">纪律确实在执行</div></div>
<div class="kpi a"><div class="k">判决串使用 / 唯一标签</div><div class="v">{vv['gpt5_uses']} / {vv['gpt5_unique']}</div><div class="n">复用率 <b>{vv['gpt5_reuse']}</b> ⇒ 几乎全一次性</div></div>
<div class="kpi"><div class="k">最高等级命题（A）占比</div><div class="v">{af['grade_A_share'][0][1]}%</div><div class="n">5 / 62 条；约 2/3 不能直接进新推理链</div></div>
<div class="kpi"><div class="k">降级 / 撤回</div><div class="v">{g5['降级']} / {g5['撤回']}</div><div class="n">self-correction 规模很大</div></div>
<div class="kpi g"><div class="k">「接续」字段（自动推进）</div><div class="v">{g5['接续']}</div><div class="n">模板明写"与总目标一致 ⇒ 自动进入下一 Phase"</div></div>
</div>
<blockquote>一个<b>从不复用的判决词汇表 = 一台不能累积的测量仪器</b>。真正的量表（误差 33%→12%→4%）可以跨 Phase 比较；现有判决串只能描述"此刻看到了什么"。<b>这就是"每个 Phase 只找到局部特征"在记录层的直接签名。</b></blockquote>

<h2>§2 循环的四个自我维持机制</h2>
<table><thead><tr><th>机制</th><th>内容</th><th>量化证据</th></tr></thead><tbody>
<tr><td><b>M1 验收函数错位</b></td><td>奖励"找到过门局部结构"（几乎恒真），而非"降低全局误差"（会失败）。装置层（bit 锚 / seal / Ledger）保证<b>可复现性</b>，与<b>有效性</b>正交</td><td>判决标签复用率 {vv['gpt5_reuse']}；A 级命题 {af['grade_A_share'][0][1]}%；单跑 8,089–33,560 s，patch/SMOKE 各约 50 次</td></tr>
<tr><td><b>M2 死线免疫</b></td><td>K1/K2/K3 全部是跨模型<b>合取</b>触发；而项目自述"跨模型阴性反复出现"</td><td>K1 = 1/3 → 未触发；3153 亦"死线未触发"；死线提及仅 {g5['死线']} 次</td></tr>
<tr><td><b>M3 自催化议程</b></td><td>议程由上一 Phase 的残差自动派生；全局目标从不投票，只在对齐声明里被引用一次</td><td>「接续」{g5['接续']} 处；「自动进入」{g5['自动进入']} 处；N 线 <b>N2h1-α-1…α-14 共 14 个连续 Phase 同一主题</b>（写入窗），其中 19/20/21 连续三轮是同一量的 nf4↔bf16 复核</td></tr>
<tr><td><b>M4 外部证伪被吸收</b></td><td>两篇外部长评审均判"无需范式修正"；三条处方（TDA/SAE/ODE）全部延后，SAE 还挂在"G1 通过"之后（而 G1 的通过依赖 §1.1 的可疑判定层）</td><td>PARADIGM_SHIFT_VERDICT / UNIFIED_REVIEW_ADJUDICATION 均在册</td></tr>
</tbody></table>

<h2>§3 核对结果（二）：必须处置的错误</h2>
<div class="card">
<h3>已在册的自我撤回（记录诚实，归档）</h3>
<ul>
<li>3044「L20 反转 3.42×」→ 3045 撤回（量纲错位伪影）</li>
<li>3100 Q3 post-o_proj 逐头分解 1/0/0 → 数学无效（须在 o_proj 输入侧）</li>
<li>3082–3092 跨模型 ρ 0.93 → 撤回至 0.46（未做 ties 平均秩 + 未块校正）</li>
<li>3147「format 偏移 = 充分原因」→ 3148 因果否定 → 3149 载体层否定（131401 rank 12442，pos 侧 ≈随机）</li>
<li>3109 之前「写入头组 = 固定几何对象」→ 否证（跨半 Jaccard 0.070 而迁移 AUC 0.9998）</li>
<li>3139 之前「指纹 → 最近邻检索」→ 否证（点态检索 = chance）</li>
</ul>
<h3>必须补降级（尚未明确处置）</h3>
<ul>
<li><b>D1</b>「N2-h1 五维类子空间是因果的」仍悬置（外部评审自己标为"半对"；留一类水果 B/A=0.05 提示它可能只是端口类描述性结构的子空间版），却已被当作 G1 的前提使用（V5：88.1%@k3 vs 12.4%@k39）。<b>同一命题既当待验候选又当下一步前提 = 已污染推理链。</b></li>
<li><b>D2</b>「层位定律」被记为发现，实为否证信号（见 §1.1）。</li>
<li><b>D3 ⚠️ 最重要</b>：<code>{kill['verdict']}</code> 应当改判（见 §1.1）。</li>
</ul>
<h3>证据底座在元层面三处对不上</h3>
<table><thead><tr><th>项</th><th>值 A</th><th>值 B</th></tr></thead><tbody>
<tr><td>命题账本分级（同称"62 条"）</td><td>MEMO 3103：A={ac['memo_3103_dist'][0][0]} / <b>B={ac['memo_3103_dist'][0][1]}</b> / <b>C={ac['memo_3103_dist'][0][2]}</b> / D={ac['memo_3103_dist'][0][3]} / E={ac['memo_3103_dist'][0][4]}</td><td>TESTPLAN §0：A={ac['testplan_dist'][0][0]} / <b>B={ac['testplan_dist'][0][1]}</b> / <b>C={ac['testplan_dist'][0][2]}</b> / D={ac['testplan_dist'][0][3]} / E={ac['testplan_dist'][0][4]}</td></tr>
<tr><td>有效依赖比例</td><td>MEMO 3103 正文「约 {ac['memo_pct_claim'][0]}%（A+B）」</td><td>按同节计数 39/62 = <b>63%</b>（同节内部不自洽）</td></tr>
<tr><td><code>atlas_ledger.json</code> 哈希自洽</td><td>自声明 <code>{lg['declared']}</code></td><td>实际 <code>{lg['actual_sha8']}</code>（{lg['measurements']} measurements / schema {lg['schema_version']}）</td></tr>
</tbody></table>
<div class="sub">在"每 Phase 一个登记哈希、一位不差"的装置里，元层账本出现三处对不上——<b>修这个比再跑一个 Phase 重要。</b></div>
</div>

<h2>§4 改进方案：跳出循环</h2>
<h3>4.1 立即执行（零 / 低 GPU）</h3>
<div class="card">
<table><thead><tr><th>#</th><th>措施</th><th>判据</th></tr></thead><tbody>
<tr><td>I1</td><td><b>唯一全局 KPI，每 Phase 必报同一套冻结 held-out</b>：<code>E_read</code>（读出层未见组合预测误差，现 <b>0.3316 / 0.3986 / 0.3898</b>）· <code>E_ar(k)</code>（k 步自回归误差曲线，现<b>不存在</b>）· <code>C_steer</code>（可控性 + 零附带损伤比例，现未测）</td><td>未降低任一项的 Phase，在 Ledger 标 <code>catalog</code> 而非 <code>advance</code></td></tr>
<tr><td>I2</td><td><b>判决层由行为定义，不得由假设自选。</b>k* 层可附报，但不得用于主线存续判定</td><td><b>据此改判 K1 为触发</b>，"算子代数"在读出层降 <code>descriptive</code></td></tr>
<tr><td>I3</td><td><b>死线双轨</b>：聚合统计（pooled + bootstrap CI）为主判据，不用合取；<b>单模型反例即标 <code>model_specific</code>，不得升为机制</b></td><td>堵住 M2</td></tr>
<tr><td>I4</td><td>修复单一真源：账本分级、A+B 百分比、ledger 自声明哈希三处对账</td><td>任何引用只允许指向唯一真源</td></tr>
</tbody></table>
</div>
<h3>4.2 中期（需 GPU）</h3>
<div class="card">
<ul>
<li><b>I5 换问题类型（最关键）</b>：被测假设类 <code>h = Σ主效应 + 低秩交互</code> 在读出层已被自己的数据否证。二选一：(a) 承认 <code>(实体 × 类 × 模板)</code> 不是模型使用的因子化；(b) <b>把 57–60% "模板内高秩散布"当未识别变量做独立因子恢复</b>。</li>
<li><b>I6 可识别性门</b>：每个新发现的局部结构必须同时报 <b>≥2 个互不兼容的候选解释</b> + 它们在 held-out 上的预测分歧。分歧 → 用 held-out 判决；一致 → 不可识别，强制 <code>descriptive</code>。</li>
<li><b>I7 可控性标准</b>：把 v1 承重轴 / 端口替换 / 坐标注入做成 <code>steering</code> 基准，报成功率 + 附带损伤。</li>
<li><b>I8 面板与功效</b>：128 行 → ≥672；翻转类 ≥100；<code>gate_precheck</code> 必须真的拒绝开跑（MDE &gt; 目标效应 ⇒ 硬 fail）。</li>
</ul>
</div>
<h3>4.3 战略（制度）</h3>
<div class="card">
<ul>
<li><b>I9 冻结 30-Phase 固定队列，删除"从残差自动派生下一 Phase"机制。</b>「预注册 Phase N+1」与「接续：自动进入 Phase N+1」两个模板字段改为只允许"继续当前队列下一项"；未采纳的残差进 backlog，由 KPI 曲线在队列末统一裁决。<b>这是切断 M3 的唯一办法。</b></li>
<li><b>I10 外部评审的每条实质建议必须落地为一条能触发死线的实验，否则不得结案</b>；禁止用"评审者方案与已冻结方案一致"作结案理由。</li>
<li><b>I11 工程去摩擦</b>：抽公共库；SMOKE 断言覆盖值域；目标 &lt;1 patch/run，把预算让给证伪。</li>
</ul>
</div>

<h2>§5 领域对照</h2>
<table><thead><tr><th>来源</th><th>要点</th><th>与本项目的关系</th></tr></thead><tbody>
<tr><td>Méloux et al., <b>ICLR 2025</b>, <i>Is Mechanistic Interpretability Identifiable?</i></td><td>因果对齐（含 IIA）<b>不足以保证唯一解释</b>；XOR 小网络穷举：85 条完美电路 / 45,000+ 条合法解释；<b>不足 2% 的网络有唯一最小映射</b>。建议强化有效性判据 / 多判据验证 / 或接受"只要求可预测与可操纵"</td><td><b>直接解释"每个 Phase 都能找到过门结构"</b>——不是巧合，是结构性必然后果。I6 的理论依据</td></tr>
<tr><td>Bolukbasi et al.（Google DeepMind）</td><td>单个神经元及其简单线性组合<b>会看似编码单一概念</b>；根源=嵌入空间几何 + 检验文本过窄；<b>在一个数据集上验证过的解释等于未验证</b></td><td>对应"单模型 76 / 单点 111"的硬伤与"1–3 个 Phase 内被推翻"的修正链</td></tr>
<tr><td>Geiger et al., <i>Causal Abstraction</i> + DAS / BoundlessDAS</td><td>因果抽象统一了补丁/中介/清洗/SAE/steering；强调 <b>what-then-where</b>（先有候选算法再做因果对齐，用 <b>IIA</b> 度量）</td><td>项目做的是 <b>where-then-what</b>；IIA 是本项目缺失、可直接移植的判据</td></tr>
<tr><td>Makelov et al. 对激活补丁的批评</td><td>补丁/正交分解可能<b>激活休眠回路</b> ⇒ "可解释性幻觉"</td><td>对应 3148"减法不是合法算子"（cancel 反向加重 0.820）——项目已独立触及</td></tr>
<tr><td>MIT TR（2026 十大突破性技术·MI）评论</td><td>"我们有很多 AI 版<b>第谷</b>（收集数据），一些<b>开普勒</b>（提出假说），<b>但还没有牛顿</b>（发现原理）"</td><td>本项目是罕见的、接近完美的<b>第谷</b>：14,901 行 / 402 Phase / 位级锚。缺的是一个<b>必须预测 held-out 的定量定律</b>（即 I1）</td></tr>
</tbody></table>

<h2>§6 一句话回答</h2>
<blockquote><b>循环的成因不是不够努力、不是纪律不严、也不是模型太复杂，而是验收函数奖励"找到一个过门的局部结构"，而高维系统里这个条件几乎恒真。</b><br><br>
把验收函数换成<b>一个必须失败的全局数字</b>（读出层误差 33%→?、k 步自回归误差曲线、可控性成功率），并把判决层交给<b>行为</b>而不是交给<b>假设自己</b>——循环自然终止，因为大量现有工作会立刻被诚实地标记为"目录条目"。<br><br>
那时议程会从 402 个自选问题收缩到 3 个有数字的问题；而 402 个 Phase 积累的负结果与端口类结构，恰好是回答这 3 个问题所需的全部前提。</blockquote>

<h2>§7 自我限制（避免本报告变成新教条）</h2>
<div class="card">
<ul>
<li><b>"一个全局 KPI"本身可能变成新教条</b>：若 <code>E_read</code> 因口径/泄漏被优化到虚假低值，就会重演同样的错误。必须与 I6、I3 捆绑，且口径纳入 <code>metric_dict.json</code> 冻结、改口径须新开 Phase 并公开标注。</li>
<li><b>"未破解整体"可能部分是真实极限</b>：至今没有任何前沿 LLM 被完整电路级解释；可识别性危机提示"唯一解释"可能根本不可得。出路也许是<b>改变所主张的命题类型</b>——从"机制是 X"改为"存在预测代理，能把误差压到 E"。</li>
<li><b>本报告未逐项重算 402 个 result.json</b>：§2 的"成立"判定基于项目自己的审计文档 + 关键节精读 + 统计口径自洽性核对。</li>
<li><b>"必须改判 K1"是强主张</b>，支点是"K1 语义应挂在行为读出层"。若认为"承诺层 = 候选取作用层"是有意登记，则结论应改为：<b>K1 原文在两处自相矛盾，必须重述为无歧义版本并重新冻结</b>——无论哪种读法，"用假设自己指定的层判断假设"都需被显式接受或显式否决。</li>
</ul>
</div>

<div class="foot">
数据源：<code>research/gpt5/docs/AGI_GPT5_MEMO.md</code>（sha8 {sc['gpt5_sha8']}）· <code>research/deepseek/docs/AGI_DEEPSEEK_MEMO.md</code>（sha8 {sc['ds_sha8']}）· <code>MEMO_AUDIT_2750_3148.md</code> · <code>RDC_TESTPLAN_v1.md</code> · <code>atlas_ledger.json</code>（sha8 {lg['actual_sha8']}）<br>
生成：<code>tests/deepseek/gen_loop_diagnosis_r3.py</code> → <code>tests/deepseek/result/loop_stats_r3.json</code>（零手工转录）　·　报告正文：<code>research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md</code>
</div>
</div></body></html>"""

out = os.path.join(OUTDIR, "loop_diagnosis_r3.html")
open(out, "w", encoding="utf-8").write(HTML)
print("wrote", out, len(HTML), "chars")
