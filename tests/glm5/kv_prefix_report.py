"""Report, visualization and append-only memorandum delivery for Phase 2949."""

import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from b36_semantic14_memo import append
import kv_prefix_run as C


A, B, P = C.A, C.B, C.P
TITLE = "14B 多前缀来源、L25H18 答码方向与自产 State 的原生 K/V 坐标"
LABELS = {
    "early-state": "K/V 路线的 State 边际损失超过同层头",
    "state-route": "State 优先路线的 State 边际损失超过同层头",
    "early-code": "K/V 路线的 P/Q 边际损失超过同层头",
    "early-local": "K/V 路线可由原生槽恢复的方向分量超过同层头",
    "kv-block": "冻结坐标清零的答码损失超过匹配坐标",
    "kv-rescue": "冻结坐标原生写回相对早层阻断的净恢复",
    "kv-rescue-specific": "冻结坐标原生写回超过匹配坐标写回",
    "native-full-rescue": "整槽原生 K/V 写回相对早层阻断的净恢复",
    "reverse": "将早层阻断后冻结坐标反写至完好 H18 的 P/Q 损失",
    "state-switch": "K/V 路线的当场 State 换 token 率超过同层头",
    "state-route-switch": "State 优先路线的当场换 token 率超过同层头",
}


def fmt(record, percent=False):
    if record is None:
        return "缺样本"
    f = "+.2%" if percent else "+.4f"
    a, b = record["ci"]
    return f"{record['mean']:{f}} [{a:{f}}, {b:{f}}]"


def route(values):
    return "、".join(f"L{l}H{h}（{site}）" for l, h, site in values)


def table(headers, values):
    return "\n".join("| " + " | ".join(map(str, x)) + " |" for x in (headers, ["---"] * len(headers), *values))


def matching(f):
    dose = np.asarray(f["coord_dose"])
    targets = np.concatenate([dose[:2, f["coordinates"][0]].sum(1), dose[2:, f["coordinates"][1]].sum(1)])
    rows = []
    for j, (k, v) in enumerate(f["random_coordinates"], 1):
        score = np.concatenate([dose[:2, k].sum(1), dose[2:, v].sum(1)])
        ratios = [f"{value / base:.2f}×" if base > 0 else "—" for value, base in zip(score, targets, strict=True)]
        rows.append((j, *ratios))
    return table(["坐标对照组", "K 写回剂量", "K 清零剂量", "V 写回剂量", "V 清零剂量"], rows)


def intro(p, f):
    selection = f["selection"]
    heads = table(["发现候选", "E 处 State 边际损失", "G K/V 写回对 H18 的预测方向分量"],
                  [(route([spec]), f"{s:+.4f}", f"{v:+.4f}") for spec, s, v in zip(selection["shortlist"],
                      selection["state_loss"], selection["local"], strict=True)])
    coalitions = table(["发现路线", "三处 Value 阻断", "方向分量", "State 边际损失", "真实选中拒绝词"],
                       [(name, route(r), f"{v:+.4f}", f"{s:+.4f}", flip)
                        for name, r, v, s, flip in zip(("答码方向排序", "State 边际排序", "合并秩排序"),
                            selection["coalitions"], selection["coalition_local"], selection["coalition_state"],
                            selection["coalition_alternatives"], strict=True)])
    k, v = f["coordinates"]
    return rf"""# {TITLE}

Phase 2948 的事实词干预已测到词面相同时自产缓存发生变化，方向特异性与终点中介没有过门。本轮 `{C.RUN}` 使用本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`，扩大搜索位置并用真正的答码敏感方向筛选单坐标。需要同时检验两件事。早层阻断能否让模型当场改选 State，且自然续写后改变 P/Q。保持词面时，早层造成的差异能否由 H18 读取指定的 G 坐标传到 P/Q。

### 材料和因果可见性

自编六行记录模板，同家族交换事实角色，保持双方候选词、码本和其他字段。两类规则分别为清单包含与排除、原始与提案决策。交叉两个生成距离及 State/Note 顺序，每家族八个视图。试跑 {C.SIZES['pilot']}、发现 {C.SIZES['discover']}、留出 {C.SIZES['confirm']} 家族。三份词表互斥，并排除所列历史路线材料中的已见词。发现词表 {len(p['materials']['discover']['words'])} 个词，留出词表 {len(p['materials']['confirm']['words'])} 个词。每格无干预行为门为 90%，保留逐 token 无约束贪心和 P/Q，只有行末空白可按模型真实预测对齐。确认包含未参与选择的词组合，使用同一任务模板和模型快照。共享构式和跨家族复用的词降低独立性。

E 是 State 出现前的最后一个 query，G 是这个 State token 的位置，C 是最终 `Code:` query。早层搜索限 L0–23，E 的 L$l$ 改动最早在 G 进入 L$l+2$ 的残差。六类互斥掩码取自用户前缀，为正确事实词、拒绝事实词、余下事实及决策文本、两个竞争码本行、规则句及 State 输出格式提示。逐样本偏移和 token ID 存入材料。只在指定 E query 和选定头把掩码内 Value 清零，保持真实分母、K、其他头与位置。自恢复取同一行同一位置的实际原生 Value。随后的 query、注意力和 MLP 继续按实际分支计算。

### 原生坐标贡献与冻结规则

从原模型的 C 读出反传 $d=\nabla_o(\ell_Q-\ell_P)$，$o$ 为 L25H18 经过注意力加权、尚未投到残差流的 128 维输出。对每个后续 block 重新加载同一原始权重，C 重新计算自己的 Q/K/V，C 之前的原生缓存由因果性固定。$d$ 是完好轨迹处的局部导数。用于筛查的坐标变化来自真实早层阻断，无需在 K/V 分析中预设目标 State 翻转。

对每个早层分支保持 C 的活体 $q$ 和其余槽，设 $u$ 是排除 G 后的归一化加权 Value，$a=\mathrm{{softmax}}(s)_G$，$s_G=q^\top k/\sqrt{{128}}$，故 $o=u+a(v-u)$。恢复的原生 G 向量为 $k^*,v^*$，对应权重为 $a^*$。每个原生 Key 坐标带来的分数变化为 $\delta s_j=q_j(k^*_j-k_j)/\sqrt{{128}}$，$\Delta s=\sum_j\delta s_j$。两个改动顺序平均后的可加分解为

$$
\begin{{aligned}}
K_j &= \delta s_j\frac{{a^*-a}}{{\Delta s}}\,d^\top\left(\frac{{v^*+v}}{{2}}-u\right),\\
V_j &= \frac{{a+a^*}}{{2}}\,d_j(v_j^*-v_j),\\
\sum_j(K_j+V_j) &= d^\top(o^*-o).
\end{{aligned}}
$$

$\Delta s$ 近零时用同点 sigmoid 导数。该等式核对 FP32 头输出投影，完整 P/Q 的实际变化另用 BF16 原模型写回测量。K 是 G 的 RoPE 后坐标，V 是 value projection 的原生输出坐标，H18 在 GQA 组 {18 // 5}。只改 C 时 H18 的一次读取，G 位本身和兄弟头继续使用实际缓存。冻结 K/V 各 {C.COORDS} 个坐标。K 按原生清零和恢复的 $|\delta s_j|$ 匹配随机剂量，V 按原生 $W_O$ 列范数乘该坐标改变量的绝对值匹配。随机选择取发现集同类剂量近邻。早层头的对照在同层同掩码匹配原生写入范数。种子、匹配池和全部坐标分数随冻结文件保存。

{heads}

{coalitions}

冻结的答码方向路线为 {route(f['early'])}。State 优先路线为 {route(f['state_route'])}。答码路线的两组对照依次是 {' ／ '.join(route(x) for x in f['early_controls'])}。State 路线的两组对照依次是 {' ／ '.join(route(x) for x in f['state_controls'])}。冻结 K 坐标 `{' '.join(map(str, k))}`，V 坐标 `{' '.join(map(str, v))}`，索引从零起算。候选和随机对照在留出响应开封前全部冻结。下表给出发现集随机坐标组与冻结坐标组的剂量比，1× 表示两组剂量相等。

{matching(f)}

十一个主终点采用双侧 {100*p['gate']['confidence']:.3f}% 家族 bootstrap 区间、至少 {p['gate']['min_family']} 家族和正的区间下界。P/Q、State 边际及方向分量要求效应至少 0.10 logit，实际换 token 率要求至少高出随机对照 2 个百分点。正态近似在标准化连续效应 0.4 下功效为 80 家族 {p['gate']['power']['80']:.3f}、{C.SIZES['confirm']} 家族 {p['gate']['power'][str(C.SIZES['confirm'])]:.3f}，未计入词汇复用、实质门槛或稀少的离散换词。原生写回、反向移植、同剂量随机阻断和恢复是共同的中介准入条件。固定句面的计算中，State 以后任何 token 分歧会取消对应臂的自然答码身份，保留反事实数值。
"""


def report(p, f, a):
    base = intro(p, f)
    nat = a["natural"]
    nc = sum(r["correct"] for r in a["behavior_cells"].values())
    nview = sum(r["n"] for r in a["behavior_cells"].values())
    head = f"""
### 留出行为与准入

原模型 {nc}/{nview} 条正确自然生成至答码，八格详见 `confirm-behavior.json`。真实提示事实发生角色互换的 {nat['attempted']} 对中，{nat['both_correct']} 对两侧行为均正确，{nat['natural_state_and_code_flips']} 对实际 State 与 P/Q 同时反转。保留 {len(a['failures'])} 条原始行为失败及逐 token 反例。
"""
    if a["state"] != "finished":
        return base + head + f"\n本轮停在 `{a['state']}`。正式留出因果判定缺少合格原始产物，发现证据与失败原样留存。\n"
    n = a["n"]
    keys = ("early", "state-route", "early-rand1", "early-rand2", "state-rand1", "state-rand2", "kv-block",
            "kv-rand-block1", "kv-rand-block2", "kv-rescue", "native-full-rescue", "reverse", "native-v-block")
    outcomes = table(["干预", "State 实际换 token", "其中换成拒绝词", "同 State 却改后缀", "验证至 C", "自然 P/Q 换码", "固定原历史的 P/Q 换码"],
                     [(k, r["state_switch"], r["recognized"], r["same_state_then_diverged"], r["prefix_valid"],
                       r["native_code_reversed"], r["conditional_code_reversed"]) for k in keys for r in [a["arms"][k]]])
    main = table(["主终点", "效应与调整区间", "删一词的效应范围", "过门"],
                 [(LABELS[k], fmt(r, k in C.ENDPOINTS[-2:]), f"{r['leave_one_lexeme'][0]:+.4f} ~ {r['leave_one_lexeme'][1]:+.4f}",
                   "是" if r["pass_gate"] else "否") for k in C.ENDPOINTS for r in [a["primary"][k]]])
    raw = table(["实际对比", "均值与描述性 95% 区间"], [(k, fmt(x)) for k, x in a["absolute"].items()])
    physical = table(["干预", "中位 K 相对量", "中位 V 相对量", "G 注意力中位数", "H18 相对 sham 的输出 L2 中位数"],
                     [(k, *(f"{r[field]:.5g}" for field in ("k_relative_median", "v_relative_median", "alpha_median", "head_l2_from_sham")))
                      for k, r in a["physical"].items()])
    tangent = table(["干预", "方向预测均值", "真实 P/Q 均值", "家族 Pearson r", "非零逐例符号一致"],
                    [(k, f"{r['tangent_mean']:+.4f}", f"{r['measured_mean']:+.4f}",
                      "—" if r["pearson_family"] is None else f"{r['pearson_family']:.3f}",
                      "—" if r["sign_agreement"] is None else f"{r['sign_agreement']:.1%}（{r['nonzero_both']}）")
                     for k, r in a["tangent_calibration"].items()])
    roll = table(["E 臂", "总换词", "续写回放", "真实到 Code", "回放中换成拒绝词", "P/Q 保持", "P/Q 翻转", "结束原因"],
                 [(k, x["total"], x["tested"], x["reached_code"], x["recognized_tested"], x["kept_original"], x["reversed_code"],
                   "复用 " + x["shared_with"] if x["shared_with"] else ", ".join(f"{k1}={v}" for k1, v in x["stops"].items()) or "—")
                  for k, x in a["rollout"].items()])
    replay = A.load("rollout.json")
    examples = []
    for arm in ("early", "state-route", "early-rand1", "early-rand2", "state-rand1", "state-rand2"):
        subset = replay[arm]
        if subset.get("shared_with"):
            continue
        for x in sorted(subset["observations"].values(), key=lambda r: not r["verified"])[:3]:
            examples.append(f"- `{arm}` ／ `{x['id']}`，State `{x['state']}`，原码 `{x['expected']}`，" +
                            (f"实际输出 `{x['code']}`。" if x["verified"] else f"停止于 `{x['stop']}`，实际已验前缀见回放。"))
    for k, group in a["examples"].items():
        if group:
            r = group[0]["row"]
            examples.append(f"- `{k}` ／ `{r['id']}`，State `{r['values']['State']}`、拒绝词 `{r['rejected']}`、原码 `{r['code']}`。逐臂实际分歧与条件分数见 `audit.json`。")
    for failed in a["failures"][:3]:
        r, o = failed["row"], failed["observation"]
        last = o["rendered"].splitlines()[-1].strip() if o["rendered"].splitlines() else "空输出"
        examples.append(f"- 原模型行为反例 `{r['id']}`，正确 State `{r['values']['State']}`、正确答码 `{r['code']}`，最后实际生成 `{last}`。")
    if a["technical_failures"]:
        ids = "、".join(f"`{x['id']}`" for x in a["technical_failures"])
        examples.append(f"- 未通过无干预贪心或数值门的视图 {ids}。其原始 token、分数与首处分歧保留，整家族从确认扣除。")
    unique = list(dict.fromkeys(f"{x['states'][0]} / {x['codes'][0]} → {x['states'][1]} / {x['codes'][1]}" for x in nat["examples"]))
    clips = [json.loads(line) for line in (C.OUT / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cuda = max(r.get("cuda_peak_bytes", 0) for r in clips) / 2**30
    shared = max(r["shared_gpu_bytes"] or 0 for r in clips) / 2**20
    drift = max(x["sham_max"] for x in a["technical"].values())
    closure = max(x["local_identity_max"] for x in a["technical"].values())
    error = max(x["physical_max"] for x in a["technical"].values())
    adj = max(x["adjoint_error"] for x in a["technical"].values())
    reread = A.load("native-recheck.json")
    assert all(B.sha(k) == h for k, h in reread["binding"].items())
    assert reread["frozen_rank_exact"]
    projection_error = max(z["max_relative_projection_error"] for z in reread["confirm"].values())
    projection_n = sum(z["reconstructed_queries"] for z in reread["confirm"].values())
    projection_edges = sum(z["softmax_boundary"] for z in reread["confirm"].values())
    margins = "，".join(f"`{key}` {q[2]:.3f}（最小 {q[0]:.3f}）" for key, q in reread["state_margin_quantiles"].items())
    same = "，".join(f"`{k}` {fmt(r)}" for k, r in a["same_word"].items())
    bridge = "通过" if a["bridge"] else "未通过"
    positive = [LABELS[k] for k, r in a["primary"].items() if r["pass_gate"]]
    passing = "、".join(positive) if positive else "无"
    return base + head + f"""
{a['eligible']} 家族通过原始逐视图行为条件，{n} 家族／{8*n} 视图共同通过技术准入，整条 E→G K/V→H18→P/Q 路线{bridge}。达到主终点判据的对比为 {passing}。同词且直到 C 的贪心后继都适用的共同家族 {a['same_word_family']} 个，描述性可比效应为 {same}。

{main}

{raw}

{physical}

表中恢复臂的 K/V 相对量以早层阻断臂为参考，其余以未更改值为参考。该范数不含方向。下表以真正的预投影 H18 输出经过原轨迹导数作预测，另列完整 BF16 logits 实测。Pearson 在家族均值上估计。

{tangent}

### 同词传播、真实换词和反例

{outcomes}

State 相对拒绝词的实际选择边际中位数为 {margins}。实际换词的判定采用全词表最高 token。

提示角色互换的可定位自然例为 {'、'.join(unique[:4])}。当干预触发首个 State token 改选时，以当场实测 token 为入口逐轮从模型贪心核对，最多保留每臂 {C.ROLLOUT_CASES} 条、{C.ROLLOUT_TOKENS} 个自产 token 或 {C.ROLLOUT_PASSES} 轮。原句后缀仅作计算草稿，每个实入历史的 token 都由当前真实前缀下的全词表 argmax 验过。原答码指原提示的正确 P/Q，换码指这两个 token 之间的交换。复用同样早层干预的两臂不得重复计数。无干预新提示角色互换是行为参照，与干预后续写分开。

{roll}

{chr(10).join(examples)}

### 机制边界与下一阶段

两条早层路线在新词表上稳定降低原 State 的选择优势。答码方向路线的 P/Q 损失为 {a['primary']['early-code']['mean']:+.4f}，相对随机头表现为小幅提高原答码的边际。其整槽 K/V 恢复均值为 {a['absolute']['native-full-rescue']['mean']:+.4f}。冻结坐标写回均值 {a['absolute']['kv-rescue']['mean']:+.4f}，只写回 K 为 {a['absolute']['kv-rescue-k']['mean']:+.4f}，只写回 V 为 {a['absolute']['kv-rescue-v']['mean']:+.4f}。局部清零预测与实测的家族相关为 {a['tangent_calibration']['kv-block']['pearson_family']:.3f}，同轨迹的微小坐标写回只有 {a['tangent_calibration']['kv-rescue']['pearson_family']:.3f}。因此对大幅阻断的局部方向校准成立于受检材料，对逐例微小自然差异的预测仍弱。清零的相对随机增量 {a['primary']['kv-block']['mean']:+.4f}，写回的相对随机增量 {a['primary']['kv-rescue-specific']['mean']:+.4f}，都低于 0.10 的实质效应门。后续修复需检验各坐标作用的抵消、Q 改动及其他槽位并发传递的份额。

本轮定位的是原生激活坐标和某个读头的有限作用，尚无单标量权重的归因。前缀来源筛查覆盖 24×40×6 个槽组，真正跑过的单头只有 12 个，任何高阶共作用、其余 query 位置、旁路及跨读头分布式机制仍可能漏掉。阻断是局部值清零，可能超出自然幅度。保持词面时，早层换了后续背景和 Q，H18 的局部恢复只置换 G 在一次读取中的原生坐标，其余历史仍沿各分支计算。用完整换词前缀重跑的自然续写有明确生成预算。早层前缀扫描用 State 投影初筛，弱 State 投影却强条件路由的头还可能遗漏。原生可微近似依赖 BF16 运算的反传及完好轨迹，最终判断由实际写回和匹配随机对照约束。

外部图谱保留事实及规则角色、两条码本和全部生成步的对应。内部图谱新增 G 的 RoPE 后 Key 与 V 坐标、活体 Q、另槽归一化和实际受影响的单头写入。跨图谱把真实换词、同词条件下的输出边际、坐标的局部可加量各自绑定到实验产物。留出门决定对应边的证据等级。有限参数复用首先要求语境能改变同一个词在历史中的角色，以及后来不同的 query 能读出当下需要的部分。下一大阶段用两个可分辨的干预点测这一关系。在 G 自产 State 已落位后，直接扫它对提示事实和码本的读取，追溯哪些写入造成 H18 的 K/V 方向变化。另在 E 上用真实的 State 输出敏感方向建立可切换的自然样本，冻结后追踪换词经 embedding 进入 G 及后续答码。两路统一比较多槽多头预算、匹配随机对照及新角色组合。当前未形成新的 RDC 统一公式，未获得跨模型或独立节点复验。

### 技术核对与归档

所有局部原生写回、未触及的 K/V 坐标和后继介入前的两个 E 候选 logits 按原生值逐元素核对。留出无干预最大原始 P/Q 差 {drift:.4g}，原生头编辑的归一化响应误差最大 {error:.4g}，局部坐标分解残差最大 {closure:.4g}，反传重算残差最大 {adj:.4g}。另用保存的 Q/K/V、注意力质量及头输出来复算 {projection_n} 个分支 query 的投影，最大归一化误差 {projection_error:.4g}，数值饱和 {projection_edges} 个，并精确核对冻结坐标排序和三份材料。以 RTX 5080 串行逐层执行，CUDA 分配峰值 {cuda:.2f} GiB，共享显存峰值 {shared:.0f} MiB。项目原生存储政策核查 D 盘，保留 20 GiB。本地运行、单节点产物和 CPU 技术复算，保留全部失败及原始输入、分词偏移、冻结计划、分片、日志、图表、来源及哈希，见 `tests/glm5/result/kv_prefix_20261116_v1/`。
"""


def figure(p, f, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axes, result) = plt.subplots(1, 2, figsize=(13.4, 4.9), layout="constrained")
    for i, (label, color) in enumerate((("K · post-RoPE", "#32809c"), ("V · native", "#9d6132"))):
        y = np.array(f["coord_strength"][i])
        chosen = f["coordinates"][i]
        axes.plot(range(128), y, color=color, alpha=.74, lw=1.2, label=label)
        axes.scatter(chosen, y[chosen], color=color, s=15)
    axes.axhline(0, color="0.6", lw=.8)
    axes.legend(frameon=False)
    axes.set(title="Discovery · native coordinate contribution", xlabel="Coordinate index", ylabel="Signed local restoration (logit tangent)")
    keys = C.ENDPOINTS[:9]
    labels = ("E block on State", "State route on State", "E block on P/Q", "E local direction", "K/V zero vs random", "K/V rescue",
              "K/V rescue vs random", "Full native slot rescue", "Reverse coordinate transplant")
    values = [a["primary"][key] for key in keys]
    x = np.array([r["mean"] for r in values])
    ci = np.array([r["ci"] for r in values])
    result.errorbar(x, range(len(labels)), xerr=np.abs(ci - x[:, None]).T, fmt="o", color="#346f71", capsize=3)
    result.set(yticks=range(len(labels)), yticklabels=labels, xlabel=f"Paired margin ({100*p['gate']['confidence']:.2f}% interval)",
               title=f"Held out · {a['n']} complete families")
    result.invert_yaxis()
    result.axvline(0, color="0.55", lw=.8)
    result.axvline(.1, color="#ad815a", ls="--", lw=1)
    for ax in (axes, result):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT / "prefix-coordinate-effects.png"), dpi=165)
    plt.close(fig)


def main(stage, p):
    f, a = A.load("freeze.json"), A.load("audit.json")
    assert all(B.sha(k) == h for k, h in a["provenance"].items())
    text = report(p, f, a)
    P.check_write(C.OUT / "report.md").write_text(text, encoding="utf-8")
    if a["state"] == "finished":
        figure(p, f, a)
    if stage == "preview":
        return
    assert not (C.OUT / "memo-receipt.json").exists()
    tip = C.ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
    assert "\n## Phase 2949:" not in tip.read_text(encoding="utf-8-sig")
    path = C.TMP.resolve()
    owned = list(path.rglob("*"))
    assert path == C.TMP and path.parent == (C.ROOT / "tests/glm5_temp").resolve()
    assert all(x.resolve().is_relative_to(path) and not x.is_symlink() and not x.is_junction() for x in (path, *owned))
    size = sum(x.stat().st_size for x in owned if x.is_file())
    shutil.rmtree(path)
    A.save("cleanup.json", dict(path=str(path), removed=size, free=shutil.disk_usage(C.OUT).free))
    paths = {*p["sources"], Path(__file__).relative_to(C.ROOT).as_posix(), "tests/glm5/kv_prefix_recheck.py",
             "tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT / "sources.zip"), "w", ZIP_DEFLATED) as zipped:
        for key in sorted(paths):
            zipped.write(C.ROOT / key, key)
    with ZipFile(C.OUT / "sources.zip") as zipped:
        assert all(zipped.read(key) == (C.ROOT / key).read_bytes() for key in paths)
    commands = [f".venv/Scripts/python.exe -B -u tests/glm5/kv_prefix_run.py {step}"
                for step in ("pilot", "plan", "discover", "confirm", "audit")]
    commands.append(".venv/Scripts/python.exe -B -u tests/glm5/kv_prefix_recheck.py")
    commands.extend(f".venv/Scripts/python.exe -B -u tests/glm5/kv_prefix_run.py {step}" for step in ("preview", "close"))
    A.save("execution.json", dict(commands=commands, bindings=a["provenance"], audit=B.sha("audit.json")))
    first = append(P.check_write(tip), TITLE, text.split("\n\n", 1)[1])
    assert first["phase"] == 2949
    second = append(P.check_write(C.ROOT / "research/claude/docs/AGI_CLAUDE_MEMO.md"),
                    "14B 多前缀来源与 L25H18 原生 K/V 坐标索引",
                    f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`，报告 `tests/glm5/result/kv_prefix_20261116_v1/report.md`。")
    A.save("memo-receipt.json", dict(records=[first, second]))
    A.save("manifest.json", dict(files={x.name: A.digest(x) for x in sorted(C.OUT.iterdir()) if x.is_file() and x.name != "manifest.json"},
                                 sources={k: A.digest(C.ROOT / k) for k in sorted(paths)}))
