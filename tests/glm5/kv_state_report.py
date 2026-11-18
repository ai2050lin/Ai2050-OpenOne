"""Render the Phase 2950 results and append the evidence ledger."""

import json
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import kv_state_run as C
from b36_semantic14_memo import append


A, B, P = C.A, C.B, C.P
TITLE = "14B 自产 State 直接读事实与码本，以及首次输出真实换词"
LABELS = {
    "g-fact-direction": "G 事实写入阻断的可恢复方向量超过随机头",
    "g-book-direction": "G 码本写入阻断的可恢复方向量超过随机头",
    "g-both-direction": "G 两处共同阻断的可恢复方向量超过随机头",
    "g-both-pq": "G 两处共同阻断的实际 P/Q 边际损失超过随机头",
    "g-kv-rescue": "共同阻断后仅恢复 H18 的 G 槽原生 K/V",
    "g-kv-reverse": "将共同阻断后的槽反写至完好 H18 的实际 P/Q 损失",
    "e-switch": "E 原生坐标换入触发候选词改选，相对匹配坐标",
    "e-native-pq": "E 原生坐标换入对固定原历史 P/Q 的损失超过匹配坐标",
    "e-switch-pq": "共同案例的真实换词且答码跟随新词，相对匹配坐标",
}


def grid(head, body):
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in (head, ["---"] * len(head), *body))


def show(z, rate=False):
    fmt = "+.2%" if rate else "+.4f"
    x, y = z["ci"]
    return f"{z['mean']:{fmt}} [{x:{fmt}}, {y:{fmt}}]"


def at(site):
    l, h, src = site
    return f"L{l}H{h} ({src})"


def intro(p, f):
    g = grid(["G 发现候选", "可恢复方向量", "实际 P/Q 损失"],
             [(at(z["site"]), f"{z['direction']:+.4f}", f"{z['pq_loss']:+.4f}") for z in f["g_search"]])
    e = grid(["E 残差层", "坐标数", "真实选中角色反转词", "所有换 token", "原 State 边际损失"],
             [(z["layer"], z["budget"], z["recognized"], z["any_switch"], f"{z['state_margin_loss']:+.4f}") for z in f["e_search"]])
    gate = p["gate"]
    targets = "、".join(at(z) for z in f["g"])
    random = " ／ ".join("、".join(at(a) for a in b) for b in f["g_random"])
    chosen = f["e"]
    n = len(chosen["coordinates"])
    return rf"""# {TITLE}

本轮 `{C.RUN}` 使用本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`。Phase 2949 的早层前缀阻断降低了 State 优势，并未触发真实换词，也没有通过经 L25H18 传向 P/Q 的全链门。本轮在两个时间点测试。G 是已生成的单 token State 自身的位置，可直接测它读取了哪些提示。E 是模型首次输出 State 之前的最后一个 query，可直接测试真实改选。C 是最终 `Code:` query。

### 材料、发现和冻结

模板交叉事实角色反转、State/Note 输出顺序和 State 至 Code 的两个距离。每家族八个视图，事实反转时码本和候选词保持。试跑 {C.SIZES['pilot']} 家族，发现 {C.SIZES['discover']} 家族，留出 {C.SIZES['confirm']} 家族。原始行为逐 token 验过，容许按实际模型输出对齐行末空白。试跑、发现、留出词表完全分开，发现和留出分别含 {len(p['materials']['discover']['words'])} 与 {len(p['materials']['confirm']['words'])} 个单 token 词。词表按历史相关运行排除已暴露的确切词身份。家族内共享词和码本，家族之间也会复用词。两种事实表达为清单包含/排除和采用提案/保持原项。每格无干预行为门为 90%，按完整家族准入。

G 搜索 L0–24、40 个头，分别用「含与不含／原始与提案及决策」的位置组和两条相互竞争的码本行。只在 G 的一个头将目标来源的全部 Value 置零，键与注意力分母保留。初筛使用该头的真实加权写入，经 $W_O$ 投到残差及其朝带角色符号的 P/Q 读出上的投影。每类从投影与实际写入范数各取四个不同层，16 个单头都执行原模型阻断，再按 L25H18 处实际 G 槽 K/V 可恢复的方向量择优。冻结 {targets}，同层同位置原生剂量匹配头 {random}。发现集剂量和匹配池在 `freeze.json` 中。L24 在 G 的写入最早能改变直接进入 L25 的 $x_{{25}}(G)$。

{g}

在 C 的完好轨迹定义 $d_C=\nabla_o(\ell_Q-\ell_P)$，$o$ 是 L25H18 投影前输出。将原权重逐层反传到该点。设 G 的 key/value 为 $k,v$，C 的活动 query 为 $q$，其余槽的归一化输出为 $u$。由 $s_G=q^\top k/\sqrt{{128}}$ 与原始 softmax 得到 G 权重 $a$。若恢复的原生 K/V 把 $(a,v)$ 变成 $(a^*,v^*)$，该读头在完好轨迹的方向量为 $d_C^\top[a^*(v^*-u)-a(v-u)]$。按任务真实码方向乘 $+1$ 或 $-1$，保留全部 128 个 post-RoPE K 和全部 128 个原生 V 坐标的可加分解。该分解沿用 Phase 2949 的 $\Delta s$ 割线和两种 K/V 更新顺序的均值，可从分片求和重构。后续 P/Q 的真实变化由 BF16 模型完整前向另行核验。原生 K、V、K+V恢复只给 H18 在 C 的一次 G 读取写入原样本值。反向臂把 G 两处阻断后的槽写进完好 H18。完整 V 清零为幅度参照。兄弟读头及其他槽保留本分支的实际缓存。

E 路线在同一模型与同一家族的角色反转样本中读取 post-block $x^{{opp}}_l(E)$。候选原生坐标按发现集平均 $[x^{{opp}}_l(E)-x_l(E)]_j\,[\nabla_{{x_l(E)}}(\ell_{{opp}}-\ell_{{orig}})]_j$ 排序。每层试 256、768、1536 个坐标，用真正的全词表 argmax 核对换词。冻结规则先取能达到 50% 指定换词的最少坐标，再比较真实换词数、边际及层。冻结 L{chosen['layer']} 的 {n} 个坐标。留出坐标只从对应角色反转 donor 的原生 BF16 值取样。匹配随机组在同层同预算按 $|x^{{opp}}-x|$ 近邻选，发现集平均逐坐标位移分别为目标组 {f['e_dose']['chosen']:.4g}、两随机组 {f['e_dose']['random'][0]:.4g} 与 {f['e_dose']['random'][1]:.4g}。同分支清零后原样写回是自恢复，冻结坐标清零和完整 L39 donor 更换为消融及正对照。后者在发现集 {f['positive_control']}/{f['discovered']*8} 真正改选对应词。E 所检层都晚于 L25，固定输入 token 时相同 G 槽 K/V 按因果结构逐元素一致。

{e}

九个主终点在留出前冻结，双侧 {100*gate['confidence']:.3f}% 家族 bootstrap 区间，连续效应要求至少 {gate['logit']:.2f} logit 且下界大于零。换词或换词且跟随新词答码要求超过随机至少 {gate['switch']:.0%}。完整家族至少 {gate['n']}，实际续写共同面板最多 {C.E_CASES} 家族，至少 {gate['natural_n']}。共同面板在已过技术门的家族里按材料顺序选取，视图按家族序号轮转，独立于各干预成败。确认读出不参与筛选或配对。未到 `Code:` 的本次联合事件记未核验，保留原因和真实前缀。
""".replace("\\\\", "\\")


def balance_section(balance):
    title = "### 写入幅度相等的补充对照"
    if balance["state"] != "finished":
        return f"{title}\n\n状态 `{balance['state']}`，等幅度终点暂无有效结果。"
    rows = grid(["来源与对照", "目标原剂量", "随机原剂量", "目标实际剂量", "随机实际剂量", "逐例差 p95"], [
        (k, f"{x['raw_target']:.4g}", f"{x['raw_random']:.4g}", f"{x['target']:.4g}", f"{x['random']:.4g}",
         f"{x['relative_error_p95']:.2%}") for k, x in balance["delivery"].items()])
    endpoints = grid(["补充终点", "效应与 98.75% 区间", "过 0.10 门"], [
        (k, show(x), "是" if x["passed"] else "否") for k, x in balance["endpoints"]["complete"].items()])
    measures = balance["endpoints"]["complete"]
    bc = measures["book-code"]["mean"]
    bd = measures["book-directed"]["mean"]
    fc = measures["fact-code"]["mean"]
    fd = measures["fact-directed"]["mean"]
    return f"""{title}

同层近邻在发现集的事实写入剂量仍存在差距。因此在新词表响应开封前另行登记等幅度对照，保留原主检验。沿用原冻结头，每条视图把目标头和对照头的源阻断写入各自投到原生 $W_O$，取两个 L2 范数的较小者为共同剂量，再让两者等幅写入。报告实际 BF16 写入，单次只动一个来源位置。首 88 个完整行为家族中 {balance['n']} 个过技术门。本补充实验检验同层同来源等剂量下的方向和行为差异。

{rows}

{endpoints}

等幅度下，L24H19 的码本阻断仍比随机头多损失 {bc:.4f} logit，达到预注册门。其经 H18 G 槽的局部方向分量为 {bd:.4f}，低于 0.10 门。L21H7 的事实阻断对比为 {fc:+.4f} logit，方向分量 {fd:+.4f}，相对随机的优势在这次匹配后未确认。联合全剂量的原生救援证明指定 H18 读取参与输出。该结果及等幅度对照共同限定证据范围，仍需检验 L24H19 经其他读头、槽位和非线性计算产生的作用，以及 L21H7 对照在更高剂量下的表现。
"""


def body(p, f, a, recheck, balance):
    nat = a["natural"]
    correct = sum(x["correct"] for x in a["cells"].values())
    total = sum(x["n"] for x in a["cells"].values())
    intro_text = intro(p, f)
    gate = f"""
### 留出行为和技术准入

无干预 {correct}/{total} 条正确自然生成至 P/Q。保持码本改变事实角色时，{nat['attempted']} 对中 {nat['natural_state_and_code_flips']} 对实际 State 与 P/Q 都随事实翻转。有 {len(a['failures'])} 条原模型行为失败。
"""
    if a["state"] != "finished":
        return intro_text + gate + f"\n本轮停止在 `{a['state']}`，因果终点未完成准入。发现产物与失败记录可复核。\n"
    main = grid(["主终点", "效应与调整区间", "删一词的效应范围", "过门"], [
        (LABELS[k], show(z, k in ("e-switch", "e-switch-pq")),
         "共同视图" if z["leave_one_lexeme"] is None else " 至 ".join(f"{x:+.4f}" for x in z["leave_one_lexeme"]),
         "是" if z["passed"] else "否") for k in C.ENDPOINTS for z in [a["primary"][k]]])
    gtable = grid(["G 或 C 操作", "G K 相对 L2 中位数", "G V 相对 L2 中位数", "G 注意力质量中位数"], [
        (k, f"{v['k']:.5g}", f"{v['v']:.5g}", f"{v['alpha']:.5g}") for k, v in a["geometry"].items()])
    gmetrics = grid(["G / C 实际操作", "平均效应及描述性 95% 区间"],
                    [(k, show(z)) for k, z in a["absolute"].items() if not k.startswith("e-")])
    calibration = grid(["C 中央读取对比", "方向预测", "原模型实测", "家族相关"],
        [(k, f"{z['predicted']:+.4f}", f"{z['measured']:+.4f}", "—" if z["pearson_family"] is None else f"{z['pearson_family']:.3f}")
         for k, z in recheck["calibration"].items()])
    egroup = grid(["E 干预", "实际改选 token", "选中候选反词", "原前缀仍可自然到 C", "只在固定原历史上 P/Q 改选", "E 变动 L2 中位数"],
                  [(k, z["state_switched"], z["recognized"], z["natural_prefix"], z["conditional_switch_code"], f"{z['median_e_l2']:.4g}")
                   for k, z in a["arms"].items() if k.startswith("e")])
    gbehavior = grid(["G 或 C 干预", "保持词和后缀自然到 C", "自然 P/Q 翻转", "固定原历史 P/Q 翻转"],
                     [(k, a["arms"][k]["natural_prefix"], a["arms"][k]["natural_switch_code"], a["arms"][k]["conditional_switch_code"])
                      for k in ("sham", "fact", "book", "both", "both-rand1", "both-rand2", "g-kv", "g-reverse", "cache-v-zero")])
    replays = grid(["真实换词臂", "回放条数", "到 Code", "指定换词且 P/Q 跟随新词", "指定换词但 P/Q 保持旧码", "共同面板的联合事件"],
                   [(k, z["tested"], z["reaches_code"], z["recognized_alternate_pq"], z["recognized_original_pq"], z["locked_joint"])
                    for k, z in a["rollout"].items()])
    examples = []
    materials = {r["id"]: r for r in A.load(p["materials"]["confirm"]["file"])}
    for arm in ("e", "e-full", "e-rand1", "e-rand2", "e-zero"):
        for kind, z in a["rollout_examples"][arm].items():
            if z:
                desc = f"最终 `{z['code']}`" if z["stop"] == "code" else f"结束于 `{z['stop']}`"
                original = materials[z["id"]]["values"]["State"]
                examples.append(f"- `{arm}` 的 `{kind}`，`{z['id']}`，原 State `{original}`，实际首词 `{z['first']}`，旧码 `{z['expected']}`，{desc}。")
    for k, v in a["examples"].items():
        if v:
            r = v[0]["row"]
            examples.append(f"- `{k}`，`{r['id']}`，原 State `{r['values']['State']}`，角色反词 `{r['rejected']}`，原码 `{r['code']}`。原生读出及首次分歧见 `audit.json`。")
    for z in a["failures"][:3]:
        r, obs = z["row"], z["observation"]
        actual = obs["rendered"].splitlines()[-1] if obs["rendered"].splitlines() else "空"
        examples.append(f"- 原行为 `{r['id']}`，预期 `{r['values']['State']}` 与 `{r['code']}`，实际末行为 `{actual}`。")
    for r in a["technical_counterexamples"][:6]:
        examples.append(f"- 技术反例 `{r['id']}`，无干预重放 P/Q 差 {r['drift']:.3g}，首处分歧位置 {r['first_mismatch']}，整家族未进入推断。")
    shards = list(a["technical"].values())
    rereads = list(recheck["rechecks"].values())
    lines = [json.loads(x) for x in (C.OUT / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cuda = max(r.get("cuda_peak_bytes", 0) for r in lines) / 2**30
    shared = max(r["shared_gpu_bytes"] or 0 for r in lines) / 2**20
    bridge = "达到" if a["bridge"] else "未达到"
    e0 = a["rollout"]["e"]
    lexical = recheck["complete_state_fields"]["e"]
    check = recheck["discovery_full_prefix"]
    full = recheck["complete_state_fields"]["e-full"]
    means = {k: v["mean"] for k, v in a["absolute"].items()}
    restored = means["g-kv-restore"] / means["both-pq-loss"]
    return intro_text + gate + f"""
{a['eligible']} 家族满足全部原始行为，{a['n']} 家族、{a['n']*8} 个视图通过完整技术门。G 通路{bridge}预注册整链判据。事实、码本、联合、同层对照与恢复都能保持实际同词前缀到 C 的共同家族为 {a['g_common']} 个。各角色和两种规则的分层留出结果见 `audit.json`。

{main}

### State 已生成后对事实和码本的读取

{gtable}

相对量用完好 G 槽归一化，其余位置和 Q 活体变动继续进入 C。G 或 C 干预中 E 的两候选 logits、实际首 token 保持精确，因果时序检查通过。下表的读出比较使用相同形状的原模型前向。

{gmetrics}

{gbehavior}

{calibration}

两个来源的全剂量联合阻断造成 {means['both-pq-loss']:+.4f} logit 的损失。单次 H18 的原生 K/V 写回补回 {means['g-kv-restore']:+.4f}，占联合损失 {restored:.1%}。单写 V 回升 {means['g-v-restore']:+.4f}，单写 K 变化 {means['g-k-restore']:+.4f}。反向移植的实际损失为 {means['g-reverse-pq-loss']:+.4f}。这组条件下，Value 贡献了大部分测到的恢复。两个单来源在主比较中的方向量均未达到 0.10 门，联合效应、单来源效应和按剂量匹配后的效应分别报告。

{balance_section(balance)}

### 真实首次改选与随后的贪心答码

{egroup}

上表的固定历史数值用原 State 的前缀比较。发生实际改选后，模型的真正生成按实测首 token 重放。后续仅接受在当前因果前缀下经全词表 argmax 验证的 token。若草稿首处分歧，接纳该处真正生成的 token 后重新核验。最多每臂回放 {C.E_CASES} 条，每条 {C.E_TOKENS} token 或 {C.E_PASSES} 次。达到 `Code:` 后记录下一 token。新码是原码本对实际改选词的映射，旧码是原码本对提示事实所指词的映射。指定换词臂回放里有 {e0['recognized_pq']} 条实际生成了 P/Q，其中 {e0['recognized_alternate_pq']} 条跟随新词，{e0['recognized_original_pq']} 条保留旧码。

在 L39 的 post-block E 操作中，固定 State token 后，{sum(r['terminal_fixed_code_cases'] for r in rereads)} 个分支的 C 位 P/Q logits 与未干预逐元素相同。这次 E 介入下，后续输出只能通过实际新 token 及其逐步后继传播。整向量与稀疏换入都仍受原提示和码本的真实竞争约束。

对到达 Code 的 {lexical['reached']} 条指定干预续写，另核对完整的 `State:` 行。有 {lexical['exact_alternative']} 条完整词恰为反角色词，其中 {lexical['alternate_word_and_new_code']} 条按新词给码，{lexical['alternate_word_and_old_code']} 条保留原事实的码。整向量正对照的 {full['exact_alternative']} 条完整反词中，{full['alternate_word_and_new_code']} 条按新词给码，{full['alternate_word_and_old_code']} 条保留旧码。共同面板的轮转视图使规则类型、距离与旧码有同现，当前样本无法单独归因给这些因素。字段异常保存在 `native-recheck.json`。

{replays}

原提示换角色自然生成的参考例为 {'、'.join(list(dict.fromkeys(f"{x['states'][0]} / {x['codes'][0]} → {x['states'][1]} / {x['codes'][1]}" for x in nat['examples']))[:4])}。冻结共同面板为 {len(a['evaluation'])} 个独立家族视图，未到达 C 的事件已记为未核验，所有停止理由和逐轮模型输出在 `rollout.json`。

{chr(10).join(examples)}

### 证据边界和下一阶段

冻结的部分坐标在发现集完整前缀复查中有 {check['recognized']}/{check['views']} 条真正选中反角色词。新词表是 {a['arms']['e']['recognized']}/{a['n']*8} 条，两组同剂量随机坐标为 {a['arms']['e-rand1']['recognized']} 与 {a['arms']['e-rand2']['recognized']} 条。发现使用城市词，留出使用姓氏，词表身份与词类一起变化。这一差距检验了较强的迁移要求，无法单独估计换词表与换词类各自的影响。完整前缀发现复查保留较高换词率，因此截短与完整前向的口径差不足以解释这一差距。完整 L39 donor 在留出实际改选 {a['arms']['e-full']['recognized']}/{a['n']*8} 条，给出了真正可切换状态及后续行为的正对照。

G 和 E 分别控制词内历史的生成后写入与生成前状态。G 原生完整槽恢复限定在 H18 一次读取，其余并行依赖及其后果继续随干预改变。头编辑只针对实际事实或码本 Value 项，Q/K 选择的来源、其他位置与兄弟头仍需直接验证。局部方向沿完好轨迹回传，大幅干预的线性预测及微小干预的测量噪声由原模型实际效果约束。E 供体含真实角色反转样本中的计算状态。随机同预算比较检验所选坐标承载切换方向的特异性。供体推入证明指定干预的充分性范围，无法据此断定哪些坐标在完好无干预时必要。须根据零化、匹配对照与原生恢复共同界定。换词重放只有有限模板和长度，病例与比较臂后继若发生格式或终止分歧都如实报告。共享词、单次 BF16 快照和单节点仍限制推广。

外部图谱增加了同码本角色反转和「原事实／实际生成词」这两套可以分别验真的答码参照。内部图谱增加了 G 处提示到 State 的写入、L25 H18 读取该 G 槽以及 E 局部原生坐标的实际行为证据。关联图谱为每条边绑定阻断、恢复、随机控制和真实续写、自身守恒的时间边界。有限参数实现可组合语言的一个可检验条件是，同一内部状态需要支持词面选择及对后续查询有用的角色记忆，且两者可以在不同位置改变。下一大阶段将沿 G 的两个来源把注意力 Q/K 选择与 V/O 写入分开，再把实际续写的原事实、原码、改口词和新码设计成正交条件，测试多槽多步冲突时依赖哪个状态。保留同预算随机与原生坐标基线，再比较跨词类、跨句式、多步骤及无供体干预的可迁移性。当前 RDC 无新统一公式。

### 运行与复算

首轮试跑在 E 自恢复时混用了整段与并排前向产出的 BF16 原生值，逐元素检查失败。发现与留出开封前将自恢复改为同一分支清零再恢复自身，保留 `pilot-first.log`，重跑试跑通过。无干预 P/Q 差最大 {max(r['raw_pq_drift'] for r in shards):.4g}。头编辑的最大归一化误差 {max(r['physical'] for r in shards):.4g}，局部坐标最大闭合误差 {max(r['closure'] for r in shards):.4g}。从保存的原生向量复算 {sum(r['heads'] for r in rereads)} 个分支的方向量，已施加输出最大归一化误差为 {max(r['applied_error'] for r in rereads):.4g}，冻结的 E 排名与三份材料逐项复核通过。RTX 5080 CUDA 分配峰值 {cuda:.2f} GiB，进程共享显存峰值 {shared:.0f} MiB。D 盘每批核对并保留 20 GiB。代码和模型身份、分片、反例、运行日志、预注册及源码哈希记录于 `tests/glm5/result/kv_state_20261116_v1/`。科学证据限于本地样本与单快照技术复算。
"""


def figure(p, f, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (e, g) = plt.subplots(1, 2, figsize=(13.0, 4.7), layout="constrained")
    tones = ("#386d7e", "#976443", "#657a5f", "#88759b")
    for l, color in zip(C.E_LAYERS, tones, strict=True):
        data = [r for r in f["e_search"] if r["layer"] == l]
        e.plot([r["budget"] for r in data], [100*r["recognized"]/(8*f["discovered"]) for r in data], marker="o", lw=1.5, color=color, label=f"L{l}")
    e.plot(len(f["e"]["coordinates"]), 100*a["arms"]["e"]["recognized"]/(8*a["n"]),
           marker="X", markersize=8, color="0.15", linestyle="none", label="Frozen L39, held out")
    e.axhline(50, color=".6", lw=1, ls="--")
    e.set(xlabel="Native residual coordinates transplanted", ylabel="Recognized State switch (%)", title="State emission · discovery and held out", ylim=(-3, 103))
    e.legend(frameon=False)
    panels = ("Fact direction", "Codebook direction", "Joint direction", "Joint P/Q loss", "Native K/V rescue", "Reverse transplant")
    r = [a["primary"][k] for k in C.ENDPOINTS[:6]]
    y = np.asarray([x["mean"] for x in r])
    ci = np.asarray([x["ci"] for x in r])
    g.errorbar(y, np.arange(len(y)), xerr=np.abs(ci-y[:, None]).T, color="#386d7e", fmt="o", capsize=3)
    g.set(yticks=np.arange(len(y)), yticklabels=panels, title=f"Held-out · {a['n']} families", xlabel="Signed logit margin or local projection")
    g.invert_yaxis()
    g.axvline(0, color=".6", lw=1)
    g.axvline(p["gate"]["logit"], color="#976443", lw=1, ls="--")
    for ax in (e, g):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT / "phase-2950-effects.png"), dpi=165)
    plt.close(fig)


def render(stage, p):
    f, a = A.load("freeze.json"), A.load("audit.json")
    recheck = A.load("native-recheck.json")
    balanced = A.load("dose-balance.json")
    assert all(B.sha(k) == v for k, v in (a["provenance"] | recheck["binding"]).items())
    assert recheck["coordinate_rank_exact"]
    assert balanced["prereg"] == B.sha("dose-balance-prereg.json")
    if balanced["state"] == "finished":
        binding = A.load("dose-balance-binding.json")
        assert balanced["binding"] == B.sha("dose-balance-binding.json") and binding["source_hash"] == A.digest(C.ROOT / binding["source"])
        assert all(B.sha(k) == v for k, v in balanced["shards"].items())
    text = body(p, f, a, recheck, balanced)
    P.check_write(C.OUT / "report.md").write_text(text, encoding="utf-8")
    if a["state"] == "finished":
        figure(p, f, a)
    if stage == "preview":
        return
    tip = C.ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
    assert not (C.OUT / "memo-receipt.json").exists()
    assert "\n## Phase 2950:" not in tip.read_text(encoding="utf-8-sig")
    scratch = C.TMP.resolve()
    owned = list(scratch.rglob("*"))
    assert scratch == C.TMP and scratch.parent == (C.ROOT / "tests/glm5_temp").resolve()
    assert all(x.resolve().is_relative_to(scratch) and not x.is_symlink() and not x.is_junction() for x in (scratch, *owned))
    released = sum(x.stat().st_size for x in owned if x.is_file())
    shutil.rmtree(scratch)
    A.save("cleanup.json", dict(path=str(scratch), bytes=released, free=shutil.disk_usage(C.OUT).free))
    sources = {*p["sources"], "tests/glm5/kv_state_report.py", "tests/glm5/kv_state_recheck.py", "tests/glm5/kv_state_balance.py",
               "tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT / "sources.zip"), "w", ZIP_DEFLATED) as z:
        for k in sorted(sources):
            z.write(C.ROOT / k, k)
    with ZipFile(C.OUT / "sources.zip") as z:
        assert all(z.read(k) == (C.ROOT / k).read_bytes() for k in sources)
    cmd = ".venv/Scripts/python.exe -B -u tests/glm5/kv_state_run.py"
    A.save("execution.json", dict(stages=[f"{cmd} {k}" for k in ("pilot", "pilot", "plan", "discover", "confirm", "audit")]
            + [".venv/Scripts/python.exe -B -u tests/glm5/kv_state_balance.py", ".venv/Scripts/python.exe -B -u tests/glm5/kv_state_recheck.py",
               f"{cmd} preview", f"{cmd} close"], experiment=a["provenance"], audit=B.sha("audit.json"),
            recheck=B.sha("native-recheck.json"), companion=B.sha("dose-balance.json")))
    main = append(P.check_write(tip), TITLE, text.split("\n\n", 1)[1])
    assert main["phase"] == 2950
    index = append(P.check_write(C.ROOT / "research/claude/docs/AGI_CLAUDE_MEMO.md"), "14B 自产 State 读事实、码本及实际换词索引",
                   f"详见 GLM5 Phase 2950，运行 `{C.RUN}`，完整报告 `tests/glm5/result/kv_state_20261116_v1/report.md`。")
    A.save("memo-receipt.json", dict(records=[main, index]))
    A.save("manifest.json", dict(files={k.name: A.digest(k) for k in sorted(C.OUT.iterdir()) if k.is_file() and k.name != "manifest.json"},
                                 sources={k: A.digest(C.ROOT / k) for k in sorted(sources)}))
