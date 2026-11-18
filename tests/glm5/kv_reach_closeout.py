"""Render and append the Phase 2944 evidence, and verify preserved artifacts."""

from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from b36_semantic14_memo import append
import kv_reach_run as C


A, B, P = C.A, C.B, C.P
TITLE = "14B 原剂量不可达的注意力失衡、路线重选与 BF16 补偿"
NAMES = {"dynamic": "动态 Q / 原生 BF16", "native": "恢复自然 Q / 原生 BF16", "once": "FP32 增量，头输出舍入",
         "late-round": "重加 Value 舍入，投影后舍入", "late": "FP32 增量，投影后舍入"}
LABELS = {"q4": "4B 固定路线", "q14-fixed": "14B 旧路线", "q14-selected": "14B 重选路线"}


def route(x):
    l, h = x["source"]
    return f"L{l}H{h} → L{x['receiver_layer']}H{x['receiver'][0]}/H{x['receiver'][1]}"


def interval(x):
    if x["ci95"] is None:
        return "—"
    lo, hi = x["ci95"]
    return f"{x['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pairline(x):
    lo, hi = x["ci97_5"]
    values = list(x["leave_one_word_out"].values())
    return (f"新增 {x['gains']}，丢失 {x['losses']}，共 {x['n']} 家族，差 {100*x['effect']:+.1f} 个百分点"
            f"（97.5% 区间 [{100*lo:+.1f}, {100*hi:+.1f}]，配对精确检验 $p={x['pair_exact_p']:.3g}$）。"
            f"按每个词分别剔除所有涉及家族，差值范围 {100*min(values):+.1f} 至 {100*max(values):+.1f} 个百分点。"
            f"预设通过条件{'满足' if x['passes'] else '未满足'}。")


def get_peaks():
    d = defaultdict(lambda: [0, 0, 0])
    for line in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        for idx, name in enumerate(("cuda_peak_bytes", "shared_gpu_bytes", "dedicated_gpu_bytes")):
            d[x["model"]][idx] = max(d[x["model"]][idx], x.get(name) or 0)
    return dict(d)


def render(a, history, freeze):
    C.verify()
    power = A.load("analysis-binding.json")["power"]["exact_two_sided"]
    old = history["pooled"]
    q = old["quantiles"]
    labels = LABELS
    b = A.load("q14-behavior-confirm.json")
    b4 = A.load("q4-behavior-confirm.json")
    check = A.load("q14-calibration.json")
    n = len(b["rows"])
    legacy = ["| 条件格 | 低于原目标下限 H29 / H37 | 2.05 限幅内不可达 H29 / H37 |",
              "|---|---:|---:|"]
    for cell, x in history["cells"].items():
        label = {"context0-order0": "否定 · State 先", "context0-order1": "否定 · Note 先",
                 "context1-order0": "更新 · State 先", "context1-order1": "更新 · Note 先"}[cell]
        legacy.append(f"| {label} | {x['below_target'][0]} / {x['below_target'][1]} | {x['cap_unreachable'][0]} / {x['cap_unreachable'][1]} |")
    roster = ["| 14B 发现路线 | 自然可达预测家族/视图 | 原生通过家族/视图 | 投影后舍入通过家族/视图 |",
              "|---|---:|---:|---:|"]
    for item in freeze["all_discovery"]:
        z, h = item["counts"], item["route"]
        pred = f"{h['predicted_families']}/{h['predicted_views']}" if "predicted_views" in h else "—"
        marker = "（冻结）" if h == freeze["selected"] else ""
        roster.append(f"| {route(h)}{marker} | {pred} | {z['native']['complete']}/{z['native']['views']} | {z['late']['complete']}/{z['late']['views']} |")
    tables = []
    for key, label in labels.items():
        z = a["results"][key]
        pooled = z["dose"]["pooled"]
        rec = a["routes"][key]
        tables.extend([f"#### {label} · {route(rec)}", "",
            "| 条件 | 合格视图 | 完整家族 | 三头目标低于门槛 | State−Note 的 P/Q 供体效应差（95% 区间） |",
            "|---|---:|---:|---:|---:|"])
        for m in NAMES:
            s = z["arms"][m]
            f = " / ".join(str(t[0]) for t in s["flags"])
            tables.append(f"| {NAMES[m]} | {s['views']} | {s['families']} | {f} | {interval(s['margin_effect'])} |")
        r = z["rounding"]
        tabs = [f"{x:.5g}" for x in pooled["quantiles"]["lo"][1]]
        target = [f"{x:.5g}" for x in pooled["quantiles"]["original"][1]]
        ex = z["above_floor_late_failures"]
        ids = "、".join(x["row"]["id"] for x in ex["rows"][:3]) or "无"
        tables.extend(["", f"- 两接收头的自然弱槽注意力中位数为 {tabs[0]} / {tabs[1]}，原目标范数中位数为 {target[0]} / {target[1]}。原目标低剂量视图为 {pooled['below_target'][0]} / {pooled['below_target'][1]}，限幅内不可达为 {pooled['cap_unreachable'][0]} / {pooled['cap_unreachable'][1]}。",
            f"- 相比原生，单次舍入救回 {r['native_to_once_rescued']} 条视图，投影后舍入救回 {r['native_to_late_rescued']} 条。投影后臂重加 Value 舍入后损失 {r['rounding_reinsert_loses']} 条；保留头输出舍入的臂相比投影后臂损失 {r['head_output_cast_loses']} 条。这些是方向性集合差，可能重合。",
            f"- 三头原目标均过下限、最迟舍入臂仍失败的视图 {ex['n']} 条。代表 ID {ids}。完整输入、原生 α、各臂要求写入与实际范数、失败标记及成对误差见审计账本。", ""])
    common = []
    for pool in ("common", "common_exact"):
        if a["families"][pool] == 0:
            continue
        desc = "两模型行为过门" if pool == "common" else "且 14B 的完整贪心前缀与答码逐 token 一致"
        common.extend([f"{a['families'][pool]} 个共同家族，{desc}。每格为完整八视图家族数。", "",
                       "| 路线 | 动态 Q | 原生 | 头输出舍入 | 重加 Value 舍入 | 投影后舍入 |",
                       "|---|---:|---:|---:|---:|---:|"])
        for key, label in labels.items():
            x = a["results"][key][pool]
            common.append(f"| {label} | "+" | ".join(str(x[m]["families"]) for m in NAMES)+" |")
        common.append("")
    earliest = "、".join(x["row"]["id"] for x in history["counterexamples"][:3])
    peaks = get_peaks()
    paths = "tests/glm5/result/kv_reach_20261115_v1/"
    picked = a["results"]["q14-selected"]["arms"]
    anchor = a["results"]["q14-fixed"]["arms"]
    q4 = a["results"]["q4"]["arms"]
    return f"""# {TITLE}

运行 `{C.RUN}`。在 Phase 2943 的 14B 旧路线复算低剂量，再在同一全新留出上比较 14B 旧路线、14B 重选路线和 4B 固定路线。核心指标是所有八种视图都通过全部剂量与写入对照的家族数。14B 冻结路线为 {route(freeze['selected'])}，旧路线为 {route(freeze['fixed'])}。

- **同精度比较 14B 重选与旧路线（两者均在原投影后舍入）**。{pairline(a['primary']['route'])}
- **同路线比较 14B 投影后舍入与恢复自然查询的原生精度**。{pairline(a['primary']['precision'])}

14B 旧路线投影后舍入通过 {anchor['late']['families']}/{a['families']['q14']} 个家族。新路线保留动态 Q 时通过 {picked['dynamic']['families']} 个，恢复自然 Q 后，原生 BF16 与投影后舍入分别通过 {picked['native']['families']} 和 {picked['late']['families']} 个。新路线的原生覆盖提供了舍入补偿增益的实测上限。4B 固定路线在同一面板为 {q4['native']['families']} → {q4['late']['families']} 个，舍入效应随头的实际剂量和数值路径而变。

### 原理与历史失败定位

在最后的 `Code:` 前缀位置，两槽已经出现。对某一头令 $a_S,a_N$ 为原模型对这两槽的 softmax 权重，$u$ 为替换 State 源词后，从同一 GQA 组查得的供体 Value 减自然 State Value。令 $l=\\min(a_S,a_N)$，$h=\\max(a_S,a_N)$，$M=l+h$。原共同目标向量是 $w^*=\\min(h,2l)u$，剂量是 $\\|w^*\\|_2$，实验下限 0.008。保持原寻址时弱槽最多施加 $2.05l\\|u\\|_2$。若这个上界仍低于 0.008，去除 BF16 舍入也无法在原限幅内达到剂量。保持双槽总质量 $M$ 而只优化两槽相对分配时，原目标的代数上界为 $\\tfrac{{2}}{{3}}M\\|u\\|_2$，于 $h=2l$ 取到。它用于定位瓶颈。对同一答码 query，两槽对数权重之差是两个 RoPE 后 QK 得分的差，softmax 的共同分母抵消。两槽共享质量 $M$ 仍受其余历史 token 影响。

在旧 384 条视图上，H29/H37 的低目标计数是 {old['below_target'][0]}/{old['below_target'][1]}，同一寻址的限幅内不可达计数是 {old['cap_unreachable'][0]}/{old['cap_unreachable'][1]}。两槽权重之比中位数 {q['ratio'][1][0]:.2f}/{q['ratio'][1][1]:.2f}，供体范数中位数 {q['d'][1][0]:.2f}/{q['d'][1][1]:.2f}。H37 的 Note 权重中位数 {q['alpha'][1][1][1]:.6f}，State 权重中位数 {q['alpha'][1][1][0]:.6f}。以各头的供体范数中位数替换逐例范数后，低目标计数 {old['floor_after_typical_value'][0]}/{old['floor_after_typical_value'][1]}。对历史失败视图保持两槽总注意力并按原公式计算理论最优分配，H29 有 {old['even_optimizing_slot_ratio_below_target'][0]} 条仍低于门槛，H37 有 {old['even_optimizing_slot_ratio_below_target'][1]} 条。此代数检查偏向解释 H37 的槽间失衡，H29 部分反例同时涉及总注意力偏低。原始反例包括 {earliest}。

{chr(10).join(legacy)}

### 发现、冻结与行为门

同一模板交叉否定/更新、State/Note 生成顺序、词身份交换。12 试跑、32 发现、96 留出家族，按词表隔离，并排除前三阶段已用词。发现先在 14B 扫描 L19–28 的自然两槽 α、供体差和经原 $W_O$ 的写入。源层取 L19–27，接收层固定为下一层，两接收头使用不同 GQA 组。按完整家族三头原剂量自然可达数排序，每个源层保留一条路线，同分再看视图数和源写入强度。对自然排序前四条做真实两臂发现计算，按投影后舍入完整比例、视图比例及自然排序冻结一条。Phase 2943 旧路线作为已指定对照也运行。冻结发生在 4B 留出自然生成和 14B 留出执行之前。发现集筛选依据是剂量及写入合格，后续 P/Q 变化分开记账。

{chr(10).join(roster)}

4B 留出按真实自回归生成，正确的 State、Note、P/Q 格式和终止合计通过 {sum(r['joint'] for r in b4['rows'])}/{n} 条，{a['families']['q4']} 个完整家族。14B 对相同前缀的全词表答码正确 {sum(r['conditional'] for r in b['rows'])}/{n} 条，{a['families']['q14']} 个完整家族。14B 贪心前缀与答码都相同的视图 {sum(r['greedy_prefix_and_code'] for r in b['rows'])}/{n}。不同前缀、与 4B 自然生成不符时的规范回放和失败原因分别记录。两模型八格全部达到 90% 的各自行为门。14B 自订逐层执行器和 HF 在两条发现样本使用相同显式因果掩码时，全词表顶部一致，P/Q 最大差 {check['pq_error']:.4g}，全词表最大差 {check['max_error']:.4g}。

### 新留出剂量、BF16 对照与答码

源写入保持原生。每臂由自然 α 定目标，并按实际槽权重 $a^{{live}}$ 取增益 $\\min(2.05,\\min(h,2l)/\\max(a^{{live}},10^{{-12}}))$。匹配精度臂恢复两个接收头的自然 Q，答码位置的 K 和实际分母仍可变化。第二接收头分别用原生 BF16、仅在头输出舍入、在投影后舍入，以及在投影后舍入并重加 Value 输入舍入。旋转对照保留供体差向量的范数。投影后臂用原 $W_O$ 对应列块乘 FP32 加权增量后叠加正常投影，再转 BF16 给后续残差和 MLP。该臂按投影输入核对剂量与方向，同时保留独立的实际 P/Q 输出。家族统计要求每种精度设置内，两个槽各自的供体和旋转四臂，其三头剂量、方向、强度及两槽配对误差在八个视图全部合格。连续 P/Q 内容差是 State 槽与 Note 槽的“供体减旋转”答码边际之差，按每种精度的通过家族汇总，描述性区间随入选范围变化。

{chr(10).join(tables)}
### 同提示、同答码前缀的跨模型对照

{chr(10).join(common)}

### 证据边界与下一大阶段

待检验的科学问题是模型自然计算会选择哪条路线、在哪个边界使用对应内容。这里获得的是指定低剂量和范围内原生网络响应的可达性证据。人工注入、恢复 Q、GQA 组以及两种有限模板共同限定其适用范围。旋转保范数；它仍可能与自然任务方向的协方差不同。跨模型比较用模型内各自的头，覆盖共同词汇和前缀，对固定提示结构以外的文本、模型独立全生成、长距离竞争、非目标任务副作用和独立节点均需额外数据。发现路由只在 L19–28 及相邻层预算内寻优。由于家族复用词，区间和精确检验按家族交换性计算，同时附删一词敏感性。预注册辅助功效调用误把损失概率当作总不一致率，留出打开前以独立配对下的精确混合重新计算。示例新增 25%、丢失 5%、96 家族、双侧 .025 时，精确检验功效 {power:.3f}，实际有效样本受行为门与共享词影响。全部原剂量阈值和主终点继续沿用冻结值。

外部图谱现在可关联槽身份、顺序、否定/更新条件与历史反例。内部图谱可记录每头两槽相对分数、合计权重、供体方向与实际精度误差。两图谱之间可先预测给定条件下可达的写入，再单独核对写入能预测多少答码边际。有限参数在组合中复用的第一性问题是同一关系换槽、换顺序时如何同时保留可寻址性和下游可读性。下一大阶段可围绕这两个条件一并验收：冻结多竞争槽和长距离、增加新句式，分别检验 QK 相对分数与槽总质量的留出可预测性；同预算比较自然路由和激活基线，再做原生阻断、恢复与副作用追踪，把 $W_O$ 后残差的实测增量和未来 token 分布纳入同一检验。RDC 暂只吸收这条局部可达性界限。

### 工程与可复核路径

4B 和 14B 在同一 RTX 5080 串行执行，使用本地原始 BF16 快照。14B 原始权重按层读取，暂存激活在 CPU，CUDA 分配峰值 4B/14B 为 {peaks['q4'][0]/2**30:.2f}/{peaks['q14'][0]/2**30:.2f} GiB，进程共享显存峰值 {peaks['q4'][1]/2**20:.0f}/{peaks['q14'][1]/2**20:.0f} MiB。复核了新旧 14B 路线 {n} 条视图的样本 ID 和无干预 P/Q logits，逐元素一致。方案、分析脚本和路线冻结时间早于留出行为开封。关键结果均在 `{paths}`。图 `reachability.png`、`q14-discover-all-head-support.npz`、`q14-freeze.json`、`historical-dose-factorization.json`、`audit.json` 和效应分片保存发现排序、边界、配对和反例。详见 `execution-receipt.json`、`memo-receipt.json` 与 `manifest.json`。研究记录按原字节前缀校验后追加。证据状态为本地实验与技术复算。
"""


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = A.load("plan.json")
    z = np.load(C.OUT/"q14-discover-all-head-support.npz")
    try:
        natural = z["alpha"][:, :, z["eligible"]]
        target = np.minimum(natural.max(-1), 2*natural.min(-1))*z["dnorm"][:, :, z["eligible"]]
    finally:
        z.close()
    coverage = (target >= p["dose"]["limits"]["norm_floor"]).reshape(*target.shape[:2], -1, 8).all(-1).mean(-1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    c = left.imshow(coverage, vmin=0, vmax=1, cmap="viridis", aspect="auto", interpolation="nearest", extent=(-.5, coverage.shape[1]-.5, 28.5, 18.5))
    left.set(yticks=range(19, 29), ylabel="Qwen3-14B transformer block", xlabel="Native query-head index",
             title="Discovery · natural dose across all eight views")
    fig.colorbar(c, ax=left, label="Fraction of complete families")
    x = np.arange(4)
    for j, (key, label, color) in enumerate((("q4", "4B fixed", "#457b9d"), ("q14-fixed", "14B old", "#a76f52"),
                                            ("q14-selected", "14B selected", "#34796e"))):
        data = a["results"][key]["common"] or a["results"][key]["arms"]
        counts = [data[m]["families"] for m in ("native", "once", "late-round", "late")]
        bars = right.bar(x+(j-1)*.24, counts, width=.23, label=label, color=color)
        right.bar_label(bars, padding=2, fontsize=8)
    right.set(xticks=x, xticklabels=["Native", "Head output\ncast", "Reinsert V\nrounding", "Projection\noutput cast"],
              ylabel="Complete eight-view families", ylim=(0, max(a["families"]["common"], 12)*1.25),
              title=f"Confirmation · {a['families']['common']} common families")
    right.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    right.grid(axis="y", alpha=.15)
    right.spines[["top", "right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"reachability.png"), dpi=180)
    plt.close(fig)


def close(preview):
    a = A.load("audit.json")
    history = A.load("historical-dose-factorization.json")
    freeze = A.load("q14-freeze.json")
    report = render(a, history, freeze)
    P.check_write(C.OUT/"report.md").write_text(report, encoding="utf-8")
    plot(a)
    if preview:
        return
    assert not (C.OUT/"memo-receipt.json").exists()
    path = C.TMP.resolve()
    assert path == C.TMP and path.parent == (C.ROOT/"tests/glm5_temp").resolve()
    paths = list(path.rglob("*"))
    assert all(x.resolve().is_relative_to(path) and not x.is_symlink() and not x.is_junction() for x in [path, *paths])
    n = sum(x.stat().st_size for x in paths if x.is_file())
    shutil.rmtree(path)
    A.save("cleanup.json", dict(path=str(path), bytes_removed=n, disk_free_bytes=shutil.disk_usage(C.OUT).free))
    A.save("execution-receipt.json", dict(commands=[f".venv/Scripts/python.exe -B -u tests/glm5/{x}" for x in (
        "kv_reach_run.py plan", "kv_reach_run.py q4-pilot", "kv_reach_run.py q4-discover", "kv_reach_q14.py discover",
        "kv_reach_audit.py historical", "kv_reach_audit.py freeze-analysis", "kv_reach_run.py q4-confirm",
        "kv_reach_q14.py confirm", "kv_reach_audit.py analyze", "kv_reach_closeout.py")],
        status="completed", peak_bytes=get_peaks(), plan=B.sha("plan.json"), audit=B.sha("audit.json"),
        early_plan_attempt="Stopped at pilot vocabulary overlap before any frozen material was written. Historical vocabulary exclusions were moved before material generation."))
    src = set(A.load("plan.json")["sources"])
    src.update(p.relative_to(C.ROOT).as_posix() for p in Path(__file__).parent.glob("kv_reach_*.py"))
    src.add("tests/glm5/b36_semantic14_memo.py")
    with ZipFile(P.check_write(C.OUT/"sources.zip"), "w", ZIP_DEFLATED) as f:
        for name in sorted(src):
            f.write(C.ROOT/name, name)
    with ZipFile(C.OUT/"sources.zip") as f:
        assert all(f.read(name) == (C.ROOT/name).read_bytes() for name in src)
    glm = C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
    claude = C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"
    first = append(P.check_write(glm), TITLE, report.split("\n\n", 1)[1])
    second = append(P.check_write(claude), "14B 原剂量可达性与 BF16 补偿的冻结对照索引",
        f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`。原三头公式和 2.05 限幅下，复算旧 14B 路线低剂量，"
        "在新留出比较模型内重选、固定旧路线和 4B 同材料参考。产物 `tests/glm5/result/kv_reach_20261115_v1/report.md`。")
    A.save("memo-receipt.json", dict(records=[first, second]))
    A.save("manifest.json", dict(files={p.name:A.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
                                  sources={k:A.digest(C.ROOT/k) for k in sorted(src)}))


if __name__ == "__main__":
    C.setup()
    close(sys.argv[1:] == ["preview"])
