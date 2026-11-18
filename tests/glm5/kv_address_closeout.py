"""Write the study ledger, render a static figure, and append both research records."""

import json
from pathlib import Path
import shutil
import sys

import kv_address_run as C
from b36_semantic14_memo import append


HEADS=("L21H19","L21H25","L20H5")
NAMES={"context":"语境（更新−否定）","order":"顺序（Note先−State先）","identity":"交叉词身份"}
TITLE="State 与 Note 的条件化 Q/K 寻址及三头写入匹配失败"


def interval(stats):
    return "—" if not stats.get("ci95") else f"{stats['mean']:+.3f} [{stats['ci95'][0]:+.3f}, {stats['ci95'][1]:+.3f}]"


def make(a,fixed):
    p=C.load("study-plan.json")
    fit=C.load("fit.json")
    choices=C.load("case-forecasts.json")
    relay=C.load("relay-gain-audit.json")
    past=C.load("phase2940-failure-by-head.json")["doses"]
    valid=[]
    for stage in ("pilot","discover","confirm"):
        b=C.load(f"{stage}-behavior.json" if stage=="pilot" else f"behavior-{stage}.json")
        valid.append(f"{stage} {sum(x['correct'] for x in b['cells'])}/{sum(x['n'] for x in b['cells'])}，八格 "+
                     "、".join(f"{x['correct']}/{x['n']}" for x in b["cells"]))
    natural=[]
    for r in a["raw_natural"]:
        natural.append("| "+("否定" if r["context"]==0 else "更新")+" | "+
                       ("State 先" if r["order"]==0 else "Note 先")+" | "+
                       " | ".join(f"{g:+.3f}（{rate:.1%}）" for g,rate in zip(r["mean_gap"],r["state_rate"],strict=True))+" |")
    factor=[]
    for label,per in a["factors"].items():
        for h,row in zip(HEADS,per,strict=True):
            factor.append(f"| {NAMES[label]} | {h} | {interval(row['signed'])} | {interval(row['absolute'])} |")
    preds=a["natural_predictions"]
    prediction=[]
    for k,label in (("order","仅顺序"),("context","仅语境"),("factorial","按控制条件的格均值"),
                    ("prototype","Q/K 原型")):
        prediction.append(f"| {label} | "+" | ".join(f"{y:.1%} / {r:.3f}" for y,r in
                           zip(preds[k]["sign"],preds[k]["rmse"],strict=True))+" |")
    patches=[]
    for k,v in a["interventions"].items():
        ix,mode,*f=k.split("-")
        key=HEADS[int(ix)]+" "+({"q":"仅 Q","k":"仅 K","rope":"仅换 RoPE 相位"}[mode])
        if f:
            key+=" · "+NAMES[f[0]]
        d="—" if v["donor_direction"] is None else f"{v['donor_direction']:.1%} / {v['donor_relative_error']:.3f}"
        patches.append(f"| {key} | {v['frozen_prediction_rmse']:.5f} | {v['gap_change_abs']:.3f} | "
                       f"{v['gap_flip']} | {v['code_margin_change_abs']:.3f} | {d} |")
    w=a["weighted"]
    o,b=w["arms"]["original"],w["arms"]["balanced"]
    before,after=w["h25_natural_alpha_median"],w["h25_balanced_alpha_median"]
    tmax=max(v["frozen_prediction_rmse"] for v in a["interventions"].values())
    pitch=a["forecast_counterexamples"][0]
    magnitude=[]
    for k in ("original","balanced"):
        row=w["arms"][k]
        magnitude.append(f"{k} 各头未通过的计数为 "+str(row["failures"]))
    first_fail=next(iter(fixed["examples"]["failed"]),None)
    first_fix=next(iter(fixed["examples"]["rescued"]),None)
    first_loss=next(iter(fixed["examples"]["lost"]),None)
    natural_a=a["raw_natural"]
    korder=a["interventions"]["0-k-order"]
    qorder=a["interventions"]["0-q-order"]
    rope=a["interventions"]["0-rope"]
    perf=[json.loads(l) for l in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peak=max(x.get("cuda_peak_bytes",0) for x in perf)/2**30
    share=max(x.get("shared_gpu_bytes") or 0 for x in perf)/2**20
    return rf"""# {TITLE}

## 问题、范围和冻结

运行 `{p['run']}`，主模型 Qwen3-4B 原始 BF16，快照 `{p['model']}`。L21H19、L21H25、L20H5 的层和查询头均从零起。目标是在答码 query 已看到 State 与 Note 之后，检查为什么偏好随场景变化，并在同一批留出材料验证只改 Q 或 K 的效果。

72 个发现家族使用日常用品词，192 个留出家族使用地理词，词表整体不相交。每家族 8 个视图，分别交叉否定/更新指令、State 先/Note 先以及两词角色互换。同一家族固定备选词、码本答案以及码本顺序、两句次序和保留/更新类型。两词角色互换保持词对不变，并保持 State 与 Note 可能使用的词来自同一池。前缀给定第一个字段标签，值和后续字段由模型自然生成。交换字段次序同时改变各槽可见的已生成历史。只交换键的 RoPE 相位是单独的位置干预。

自然行为联合门要求每格至少 90% 严格正确，含 State、Note、P/Q、终止及单 token 槽位。{chr(10).join(valid)}。确认按完整八视图家族分析，共 {a['families']} 家族、{a['n_eligible']} 视图，未纳入的视图保留全部实际生成账本。

于 {fit['frozen']} 在读取任何确认响应以前冻结发现集 Q/K 原型和三个因素基线。于 {choices['time']} 在任何留出干预以前冻结逐案例 Q/K 干预预测、供体索引及阈值。确认自然原型预测用语境、输出顺序、码本及格式相对距离，输入不含目标样本的 Q/K。干预预测读取同一确认家族自然响应中的已声明供体，回答局部计算是否可由实际向量重建。二者的输入不同，分别记结果。

功效预注册用项目 McNemar 工具按 8.5 个百分点和 .10 配对不一致率规划，预设 α=.025，需 {p['power']['n_required']} 家族，192 家族规划功效 {p['power']['achieved_power']:.3f}。实际主效应是连续 Q/K 分差，家族 bootstrap 给描述性区间。共享词、模板和同一快照限制外推，不能把所有视图当独立。

## 寻址分差与定点干预

令 $g_h=\log(\alpha_{{h,S}}/\alpha_{{h,N}})$。相同 softmax 的分母抵消，因此：

$$g_h = \frac{{q_h^\top(k_{{S,g(h)}}-k_{{N,g(h)}})}}{{\sqrt{{128}}}},\qquad g(h)=\lfloor h/4\rfloor.$$

$g_h>0$ 表示此头更关注 State。原生捕获 RoPE 后的 $q,k$、两槽的 $V$、完整注意力 softmax 的两槽权重和 P/Q logits。自然回放两条未编辑路径的全词表输出逐值相等，再核对编辑批次的原始 Q/K 和 P/Q。冻结原型先把发现向量逆转到零位置，相同语境、顺序和代码条件内取均值，再按目标相对位置复相位。该预测与只按顺序、只按语境及条件格均值比较。

固定同一答码 query 和其自然历史，在同家族仅翻转一个因素作为供体。Q 臂把供体查询旋转至当前 query 位置，K 臂把两条供体键旋转至各自原有槽位，V 和未选头保持原生。单纯 RoPE 臂保留两个槽的键内容并交换对方的相位，以测试相对位置编码本身。原生 SDPA 重算及原生下游继续前向，记录已修改头实际写出差、读出偏好翻转、P/Q 分差。预测分差是局部代数的显式核对，因果范围由实际改写后出现的头写入及 P/Q 变化确定。

| 语境 | 输出顺序 | L21H19 平均 g（State占优率） | L21H25 | L20H5 |
|---|---|---:|---:|---:|
{chr(10).join(natural)}

L21H19 的反转集中在否定且 State 先输出。此格平均 $g={natural_a[0]['mean_gap'][0]:+.3f}$，有 {1-natural_a[0]['state_rate'][0]:.1%} 的视图关注 Note 更多。否定语境里把 State 放到第二位，平均 $g$ 变为 {natural_a[1]['mean_gap'][0]:+.3f}。同样是 State 先，在更新语境平均 $g$ 为 {natural_a[2]['mean_gap'][0]:+.3f}。词角色互换在 L21H19 上的绝对变化为 {a['factors']['identity'][0]['absolute']['mean']:.3f}，有符号总体均值为 {a['factors']['identity'][0]['signed']['mean']:+.3f}，说明具体词对有影响，作用方向随配对而异。L21H25 在全部 {a['n_eligible']} 个确认视图中均偏 State。

| 因素 | 查询头 | 家族平均有符号变化与 95% 区间 | 家族平均绝对变化与 95% 区间 |
|---|---|---:|---:|
{chr(10).join(factor)}

| 冻结预测 | L21H19 符号一致 / RMSE | L21H25 | L20H5 |
|---|---:|---:|---:|
{chr(10).join(prediction)}

原型准入条件为符号一致率至少 .70 且相对顺序基线改善至少 .05。三头的逐项判定为 {preds['prototype_pass']}。已有较简单格均值可以达到或超过原型时须保留。代表性失败 `{pitch['id']}`，三个头实际 $g$ 为 {pitch['observed']}，冻结原型为 {pitch['pred']}。定位到当天的原生张量，不将其升级为某类词的普遍功能。

| 单次干预 | 预测 g RMSE | g 平均绝对变化 | 偏好翻转 | P/Q 分差平均绝对变化 | 朝供体方向比例 / 相对误差 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(patches)}

L21H19 的字段反序对照中，位置对齐后只换 K 使分差平均变动 {korder['gap_change_abs']:.3f}，{korder['donor_direction']:.1%} 朝供体方向。只换 Q 的平均变动为 {qorder['gap_change_abs']:.3f}，只交换 RoPE 相位为 {rope['gap_change_abs']:.3f}。位置对齐的 K 仍保留生成顺序造成的上下文差异，结果支持先前输出历史影响缓存键。语境对照的 Q 与 K 两种替换均分别产生作用，后续可以在上游状态中继续定位。

21 个预定干预臂最大 RMSE 为 {tmax:.5f}，阈值 .03。朝供体方向只度量 $g$ 的变化方向。相对误差是 $mathrm{{RMS}}[(g_{{编辑}}-g_{{自然}})-(g_{{供体}}-g_{{自然}})]/mathrm{{RMS}}[g_{{供体}}-g_{{自然}}]$，小于 1 表示该次替换相对零改动更接近跨条件差异，未视作中介比例。

## 三头匹配失败与修复

逐分片复算 Phase 2940，低剂量 {past['low']['failed']} 个失败视图全部且仅涉及 L21H25，共 59 个解析写入误差和 2 个方向误差（有重叠）。高剂量 {past['high']['failed']} 个失败，L21H19 缩放触顶 36 次、L21H25 64 次，存在重叠。换回同一行的自然 α 去算，三个头的触顶计数为 {past['high']['cap_from_sham_count']}。高剂量原定两倍较小权重为上界，触顶来自 L20 先改动后 L21 的**实际权重比自然锚更小**。低剂量误差则集中在 L21H25 的 BF16 Value 舍入后有效写入，不能用同一个触顶理由解释。

新留出上的修复只编辑 L21H25 的 State/Note 两条 K 沿 $q$ 的投影，把两条注意力 logit 移到同一均值，目标保持两槽指数质量之和。之后在原生 SDPA 重新做三头 Value 供体与旋转对照，并核对实际头写入。修正前后该头的自然 State/Note 权重中位数为 {before[0]:.4f}/{before[1]:.4f} 与 {after[0]:.4f}/{after[1]:.4f}，单批最大余额误差 {w['balance_alpha_max_error']:.5f}。K-only 对原有 P/Q 分差的平均绝对副作用为 {w['balance_baseline_margin_abs']:.3f} logits。对两槽的 Value 内容比较采用同一 K 条件。

| 三头高剂量方案 | 合格视图 | 全家族通过 | 匹配的 State−Note 内容差 |
|---|---:|---:|---:|
| 原始寻址 | {o['views']} | {o['families']} | {interval(o['source'])} |
| 平衡 H25 后重算共同剂量 | {b['views']} | {b['families']} | {interval(b['source'])} |
| 平衡 H25，固定原目标剂量 | {fixed['views'].get('same',0)+fixed['views'].get('rescued',0)} | {fixed['repaired_families']} | {interval(fixed['source'])} |

重算目标剂量时，救回 {w['rescued_views']} 条、损失 {w['lost_views']} 条。救回 {w['rescued_families']} 个完整家族、损失 {w['lost_families']} 个。固定原目标剂量时，救回 {fixed['views'].get('rescued',0)} 条、损失 {fixed['views'].get('lost',0)} 条，完整家族分别为 {fixed['rescued_families']} 和 {fixed['lost_families']}。固定目标用每头相同的原始 αΔV 目标，不用降低剂量改善覆盖。

{'。'.join(magnitude)}。固定目标残留原因按头为 {fixed['fail_heads']}。固定目标代表救回 {first_fix}，代表失败 {first_fail}，丢失的视图为 {first_loss}。更多逐视图反例见 audit.json、fixed-dose-audit.json 和哈希分片。

用新留出原生权重重算原目标，对 L21H19、L21H25、L20H5 的触顶数为 {relay['head_overflow_anchored']}。改用三头干预时实时权重，触顶变为 {relay['head_overflow_live']}，逐行复现原干预标记。该时间点前唯一实际改动是 L20H5 写入。这将缺口定位为 L20H5 写入后 L21 实时权重对冻结剂量的敏感性，L20 与 L21 之间的上游参数传递仍待定位。按语境和顺序的计数、增益最大的反例见 relay-gain-audit.json。

## 图谱和边界

这次能区分三种变量对寻址的作用。词对角色互换改变内容，字段反序改变位置和此前生成历史。RoPE 交换只改变位置相位。否定与更新是两套完整指令环境，结论限定在本组题面。答码 P/Q 的边界可复算，Q/K 干预的含义限定在该 query 指定的头及其原生后续计算。注意力偏好翻转、最终答码翻转和真实语义推理分别计量。局部投影可以做反事实解释，注意力何以在训练中形成、原始 Q 与 K 如何由上游状态生成、以及真实长文中的地址竞争均待验证。

外部图谱新增词对、角色、输出顺序、语境的交叉条件及失败样本。内部图谱把 Q/K、RoPE 及同层的地址权重定位到三个查询头，也保留了下游 P/Q 和三头写入的复核入口。关联图增加可计算的局部寻址候选。RDC 主体公式维持原状。下一阶段应冻结自然化题面的地址预测，用上游原生状态或同预算稀疏特征直接预测 Q/K 缺口，评估留出句式、长距离和不同 Note 语义。再追踪 L20H5 的写入通过哪些计算改变 L21H19、L21H25 的查询，分开检验源位置质量、内容和生成时刻。

## 可复核产物

冻结设计 SHA256 `{a['design']}`，逐案例冻结预测 `{a['forecast_sha']}`。分词、自然行为、预测、指定头读出和失败都有样本 ID。保留两个套组首批四视图的生成步 × 全层 × 原生坐标 BF16，其余保存选中头的张量与实际分片。RTX 5080 上 CUDA 分配峰值 {peak:.2f} GiB，DXGI 共享显存采样峰值 {share:.1f} MiB。所有 GPU 测量来自同一物理节点，旧 Phase 2940 工单的独立 GPU 复算未在本轮发生。全程仅运行用户明确指定的 Qwen3-4B。

命令与恢复入口见 execution-receipt.json。完整产物位于 `tests/glm5/result/kv_address_20261115_v1`。
"""


def chart(a,fixed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,(left,right)=plt.subplots(1,2,figsize=(11.8,4.6),layout="constrained",
                                  gridspec_kw={"width_ratios":[1.3,1]})
    import numpy as np
    x=np.arange(4)
    for i,(head,col) in enumerate(zip(HEADS,("#316f94","#ac654e","#2f8876"),strict=True)):
        left.plot(x,[r["mean_gap"][i] for r in a["raw_natural"]],"o-",label=head,color=col)
    left.axhline(0,color="#6c6e77",ls="--",lw=1)
    left.set_xticks(x,["Negation\nState first","Negation\nNote first","Update\nState first","Update\nNote first"])
    left.set(ylabel="Mean log attention ratio  State / Note",title="The addressing preference changes by condition")
    left.legend(frameon=False)
    w=a["weighted"]["arms"]
    ticks=["Native\naddressing","Balance H25\nnew target","Balance H25\nsame target"]
    v=[w["original"]["families"],w["balanced"]["families"],fixed["repaired_families"]]
    bars=right.bar(ticks,v,color=["#8795a4","#3d7d9c","#348575"],width=.58)
    right.bar_label(bars,padding=2)
    right.axhline(153,color="#7b5353",ls="--",lw=1,label="153-family gate")
    right.set(ylabel="Complete families that met dose gates",ylim=(0,201),title="Three heads · matched high dose")
    right.legend(frameon=False,loc="upper left")
    for ax in (left,right):
        for edge in ("right","top"):
            ax.spines[edge].set_visible(False)
        ax.grid(axis="y",alpha=.17)
    fig.savefig(C.R.Base.Previous.check_write(C.OUT/"address-results.png"),dpi=170)
    plt.close(fig)


def close(preview=False):
    a=C.load("audit.json")
    f=C.load("fixed-dose-audit.json")
    report=make(a,f)
    C.R.Base.Previous.check_write(C.OUT/"report.md").write_text(report,encoding="utf-8")
    chart(a,f)
    if preview:
        return
    assert not (C.OUT/"memo-receipt.json").exists()
    root=C.Base.ROOT
    temp=C.TEMP
    assert temp.resolve()==temp and temp.parent==(root/"tests/glm5_temp").resolve() and not temp.is_junction() and not temp.is_symlink()
    disposable=list(temp.rglob("*"))
    assert all(x.resolve().is_relative_to(temp) and not x.is_junction() and not x.is_symlink() for x in disposable)
    n=sum(s.stat().st_size for s in disposable if s.is_file())
    shutil.rmtree(temp)
    C.save("cleanup.json",dict(path=str(temp),bytes=n,free=shutil.disk_usage(C.OUT).free))
    C.save("execution-receipt.json",dict(
        commands=[f"python -B -u tests/glm5/kv_address_run.py {s}" for s in ("pilot","plan","discover","fit","confirm")]+[
            "python -B -u tests/glm5/kv_address_history.py",
            "python -B -u tests/glm5/kv_address_intervene.py freeze",
            "python -B -u tests/glm5/kv_address_intervene.py run",
            "python -B -u tests/glm5/kv_address_fixed_dose.py freeze",
            "python -B -u tests/glm5/kv_address_audit.py",
            "python -B -u tests/glm5/kv_address_fixed_dose.py run",
            "python -B -u tests/glm5/kv_address_fixed_dose.py audit",
            "python -B -u tests/glm5/kv_address_relay_audit.py",
            "python -B -u tests/glm5/kv_address_closeout.py"],
        qualified="Methods/material/integrity and GPU audited. State-first and Note-first are generated after a fixed leading label. No official-center publication or second physical CUDA host.",
        provenance=a["provenance"],main_shards=len(C.load("interventions.json")["shards"]),
        fixed_target_shards=len(C.load("fixed-dose-index.json")["shards"])))
    bodies="\n".join("#"+l if l.startswith("## ") else l for l in report.split("\n\n",1)[1].splitlines())
    first=append(root/"research/glm5/docs/AGI_GLM5_MEMO.md",TITLE,bodies)
    second=append(root/"research/claude/docs/AGI_CLAUDE_MEMO.md","State 与 Note 原生寻址及三头匹配失败索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 {C.RUN}。材料和 Q/K 预测在留出前冻结，"
                  "可查冻结预测、实际查询/键干预、RoPE 相位对照、历史失败逐头定位和固定剂量修复。"
                  "完整产物在 tests/glm5/result/kv_address_20261115_v1/。")
    C.save("memo-receipt.json",dict(records=[first,second]))
    paths=sorted(Path(__file__).parent.glob("kv_address_*.py"))
    from zipfile import ZIP_DEFLATED, ZipFile
    with ZipFile(C.R.Base.Previous.check_write(C.OUT/"source.zip"),"w",ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    with ZipFile(C.OUT/"source.zip") as z:
        assert all(z.read(p.name)==p.read_bytes() for p in paths)
    C.save("manifest.json",dict(files={p.name:C.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},
                                sources={p.name:C.digest(p) for p in paths}))


if __name__=="__main__":
    C.configure()
    close(preview=sys.argv[1:]==["preview"])
