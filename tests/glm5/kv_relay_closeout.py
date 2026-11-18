"""Report the fixed-dose L20→L21 relay, with a manifest and append-only records."""

import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import kv_relay_run as C
import kv_relay_study as S
from b36_semantic14_memo import append


A=C.A
TITLE="L20H5 Value 写入经残差与 MLP 改变 L21 寻址的冻结验证"
DOSE={
    "trio":"原三头",
    "trio-mlp":"阻断 L20 MLP 变化",
    "trio-q19":"恢复 H19 原查询",
    "trio-q":"恢复 H19 与 H25 原查询",
    "trio-qk":"恢复两头原查询及当前键",
}
PRED={
    "full":"完整局部原生计算",
    "direct":"残差直达，保留自然 MLP",
    "mlp":"仅插入受扰 MLP 输出",
    "sparse":"直达 + 冻结 192 单元",
    "random":"直达 + 同数随机单元",
    "cell":"发现格均值",
    "q_only":"只替换 Q 的计算分解",
    "k_only":"只替换当前 K 的计算分解",
}
BLOCKS={
    "mlp":"全 MLP",
    "units":"冻结 192 单元",
    "null-units":"192 随机单元",
    "q":"两头 Q",
    "k":"两头当前 K",
    "qk":"两头 Q + 当前 K",
}


def band(r):
    return "—" if not r["ci95"] else f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def render(a):
    p=A.load("plan.json")
    f=A.load("freeze.json")
    predictions=A.load("forecast.json")
    blocks=a["interventions"]
    d=a["doses"]
    natural=[]
    for stage,table in a["behavior"].items():
        natural.append(f"- {stage}，{sum(x['correct'] for x in table)}/{sum(x['n'] for x in table)}，逐格 "
                       +"、".join(f"{x['correct']}/{x['n']}" for x in table))
    forecast_table=[]
    for k,title in PRED.items():
        x=a["predictors"][k]
        u,v=x["logalpha_rmse"]
        q,w=x["cap_balanced_accuracy"]
        fn=x["cap_fn"]
        forecast_table.append(f"| {title} | {u:.3f} / {v:.3f} | {q:.1%} / {w:.1%} | {fn[0]} / {fn[1]} |")
    bridge=[]
    for k,title in BLOCKS.items():
        x=blocks[k+"-block"]
        y=blocks[k+"-restore"]
        duo=lambda xs:" / ".join(f"{z:.1%}" for z in xs)
        bridge.append(f"| {title} | {duo(x['removal_ratio'])} | {duo(y['restoration_ratio'])} | "
                      f"{x['alternative_code_margin_mean_abs']:.3f} / {y['alternative_code_margin_mean_abs']:.3f} |")
    rescue=[]
    for k,title in DOSE.items():
        x=d[k]
        rescue.append(f"| {title} | {x['views']} | {x['rescued_views']} / {x['lost_views']} | {x['families']} | "
                      f"{x['h19_caps']} / {x['h25_caps']} | {band(x['source_difference'])} | "
                      f"{'达到' if x['family_coverage'] else '未达'} |")
    h19=a["overflow_components"]["0"]
    switch=a["mlp_block_cap_switches"]
    quote=a["counterexamples"][0]
    best=d["trio-qk"]
    label=("幅度小","触顶","解析误差","原生头输出误差","写入塌缩","方向偏差")
    remaining=[]
    for h,nums in enumerate(best["by_head_reason"]):
        msg="、".join(f"{l} {n}" for l,n in zip(label,nums,strict=True) if n) or "无单槽失败"
        remaining.append(f"{('L21H19','L21H25','L20H5')[h]}，{msg}，两槽配对失败 {best['pair_failure_per_head'][h]}")
    logs=[json.loads(x) for x in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peak=max(x.get("cuda_peak_bytes",0) for x in logs)/2**30
    shared=max(x.get("shared_gpu_bytes") or 0 for x in logs)/2**20
    return rf"""# {TITLE}

运行 {p['run']}。在答码 query 对 L20H5 的 State 或 Note Value 使用原始高剂量供体与旋转对照，跟踪 L21H19、L21H25 的原生 Q/K、实时注意力和三头剂量匹配。

主要结果是实时注意力的改变几乎全部可由两头的新 Q 重现。L21H19 的 {d['trio']['h19_caps']} 个触顶视图在恢复其自然 Q 后全部消失。恢复两头自然 Q 后，合格完整家族从 {d['trio']['families']} 增至 {d['trio-q']['families']}，覆盖门是 153。冻结 192 单元的预测和原生恢复均优于等量随机，但全量阻断 L20 MLP 使剂量通过数下降。记录中保留这两种方向的反例。

### 材料、时序与证据范围

Qwen3-4B 原始 BF16，快照 {p['snapshot']}。发现 72 个家族、确认 192 个家族，词池不相交。确认日用品词全部排除 Phase 2941 的发现和确认词表。每家族八视图交叉否定/更新、State/Note 顺序和词角色互换。预填到首字段标签，字段值、后续标签和终止由模型自然生成。State、Note、P/Q、停止与单 token 槽位联合正确率每格需 ≥90%。

{chr(10).join(natural)}

双向干预与剂量比较以 {a['eligible_families']} 个八视图完整家族、{a['eligible_views']} 条视图为分母。先于 {f['time']} 冻结发现排名、192 个待检 MLP 单元、同数随机对照与格均值。再用原缓存和上游的原生局部运算，在 {predictions['time']} 封存所有确认集逐例预测，之后才进行确认集真实编辑。发现后未重新选择路径。二值功效规划用项目 McNemar 工具，8.5 个百分点、.10 配对不一致率及 α=.025，需 {p['power']['n_required']} 家族，规划功效 {p['power']['achieved_power']:.3f}。主要机制指标是连续 log α，区间按家族描述。

### 计算路径与预测

记 $x$ 为 L20 注意力前残差，$o$ 为全部头的写出，注意力后的残差 $a=x+W_Oo$，门控乘积 $z(a)=\mathrm{{SiLU}}(W_gN(a))\odot W_u N(a)$。

$$r_{{21}}=a+W_dz(a).$$

用相同 Q/K 重算 L20H5 Value 改动后头输出，保留其余头并经原生 $W_O$ 得到受扰 $a$。分别组合自然或受扰的 $a$ 与 MLP 输出，再执行 L21 输入 RMSNorm、原生 Q/K 投影、头内归一化及 RoPE。每次真实运行均核对 L21 两个 GQA 组的全部历史 K 与自然缓存逐元素相同。State/Note 键早已写入，当前时刻新形成的 K 位于答码 query 自身。预测将这一 K 放回原历史库，再重算完整 softmax。

完整局部原生计算与实际受扰 Q、当前 K 逐值对齐，α 最大偏差 {a['exact_full_alpha_error']:.2e}。它使用自然时点的向量、模型权重及上游已知写入，属于接力算式的核对。发现单元按 $\mathbb E[(\Delta z_i)^2]$ 乘 $W_d$ 第 $i$ 列到 L21 目标 Q/K 投影的平方量排序。192 个单元占这一排序分数 {a['discovery_unit_score_fraction']:.1%}。稀疏分支需比直达 RMSE 降低至少 10%，且优于随机单元，H19/H25 判定分别为 {a['sparse_pass']}。

| 冻结预测或计算分解 | log α RMSE H19 / H25 | 触顶平衡准确率 H19 / H25 | 漏报触顶臂 H19 / H25 |
|---|---:|---:|---:|
{chr(10).join(forecast_table)}

自然到受扰的两头 Q/当前 K 中位向量变化为 {a['qk_change_median_norm']}。L20 注意力后残差与 MLP 输出的改动范数中位数为 {a['median_direct_residual_norm']:.3f} 和 {a['median_delta_mlp_norm']:.3f}，二者对 log α 的非加性 RMS 为 {a['residual_mlp_logalpha_interaction_rms']}，不能直接按范数折算作用占比。

### 原生阻断与恢复

阻断保持 L20H5 源扰动，但将中间节点换成同前缀自然值。恢复在关闭源扰动后，移入相同源扰动生成的中间节点值。两条方向分别执行。以自然到受扰的 log α RMS 距离为分母，阻断移除率为 $1-d(阻断,自然)/d(受扰,自然)$，恢复率为 $1-d(恢复,受扰)/d(受扰,自然)$。负值说明留下的误差比原改动更大。

| 中间节点 | 阻断移除率 H19 / H25 | 恢复率 H19 / H25 | P/Q 分差平均绝对改动，阻断 / 恢复 |
|---|---:|---:|---:|
{chr(10).join(bridge)}

保持源写入但阻断 MLP 后，原本触顶而消失的臂数为 {switch['prevented']}（依次为 H19、H25），新出现触顶的臂数为 {switch['introduced']}。因此 MLP 在部分条件下会抵消残差直达引起的注意力变化，满足三头约束需要结合逐臂结果判断。

H19/H25 触顶视图为 {a['original_caps']}，按单臂计为 {a['original_cap_arms']}。H19 跨 State/Note 和旋转/供体的计数 {a['h19_cap_by_slot_signal']}。原剂量为 $a_*\Delta V$，其中 $a_*=\min(\max(\alpha_S^0,\alpha_N^0),2\min(\alpha_S^0,\alpha_N^0))$，所需原生 Value 缩放为 $a_*/\alpha_j^{{live}}$，大于 2.05 即触顶。历史 State/Note 键固定时，当前 K 只会改变 softmax 分母，Q 会同时改变与历史键的匹配和归一化。只恢复受扰当前 K 时，两槽对数比相对自然的最大误差为 {a['key_only_pair_odds_max_error']:.2e}。

将 $\alpha_j$ 拆成两槽合计质量乘槽内份额。在 H19 的 {h19['events']} 个触顶臂中，两项的对数变化中位数分别为 {h19['median_log_total_mass_change']:.3f} 与 {h19['median_log_within_pair_share_change']:.3f}。各自固定另一项比较，仅质量变化可复算 {h19['reproduced_by_mass_only']} 个触顶，仅槽内重分配可复算 {h19['reproduced_by_share_only']} 个，有重叠。这是概率分解，向量级副作用以原生阻断/恢复为准。

### 原剂量下的三头救回

始终保留原 $a_*\Delta V$、2.05 上限、供体与旋转方向。档案逐分片断言所有候选方案的目标向量范数完全相同。沿用 Phase 2940 解析写入、头输出、方向与两槽匹配门槛；每个家族八视图全部通过才计覆盖。预定门为 153 家族与 P/Q State−Note 的 ±0.25 logits 等效区间。

| 原剂量方案 | 合格视图 | 新救回 / 丢失 | 完整家族 | H19 / H25 触顶视图 | State−Note logits 与 95% 区间 | 覆盖门 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rescue)}

只恢复 H19 原查询时，H25 注意力与原三头数值保持一致。两头 Q 和当前 K 都恢复后，剩余失败为 {'。'.join(remaining)}。最大 H19 所需系数的例子 {quote['id']}，该槽自然 α {quote['head19_natural_alpha']:.4f}，受扰时 {quote['head19_live_alpha']:.4f}，所需缩放 {quote['head19_gain']:.3f}。仍失败的实例包括 {best['remaining_examples'][:3]}。逐例提示、触顶臂、阻断/恢复反例和全部剂量分片均保留。

### 图谱增量与下一阶段

新增可回查的源 Value、O 写入、L20 残差与 MLP、L21 新 Q/当前 K 以及固定历史键的局部接续。原生读出、层内代数、反事实阻断、答码分差是不同层次。单一答码 query 的阻断可检验短因果链，早期步骤、自由生成、其他模型和跨节点复算尚未覆盖。RDC 主体公式沿用既有版本。下一大阶段先冻结自然句式和多个竞争槽的预期寻址，联合拟合残差直达与稀疏 MLP 特征，再检验能否在生成过程中预测何时发生剂量饱和。优先解释失败中剩余的 BF16 写入误差与逐案例累积。

### 工程产物

RTX 5080 CUDA 最高分配 {peak:.2f} GiB，DXGI 进程共享显存采样峰值 {shared:.1f} MiB。保留首批四视图的原生 BF16 过渡场、缓存键及后续全部视图的选中 Q/K、注意力、P/Q 和剂量账本。初次试跑 BF16 转 NumPy 的工程失败保留日志，修正后通过校准并冻结。设计、材料、分片、脚本与记录有 SHA256；复核命令见 execution-receipt.json。实验为同节点真实测量与 CPU 审计。
"""


def chart(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,(l,r)=plt.subplots(1,2,figsize=(12.3,5.1),layout="constrained")
    methods=("direct","sparse","random","q_only","k_only")
    loc=np.arange(5)
    for i,head in enumerate(("L21H19","L21H25")):
        l.barh(loc+(i-.5)*.30,[a["predictors"][k]["logalpha_rmse"][i] for k in methods],
               height=.28,color=("#3b7599","#c27b58")[i],label=head)
    l.set_yticks(loc,["Direct residual","Direct + 192 selected","Direct + 192 random","Query only","Current key only"])
    l.set(xlabel="Holdout log-attention RMSE (lower is better)",title="Predicting the live attention change")
    l.legend(frameon=False)
    count=[a["doses"][k]["families"] for k in DOSE]
    bars=r.barh(loc,count,color=["#8895a5","#83a4b6","#487c9c","#3c8d82","#247364"],height=.54)
    r.bar_label(bars,padding=3)
    r.axvline(153,ls="--",lw=1,color="#a86350",label="153-family gate")
    r.set_yticks(loc,["Original trio","Block MLP change","Restore H19 query","Restore both queries","Restore queries + keys"])
    r.set(xlim=(0,207),xlabel="Complete eight-view families",title="Matching at the original dose")
    r.legend(frameon=False,loc="upper right")
    for ax in (l,r):
        ax.invert_yaxis()
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="x",alpha=.15)
    fig.savefig(S.check("relay-results.png"),dpi=170)
    plt.close(fig)


def close(preview=False):
    a=A.load("audit.json")
    text=render(a)
    S.check("report.md").write_text(text,encoding="utf-8")
    chart(a)
    if preview:
        return
    assert not (C.OUT/"memo-receipt.json").exists()
    temp=C.TMP.resolve()
    assert temp==C.TMP and temp.parent==(C.ROOT/"tests/glm5_temp").resolve() and not temp.is_symlink() and not temp.is_junction()
    children=list(temp.rglob("*"))
    assert all(x.resolve().is_relative_to(temp) and not x.is_symlink() and not x.is_junction() for x in children)
    n=sum(x.stat().st_size for x in children if x.is_file())
    shutil.rmtree(temp)
    A.save("cleanup.json",dict(path=str(temp),bytes=n,free=shutil.disk_usage(C.OUT).free))
    A.save("execution-receipt.json",dict(provenance=a["provenance"],
        commands=[f".venv/Scripts/python.exe -B -u tests/glm5/kv_relay_run.py {s}" for s in
                  ("pilot","engine-pilot","plan","discover","freeze","forecast","edit")]+[
            ".venv/Scripts/python.exe -B -u tests/glm5/kv_relay_audit.py",
            ".venv/Scripts/python.exe -B -u tests/glm5/kv_relay_closeout.py"],
        forecast_shards=len(A.load("forecast.json")["shards"]),edited_shards=len(A.load("edited.json")["shards"]),
        independent_cuda=False,production_registration=False))
    first=append(C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",TITLE,text.split("\n\n",1)[1])
    second=append(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md","L20H5 到 L21 的接力与原剂量救回索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 {C.RUN}。发现与逐例预测先冻结，之后在同一确认集验证原生阻断、恢复和原剂量三头匹配。"
                  "数据在 tests/glm5/result/kv_relay_20261115_v1/report.md。")
    A.save("memo-receipt.json",dict(records=[first,second]))
    paths=set(S.SOURCES)
    paths.update(p.relative_to(C.ROOT).as_posix() for p in Path(__file__).parent.glob("kv_relay_*.py"))
    with ZipFile(S.check("sources.zip"),"w",ZIP_DEFLATED) as z:
        for name in sorted(paths):
            z.write(C.ROOT/name,name)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(name)==(C.ROOT/name).read_bytes() for name in paths)
    A.save("manifest.json",dict(files={p.name:A.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},
                               sources={name:A.digest(C.ROOT/name) for name in sorted(paths)}))


if __name__=="__main__":
    C.configure()
    close(preview=sys.argv[1:]==["preview"])
