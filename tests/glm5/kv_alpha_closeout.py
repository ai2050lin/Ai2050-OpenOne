"""Report the weighted-write experiments, preserve raw evidence and append the memo."""

from pathlib import Path
import json
import shutil
import sys

import kv_alpha_run as C
from b36_semantic14_memo import append


R=C.R
ROOT=C.ROOT
V1=C.OUT
V2=V1.with_name("kv_alpha_20261115_v2")
TITLE="匹配注意力加权写入量后，否定组的 State 与 Note 槽效应"
PANELS={"update":"更新对照","polarity":"否定复验"}
HEADS={"h19":"L21H19","trio":"冻结三头"}


def load(folder,name):
    return R.Base.Previous.load(folder/name)


def values():
    records=[load(p,"audit.json") for p in (V1,V2)]
    return records,[s for d in records for s in d["panels"] if s["status"]=="complete"]


def fmt(x,places=3):
    return "—" if x is None else f"{x:+.{places}f}"


def interval(s):
    if s["ci95"] is None:
        return "—"
    return f"{fmt(s['mean'])} [{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}]"


def behavior_line(where,p):
    cells=p["behavior"]
    failed="，".join(f"`{c['cell']}` {c['correct']}/{c['n']}" for c in cells if not c["passed"])
    return (f"| {where} | {p['panel']} | {sum(c['correct'] for c in cells)}/{sum(c['n'] for c in cells)} "
            f"| {sum(bool(c['passed']) for c in cells)}/{len(cells)} | {failed or '全部过门'} |")


def narrative():
    records,panels=values()
    plan=load(V2,"plan.json")
    old=load(V1,"plan.json")
    job=load(V1,"replication-job.json")
    preflight=load(V1,"replication-preflight.json")
    lines=[]
    for p in panels:
        lines.extend((f"### {PANELS[p['panel']]}","",
                      "| 查询头 | 原始 State / Note 内容效应 | 原始 State−Note | 只用 α 的预测 State−Note | 单视图 RMSE / 均值偏差 / 秩相关 |",
                      "|---|---:|---:|---:|---:|"))
        for g,a in p["group"].items():
            e=a["prediction_check"]
            lines.append(f"| {HEADS[g]} | {fmt(a['raw']['state']['mean'])} / {fmt(a['raw']['note']['mean'])} "
                         f"| {interval(a['raw']['source'])} | {interval(a['predicted']['source'])} "
                         f"| {e['rmse']:.3f} / {fmt(e['bias'])} / {e['rho']:.3f} |")
        lines.extend(("", "| 查询头 / 共同剂量 | 完整家族 / 192 | 同一家族的原始位置差 | 匹配后位置差 | 匹配后 State / Note 内容效应 | 同集预测偏差 | 家族均值等效 |",
                      "|---|---:|---:|---:|---:|---:|---|"))
        for g,a in p["group"].items():
            for z,d in a["doses"].items():
                passed="通过" if d["equivalent"] else ("覆盖不足" if not d["sufficient_coverage"] else "未证实")
                lines.append(f"| {HEADS[g]} / {z} | {d['families']}/192 | {interval(d['raw_on_matched'])} "
                             f"| {interval(d['matched']['source'])} | {fmt(d['matched']['state']['mean'])} / {fmt(d['matched']['note']['mean'])} "
                             f"| {interval(d['prediction_bias_on_matched'])} | {passed} |")
        for g,a in p["group"].items():
            stat=a["prediction_check"]
            avoid="、".join(f"`{x['id']}` {fmt(x['gap'])}" for x in stat["failures"][:2])
            natural=a["alpha_medians"]
            lines.extend(("", f"- {HEADS[g]} 的自然注意力 State/Note 中位数，按冻结头顺序为 "
                          f"{'、'.join(f'{x[0]:.4f}/{x[1]:.4f}' for x in natural)}。α 预测单视图误差门{'通过' if stat['passed'] else '未通过'}。"
                          f"误差最大的反例包括 {avoid}。"))
            for z,d in a["doses"].items():
                m=d["dose_medians"]
                omitted="、".join(f"{k} {v}" for k,v in sorted(d["reasons"].items())) or "无"
                cases="、".join(f"`{x['id']}` {fmt(x['source'])}" for x in d["counterexamples"][:2]) or "—"
                lines.append(f"- {HEADS[g]} / {z}，合格视图 {d['qualifying_rows']}/768，完整家族 {d['families']}。"
                             f"在完整家族内，最弱实际写入范数的中位数 {fmt(m['actual_shared'])}，两槽最大实际向量差的中位数 {fmt(m['actual_gap'])}。"
                             f"失败原因计数允许重叠，{omitted}。最大匹配后残差示例 {cases}。")
    polarity=next((p["group"] for p in panels if p["panel"]=="polarity"),None)
    if polarity:
        h=polarity["h19"]
        t=polarity["trio"]
        result=h["doses"]["low"]["population_explanatory_check"]
        first=(f"在低剂量共同支持上，L21H19 {'通过' if result else '未通过'}预注册的组平均解释检验。"
               f"两槽效应的原始差为 {interval(h['doses']['low']['raw_on_matched'])}，匹配后为 {interval(h['doses']['low']['matched']['source'])}。"
               f"三头的原始差为 {interval(t['raw']['source'])}，需按其本身的方向与覆盖判断，不能沿用单头的反转命题。")
    else:
        first="新否定组未进入完整干预审计，否定反转的解释没有达到本阶段验收条件。"
    b="\n".join(behavior_line(f"v{i+1}",p) for i,d in enumerate(records) for p in d["panels"])
    logs=[json.loads(line) for f in (V1,V2) for line in (f/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    mem=max(x.get("cuda_peak_bytes",0) for x in logs)/2**30
    shared=max(x.get("shared_gpu_bytes") or 0 for x in logs)/2**20
    pp=plan["power"]
    binary=pp["binary_plan"]
    return rf"""# {TITLE}

本阶段沿用 Phase 2939 冻结的 Qwen3-4B 查询头 L21H19、L21H25、L20H5（零起始）。检验对象是模型已生成 State 和 Note 后，原生 Value 读取怎样影响下一枚 P/Q token。{first}

## 材料、冻结与执行边界

Qwen3-4B 原始非量化 BF16，模型快照 `{plan['provenance']['model_revision']}`，RTX 5080 上串行 CUDA，batch 16，左 padding，显式 position IDs，Transformers SDPA，禁用思考模式。计划分别绑定 `{R.sha(V1/'plan.json')}` 和 `{R.sha(V2/'plan.json')}`。模型文件名称与 SHA256、运行时版本、源码哈希、家族词对与种子均见计划。每组 192 个无序词对家族 × 4 个视图，交叉替换正确 State 和 P/Q 码本；否定复验另平衡否定句在前或在后。试跑使用独立词对。v1 更换了否定措辞和 Note 词表，部分行为格失败，因此 v2 沿用 Phase 2939 的句式和颜色词，但重新冻结了未暴露的 State 词对。v1 的失败样本全部保留，只作为行为记录。确认材料均为原创模板。

| 冻结版本 | 套组 | 联合合格视图 | 过门的格 | 未过门明细 |
|---|---|---:|---:|---|
{b}

每个格要求 90% 的 State、Note、Code、停止及 token 对齐联合正确率。匹配位置的主要统计量以完整四视图家族为单位。等效界定为位置差的家族 bootstrap 95% 区间在 ±0.25 logits 内，至少 {pp['equivalence_retained_family_n']} 个家族可匹配。规划假设家族标准差 0.85 logit、真实位置差为 0，采用上述更保守区间估计，近似等效检验力 {pp['equivalence_power_centered_zero']:.3f}。二值规划用项目 McNemar 工具，预期变化 {binary['expected_delta_pt']} 个百分点、配对不一致率 .10、α=.025，需 {binary['n_required']} 个家族，计划数下功效 {binary['achieved_power']:.3f}。正文的主要效应为连续 logits，二值规划不作为实际功效证明。区间以家族重抽样，对共享词汇、同模板和多指标造成的相关性仍需谨慎。

## 原生算法

固定自然历史和预测答码的 query。对同一头 $h$，注意力输出为

$$o_h=\sum_j\alpha_{{hj}}V_{{j,g(h)}},\qquad g(h)=\lfloor h/4\rfloor.$$

替代 State 供体从相同句子的反事实首个生成词分支获得，定义 $d=V_{{alt}}-V_{{state}}$，方向负控制为范数相同的层固定循环旋转 $Rd$。插入 State 与 Note 使用同一 $d$。跨层时实时读取当前干预臂的 Q/K，以便 L20 改动后的 L21 仍匹配其实际系数。

原始干预把每槽的 V 增量设为 $d$ 或 $Rd$，由原生 SDPA 产生写入。匹配干预先从自然注意力取共同目标系数 $a_{{low}}=\min(\alpha_S^0,\alpha_N^0)$，$a_{{high}}=\min[\max(\alpha_S^0,\alpha_N^0),2a_{{low}}]$，然后对当前槽缩放 $V'=\mathrm{{BF16}}(V+d\cdot a/\alpha_{{live}})$。缩放上限 2.05。CPU 审计用保存的原值、供体和权重重算 $\alpha_{{live}}(V'-V)$，并核对同一 Q/K 下两次原生 SDPA 的实际头输出差 $\widehat{{w}}=o'_h-o_h$。同时限制最小目标幅度、BF16 解析误差、向量方向、实际剂量误差和两个槽的实际向量差。无写入、剂量过小和超出合理缩放的样本进入失败账本。每臂每头一次读取，只替换指定 query 的头输出，缓存历史保持固定。自然回放在答码处对齐完整词表 logits。

并行的 α 预测绕开 Value 编辑，把 $\alpha_{{live}} d$ 或 $\alpha_{{live}} Rd$ 直接加到原生头输出，再继续原模型前向。两槽共享方向及后续模型，预测依赖两槽的 α。当前组响应过程中早层的写入可改变后层 α。这里的预测针对位置效应的代数来源，使用了模型真实局部状态。

结果单位是 $m=z_{{替代码}}-z_{{正确码}}$。槽的内容净效应 $D_j=m_{{j,d}}-m_{{j,Rd}}$，表格位置差为 $D_S-D_N$。负值对应 Note 影响更强。若原生头在同一个 query 收到了完全相同的输出写入，固定前缀下后续计算相同是算子性质。本实验还要求原始反转复现、有效剂量可测、按 α 预测的组均值在相同家族内复现原始差，并保留单视图预测不准的案例。

## 实测结果与反例

{chr(10).join(lines)}

## 解释与下一阶段

实验只覆盖固定单词 Note、指定答码时刻和两种人工报告族。在匹配剂量内，接近零的位置差说明把相同写入交给同一时刻的相同原生头即可消掉这部分槽位优势。较高剂量有的头受限于动态注意力和缩放上限；覆盖不足时只能描述保留下来的家族。注意力权重本身还可能受 query、历史、token 身份及下游非线性共同影响。本轮未测状态词生成机制、完整自由续写、一般语义否定或自然长文。表中正反对照、离群误差与不合格视图均可用原始 ID 回查。

外部图谱新增位置平衡的否定词对与改写失败记录。内部图谱新增三个 query head 的逐样本真实注意力、BF16 Value 变化和注意力头写入账本。关联图可以按材料和剂量条件定位原始读出及匹配边界。RDC 主体公式未更新。下一大任务是冻结自然材料上的可计算寻址预测，把“谁生成 State”“谁在多个可见记录中被寻址”“每个头写入的内容与后续残差如何合成”连成一个能预测未见条件的模型。先分开估计条件化 α 与单位有效写入的下游响应，再冻结预测，比较词法、位置基线与原生头组合，最后由独立节点复算并检查写入之外的失败样本。

## 独立 GPU 复算与可追溯性

独立工单状态 `{job['status']}`。包内包含两次冻结计划、材料、源码、运行命令和接收标准，不带本节点输出。真实复算需要另一物理主机和 GPU，严格匹配冻结计划、权重、tokenizer 及运行时。回传后比较有效覆盖（≥90% 的入选交并比）、逐视图分差（≥95% 落在 0.35 logit 容差）和完整家族的均值差（≤0.25 logit）。工单已在隔离目录通过两个冻结版本的文件和导入校验，同一节点提交已被比较器正确拒绝，核对记录为 `{preflight['status']}`。当前仅有本节点实测与同节点 CPU 数值审计，未接到独立节点回执。工单 SHA256 `{job['package_sha']}`。

运行入口为 `kv_alpha_run.py`、`kv_alpha_confirm.py` 与 `kv_alpha_audit.py`，复算入口为 `kv_alpha_replication.py`。两个结果目录分别为 `tests/glm5/result/kv_alpha_20261115_v1` 和 `tests/glm5/result/kv_alpha_20261115_v2`。逐视图结果在 `audit-rows-*.json`。每个 `writes-*.npz` 保存选中 KV 组的 State、Note、供体、自然注意力和实测干预头向量，首批另存四视图多生成步各层全坐标 BF16。进度逐批刷新，CUDA 分配峰值 {mem:.2f} GiB，DXGI 共享显存采样峰值 {shared:.1f} MiB。早期快照计算作用域不一致、试跑词表可用单 token 不足和 v1 行为失败均在开封正式干预前处理，并保留可用日志。正式中心登记本轮未启动。
"""


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=values()[1]
    fig, ax=plt.subplots(figsize=(10.0,5.6),layout="constrained")
    aliases={"update":"Register update","polarity":"Polarity"}
    typ=(("raw","Raw donor",-.19,"#bd624e"),("predicted","Alpha-only reconstruction",0,"#426e91"),
         ("low","Matched native write",.19,"#318673"))
    labels=[]
    for p in selected:
        for g,d in p["group"].items():
            i=len(labels)
            dose=d["doses"]["low"]
            name="L21H19" if g=="h19" else "Three heads"
            labels.append(f"{aliases[p['panel']]} | {name}  ({dose['families']}/192 matched)")
            for k,name,offset,color in typ:
                stats={"low":dose["matched"]["source"],"raw":dose["raw_on_matched"],
                       "predicted":dose["prediction_on_matched"]}[k]
                if stats["ci95"] is None:
                    continue
                x=stats["mean"]
                lo,hi=stats["ci95"]
                er=ax.errorbar(x,i+offset,xerr=[[x-lo],[hi-x]],color=color,fmt="o",
                               capsize=3,label=name if i==0 else "_nolegend_")
                if k=="low" and not dose["sufficient_coverage"]:
                    er.lines[0].set_markerfacecolor("white")
    ax.set_yticks(range(len(labels)),labels)
    ax.invert_yaxis()
    ax.axvspan(-.25,.25,color="#dce4e7",alpha=.7,label="±0.25 logit equivalence band")
    ax.axvline(0,color="#546574",ls="--",lw=1)
    for key in ("top","right"):
        ax.spines[key].set_visible(False)
    ax.set(xlabel="State minus Note content effect (logits)",
           title="Qwen3-4B | Raw, reconstructed and matched on the same families")
    ax.grid(axis="x",alpha=.2)
    ax.legend(loc="best",frameon=False)
    fig.savefig(R.Base.Previous.check_write(V2/"weighted-write-results.png"),dpi=170)
    plt.close(fig)


def cleanup():
    base=(ROOT/"tests/glm5_temp").resolve()
    cleaned=[]
    for name in (V1.name,V2.name,"kv_alpha_replica_preflight"):
        p=base/name
        if not p.exists():
            continue
        assert p.resolve()==p and p.parent==base and not p.is_symlink() and not p.is_junction()
        members=list(p.rglob("*"))
        assert all(x.resolve().is_relative_to(p) and not x.is_symlink() and not x.is_junction() for x in members)
        size=sum(q.stat().st_size for q in members if q.is_file())
        shutil.rmtree(p)
        cleaned.append(dict(path=str(p),bytes=size))
    R.write("cleanup.json",dict(removed=cleaned,disk_free=shutil.disk_usage(V2).free))


def close():
    if (V2/"memo-receipt.json").exists():
        raise RuntimeError("memo already appended")
    output=narrative()
    R.Base.Previous.check_write(V2/"report.md").write_text(output,encoding="utf-8")
    plot()
    cleanup()
    body="\n".join("#"+line if line.startswith(("## ","### ")) else line
                    for line in output.split("\n\n",1)[1].splitlines())
    first=append(ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",TITLE,body)
    second=append(ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md","注意力加权剂量、否定槽位效应与独立复算索引",
                  f"详见 GLM5 Phase {first['phase']}。配对原生 Value 试验、α 直接写入预测、两个共同剂量、反例及行为失败均可回查。"
                  "独立 GPU 的冻结工单与回传容差见 tests/glm5/result/kv_alpha_20261115_v1/replication-job.json。"
                  "完整报告在 tests/glm5/result/kv_alpha_20261115_v2/report.md。")
    R.write("memo-receipt.json",dict(records=[first,second]))
    R.write("manifest.json",{d.name:{p.name:R.sha(p) for p in sorted(d.iterdir())
                                     if p.is_file() and p.name!="manifest.json"} for d in (V1,V2)})


if __name__=="__main__":
    C.configure(V2)
    if len(sys.argv)>1 and sys.argv[1]=="preview":
        R.Base.Previous.check_write(V2/"report.md").write_text(narrative(),encoding="utf-8")
        plot()
    else:
        close()
