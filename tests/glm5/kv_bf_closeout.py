"""Publish an append-only account of the fixed-dose BF16 and cross-size tests."""

import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import kv_bf_study as S
from b36_semantic14_memo import append


C=S.C
A=C.A
TITLE="原剂量下的 BF16 写入舍入干预与 Qwen3-4B／14B 对照"
NAMES={"dynamic":"保留动态 Q","native":"恢复 Q + 原生 BF16","once":"增量 FP32，头输出再舍入","late-round":"重加 Value 舍入，投影后再舍入","late":"增量 FP32，投影后再舍入"}


def interval(x):
    if not x["ci95"]:
        return "—"
    return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def render(a):
    p=A.load("plan.json")
    f4=A.load("freeze-q4.json")
    f14=A.load("freeze-q14.json")
    h=A.load("q14-selection.json")
    check=A.load("q14-calibration.json")
    old=A.load("phase2942-reassessment.json")["counts"]
    traces=A.load("direction-replay.json")
    sections=[]
    for size,ket in (("4B","q4"),("14B","q14")):
        z=a["full"][ket]
        if z["status"]!="finished":
            sections.append(f"- {size} 确认状态 {z['status']}。冻结的发现选头和行为记录保留。")
            continue
        sections.extend((f"#### {size}","",
            "| 条件 | 合格视图 | 完整家族 | 新救回 / 丢失 | 三头中第一／第二接收头触顶 | State−Note 的 P/Q 内容差 |",
            "|---|---:|---:|---:|---:|---:|"))
        for k,title in NAMES.items():
            x=z["arms"][k]
            sections.append(f"| {title} | {x['views']} | {x['families']} | {x['gain']} / {x['loss']} | "
                            f"{x['flags'][0][1]} / {x['flags'][1][1]} | {interval(x['margin_effect'])} |")
        t=z["rounding"]
        sep="、".join(x["id"] for x in t["examples"][:3]) or "本组无该类失败"
        sections.extend(("", f"- 第二接收头在恢复 Q 后的原生失败视图 {t['native_target_failure_views']}，"
                            f"方向失败 {t['direction_views']}。单次舍入救回 {t['native_to_once_rescued']} 条。"
                            f"重新加入 Value 舍入使 {t['rounding_reinsert_loses']} 条重新失败，"
                            f"保留头输出舍入而只拿掉输入舍入相对投影后舍入少通过 {t['head_output_cast_loses']} 条。"
                            f"这些计数按同一套剂量、原生头输出、方向和两槽配对判据。代表 ID {sep}。",
                            f"- CPU 对所有保留的 Value 原值、供体与注意力重算显式舍入，"
                            f"输入误差范数中位数 {t['med_input_error']:.6f}，一次舍入下头输出误差中位数 {t['med_output_error_after_once']:.6f}。"))
        b=z["arms"]["late"]["flags"]
        sections.append(f"- 投影后再舍入仍有绝对目标低于最小有效剂量的视图，三个头分别为 {[x[0] for x in b]}。"
                        f"至少一视图因这条判据失败的完整家族有 {z['arms']['late']['dose_floor_blocked_families']} 个。"
                        f"自然 State／Note 注意力权重的接收头中位数为 {z['baseline_state_note_median']}。")
    paired=a["paired_common"]
    common=[]
    if paired:
        common.extend((
            "| 同样完整家族 | 保留动态 Q | 恢复 Q + 原生 BF16 | 增量 FP32 | 重加 Value 舍入 | 投影后再舍入 |",
            "|---|---:|---:|---:|---:|---:|"))
        for name,code in (("4B","q4"),("14B","q14")):
            d=paired[code]
            common.append(f"| {name}，{paired['families']} 家族 | "+
                          " | ".join(str(d[m]["families"]) for m in NAMES)+" |")
        common.extend(("", "| 模型 / 自然语境 | State 先时两头 State 占优率 | Note 先时两头 State 占优率 |",
                       "|---|---:|---:|"))
        for name,code in (("4B","q4"),("14B","q14")):
            d=paired[code]["addressing"]["preference"]
            for c in range(2):
                n1,n2=(next(x for x in d if x["context"]==c and x["order"]==o) for o in (0,1))
                seen=lambda x:" / ".join(f"{u:.1%}" for u in x["state_preference"])
                common.append(f"| {name} {'否定' if c==0 else '更新'} | {seen(n1)} | {seen(n2)} |")
    else:
        common.append("共同材料中可比较的机制数据未达到执行门。")
    hist=[]
    for item in traces["cases"]:
        hist.append(f"| {item['id']} | {item['directional']['native']} | {item['directional']['once']} "
                    f"| {item['directional']['late-round']} | {item['directional']['late']} |")
    still_once=sum(bool(x["directional"]["once"]) for x in traces["cases"])
    still_late_round=sum(bool(x["directional"]["late-round"]) for x in traces["cases"])
    q4=a["full"]["q4"]
    q14=a["full"]["q14"]
    chosen=q4["arms"][f4["primary"]]
    natural=a["q4_behavior"]
    forced=a["q14_behavior"]
    logs=[json.loads(x) for x in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    stats={}
    for m in ("q4","q14"):
        use=[x for x in logs if x["model"]==m]
        stats[m]=(max(x.get("cuda_peak_bytes",0) for x in use)/2**30,
                  max(x.get("shared_gpu_bytes") or 0 for x in use)/2**20)
    exact_prefix=f"{forced['exact_prefix_code']}/{forced['n']}"
    exc=forced["whole_greedy_compatible_families"]
    greed=(f"仅取 14B 自己的贪心前缀与答码均一致的 {exc['families']} 个完整家族，"
           f"原生、单次舍入、重加 Value 舍入与投影后舍入的合格视图分别是 {[exc['arms'][m]['views'] for m in ('native','once','late-round','late')]}。"
           if exc else "")
    return rf"""# {TITLE}

运行 {p['run']}。目标是判断 Phase 2942 恢复 L21H19、L21H25 自然 Q 后，三头组剩余匹配失败的数值来源，并验证固定原 αΔV 和 2.05 缩放上限能否过 153 家族门。本阶段另补 Qwen3-14B 的模型内发现与同前缀对照。此前 Phase 2941–2942 的任务明确要求 Qwen3-4B；更早的 14B 参考主要是行为校准，因此缺少相同程度的机制证据。

4B 的冻结方法在留出达到 {chosen['families']} 个家族，原生 BF16 为 {q4['arms']['native']['families']}，153 门判定为 {chosen['threshold_153']}，输出位置差等效判定为 {chosen['equivalence_025']}。本阶段新增的方向偏差解释来自旧反例的定向复放，单独记载。

### 材料、模型与时序

4B 快照 {a['q4_snapshot']}，14B 快照 {a['q14_snapshot']}，均为未量化原始 BF16。分别核对模型与分词器文件和运行时。4B 在同样的八格设计中用 72 个家族发现、192 个新家族确认。确认动物词表与 Phase 2942 的发现、确认词表分离。每家族八视图交叉语境、输出顺序和词角色。4B 自然 State、Note、P/Q、停止和 token 对齐联合正确 {natural['strict']}/{natural['n']}，逐格 90% 后要求八视图全部合格。功效规划为 {p['power']['achieved_power']:.3f}，对应配对差 {p['power']['expected_delta_pt']} 个百分点、所需 {p['power']['n_required']} 家族，连续误差另记。

4B 于 {f4['stamp']} 冻结 {f4['primary']}，候选选择在确认生成前完成。发现先比较一次舍入与投影后舍入，完整家族数相同优先保留较小的精度改动。14B 预定 16 个发现家族、48 个确认家族，先扫描 L19–L27 源头的加权 $W_O$ 写入范数，再用源 Value 原生编辑选择下一层两个有实时注意力变化且 GQA 组不同的头。冻结路线为源 L{h['source'][0]}H{h['source'][1]} → L{h['receiver_layer']}H{h['receiver'][0]} 与 H{h['receiver'][1]}（零起始）。14B 自身于 {f14['stamp']} 冻结 {f14['primary']}。头号、坐标和权重没有跨模型对齐。

14B 在预定共同材料上验证给定 State/Note 前缀后的 P/Q，合格 {forced['conditional']}/{forced['n']}。全部前缀 token 及 P/Q 同时符合该模型贪心选择的计数为 {exact_prefix}，逐 token 验证了这些路径；评分范围到首个 P/Q token。机制表中 14B 使用与 4B 字面相同的前缀，条件 P/Q 每格 90% 后取完整家族。14B 两例与原生标准 HF 前向比较，P/Q 最大差 {check['pq_error']:.3f} logits，全词表 top1 一致。14B 为规模参照，153 家族门专用于 4B 的完整确认。

### 固定剂量、两处舍入和物理边界

对固定头和槽，原自然权重 $alpha_S^0,alpha_N^0$ 定义共同目标 $a_*=min(max(alpha_S^0,alpha_N^0),2min(alpha_S^0,alpha_N^0))$。使用同一供体或固定旋转方向 $d$，期望向量 $w_*=a_*d$。恢复接收两头自然 Q，实际所需缩放 $s=a_*/alpha_{{live}}$，严格使用 $min(s,2.05)$；阈值事件仍按未截断 $s$ 判定。各精度臂的目标向量范数、Q/K 注意力及所需缩放逐分片相等。历史 State、Note 值和键作为本任务固定前缀。

原生 Value 先存回 BF16，实际解析增量为

$$w_V=alpha_{{live}}\left(\mathrm{{BF16}}(V+min(s,2.05)d)-V\right).$$

显式对照保持供体与旋转的原目标，单次舍入令 $w=\alpha_{{live}}\min(s,2.05)d$，在原生头输出上加未先舍入的增量，然后转 BF16 并经原 $W_O$。投影后舍入把该增量沿真实 $W_O$ 的该头列块，用 FP32 累加到输出再转 BF16。另一臂把显式 Value 舍入重加，但仍使用相同投影后舍入。两处对照针对写入数值路径，网络权重、源写入、P/Q 候选、旋转对照和限幅均保持同一设置。投影后舍入以物理生效的 FP32 头增量核对入投影剂量。它是指定局部精度干预，报告的 P/Q 来自原生其余计算。其他两头使用原生方法。

### 新留出结果

{chr(10).join(sections)}

恢复 Q、改变舍入的同时，仍报告实际 P/Q 内容差和每个家族的区间，不以数值目标通过来推断普遍答码机制。4B 原生模式、单次舍入、重加 Value 舍入和投影后舍入对齐比较。14B 第一和第二接收头采用独立发现排序。模型间运行接口的区别是 4B 缓存自然回放，14B 从同一固定前缀逐层推进，随后对最后 query 分支；两者均用各自原生 SDPA。

### 完整共同家族比较

{chr(10).join(common)}

共同分母要求两模型对应的行为门和八视图全覆盖。{greed}比较限于该材料和模型内所选路线，差异可来自寻址、门槛、V 的局部量化格距或本次发现的头。仅比较物理现象和交集，不将某一头编号当作跨模型对象。

### 旧方向偏差的回放和反例

CPU 复算 Phase 2942 恢复两头自然 Q 的账本，解析误差视图 {old['analytic']}，方向偏差 {old['direction']}，重叠 {old['overlap']}。在本次新留出，恢复 Q 后 4B 原生方向失败 {q4['rounding']['direction_views']} 条。用原来的批形状和原始自然前缀回放 {traces['n']} 个旧方向反例，先核对原生 gain 与 H25 剂量门标签逐元素一致，再比较四个精度变体。下表是各条件仍未满足方向门的臂索引，索引 0/1 是 State 旋转/供体，2/3 是 Note 旋转/供体。

| 旧 ID | 原生 | 单次舍入 | 重加 Value 舍入 | 投影后舍入 |
|---|---|---|---|---|
{chr(10).join(hist)}

单次舍入仍留下 {still_once} 样本的方向偏差。把舍入推迟到投影后，即使显式重加输入 Value 舍入，方向偏差仍剩 {still_late_round} 样本，解析剂量误差可以同时保留。这把旧方向偏差定位到头输出 BF16 舍入边界，和新留出的输入 Value 舍入是两个数值环节。

旧反例回放用于定位失败，未进入新留出的泛化统计。提示全文、输入和输出误差、归零坐标率、方向和实际输出保存在 audit.json、direction-replay.json 及对应数值分片。

### 机制拼图与下一阶段

三图谱新增同一语言条件下的注意力目标、Value 浮点表示、原生头输出和接收端写入的可复算关系。在局部把 α、方向、剂量和精度分开，可找出“寻址可用但写入被舍入改变”的例子。更一般的预测仍需解释自然语料上哪些结构会落在较小的有效写入区间。在外部图谱引入更长距离和多个竞争槽的模板，在内部图谱冻结按层的误差传播与下游输出预测，再在同一材料比较两模型和对照精度。RDC 主体公式维持现状。当前结果来自同一物理 GPU，未做跨节点独立复算。

### 工程交付

本地串行 4B、14B、4B 历史反例回放。14B 首次尝试较大常驻时分配失败，恢复为少量权重常驻、原始 safetensors 逐层读取、CPU 暂存激活，避免复制权重与磁盘卸载。HF 自动掩码与显式掩码的首轮 P/Q 比对超出容差，保留日志与旧计划。正式数据开封前统一因果掩码，再按同一容差核对。CUDA 峰值 GiB／进程共享显存 MiB 分别为 4B {stats['q4'][0]:.2f}/{stats['q4'][1]:.1f}，14B {stats['q14'][0]:.2f}/{stats['q14'][1]:.1f}。源文件、冻结计划、分词、失败、剂量和复算文件通过清单核对。命令见 execution-receipt.json。
"""


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(11.7,4.7),layout="constrained")
    variants=("native","once","late-round","late")
    bars=left.bar(range(4),[a["full"]["q4"]["arms"][m]["families"] for m in variants],
                  color=["#728497","#3f8b7e","#a56e58","#508092"])
    left.bar_label(bars,padding=3)
    left.axhline(153,ls="--",lw=1,color="#ad6754",label="153-family gate")
    left.set(xticks=list(range(4)),xticklabels=["Native","One cast","Reinsert\nV rounding","Cast after\nprojection"],
             ylabel="Complete eight-view families",ylim=(0,208),title="Qwen3-4B | original dose and cap")
    left.legend(loc="upper left",frameon=False)
    paired=a["paired_common"]
    if paired:
        labels=["Native","One cast","Reinsert\nV rounding","Late cast"]
        for model,name,col in (("q4","4B","#3b7b99"),("q14","14B","#b77253")):
            vv=[paired[model][m]["views"] for m in variants]
            right.plot(range(4),vv,"o-",label=name,color=col)
            for j,y in enumerate(vv):
                right.annotate(str(y),(j,y),xytext=(0,8),textcoords="offset points",ha="center",color=col)
        right.set(xticks=list(range(4)),xticklabels=labels,ylabel="Passing views",ylim=(0,paired["families"]*8+32),
                  title=f"Common prefix panel | {paired['families']} eligible families")
        right.legend(loc="lower right",frameon=False)
    else:
        right.axis("off")
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.14)
    fig.savefig(C.OUT/"bf16-results.png",dpi=170)
    plt.close(fig)


def close(preview=False):
    a=A.load("audit.json")
    text=render(a)
    R=A.R.Base.Previous
    R.check_write(C.OUT/"report.md").write_text(text,encoding="utf-8")
    R.check_write(C.OUT/"bf16-results.png")
    figure(a)
    if preview:
        return
    assert not (C.OUT/"memo-receipt.json").exists()
    t=C.TMP.resolve()
    assert t.parent==(C.ROOT/"tests/glm5_temp").resolve() and t==C.TMP and not t.is_symlink() and not t.is_junction()
    paths=list(t.rglob("*"))
    assert all(x.resolve().is_relative_to(t) and not x.is_symlink() and not x.is_junction() for x in paths)
    n=sum(x.stat().st_size for x in paths if x.is_file())
    shutil.rmtree(t)
    A.save("cleanup.json",dict(bytes=n,path=str(t),free=shutil.disk_usage(C.OUT).free))
    A.save("execution-receipt.json",dict(
        steps=[f"python -B -u tests/glm5/kv_bf_run.py {x}" for x in ("pilot","engine-pilot","plan","discover")]+[
            "python -B -u tests/glm5/kv_bf_freeze.py","python -B -u tests/glm5/kv_bf_run.py confirm",
            "python -B -u tests/glm5/kv_bf_q14.py","python -B -u tests/glm5/kv_bf_direction.py",
            "python -B -u tests/glm5/kv_bf_audit.py","python -B -u tests/glm5/kv_bf_closeout.py"],
        provenance=a["provenance"]))
    result=append(C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",TITLE,text.split("\n\n",1)[1])
    ix=append(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md","BF16 原剂量补偿及两模型比较索引",
              f"详见 GLM5 Phase {result['phase']}，4B 新材料、14B 模型内选头、同前缀对照及历史方向反例均可复核。"
              "完整结果在 tests/glm5/result/kv_bf_20261115_v1/report.md。")
    A.save("memo-receipt.json",dict(records=[result,ix]))
    names=set(S.SOURCES)
    names.update(p.relative_to(C.ROOT).as_posix() for p in Path(__file__).parent.glob("kv_bf_*.py"))
    with ZipFile(R.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for k in sorted(names):
            z.write(C.ROOT/k,k)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(k)==(C.ROOT/k).read_bytes() for k in names)
    A.save("manifest.json",dict(
        files={p.name:A.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},
        sources={k:A.digest(C.ROOT/k) for k in sorted(names)}))


if __name__=="__main__":
    C.setup()
    close(preview=sys.argv[1:]==["preview"])
