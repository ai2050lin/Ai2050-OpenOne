"""Write research deliverables, keep complete negatives, and append dated memo entries."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile,ZIP_DEFLATED

import kv_read_run as R
from b36_semantic14_memo import append

S={"discovery":"暴露发现","frames":"新句式","multi":"两次更新","gap":"生成间隔干扰"}
T={"first":"状态词后首查询","interior":"中间区间","answer":"答码预测查询"}


def table(cols,items):
    return "\n".join("| "+" | ".join(str(x) for x in row)+" |" for row in [cols,["---"]*len(cols),*items])


def charts(data):
    d=data["discovery"]
    if d.get("status")!="recomputed discovery selection":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels=[s for s in data["confirmation"] if s["status"]=="complete"]
    fig,ax=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    for f,color in (("keys","#486da8"),("values","#a04d33")):
        scan=sorted(d["by_field"][f]["layers"],key=lambda x:x["layer"])
        ax[0].plot([p["layer"] for p in scan],[p["delta"] for p in scan],label=f,color=color)
    ax[0].axhline(0,color="#aaaaaa",lw=1)
    ax[0].set(title="Exposed discovery: one layer edited at a time",xlabel="Zero-based block",ylabel="Change in alternate-code logit margin")
    ax[0].legend()
    labels=[]
    for i,suite in enumerate(panels):
        for j,(field,color) in enumerate((("keys","#486da8"),("values","#a04d33"))):
            row=suite["findings"][field]["sparse"]
            at=2*i+j
            lo,hi=row["ci95"]
            ax[1].errorbar(row["mean"],at,xerr=[[max(0,row["mean"]-lo)],[max(0,hi-row["mean"])]],fmt="o",color=color,capsize=3)
            labels.append(f"{suite['suite']} · {field}")
    ax[1].axvline(0,color="#aaaaaa",lw=1)
    ax[1].axvline(.5,color="#a04d33",lw=1,ls=":")
    ax[1].set(yticks=range(len(labels)),yticklabels=labels,title="Frozen three heads vs. dose-matched control",
              xlabel="Family mean logit contrast (95% bootstrap interval)")
    fig.savefig(R.Base.Previous.check_write(R.OUT/"layer-and-heldout-results.png"),dpi=150)
    plt.close(fig)


def narrative(data):
    conf=[s for s in data["confirmation"] if s["status"]=="complete"]
    status={s["suite"]:s["status"] for s in data["confirmation"]}
    v1=R.Base.Previous.load(R.OUT.with_name("kv_read_20261115_v1")/"q4-discovery-behavior.json")
    behavior=table(["版本或范围","任务","正确/视图","格通过","说明"],[
        ("V1 4B","失败发现",f"{sum(c['correct'] for c in v1['cells'])}/32",
         f"{sum(c['passed'] for c in v1['cells'])}/4","保留格各2/8"),
        *[("V2 4B",S[x["suite"]],f"{sum(c['correct'] for c in x['cells'])}/{sum(c['n'] for c in x['cells'])}",
           f"{sum(c['passed'] for c in x['cells'])}/{len(x['cells'])}",status.get(x["suite"],"发现材料"))
          for x in data["behavior"]],
        *[(key,S[suite],f"{sum(c['correct'] for c in s['cells'])}/{sum(c['n'] for c in s['cells'])}",
           f"{sum(c['passed'] for c in s['cells'])}/{len(s['cells'])}","行为小样本")
          for key,v in data["reference"].items() for suite,s in v.items()]])
    selected=table(["方式","筛出的层","冻结的(层,查询头,时间)"],[
        (f,record["layers"],", ".join(f"{l}.{h} {T[t]}" for l,h,t in record["points"]))
        for f,record in data["selection"].get("choices",{}).items()])
    indexed=[]
    for suite in conf:
        for f,record in suite["findings"].items():
            for short,key in (("全层换/等幅","broad"),("三头换/等幅","sparse"),("冻结时间/其余时间","timing"),
                              ("三头归零/等幅","removal"),("全层归零/选中恢复","rescue"),
                              ("匹配恢复/选中恢复","rescue_specific")):
                s=record[key]
                indexed.append((S[suite["suite"]],f,short,f"{s['mean']:+.3f}",f"{s['p']:.2g}",
                                "通过" if s["passed"] else "失败"))
    scores=table(["留出","只改","配对比较","均值(logit)","p","门"],indexed)
    heads=table(["留出","方式","层.查询头","冻结时刻","同剂量差","相对其他时刻","效应门/时刻门"],[
        (S[s["suite"]],f,f"{h['point'][0]}.{h['point'][1]}",T[h["point"][2]],
         f"{h['specific']['mean']:+.3f}",f"{h['clock']['mean']:+.3f}",
         f"{int(h['intervention_grade'])}/{int(h['timing_grade'])}")
        for s in conf for f,x in s["findings"].items() for h in x["individual"]])
    free=table(["留出","完整自由续写","全层K翻转","全层V翻转","全层V等幅翻转","选中V头翻转"],[
        (S[s["suite"]],s["n"],*[s["free"][k]["alternate"] for k in
                                 ("keys-all_swap","values-all_swap","values-all_scramble","values-selected")]) for s in conf])
    negative=[]
    for suite in conf:
        chosen=suite["findings"]["values"]["sparse"]["least_effect"]
        fail=suite["free"]["values-all_swap"]["failure"][:1]
        negative.append(f"- {S[suite['suite']]}：三头最弱配对 {chosen['id']}，差 {chosen['difference']:+.4f} logit。"
                        + (f" 全层V自由续写反例 {fail[0]['id']}，预期另一词编码 {fail[0]['expected']}，实际 {repr(fail[0]['actual'])}。" if fail else
                           " 全层V自由续写未见此类失败。"))
    clock=table(["发现阶段","全程","首查询","中间","答码查询"],[
        (field,*[f"{found['broad_clock'][k]:+.3f}" for k in ("all","all-first","all-interior","all-answer")])
        for field,found in data["discovery"].get("by_field",{}).items()])
    receipt=R.read("production.json") if (R.OUT/"production.json").exists() else {}
    if receipt.get("verified"):
        assert (R.OUT/"production-client.jpg").exists()
        counts=R.read("publication.json")
        release=(f"正式中心 {receipt['version']} 已登记并回读 {len(receipt['sha256'])} 对象。{counts['cards']} 张卡、"
                 f"{counts['edges']} 条边按范围干预登记。客户端已打开报告和定位卡，报告SHA256 {receipt['study']['report_sha256']}，"
                 f"回读{receipt['time']}，截图 production-client.jpg。未晋级正式主线证据，独立GPU复算待办。")
    else:
        release="没有获得正式服务器读回证明；本地发布状态按 publication.json 记账。"
    stats=[json.loads(line) for line in (R.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    perf={key:(round(max([r.get("cuda_peak_bytes",0) for r in stats if r["model"]==key],default=0)/2**30,3),
               round(max([r.get("shared_gpu_bytes",0) for r in stats if r["model"]==key],default=0)/2**20,1))
          for key in sorted({r["model"] for r in stats})}
    return f"""**范围。** 承接Phase 2937。区分状态词缓存的寻址与内容，检验下游哪一层和哪一个查询头在何时能读取它。主模型本地Qwen3-4B原始BF16，参考Qwen3-14B和GLM4串行仅跑行为。V1短KEEP发现题因行为未过门而停止，失败原样保存。V2以Phase2937的32个已公开示例作为8家族发现；三组留出各96个新词对家族、每家族4种更新×映射组合。多次更新进一步分第二步保留与替换。间隔干扰要求模型在自生成状态与答码之间另写无关Note行。多次更新以最终状态为被干预位置，不定位第一次更新的生成过程。观察须按行为、层头确认、恢复对照分别记账。

{behavior}

**模型计算。** 在真实一次生成中处理自然状态词 g₁ 后保存KV，保留自然 g₂。另从该次原 prompt 缓存处理另一候选词，获得同位置供体 K'、V'。设读头 h 的 query 为 qₕ，共用KV组 g=⌊h/4⌋，原生输出 zₕ=Σₛsoftmaxₛ(qₕ·K(g,s)/√128)V(g,s)。K为已过RoPE的原生键，换K保持V，换V保持K。只在冻结步骤和指定query头重算被改动的 native SDPA 输出，在 o_proj 前替换该头的结果。未选择的头在同次 attention 调用内保留原生值。由此定位的是查询头，不能把同组四个查询头当作四组独立原生KV参数。分支从该次生成的缓存克隆，检查自由回放完整token及自然答案边界全词表logits逐位一致，供体分叉前的prompt缓存逐层哈希不变。主指标取固定自然非答案前缀上的 m=logit(另一词编码)−logit(正确编码)。仅测定答码，不将答案提前输入分支。另从 State 后做一组真正自由续写，并要求有效格式及干扰字段保持。

**时间与负对照。** 「首查询」是状态词之后首个被处理的token，「答码查询」是自然答码前一token，「中间」是二者间。时间皆在自然轨迹计数；中间或补集可包含多次读取，比较限制在此定义。等幅乱序施加 x+roll(x'−x)，按BF16舍入后剂量误差≤3%。归零测试只移除选中读取的K或V向量信息；K归零不会屏蔽该位置，注意力仍可能读到它。恢复从各层所有后续查询均归零的分支恢复选定的真实原生读，另用同层同数其他查询头恢复作匹配。功能恢复分数上升和通过预注册完整移除/恢复链是两件事。无广泛无关任务副作用检查。

**冻结候选。** 发现比较226臂。先扫每层，选每方向两个层，再在各32个查询头上发现三头，然后对其三个时间窗口排名。初筛不计作留出。

{selected}

{clock}

**留出结果。** 一家族内4变体先取均值，96家族做单侧符号检验，最多90项分配α=.05/90。配对效应同时须达到计划规定的logit阈值、各格均值非负。bootstrap区间为描述。若查询头匹配对照有效且时刻补集对照失败，保留读干预作用，不认定独占时刻。家族仍重复值词及有限模板，不能将其作为对任意语言分布的独立采样。

{scores}

{heads}

{free}

**保留反例。**

{chr(10).join(negative)}

**解释、第一性原理和下一大阶段。** 值槽在这一程序中作为以后可读取的内容入口。供体K替换成效、归零后恢复以及V内容替换分别回答不同问题；若K正向效应未过门，不能推出模型在这些任务里不使用Key寻址。单头可能影响输出的竞争，即使幅度不足以翻转最终答案。三组变动了任务措辞、词表和输出结构，组间效应差不能单独归因于延迟或干扰。跨句式复现后仍不等于最小、唯一、必要或任意任务共享的电路。下一阶段用更分散的新语言功能单独预注册源位置寻址、目标值写入与头组恢复，匹配读取次数和自然注意力分数，测试组合更新、较长干扰后的路径替换，并比较原生路径和同预算稀疏方法对完整后续分布与干预效应的预测。RDC主体公式不改。

**工程与发布。** CUDA BF16串行，14B逐层装载。CUDA分配峰值/共享显存采样最大值GiB/MiB：{perf}。CPU复核全部执行过的输入分词、真值、实际生成、输出分差与冻结排行，复算了抽样原生向量的BF16剂量。所有对照失败、非显著、零效应和自由续写原文均保留。抽样4视图保存每步的embedding、各层和归一化后的全坐标；保存目标位两套K/V样本与原始KV哈希，不长期保存完整prompt缓存。含启动解析和脚本作用域问题及V1行为失败的旧日志、版本冻结和原输入一并保留。文件位于 tests/glm5/result/kv_read_20261115_v1 和 tests/glm5/result/kv_read_20261115_v2。{release}"""


def cleanup():
    directory=(R.Base.ROOT/"tests/glm5_temp").resolve()
    removed=[]
    for name in ("kv_read_20261115_v1","kv_read_20261115_v2"):
        p=directory/name
        assert p.parent==directory and p.resolve()==p and not p.is_symlink() and not p.is_junction()
        if not p.exists():
            continue
        members=list(p.rglob("*"))
        assert all(x.resolve().is_relative_to(p) and not x.is_symlink() and not x.is_junction() for x in members)
        bytes_freed=sum(x.stat().st_size for x in members if x.is_file())
        shutil.rmtree(p)
        removed.append(dict(path=str(p),bytes=bytes_freed))
    R.write("cleanup.json",dict(removed=removed,free=shutil.disk_usage(R.OUT).free))


def close():
    assert not (R.OUT/"memo-receipt.json").exists()
    data=R.read("audit.json")
    description=narrative(data)
    R.Base.Previous.check_write(R.OUT/"report.md").write_text("# 自生成状态的 Key/Value 与读取时刻\n\n"+description+"\n",encoding="utf-8")
    if not (R.OUT/"layer-and-heldout-results.png").exists():
        charts(data)
    files=sorted(Path(__file__).parent.glob("kv_read_*.py"))
    files.extend(Path(__file__).with_name(p) for p in R.read("plan.json")["sources"] if not p.startswith("kv_read_"))
    files.extend(Path(__file__).with_name(p) for p in ("own_state_audit.py","own_state_explicit.py","source_binding_resources.py","b36_switch14_engine.py"))
    name=R.Base.Previous.check_write(R.OUT/"source.zip")
    with ZipFile(name,"w",ZIP_DEFLATED) as dst:
        for p in files:
            dst.write(p,p.name)
    with ZipFile(name) as src:
        assert all(src.read(p.name)==p.read_bytes() for p in files)
    cleanup()
    previous=R.OUT.with_name("kv_read_20261115_v1")
    R.write("manifest.json",{p.name:{f.name:R.sha(f) for f in sorted(p.iterdir()) if f.is_file()} for p in (previous,R.OUT)})
    memos=[R.Base.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",R.Base.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"]
    result=append(memos[0],"自生成状态的 Key/Value 分离及读取头和时刻：跨句式、两次更新与间隔验证",description)
    index=f"""详见GLM5 Phase {result['phase']}。Qwen3-4B原生SDPA测试区分已生成状态的Key与Value内容，冻结缩小到查询头/时刻后才开封新材料。完整证据区分预填回放、固定自然前缀分差、自由答码、等幅对照及归零/恢复；未过门场景和反例保留。正式状态与独立复算边界以主记录和 tests/glm5/result/kv_read_20261115_v2/production.json 为准。"""
    records=[result,append(memos[1],"自生成状态的原生K/V与读取路径范围验收索引",index)]
    for file,row in zip(memos,records,strict=True):
        assert hashlib.sha256(file.read_bytes()[:row["original_bytes"]]).hexdigest()==row["original_sha256"]
    R.write("memo-receipt.json",dict(records=records))
    print("Appended",result["phase"],records[1]["phase"],flush=True)


if __name__=="__main__":
    R.setup()
    if sys.argv[1]=="chart":
        charts(R.read("audit.json"))
    elif sys.argv[1]=="close":
        close()
