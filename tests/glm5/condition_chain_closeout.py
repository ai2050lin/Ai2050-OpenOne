"""Report the five frozen substudies, keep their evidence, and append the research memos."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from collections import defaultdict

import condition_chain_audit as A
import condition_chain_run as R
from b36_semantic14_memo import append

DEST=A.BASE/A.FOLDERS[0]
OUTPUTS=[A.BASE/name for name in A.FOLDERS]


def draw():
    A.setup_version(5)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    isolated=R.load(OUTPUTS[3]/"audit.json")
    audit=R.load(OUTPUTS[4]/"audit.json")
    tests=next(c for c in audit["analysis_summaries"][0] if c["task"]=="assessment")["tests"]
    groups={(g["split"],g["cue"]):g for g in audit["native_descriptive"] if g["task"]=="assessment"}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    order=["select","update","maintain","report","composed"]
    for s in isolated["sections"]:
        if s["split"]=="pilot":
            continue
        for domain in ("stock","calendar"):
            cells={r["stage"]:r for r in s["summary"] if r["task"]==domain}
            axes[0].plot(range(5),[cells[t]["correct"]/cells[t]["n"] for t in order],
                         marker="o",label=f"{domain} / {s['split']}")
    axes[0].set(xticks=range(5),xticklabels=order,ylim=(.5,1.03),ylabel="Forced-choice accuracy",
                title="Supplied-state probes and composition")
    axes[0].axhline(.85,color="#8798aa",ls=":")
    axes[0].legend(fontsize=8,loc="lower left")
    for i,row in enumerate(tests):
        grouped=groups[row["split"],row["cue"]]["G_cluster_sensitivity"]
        for j,x in enumerate([row,grouped]):
            lo,hi=x["ci95"]
            axes[1].errorbar(x["mean"],i+(.12 if j else -.12),
                             xerr=[[x["mean"]-lo],[hi-x["mean"]]],fmt="o",capsize=3,
                             color=("#19678b" if not j else "#c55536"),
                             label=("96 families" if not j else "6 context groups") if i==0 else None)
    axes[1].set(yticks=range(len(tests)),yticklabels=[f"{t['split']} / keyword {t['cue']}" for t in tests],
                xlabel="Mean paired G difference (measured minus guessed)",title="Qwen3-4B B31 / U2619")
    axes[1].axvline(0,color="#8798aa",ls=":")
    axes[1].legend(fontsize=8,loc="lower right")
    fig.savefig(R.check_write(DEST/"research-results.png"),dpi=150)
    plt.close(fig)


def table(rows):
    return "\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)


def tables(data):
    runs=table([["版本","运行 ID","实测审计条目"],["---","---","---:"],
                *[(p["version"],p["run_id"],p["verified_rows"]) for p in data]])
    chain=[["模型","留出","库存","日程","过门格"],["---","---","---:","---:","---:"]]
    for s in sorted(data[1]["sections"],key=lambda x:(x["model"],x["split"])):
        if s["split"] in ("words","frames"):
            scores={r["task"]:f"{r['correct']}/{r['n']}" for r in s["summary"]}
            chain.append([s["model"],s["split"],scores["inventory"],scores["calendar"],
                          f"{sum(k['pass_gate'] for k in s['gates'])}/{len(s['gates'])}"])
    isolated=[["领域","拆分","选择","更新","保持","报告","组合"],["---","---","---:","---:","---:","---:","---:"]]
    for s in sorted(data[3]["sections"],key=lambda x:x["split"]):
        for name in ("stock","calendar"):
            scores={r["stage"]:f"{r['correct']}/{r['n']}" for r in s["summary"] if r["task"]==name}
            isolated.append([name,s["split"],*[scores[k] for k in ("select","update","maintain","report","composed")]])
    candidates=[["候选","留出","强制选择","行为门"],["---","---","---:","---:"]]
    for s in data[4]["sections"]:
        if s["split"] in ("words","frames"):
            for r in s["summary"]:
                candidates.append([r["task"],s["split"],f"{r['correct']}/{r['n']}",
                                   f"{sum(g['pass_gate'] for g in s['gates'] if g['cell'][0]==r['task'])}/4"])
    effects=[["留出","关键词","均值差 G","家族 dz","家族 p","场景聚类 95% CI","MSE 比"],["---","---:","---:","---:","---:","---:","---:"]]
    groups={(n["split"],n["cue"]):n for n in data[4]["native_descriptive"] if n["task"]=="assessment"}
    measured=next(c for c in data[4]["analysis_summaries"][0] if c["task"]=="assessment")
    for q in measured["tests"]:
        c=groups[q["split"],q["cue"]]["G_cluster_sensitivity"]["ci95"]
        effects.append([q["split"],q["cue"],f"{q['mean']:.4f}",f"{q['dz']:.3f}",f"{q['p']:.3g}",
                        f"[{c[0]:.3f}, {c[1]:.3f}]",f"{q['mse_ratio']:.3f}"])
    duplicate=[["V3候选","留出","唯一前缀家族","名义家族"],["---","---","---:","---:"]]
    for s in data[2]["sections"]:
        if s["split"] in ("words","frames"):
            duplicate.extend([r["task"],s["split"],r["exact_distinct_first_prefix_families"],r["families"]] for r in s["summary"])
    return runs,*(table(rows) for rows in (chain,isolated,candidates,effects,duplicate))


def memo_text(data):
    runs,chain,isolated,candidates,effects,duplicate=tables(data)
    logs=[json.loads(line) for p in OUTPUTS for line in (p/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peaks={m:(round(max([r.get("cuda_peak_bytes",0) for r in logs if r["model"]==m],default=0)/2**30,3),
              round(max([r.get("shared_gpu_bytes",0) for r in logs if r["model"]==m],default=0)/2**20,1))
           for m in ("q4","q14","glm")}
    proof=(OUTPUTS[4]/"center-sandbox-proof.json").exists()
    remote=R.load(OUTPUTS[4]/"publication.json").get("remote_receipt")
    release=("线上已登记并由公开接口回读，界面渲染未验。" if remote else
             "正式健康接口为200、版本1.0.29。缺少可用管理员身份，未上传或正式登记。浏览器无法确认当前URL，正式客户端未验。")
    proof_scope=("隔离的发布版存档路由收录并读回6个对象，哈希一致。" if proof else
                 "归档6个对象通过本地依赖顺序与哈希校验，隔离服务接口尚未验收。")
    return f"""**范围。** 承接 Phase 2935。用非算术的库存标签和日程地点任务检验条件选择、内容更新、终态保持和答案报告。另从 Phase 2920 冻结4B的越界 B13/U9571、付费责任 B16/U9414、测量判定 B31/U2619。材料均为原创虚构文本，版本与模型分开记账。同一家族有多条配对视图，表中的条目数不是独立样本数。

{runs}

**测试原理。** 对选源 s∈{{0,1}}、替换 u∈{{0,1}}，候选按来源与初始/替换排列，外部真值 z₀=2s、z₁=2s+u。无关示例不改真值，保持 z₂=z₁。直接报告 z₂ 对应内容，转码先计算题面映射 π(z₂)，再匹配选项字母。这些是任务定义。V1/V2每家族16格。V2同时更换对话模板、说明与材料，不能将V2相对V1的差异只归于模板。GLM4和14B参考各使用每留出前32家族/域及8家族试跑，不比较通用模型排名。V4把上一步真值给在题面上，逐段检验能力；每留出96家族/域。组合题只测试保留选中的值，不含真正替换。新句式保持题把示例放在真值之前，所以只检验该提示结构下排除示例，不能从中推出模型保留了自己生成的状态。

**评分。** 强制选择要求正确选项 logit 严格领先。候选化负对数概率为 logΣⱼexp(ℓⱼ)−ℓ_true，完整单 token 的负对数概率为 logΣᵥexp(ℓᵥ)−ℓ_true。两者分开；仅试跑自由生成8 token。V4试跑首字母145/160，完整裸字母0/160，需把额外复述或截断与内容错误分开。模型行为逐格门为≥85%，不从正确答题子集补门。读出采用前缀稳定的最后一个 Checkpoint token；V1为 Record ends。问题、选项及答案均不进入前段候选读出。

{chain}

{isolated}

V4给定真值的保持题两组均384/384；组合题为275/384与341/384。更新与报告也有未过门的格。V1/V2联合行为门未通过，原定全层HiddenState采集、输出相关原生状态干预和稀疏对比未执行。V4给出中间状态不能替代自主形成与保持的验证。

**原生候选。** 对归一化后的当前输入 z，第j个MLP单元有 Gⱼ=gⱼᵀz、Vⱼ=uⱼᵀz、Aⱼ=SiLU(Gⱼ)Vⱼ，写出为 Aⱼdⱼ。G的相关变化不能直接当作实际写入或输出因果。V3出现唯一前缀少于名义家族的问题，保留为探索诊断。V5完全重新生成材料，每个留出每候选的96个家族由6个场景与16个实体构成，精确前缀互异，场景内仍共享结构。

{duplicate}

{candidates}

主检验按家族在固定关键词格内计算 G(成立)−G(不成立)，要求正向 dz≥0.35、12项Bonferroni α=.05/12，含关系标签的岭预测相对于词与双词、前缀平均embedding及长度基线的MSE比≤0.90。独立家族近似下，96个家族对dz=.35的双侧t检验事后计算理想功效约0.688，不能推广到6个聚类或联合门。同一留出两个关键词格共享MSE比；关系标签由实验者提供，不能当作模型自行构造特征。V5测量判定四组同向，但MSE比.912/.917高于冻结门槛。按场景聚类后有一组描述性区间含零。V3同候选多组方向反转，且有重复前缀，两版不合并。越界、付费的部分行为格失败，原生候选联合证据不完整。所有三候选均未达到机制晋级条件，匹配消融未进入原模型执行。

{effects}

**结论与理论边界。** 先区分模型能单独执行的运算、组合时实际维持的状态和作答映射。已有证据支持具体条件下的行为差异，以及4B B31/U2619在V5材料中的观察性G响应。内部并未验证可复用因果链，没有训练检查点验证、同预算稀疏字典、未来完整续写预测、独立节点复跑。外部图谱保存分阶段材料与反例，内部档案保存通过行为门后的G/V/A，关联层只保留范围内观察。RDC主体不变。

**第一性原理与下一阶段。** 有限参数的组合能力需要解释什么状态在何时被写入、何时仍可读取、后续规则怎样使用它。下一大阶段在新家族上正交改变保留或实质替换、示例位置和答案映射，优先打通自然成功与给定状态之间的差距。随后在同一留出对比原生坐标、前缀状态和同预算字典，预测未来输出与未见条件，再在同缓存自然分支做匹配消融、恢复和副作用检查。将可迁移规则的预测和干预同时作为升级条件。

**工程与发布。** 本地原始BF16，CUDA依次运行。14B自动装载异常退出0xC0000005，首个评分前以8GiB内常驻预算法和原权重逐块装载替代。CUDA分配峰值/共享显存日志采样峰值（GiB/MiB）：{peaks}。所有实测条目CPU复核前缀分词、读出、分数和准入，模型身份按官方文件口径重建。边界修订发生在首个评分前，保留启动失败日志。没有生产端代码或客户端改动。{proof_scope} {release} V4完整发布门失败，V5仅准备观察归档，不提升图谱证据。结果和脚本分别见 tests/glm5/result/condition_chain_*、tests/glm5/condition_chain_*.py。"""


def cleanup():
    diagnostics=[]
    parent=(R.ROOT/"tests/glm5_temp").resolve()
    for directory in (parent/name for name in A.FOLDERS):
        assert directory.resolve()==directory and directory.parent==parent and not directory.is_symlink() and not directory.is_junction()
        if not directory.exists():
            continue
        files=list(directory.rglob("*"))
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(directory) for p in files)
        size=sum(p.stat().st_size for p in files if p.is_file())
        shutil.rmtree(directory)
        diagnostics.append(dict(path=str(directory),bytes=size))
    R.save_json(R.check_write(DEST/"cleanup.json"),dict(trees=diagnostics,free_after=shutil.disk_usage(DEST).free))


def paired_diagnostics():
    """Use shared families and preserve model identity and per-task results."""
    from scipy.stats import binomtest
    comparisons=[]
    reference=OUTPUTS[1]
    for split in ("words","frames"):
        material=R.load(reference/f"materials-{split}-chain.json")
        scores={}
        for key in ("q4","q14","glm"):
            rows=R.load(reference/f"{key}-{split}-chain.json")["results"]
            scores[key]={r["id"]:r["correct"] for r in rows}
        for domain in ("inventory","calendar"):
            same=[r for r in material if r["task"]==domain and r["index"]<32]
            assert len(same)==512 and all(r["id"] in scores[k] for r in same for k in scores)
            by_family=defaultdict(list)
            for row in same:
                by_family[row["family"]].append(row["id"])
            comparisons.append(dict(panel="matched_reference",split=split,domain=domain,n=len(same),families=len(by_family),
                accuracy={m:sum(s[r["id"]] for r in same)/len(same) for m,s in scores.items()},
                all_16={m:sum(all(s[r] for r in v) for v in by_family.values()) for m,s in scores.items()}))
        isolated=OUTPUTS[3]
        material=R.load(isolated/f"materials-{split}.json")
        actual=R.load(isolated/f"q4-{split}.json")["results"]
        for domain in ("stock","calendar"):
            paired=defaultdict(lambda:defaultdict(list))
            for r,o in zip(material,actual,strict=True):
                if r["task"]==domain:
                    paired[r["family"]][r["stage"]].append(o["correct"])
            lost=sum(all(item["maintain"]) and not all(item["composed"]) for item in paired.values())
            gained=sum(not all(item["maintain"]) and all(item["composed"]) for item in paired.values())
            p=binomtest(min(lost,gained),lost+gained).pvalue if lost+gained else 1.0
            comparisons.append(dict(panel="isolated_vs_composed",split=split,domain=domain,families=len(paired),
                                    maintenance_both_flip=sum(all(v["maintain"]) for v in paired.values()),
                                    composition_both_flip=sum(all(v["composed"]) for v in paired.values()),
                                    lost=lost,gained=gained,mcnemar_descriptive=p))
    R.save_json(R.check_write(DEST/"paired-diagnostics.json"),dict(scope="descriptive: stages have different task structures",results=comparisons))


def finish():
    data=[R.load(folder/"audit.json") for folder in OUTPUTS]
    assert sum(s["model"]=="q14" for s in data[1]["sections"])==3
    assert sum(s["model"]=="glm" for s in data[1]["sections"])==3
    assert not (DEST/"memo-append-receipt.json").exists()
    paired_diagnostics()
    report=memo_text(data)
    R.check_write(DEST/"report.md").write_text("# 条件选择、内容更新、终态保持与答案报告\n\n"+report+"\n",encoding="utf-8")
    scripts=sorted(Path(__file__).parent.glob("condition_chain_*.py"))
    source=R.check_write(DEST/"source.zip")
    with ZipFile(source,"w",ZIP_DEFLATED) as archive:
        for script in scripts:
            archive.write(script,script.name)
    with ZipFile(source) as archive:
        assert all(archive.read(script.name)==script.read_bytes() for script in scripts)
    cleanup()
    coverage={q.name:{p.name:R.file_sha(p) for p in sorted(q.iterdir()) if p.is_file()} for q in OUTPUTS}
    R.save_json(R.check_write(DEST/"manifest.json"),coverage)
    paths=[R.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",R.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"]
    first=append(paths[0],"非算术条件链与三类原生候选：分阶段行为、泛化反例及预测门失败",report)
    index=f"""详见GLM5 Phase {first['phase']}。{sum(s['verified_rows'] for s in data):,}条实测视图经CPU复核，不等于独立样本量。非算术全链联合门未过。Qwen3-4B B31/U2619四个对照G同向，预测MSE比0.912/0.917未达到0.90，其余两候选保留行为失败。原生机制未晋级，观察包的生产状态见主记录与publication.json。"""
    receipts=[first,append(paths[1],"非算术条件链与Goodfire三候选留出验收索引",index)]
    for path,receipt in zip(paths,receipts,strict=True):
        assert hashlib.sha256(path.read_bytes()[:receipt["original_bytes"]]).hexdigest()==receipt["original_sha256"]
    R.save_json(R.check_write(DEST/"memo-append-receipt.json"),dict(records=receipts))
    print("Appended GLM5",first["phase"],"Claude",receipts[1]["phase"],flush=True)


if __name__=="__main__":
    {"draw":draw,"finish":finish}[sys.argv[1]]()
