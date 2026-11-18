"""Write the append-only Phase 2945 report and verify its archived inputs."""

import json
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from b36_semantic14_memo import append
import kv_natural_run as C


A,B,P = C.A,C.B,C.P
TITLE = "14B 自然答码中的 L25H18 → L26H4/H34 阻断、原生恢复与竞争槽"
NAMES = {"u":"L25H18 阻断", "r4":"L26H4 阻断", "r34":"L26H34 阻断", "pair":"两个接收头阻断",
         "rescue-specific":"接收头救援超过随机救援", "rescue-absolute":"上游阻断下的接收头净救援"}
CELLS = {"00":"一竞争槽 · 短", "01":"一竞争槽 · 长", "10":"三竞争槽 · 短", "11":"三竞争槽 · 长"}


def effect(z):
    return f"{z['mean']:+.3f} [{z['ci'][0]:+.3f}, {z['ci'][1]:+.3f}]"


def report(a,p,f):
    pilot1=A.load("pilot-behavior-v1.json")
    pilot2=A.load("pilot-behavior-v2.json")
    pilot3=A.load("pilot-behavior.json")
    disc=A.load("discover-behavior.json")
    hold=A.load("confirm-behavior.json")
    stock=A.load("stock-check.json")
    controls=f["control"]
    print_stats=lambda z:sum(x["correct"] for x in z["cells"].values())
    writes=[json.loads(l) for l in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cuda=max(l.get("cuda_peak_bytes",0) for l in writes)/2**30
    shared=max(l["shared_gpu_bytes"] or 0 for l in writes)/2**20
    spaces=sum(bool(x["whitespace"]) for x in hold["rows"])
    top=f"""# {TITLE}

使用本地 Qwen3-14B 原始 BF16 快照 `{p['identity']['snapshot']}`。运行 `{C.RUN}`。Phase 2944 证明冻结接收头的受控写入剂量可达。本轮问的是模型自己生成到答码时，三个头的自然写入是否支撑正确的 P/Q，以及上游作用能否由两个接收头的原生写入补回。

### 行为与实验边界

原创词表 12 试跑、32 发现、128 留出家族。每家族八视图交叉竞争槽数、生成距离、State/Note 顺序，家族间平衡否定/更新和 P/Q 身份。四个生成槽始终存在。低竞争时 Note 指向相反答码，高竞争时 Spare 和 Archive 也指向相反答码。长条件在已生成槽和 `Code:` 之间写一个更长的 Receipt，使无其他格式变化时每槽距离增加 36 个 token。最远槽在短条件距 query 22 token、长条件 58 token，最后一槽为 9/45。14B 的权重与 tokenizer 文件哈希同 Phase 2944。

逐 token 在全词表求贪心预测，从 chat 的 assistant 边界检查到 `Code:`。一次因果前向可验证整个候选续写，因为每个位置只能读自己的前缀。V1 试跑按严格 LF 预测，{print_stats(pilot1)}/96 条通过，首次分歧集中在 Markdown 行末双空格加换行或 CRLF。V2 核对独立换行 token，{print_stats(pilot2)}/96 条通过，剩余反例的 CRLF 与收据末尾句号结合成一个 token。V3 允许固定的有限行末空白，保留字面句号，根据模型实际选中的 token 合并边界，并逐 token 重新核验。State、竞争槽、Receipt 文本和 P/Q 决策沿用同一口径，答码使用未限制的全词表 argmax。V3 试跑 {print_stats(pilot3)}/96，发现 {print_stats(disc)}/256，留出 {print_stats(hold)}/1024。留出中 {spaces} 条使用了实际生成的替代换行。进入干预的视图中，所有生成槽先通过贪心检查，再在 `Code:` 的最后 token 阻断或恢复并记录实际生成的一个答码 token。未进入干预的视图保留真实前缀的首次分歧，观察止于分歧或答码。HF 原版前向与逐层实现，在 {stock['positions']} 个试跑生成位置上全词表 argmax 一致，最大 logit 差 {stock['max_logit_error']:.4g}。核对使用相同显式因果掩码和有限数值精度。

### 冻结干预与统计

在 L25/L26 最后 query 的 $W_O$ 前截断指定头。每个反事实分支重算动态 Q、K 和后续计算，其先前 token 历史按因果边界共享。上游、两个接收头分别阻断，另测联合接收头阻断及上游与接收头同时阻断。同位原生恢复先归零，再将该点本身的 BF16 头向量写回，验证编辑链路的可逆性。交叉救援保持 L25H18 阻断，把完好执行中 L26H4/H34 的原 BF16 输出写回接收位置；接收端 Q 和 K 仍取各干预分支实算。这个反事实测试指定头的写入能否解释上游阻断的输出影响。两份同层随机头依据发现集原生 $\\|W_{{O,h}}z_h\\|_2$ 的几何均值选近邻四分位后固定随机种子抽取。L25 对照头 {controls['source']}，L26 的对照组 {controls['receivers']}。每组接收头来自不同 GQA 组。用相同预算做阻断和交叉恢复。

令 $m=L_{{gold}}-L_{{alt}}$。阻断损失是 $m_{{sham}}-m_{{block}}$，主终点为它超出两组随机阻断平均损失的量。接收头净救援为 $m_{{u+restore}}-m_u$，特异救援为 $m_{{u+restore}}-\\overline{{m_{{u+restore(random)}}}}$。家族等权，八视图平均。六个主终点用每项 99.167% 家族自助区间及 0.10 logit 效应门，至少 80 个完整家族。逐一删词复算，避免单个高复用词主导。真实答码翻转和分条件结果是描述性。抢先写入的固定组件是已冻结的，不用留出选择头。
"""
    if a["state"]!="finished":
        return top+f"\n### 结果\n\n留出行为门未通过，按冻结设计暂停机制确认。全部 {len(a['natural_fails'])} 个失败前缀和冻结分条件账本在 `audit.json`。下一阶段先针对错误格校准自然生成，再冻结新确认集。\n"
    n=a["eligible_families"]
    main=["| 预注册终点 | 效应 logit [99.167% 区间] | 删单词效应范围 | 通过 |","|---|---:|---:|:---:|"]
    for k,label in NAMES.items():
        z=a["primary"][k]; x,y=z["leave_one_word_out"]
        main.append(f"| {label} | {effect(z)} | {x:+.3f} ~ {y:+.3f} | {'是' if z['passed'] else '否'} |")
    rank=["| 干预臂 | 答错视图 | 保持全八视图正确的家族 | 相对 sham 的答码边际损失 [95%] |","|---|---:|---:|---:|"]
    for k,z in a["behavior"].items():
        rank.append(f"| `{k}` | {z['flips']}/{z['n']} | {z['complete']}/{n} | {effect(z['margin_loss'])} |")
    grid=["| 条件 | L25H18 对照校正损失 | 双接收头对照校正损失 | 特异接收救援 | 上游阻断/双接收阻断/救援后答错 |",
          "|---|---:|---:|---:|---:|"]
    for k,label in CELLS.items():
        z=a["cells"][k]
        grid.append(f"| {label} | {effect(z['endpoints']['u'])} | {effect(z['endpoints']['pair'])} | {effect(z['endpoints']['rescue-specific'])} | "+" / ".join(str(z['flips'][j]) for j in ("u","pair","rescue"))+f"（共 {z['n']}） |")
    x=a["physical"]; s=a["numeric"]
    mid=lambda l,arm,k:x[str(l)]["actual_deltas"][arm][k][1]
    l25=mid(25,"u","actual_projection_delta")
    l26=mid(26,"u","actual_projection_delta")
    q26=mid(26,"u","query_change")
    cur=x["26"]["head_write_change"]["u"]["before_with_live_q"][1]
    resc=mid(26,"rescue","actual_residual_delta")
    details=["| 条件 | H4 当次生成的四槽 α 中位数 | H34 当次生成的四槽 α 中位数 |",
             "|---|---:|---:|"]
    for k,label in CELLS.items():
        v=x["26"]["sham"]["alpha"][k][1]
        details.append(f"| {label} | "+" / ".join(f"{j:.4g}" for j in v[0])+" | "+" / ".join(f"{j:.4g}" for j in v[1])+" |")
    negatives=[]
    for k,z in a["counterexamples"].items():
        if z:
            e=z[0]
            negatives.append(f"- `{k}`，样本 `{e['row']['id']}`，上游阻断的正确码边际损失 {e['signed_source_loss']:+.3f}，接收写入净救援 {e['signed_rescue']:+.3f}。")
    verdict=("预注册的接力判据通过。在此读出时刻和留出范围内，源头阻断、接收头联合阻断及原生交叉救援一起支持这条路线参与自然答码。"
             if a["chain_passed"] else "预注册的整条接力判据未通过。各头贡献、物理写入变化和交叉救援应依据各自的结果判断。")
    return top+f"""
### 留出结果

{n} 个家族完成干预账本，共 {8*n} 条视图，自然失败 {a['n_natural_failures']} 条。{verdict}

{chr(10).join(main)}

绝对上游损失 {effect(a['effects']['absolute_source'])}，绝对双接收头损失 {effect(a['effects']['absolute_pair'])}，双阻断遮挡量 {effect(a['effects']['occlusion'])}。随机受体恢复本身的平均净救援 {effect(a['effects']['mean_rnd_rescue'])}。自恢复与 sham 的最大 P/Q 差 {max(s['self_restore_vs_sham'].values()):.4g}，sham 与初始自然检查 P/Q 最大差 {s['sham_vs_open']:.4g}。逐层全前缀和并行单查询路径使用不同形状的 BF16 内核。试跑时全词表贪心 token 一致，原始 P/Q 最大差 .50，因此开封前冻结跨形状原始 logit 容差 1.0、同形状自恢复容差 .125，以及全词表首 token 一致条件。效果在并行形状相同的分支中做配对。

{chr(10).join(rank)}

{chr(10).join(grid)}

### 物理路径、实际写入与反例

设 $z_h=\\sum_t\\alpha_{{h,t}} V_{{g(h),t}}$，位于投影前的真实 BF16 头输出由 CUDA SDPA 返回。按槽拆分 $W_{{O,h}}(\\alpha V)$ 以原权重 FP32 重算，实际含舍入的全头合投影与残差从 BF16 前向取值。L25H18 阻断后，L25 全投影偏移范数中位数 {l25:.4f}，L26 为 {l26:.4f}。L26 两个接收 query 偏移中位数 {q26[0]:.4f}/{q26[1]:.4f}，接收头在施加任何 L26 编辑前的自然写入偏移 {cur[0]:.4f}/{cur[1]:.4f}。交叉救援使实际 L26 投影后残差偏移中位数为 {resc:.4f}。两接收头写回向量与活体 sham 的最大偏差 {s['rescue_recipient_max_delta_from_live_sham']:.4g}。救援和只阻断上游的接收 Q 完全一致，因此这项比较在同一动态 Q 上改变了指定预投影输出。

{chr(10).join(details)}

表中顺序为 State、Note、Spare、Archive。此前提示里的相同 token 也可能被寻址，按槽汇总该部分 α，数据在 `audit.json/physical` 与逐样本 NPZ。把 Value 按原 $W_O$ 投影的范数描述了写入大小，完整结果同时记录物理写入与 P/Q，方便识别写入大而答码稳定的案例。

{chr(10).join(negatives)}

### 严格边界与下一大阶段

干预作用于 `Code:` 的最后 query，先前已经生成的词与角色保留。得到的依赖结论只覆盖这个位置和原模型的贪心行为；还需检验模型写出 State 等早期 token 时的依赖，以及答码之后的停止和继续生成。归零干预可移除同一头在所有可见 token 上的写入，由接收头原生写回检验写入贡献，精确的 QK 查询、V 选择来源和阻断的非目标用途仍需更局部的干预。随机样本只有每层两组，匹配基于发现写入范数，方向及其他功能特性可能不同。显著区间按家族重采样，词汇在家族间复用。独立配对正态近似的 .40 标准化差功效 {a['independent_power_dz_040']:.3f}，实际推广权重受词汇、两个受控模板、位置和一份快照限制。竞争条件改了码本映射，长条件同时带来多段固定语义文本，均需在其他措辞与结构上复验。自然早错或非规定输出不被计为机制正例，独立节点和其他任务副作用尚待测量。

外部图谱新增实际生成的槽身份、到读出距离和竞争码数。内部图谱新增动态查询变化、原生头写入、实际投影残差及后续 P/Q。关联图谱可把「头写入变化」「当地物理接续」「输出特异性」三条证据分别绑定到同一记录。有限参数组合的第一性命题是角色顺序和相距位置变化时，模型能否用可迁移的寻址规则保证组成项继续可读。下一大阶段把自然生成沿时间展开，冻结从写 State 到读答码的跨 token 预测，比较提示原词和自产槽这两类来源的遮断与各自救援；在同一新语法上用匹配任务和远处候补路径检测副作用。以每种条件下真实答码分布和特征写入的预测保真度继续检验 RDC 的局部拼图。

### 工程与复算

使用原始 safetensors 逐层送 CUDA，CPU 暂存，中间目录按 D 盘策略管理。首个 V3 干预试跑发现 Q 追踪张量有额外序列轴，修正后单独重跑该校准，随后才冻结。RTX 5080 CUDA 分配峰值 {cuda:.2f} GiB，进程共享显存峰值 {shared:.0f} MiB。产物位于 `tests/glm5/result/kv_natural_20261115_v1/`，包括行为行、完整分片、原生对照、恢复和反例、日志、结果图及哈希清单。已核对冻结身份及追加前字节前缀。证据状态为本地实验与技术复算。
"""


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
    nm=("L25H18 block","L26H4 block","L26H34 block","Receiver pair block","Specific rescue","Absolute rescue")
    vals=[a["primary"][k] for k in NAMES]
    v=np.array([x["mean"] for x in vals]); ci=np.array([x["ci"] for x in vals])
    ax.errorbar(v,range(len(v)),xerr=np.abs(ci-v[:,None]).T,fmt="o",color="#346b80",capsize=3)
    ax.axvline(0,color="#8e9296",lw=1); ax.axvline(.1,color="#a16f50",lw=1,ls="--")
    ax.set(yticks=range(len(v)),yticklabels=nm,xlabel="Paired effect in logit units (99.167% intervals)",title="Preregistered paired contrasts")
    ax.invert_yaxis()
    keys=("u","r4","r34","pair","u-pair","rescue")
    counts=[a["behavior"][k]["flips"] for k in keys]
    bars=bx.bar(range(len(keys)),counts,color=["#586b80"]*5+["#4e887d"])
    bx.bar_label(bars,padding=3)
    bx.set(xticks=range(len(keys)),xticklabels=["L25H18","L26H4","L26H34","Pair","All three","Rescue"],
           ylim=(0,max(2,max(counts)*1.23)),ylabel="Incorrect greedy next tokens",title=f"{a['eligible_families']*8} eligible held-out views")
    for sub in (ax,bx):
        sub.spines[["top","right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"causal-effects.png"),dpi=175)
    plt.close(fig)


def close(preview):
    p=C.verify()
    a=A.load("audit.json")
    f=A.load("freeze.json")
    text=report(a,p,f)
    P.check_write(C.OUT/"report.md").write_text(text,encoding="utf-8")
    if a["state"]=="finished":
        figure(a)
    if preview:
        return
    assert not (C.OUT/"memo-receipt.json").exists()
    scratch=C.TMP.resolve()
    assert scratch==C.TMP and scratch.parent==(C.ROOT/"tests/glm5_temp").resolve()
    owned=list(scratch.rglob("*"))
    assert all(x.resolve().is_relative_to(scratch) and not x.is_symlink() and not x.is_junction() for x in [scratch,*owned])
    clean=sum(x.stat().st_size for x in owned if x.is_file())
    shutil.rmtree(scratch)
    A.save("cleanup.json",dict(path=str(scratch),bytes_removed=clean,free=shutil.disk_usage(C.OUT).free))
    sources=set(p["sources"])|{"tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for name in sorted(sources):
            z.write(C.ROOT/name,name)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(name)==(C.ROOT/name).read_bytes() for name in sources)
    A.save("execution.json",dict(run=C.RUN, used=[f".venv/Scripts/python.exe -B -u tests/glm5/kv_natural_run.py {x}" for x in ("pilot (V1,V2,V3)","pilot-trace","plan","discover","confirm")]+[
        ".venv/Scripts/python.exe -B -u tests/glm5/kv_natural_audit.py", ".venv/Scripts/python.exe -B -u tests/glm5/kv_natural_closeout.py"],
        plan=B.sha("plan.json"),freeze=B.sha("freeze.json"),audit=B.sha("audit.json"),state=a["state"]))
    glm=C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
    claude=C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"
    first=append(P.check_write(glm),TITLE,text.split("\n\n",1)[1])
    second=append(P.check_write(claude),"14B 自然答码中的原生头阻断与恢复索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`。新材料交叉三竞争槽、长距离与生成槽顺序，"
                  "保留动态 Q，冻结后执行原生三头阻断、恢复及匹配随机头对照。完整报告在 `tests/glm5/result/kv_natural_20261115_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[first,second]))
    A.save("manifest.json",dict(files={p.name:A.digest(p) for p in sorted(C.OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},
                                sources={k:A.digest(C.ROOT/k) for k in sorted(sources)}))


if __name__=="__main__":
    C.setup(); close(sys.argv[1:]==["preview"])
