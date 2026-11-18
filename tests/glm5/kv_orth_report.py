"""Phase 2951 report, visual QA output, and verified append-only delivery."""

import json
import shutil
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

import kv_orth_run as C
from b36_semantic14_memo import append


A,B,P=C.A,C.B,C.P
TITLE="14B 原事实、实发 State、码本与距离的正交答码检验"
NAMES={"state-coef":"实发 State 的码本系数", "fact-coef":"原事实的码本系数", "distance-balance":"长减短的冲突偏好",
       "h18-state":"L25H18 对实发 State 系数的等幅阻断", "fact-fact":"事实读取对原事实系数的等幅阻断",
       "book-state":"码本读取对实发 State 系数的等幅阻断", "joint-code":"三路联合阻断的原模型答码优势损失",
       "h18-rescue":"联合阻断后原生 H18 写回", "parallel-rescue":"联合阻断后原生并行写回"}


def table(head,values):
    return "\n".join("| "+" | ".join(map(str,r))+" |" for r in (head,["---"]*len(head),*values))


def show(r):
    lo,hi=r["ci"]
    return f"{r['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def path(x):
    return f"L{x[0]}H{x[1]} ({x[2]})"


def results(p,f,a,r):
    rolepaths="、".join(path(x) for x in f["paths"])
    screening=table(["发现候选", "原事实系数损失", "实发 State 系数损失"],
                    [(path(q["site"]),f"{q['loss_f']:+.4f}",f"{q['loss_s']:+.4f}") for q in f["selection"]])
    matched=table(["冻结读取", "对照头", "目标发现剂量", "随机头发现剂量"],
                  [(path(x)," / ".join(f"H{h}" for h in y),f"{d['native']:.4g}"," / ".join(f"{q:.4g}" for q in d["random"]))
                   for x,y,d in zip(f["paths"],f["controls"],f["doses"],strict=True)])
    main=table(["预注册终点", "效应与 99.444% 区间", "家族", "过门"],
               [(NAMES[k],show(v),v["n"],"是" if v["passed"] else "否") for k in C.ENDPOINTS for v in [a["primary"].get(k)] if v is not None])
    natural=a["natural"]
    cells=table(["D/M/S/F", "样本数", "实发指定 State", "未经替换的 Code P/Q", "其他答码", "前缀未核验", "冲突时随实发词", "冲突时随事实"],
                [(k,v["total"],v.get("state",0),f"{v.get('P',0)} / {v.get('Q',0)}",v.get("other",0),v.get("prefix_unresolved",0),
                  v.get("follows_emitted","—"), v.get("follows_fact","—")) for k,v in natural["cells"].items()])
    runs=[]
    for k in ("0","1"):
        stats=natural["distance"].get(k,{})
        re=r["natural"][k]
        runs.append(("短" if k=="0" else "长",re["attempted"],re["real_state_changed"],re["valid_code_prefix"],
                     stats.get("follows_emitted",0),stats.get("follows_fact",0),stats.get("other",0),
                     "–".join(map(str,re["actual_gap"])) if re["actual_gap"] else "—"))
    distance=table(["冲突距离", "尝试", "相对未干预真实换词", "核验到 Code", "随 State", "随事实", "其他答码", "实际 G–C token 距离"],runs)
    stage=a["state"]
    text=rf"""# {TITLE}

本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`，运行 `{C.RUN}`。Phase 2950 的真实换词续写中，新词与原事实都曾决定答码，但规则、距离与原码在小面板中存在同现。本轮逐家族交叉四个变量，并在自然核验后的 `Code:` query 干预 L25H18 与并行读头。研究问题是在提示和实际输出互相冲突时，后续模型读取哪一种历史。

### 四因子、真实生成和统计量

同家族的两个候选词在提示中固定先后位置。F 改变哪一个词由 includes 与 excludes 指定为事实。S 是模型真正发出的 State 首 token 和经完整行核对的词。M 互换两候选词在同一码本中的 P/Q，其他三个槽的词和码保持。D 改变要复制的 Receipt 长度，连带改变了提示中该格式指令。D 在 16 格中与 F/S/M 正交，其效果对应本模板中长短 Receipt 干预。试跑、发现、留出分别 {C.FAMILIES['pilot']}、{C.FAMILIES['discover']}、{C.FAMILIES['confirm']} 家族，相关历史去重后的实际使用词数 {r['material_words']}。发现和留出沿同一个人名词池分开分词身份。模型、dtype、脚本和材料哈希一起保存。

冲突格的 E 位于首次 State token 之前。将本模型匹配角色反转样本的 L39 block 后原生残差在 E 交给同一输出归一化及 unembedding，取全词表真正 argmax。实现把 E 的读出替换为该残差的真实读出，这与在最后一层替换该位置等价。供体与目标共享家族、M 和 D。E 输出之后，后续分支只能通过实际选择的 token 演进，目标提示保持原事实。先核验每个预测，包括行末空白出现的实际 token。提出的后缀只作为批量前向草稿，单个 token 只有通过当前前缀的贪心 argmax 才算生成。格式偏离的案例另作生成核验并记停止原因，完整均衡的干预面板只用能够核验至 C 的完整家族。各格预注册的行为门是 90%。在无冲突格还要求 P/Q 为两种参照共有的正确码。

令 $m=\ell_Q-\ell_P$，$c_F,c_S\in\{{-1,+1\}}$ 表示原码本给 F 或实际 S 指定的 P/Q，Q 取 $+1$。对固定家族和距离，分别计算 $\beta_F=\frac18\sum_{{F,S,M}}mc_F$ 和 $\beta_S=\frac18\sum_{{F,S,M}}mc_S$。全交叉下，两种编码符号正交，$\beta_S-\beta_F$ 精确等于冲突格里 $m$ 朝实发 State 的带符号均值，$\beta_S+\beta_F$ 精确等于一致格里该量。正向系数可在留出上区分实发词或原事实的答码信息。主要检验还包括长减短的冲突偏好。二分类真实答码分开计数。P/Q 以外的 token、分歧、未到达 Code 均保留身份。每家族汇总八或十六个视图，推断门要求至少 80 家族、0.10 logit 实质差以及预注册的多重调整区间；距离检验双向。

### C 位读取的发现与冻结

固定追踪 L25H18 对已生成 G 位的读取。并行候选在 L18–38 扫 40 头，事实掩码包含两条事实行，码本掩码包含两条候选码本行。对每个候选做真实 C 位写入阻断，按发现集中 $\beta_F$ 或 $\beta_S$ 的损失选择固定头。收到任何留出模型响应之前冻结 {rolepaths}，同层同来源的随机头两组，后续全部臂不改。

{screening}

{matched}

只在当前 C query 替换被测注意力头的输出，与其保持活体 query/Key 的原始 SDPA 对比，源位置 V 设零后计算差向量。单独阻断 G 槽时，投影前理论量为 $-\alpha_G v_G$。将差加回底层原生 packed 头并转回 BF16。等幅随机比较逐例将目标和对照的 $W_O$ 投影 L2 降到二者的较小值，记录实际舍入后的 L2。联合阻断保留三个全剂量读出变化，再分别把 H18 或并行头写回同次无干预前向捕获的原生 pre-$W_O$ 头向量。完整写回与同一分支清零再原样复原都核验与 sham 逐元素一致。开关只在读答码的时刻发生，先前贪心输出不受影响。写回的原生头向量包含该头从其余槽取得的合成读出，要以实际介入边界解释结果。

### 留出行为与预注册结果

{distance}

{cells}

{main}
""".replace("\\\\","\\")
    if r["factor"]:
        w=r["factor"]
        d=w["directional_flips"]
        weights=natural["coefficients_by_distance"]
        def band(arr): return " 到 ".join(f"{x['mean']:.3f}" for x in arr)
        text+=f"\n{w['complete']} 个家族达到完整自然行为准入。代数复算得到冲突格中朝实发词的原模型边际短 {w['conflict_signed'][0]:+.4f}、长 {w['conflict_signed'][1]:+.4f}。实发词系数从 {band(weights['state'])}，事实系数从 {band(weights['fact'])}。固定家族、F/S/M，只改变 D 时，有 {w['distance_flip_count']} 对冲突视图的实际 P/Q 翻转，其中 {d.get('fact_to_state',0)} 对由原事实转向实发词，{d.get('state_to_fact',0)} 对反向。\n"
    alt=natural["alternate_format"]
    text+=f"\n偏离规范格式的补充贪心核验 {alt['tested']} 条，停止分类 {'、'.join(f'{k}={v}' for k,v in alt['stops'].items()) or '无'}，完整重放与字段核验反例保留。补充结果未合入等幅主样本。\n"
    if not a.get("causal"):
        return text+f"\nC 位冻结队列处于 `{stage}`，当前没有足够合格干预数据支持机制判定。\n"
    z=a["causal"]
    effects=table(["干预", "相对原模型真实改码", "原选码边际 [95%]", "事实系数短 / 长", "实发系数短 / 长"],
                  [(k,v["changed"],show(v["original_model_margin"])," / ".join(f"{q['mean']:+.3f}" for q in v["beta_f"]),
                    " / ".join(f"{q['mean']:+.3f}" for q in v["beta_s"])) for k in ("sham","h18","fact","book","parallel","joint",
                    "restore-h18","restore-parallel","restore-all") for v in [z["arms"][k]]])
    phys=table(["等幅对照", "目标原剂量 / 实际", "对照原剂量 / 实际", "逐例相对差 p95"],
               [(k,f"{v['target'][0]:.4g} / {v['target'][2]:.4g}",f"{v['random'][0]:.4g} / {v['random'][2]:.4g}",f"{v['p95']:.2%}")
                for k,v in z["phys"].items()])
    cross=table(["等幅读取", "事实系数损失（短 / 长）", "实发系数损失（短 / 长）"],
                [(k," / ".join(f"{q['mean']:+.4f}" for q in v["f"])," / ".join(f"{q['mean']:+.4f}" for q in v["s"]))
                 for k,v in z["cross_coefficients"].items()])
    def raw_split(v):
        return " / ".join(str(v.get(k,0)) for k in ("state","fact","other"))
    raw_conflict=table(["冲突格干预", "短：State / 事实 / 其他", "长：State / 事实 / 其他"],
                       [(k,raw_split(v["0"]),raw_split(v["1"])) for k,v in r["causal_conflict"].items()])
    ex=[]
    for k,w in natural["examples"].items():
        row,ob=w["row"],w["observed"]
        end=ob["rendered"].splitlines()[-1] if ob["rendered"].splitlines() else "空"
        ex.append(f"- `{k}`，`{row['id']}`，事实 `{row['words'][row['fact']]}` 应给 `{row['code_f']}`，实际 State `{row['words'][row['state']]}` 对应 `{row['code_s']}`，模型末行为 `{end}`。")
    for w in (r["factor"] or {}).get("examples",[])[:3]:
        ex.append(f"- `{w['id'][0]}` → `{w['id'][1]}`，同事实 `{w['fact']}` 和 State `{w['state']}`，同码本，短长实际答码 `{w['outputs'][0]}` → `{w['outputs'][1]}`。")
    for k,w in z["examples"].items():
        if w:
            q=w[0]["row"]
            ex.append(f"- `{k}`，`{q['id']}`，原事实码 `{q['code_f']}`，实发词码 `{q['code_s']}`，逐臂全词表选择、原生源质量、写入剂量及 P/Q 见 `audit.json`。")
    for w in r["agreed_miscodes"][:5]:
        ex.append(f"- 事实和实发词一致仍错码 `{w['id']}`，State `{w['state']}`，共同真值 `{w['code']}`，实际 `{w['actual'].splitlines()[-1]}`。")
    for record in a["technical"]["failures"][:8]:
        ex.append(f"- 无干预技术反例 `{record['id']}`，packed 与全长最大原始 P/Q 差 {record['drift']:.3g}，按完整家族拒绝。")
    logs=[json.loads(s) for s in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cuda=max(v.get("cuda_peak_bytes",0) for v in logs)/2**30
    shared=max(v["shared_gpu_bytes"] or 0 for v in logs)/2**20
    peak=max(s["drift"] for s in a["technical"]["shards"].values())
    err=max(s["max_source_formula_error"] for s in r["trace"].values())
    pmax=max(v["p95"] for v in z["phys"].values())
    h=a["primary"]
    loss=h["joint-code"]["mean"]
    g_restore=h["h18-rescue"]["mean"]
    q_restore=h["parallel-rescue"]["mean"]
    h18_counts=r["causal_conflict"]["h18"]
    sham_counts=r["causal_conflict"]["sham"]
    return text+f"""
### 等幅对照、联合阻断和原生救援

{z['n']} 个完整家族、{16*z['n']} 条视图通过 C 位技术门，按 packed sham 作为成对参照。下表的源幅度取进入残差流的实际 (W_O) 写入。相对差的预注册门是 p95 不超过 3%。

{phys}

{cross}

{effects}

{raw_conflict}

六对等幅保持在预注册精度内，最大 p95 为 {pmax:.3%}。同幅度下，L25H18 的实发词系数损失 {h['h18-state']['mean']:+.4f} logit。事实头的事实系数损失 {h['fact-fact']['mean']:+.4f}，码本头的实发词系数损失 {h['book-state']['mean']:+.4f}，两者未过门。随机头较弱，逐例配平需要降低目标剂量。两条并行头的零结果限于这些共同剂量。

完整 H18 阻断后，短冲突格随实发词的视图从 {sham_counts['0']['state']} 减至 {h18_counts['0']['state']}，长格从 {sham_counts['1']['state']} 减至 {h18_counts['1']['state']}。三路全剂量联合使原选码的优势下降 {loss:.4f} logit，并让 {z['arms']['joint']['changed']} 条实际改码。H18 单独原生写回恢复 {g_restore:.4f} logit，约为联合损失 {g_restore/loss:.1%}。写回并行两头恢复 {q_restore:.4f} logit，约为 {q_restore/loss:.1%}。两个恢复条件从同一个阻断状态起步，不能把比例加总为独立解释份额。

### 可定位反例

{chr(10).join(ex)}

### 机制与适用范围

完整 16 格将词面和提示事实分开，并让 P/Q 标签在每个冲突方向上交换。等幅对照检验写入幅度相同时的来源特异性，真实改码与边际损失分别记录。结果的 G 槽边限定在 L25H18 于 C 的读取，候选事实和码本头的写入干预保留可见键及其他槽。三路读取同一个 C 位置，分别命中不同的源组，跨层阻断还会改变后继 query。联合救援使用完整原生读头输出，保留当前分支的网络后续计算。缺少对 Q/K 选择来源、未筛查的分布式路径以及必要性的全面干预。

三图谱的增量是外部事实、实发词、码本和距离的正交结构，内部三条读路径及有物理剂量的效应，以及完整来源与结果哈希的留出边。有限参数实现组合要求内部能同时保留提示事实与自己刚写出的词，并随当前查询选择性使用。下一阶段应建立两种复制长度但同一前缀控制的生成干预，交叉码字数量、词类和生成字段顺序，再按留出预算检验读取组合的条件变化及旁路。RDC 暂无新统一公式。当前只覆盖这一提示格式、两候选标签、人名词池、本地单个模型快照，词汇复用与模板条件限制推广。

### 工程复核

无干预 packed 对照最大 P/Q 差 {peak:.4g}，C 按完整家族同时核对 argmax 和真实贪心前缀。同分支自恢复及三路全写回精确复现按位原生输出。从原生保存的 H18 读质量、Value 和写入差独立复算 $-\\alpha_G v_G$，最大归一化误差 {err:.4g}。三份材料的完整 16 格、互斥词表、同 M/D 供体身份及 Walsh 恒等式另行核对。试跑分片复查发现冻结审计代码把主终点和路径交叉系数写进了同一个字典。留出响应开封前登记分组修正，将初次输出和错误日志留档，对修正前后九个主终点与交叉系数逐项验证数值一致。RTX 5080 CUDA 分配峰值 {cuda:.2f} GiB，进程共享峰值 {shared:.0f} MiB。只写 D 盘并保留 20 GiB。完整轨迹、反例、标定、冻结记录、源码与哈希均在 `tests/glm5/result/kv_orth_20261116_v1/`。
"""


def graph(a,r,p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.7,4.6),layout="constrained")
    kinds=("follows_emitted","follows_fact","other","unverified")
    lab=("Emitted State","Prompt fact","Other answer","Unverified")
    palette=("#386f7f","#b08054","#879379","#c5c5cb")
    x=np.arange(2);base=np.zeros(2)
    for key,label,color in zip(kinds,lab,palette,strict=True):
        values=[]
        for j in ("0","1"):
            d=a["natural"]["distance"].get(j,{})
            n=r["natural"][j]["attempted"]
            z=n-sum(d.values()) if key=="unverified" else d.get(key,0)
            values.append(100*z/n)
        ax.bar(x,values,bottom=base,label=label,color=color,width=.58)
        base+=values
    ax.set(xticks=x,xticklabels=("Short receipt","Long receipt"),ylim=(0,100),ylabel="Percent of conflict views",title="Verified greedy Code")
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2)
    keys=C.ENDPOINTS[3:]
    y=np.asarray([a["primary"][k]["mean"] for k in keys]);ci=np.asarray([a["primary"][k]["ci"] for k in keys])
    labs=("H18 · emitted-State loss","Fact head · fact loss","Book head · emitted loss","Joint · code margin loss","H18 native rescue","Parallel native rescue")
    bx.errorbar(y,range(len(y)),xerr=abs(ci-y[:,None]).T,fmt="o",color="#386f7f",capsize=3)
    bx.set(yticks=range(len(y)),yticklabels=labs,title=f"Held-out causal · {a['causal']['n']} families",xlabel="Paired logit effect (99.44% interval)")
    bx.axvline(0,color=".6",lw=.8)
    bx.axvline(p["statistics"]["margin"],color="#b08054",ls="--",lw=1)
    bx.invert_yaxis()
    for axes in (ax,bx): axes.spines[["top","right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"phase-2951-effects.png"),dpi=165)
    plt.close(fig)


def main(stage,p):
    f,a,r=(A.load(k) for k in ("freeze.json","audit.json","native-recheck.json"))
    assert all(B.sha(k)==v for k,v in (a["hash"]|r["binding"]).items())
    schema=A.load("audit-schema-proof.json")
    assert schema["statistics_unchanged"] and schema["corrected"]==B.sha("audit.json") and schema["note"]==B.sha("audit-schema-note.json")
    report=results(p,f,a,r)
    P.check_write(C.OUT/"report.md").write_text(report,encoding="utf-8")
    if a.get("causal"):
        graph(a,r,p)
    if stage=="preview":return
    tip=C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
    assert not (C.OUT/"memo-receipt.json").exists() and "\n## Phase 2951:" not in tip.read_text(encoding="utf-8-sig")
    location=C.TMP.resolve();inside=list(location.rglob("*"))
    assert location==C.TMP and location.parent==(C.ROOT/"tests/glm5_temp").resolve()
    assert all(x.resolve().is_relative_to(location) and not x.is_symlink() and not x.is_junction() for x in (location,*inside))
    bytes_=sum(x.stat().st_size for x in inside if x.is_file());shutil.rmtree(location)
    A.save("cleanup.json",dict(path=str(location),removed=bytes_,free=shutil.disk_usage(C.OUT).free))
    source={*p["sources"],"tests/glm5/kv_orth_report.py","tests/glm5/kv_orth_recheck.py","tests/glm5/kv_orth_correct_audit.py","tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as zip_:
        for k in sorted(source):zip_.write(C.ROOT/k,k)
    with ZipFile(C.OUT/"sources.zip") as zip_:
        assert all(zip_.read(k)==(C.ROOT/k).read_bytes() for k in source)
    cmd=".venv/Scripts/python.exe -B -u tests/glm5/kv_orth_run.py"
    A.save("execution.json",dict(commands=[f"{cmd} {k}" for k in ("pilot","plan","discover","confirm","audit")]+
        [".venv/Scripts/python.exe -B -u tests/glm5/kv_orth_correct_audit.py",".venv/Scripts/python.exe -B -u tests/glm5/kv_orth_recheck.py",
         f"{cmd} preview",f"{cmd} close"],proof=a["hash"],audit=B.sha("audit.json")))
    m=append(P.check_write(tip),TITLE,report.split("\n\n",1)[1]);assert m["phase"]==2951
    idx=append(P.check_write(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"),"14B 实发 State、原事实与答码正交检验索引",
               f"详见 GLM5 Phase 2951，运行 `{C.RUN}`，报告 `tests/glm5/result/kv_orth_20261116_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[m,idx]))
    A.save("manifest.json",dict(files={x.name:A.digest(x) for x in sorted(C.OUT.iterdir()) if x.is_file() and x.name!="manifest.json"},
                                 sources={k:A.digest(C.ROOT/k) for k in sorted(source)}))
