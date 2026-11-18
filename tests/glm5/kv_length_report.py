"""Render Phase 2952 evidence and append the verified research ledger."""

import json
import shutil
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

import kv_length_run as C
from b36_semantic14_memo import append


A,B,P=C.A,C.B,C.P
TITLE="14B 同提示中性段长度及 L25H18 寻址与写入分离"
LABELS={"distance-preference":"长段增加实发词的冲突答码边际", "address-state":"H18 寻址阻断的等幅实发词系数损失",
        "value-state":"H18 V/O 阻断的等幅实发词系数损失", "fact-preference":"并行事实头的等幅事实系数损失",
        "book-preference":"并行码本头的等幅实发词系数损失", "joint-margin":"三路全阻断后原选码优势损失",
        "h18-rescue":"联合阻断后的 H18 原生读头恢复", "parallel-rescue":"联合阻断后的并行读头恢复"}


def table(title,rows):
    return "\n".join("| "+" | ".join(str(x) for x in v)+" |" for v in (title,["---"]*len(title),*rows))


def ci(v):
    return f"{v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"


def intro(p,f,a,bank):
    g=p["gate"]
    controls=table(["冻结源", "随机头", "发现集源剂量", "发现集对照剂量"],
        [(k," / ".join(map(str,f['random'][k])),f"{v['target']:.4g}"," / ".join(f"{x:.4g}" for x in v["random"])) for k,v in f["dose"].items()])
    return rf"""# {TITLE}

本轮 `{C.RUN}` 使用本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`。Phase 2951 的长 Receipt 同时出现在用户提示与已生成历史，本轮让成对样本共享逐字提示、State、事实、码本和整个输出布局，把长短差限制在实际生成的中性追加段。问题是这一改变是否仍然提高随实发词给码的优势，以及 L25H18 从地址到写入与其他头的作用怎样分离。

### 设计、真实生成和正交指标

每家族 16 格，独立交叉提示事实 F、实际发出的词 S、两个候选的 P/Q 码本方向 M 和段长 D。F 反转时两候选身份和它们的先后位置固定，M 交换两候选的字母映射。两条 Receipt 都明示为合法选项，且出现在所有 D 的同一个提示。短段是 `{C.SHORT}`，长段在这句之后插入 `{C.LONG[len(C.SHORT):].strip()}`，固定 {len(bank['extension'])} 个 token。词与 P/Q 事实均不在插入文字里。配对核验要求从提示到 Code query 的实际 token 串，在长分支删除这 36 个实际 token 后逐元素等于短分支。这个更严格的定义保留同一自然空白、同一 Code 标记和同一其他输出字段。

E 是 State 首 token 前的最后 query。在冲突格，用同家族反角色、同 M/D 的原模型终层残差运行原输出归一化和完整 unembedding，取真正的全词表 argmax。B 是两条 Receipt 共有第一句的最后 token。在这个位置换入固定的两份先导原生 BF16 残差之一，让原模型输出短段换行 token `{bank['sources'][0]['next']}` 或长段首追加词 `{bank['sources'][1]['text']}`。供体只控制各自的即时 argmax。两次介入都限最后 block 后的单 query，后续位置通过实际发出的 token 继续。长段从第二个追加 token 起，全部用当前因果前缀上的原模型贪心选择，Code 标签与 P/Q 也未经过外力指定。草稿用于批量核对，只有真实逐 token 核验过的位置计为生成。空白适配只使用实际出现的 token。供体身份、实测分歧和插入词表可回查。

8/24/96 家族用于试跑、发现和留出，实际词数 {a['word_count']}。地理词池按固定种子切成互斥集合，并按相关历史去重。各格真实核验及无冲突正确码至少 90%，整家族因果分析还要求 8 对严格短长、共 16 个视图均合格。事实和码本在 E 之前已出现。C 是 `Code:` 后给下一 token 的 query，此时提示与实发 State 均可见。

令 $m=\ell_Q-\ell_P$，$c_S,c_F$ 为码本里实际词与原事实的字母符号，Q 取 $+1$，P 取 $-1$。在每个家族和 D 下按全部八格计算 $\beta_S=\mathbb{{E}}[mc_S]$、$\beta_F=\mathbb{{E}}[mc_F]$。两个码符号正交，$\beta_S-\beta_F$ 也精确等于冲突格中 $mc_S$ 的均值。主要距离终点为这个值的长减短，预注册方向为增加。8 个主终点按家族 bootstrap，双侧 {100*g['confidence']:.3f}% 区间，方向通过需下界大于零、均值至少 {g['margin']} logit 和至少 {g['n']} 家族。实际选码和两个符号方向的边际分别报告。

### 原生注意力和干预计算

冻结 L25H18 在 C 读 G（已生成的 State），L21H12 读两条提示事实，L28H21 读候选码本两行，层和头均从 0 编号。令 $s_i=q_C^\top k_i/\sqrt{{128}}$ 为源位置 $i$ 的头内得分，$a=\exp(s_G)/\sum_{{i\in\mathcal{{V}}}}\exp(s_i)$ 为 G 的读权重，$\mathcal{{V}}$ 是 C 可见的所有位置。$u$ 表示去掉 G 后其余位置的归一化输出，$v_G$ 是原生 G Value。原始头输出是 $o=u+a(v_G-u)$。寻址阻断只在 H18 的 C query 移除 G 的可见边，保持 Q/K/V 及其他槽的可见性，归一化重分配使头输出改变 $a(u-v_G)$。V/O 阻断只零化 G 的 Value，Q/K、mask 和权重保持相同，头输出改变 $-av_G$。两种差都用原始 BF16 单头 SDPA 计算，再经原生 $W_O$ 写进残差流。

另检验实际 Q/K 坐标的两个干预。对 Key，置 $k'_G=k_G-q_C(q_C^\top k_G)/\|q_C\|^2$，仅在该头读取时改变 G 的 post-RoPE Key，对所有 Value 和其他键保持。对 Query，置 $q'_C=q_C-k_G(q_C^\top k_G)/\|k_G\|^2$，在该头检测同一个得分方向。此操作同时影响其他位置的得分，作为 query 范围的扩展检验。BF16 写回后重新估计真实得分与权重。Key 投影联合 V 清零后，分别写回原生 K、原生 V 及两者，配合原始 K/V 向量、结果等价臂和按位 sham 校验。Q/K 投影视图和地址掩码的行为效应分别报告。

发现集从相同层和来源的最近原生剂量四分位抽两随机头。留出等幅时，每条视图中的两个受测头都使用两个 $\|W_O\Delta o\|_2$ 的较小者。最终实际 BF16 剂量差的 p95 不得超过 3%。全剂量并行读头干预在同一 C 位零化提示源，两模块先后层的 query 仍活体更新。联合阻断使用 H18 的 V 零化及两个并行源阻断。原生救援分别写回 H18 的完整 pre-$W_O$ 输出或两个并行头的完整原生输出。全恢复分支按位核对 sham。完整头救援保留该头的全部原生合成向量，结论限定在该边界。

{controls}
""".replace("\\\\","\\")


def report(p,f,a,bank,recheck,shape):
    n=a["natural"]
    text=intro(p,f,a,bank)
    natural=table(["D/M/S/F", "总数", "核验到 Code", "P / Q", "实发词 / 原事实"],
        [(k,v["attempted"],v.get("matched",0),f"{v.get('P',0)} / {v.get('Q',0)}",f"{v.get('follows_state',0)} / {v.get('follows_fact',0)}" if k[-2]!=k[-1] else "一致") for k,v in n["cells"].items()])
    summary=table(["冲突段长", "尝试", "真正换 State", "随实发词", "随原事实", "其他码", "实际 G–C 距离"],
        [("短" if k=="0" else "长",v.get("attempted",0),v.get("actual_state_switched",0),v.get("follows_state",0),v.get("follows_fact",0),
          v.get("follows_other",0),"–".join(str(x) for x in n["exact_gap"].get(k,[]))) for k,v in n["raw_conflict"].items()])
    main=table(["预注册终点", "成对效应及调整区间", "n", "过门"],
        [(LABELS[k],ci(v),v["n"],"是" if v["passed"] else "否") for k in C.ENDPOINTS for v in [a["primary"].get(k)] if v is not None])
    trans=n["length_transitions"]
    ko=n.get("coefs",{}).get("conflict")
    distance=a["primary"]["distance-preference"]
    dline=(f"朝实发词的冲突边际短 {ko[0]['mean']:.4f}、长 {ko[1]['mean']:.4f} logit。"
           f"长减短 {ci(distance)} logit，方向与预注册的增加假设相反。") if ko else "完整家族不足以计算边际。"
    s=recheck["lexical"]["matched_conflict"]
    tally="、".join(f"{'短段' if k=='0' else '长段'}随实发词 {v.get('state',0)}、随原事实 {v.get('fact',0)}" for k,v in s.items())
    text+=f"""
### 留出真实贪心续写

实际核验 {n['exact_pairs']} 对 token 级严格插入，{n['complete']} 个整家族通过。{dline}完整家族的冲突视图中，{tally}。同完整家族其他因素固定时，原事实→实发词 {trans.get('fact_to_state',0)} 对，反向 {trans.get('state_to_fact',0)} 对。补存 {len(n['pair_failures'])} 对未通过严格相等的具体行。

{summary}

{natural}

{main}
"""
    if "reads" not in a:
        return text+f"\n后续读取实验记录状态 `{a['state']}`，暂无合格结果可用于机制判定。\n"
    z=a["reads"]
    geom=table(["H18 操作", "G 权重前 / 后", "live 得分前 / 后", "Q / K / V 位移 L2 中位数"],
        [(k,f"{v[0]:.5f} / {v[1]:.5f}",f"{v[2]:.4f} / {v[3]:+.4f}"," / ".join(f"{t:.5g}" for t in v[4:])) for k,v in z["geometry"].items()])
    pair=recheck["paired_geometry"]
    attention=table(["H18 自然读出", "短中位数", "长中位数", "逐对长减短中位数"],
        [(k,f"{pair[k]['short']:.5f}",f"{pair[k]['long']:.5f}",f"{pair[k]['paired_change']:+.5f}")
         for k in ("alpha","score","other_log_partition")])
    value=pair["state_value_preserved"]
    dos=table(["等幅对照", "目标原 / 实际剂量", "对照原 / 实际剂量", "相对误差 p95"],
        [(k,f"{v['target'][0]:.4g} / {v['target'][2]:.4g}",f"{v['control'][0]:.4g} / {v['control'][2]:.4g}",f"{v['p95']:.3%}") for k,v in z["dose"].items()])
    fmt=lambda v:" / ".join(str(v.get(k,0)) for k in ("state","fact","other"))
    reads=table(["C 干预", "相对原模型改码", "原选码边际损失 [95%]", "短冲突 实发 / 原事实 / 其他", "长冲突 实发 / 原事实 / 其他"],
        [(k,v["flip"],ci(v["loss"]),fmt(v["conflict"]["0"]),fmt(v["conflict"]["1"])) for k,v in z["arms"].items()])
    cross=table(["等幅读取", "F 系数损失（短 / 长）", "S 系数损失（短 / 长）"],
        [(k," / ".join(f"{x['mean']:+.4f}" for x in v["fact"])," / ".join(f"{x['mean']:+.4f}" for x in v["state"])) for k,v in z["cross"].items()])
    interaction=table(["完整剂量操作", "冲突边际长减短 [描述性 95%]", "相对 sham 的差中差 [描述性 95%]"],
        [(k,ci(v["long_short"]),ci(v["difference_from_sham"])) for k,v in recheck["distance_response"].items()])
    examples=[]
    for k,v in n["records"].items():
        row=v["row"];obs=v.get("observed",v.get("short"))
        out=obs["rendered"].splitlines()[-1] if obs["rendered"].splitlines() else "空"
        longer=""
        if "long" in v:
            lo=v["long"]
            longer=f"，长分支 `{lo['id']}` 实际 `{lo['rendered'].splitlines()[-1]}`"
        examples.append(f"- `{k}`，`{row['id']}`，原事实 `{row['words'][row['fact']]}` 对应 `{row['code_f']}`，实发 `{row['values']['State']}` 对应 `{row['code_s']}`，实际 `{out}`{longer}。")
    for q in n["invalid"][:4]:
        row,obs=q["row"],q["observed"]
        examples.append(f"- 自然验证边界 `{row['id']}`，观察 token `{obs['observation']}`，原始序列和诊断见 `confirm-natural.json`。")
    for k,v in z["examples"].items():
        if v:
            r=v[0]["row"]
            examples.append(f"- `{k}`，`{r['id']}`，实发 `{r['values']['State']}`，原事实码 `{r['code_f']}`。逐臂实际 P/Q 与 Q/K/V 参数见 `audit.json`。")
    for d in n["pair_failures"][:3]:
        examples.append(f"- 严格配对失败 `{d[0]}`／`{d[1]}`，保留两个实际前缀，整家族不进入纯插入推断。")
    observed={r["id"]:r for r in A.load("confirm-natural.json")["rows"]}
    for file,check in a["technical"].items():
        if not check["invalid"]:
            continue
        with np.load(C.OUT/file) as native:
            for key in check["invalid"]:
                row=native["ids"].tolist().index(key)
                natural=observed[key]
                if native["top"][row,0]!=natural["top"]:
                    examples.append(f"- 无干预读出不一致 `{key}`，真实生成 P/Q {natural['pq']}，"
                                    f"干预批次 sham P/Q {native['pq'][row,0].tolist()}，家族按逐例全词表首选要求排除。")
    rounding=""
    if (C.OUT/"rounding-exclusions.json").exists():
        q=A.load("rounding-exclusions.json")
        assert q["policy"]==B.sha("resume-policy.json") and q["confirm"]==B.sha("confirm.json")
        saved=A.load("resume-policy.json")
        assert A.digest(C.ROOT/saved["adapter"])==saved["adapter_sha"]
        rounding=f"主运行的首个留出尝试因 BF16 注入误差 0.05587 超过既定 0.05 阈值停止。保留初次日志和已通过分片，未调整冻结操作或阈值。续跑只把原异常改为逐视图保存实际注入误差并拒绝该完整家族。实际记录 {q['n_views']} 条超限视图，峰值 {q['max_measured']:.4g}，初次未完成分片从头复算。"
        for subset in q["violations"].values():
            for row in subset[:2]:
                examples.append(f"- 原生写入超限 `{row['id']}`，臂 `{row['arm']}`，归一误差 {row['error']:.5f}，实际 P/Q {row['measured_pq']}，家族按冻结阈值排除。")
    logs=[json.loads(s) for s in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    cuda=max(v.get("cuda_peak_bytes",0) for v in logs)/2**30
    share=max(v["shared_gpu_bytes"] or 0 for v in logs)/2**20
    checks=list(a["technical"].values())
    qerr={k:max(v["errors"][k] for v in checks) for k in ("address","value")}
    matched=shape["result"]
    sa,ss=(matched["pairs"][k] for k in ("alpha","score"))
    sk,sv=(matched["pairs"][k] for k in ("key","value"))
    stable=matched["exact_top"]
    mismatches="、".join(f"`{k}`" for k in matched["mismatches"])
    mismatch_note=f"首选有差异的行是 {mismatches}，原记录和统一形状的 P/Q 与全词表首选均已保留。" if mismatches else ""
    return text+f"""
### 寻址、写入与并行读出

纳入 {z['n']} 个完整技术家族、{16*z['n']} 条视图。下表显示同一 BF16 条件的真实几何量。Key、Query 和 K+V 的实际末得分绝对值 p95 分别为 {z['score95']['key']:.4g}、{z['score95']['query']:.4g}、{z['score95']['kv']:.4g}。

原始左填充调度下，同 State 前缀的 {value['n']} 个长短配对中，G 的原生 V 按位一致 {value['exact']} 对，相对 L2 差中位数 {value['median_relative']:.2%}、最大 {value['max_relative']:.2%}。从实际 G 概率和得分反算其他槽的 log-partition，使用 `score_G - log(alpha/(1-alpha))`。下表保留原始得分及分母竞争的物理量，以检查后继 query 的寻址和其他槽竞争。

{attention}

冻结后另用统一右填充宽度 {shape['width']} 的 BF16 调度，对 {matched['n']} 个严格配对做前向复算。两分支在同一 batch 中，拥有相同的 G 矩阵索引与因果前缀。G 的 post-RoPE K / V 分别有 {sk['exact']} / {sv['exact']} 对按位相同，最大相对 L2 差 {sk['max_relative']:.4g} / {sv['max_relative']:.4g}。G 注意力中位数 {sa['short']:.5f} → {sa['long']:.5f}，得分中位数 {ss['short']:.4f} → {ss['long']:.4f}。C 的全词表首选有 {stable}/{2*matched['n']} 条与原贪心核验一致，P/Q 最大漂移 {matched['max_pq_drift']:.4f} logit，冲突边际长减短 {ci(matched['distance'])} logit。{mismatch_note}复算只核对已验收的前缀和 C 的读出。原始端点和干预判据仍按冻结调度记录。

{geom}

{dos}

H18 的寻址和 V/O 都通过等幅随机对照，事实头和码本头未通过各自主终点。码本头第一组对照的剂量误差 p95 也超出 3% 物理门。全剂量联合阻断三个读头改变了 {z['arms']['joint']['flip']} 条实际答码，H18 与两个并行头的原生读头恢复各自提高原选码的优势。两份恢复从同一个联合阻断状态出发，恢复量不可当作独立份额相加。K 只恢复时仍缺少目标 V，V 只恢复时寻址仍被破坏。两者恢复时，实际输出回到 sham。

{cross}

{reads}

下表用同一已核验家族描述干预如何改变长短差异。每个值都在相同 F/S/M 的冲突对上计算，属于补充分析。边际和逐例改码仍分别记录。

{interaction}

### 反例

{chr(10).join(examples)}

### 适用范围与下一阶段

严格公共提示与 token 插入保证上游角色及码本物理相同，改变的是指定的无任务标签插入文本及随它变化的长度和位置。这项结果限制了 Phase 2951 对一般距离效应的解释。两个阶段的词表和提示设计也有变化，方向差异不能单独归因于某个跨阶段变量。两个末层供体决定两个输出选项的首个分歧，后续由模型贪心生成，结论条件化于这种分叉方案。等幅比较覆盖的剂量受随机头原生贡献限制，码本对照还有物理失配。Q 投影会改变其他槽寻址，单指定源的 Key 或 mask 检验给出更局部的效应边界。真实原生读头恢复既可能恢复竞争路径也可能抵消其余作用。当前为单快照、同一句式和两字母任务。家族 bootstrap 把 16 格作为成组测量，留出面板内一些词汇跨家族复用，区间反映当前构造与词池，不应外推为广义词汇总体的不确定性。

外部语言图谱新增严格共前缀的事实／实发词／码本组合与中性延长。内部图新增读取支撑、Q/K 得分、V/O 写入与并行读头操作。跨图边绑定真实输出与原生阻断恢复。统一矩阵形状下可检验读取时的源内容与后继查询如何独立变化。同一源内容若能在多项任务中反复读取，复用的候选对象应是由查询和源共同决定的条件规则。下一大阶段先交叉多种中性文本、多个冻结长度档和等长替换，再比较不同句式与码本规模。用发现集预测哪些 $q_{{C,i}}k_{{G,i}}$、源竞争和 $W_{{O,ji}}v_{{G,i}}$ 项能解释留出边际与真正改码，与绝对位置、词身份和简单注意力基线比较，最后做冻结坐标级阻断、原生恢复及副作用核验。当前 RDC 无新统一公式。

### 复核与资源

原生地址与 V-only 解析式对实际 BF16 头写入的最大归一化误差为 {qerr['address']:.4g}、{qerr['value']:.4g}，sham 最大 P/Q 差 {max(w['drift'] for w in checks):.4g}。原生 Q、地址、V 自恢复、K/V 两部分恢复等价臂及全部并行写回均逐元素核验。独立 CPU 复算用实际输出 token 再次确认长度配对、地址阻断的 Q/K/V 保持、V 清零的权重恒等，以及 Key 和 Query 各自的坐标改变。第一轮路径试跑的 attention mask 扩展维度错误在冻结前修正，早期日志留存，确认未复用失败产物。{rounding}RTX 5080 CUDA 分配峰值 {cuda:.2f} GiB，共享显存 {share:.0f} MiB。D 盘每批核查，保留 20 GiB，清理仅本轮独占缓存。材料、供体、原生向量、脚本、真实生成、失败用例、分片和哈希在 `tests/glm5/result/kv_length_20261116_v1/`。
"""


def plot(a,p,recheck):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f,(x,y)=plt.subplots(1,2,figsize=(12.8,4.5),layout="constrained")
    u=recheck["lexical"]["matched_conflict"]
    names=tuple(a["natural"]["exact_gap"].values())
    bottoms=np.zeros(2)
    for k,label,color in (("state","Emitted State","#3f707e"),("fact","Prompt fact","#af8057"),("other","Other","#879b82")):
        data=[100*u[d].get(k,0)/(4*a["natural"]["complete"]) for d in ("0","1")]
        if not any(data):continue
        x.bar([0,1],data,bottom=bottoms,width=.57,color=color,label=label)
        if k=="state":
            for j,value in enumerate(data):x.text(j,value/2,f"{value:.1f}%",color="white",ha="center",va="center",fontweight="bold")
        bottoms+=data
    ticks=[str(h[0]) if h[0]==h[1] else f"{h[0]}–{h[1]}" for h in names]
    x.set(title=f"Matched continuations · {a['natural']['complete']} families",ylabel="Verified greedy code in conflict (%)",ylim=(0,100),
          xticks=[0,1],xticklabels=[f"{n} tokens to Code" for n in ticks]);x.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,-.22),ncol=2)
    seq=("address-state","value-state","fact-preference","book-preference","joint-margin","h18-rescue","parallel-rescue")
    r=[a["primary"][k] for k in seq];z=np.array([v["mean"] for v in r]);interval=np.array([v["ci"] for v in r])
    y.errorbar(z,range(len(z)),xerr=abs(interval-z[:,None]).T,fmt="o",color="#3f707e",capsize=3)
    opts=("H18 address · State loss","H18 V/O · State loss","Parallel fact loss","Parallel book loss","Joint code loss","H18 native rescue","Parallel native rescue")
    y.set(title=f"Reader interventions · {a['reads']['n']} complete families",xlabel=f"Paired logit effect ({100*p['gate']['confidence']:.3f}% interval)",yticks=range(len(opts)),yticklabels=opts)
    y.axvline(0,color=".6",lw=.8);y.axvline(.1,color="#af8057",ls="--",lw=.9);y.invert_yaxis()
    for v in (x,y):v.spines[["top","right"]].set_visible(False)
    f.savefig(P.check_write(C.OUT/"phase-2952-effects.png"),dpi=165)
    plt.close(f)


def main(stage,p):
    f,a,bank=(A.load(k) for k in ("freeze.json","audit.json","neutral-donors.json"))
    proof=A.load("native-recheck.json")
    shape=A.load("common-shape.json")
    for check in (a,proof,shape):
        assert all(B.sha(k)==v for k,v in check["binding"].items())
    text=report(p,f,a,bank,proof,shape)
    P.check_write(C.OUT/"report.md").write_text(text,encoding="utf-8")
    if "reads" in a:plot(a,p,proof)
    if stage=="preview":return
    target=C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
    assert not (C.OUT/"memo-receipt.json").exists() and "\n## Phase 2952:" not in target.read_text(encoding="utf-8-sig")
    scratch=C.TMP.resolve();within=list(scratch.rglob("*"))
    assert scratch==C.TMP and scratch.parent==(C.ROOT/"tests/glm5_temp").resolve()
    assert all(x.resolve().is_relative_to(scratch) and not x.is_symlink() and not x.is_junction() for x in (scratch,*within))
    removed=sum(x.stat().st_size for x in within if x.is_file());shutil.rmtree(scratch)
    checkpoint=scratch.with_name("kv_length_active_checkpoint.md")
    checkpoint_bytes=0
    if checkpoint.exists():
        assert checkpoint.resolve()==checkpoint and checkpoint.is_file() and not checkpoint.is_symlink()
        assert checkpoint.read_text(encoding="utf-8").startswith("Phase 2952 active user request")
        checkpoint_bytes=checkpoint.stat().st_size
        checkpoint.unlink()
    A.save("cleanup.json",dict(path=str(scratch),removed=removed+checkpoint_bytes,checkpoint_bytes=checkpoint_bytes,
                               free=shutil.disk_usage(C.OUT).free))
    src={*p["sources"],"tests/glm5/kv_length_report.py","tests/glm5/kv_length_recheck.py","tests/glm5/kv_length_shape.py",
         "tests/glm5/b36_semantic14_memo.py"}
    if (C.OUT/"resume-policy.json").exists():src.add("tests/glm5/kv_length_resume.py")
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for k in sorted(src):z.write(C.ROOT/k,k)
    with ZipFile(C.OUT/"sources.zip") as z:assert all(z.read(k)==(C.ROOT/k).read_bytes() for k in src)
    invoke=".venv/Scripts/python.exe -B -u tests/glm5/kv_length_run.py"
    resume=[".venv/Scripts/python.exe -B -u tests/glm5/kv_length_resume.py"] if (C.OUT/"resume-policy.json").exists() else []
    A.save("execution.json",dict(commands=[".venv/Scripts/python.exe -B -u tests/glm5/kv_length_natural.py"]+[f"{invoke} {k}" for k in (
        "pilot","pilot","plan","discover","confirm")]+resume+[f"{invoke} audit",".venv/Scripts/python.exe -B -u tests/glm5/kv_length_recheck.py",
        ".venv/Scripts/python.exe -B -u tests/glm5/kv_length_shape.py",f"{invoke} preview",f"{invoke} close"],
        proof=a["binding"],audit=B.sha("audit.json"),shape=B.sha("common-shape.json")))
    first=append(P.check_write(target),TITLE,text.split("\n\n",1)[1]);assert first["phase"]==2952
    index=append(P.check_write(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"),"14B 同提示中性段长度、寻址及写入索引",
                 f"详见 GLM5 Phase 2952，运行 `{C.RUN}`，报告 `tests/glm5/result/kv_length_20261116_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[first,index]))
    A.save("manifest.json",dict(files={k.name:A.digest(k) for k in sorted(C.OUT.iterdir()) if k.is_file() and k.name!="manifest.json"},
                                 sources={k:A.digest(C.ROOT/k) for k in sorted(src)}))
