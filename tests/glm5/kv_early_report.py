"""Render Phase 2948 and append verified byte-prefix research records."""

import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from b36_semantic14_memo import append
import kv_early_run as C


A,B,P=C.A,C.B,C.P
TITLE="14B 早层事实写入、L25 自产 State K/V 与末端联合救援"
LABELS={"e-state":"首次 State 事实阻断超过同层随机头","kv-physical":"L25 自产槽 K/V 距离超过同层随机头",
        "e-code":"早层事实阻断的答码损失超过同层随机头","relay-abs":"H18 写回原生 G 键和值的净救援",
        "relay-specific":"H18 写回比同 GQA 组随机头多救回的量","reverse":"把阻断后的 G K/V 反写至完好 H18 的损失",
        "c-vblock":"末端 H18 读槽阻断超过同位置随机头","c-joint":"末端三处联合超出同层随机组合",
        "ec-joint":"早层和末端均阻断后的联合净救援","ec-over-partial":"联合相对逐例较强部分救援的增量"}


def interval(x, percent=False):
    f="+.2%" if percent else "+.4f"
    return f"{x['mean']:{f}} [{x['ci'][0]:{f}}, {x['ci'][1]:{f}}]"


def fmt_frac(x):
    return "—" if x is None else f"{x:.1%}"


def report(a,p,f):
    names="、".join(f"L{l}H{h}" for l,h in f["early"])
    ctrl=["、".join(f"L{l}H{h}" for l,h in h0) for h0 in f["early_random"]]
    base=A.load("confirm-behavior.json")
    ready=sum(x["correct"] for x in base["cells"].values())
    natural=a["natural"]
    head=f"""# {TITLE}

本轮 `{C.RUN}` 使用本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`。Phase 2947 的早期候选都晚于 L28，在 State 词面保持时无法改变 L25。本轮先确定能改变 L25 记忆的受检早层头，再检验该物理变化能否由 L25H18 写入后续答码并传至 L27、L28H20 和 L37。

### 行为、材料与介入位置

新词表与 Phase 2947 及此前相关材料排斥。32 个发现家族，96 个留出家族，每家族八个角色互换、输出顺序、短长距离视图。角色互换修改提示里的事实选择，保持同家族候选词及码本，正确 State 及 P/Q 同向切换。留出自然生成 {ready}/768，逐 token 用实际全词表贪心结果核验到 `Code:`，只接受模型自身选择的行末空白。角色互换的两侧均正确的配对 {natural['both_correct']}/{natural['attempted']}，其中完整自然生成 State 与 P/Q 都换向 {natural['natural_state_and_code_flips']} 对。这是改变事实的自然行为参照，单独保留与模型内部受控干预的证据。

E 是首次生成 State 前一 query，G 是已生成 State token 的位置，C 是最终 `Code:` query。零起算 block，E 在第 $l$ 层的写入先改变 $x_{{l+1}}(E)$，此变化通过下一层可进入 G 的残差，最早成为 $x_{{l+2}}(G)$。L25 的 G K/V 取 $x_{{25}}(G)$，所以本次搜索限 $l\\le23$。若 E 实际改选，下一 token 的 embedding 也变化，时序传播另记。原始同一词面时，路径依赖实测 K/V 与后续动态 Q。

E 只在目标 query 和头将提示事实词 V 置零，键与分母保留。另做原生写回、同头提示拒绝词阻断和同层随机头两个对照。C 只给一个 query 头切换 G 的 K 或 V，不修改 GQA 兄弟头及历史本身。L25H18 使用 GQA 组 3。仅 K 恢复会按活体 Q 重算注意力权重，仅 V 恢复保留路由。H18 的 G 键和值同时恢复时，从无干预完整分支读取原生 BF16 源向量，保持受干预分支 Q。反向臂把受早层阻断的 G K/V 写到原本完好的 H18，只改变 C 的该次读取。对同 GQA 组随机头 {f['relay_controls']} 做同一原生源恢复，从而比较共享 K/V 上不同读头的效应。C 被单独阻断时只把自产槽 V 置零。早层及 C 同时阻断后，分别恢复 H18 G 记忆、恢复下游三处，及做两者的联合。

若 E 阻断直接改变了实际首个 State token，则对至多 16 条按冻结顺序尝试后续贪心提案，以实际选择的 token 起头。若冻结臂未触发换词，启动独立词面诊断，在 E 的全词表只把原 State 单 token 的 logit 设为负无穷，后续自然贪心继续。用因果前缀逐 token 验证，每次至多接纳首个由模型实际发出的新 token，相同提案最多五趟。必须真实达到 `Code:` 才记自然答码。格式分歧、EOS、预算终止均留原始前缀和首次分歧。该回放给出词面改选后的自然后继，对所选早层头的改词作用只用 E 阻断本身评估。

### 发现、冻结和数学量

在 L0–23 用提示事实位置的原生 $\\alpha V$ 经 $W_O$ 的实际投影范数和朝 State/拒绝词读出差方向上的投影分别筛分，两个排序各取四个不同层。真实阻断这些头后，从经过 G 而未见后继输出的前缀读取 L25，计算 $D=\\frac12(\\|K_u-K_0\\|/\\|K_0\\|+\\|W_{{O,18}}(V_u-V_0)\\|/\\|W_{{O,18}}V_0\\|)$，按逐视图中位数冻结三层 {names}。向量以对应实际 RoPE 位置和原生值记录，分母下限 $10^{{-4}}$。同层随机头两组 {ctrl[0]} 和 {ctrl[1]}，按发现集原生事实投影范数近邻抽取。复用 Phase 2947 冻结的 L27/L37 各 128 单元、L28H20 及该研究的原随机组合，留出前将身份绑定进方案。末端 H18 同位置对照头 {f['c_controls']}。全部后继 Q 自行计算。

定义 $m_E=\\ell_{{State}}-\\ell_{{rejected}}$、$m_C=\\ell_{{gold}}-\\ell_{{alt}}$，阻断、恢复和反向移植均用同一视图配对。十个主终点使用 99.5% 家族 bootstrap 区间、至少 80 家族。边际效应门为 0.10 logit。物理量要求较随机的增量大于零且达目标组扰动的 20%。桥接要求物理、早层答码、绝对救援、随机头对照及反向移植同时成立。整链还要求末端组特异性、联合补回比例至少 50% 且净救援超过逐例较佳的部分恢复。预注册独立家族正态近似在标准化差 .40 的 96/80 家族功效分别为 {p['gate']['power_n96']:.3f}/{p['gate']['power_n80']:.3f}，未纳入共享词及实质效应门。
"""
    scan=["| 发现头 | 筛分方法 | 筛分量 | L25 K 中位相对量 | L25 投影 V 中位相对量 | 联合 D 中位数 |","|---|---|---:|---:|---:|---:|"]
    for q in f["screening"]:
        scan.append(f"| L{q['layer']}H{q['head']} | {q['criterion']} | {q['screen']:.4g} | {q['median_K']:.4g} | {q['median_V']:.4g} | {q['kv']:.4g} |")
    head+="\n"+"\n".join(scan)+"\n"
    if a["state"]!="finished":
        return head+"\n### 准入\n\n留出行为或技术门未通过。原生行为和技术反例均保留，冻结证据不足以执行正式因果判定。\n"
    lines=["| 主终点 | 效应 [99.5% 区间] | 删一词后的区间 | 过门 |","|---|---:|---:|:---:|"]
    for key,label in LABELS.items():
        v=a["primary"][key]; x,y=v["leave_one_lexeme"]
        lines.append(f"| {label} | {interval(v,key=='kv-physical')} | "+(f"{x:+.2%} ~ {y:+.2%}" if key=="kv-physical" else f"{x:+.4f} ~ {y:+.4f}")+f" | {'是' if v['passed'] else '否'} |")
    physical=["| 操作 | G K 相对量中位数 | G V 投影相对量中位数 | 当前读头 α 中位数 | 读头写入偏移中位数 |","|---|---:|---:|---:|---:|"]
    for k in ("sham","e-block","e-random1","e-random2","relay-k","relay-v","relay-kv","relay-reverse","ec-block","ec-joint"):
        g=a["geometry"][k]
        physical.append(f"| `{k}` | {g['median_ratios'][0]:.4g} | {g['median_ratios'][1]:.4g} | {g['median_g_attention']:.4g} | {g['median_head_change']:.4g} |")
    behave=["| 时序臂 | 实际改选 State | 核验到 C | 核验后自然错码 | 固定词面条件错码 |","|---|---:|---:|---:|---:|"]
    for k,x in a["arms"]["temporal"].items():
        behave.append(f"| `{k}` | {x['state_changed']} | {x['verified']} | {x['actual_code_wrong']} | {x['conditional_code_wrong']} |")
    core=a["absolute"]
    run=a["rollout"]
    alt=[(k,z) for k,z in run["observations"].items() if z["verified"]]
    gold={r["id"]:r["code"] for r in A.load("material-confirm.json")}
    same_code=sum(z["code"].strip()==gold[k] for k,z in alt)
    summary="、".join(f"`{k}` {z['state'].strip()} → {z['code'].strip()}（原答码 {gold[k]}）" for k,z in alt[:4]) or "无完整的干预续写"
    roles="、".join(f"{x['states'][0]} / {x['codes'][0]} → {x['states'][1]} / {x['codes'][1]}" for x in a["natural"]["examples"][:3])
    refs=[]
    for k,v in a["examples"].items():
        if v:
            rr=v[0]["row"]
            refs.append(f"- `{k}`，`{rr['id']}`，State {rr['values']['State']}，拒绝词 {rr['rejected']}。逐臂 State、K/V、P/Q 和首处分歧保留。")
    prior=[json.loads(x) for x in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peak=max(x.get("cuda_peak_bytes",0) for x in prior)/2**30
    shared=max(x["shared_gpu_bytes"] or 0 for x in prior)/2**20
    return head+f"""
### 留出结果与本地复算

{a['eligible_original']} 个家族过原行为门，{a['n']} 个家族、{8*a['n']} 个视图在两个面板共同过无干预技术门。整条 H18 桥接判据{'通过' if a['bridge_passed'] else '未通过'}，跨阶段整链{'通过' if a['whole_passed'] else '未通过'}。早层事实阻断的原始 State 损失 {interval(core['e-state'])}，同头原生恢复 {interval(core['self-e'])}。早层对条件答码的原始损失 {interval(core['e-code'])}。H18 只恢复 K、只恢复 V、同时恢复的净值依次为 {interval(core['res-k'])}、{interval(core['res-v'])}、{interval(core['res-kv'])}。独立遮挡量定义为 E 单阻的损失减去 C 已阻断时再加入 E 阻断的损失，得到 {interval(core['occlusion'])}。

{chr(10).join(lines)}

早层组对 L25 自产缓存的扰动中，超出同层随机头的部分约为 {a['fractions']['physical']:.2%}，低于预注册的 20%。

{chr(10).join(physical)}

表中前两列为该臂介入 C 读出前的自产缓存，与完好值的相对范数。后两列来自实际介入后的当前读取。相同 K/V 在不同 Q 和投影下可以改写不同方向。早层、K/V 完好双向写回及分离恢复逐元素核对，所有后续操作的首次 State logits 与相应 E 阻断臂逐元素相同。

| 后继恢复位点 | 仅阻断 C 后救回边际 [95%] |
|---|---:|
| L27 128 单元 | {interval(core['single-0'])} |
| L28H20 | {interval(core['single-1'])} |
| L37 128 单元 | {interval(core['single-2'])} |
| 上述三处联合 | {interval(core['c-joint'])} |

仅 C 阻断损失 {interval(core['c-vblock'])}，三处联合补回比例 {fmt_frac(a['fractions']['c'])}。E+C 同时阻断损失 {interval(core['ec-block'])}，H18 局部原生缓存恢复 {interval(core['ec-cache'])}，保持 C 阻断只恢复末端三处 {interval(core['ec-end'])}，缓存加末端联合 {interval(core['ec-joint'])}，联合补回比例 {fmt_frac(a['fractions']['ec'])}。恢复 H18 缓存本身会解除这个接收头的 G Value 阻断，因此这个比较衡量各项原生恢复是否互补，不能用它替代 E 造成的输出损失是否经 H18 中介。

### 真实改选、同词传播和反例

{chr(10).join(behave)}

自然角色互换例 {roles}。六个关键时序臂都保持真实贪心前缀的家族为 {a['natural_common']} 个。冻结 E 事实阻断触发的已验收实际换词 {run['real_early_switches']} 条。补充续写模式 `{run['mode']}`，尝试 {run['tested']} 条，核验至 C {len(alt)} 条。其中 {same_code} 条保留原答码，{len(alt)-same_code} 条换码。代表续写 {summary}。其余尝试保留已核验前缀和分歧位置。早层无干预技术失败视图 {len(a['technical']['temporal'])}，末端 {len(a['technical']['code'])}。原始自然行为失败 {len(a['original_failures'])} 条。

{chr(10).join(refs)}

### 严格边界、机制拼图与下一阶段

相对范数测到相同词面下编码状态的变化，方向是否携带该条件的答码信息由恢复、反向移植及随机对照分别约束。角色反转还改变事实文本和可见索引，给出真实词面和后继行为的参照。词面抑制是独立介入，能够考察旧提示在面对已改口的 State 后是否保持原 P/Q。早层头搜索只保留八个精确候选，未覆盖早层的多数头、多位置组合、其他 GQA 读头及所有自然文本。词汇跨家族复用，区间按家族尺度计算。状态修改可能及于后继多条读取，原生局部替换只能识别指定位置和模块范围内的路径。全词表语料、独立节点、长自由生成和非目标任务副作用仍待复验。

三图谱新增事实角色的可互换配对、可到达 L25 的因果深度约束、自产槽物理 K/V 与条件答码的分立证据。有限参数复用的第一性问题是词身份与角色条件如何附着到同一位置，不同读头从该位置读取的究竟是什么。下一大阶段应把词面改变和词面稳定分别建立自然生成预测，扩大早层前缀多位置搜索，冻结与 H18 输出方向耦合的 K/V 坐标而非仅按范数找头，并用角色与词汇交叉反转检验未解释的读写路径。

### 工程归档

RTX 5080 上串行加载本地 BF16 快照，CUDA 分配峰值 {peak:.2f} GiB，进程共享峰值 {shared:.0f} MiB。D 盘逐批核对、保留 20 GiB。材料、全 24×40 头筛查、精确早层候选、冻结计划、实际局部 K/V、经核验的真实改词续写、技术反例、审计、绘图及哈希均列于 `tests/glm5/result/kv_early_20261116_v1/`。证据来自本地单节点实验与技术复算。
"""


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(p,q)=plt.subplots(1,2,figsize=(12.7,4.6),layout="constrained")
    labels=("e-code","relay-abs","relay-specific","reverse","c-vblock","c-joint","ec-joint","ec-over-partial")
    names=("Early block on code","H18 native K/V rescue","H18 vs shared-group controls","Reverse K/V transplant","H18 final V block","Final trio vs random","Cross-stage joint","Joint vs best partial")
    x=np.array([a["primary"][k]["mean"] for k in labels]); ci=np.array([a["primary"][k]["ci"] for k in labels]);
    p.errorbar(x,range(len(labels)),xerr=abs(ci-x[:,None]).T,fmt="o",color="#3c6a7c",capsize=3)
    p.set(yticks=range(len(labels)),yticklabels=names,xlabel="Paired logit margin (99.5% interval)",title=f"Held-out causal contrasts · {a['n']} families")
    p.invert_yaxis(); p.axvline(0,color=".5",lw=1); p.axvline(.1,color="#b38861",lw=1,ls="--")
    t=a["trace"]; layers=[r["layer"] for r in t]
    for key,label,color in ((3,"Generated State residual","#326d80"),(4,"Generated State K","#9b6b58"),(5,"Generated State V","#52826a")):
        q.plot(layers,[r["median"][key] for r in t],lw=1.8,label=label,color=color)
    q.set(xlabel="Transformer block",ylabel="Actual L2 change (median)",title="Propagation with identical State token")
    q.axvline(25,color=".5",lw=1,ls="--"); q.legend(frameon=False)
    for x0 in (p,q): x0.spines[["top","right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"early-kv-effects.png"),dpi=175)
    plt.close(fig)


def close(preview):
    p=C.verify(); a=A.load("audit.json"); f=A.load("freeze.json")
    txt=report(a,p,f)
    P.check_write(C.OUT/"report.md").write_text(txt,encoding="utf-8")
    if a["state"]=="finished": figure(a)
    if preview: return
    assert not (C.OUT/"memo-receipt.json").exists()
    path=C.TMP.resolve(); owned=list(path.rglob("*"))
    assert path==C.TMP and path.parent==(C.ROOT/"tests/glm5_temp").resolve()
    assert all(z.resolve().is_relative_to(path) and not z.is_symlink() and not z.is_junction() for z in [path,*owned])
    m=sum(z.stat().st_size for z in owned if z.is_file())
    shutil.rmtree(path)
    A.save("cleanup.json",dict(path=str(path),removed=m,free=shutil.disk_usage(C.OUT).free))
    src={*p["sources"],Path(__file__).relative_to(C.ROOT).as_posix(),"tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for key in sorted(src): z.write(C.ROOT/key,key)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(key)==(C.ROOT/key).read_bytes() for key in src)
    A.save("execution.json",dict(shell=[f".venv/Scripts/python.exe -B -u tests/glm5/kv_early_run.py {stage}" for stage in (
        "pilot-behavior","pilot-engine","plan","discover","confirm","audit","preview","close")],plan=B.sha("plan.json"),freeze=B.sha("freeze.json"),
        confirm=B.sha("confirm.json"),audit=B.sha("audit.json")))
    first=append(P.check_write(C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"),TITLE,txt.split("\n\n",1)[1])
    second=append(P.check_write(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"),"14B 早层事实到 L25 自产 State K/V 及后继救援索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`，完整报告 `tests/glm5/result/kv_early_20261116_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[first,second]))
    A.save("manifest.json",dict(files={x.name:A.digest(x) for x in sorted(C.OUT.iterdir()) if x.is_file() and x.name!="manifest.json"},
                                sources={k:A.digest(C.ROOT/k) for k in sorted(src)}))
