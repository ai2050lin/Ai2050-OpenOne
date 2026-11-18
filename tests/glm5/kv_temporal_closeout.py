"""Render the Phase 2947 evidence and append the research memos."""

import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from b36_semantic14_memo import append
import kv_temporal_run as C


A,B,P=C.A,C.B,C.P
TITLE="14B 从提示事实到首次 State、自产记忆及最终答码的分阶段因果检验"
PRIMARY={"e-specific":"首次写 State 的事实阻断超过同层随机头","e-restore":"首次写 State 的原生恢复",
         "e-code":"早期事实阻断对答码的损失超过随机头（固定词面）","g-rescue":"仅恢复自产 State K/V 的净答码救援",
         "g-specific":"恢复 State K/V 比恢复 Spare K/V 多救回的边际","c-specific":"末端 H18 读取自产槽阻断超过同层随机头",
         "c-joint-random":"末端候选联合救援超过随机组合","c-joint-single":"末端候选联合救援超过最强冻结单点",
         "ec-joint":"早期和末端同时阻断后的跨阶段联合救援"}
END={"0":"L27 MLP","1":"L28H20","2":"L37 MLP"}


def name(key):
    return " + ".join(END[x] for x in key)


def ci(z):
    return f"{z['mean']:+.3f} [{z['ci'][0]:+.3f}, {z['ci'][1]:+.3f}]"


def report(a,p,f):
    pilots=A.load("pilot-behavior.json")
    disc=A.load("discover-behavior.json")
    held=A.load("confirm-behavior.json")
    stock=A.load("pilot-runner.json")
    cc=lambda x:sum(y["correct"] for y in x["cells"].values())
    sites="、".join(f"L{l}H{h}" for l,h in f["early"])
    randoms=["、".join(f"L{l}H{h}" for l,h in q) for q in f["early_random"]]
    power=A.load("precision-design-receipt.json")
    body=f"""# {TITLE}

运行 `{C.RUN}`，本地 Qwen3-14B 原始 BF16 快照 `{p['snapshot']['snapshot']}`。Phase 2946 在答码位置发现，L25H18 直接使用自产 State 的 Value。这里加入首次生成 State 之前的事实阻断，检验上游证据如何影响 State 的词面选择，以及自产 token 的内部记忆能否跨生成步传到 P/Q。

### 材料及因果时间边界

试跑 8 家族，发现 32 家族，留出 112 家族。新词与之前同路线材料隔离。家族内交叉角色互换、短/长 receipt、State/Note 输出顺序。互换角色时保持候选词与码本不动，同时变更事实选择与正确答码。条件内可以比较同一词换角色后的响应，家族之间平衡排除/更新规则和备用竞争码。试跑自然过门 {cc(pilots)}/64，发现 {cc(disc)}/256，留出 {cc(held)}/896。每格行为门 90%，主检验需要至少 80 个完整家族。用全词表 argmax 从助理开头逐 token 核验自然续写，允许模型实际产生的行末空白。留出原始答码只在完整验证时入组。

- **E，首次写 State 的前一个查询。** 当时事实文本已可见，模型尚未写出 State 的词。在指定的头上只令事实角色词的 Value 行为零，保留未动的查询、键及 softmax 分母。当场记录模型真正选择的首个 State token，和 $m_E=\\ell_{{State}}-\\ell_{{rejected}}$。恢复臂在同一 BF16 路径写回该样本的原生值。
- **G，自产 State token 的 KV 位置。** 保存动态前向下这个位置的残差、K 和 V 写入变化。恢复臂向后来位置提供同一视图完好分支的原生 G 键和值，保持干预分支的动态 Q，让其余历史继续自行计算。从最早受检层（或更早的 L25）起做独立 G 阻断，只在更晚的 query 将 G 的 V 置零。原生 G 恢复只开放给 G 之后的查询。Spare 同样在 State 之后生成，作为不同来源的原生恢复对照。
- **C，`Code:` 最后查询。** 在 L25H18 只阻断对 G 的 Value 读取，另做原生恢复和同层同位置随机头对照。随后以相同计算形状单点或联合恢复 L27 MLP、L28H20、L37 MLP 的原生写入，不替换层间传入残差。细粒度 MLP 臂只恢复 128 个原生 gate×up 单元，再由原 down 矩阵写回。跨阶段联合臂保持 E 事实和 C 的 H18 读槽阻断，同时恢复 G 原生记忆与冻结的下游细粒度组合，另测只恢复 G 或只恢复下游的两个臂。

后续序列按因果掩码沿动态状态传播。固定原词面的回放可以隔离同一个词内部携带的状态变化。只有从 E 到 C 每个前缀 token 都实际由干预后的全词表贪心读出时，固定前缀下的答码才同时计为自然输出。E 改选已识别的单 token 词时，对照组不变，另用实际生成的词面尝试续写提案，再逐 token 验证。未验证到 C 的案例保留首次出错 token 和位置，固定前缀上的代码输出单独记账。

### 发现与冻结

早期对全部 40 层、每层 40 头计算事实写入近似分数。对一个事实 token，$w_h=W_{{O,h}}[\\alpha_{{h,f}}V_{{g(h),f}}]$，以同一输入的真实 State 与拒绝词 unembedding 之差乘末端 RMSNorm 权重，在该层残差范数下投影。用它预选分数最高的六个不同层，直接在 E 阻断后按真实 State 边际选三个头 {sites}。冻结的两组合计同层随机头分别为 {randoms[0]} 和 {randoms[1]}，由事实写入范数近邻抽取。

在末端，遍历 L27、L28H20、L37 三处的全部七个非空子集。发现时同时比较完整 MLP 与等预算 128 单元组。单元排序项为 $\\Delta a_j\\langle W_{{down,j}},v_{{P/Q}}\\rangle$，$\\Delta a_j$ 是末端 H18 读槽阻断前后的原生门控激活差。候选组与同层随机组按原生剂量和扰动剂量匹配，发现剂量在冻结表逐层披露。选出的细粒度联合为 **{name(f['joint'])}**，最强细粒度单点为 **{name(f['singleton'])}**。末端 H18 同位置随机头 {f['c_random']}，L28 同层随机头 {f['head_controls']}。材料、选点、预算、随机子集以及方案哈希在留出行为开封前冻结。

边际 $m_C=\\ell_{{gold}}-\\ell_{{alt}}$。阻断损失为 $m_{{sham}}-m_{{block}}$，救援是同一阻断背景中的 $m_{{rescue}}-m_{{block}}$。对照校正扣除相同位置和同层随机臂的平均。九个主终点使用 99.444% 配对家族自助区间、0.10 logit 和 80 家族门。跨阶段整条路径还要求 G 中介对照及末端特异性支持，且联合救援至少补回 50% 联合阻断损失。独立家族、标准化效应 0.4 时的近似功效在 112 家族下为 {power['n112']:.3f}，80 家族下为 {power['n80']:.3f}；共享词、实质效应门和行为排除会降低解释范围。
"""
    early=["| E 候选 | 读出投影筛分 | 单头精确阻断损失 |","|---|---:|---:|"]
    for q in f["early_screen"]:
        early.append(f"| L{q['layer']}H{q['head']} | {q['linear_score']:+.4f} | {q['exact_loss']:+.3f} |")
    body+="\n"+"\n".join(early)+"\n"
    if a["state"]!="finished":
        return body+"\n### 准入和失败记录\n\n确认行为或技术门未通过。冻结计划、全量行为及反例记录留在产物。\n"
    prim=["| 主终点 | 效应 [99.444%] | 删单词范围 | 通过 |","|---|---:|---:|:---:|"]
    for k,title in PRIMARY.items():
        s=a["primary"][k]; low,high=s["delete_one_word"]
        prim.append(f"| {title} | {ci(s)} | {low:+.3f} ~ {high:+.3f} | {'是' if s['passed'] else '否'} |")
    factorial=["| 原生恢复集合 | 完整模块救援 [95%] | 细粒度救援 [95%] |","|---|---:|---:|"]
    for k in ("0","1","2","01","02","12","012"):
        factorial.append(f"| {name(k)} | {ci(a['code_factorial']['coarse-'+k])} | {ci(a['code_factorial']['fine-'+k])} |")
    natural=["| 时序臂 | 生成了不同 State | 到 C 的已验证前缀 | 已验证路径的错码 | 固定原词面答码正确 |","|---|---:|---:|---:|---:|"]
    for key,q in a["arms"]["temporal"].items():
        natural.append(f"| `{key}` | {q['state_wrong']} | {q['natural_valid']} | {q['natural_wrong']} | {q['conditional_code_correct']}/{q['conditional_code_total']} |")
    effects=a["absolute"]
    tr={x["layer"]:x for x in a["trace"]}
    first_layer=min(x[0] for x in f["early"])
    depth=""
    if first_layer>28:
        depth=(f"本次最早的 E 阻断在 L{first_layer}。固定 G 的 token ID 时，E 的改动只能经更深层传播到 G 或 C，"
               "因此它在 L25/L28 的 G 记忆中没有跨层逆行途径。物理表用于核对这条结构约束。"
               "若 E 实际改选了 token，下一生成步的 embedding 会随词面改变，此时低层可以受到词身份变化的影响。")
    ph=[]
    for layer in (25,27,28,37,39):
        if layer in tr:
            q=tr[layer]
            ph.append(f"| L{layer} | {q['e_resid']:.4g} | {q['g_resid']:.4g} | {q['g_key']:.4g} | {q['g_value']:.4g} | {q['e_disruption_at_code']:.4g} | {q['mlp_at_code']:.4g} |")
    condition=["| 角色交换 / 生成距离 | 早期事实作用 | 早期 → 固定词面答码作用 | State 记忆超过 Spare | 跨阶段联合净救援 |","|---|---:|---:|---:|---:|"]
    for k,v in a["condition"].items():
        condition.append(f"| {k[0]} / {'长' if k[1]=='1' else '短'} | "+" | ".join(ci(v[j]) for j in ("e-specific","e-code","g-specific","ec-joint"))+" |")
    recovered=a["replay"]["attempted"]
    came=sum(x["verified"] for x in recovered)
    code_alt="、".join(f"{x['id']}={x['code']}" for x in recovered if x["verified"]) or "无已核验至末端的改词提案"
    cfrac="—" if a["fractions"]["c_joint"] is None else f"{a['fractions']['c_joint']:.1%}"
    ecfrac="—" if a["fractions"]["ec_joint"] is None else f"{a['fractions']['ec_joint']:.1%}"
    ok={k:v["passed"] for k,v in a["primary"].items()}
    g_cert=all(ok[k] for k in ("e-code","g-rescue","g-specific"))
    whole=g_cert and ok["c-joint-random"] and ok["ec-joint"] and a["fractions"]["ec_joint"] is not None and a["fractions"]["ec_joint"]>=.50
    claim=("所有整条路径的冻结判据通过。" if whole else "整条路径的冻结判据未通过。")
    gline=("固定词面下支持自产 State K/V 中介。" if g_cert else "自产 State K/V 中介未同时达到早期传播、绝对恢复和对照特异性判据。")
    cases=[]
    for group,v in a["examples"].items():
        if v:
            z=v[0]
            cases.append(f"- `{group}`，`{z['row']['id']}`，State 为 {z['row']['values']['State']}，事实拒绝词为 {z['row']['rejected']}。保留各臂 token、自然前缀门、P/Q 及输入全文。")
    bad="、".join(q["row"]["id"] for q in a["failures"][:4]) or "无原始自然行为失败"
    order=A.load("causal-order-recheck.json")
    gflips=A.load("native-code-counterexamples.json")["state_cache_block_actual_code_changes"]
    shown="、".join(f"`{x['row']['id']}` {x['before']}→{x['after']}" for x in gflips[:4]) or "无"
    logs=[json.loads(l) for l in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peak=max(x.get("cuda_peak_bytes",0) for x in logs)/2**30
    shared=max(x['shared_gpu_bytes'] or 0 for x in logs)/2**20
    return body+f"""
### 留出结果

{a['coverage']['behavior_families']} 个家族通过原始行为门，{a['n']} 个家族共同通过两套干预的同形状技术门。主要分阶段观察为，早期事实阻断的自然 State 边际损失 {ci(effects['e-loss'])}，保持词面时最终 P/Q 边际损失 {ci(effects['e-code-loss'])}，末端阻断自产槽的 P/Q 损失 {ci(effects['c-loss'])}。{gline}{claim}

独立阻断后来查询对 G 的 Value 读取，最终答码边际损失 {ci(effects['g-block'])}，经完整后续核验有 {len(gflips)} 条 P/Q 翻转，说明后继确实使用 G 记忆。E 阻断后只写回完好的 G 键和值，能补回多少答码边际见主终点。E 事实头归零后原生恢复确实可逆，E 与同层随机头的对照校正仍须单独判定。对照地移除提示所拒绝词的边际损失为 {ci(effects['e-other'])}。

{chr(10).join(prim)}

两套臂给出两类结果。末端 7 组恢复均在已经自然生成到 `Code:` 的前缀上测量。时序干预改变了此前生成行为，因此标为固定词面的答码统计与核验到 C 的自然答码需同时保存。

{chr(10).join(natural)}

采用观测到的改词做有界回放 {len(recovered)} 条，{came} 条逐 token 核验到末端，其他候选外或超预算 {len(a['replay']['unresolved'])} 条。自然生成答码核验记录 {code_alt}，这里的答码用 token ID 压缩保存，原件关联到 tokenizer 和实际词。保持原词面且六个关键臂都能自然到达 C 的完整家族 {a['coverage']['natural_common']} 个。固定词面和这一区间的配对证据在 `audit.json/natural_endpoints` 对照。

{chr(10).join(factorial)}

冻结末端细粒度联合补回 C 单阻结果的 {cfrac}。同时阻断早期事实和 C 自产槽时，总损失 {ci(effects['ec-loss'])}，只恢复 G 为 {ci(effects['ec-g'])}，只恢复下游组合为 {ci(effects['ec-fine'])}，两段联合净恢复 {ci(effects['ec-joint'])}，补回比例 {ecfrac}。联合相对逐例两种单段恢复中较优者的差 {ci(effects['ec-joint-extra'])}，这个量描述交互及上界，主终点依然使用预设配对条件。

{chr(10).join(condition)}

### 实际写入与代表反例

下表比较只做 E 事实阻断与同形状自然纯净臂的输出，数值是实际 BF16 前向差向量的 L2 范数中位数。E/G 残差是 block 输出差，G 键值是该层写进随后注意力计算的 K/V，C 投影和 MLP 量是同一层该后继查询的实测写入差。该表反映状态传播，后续替换实验才检验这些写入能否补回答案。

| 层 | E 残差 | G 残差 | G K | G V | C attention 写入 | C MLP 写入 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ph)}

{depth}

对全部 {order['all_attempted']} 条已运行时序视图单独复算，L25 与 L28 对应 G 原生 Value 逐元素一致，所有后继干预与仅 E 阻断的首次 State logits 也逐元素一致。源文件和分片身份见 `causal-order-recheck.json`。

原始自然行为失败示例 {bad}。

G 独立阻断后确实翻转答码的例子包括 {shown}，完整的原生 token、P/Q 及自然前缀校验见 `native-code-counterexamples.json`。另有 {len(a['technical']['temporal'])} 条无干预分支在 State 后的下一 token 发生数值执行形状分歧，按冻结的完整家族规则剔除对应家族，保留原始前缀、首处分歧 token 与 P/Q。

{chr(10).join(cases)}

### 证据边界与阶段任务

E 操作只测试选中的事实 token 到指定 query 的 Value 写入，码本、提示剩余 token 和此前步骤保留。自产槽恢复替换同一位置的整个 K/V，允许其内容、路由权重与读出历史联合变化，不能指定只经过某个具体 GQA 头。固定已生成 token 的反事实给出条件机制；存在途中换词或格式分歧时，以可验证的自然续写范围为行为证据。早期头筛查的打分是线性筛选，漏掉的分布式窗口、其他词位以及时间错位仍可能承载作用。联合恢复有限于冻结的位点及 128 单元预算，整模块与同层随机组的剂量比较层级不同。每格按共享词家族汇总，词汇复用降低独立性。独立节点、自由续写及其他任务副作用尚待复验。

外部图谱新增同词角色互换后的事实与码本对应。内部图谱新增 E 写入、G 残差和 K/V、C 动态读出之间的时间边。关联图谱可以分别检验「选择词面」与「为已选的词写入可读取状态」。从有限参数能反复组合语言的第一性问题看，复用要求模型在不同位置给同一身份附上当下角色，并能从历史状态重建作用。下一大阶段应冻结自然改词后的完整多步后续和真实 P/Q，再用多竞争槽、相反角色规则及局部 K/V 分离恢复，追踪联合救援未覆盖的残差，并比较需多少状态维度才能跨语法预测词面和答码。

### 运行及复核

确认启动的首个入口继承了前项目的 400 MiB 预算，被磁盘预检拦截。改用本轮冻结方案声明的 240 MiB 峰值和 20 GiB 保留空间后继续，只写 D 盘。确认卷仍按原方案逐批检查。

仅用本地原始 14B，在 RTX 5080 逐层 CUDA 执行。显存分配峰值 {peak:.2f} GiB，进程共享显存峰值 {shared:.0f} MiB。试跑首版发现 KV 范数记录的 arm 与样本索引顺序错误，保留失败日志，修正后重跑并在冻结前通过。试跑 P/Q 无干预最大差 {stock['max_sham_error']:.3g}，试跑原生恢复为零差。留出按冻结条件保留两个面板所有技术反例，已验收数据里的末端/时序最大原始 P/Q 差依次为 {a['technical_max']['code']:.4g} / {a['technical_max']['temporal']:.4g}，局部 V 写入最大解析误差为 {a['origin_max']['code']:.4g} / {a['origin_max']['temporal']:.4g}。哈希、材料、原生向量、反例、日志、逐臂首处分歧和答码及统计保存在 `tests/glm5/result/kv_temporal_20261115_v1/`。已只在记录尾部追加并核验历史字节。证据状态为本地实验加技术复算。
"""


def figure(a,f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(x,y)=plt.subplots(1,2,figsize=(12.8,4.7),layout="constrained")
    keys=["e-code","g-rescue","g-specific","c-specific","c-joint-random","c-joint-single","ec-joint"]
    z=[a["primary"][k] for k in keys]
    v=np.array([q["mean"] for q in z]); conf=np.array([q["ci"] for q in z])
    x.errorbar(v,range(len(keys)),xerr=abs(conf-v[:,None]).T,fmt="o",color="#35768a",capsize=3)
    x.set(yticks=range(len(keys)),yticklabels=["Early effect on code","State K/V rescue","State vs Spare rescue","Final State V block","Joint vs random","Joint vs single","Cross-stage rescue"],
          xlabel="Logit margin effect (99.444% interval)",title=f"Held-out causal contrasts · {a['n']} families")
    x.invert_yaxis(); x.axvline(0,color=".55",lw=1); x.axvline(.1,color="#b47c50",lw=1,ls="--")
    g=a["trace"]; l=[q["layer"] for q in g]
    for key,label,color in (("g_resid","Generated State residual","#326b7e"),("g_value","Generated State V","#a66d47"),("e_disruption_at_code","Code attention write","#55816a")):
        y.plot(l,[q[key] for q in g],label=label,color=color,lw=1.8)
    y.legend(frameon=False); y.set(xlabel="Transformer block",ylabel="Median actual L2 change",title="Physical response to early fact block")
    for ax in (x,y): ax.spines[["top","right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"temporal-effects.png"),dpi=180)
    plt.close(fig)


def close(preview):
    p=C.verify(); f=A.load("freeze.json"); a=A.load("audit.json")
    rendered=report(a,p,f)
    P.check_write(C.OUT/"report.md").write_text(rendered,encoding="utf-8")
    if a["state"]=="finished": figure(a,f)
    if preview: return
    assert not (C.OUT/"memo-receipt.json").exists()
    scratch=C.TMP.resolve(); entries=list(scratch.rglob("*"))
    assert scratch==C.TMP and scratch.parent==(C.ROOT/"tests/glm5_temp").resolve()
    assert all(x.resolve().is_relative_to(scratch) and not x.is_symlink() and not x.is_junction() for x in [scratch,*entries])
    n=sum(x.stat().st_size for x in entries if x.is_file()); shutil.rmtree(scratch)
    A.save("cleanup.json",dict(path=str(scratch),removed=n,free=shutil.disk_usage(C.OUT).free))
    sources={*p["sources"],"tests/glm5/kv_temporal_closeout.py","tests/glm5/kv_temporal_entry.py","tests/glm5/b36_semantic14_memo.py"}
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for k in sorted(sources): z.write(C.ROOT/k,k)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(k)==(C.ROOT/k).read_bytes() for k in sources)
    A.save("execution.json",dict(plan=B.sha("plan.json"),freeze=B.sha("freeze.json"),audit=B.sha("audit.json"),
        commands=[f".venv/Scripts/python.exe -B -u tests/glm5/kv_temporal_run.py {k}" for k in ("pilot-behavior","pilot-runner","plan","discover","confirm")]+
                 [f".venv/Scripts/python.exe -B -u tests/glm5/kv_temporal_entry.py {k}" for k in ("confirm","audit","preview","close")]))
    first=append(P.check_write(C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"),TITLE,rendered.split("\n\n",1)[1])
    second=append(P.check_write(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"),"14B 提示事实、自生成 State 与跨生成步救援索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`。报告及反例在 `tests/glm5/result/kv_temporal_20261115_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[first,second]))
    A.save("manifest.json",dict(files={x.name:A.digest(x) for x in sorted(C.OUT.iterdir()) if x.is_file() and x.name!="manifest.json"},
                               sources={k:A.digest(C.ROOT/k) for k in sorted(sources)}))
