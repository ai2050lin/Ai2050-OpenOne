"""Append Phase 2946 with auditable summaries and the complete run manifest."""

import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from b36_semantic14_memo import append
import kv_origin_run as C


A,B,P=C.A,C.B,C.P
TITLE="14B 自然答码中的提示词、自生成 State 与 L25H18 下游路径"
ORIGINS={"fact":"提示事实原词","book":"提示码本键","gen":"自产 State","prompt":"提示事实与码本","all":"三个 State 来源","other":"提示拒绝词","u":"L25H18 整头"}
PRIMARY={"fact":"事实原词阻断超出随机头","book":"码本键阻断超出随机头","gen":"自产 State 阻断超出随机头",
         "prompt-gen":"提示联合阻断减自产 State 阻断","rest-fact":"三位置阻断时仅恢复事实原词","rest-gen":"三位置阻断时仅恢复自产 State",
         "res-head-specific":"候选头救援超出随机头","res-head-absolute":"候选头净救援","res-unit-specific":"候选单元组救援超出随机组","res-unit-absolute":"候选单元组净救援"}


def interval(x):
    return f"{x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def output_report(a,p,f):
    stock=A.load("pilot-calibration.json")
    pilot=A.load("pilot-behavior.json")
    disc=A.load("discover-behavior.json")
    held=A.load("confirm-behavior.json")
    correct=lambda x:sum(v["correct"] for v in x["cells"].values())
    lay,h=f["head"]
    mlp=f["mlp"]
    dose=f["unit_dose"]
    power=A.load("precision-design-receipt.json")
    if (C.OUT/"recovery-counterexamples.json").exists():
        recovery=A.load("recovery-counterexamples.json")["families"]
    else:
        recovery=[]
    text=f"""# {TITLE}

运行 `{C.RUN}`。本地 Qwen3-14B 原始 BF16 快照 `{p['model']['snapshot']}`。Phase 2945 证明阻断 L25H18 会拉低正确答码边际，同时 L26H4/H34 的原生写回几乎未补回损失。本阶段追踪其他下游路径，并分辨 H18 在读提示中事实原词、码本中的键，还是模型自产的 State。

### 自然行为及词源定位

复用 Phase 2945 校准的语法，重新抽取互斥词表与家族，排除已开展的相关路线实验所用词。试跑 {correct(pilot)}/64，发现 {correct(disc)}/256，留出 {correct(held)}/1024。八种视图交叉一个或三个竞争槽、短或长生成距离、State/Note 顺序。按全词表 argmax 从 chat 边界核对自然续写，只接受实际生成的行末空白变体。过门后干预在 `Code:` 的最后查询，记录随后第一个未限词表的 token。干预不改变已经核验的历史前缀。每格行为门 90%，主检验用八视图都正确的完整家族且至少 80 家族。

提示事实原词由任务事实片段的角色跨度定位。码本键由单独的字面行定位，整个键跨度入组，即使同一词分成多个 token。模型生成的 State 用核验后的 State 行词位定位；提示中的拒绝词提供同头、同查询的词源对照。每个来源的阻断只对该头在这个查询访问的对应 Value 行置零。Q、K、softmax 分母、其余 Value 与其他 GQA 头采用原计算。通过同形状单头原生 SDPA 比较置零前后结果，将响应差施加于原全头 SDPA 的对应输出并转 BF16。原生恢复把该视图保存的 Value 写回再运算；另在三位置联合阻断时单独恢复事实、码本或自产 State。

### 发现、冻结和数学口径

整头阻断 L25H18 后，在 L26–39 逐个恢复 14 个完整注意力投影写入，在 L25–39 逐个恢复 15 个完整 MLP 写入。每次用同形状完好分支的原 BF16 写入，保留当下传入残差。按发现集配对边际救援选出 L{lay} 注意力和 L{mlp} MLP。在 L{lay} 继续对全部 40 个原生 query 头逐头恢复，选出 H{h}；在 L{mlp} 按原生门控激活差乘 down 列在 P/Q 读出方向上的有号投影，冻结前 96 个单元。候选、筛选分数、种子、匹配池和所有对照先冻结，再开封留出。

L25H18 整体同层对照 {f['source_controls']['u']}。同位置随机头按发现集 $\\|W_{{O,h}}\\sum_{{t\\in S}}\\alpha_{{h,t}}V_{{g(h),t}}\\|_2$ 的几何均值匹配，事实、码本、自生成三组分别为 {f['source_controls']['fact']}、{f['source_controls']['book']}、{f['source_controls']['gen']}。候选头同层随机为 {f['head_controls']}。两份同层随机单元组各 96 个，在发现集按 down 列范数乘原生激活绝对量及激活差绝对量的对数近邻抽取。留出实际比较候选头和候选单元组的完整向量阻断、受上游阻断时原生救援及匹配对照，并保留完整注意力与完整 MLP 的粗粒度救援和阻断。

发现集的单位写入范数之和受单元方向影响而与联合向量范数不同。候选组和两组随机在原生剂量上的上述匹配量为 {dose['target_intact']:.2f}、{dose['control_intact'][0]:.2f}、{dose['control_intact'][1]:.2f}，上游阻断差剂量为 {dose['target_change']:.2f}、{dose['control_change'][0]:.2f}、{dose['control_change'][1]:.2f}。随机组的匹配仍有约 10–22% 的差异，结论需同时看绝对救援和扰动方向。

最后残差中定义实际 BF16 写入 $\u0394r_j$ 为同一子层加法后减加法前。整头阻断与自然执行的差满足 $r_{{nat}}-r_u=\\sum_j (\\Delta r_{{j,nat}}-\\Delta r_{{j,u}})$。令 $g(r)$ 为连续 FP64 RMSNorm 加 P/Q 有号 unembedding。沿末端两残差的直线，积分梯度 $\\bar v=\\int_0^1\\nabla g(r_u+t(r_{{nat}}-r_u))dt$，每个子层的读出账为 $\\bar v\\cdot(\\Delta r_{{j,nat}}-\\Delta r_{{j,u}})$。16 点 Gauss–Legendre 积分核对望远镜求和，并单记 BF16 原生 logits 的舍入偏差。这是输出差的代数分解。恢复特定头或单元实测的反事实才用于路径判断。

边际 $m=L_{{gold}}-L_{{alt}}$；阻断损失为 $m_{{sham}}-m_{{block}}$，原生救援为 $m_{{patched}}-m_u$ 或 $m_{{patched}}-m_{{all}}$。十个主终点按家族等权、八视图平均，使用 99.5% 配对家族自助区间及绝对 0.10 logit 门。提示联合与自产的差用双向判定，其余检验正向。主要中介还须对照校正及绝对恢复过门，实际补回至少 20% 上游损失，且独立阻断使正确码边际下降。所有贪心答码及转错反例另计。预先保存的独立配对正态近似在标准化差 0.40、128 家族、每项 .005 时功效 {power['n128_power']['0.4']:.3f}，以 80 家族计为 {power['n80_power']['0.4']:.3f}；实际受词汇复用与实质效应门影响。
"""
    if a["state"]!="finished":
        return text+"\n### 执行结论\n\n留出行为门失败，按方案保留失败格及反例。无合格的留出因果确认。\n"
    blocks=["| 阻断操作 | 边际损失 [95%] | 错码数 | H18 输出变化中位数 | L25 实际投影变化中位数 |","|---|---:|---:|---:|---:|"]
    for k,label in ORIGINS.items():
        v=a["behavior"][k]; g=a["output_geometry"][k]
        blocks.append(f"| {label} | {interval(v['drop'])} | {v['errors']}/{a['n']} | {g['source_head_change']:.4f} | {g['block25_projected_change']:.4f} |")
    endpoints=["| 主终点 | 效应 [99.5%] | 去掉单个词的范围 | 通过 |","|---|---:|---:|:---:|"]
    for k,label in PRIMARY.items():
        v=a["primary"][k]; lo,hi=v["leave_lexeme"]
        endpoints.append(f"| {label} | {interval(v)} | {lo:+.3f} ~ {hi:+.3f} | {'是' if v['passed'] else '否'} |")
    paths=["| 候选粒度 | 独立阻断损失 [95%] | 上游阻断后净救援 [95%] | 上游边际补回比例 | 救援后错码 |","|---|---:|---:|---:|---:|"]
    for k,lbl in (("att",f"L{lay} 完整 attention"),("head",f"L{lay}H{h}"),("mlp",f"L{mlp} 完整 MLP"),("unit",f"L{mlp} 原生 96 单元")):
        v=a["path"][k]; x=a["behavior"][f"res-{k}"]
        frac="—" if v["fraction"] is None else f"{v['fraction']:+.1%}"
        paths.append(f"| {lbl} | {interval(v['block'])} | {interval(v['rescue'])} | {frac} | {x['errors']}/{a['n']} |")
    cc=["| 条件 | 事实原词损失 | 码本键损失 | 自产 State 损失 | 提示联合减自产 |","|---|---:|---:|---:|---:|"]
    for code,label in (("00","单竞争短"),("01","单竞争长"),("10","三竞争短"),("11","三竞争长")):
        xx=a["cells"][code]
        cc.append(f"| {label} | "+" | ".join(interval(xx[k]) for k in ("fact","book","gen","prompt-gen"))+" |")
    notes=[]
    for label,lst in a["examples"].items():
        if lst:
            q=lst[0]
            notes.append(f"- `{label}`，样本 `{q['row']['id']}`。整头、事实原词、码本键和自产槽阻断的边际损失分别为 {q['drop_u']:+.3f}、{q['drop_fact']:+.3f}、{q['drop_book']:+.3f} 和 {q['drop_gen']:+.3f}。")
    ranks=sorted([(m[k]["mean"],m["layer"],k) for m in a["continuous"]["layers"] for k in ("attention","mlp")],key=lambda e:-abs(e[0]))[:6]
    largest="、".join(f"L{l} {k} {x:+.3f}" for x,l,k in ranks)
    certs=[k for k,v in a["path"].items() if v["major"]]
    flag="通过主要中介门的细粒度候选为 "+"、".join(certs)+"。" if certs else "两个细粒度候选均未通过主要中介门。"
    logs=[json.loads(l) for l in (C.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peak=max(x.get("cuda_peak_bytes",0) for x in logs)/2**30
    shared=max(x["shared_gpu_bytes"] or 0 for x in logs)/2**20
    natural="、".join(x["row"]["id"] for x in a["natural_failures"][:4]) or "本次无自然行为失败"
    numeric=a["numeric"]
    recovered=""
    if recovery:
        names="、".join(z["family"] for z in recovery)
        sensitivity=A.load("recovery-sensitivity.json")
        recovered=(f"并行单查询原先在 {len(recovery)} 个家族触发无干预首 token 或原始 logit 一致性门。"
                   f"这些家族（{names}）对全套干预采用与自然评分相同的原序列形状复算，完整保留先前日志、两次无干预 P/Q 及实际 token。"
                   f"原冻结、阈值和统计口径沿用，分支在每个家族内对齐。剔除后十项判定相同，效应均值最大变化 {sensitivity['max_endpoint_change']:.4f} logit。")
    return text+f"""
### 留出证据

完成 {a['eligible']} 个完整家族、{a['n']} 个新视图。发现集上选出的 L{lay} 整体注意力和 L{mlp} 整体 MLP 救援分别为 {f['module_scores'][f'res-{lay}-att']:+.3f}、{f['module_scores'][f'res-{mlp}-mlp']:+.3f}，候选头 H{h} 的发现救援为 {f['head_scores'][h]:+.3f}。下面为开封后同材料结果。

H18 直接读取的主要受检来源是自然生成的 State。按完整八视图家庭计，自产 State 阻断使边际降低 {a['behavior']['gen']['drop']['mean']:.3f}，事实和码本联合阻断为 {a['behavior']['prompt']['drop']['mean']:.3f}。自产槽单独恢复在其余两位置仍置零时补回 {a['primary']['rest-gen']['mean']:.3f}。L{lay}H{h} 的原生救援通过绝对及随机头校正检验，补回 {a['path']['head']['fraction']:.1%}。较粗粒度的 L{mlp} MLP 有更大的绝对救援，整个 MLP 独立阻断的区间跨零。三个位置的取证发生在 `Code:`，提示可能早已参与形成自产槽的状态。

{chr(10).join(endpoints)}

{chr(10).join(blocks)}

H18 自然注意力中位数按事实、码本、自生成、拒绝词依次为 {a['alpha']['fact']:.5f}、{a['alpha']['book']:.5f}、{a['alpha']['gen']:.5f}、{a['alpha']['other']:.5f}。三个 State 来源联合损失相对分别相加的交互为 {interval(a['interaction'])}。三位置阻断后仅恢复码本的净变化 {a['behavior']['all']['drop']['mean']-a['behavior']['restore-book']['drop']['mean']:+.3f}，对应两条预注册的恢复检验在上表。

{chr(10).join(cc)}

{chr(10).join(paths)}

{flag} 受上游阻断时，候选接收 query 的实测偏移范数中位数 {a['receiver_qdiff_u']:.4f}。候选 MLP 组原生 down 写入范数的中位数按自然、上游阻断、原生组写回依次为 {', '.join(f'{x:.4f}' for x in a['unit_write_norm'])}。原生候选头和 96 单元写回逐元素核对，独立归零同样核对。连续读出积分得到平均损失 {a['continuous']['mean']:+.4f}，分解闭合最大误差 {a['continuous']['close']:.3g}，与 BF16 输出逐例绝对差中位数 {a['continuous']['bf16_abs_error_median']:.4f}。绝对均值最大的六个写入账为 {largest}。原生末端效果由上游写入、后续放大及抵消共同决定，账本把各子层有号项保留。

### 反例、限制和下一大阶段

自然行为失败记录 {natural}。其他逐例反例保留原提示、选择前缀、三种词源头权重、各臂 P/Q、实际贪心 token 和分层读出贡献。

{chr(10).join(notes)}

实验定位于最后答码查询，检验真实历史中不同位置的 Value 写入。提示原词与自产 State 在序列里的时间和因果可见性不同，后者可能已通过早期计算吸收前者，单次阻断覆盖不了此前形成的状态。位置编辑保持 K 与 softmax 分母，因此结论限定为当前注意力路由下的位置写入。直接残差账和单处救援都可能受路径交互影响。完整模块替换同时改变多维写入，配套头和单元随机对照只比较各自粒度。发现只扫描 L25–39、最佳注意力层全头和最佳 MLP 的固定 96 单元预算；网络中其他冗余组合仍有待查。置信区间按家族交换性计算，词汇有跨家族复用。独立节点、全网络副作用和自然生成早期位置仍需另验。

三图谱新增同一 State 对象在事实、码本和自产槽的可追踪定位，内部边区分动态注意力读写、加法残差流及末端输出。语言组合的第一性问题是哪些信息来自当前读取，哪些已被上下文写进可持存的状态。下一大阶段可在不同语法、距离与反向码本上冻结同词多位置的留出预测，在 State 初写时分别阻断事实读取和自产历史形成，再测多个下游恢复位点能否联合补回剩余输出损失。该阶段应保留直接残差基线，并用同预算比较正交对照和跨生成步的复用。

### 执行与复算

在单卡 RTX 5080 串行使用原始 safetensors 逐层 CUDA 计算。显存分配峰值 {peak:.2f} GiB，进程共享显存峰值 {shared:.0f} MiB。试跑逐位置写入解析与 BF16 响应最大向量误差 {stock['max_v_response_error']:.4g}，留出同项 {numeric['v_kernel_error']:.4g}。留出同形状完整原生恢复最大 P/Q 差 {numeric['same_shape_restoration_max']:.4g}，全前缀与验收对照最大原始 logit 差 {numeric['sham_max']:.4g}，贪心 token 校验通过。{recovered}保存材料、读出索引、所有 29 模块/40 个候选头筛分数据、冻结候选、分片、反例和图，日志含显存与耗时。研究记录仅追加并校验历史字节前缀。证据状态为本地实验与技术复算。
"""


def figure(a,f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14.5,4.6),layout="constrained")
    pnames=["fact","book","gen","prompt","all"]
    x0=[a["behavior"][k]["drop"] for k in pnames]
    x=np.array([v["mean"] for v in x0]); ci=np.array([v["ci"] for v in x0])
    ax=axs[0]
    ax.errorbar(x,range(len(x)),xerr=abs(ci-x[:,None]).T,fmt="o",capsize=3,color="#366d82")
    ax.set(yticks=range(len(x)),yticklabels=["Prompt fact","Codebook key","Generated State","Both prompt sites","All three"],
           xlabel="Correct-code margin loss (95% CI)",title=f"Source positions · {a['eligible']} families")
    ax.invert_yaxis(); ax.axvline(0,color="#a8aaad",lw=1)
    ax=axs[1]; data=a["continuous"]["layers"]; layer=np.array([v["layer"] for v in data]);
    for off,name,col in ((-.18,"attention","#537c8c"),(.18,"mlp","#9b7754")):
        ax.bar(layer+off,[v[name]["mean"] for v in data],width=.34,label=name,color=col)
    ax.axhline(0,color="#a8aaad",lw=1); ax.set(xlabel="Transformer block",ylabel="Integrated margin contribution",title="Realized residual accounting")
    ax.legend(frameon=False)
    ax=axs[2]; keys=["att","head","mlp","unit"]
    est=[a["path"][k]["rescue"] for k in keys]; y=np.array([v["mean"] for v in est]); ci=np.array([v["ci"] for v in est]);
    ax.errorbar(y,range(len(y)),xerr=abs(ci-y[:,None]).T,fmt="o",capsize=3,color="#497f72")
    l,h=f["head"]; j=f["mlp"]
    ax.set(yticks=range(len(y)),yticklabels=[f"L{l} attention",f"L{l}H{h}",f"L{j} MLP",f"L{j} 96 units"],
           xlabel="Recovery from upstream block (95% CI)",title="Native write restoration")
    ax.invert_yaxis(); ax.axvline(0,color="#a8aaad",lw=1)
    for ax in axs: ax.spines[["top","right"]].set_visible(False)
    fig.savefig(P.check_write(C.OUT/"position-path-effects.png"),dpi=175)
    plt.close(fig)


def close(preview):
    p=C.verify(); f=A.load("freeze.json"); a=A.load("audit.json")
    r=output_report(a,p,f)
    P.check_write(C.OUT/"report.md").write_text(r,encoding="utf-8")
    if a["state"]=="finished": figure(a,f)
    if preview: return
    assert not (C.OUT/"memo-receipt.json").exists()
    path=C.TMP.resolve()
    assert path==C.TMP and path.parent==(C.ROOT/"tests/glm5_temp").resolve()
    owned=list(path.rglob("*"))
    assert all(x.resolve().is_relative_to(path) and not x.is_symlink() and not x.is_junction() for x in [path,*owned])
    n=sum(x.stat().st_size for x in owned if x.is_file())
    shutil.rmtree(path)
    A.save("cleanup.json",dict(path=str(path),removed=n,free=shutil.disk_usage(C.OUT).free))
    src={*p["sources"],str(Path(__file__).relative_to(C.ROOT)).replace('\\','/'),"tests/glm5/b36_semantic14_memo.py"}
    if (C.OUT/"recovery-counterexamples.json").exists():
        src.add("tests/glm5/kv_origin_recover.py")
    with ZipFile(P.check_write(C.OUT/"sources.zip"),"w",ZIP_DEFLATED) as z:
        for k in sorted(src): z.write(C.ROOT/k,k)
    with ZipFile(C.OUT/"sources.zip") as z:
        assert all(z.read(k)==(C.ROOT/k).read_bytes() for k in src)
    A.save("execution.json",dict(steps=[f".venv/Scripts/python.exe -B -u tests/glm5/kv_origin_run.py {k}" for k in ("pilot-behavior","pilot","plan","discover","confirm")]+[
        f".venv/Scripts/python.exe -B -u {x}" for x in ("tests/glm5/kv_origin_recover.py","tests/glm5/kv_origin_audit.py","tests/glm5/kv_origin_closeout.py")],
        plan=B.sha("plan.json"),freeze=B.sha("freeze.json"),audit=B.sha("audit.json")))
    first=append(P.check_write(C.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"),TITLE,r.split("\n\n",1)[1])
    second=append(P.check_write(C.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"),"14B 提示词源及 L25H18 下游路径索引",
                  f"详见 GLM5 Phase {first['phase']}，运行 `{C.RUN}`。动态 Q 下区分事实、码本与自产 State，冻结新的下游头和 MLP 候选。报告在 `tests/glm5/result/kv_origin_20261115_v1/report.md`。")
    A.save("memo-receipt.json",dict(records=[first,second]))
    A.save("manifest.json",dict(files={x.name:A.digest(x) for x in sorted(C.OUT.iterdir()) if x.is_file() and x.name!="manifest.json"},
                                sources={k:A.digest(C.ROOT/k) for k in sorted(src)}))


if __name__=="__main__":
    C.setup()
    close(sys.argv[1:]==["preview"])
