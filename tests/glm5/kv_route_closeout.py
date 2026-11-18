"""Render the experiment ledger, index the source, and append the research memo."""

import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from b36_semantic14_memo import append
from kv_route_contrast import configure
import kv_route_run as R


TITLE = "自生成状态 Value 的源位置、读取时刻、内容分离与头组恢复"
SUITES = {"long":"三词干扰","multi":"两次更新","role":"施受关系","polarity":"否定与肯定条目选择"}
LABEL = {"source":"状态槽−Note 槽", "clock":"答码时刻−早期时刻", "content":"答码时刻状态槽的供体−旋转"}
STATUS = {"complete":"原生干预与复算完成", "behavior gate failed":"行为门失败，未进入干预"}


def totals(data):
    return f"{sum(x['correct'] for x in data)}/{sum(x['n'] for x in data)}"


def table(data):
    lines = []
    for v, study in data:
        for suite, cells in study["behavior"].items():
            if suite not in SUITES:
                continue
            cells_ok = sum(x["passed"] for x in cells)
            cells_txt = "、".join(f"{x['correct']}/{x['n']}" for x in cells)
            lines.append(f"| {v} | {SUITES[suite]} | {totals(cells)} | {cells_ok}/{len(cells)} | {cells_txt} |")
    return "\n".join(lines)


def decision(record):
    return "通过" if record["passed"] else "未通过"


def describe(audit, contrast, prior, transfer, trans_contrast, production):
    retro = audit["retrospective"]
    m = retro["medians"]
    all_holdout = audit["holdout"] + transfer["holdout"]
    shorts = "\n".join(f"- {SUITES[s['suite']]} · {STATUS[s['status']]}" for s in all_holdout)
    refs = "、".join(f"{key} {totals(v)}" for key,v in sorted(prior["references"].items()))
    bodies = []
    for row in contrast["suites"] + trans_contrast["suites"]:
        sid = row["suite"]
        stats = next(x for x in all_holdout if x["suite"] == sid)
        lines = [f"### {SUITES[sid]}",
                 f"{stats['n']} 个配对视图，{stats['exact_batches']} 批。自然回放在测量位核对全词表 logits 逐值相同；原始 Value 剂量最大相对偏差 {stats['equal_dose_error']:.3%}。",
                 "", "| 查询头 | 对照 | 均值 | 家族 bootstrap 95% 区间 | 单侧 p | 判定 | 最低视图（ID，分差） |",
                 "|---|---|---:|---:|---:|---|---|"]
        for group, batch in row["groups"].items():
            for k, test in batch["tests"].items():
                lo, hi = test["ci95"]
                worst = test["worst"]
                lines.append(f"| {group} | {LABEL[k]} | {test['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {test['p']:.3g} | {decision(test)} | `{worst['id']}` {worst['contrast']:+.3f} |")
        lines.append("")
        for group, batch in row["groups"].items():
            vals = "、".join(f"{k} {v:+.3f}" for k,v in batch["raw_means"].items())
            lines.append(f"- {group} 四格内容效应 {vals}。位置与时刻交互的描述性均值 {batch['interaction']:+.3f}。")
        lines.extend(["", "| L21H19=0 / L21H25=1 / L20H5=2 | 供体替换净效应 | 归零−等幅旋转 | 真值恢复相对旋转的优势 | 从三头归零恢复的实际收益 | 移除与恢复联合门 |",
                      "|---|---:|---:|---:|---:|---|"])
        for group, val in stats["subsets"].items():
            out = "通过" if val["controlled_pair"] else "未通过"
            swap, removal, restoration = (val[k] for k in ("swap","remove","restore"))
            lines.append(f"| {group} | {swap['mean']:+.3f}（{decision(swap)}） | {removal['mean']:+.3f}（{decision(removal)}） | {restoration['mean']:+.3f}（{decision(restoration)}） | {val['gain']:+.3f} | {out} |")
        med = "、".join(f"{k} {v:+.3f}" for k,v in stats["field_median_margins"].items())
        crossings = "、".join(f"{k} {v['positive']}/{v['n']}" for k,v in row["alternative_crossings"].items())
        rem = stats["measures"]["remaining"]
        broad = stats["measures"]["whole_content"]
        free = "、".join(f"{k} {v['flips']}/{v['n']}" for k,v in stats["free"].items()) or "无整批对齐的答码时刻"
        ties = "、".join(f"{k} 并列翻转 {v['tie_flip']} 条、其他不一致 {v['other_disagreement']} 条"
                        for k,v in row["free_vs_strict_crossing"].items() if v["tie_flip"] or v["other_disagreement"])
        attention = "、".join(f"{k} 状态 {v['state_median']:.4f} / Note {v['note_median']:.4f}" for k,v in stats["attention"].items())
        lines.extend(["", f"- 各臂 P/Q 分差中位数为 {med}。固定前缀下替代码分差大于零的比例为 {crossings}。",
                      f"- 余下 1,149 查询头相对自然的分差均值变化 {rem['mean']:+.3f}，{decision(rem)}。全 1,152 查询头的供体相对旋转均值 {broad['mean']:+.3f}，{decision(broad)}。全组相对两个部分组的非加性项均值 {stats['interaction_mean']:+.3f}。",
                      f"- 自由续写保持原 State、Note 并翻转 Code 的计数为 {free}。答码偏移不统一的批次仅计固定前缀。{ties+'。' if ties else ''}",
                      f"- 由原生 Q/K 重构的注意力质量中位数为 {attention}。这一测量用于解释可能的权重中介。",
                      "- 全三头恢复后的 P/Q 分差逐行回到自然值。部分恢复的因果范围限定在三头归零背景内。"])
        bodies.append("\n".join(lines))
    state = ("中心公开索引及全部对象 SHA256 已核对，研究客户端实屏验收见同目录 client.jpg。"
             if production and production.get("verified") else "中心状态在 production.json 另记。")
    rest = R.OUT/"restart.json"
    if rest.exists():
        check = R.Base.Previous.load(rest)["check"][0]
        state += (f" 两次更新在显存压力下重启，沿用冻结的 32 条批次和原干预。首批干预分数、自由续写、注意力与读次数逐项完全一致。"
                  f"K/V 哈希变化层数为 {check['prompt_k_layers_with_hash_changes']}/"
                  f"{check['prompt_v_layers_with_hash_changes']}，两份原始分片同时保留。一致性结论限定在已核对项。")
    sensitivity = [
        (row["suite"],[(g,s["tests"]["source"]["mean"],s["tests"]["source"]["passed"]) for g,s in row["groups"].items()])
        for row in contrast["suites"] + trans_contrast["suites"]
    ]
    source_summary = "；".join(
        f"{SUITES[sid]} " + "、".join(f"{group} {value:+.3f}（{'通过' if passed else '未通过'}）"
                                       for group,value,passed in group_tests)
        for sid,group_tests in sensitivity
    )
    return f"""# {TITLE}

Qwen3-4B 原始 BF16，快照 `{audit['snapshot']}`。查询头编号从零起。本轮核对已经生成的 State 如何通过原模型的 Value 读取影响 P/Q 竞争，拆开源位置、读取时刻和方向。发现沿用已开封的 16 个家族、64 个配对视图筛查，冻结三头 (21,19)、(21,25)、(20,5)，L21H19 在发现段单头供体相对等幅对照的均值最高。新版试跑使用独立词表，不参与留出机制分析。所有题面为自写虚构英文记录，确认按无序词对定义家族，每组 96 家族 × 状态及编号轮换四视图。

## 上一轮为什么只有 19/384

重算 Phase 2938 干扰组全部 384 条原始分片。令 `m = z[替代码] − z[正确码]`。原模型的替代码分差中位数为 {m['original']:+.3f}。单次三头干预把全部 {retro['rises']}/{retro['n']} 条推向替代码，推力中位数为 {m['lift']:+.3f}，干预后分差中位数仍为 {m['triad']:+.3f}。严格跨过 0 的 {retro['flips']} 条逐条对应自然格式下的自由答码翻转，判定分歧 {retro['sign_disagreements']} 条。全头替换分差中位数为 {m['whole']:+.3f}。翻转少的直接原因可由原有优势和干预推力逐条解释；本轮头组补集提供了额外干预证据，适用范围见各新材料结果。完整账本见 audit.json 的 retrospective.rows。

## 材料门与执行状态

各预注册格都需达到 90% 联合正确。State、Note、Code、停止符和可对齐 token 位置同时合格。第一版三项都失败，保留完整输出。v2 施受组失败后，v3 单独加入否定条目选择。两条句子中一个条目肯定出现，另一条明确否定，首句极性平衡，两个词仍要使用题面中的编号映射。v3 词汇与其试跑分开，首条语句极性固定在家族内。新版过门套组中，少量自然不合格视图保留为零干预分差，并纳入固定前缀分析分母。

| 版本 | 套组 | 联合正确 | 通过格 | 各预注册格 |
|---|---|---:|---:|---|
{table([("v1",prior),("v2",audit),("v3",transfer)])}

{shorts}

参考行为在 RTX 5080 串行执行，{refs}，仅各四条已暴露提示。14B 逐层 BF16 CUDA。参考行为仅用于调通和识别快照，未用于主模型查询头结论。

## 原生运算与统计

在真实生成第一个 State token 后固定其缓存。令 `δ = V_alt − V_state`，其中供体来自相同 prompt 生成相反 State 的反事实分支。只在指定 query head、位置 `j` 和时刻 `t` 重算该头 SDPA 输出，沿原模型后续模块走到答码。固定该头的 Q/K 时，令 `g = floor(h/4)`，头输出的局部算子为 `o_h(t) = Σ_j α_h(t,j) V_g(j)`，作用到这一头的局部代数项为 `α_h(t,j) δ`，经过原生输出投影和下游处理后才形成答码效应。其他位置的缓存保留同源历史，供体内容不会成为模型新生成的 State。

每组采用同一 `δ` 和按层固定循环旋转 `Rδ`，比较状态槽 / 第一个 Note 词，以及紧随首 Note 词的早期 query / 预测答码的 query。两时刻两个源均已在历史可见。每臂每适用查询头恰好一次读取。以 `D(j,t)=m(δ,j,t)−m(Rδ,j,t)` 度量内容对照，位置对比固定答码时刻，时刻对比固定状态槽。原始 Value 改量范数匹配，注意力质量、实际写入残差和下游雅可比可以不同。循环旋转是等幅方向对照，其方向仍可能具有作用。归零对照是在原值上加同范数的旋转负值；从三头归零出发，逐一恢复单头、双头或三头的原值，再与等幅旋转的恢复比较。

先算四视图家族均值，以家族进行单侧符号检验（α = .05 / 96），要求冻结效应下限和每格均值非负两条件同时满足。95% 家族 bootstrap 区间用于描述。词对共享词汇和模板，家族也可能相关，符号 p 值和区间的外推要受此约束。全头和补集试验在同一预测位置，修改读取数分别为 1,152 和 1,149，三头为 3。三者的非加性项仅为该干预分解的描述量，不能折算为自然网络独立通路的份额。

{chr(10).join(bodies)}

位置对照跨材料的结果为 {source_summary}。特别在否定组，L21H19 的答码时刻与内容对照复现，同时给 Note 槽施加匹配向量的净效应更大。这个反例限制了“同一头在各种语言任务中优先从同一位置读取”的主张。

## 证据边界与下一阶段

固定前缀后自由生成只从答码开始，故前面的 State 和 Note 属于已固定历史；模型主动重写状态、长程对话或跨任务副作用尚未受检。干扰组多复制两个英文词，考察短受控间隔，无法隔离自然长文本中的时间距离、语义竞争和注意力质量。两次更新的题面提供了固定算法，施受组提供了句法提示。否定组可通过识别固定措辞与紧邻条目完成，证据局限在生成状态之后的 Value 使用，不能检验模型内部如何表示否定。需要在独立动词、句式和跨语言材料上复核，再判断语言功能的推广。候选头起源于相同主模型的既有发现。全层原生 BF16 状态仅保存每个通过套组首批的四个视图和生成步，全部样本保存材料、分词、生成、干预、读计数和划分身份。无独立 GPU 节点复算，无参数级最小性证明，无同预算 SAE / transcoder 比较。

外部图谱中的对象是当前状态、更新和角色。内部图谱中的对象是固定模型、生成位置、query head 与实际读出的 Value。关联图登记通过的局部干预边，失效条件仍留在实验记录。有限参数产生可组合语言行为，需要解释状态如何经任务变化生成、如何被后续 query 寻址，以及多个读取结果通过下游运算转成输出。下一阶段把三段作为同一预测任务，先冻结真实长距离、多更新和多语种的行为材料与自然错例，再比较等原始剂量及按 `αδ` 匹配的选位实验，用原生路径、词法/位置基线和同预算稀疏方法预测未见条件的下一 token 分布与移除/恢复分差。冻结模型快照与计算预测后，由独立节点复算并测试通路逐层接续。多个任务的通路组织是否共享仍待检验，RDC 主体规则没有由本次实验更新。

## 复核与登记

运行入口在 kv_route_run.py、kv_route_native.py、kv_route_revised.py、kv_route_transfer.py。CPU 分别复算 v1 的 audit、v2 的 revised audit、v3 的 transfer audit；最后运行 kv_route_finalize_audit.py v2 v3，把 JSON 单元格标签还原为内存元组并核对全部匹配对照。计划、模型文件身份、原始分片、失败试跑、资源日志和源文件哈希与本报告同目录或在 v1 和 v3 目录。原场 `native-*.npz` 保留 BF16 bit patterns。图 `route-results.png` 对照历史逐条阈值与新留出均值。{state}
"""


def close():
    assert not (R.OUT/"memo-receipt.json").exists()
    root = R.Base.ROOT
    targets = [root/"research/glm5/docs/AGI_GLM5_MEMO.md", root/"research/claude/docs/AGI_CLAUDE_MEMO.md"]
    report = (R.OUT/"report.md").read_text(encoding="utf-8")
    memo = "\n".join("#"+line if line.startswith("## ") or line.startswith("### ") else line
                     for line in report.split("\n\n",1)[1].splitlines())
    records = [append(targets[0],TITLE,memo)]
    n = records[0]["phase"]
    records.append(append(targets[1],"Qwen3-4B 自生成状态单头与三头读取索引",
                          f"详见 GLM5 Phase {n}、运行 KV-ROUTE-20261115-v2 和 tests/glm5/result/kv_route_20261115_v2/report.md。"
                          "保留上一轮 384 条分差回算、新版源位置/读取时刻/Value 内容对照、单头/子集移除恢复，以及第一版失败和新版准入边界。"
                          "可信范围以计划、审计和中心核对文件为准。"))
    R.write("memo-receipt.json",dict(records=records))


def report():
    other = R.OUT.with_name("kv_route_20261115_v1")
    transfer = R.OUT.with_name("kv_route_20261115_v3")
    load = R.Base.Previous.load
    maybe = R.OUT/"production.json"
    p = R.Base.Previous.load(maybe) if maybe.exists() else None
    text = describe(R.read("audit.json"),R.read("matched-contrast.json"),load(other/"audit.json"),
                    load(transfer/"audit.json"),load(transfer/"matched-contrast.json"),p)
    R.Base.Previous.check_write(R.OUT/"report.md").write_text(text,encoding="utf-8")


def bundle():
    src = Path(__file__).parent
    files = set(src.glob("kv_route_*.py")) | {src/name for name in R.read("plan.json")["sources"]}
    paths = [R.OUT.with_name("kv_route_20261115_v1"), R.OUT, R.OUT.with_name("kv_route_20261115_v3")]
    with ZipFile(R.Base.Previous.check_write(R.OUT/"source.zip"),"w",ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f,f.name)
    with ZipFile(R.OUT/"source.zip") as z:
        assert all(z.read(f.name)==f.read_bytes() for f in files)
    root = (R.Base.ROOT/"tests/glm5_temp").resolve()
    removed = []
    for p in (root/name.name for name in paths):
        if p.exists():
            assert p.parent==root and p.resolve()==p and not p.is_symlink() and not p.is_junction()
            members = list(p.rglob("*"))
            assert all(x.resolve().is_relative_to(p) and not x.is_symlink() and not x.is_junction() for x in members)
            size = sum(x.stat().st_size for x in members if x.is_file())
            shutil.rmtree(p)
            removed.append(dict(path=str(p),bytes=size))
    R.write("cleanup.json",dict(removed=removed,free_bytes=shutil.disk_usage(R.OUT).free))
    R.write("manifest.json",{p.name:{f.name:R.sha(f) for f in sorted(p.iterdir()) if f.is_file() and f.name!="manifest.json"}
                             for p in paths})


if __name__ == "__main__":
    configure()
    {"report":report,"memo":close,"bundle":bundle}[sys.argv[1]]()
