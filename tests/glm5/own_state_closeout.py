"""Archive audited results and append the two research memos."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import own_state_explicit as E
from own_state_audit import V1,V2
from b36_semantic14_memo import append
R=E.R
NAMES={"dispatch":"包裹路由","permit":"建筑通行许可"}
ARMS=("replay","state_token_swap","value_patch","matched_null")


def table(header,data):
    return "\n".join("| "+" | ".join(str(x) for x in r)+" |" for r in [header,["---"]*len(header),*data])


def glm_diagnostic():
    materials={r["id"]:r for r in R.read("materials-pilot.json")}
    result=[]
    for item in R.read("glm-pilot.json")["records"]:
        doc=materials[item["id"]]
        text=item["natural"]["text"]
        m=re.fullmatch(r"\s*([a-z]+)\s+and\s+Code\s*:\s*([PQ])\s*",text,flags=re.I)
        result.append(dict(id=doc["id"],output=text,one_line=bool(m),
                           correct=bool(m and (m[1].lower(),m[2].upper())==(doc["state"],doc["code"]))))
    summary=dict(n=len(result),matched=sum(r["one_line"] for r in result),correct=sum(r["correct"] for r in result),
                 scope="Exploratory single-line diagnostic, no gate changes",results=result)
    R.write("glm-format-diagnostic.json",summary)
    return summary


def body(audit,d):
    c=audit["causal"]
    studies=c["tasks"]
    behavior=table(["版本","模型","划分","严格联合","过门格"],[
        (v,r["model"],r["split"],f"{sum(g['joint'] for g in r['cells'])}/{r['view_count']}",
         f"{sum(g['passed'] for g in r['cells'])}/{len(r['cells'])}")
        for v in ("v1","v2") for r in audit[v]["runs"] if r["cells"]])
    effects=table(["任务","原样回放","换词","仅换 Value","等幅对照"],[
        (NAMES[t["task"]],*[f"{t['arms'][arm]['flips']}/{t['n']}" for arm in ARMS]) for t in studies])
    inference=table(["任务","V − 对照","家族 bootstrap 95% 区间","单侧符号置换 p","逐一排除词后"],[
        (NAMES[t["task"]],f"{100*t['value']['difference']:.2f}个百分点",
         f"[{100*t['value']['ci95'][0]:.2f}%, {100*t['value']['ci95'][1]:.2f}%]",
         f"{t['value']['p_sign_flip']:.3g}",
         f"[{100*min(a['delta'] for a in t['exclude_one_value']):.2f}%, {100*max(a['delta'] for a in t['exclude_one_value']):.2f}%]")
        for t in studies])
    cells=table(["任务","替换u","干扰在后d","反转码表m","换词翻转","V翻转","等幅翻转"],[
        (NAMES[t["task"]],*r["factors"],r["state_token_swap"],r["value_patch"],r["matched_null"])
        for t in studies for r in t["cells"]])
    rows={r["id"]:r for f in R.read("causal-run.json")["batches"] for r in R.read(f["name"])["rows"]}
    examples=[]
    for item in studies:
        ex=item["example"]
        bad=rows[item["failures"]["value_patch"][0]["id"]]
        val=lambda key:ex[key]["text"].strip().replace("\n"," / ")
        examples.append(f"- {NAMES[item['task']]} {ex['id']}，自然输出 {val('natural')}，仅换 V 后输出 {val('value_patch')}。反例 {bad['id']} 的目标码为 {bad['alt_code']}，实际仍输出 {bad['value_patch']['code']}。")
    norm=100*max(x["native"]["max_norm_error"] for x in c["checks"])
    perf=[json.loads(line) for line in (V2/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    peaks={m:(round(max(x.get("cuda_peak_bytes",0) for x in perf if x["model"]==m)/2**30,3),
              round(max(x.get("shared_gpu_bytes",0) for x in perf if x["model"]==m)/2**20,1)) for m in R.MODELS}
    receipt=R.read("production-readback.json")
    return f"""**问题与材料。** 承接 Phase 2936。用新创作的英语包裹路由与建筑通行许可检验模型会不会读取自己已经写出的值来作答。三因素2×2×2：是否实质替换 u、无关培训例在操作之前或之后 d、P/Q码表是否交换 m。示例可等于原值或替换值，家族内固定。试跑每任务12家族，确认每任务96个新词对家族，每家族8视图。词表跨划分分离，同一划分内词仍复用。材料原创虚构。主模型本地原始BF16 Qwen3-4B，运行 {R.RUN}；14B和GLM4各64试跑视图，仅记行为。

**行为门。** 外部真值 s=sᵤ，其中 s₀为原值、s₁为替换值；目标答码 c=πₘ(s)。对话模板只给助手出场和 State:，不给状态或答码。贪心自由生成最多11 token，联合成功需单token定位、正确状态与代码、严格两行及EOS。8格各≥90%才准入。V1 符号 APPLY 使4B的替换格失败，派送替换0/48、许可替换35/48，两任务保留各48/48。V2同时改直接操作说明及家族，门槛保持，不能将变化全归于措辞。GLM4未过两行格式门。事后单行诊断中 {d['matched']}/{d['n']} 输出符合 “state and Code: X”，{d['correct']} 条字段内容正确。该诊断不补门。

{behavior}

**同次生成的干预。** 模型自己生成 g₁=s 后，冻结尚未处理 g₁ 的 prompt 缓存 C₀；处理 g₁ 后，各层生成该位置的原生 Kₗ、Vₗ，并自由选定第二个token g₂。g₂未包含P/Q答码。无编辑从C₀及g₁续写，要求首续写全词表BF16 logits逐位相同、全程token一致且 C₀ 逐层哈希不变。换词臂从同一C₀输入另一候选 s' 后自由续写。仅换 V 臂从同一C₀输入 s' 得供体原生 V'ₗ，仅替换状态词位置的V，保持自然K、其他位置、可见 g₁ 和原有 g₂，再自由续写。等幅对照在相同位置和层施加 Vₗ+roll(V'ₗ−Vₗ)，保存BF16舍入。对照实际合并剂量最大相对误差 {norm:.4f}%，低于3%。干预覆盖36层联合 Value，未定位单头。

**主要结果。** 最终答码精确改变为 πₘ(s')，且两行及EOS合格的视图数：

{effects}

48批同次生成无编辑回放全部通过，完整token核验覆盖1,536视图。仅换V时自然 g₁、g₂ 保持；等幅对照0翻转，所有臂保持合格两行与EOS。换词主门差≥25个百分点且8格均正向，V对照差≥10个百分点、两反向码表正向且无逆向格。对照允许翻转≤25%、格式损失≤10个百分点。四主检验用 α=.05/4。每家族把8变体的配对差合计为 Dᵢ=Σₐ[I(干预得到另一词的码)−I(对照得到另一词的码)]，在家族层面对 ±Dᵢ 精确求单侧符号随机化概率。家族的重复词和固定框架限制推断范围，6000次bootstrap区间及逐词排除为描述。换词和V臂的g₂自由度不同，不能以两臂的率差直接判断强弱。

{inference}

每格96个视图，跨格关联变体不累加成独立N：

{cells}

**反例。** 仅换V未跟随的案例为13条派送、7条许可，换词未跟随分别15、14。保留所有行、原始词、格式和真实答码。

{chr(10).join(examples)}

**严格结论。** 在这一模型快照和这两个受控英语任务中，模型处理自己生成的状态词后写出的 Value 会参与后续答码。仅改变该槽位的原生 V 即足以在多数行改变最终答案，可见词和原始 K 均维持。两领域的程序结构、近似句式、生成格式及有限词表是明确边界。本文没有检查 V 的必要性、最小充分通路、未见结构、语法机制、广泛副作用或跨模型机制。等幅对照只覆盖本次置换与剂量，不能排除所有干预偏差。外部图谱增加更新/干扰/映射材料，内部与关联图谱增加两条原生作用记录。RDC主体保持。

**第一性原理与下一阶段。** 语言计算除了层内状态，还需要可被后续生成访问的写入。下一大阶段围绕“何时写入、哪些头读取、在哪些条件保持”展开。用独立发现材料分别缩小 Key、Value、层及头，再冻结新句式、多次更新和间隔干扰材料；比较时序保留、局部移除、同剂量恢复和无关任务损伤。在同一留出按原生位置、词汇基线和同预算稀疏特征比较自然状态、后续输出与干预效应的预测，只有共同通过才扩展为可复用规则。

**执行与审计。** 3模型本地BF16 CUDA串行；14B原生逐块装载，GLM4用CPU调度部分权重。CUDA分配峰值/共享显存采样最大值（GiB/MiB）{peaks}。D盘准入检查和逐批换行日志合格。CPU独立重做文本真值、实际分词、输出解码、行为门和每行BF16匹配剂量，同缓存首续写与完整token由运行时逐批核验。原 prompt 缓存只在批内驻留；保留哈希、全部干预位置两路原生BF16 Value。每任务前两家族保存各生成步的embedding、全部36层及最终归一化原生场。内存预检只提前释放不再用的连续缓存，旧计划保留，科学门不变。未独立重跑GPU节点。

**正式归档。** 正式中心 {receipt['version']} 已登记6个新对象。客户端报告展示 755、761，并显示4卡2边皆为“范围干预”；公开索引及报告已读回。报告SHA256 {receipt['study']['report_sha256']}，读回时刻 {receipt['checked_at']}。正式主线证据晋级和独立节点复算仍分别待办。报告、反例、冻结计划、源代码、原始数组、生产回执和客户端截图位于 tests/glm5/result/own_state_explicit_20261115_v2；试跑V1保留于 tests/glm5/result/own_state_20261115_v1。记录仅尾部追加。"""


def cleanup():
    parent=(R.ROOT/"tests/glm5_temp").resolve()
    cleaned=[]
    for result in (V1,V2):
        p=parent/result.name
        assert p.resolve()==p and p.parent==parent and not p.is_symlink() and not p.is_junction()
        if not p.exists():
            continue
        files=list(p.rglob("*"))
        assert all(not f.is_symlink() and not f.is_junction() and f.resolve().is_relative_to(p) for f in files)
        n=sum(f.stat().st_size for f in files if f.is_file())
        shutil.rmtree(p)
        cleaned.append(dict(path=str(p),bytes=n))
    R.write("cleanup.json",dict(paths=cleaned,free_after=shutil.disk_usage(V2).free))


def finish():
    audit=R.read("audit.json")
    assert {item["model"] for item in audit["v2"]["runs"]}==set(R.MODELS)
    assert R.read("publication.json")["remote"]["client_index_verified"]
    assert not (V2/"memo-append-receipt.json").exists()
    text=body(audit,glm_diagnostic())
    R.Previous.check_write(V2/"report.md").write_text("# 自生成中间状态如何影响最终答码\n\n"+text+"\n",encoding="utf-8")
    sources=sorted(Path(__file__).parent.glob("own_state_*.py"))
    sources.extend(Path(__file__).with_name(n) for n in ("condition_chain_run.py","condition_chain_reference_stream.py","b36_switch14_engine.py"))
    src=R.Previous.check_write(V2/"source.zip")
    with ZipFile(src,"w",ZIP_DEFLATED) as f:
        for p in sources:
            f.write(p,p.name)
    with ZipFile(src) as f:
        assert all(f.read(p.name)==p.read_bytes() for p in sources)
    cleanup()
    R.write("manifest.json",{p.name:{f.name:R.digest(f) for f in sorted(p.iterdir()) if f.is_file()} for p in (V1,V2)})
    files=[R.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",R.ROOT/"research/claude/docs/AGI_CLAUDE_MEMO.md"]
    first=append(files[0],"自生成中间状态的缓存复用：非算术同生成分支干预与正式客户端核验",text)
    index=f"""详见GLM5 Phase {first['phase']}。4B两类冻结非算术任务自由行为严格获准1,536/1,536。同批同次生成仅换状态词位置36层V，派送与许可分别使最终答码翻转755/768、761/768，等幅对照0/1536，可见词与K保持。正式中心新增6对象，报告和范围内干预等级已由客户端和公开接口复核。独立节点复算与主线证据晋级待办。失败、数组、回执见 tests/glm5/result/own_state_explicit_20261115_v2。"""
    receipts=[first,append(files[1],"自生成状态复用的范围内干预及正式归档索引",index)]
    for p,entry in zip(files,receipts,strict=True):
        assert hashlib.sha256(p.read_bytes()[:entry["original_bytes"]]).hexdigest()==entry["original_sha256"]
    R.write("memo-append-receipt.json",dict(records=receipts))
    print("Appended GLM5",first["phase"],"Claude",receipts[1]["phase"],flush=True)


if __name__=="__main__":
    E.select()
    R.configure()
    finish()
