"""Close the gated behavioral study after its real outputs and audits are available."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import content_origin_run as R
import content_origin_report as report
import content_origin_forecast as forecast


def close():
    R.setup()
    if (R.OUT/"memo-receipt.json").exists():
        raise FileExistsError("This run already has a memo receipt")
    decision=R.read("confirm-summary.json")
    if decision["passed"]:
        raise RuntimeError("Native branch required before closing this phase")
    forecast.evaluate()
    report.audit()
    evidence=R.read("analysis.json")
    confirm=evidence["splits"]["confirm"]
    assert confirm["stopped"]==confirm["n"], "Inspect incomplete answers before phase closure"
    plan=R.read("design.json")
    secondary=R.read("lexical-review.json")["changes"]
    forecast_metrics=R.read("forecast-evaluation.json")["metrics"]
    examples=R.read("examples.json")["confirm"]
    telemetry=[json.loads(s) for s in (R.OUT/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    shared=max(x["shared_gpu_bytes"] or 0 for x in telemetry)
    allocated=max(x.get("cuda_peak_bytes",0) for x in telemetry)
    lines=[
        "实际模型为本地 Qwen3-14B 原 BF16 快照 "+plan["model"]["snapshot"]+"。冻结时逐个核对模型权重和 tokenizer，逐对检查真实 token 长度、载荷位置及指定来源词。项目自编材料分别改变档案样式、档案对象及数量、指定来源，完整交叉 2×2×2，每个确认家族走两套新句式。档案早晚与是否自带括号在家族间配平。任务以英文提供材料，要求生成中文、对象及方括号中的数量、一个 Python 打印表达式。所有生成从空回答开始，采用原模型全词表贪心，停止符与 64-token 上限逐条记录。",
        "主量测每个家族 f、句式 j，在要求修订记录时令 A=1 表示输出中唯一识别的对象与唯一数值同时属于档案。对两份载荷 p 算 Δ(f,j)=mean_p[A(代码,p)−A(叙述,p)]。两套留出句式均需平均 Δ≥0.15，且同时 99% 的家族配对区间下界大于零。冻结了每套句式的双侧 Bonferroni 区间，联合家族重复抽样 30,000 次。初始功效预算 "+str(plan["power"]["achieved_power"])+"，按发现组实测方差的 256 族设计复核也通过。相同家族的多个提示共享样本单位。",
        (R.OUT/"report.md").read_text(encoding="utf-8").split("\n\n",1)[1].strip(),
    ]
    for k,cell in sorted(confirm["totals"].items()):
        style,req=map(int,k.split("/"))
        lines.append(f"{('叙述','代码')[style]}档案，指定{('修订','档案')[req]}，严格混用中的档案对象配修订数 {cell.get('archive_object_revised_number',0)}/{cell['n']}，任务对象配档案数 {cell.get('task_object_archive_number',0)}/{cell['n']}。出现多个不同数值 {cell.get('multiple_number',0)} 条，对象无法唯一查到词表中的任一来源 {cell.get('unknown_object',0)} 条。前者保留全部数值，后者可能包含漏识别译名。")
    for frame in decision["effects"]:
        for req in (0,1):
            values=[confirm["pairs"][f"payload_follow/{frame}/{s}/{req}/both"]["mean"] for s in (0,1)]
            lines.append(f"句式 {frame}，指定{('修订','档案')[req]}。同家族互换档案载荷时，两个答案均取各自档案对象和数字的比例，叙述／代码分别为 {values[0]:.1%}／{values[1]:.1%}。")
    for k in ("archive_object_revised_number/requested_0","task_object_archive_number/requested_0","multiple_mentioned/requested_0","archive_both/requested_0"):
        if k in examples:
            v=examples[k]["example"]
            lines.append(f"例 {v['id']}。档案 {v['archive_n']} {v['archive_word']}，任务 {v['old_n']}→{v['new_n']} {v['task_word']}。要求修订记录。原回答 "+repr(v["text"])+f"，对象／数字按冻结规则归为 {v['score']['obj']}／{v['score']['num']}。")
    lines.append("校准发现个别汉字简称嵌在另一个完整名词中，英文单复数也会影响自动判分。在首个确认分片前登记了最长中文词优先、补充少量别名与英文单数的敏感性分析。确认中对象归属改变 "+str(sum(x["split"]=="confirm" for x in secondary))+" 条。两句式该口径的 Δ 分别为 "+"、".join(f"{v:+.2%}" for v in confirm["secondary_delta"].values())+"。主判沿用原评分。未识别译名、多对象、多数值与超出白名单的 Python 表达式按实际状态保留。")
    lines.append("第二项确认前登记的预测以发现家族拟合，对确认输出不再训练。在指定修订记录的子集上预测对象和数字均取档案，Brier 均方误差（越小越好）依次为 "+"，".join(f"{label} {forecast_metrics[name+'/revised/archive_both']['brier']:.4f}" for name,label in (("prevalence","发现总体基率"),("instructed","只看指令"),("nearest","只看最近记录"),("factors","四输入因子与两两交互"),("text","再加提示字符三元组")))+"。各来源、少数类召回与平衡准确率见 forecast-evaluation.json。")
    lines.append("副分析复算时，最近记录基线的布尔数组不能在 NumPy 中直接相减。修复仅在计算平方误差时转为浮点，旧训练源码、系数与全部正常输出均保留。补充预测系数封存在确认首批启动之后、首批回答落盘之前；严格按前向启动时间留出的子集另见 verification.json。")
    lines.append("冻结的正向门决定是否值得把之前的 B26/H36 查表线索用于本任务。本轮在行为层停止，B26/H36 的选源、V/O 写入、等幅阻断和原生恢复没有新增测量。既有查表干预仍属于原任务。旧轮的固定两种载荷、代码字符串和句式与本轮有多处区别；跨轮差异不能单独归到其中一个改动。同一框架内的成对切换可解释其行为效应。句式、英文词表、中文别名和单模型快照限定外推范围。对象和数字分开评分使来源混用、格式正确和实际取值成为可独立检验的事件。下一大阶段可同族交叉显式／隐式取源指令与参考示例的用途，保留当前反例，先预测不同条件的取值方向，再进入相应原生边。")
    lines.append(f"运行入口 content_origin_run.py freeze / run，随后 content_origin_forecast.py 与 content_origin_closeout.py 复核。校准首四条的头两次贪心选择以整前缀重算对账，通过。逐条 token 解码及全部主判从已存答案重算。CUDA 采样峰值 {allocated/2**30:.2f} GiB，共享显存峰值 {shared/2**20:.1f} MiB。128 个锁箱家族只作材料哈希，模型前向为零。归档位置 tests/glm5/result/{R.OUT.name}，包括输入、逐 token 输出、保留混合项的全表、反例、来源哈希和可视化。")
    audit_paths=[p for p in R.OUT.iterdir() if p.is_file() and p.name not in ("manifest.json","memo-receipt.json")]
    code_plan=R.read("forecast-plan.json")
    forecast_source=Path(forecast.__file__).with_name("content_origin_forecast_frozen_v1.py")
    assert code_plan["code"]==R.file_sha(forecast_source)
    notes="\n\n".join(lines)
    with (R.OUT/"report.md").open("a",encoding="utf-8") as f:
        f.write("\n\n"+"\n\n".join(lines[3:])+"\n")
    manifest=dict(run=R.OUT.name,model=plan["model"]["snapshot"],
                  inputs=plan["source"],extras={p.name:R.file_sha(p) for p in [Path(__file__),Path(report.__file__),Path(forecast.__file__),forecast_source]},
                  files={p.name:dict(bytes=p.stat().st_size,sha256=R.file_sha(p)) for p in sorted(audit_paths)})
    manifest_sha=R.save("manifest.json",manifest)
    for name,meta in manifest["files"].items():
        assert R.file_sha(R.OUT/name)==meta["sha256"]
    memo=R.check_write(R.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md",128*1024)
    with memo.open("r+b") as f:
        before=f.read()
        headers=re.findall(rb"(?m)^## Phase (\d+):",before)
        n=int(headers[-1])+1
        title=f"## Phase {n}: 独立档案样式、内容与指定来源的自由生成验证 [{datetime.now():%Y-%m-%d %H:%M}]"
        addition=("\n\n"+title+"\n\n"+notes+"\n").encode()
        assert R.OUT.name.encode() not in before, "Check possible previously appended run"
        f.seek(0,2)
        assert f.tell()==len(before)
        f.write(addition)
        f.flush()
        f.seek(0)
        assert f.read(len(before))==before and f.read()==addition
    R.save("memo-receipt.json",dict(phase=n,title=title,manifest=manifest_sha,
                                     prefix_sha256=hashlib.sha256(before).hexdigest(),
                                     append_sha256=hashlib.sha256(addition).hexdigest(),
                                     line=before.count(b"\n")+3))
    print(title,manifest_sha,flush=True)


if __name__=="__main__":
    close()
