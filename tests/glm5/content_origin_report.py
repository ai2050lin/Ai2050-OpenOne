"""Rejoin native answers to sealed inputs, retain every mixed answer and counterexample."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import content_origin_run as R
import content_origin_material as M
import content_origin_lexical_audit as lexical


def archive_kind(s):
    if s["obj"]=="archive" and s["num"]=="archive":
        return "both"
    if s["obj"]=="archive":
        return "object_only"
    if s["num"]=="archive":
        return "number_only"
    return "neither"


def family_interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(29811008)
    draws=rng.choice(values,(15000,len(values)),replace=True).mean(1)
    return dict(mean=float(values.mean()),ci99=np.quantile(draws,[.005,.995]).tolist(),families=len(values))


def examples(rows):
    chosen={}
    counts=Counter()
    for r in rows:
        s=r["score"]
        tags=[]
        if s["obj"]=="archive" and s["num"] in ("revised","old"):
            tags.append("archive_object_"+s["num"]+"_number")
        if s["obj"]=="task" and s["num"]=="archive":
            tags.append("task_object_archive_number")
        if s["num"]=="multiple" or s["obj"]=="both":
            tags.append("multiple_mentioned")
        if s["archive_both"]:
            tags.append("archive_both")
        if s["correct"] and not s["joint"]:
            tags.append("source_correct_presentation_failed")
        if s["obj"]=="other_or_missing" or s["num"]=="missing":
            tags.append("unresolved")
        for tag in tags:
            key=f"{tag}/requested_{r['requested']}"
            counts[key]+=1
            # A stable, answer-independent priority selects one illustration per category.
            if key not in chosen or R.digest(r["id"])<R.digest(chosen[key]["id"]):
                chosen[key]=r
    return {k:dict(n=counts[k],example=r) for k,r in sorted(chosen.items())}


def plot_confirmation(part):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    fig,(top,lower)=plt.subplots(2,1,figsize=(9.4,6.5),height_ratios=[2.1,1.25],layout="constrained")
    order=("0/0","1/0","0/1","1/1")
    labels=["取修订 · 叙述档案","取修订 · 代码档案","取档案 · 叙述档案","取档案 · 代码档案"]
    left=np.zeros(4)
    for key,label,color in (("both","对象与数字","#174b66"),("object_only","仅对象","#4d9a91"),
                            ("number_only","仅数字","#edaf53"),("neither","其余回答","#e2e7ea")):
        frac=np.array([part["totals"][k].get(key,0)/part["totals"][k]["n"] for k in order])
        top.barh(labels,frac,left=left,label=label,color=color)
        left+=frac
    top.invert_yaxis()
    top.set_xlim(0,1)
    top.xaxis.set_major_formatter(PercentFormatter(1))
    top.set_title("回答中哪些内容可唯一归给档案",loc="left",weight="bold")
    top.legend(ncols=4,loc="upper left",bbox_to_anchor=(0,-.1),frameon=False)
    effects=part["decision"]["effects"]
    lower.axvline(0,color="#727d82",lw=1)
    lower.axvline(15,color="#b4671d",ls="--",lw=1.2,label="预定门槛 +15 个百分点")
    for i,(frame,v) in enumerate(effects.items()):
        x=100*v["delta"]
        lo,hi=np.array(v["ci"])*100
        lower.plot([lo,hi],[i,i],lw=2,color="#174b66")
        lower.scatter(x,i,color="#174b66",s=40,zorder=3)
    lower.set_yticks(range(len(effects)),[f"新句式 {f}" for f in effects])
    lower.set_ylim(-.65,1.65)
    lower.set_title("指定修订记录时，改用代码档案引起的双项取档案率变化",loc="left",weight="bold")
    lower.set_xlabel("百分点，线段为按家族配对的同时 99% 区间")
    lower.legend(loc="upper right",frameon=False)
    for ax in (top,lower):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.savefig(R.check_write(R.OUT/"source-outcomes.png",512*1024),dpi=165)
    plt.close(fig)


def contrast(rows):
    lookup={(r["family"],r["frame"],r["payload"],r["style"],r["requested"]):r for r in rows}
    family_groups=defaultdict(list)
    for (f,frame,p,s,req),r in lookup.items():
        if p==0:
            alternate=lookup[f,frame,1,s,req]
            sa,sb=r["score"],alternate["score"]
            family_groups["payload_follow",frame,s,req,f].append({
                "object":sa["obj"]==sb["obj"]=="archive","number":sa["num"]==sb["num"]=="archive",
                "both":sa["archive_both"] and sb["archive_both"],"text_changed":r["text"]!=alternate["text"]})
        if req==0:
            alternate=lookup[f,frame,p,s,1]
            sa,sb=r["score"],alternate["score"]
            family_groups["instruction_switch",frame,s,"both",f].append({
                "object":sa["obj"]=="task" and sb["obj"]=="archive",
                "number":sa["num"]=="revised" and sb["num"]=="archive",
                "both":sa["correct"] and sb["correct"],"text_changed":r["text"]!=alternate["text"]})
    aggregates=defaultdict(list)
    for (kind,frame,s,req,_),records in family_groups.items():
        for metric in records[0]:
            aggregates[f"{kind}/{frame}/{s}/{req}/{metric}"].append(np.mean([r[metric] for r in records]))
    return {k:family_interval(xs) for k,xs in aggregates.items()}


def audit():
    from tokenizers import Tokenizer
    R.setup()
    d=R.read("design.json")
    binding=R.file_sha(R.OUT/"design.json")
    assert d["source"]==R.sources()
    secondary_plan=R.read("lexical-audit-plan.json")
    assert secondary_plan["design_sha"]==binding and secondary_plan["script_sha"]==R.file_sha(Path(lexical.__file__))
    tokpath=R.MODEL/"tokenizer.json"
    assert R.file_sha(tokpath)==d["model"]["files"]["tokenizer.json"]["sha256"]
    tok=Tokenizer.from_file(str(tokpath))
    overall={}
    cases=[]
    label_changes=[]
    illustrations={}
    for split in ("pilot","discovery","confirm"):
        if not (R.OUT/(split+"-summary.json")).exists():
            continue
        rows={r["id"]:r for r in R.material(d,split)}
        joined=[]
        sources=[]
        ids=set()
        for path in sorted(R.OUT.glob(split+"-[0-9][0-9][0-9].json")):
            data=json.loads(path.read_bytes())
            assert data["binding"]==binding
            sources.append(dict(name=path.name,sha256=R.file_sha(path)))
            for answer in data["rows"]:
                k=answer["id"]
                assert k not in ids
                ids.add(k)
                q=rows[k]
                assert tok.decode(answer["tokens"],skip_special_tokens=True)==answer["text"]
                s=M.score(answer["text"],q,answer["stopped"])
                meta={k:v for k,v in q.items() if k not in ("ids","spans","prompt")}
                joined.append(dict(meta,score=s,text=answer["text"],stopped=answer["stopped"],length=len(answer["tokens"])))
                if not s["correct"] or not s["joint"]:
                    cases.append(dict(meta,split=split,prompt=q["prompt"],text=answer["text"],score=s,
                                      tokens=answer["tokens"],stopped=answer["stopped"]))
        assert ids==set(rows)
        illustrations[split]=examples(joined)
        decision=R.read(split+"-summary.json")
        stats=R.summary(joined,split)
        assert json.loads(json.dumps(stats))=={k:v for k,v in decision.items() if k not in ("at","binding")}
        secondary_rows=[]
        for row in joined:
            obj=lexical.object_origin(row["text"],row)
            s=row["score"]
            secondary_rows.append(dict(row,score=dict(s,obj=obj,archive_both=obj=="archive" and s["num"]=="archive")))
            if s["obj"]!=obj:
                label_changes.append(dict(id=row["id"],split=split,text=row["text"],task=row["task_word"],
                                          archive=row["archive_word"],primary=s["obj"],secondary=obj))
        cells=defaultdict(Counter)
        totals=defaultdict(Counter)
        positions=defaultdict(Counter)
        for r in joined:
            s=r["score"]
            detail=dict(n=1,correct=int(s["correct"]),joint=int(s["joint"]),python=int(s["python"]),
                        stopped=int(r["stopped"]),unknown_object=int(s["obj"]=="other_or_missing"),
                        multiple_number=int(s["num"]=="multiple"),absent_number=int(s["num"]=="missing"),
                        archive_object_revised_number=int(s["obj"]=="archive" and s["num"]=="revised"),
                        task_object_archive_number=int(s["obj"]=="task" and s["num"]=="archive"),
                        task_object_old_number=int(s["obj"]=="task" and s["num"]=="old"),
                        archive_object_old_number=int(s["obj"]=="archive" and s["num"]=="old"))
            detail[archive_kind(s)]=1
            totals[f"{r['style']}/{r['requested']}"].update(detail)
            cells[f"{r['frame']}/{r['style']}/{r['requested']}"].update(detail)
            positions[f"{int(r['early'])}/{int(r['brackets'])}/{r['style']}/{r['requested']}"].update(detail)
        overall[split]=dict(sources=sources,n=len(joined),families=len({r["family"] for r in joined}),
                            max_tokens=max(r["length"] for r in joined),stopped=sum(r["stopped"] for r in joined),
                            decision=decision,totals=totals,frame_cells=cells,position_brackets=positions,
                            pairs=contrast(joined) if split!="pilot" else {},
                            secondary_delta={str(f):float(R.stat_panel(secondary_rows,f).mean())
                                             for f in M.FRAMES_BY_SPLIT[split]} if split!="pilot" else {})
    assert overall
    evidence=dict(at=R.stamp(),design_sha=binding,early_order_bracket_panels="between-family descriptive",
                  requested_values=["revised","archive"],styles=["prose","code"],splits=overall,
                  script_sha256=R.file_sha(Path(__file__)),lockbox_generations=0)
    R.save("lexical-review.json",dict(plan=R.file_sha(R.OUT/"lexical-audit-plan.json"),changes=label_changes))
    R.save("examples.json",illustrations)
    R.save("analysis.json",evidence)
    with R.staging_file(R.check_write(R.OUT/"counterexamples.json.gz",6*2**20)) as path:
        with gzip.open(path,"wt",encoding="utf-8") as f:
            json.dump(cases,f,ensure_ascii=False,separators=(",",":"))
    pilot=overall["pilot"]
    summary=[f"运行 {R.OUT.name}。采用本地 Qwen3-14B 原生 BF16，全词表贪心自由生成。最多 64 token，逐条保留实际停止标识。",
             f"单来源校准 {pilot['families']} 家族、{pilot['n']} 回答。按句式、档案样式和指定来源分格，内容与 Python 两项最低正确率分别为 "+
             " / ".join(f"{min(g[m] for g in pilot['decision']['cells'].values()):.1%}" for m in ("correct","python"))+
             f"。校准通过 {pilot['decision']['passed']}。"]
    for split in ("discovery","confirm"):
        if split not in overall:
            continue
        part=overall[split]
        summary.append(f"{split} 有 {part['families']} 家族、{part['n']} 个结果，{part['stopped']} 个生成到 EOS，最长 {part['max_tokens']} token。")
        for frame,rec in part["decision"]["effects"].items():
            l,h=rec["ci"]
            summary.append(f"句式 {frame} 在要求修订数据时，代码档案减叙述档案的对象和数字都来自档案的差为 {100*rec['delta']:+.2f} 个百分点，家族同时 99% 区间 [{100*l:+.2f}, {100*h:+.2f}]。")
        for k,g in sorted(part["totals"].items()):
            style,req=map(int,k.split("/"))
            summary.append(f"{('叙述档案','代码档案')[style]}，要求{('修订记录','档案')[req]}，共 {g['n']} 条。对象和数字同取档案 {g['both']}，仅对象 {g['object_only']}，仅数字 {g['number_only']}，两项均未唯一归给档案 {g['neither']}。指定来源两项均正确 {g['correct']}，严格全对 {g['joint']}。")
    if "confirm" in overall and not overall["confirm"]["decision"]["passed"]:
        summary.append("两套新句式未同时达到预定的 +15 个百分点及区间下界大于零的行为门。按冻结规则，本任务的 B26/H36 选源、等幅阻断和原生恢复均未启动。")
    elif "confirm" not in overall:
        summary.append("确认阶段未启动，按冻结门记录停点。")
    if "confirm" in overall:
        plot_confirmation(overall["confirm"])
    (R.OUT/"report.md").write_text("# 内容来源自由生成实验\n\n"+"\n\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary),flush=True)


if __name__=="__main__":
    audit()
