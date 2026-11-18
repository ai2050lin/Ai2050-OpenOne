"""Phase 2947 material, snapshot binding, and dynamic-query driver."""

from pathlib import Path
import sys
import time

import numpy as np

import kv_origin_run as O
import kv_natural_run as N
import kv_natural_material as M


A,B,P=O.A,O.B,O.P
ROOT=O.ROOT
OUT=ROOT/"tests/glm5/result/kv_temporal_20261115_v1"
TMP=ROOT/"tests/glm5_temp/kv_temporal_20261115_v1"
RUN="KV-TEMPORAL-20261115-v1"
SEED=29472026
SIZES={"pilot":8,"discover":32,"confirm":112}
VOCAB={
    "pilot":"acorn hay grain oatmeal biscuit cracker toast jam yogurt gravy chili stew cereal pretzel cobbler noodle waffle pancake muffin donut cookie brownie chocolate caramel candy marshmallow biscuit dumpling steak chop sausage yogurt olive vinegar mustard ketchup salsa pickle radish lettuce cabbage cucumber pumpkin squash onion potato celery spinach kale",
    "discover":"storm hurricane typhoon tornado cyclone rain snow sleet frost hail fog mist dew cloud vapor wind breeze gale drought flood thunder lightning rainbow snowfall blizzard heat sunshine weather monsoon climate season tide horizon current surface aurora twilight dawn dusk smoke haze temperature humidity pressure gust shadow sunset daylight moon sun star planet eclipse comet asteroid",
    "confirm":"kernel buffer cursor socket database server router cache node vector matrix tensor bit byte binary decimal parser script module program function variable constant pointer address index integer float string boolean macro register token instruction pipeline device process thread stack heap block queue stream packet frame pixel image cipher key password credential signal channel packet port terminal console shell protocol kernel memory controller monitor display engine logger client proxy service worker listener relay debugger compiler interpreter emulator simulator sandbox firewall socket payload scanner reader writer editor browser desktop app graphics texture geometry mesh spline shader grid vertex polygon raster bitmap video codec encoder decoder pager scheduler counter timer clock latch switch engine circuit voltage cable wire fiber cluster array record log query table graph tree map dictionary list loop iterator generator lambda argument return result value predicate operand operator regex pattern sequence runtime budget allocation sentinel context modem connector switch cartridge drive disk chip keyboard processor core adapter gateway bridge router processor", 
}
END_COMPONENTS=((27,"mlp"),(28,"head"),(37,"mlp"))
UNITS=128
SLAB=12
CONFIDENCE=1-.05/9


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    # Fixed-phase outputs are summaries and selected native vectors. Reserve 240 MiB of new peak.
    P.check_write(OUT,240*2**20,operation=RUN)
    P.check_write(TMP,240*2**20,operation=RUN)
    O.A.R.OUT,O.A.R.TEMP,O.A.R.RUN=OUT,TMP,RUN
    O.A.R.setup()
    B.C.OUT=OUT


def excluded():
    seen=O.exclude()
    for p in O.OUT.glob("material-*.json"):
        seen.update(N.words(P.load(p)))
    return seen


def add_sites(tok,row):
    z=M.token_plan(tok,row)
    chat=tok.apply_chat_template([dict(role="user",content=row["prompt"])],tokenize=False,
                                add_generation_prompt=True,enable_thinking=False)
    enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
    assert enc.input_ids==z["ids"][:z["start"]]
    start=chat.index(row["prompt"])+row["prompt"].index("\n\n")+2
    stop=chat.index("\nNote entry:",start)
    word=row["values"]["State"]
    idx=lambda word:[i for i,(l,r) in enumerate(enc.offset_mapping) if start-1<=l<stop and chat[l:r].strip()==word]
    fact,other=idx(word),idx(row["rejected"])
    after=chat.index("\nCodebook:\n",stop)+len("\nCodebook:\n")
    b=chat.index("\n"+word+" ->",after-1)+1
    book=[i for i,(u,v) in enumerate(enc.offset_mapping) if v>b and u<b+len(word)]
    assert len(fact)==len(other)==1 and book and fact[0]<min(book)<z["start"]<=z["slots"][0]-1
    row["token"],row["origins"]=z,dict(fact=fact,other=other,book=book)
    row["state_ids"]=[z["ids"][z["slots"][0]],tok.encode(" "+row["rejected"],add_special_tokens=False)[0]]
    return row


def material(stage,tok,blocked):
    old=M.VOCAB,M.SEED
    try:
        M.VOCAB,M.SEED=VOCAB,SEED
        rs=M.generate(stage,tok,blocked,families=SIZES[stage])
    finally:
        M.VOCAB,M.SEED=old
    out=[]
    for j in range(SIZES[stage]):
        part=rs[j*8:(j+1)*8]
        comp=(j//4)%2
        for side in (0,1):
            for base in [r for r in part if r["compete"]==comp]:
                row=dict(base,side=side,id=f"{base['family']}-{side}{base['distance']}{base['order']}")
                if side:
                    s,x=row["values"]["State"],row["rejected"]
                    p=row["prompt"]
                    if row["context"]:
                        a="Decision: Adopt the proposal."; b="Decision: Keep the original."
                        row["prompt"]=p.replace(a,b,1) if a in p else p.replace(b,a,1)
                    else:
                        row["prompt"]=p.replace(f"includes {s}.",f"excludes {s}.",1).replace(f"excludes {x}.",f"includes {x}.",1)
                    row["values"]=dict(row["values"],State=x)
                    row["rejected"]=s
                    row["code"]="Q" if row["code"]=="P" else "P"
                    row["prefix"]=row["prefix"].replace(f"State: {s}",f"State: {x}",1)
                book=row["prompt"].split("Codebook:\n",1)[1].split("\n\n",1)[0]
                cmap=dict(line.split(" -> ") for line in book.splitlines())
                assert cmap[row["values"]["State"]]==row["code"] and cmap[row["rejected"]]!=row["code"]
                row["competitors"]=[k for k in M.FIELDS[1:] if cmap[row["values"][k]]!=row["code"]]
                out.append(add_sites(tok,row))
    assert len(out)==8*SIZES[stage] and len({r["prompt"] for r in out})==len(out)
    return out


def plan():
    assert not (OUT/"plan.json").exists()
    b=A.load("pilot-behavior.json")
    assert b["passed"],b["cells"]
    assert A.load("pilot-runner.json")["passed"]
    tok=A.Base.tok(); taken=excluded()|N.words(A.load("material-pilot.json")); m={}
    for stage in ("discover","confirm"):
        rows=material(stage,tok,taken)
        vocab=N.words(rows)
        assert not vocab&taken
        taken|=vocab
        name=f"material-{stage}.json"; A.save(name,rows)
        m[stage]=dict(name=name,sha=B.sha(name),n=len(rows),families=SIZES[stage],vocab=sorted(vocab))
    p=P.load(O.OUT/"plan.json")
    ident=B.model_identity("q14"); assert p["model"]==ident
    src=set(p["sources"]) | {"tests/glm5/kv_origin_run.py","tests/glm5/kv_origin_trace.py","tests/glm5/kv_origin_study.py"}
    src.update(q.relative_to(ROOT).as_posix() for q in Path(__file__).parent.glob("kv_temporal_*.py"))
    A.save("plan.json",dict(run=RUN,stamp=O.stamp(),snapshot=ident,seed=SEED,materials=m,runtime=A.Base.versions(),
        sources={k:A.digest(ROOT/k) for k in sorted(src)},pilot={k:B.sha(k) for k in ("material-pilot.json","pilot-behavior.json","pilot-runner.json")},
        material="Three disjoint new vocabularies, exclude previous related phases. Eight views = fact/decision role swap (same codebook, different true State and code) × receipt length × emitted field order. Rules/competition and codebook order counterbalanced across families. Family as resampling unit. Every original step through Code: unforced greedy prefix verified with only observed whitespace variants and at least 90% in every cell; report original errors. At least 80 complete confirmation families for a positive claim.",
        stages="E=last query before the model first emits the verified single-token State. G=that token's activation/cache position. C=last Code: query. At E zero only role-bearing prompt fact V terms in a chosen head, preserving Q/K/denominator/other heads. At G do whole native K/V memory substitution from the intact same sample only for reads by positions after G; Q remains live. From the earliest tested E layer (or L25 if earlier), G memory zero gates V and keeps K, with a same-shape replay. Earlier G memory is intact and unaffected by E. Report conditional same-token memory effect and separately any actual argmax State or suffix divergence. Replay up to 32 recognized one-token emitted State alternative prefixes in row order; never treat a broken greedy suffix as natural. At C remove generated-State V only in L25H18, reinsert native, and apply one or more fixed downstream restorations.",
        discovery="All layers/40 heads screened at E with signed native output contribution from factual alpha*V projected toward true-vs-rejected State readout; six layers/one ranked head per layer take direct E factual blocking. Freeze strongest three heads at distinct layers after exact head screening. Same-layer random head for each matched factual projected norm. For code candidate test the 7 nonempty subsets of L28H20, full L27 MLP, full L37 MLP by exact native restore under C block. At both MLP sites rank 128 units by signed native activation-difference times down column readout with C block, and repeat 7 subsets using unit sets. Freeze best fine pair or triple and strongest singleton by mean paired contribution. Save all discovery scores and fit targets then hash freeze before confirm. Compare full modules descriptively and fine sets with random control modules (matching actual unit up/down dose) and combination specificity.",
        primary=dict(confidence=CONFIDENCE,floor=.10,n=80,labels=["E factual block loss vs same-layer random","E native restore vs E block","E block code loss vs random under fixed original token history","G native memory rescue vs E block","G rescue above negative memory (generated Spare)","C State V block vs matched heads","C fine combined rescue above random fine combination","C fine combined rescue above strongest frozen singleton","cross-stage joint recovery absolute"],
                     rule="Family means and two-sided 99.444% paired bootstrap intervals. Positive if >=.10 and lower bound >0 with n>=80. Actual unforced State/code flips tracked separately. G memory mediation claim needs E effect on code and G native above Note plus absolute positive and validated patch; whole candidate closure needs joint recovery and >=50% recovery under combined E and C blockade, with controls. Report all scope limitations."),
        technical="BF16 snapshots and original module weights, native SDPA per-head difference to full head at causal site with original Q/K. Full-vocabulary argmax and raw P/Q/State agreement with untreated validation under fixed acceptance. Identical-physics rescue sentinel, logged numeric mismatch exclusion by complete family or original-shape repair. No retuning on holdout. Disk preflight 240 MiB and 20-GiB reserve."))


def verify():
    p=A.load("plan.json")
    assert p["runtime"]==A.Base.versions()
    assert all(B.sha(v["name"])==v["sha"] for v in p["materials"].values())
    assert all(A.digest(ROOT/k)==h for k,h in p["sources"].items())
    assert all(B.sha(k)==v for k,v in p["pilot"].items())
    return p


class Engine(N.Engine):
    def base_step(self,l,i,p,mode,operations):
        if mode in ("temporal","code","early","screen-e","scan-c","pilot-runner"):
            from kv_temporal_native import step
            return step(self,l,i,p,mode,operations)
        return super().base_step(l,i,p,mode,operations)


def run(cmd):
    import torch

    p=None if cmd.startswith("pilot") else verify()
    bound=P.load(O.OUT/"plan.json")["model"] if p is None else p["snapshot"]
    assert B.model_identity("q14")==bound
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    with P.GpuLease(),torch.inference_mode():
        clock=time.monotonic(); N.progress("temporal-load",0,1,clock)
        m=Engine(); N.progress("temporal-load",1,1,clock)
        try:
            if cmd=="pilot-behavior":
                rows=material("pilot",m.tok,excluded()); A.save("material-pilot.json",rows)
                seen,_=m.observe(rows,"pilot-natural"); b=N.behavior(rows,seen); A.save("pilot-behavior.json",b)
                assert b["passed"],b["cells"]
                print("Pilot behavior",b["cells"],flush=True)
            else:
                from kv_temporal_study import pilot,discover,confirm
                {"pilot-runner":pilot,"discover":discover,"confirm":confirm}[cmd](m,p)
        finally:
            m.close()


if __name__=="__main__":
    sys.modules["kv_temporal_run"]=sys.modules[__name__]
    setup(); cmd=sys.argv[1]
    if cmd=="plan": plan()
    else: run(cmd)
