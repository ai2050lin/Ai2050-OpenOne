"""Phase 2946: trace the downstream effects of native L25H18 at Code:."""

from datetime import datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

import kv_natural_run as Prev
import kv_natural_material as M


A,B,P=Prev.A,Prev.B,Prev.P
ROOT=Prev.ROOT
OUT=ROOT/"tests/glm5/result/kv_origin_20261115_v1"
TMP=ROOT/"tests/glm5_temp/kv_origin_20261115_v1"
RUN="KV-ORIGIN-20261115-v1"
SEED=29462026
SIZE={"pilot":8,"discover":32,"confirm":128}
WORDS={
    "pilot":"harp flute clarinet piano harp keyboard mandolin violin cello trumpet trombone cymbal banjo trombone cymbal tuba fiddle bass gong oboe drum pipe flute pedal recorder bell reed accordion amp switch adapter stand headset speaker charger cable joystick sensor dial lens tablet buzzer printer scanner cartridge remote alarm battery plug resistor diode jack relay rotor stator motor bezel microphone fan", 
    "discover":"ankle wrist elbow knee thigh calf femur tibia skull spine chin neck shoulder finger toe nose tongue cheek scalp throat heart lung brain liver stomach pancreas kidney bladder spleen tendon cartilage ligament vein artery retina pupil iris eyelid lip thumb biceps triceps torso pelvis collar abdomen muscle jaw teeth molar gum enamel bone nerve tendon skin knuckle heel wrist appendix intestine colon organ gland airway tissue limb",
    "confirm":"wallet passport stamp ticket invoice receipt voucher coupon fee fare credit debit bond equity stock cash coin bank budget loan mortgage tax lease permit license contract draft policy record log balance ledger audit trust tariff grant fund pension savings price rent sale bid fee margin profit loss trade import export patent trademark share quota rate reserve warrant premium option dividend security deposit advance debt allowance subsidy bond yield rebate surcharge toll card token check cheque note draft bill account branch register folder archive notice letter memo report form badge award prize medal banner flag parcel package freight cargo shipment crate manifest docket summons claim deed title license ballot petition survey transcript consent pledge charter bill cash envelope packet dispatch mail booklet file slip stub spreadsheet schedule calendar", 
}
LAYERS=tuple(range(25,40))
NUNIT=96
CONFIDENCE=1-.05/10


def stamp():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.configure()
    B.C.OUT=OUT


def material(key,tok,excluded):
    old=M.VOCAB,M.SEED
    try:
        M.VOCAB,M.SEED=WORDS,SEED
        rows=M.generate(key,tok,excluded,families=SIZE[key])
    finally:
        M.VOCAB,M.SEED=old
    for row in rows:
        row["token"]=z=M.token_plan(tok,row)
        state=z["ids"][z["slots"][0]]
        chat=tok.apply_chat_template([dict(role="user",content=row["prompt"])],tokenize=False,
                                    add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
        word=row["values"]["State"]
        assert enc.input_ids==z["ids"][:z["start"]]
        code_at=chat.index("Codebook:\n")+len("Codebook:\n")
        fact_first=chat.index(row["prompt"])+row["prompt"].index("\n\n")+2
        fact_end=chat.index("\nNote entry:",fact_first)
        factual=[j for j,(tid,(lo,hi)) in enumerate(zip(enc.input_ids,enc.offset_mapping,strict=True))
                 if tid==state and fact_first-1<=lo<fact_end and chat[lo:hi].strip()==word]
        # Codebook row starts after a newline and can have a different token ID from the in-line word.
        key_at=chat.index("\n"+word+" ->",code_at-1)+1
        code=[j for j,(lo,hi) in enumerate(enc.offset_mapping) if hi>key_at and lo<key_at+len(word)]
        assert len(factual)==1 and len(code)>=1,(row["id"],factual,code,row["prompt"])
        assert max(factual)<min(code)<z["start"]<z["slots"][0]<len(z["ids"])-1
        row["origins"]=dict(fact=factual,book=code)
        # Match a non-State factual word from the same decision, explicitly not the dictionary occurrence.
        alt=row["rejected"]
        other=[j for j,(lo,hi) in enumerate(enc.offset_mapping) if fact_first-1<=lo<fact_end and chat[lo:hi].strip()==alt]
        assert len(other)==1,(row["id"],other)
        row["origins"]["other"]=other
    return rows


def exclude():
    used=Prev.old_words()
    for f in Prev.OUT.glob("material-*.json"):
        used.update(Prev.words(P.load(f)))
    return used


def plan():
    assert not (OUT/"plan.json").exists()
    b=A.load("pilot-behavior.json")
    assert b["passed"],b["cells"]
    assert A.load("pilot-calibration.json")["passed"]
    tok=A.Base.tok()
    used=exclude()|Prev.words(A.load("material-pilot.json"))
    rows={}
    for k in ("discover","confirm"):
        rr=material(k,tok,used)
        words=Prev.words(rr)
        assert not (words&used)
        used.update(words)
        name=f"material-{k}.json"
        A.save(name,rr)
        rows[k]=dict(name=name,sha=B.sha(name),families=SIZE[k],vocab=sorted(words))
    prior=P.load(Prev.OUT/"plan.json")
    identity=B.model_identity("q14")
    assert identity==prior["identity"]
    sources=set(prior["sources"]) | {"tests/glm5/kv_natural_run.py","tests/glm5/kv_natural_material.py"}
    sources.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob("kv_origin_*.py"))
    A.save("plan.json",dict(run=RUN,stamp=stamp(),model=identity,runtime=A.Base.versions(),seed=SEED,sources={k:A.digest(ROOT/k) for k in sorted(sources)},
        pilot={k:B.sha(k) for k in ("material-pilot.json","pilot-behavior.json","pilot-calibration.json")},materials=rows,
        design="Reused validated Phase 2945 grammar with distinct unexposed role vocabularies and family sets. 8 views: code competitors × emitted receipt distance × State/Note order. Balanced rule and code letter. Causal verified unforced greedy through Code:, adapt only observed whitespace, check 90% correctness per cell. Require >=80 whole held-out families for main inference.",
        positions="Single last Code: query. Distinct key/value positions: fact is the in-prompt role-bearing mention of the active State; book is its complete codebook key span (may have a different tokenizer ID); gen is the State word naturally emitted by model; other is the rejected word in the factual prompt. Block a chosen head's effective V at these positions at this query, keeping live Q, K, all other V and normalization denominator. Same-layer random heads at the same positions and same H18 rejected factual word are controls. Both prompt positions are also compared jointly; native rescue after combined exclusion restores only fact, book or gen separately.",
        discovery="With H18 blocked at Code:, restore from matched intact sham each attention after L25 and each MLP L25–39 separately; rank the two module types by mean whole-family signed code-margin recovery. No whole residual replacement. Scan all 40 individual heads at best-attention layer by the same exact restoration; rank from discovery. At best MLP rank 96 units by mean signed (natural minus blocked activation) times native down column dotted with signed LM P/Q weight. Freeze one head, one 96-unit group, the two module sites, and two native projected-dose-matched random head/unit groups. Save trace and full search scores. Show whether top candidates explain 20% of upstream loss on holdout; if none do, no major experimentally isolated mediator in this search.",
        readout="Integrated gradient of continuous final RMSNorm plus unembed along the intact-to-blocked final residual segment applied to exact realized BF16 attention/MLP residual increments; verify telescoping in FP64. This is endpoint accounting, report BF16 score drift separately. Counterfactual restores and blocks tested independently with dynamic Q/K.",
        primary=dict(confidence=CONFIDENCE, floor=.10, coverage=80, fraction=.20,
            list=["fact loss above same-position matched heads","book loss above matched heads","gen loss above matched heads",
                  "combined prompt minus generated loss (two-sided)","fact native gain within all-source block","gen native gain within all-source block",
                  "frozen-head rescue vs matched head rescue","frozen-head absolute rescue","frozen-MLP-group rescue vs matched groups","frozen-MLP-group absolute rescue"],
            gate="99.5% paired family bootstrap interval excludes zero in specified direction (two-sided only for prompt-minus-gen), effect >=0.10 logits, n>=80. Calling a major mediator additionally needs absolute net rescue at least 20% of whole upstream loss with positive specificity, its standalone knockout hurting the signed margin, and physical rescue verification. Module-level scan endpoints and interactions are descriptive."),
        calibration=dict(untreated_vs_packed_raw_logit=Prev.SHAM_TOL,same_shape_reversible=Prev.RESTORE_TOL,
                         description="Full-prefix versus packed greedy next token agrees; separately verify each physical V-zero and recovery, full-block same shape controls. Keep packed sham matched to intervention. Zeroing a V preserves attention probabilities and denominator, rather than remasking the slot.")))


def verify():
    p=A.load("plan.json")
    assert p["runtime"]==A.Base.versions()
    assert all(A.digest(ROOT/k)==v for k,v in p["sources"].items())
    assert all(B.sha(x["name"])==x["sha"] for x in p["materials"].values())
    assert all(B.sha(k)==v for k,v in p["pilot"].items())
    return p


def resolved(rows,behavior):
    allowed=set(behavior["eligible"])
    s={x["id"]:x for x in behavior["rows"]}
    return [dict(r,token=s[r["id"]].get("resolved_token",r["token"])) for r in rows if r["family"] in allowed]


class Engine(Prev.Engine):
    def base_step(self, layer, i, p, mode, commands):
        if mode not in ("module-scan","unit-scan","confirm-path","pilot-path") or i<25:
            return super().base_step(layer,i,p,mode,commands)
        from kv_origin_trace import step
        step(self,layer,i,p,mode,commands)


def pilot_behavior(m):
    rows=material("pilot",m.tok,exclude())
    A.save("material-pilot.json",rows)
    observations,_=m.observe(rows,"origin-pilot")
    b=Prev.behavior(rows,observations)
    A.save("pilot-behavior.json",b)
    if not b["passed"]:
        raise RuntimeError(b["cells"])
    print("Pilot",b["cells"],flush=True)


def pilot(m):
    from kv_origin_trace import panel,pilot_arms
    if not (OUT/"pilot-behavior.json").exists():
        pilot_behavior(m)
    b=A.load("pilot-behavior.json")
    rows=A.load("material-pilot.json")
    assert b["passed"]
    sel=resolved(rows,b)[:8]
    a,meta=panel(m,sel,"pilot-path",pilot_arms(),b)
    A.save("pilot-calibration.json",dict(passed=True,n=len(sel),max_pq_drift=float(abs(a["pq"][:,0]-a["untreated"]).max()),
                 max_integral_error=float(abs(a["readout_closure"]).max()),shapes={k:list(x.shape) for k,x in a.items()},
                 max_v_response_error=float(max(z["origin_err"].max() for z in meta))))


def run(stage):
    import torch
    p=None if stage in ("pilot","pilot-behavior") else verify()
    snapshot=P.load(Prev.OUT/"plan.json")["identity"] if p is None else p["model"]
    assert B.model_identity("q14")==snapshot
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    with P.GpuLease(),torch.inference_mode():
        clock=time.monotonic()
        Prev.progress("origin-load",0,1,clock)
        m=Engine()
        Prev.progress("origin-load",1,1,clock)
        try:
            if stage=="pilot-behavior":
                pilot_behavior(m)
            elif stage=="pilot":
                pilot(m)
            else:
                from kv_origin_trace import discover,confirm
                {"discover":discover,"confirm":confirm}[stage](m,p)
        finally:
            m.close()


if __name__=="__main__":
    sys.modules["kv_origin_run"]=sys.modules[__name__]
    setup()
    cmd=sys.argv[1]
    if cmd=="plan":
        plan()
    else:
        run(cmd)
