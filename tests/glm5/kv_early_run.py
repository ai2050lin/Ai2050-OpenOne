"""Phase 2948 materials and pre-holdout execution contract."""

from math import erfc,sqrt
from pathlib import Path
from statistics import NormalDist
import sys
import time

import kv_temporal_run as Prev
import kv_temporal_native as PN


A,B,P,N,O=Prev.A,Prev.B,Prev.P,Prev.N,Prev.O
ROOT=Prev.ROOT
OUT=ROOT/"tests/glm5/result/kv_early_20261116_v1"
TMP=ROOT/"tests/glm5_temp/kv_early_20261116_v1"
RUN="KV-EARLY-20261116-v1"
SEED=29482026
SIZES={"pilot":8,"discover":32,"confirm":96}
SLAB=12
CONF=1-.05/10
VOCAB={
    "pilot":"soccer tennis football baseball basketball cricket rugby hockey golf lacrosse squash archery fencing wrestling boxing skating skiing cycling rowing racing throwing diving running sprint marathon triathlon hurdle relay racket bat racket helmet ball target net puck glove ring track pitch court field pool award arena stadium derby tournament trophy medal trophy scoreboard whistle judge referee uniform ribbon baton jersey sprint lap penalty foul", 
    "discover":"sorrow anger fear joy pride shame fatigue anger hope trust delight guilt envy peace calm grief panic shock anxiety awe relief fury rage terror surprise regret pity laughter pleasure pain courage stamina energy mercy grace delight passion virtue humor honor doubt desire devotion wonder sorrow weakness strength warmth honesty sorrow justice glory triumph defeat victory failure triumph success loss wisdom passion patience anxiety sadness compassion hunger thirst sleep wake dream thought belief reason logic concern care love hate need wish want devotion distress sadness freedom", 
    "confirm":"triangle circle square hexagon oval cube sphere cone prism wedge rectangle diagonal line angle curve spiral helix arc radius diameter perimeter edge corner plane slope axis point center origin span segment figure shape polygon pentagon symmetry rotation mirror reflection distance width height depth length volume surface area sample mixture liquid solid powder water mist vapor smoke flame spark heat cold ice steam ash sediment soil dust fiber thread rope string cord cloth canvas niche pillar tower dome arch beam beam brace slab plate sheet wall stair staircase chimney gutter roof attic cellar balcony porch stoop garden lawn hedge fence grass field slope hill valley alley lane road river ditch pond stream brook creek shore coast harbor bay dune delta plain forest jungle grove meadow orchard cliff canyon gully glacier tundra desert ridge summit peak crater cavern tunnel cave shelf ledge archipelago basin peninsula headland tunnel sidewalk crossroad crossing bridge plaza park driveway ramp alley window door hinge handle bolt chain hinge joint axle wheel gear pump pulley belt spring lever hammer drill hook clasp clip pin nail screw wrench rod pole beam anchor bearing crank nozzle valve piston cylinder grip knot clamp pulley latch hinge frame scaffold grate lantern kettle grate", 
}


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.R.OUT,A.R.TEMP,A.R.RUN=OUT,TMP,RUN
    P.OUT,P.TMP,P.RUN=OUT,TMP,RUN
    P.BUDGET=200*2**20
    P.check_write(TMP,P.BUDGET,operation=RUN)
    P.setup()
    B.C.OUT=OUT


def excluded():
    used=Prev.excluded()
    for x in Prev.OUT.glob("material-*.json"):
        used.update(N.words(P.load(x)))
    return used


def material(stage,tok,used):
    saved=Prev.VOCAB,Prev.SEED,Prev.SIZES
    try:
        Prev.VOCAB,Prev.SEED,Prev.SIZES=VOCAB,SEED,SIZES
        return Prev.material(stage,tok,used)
    finally:
        Prev.VOCAB,Prev.SEED,Prev.SIZES=saved


def prior():
    p=P.load(Prev.OUT/"plan.json")
    f=P.load(Prev.OUT/"freeze.json")
    assert f["joint"]=="012" and f["features"][1]==20
    return p,f


def plan():
    assert not (OUT/"plan.json").exists()
    assert A.load("pilot-behavior.json")["passed"] and A.load("pilot-engine.json")["passed"]
    tok=A.Base.tok(); used=excluded()|N.words(A.load("material-pilot.json")); panels={}
    for stage in ("discover","confirm"):
        rows=material(stage,tok,used); words=N.words(rows)
        assert not words&used
        used|=words
        file=f"material-{stage}.json"; A.save(file,rows)
        panels[stage]=dict(file=file,sha=B.sha(file),n=len(rows),families=SIZES[stage],words=sorted(words))
    p,f=prior(); identity=B.model_identity("q14"); assert identity==p["snapshot"]
    codes=set(p["sources"])|{f"tests/glm5/kv_temporal_{s}.py" for s in ("run","native","study","audit")}
    codes.update(x.relative_to(ROOT).as_posix() for x in Path(__file__).parent.glob("kv_early_*.py"))
    z=NormalDist().inv_cdf((1+CONF)/2)
    power=lambda n: .5*(erfc((z-.4*sqrt(n))/sqrt(2))+erfc((z+.4*sqrt(n))/sqrt(2)))
    A.save("plan.json",dict(run=RUN,stamp=O.stamp(),seed=SEED,snapshot=identity,runtime=A.Base.versions(),materials=panels,
        sources={k:A.digest(ROOT/k) for k in sorted(codes)},pilot={k:B.sha(k) for k in ("material-pilot.json","pilot-behavior.json","pilot-engine.json")},
        phase2947_freeze=A.digest(Prev.OUT/"freeze.json"),features=f["features"],control_features=f["random"],c_controls=f["c_random"],
        design="New disjoint lexemes. Cross prompt role reversal with same codebook, short/long, field order. Eight views per family. Source prior to first State emission E, generated State G, Code query C. Only actual full-vocabulary greedy prefix receives behavioral eligibility. Per-cell >=90% and inference requires 80 complete families. Preserve actual altered token and first incompatible continuation separately.",
        discovery="Screen only L0–23: per head measure native prompt fact alpha*V write, signed projection on gold-vs-rejected State and native projected norm. Shortlist eight distinct layers: four largest absolute decision projections, four additional by physical dose, with one head per layer. Directly block their fact V at E on greedy prefix through G and stop after computing L25 K/V. Rank by median of normalized G changes at L25 H18's shared K/V group, equal weight to K norm ratio and projected V norm ratio. Choose three distinct layers (top three). Same-layer random head controls each from nearest norm quartile, two draws. Freeze before any confirm behavior outputs. Reuse Phase2947 finer 128+head+128 combination and its random groups and C same-slot heads, with same frozen scope. Publish source scores and controls.",
        intervention="E fact block zeros one or several native fact Value terms while keeping live Q/K/denominator; matching same-head restore and rejected-word control. At C measure L25H18 L25/G K/V writing; after E block selectively copy intact K, V or both at G for H18 only (all other heads/layers/positions remain live); reverse copy E-perturbed G K/V into intact H18, and compare matched other-head K/V rescue. C block zeros only generated G V in H18. Perform E+C block plus native same-source repair, Phase2947 downstream repair, and joint; also C-only native restore, three singletons, reusable combination and two random combinations. Exact term change by BF16 one-head SDPA difference with shape-matched native full baseline. Later Q evolves dynamically.",
        natural="Compare truly greedy role-reversal pairs from new prompts, and explicitly separate same-token replay. If E block produces a different recognized token, replace the proposal token only with the observed choice and greedily verify remaining suffix. If E block cannot induce a switch, include a diagnostic lexical-only intervention setting the original State token logit to -inf at E; greedily choose among remaining full vocab. Verify continuation after the actual selected token under unchanged model, preserving noncanonical or unresolved trajectories. Lexical diagnostic is not evidence that the earlier heads caused that switch or that alternate prompts isolate only that route. Bounded complementary rollout cap 16 prompts selected by fixed family order. Neither future rejected token nor expected P/Q is supplied as causal input in model choices.",
        gate=dict(alpha=1-CONF,confidence=CONF,margin=.10,n=80,physical_fraction=.20,joint_fraction=.50,
                  labels=["E State loss vs random","E change in L25 G KV vs random","E effect on final code vs random","Intact G KV rescue absolute","H18 G KV rescue vs random heads","H18 G reverse transport loss","H18 C block vs same-slot random","C final combination vs random","Joint early+end rescue absolute","Joint beyond best partial"],
                  scope="Two-sided adjusted family intervals, positive directional estimates. Physical threshold additionally relative 20% of targeted pre-control excursion (rather than .10 logit). Bridge requires physical effect + positive source code effect + absolute and specific rescue + reverse transport. Full joint additionally requires final specificity, combined rescue >=50% and beyond the larger of the two partial margins per view. No threshold changes on held-out data.",power_n96=power(96),power_n80=power(80)),
        storage="Windows D: only; 200 MiB initial peak audit estimate, reserve 20 GiB, check each layer and batch. Original 14B only, single GPU, full snapshot hash."))


def verify():
    p=A.load("plan.json")
    assert p["runtime"]==A.Base.versions()
    assert p["phase2947_freeze"]==A.digest(Prev.OUT/"freeze.json")
    assert all(A.digest(ROOT/k)==h for k,h in p["sources"].items())
    assert all(B.sha(x["file"])==x["sha"] for x in p["materials"].values())
    assert all(B.sha(k)==h for k,h in p["pilot"].items())
    return p


class Engine(N.Engine):
    def base_step(self,l,i,p,mode,operations):
        if mode in ("screen-e","probe","temporal","code","rollout"):
            from kv_early_native import step
            return step(self,l,i,p,mode,operations)
        return super().base_step(l,i,p,mode,operations)


def run(stage):
    import torch

    p=None if stage.startswith("pilot") else verify()
    snap=prior()[0]["snapshot"] if p is None else p["snapshot"]
    assert B.model_identity("q14")==snap
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    with P.GpuLease(),torch.inference_mode():
        clock=time.monotonic(); N.progress("early-load",0,1,clock)
        m=Engine(); N.progress("early-load",1,1,clock)
        try:
            if stage=="pilot-behavior":
                rows=material("pilot",m.tok,excluded()); A.save("material-pilot.json",rows)
                seen,_=m.observe(rows,"pilot-natural"); b=N.behavior(rows,seen); A.save("pilot-behavior.json",b)
                assert b["passed"],b["cells"]
                print("Pilot",b["cells"],flush=True)
            else:
                from kv_early_study import pilot,discover,confirm
                {"pilot-engine":pilot,"discover":discover,"confirm":confirm}[stage](m,p)
        finally:
            m.close()


if __name__=="__main__":
    sys.modules["kv_early_run"]=sys.modules[__name__]
    setup(); k=sys.argv[1]
    if k=="plan": plan()
    elif k=="audit":
        from kv_early_audit import main
        main()
    elif k in ("preview","close"):
        from kv_early_report import close
        close(k=="preview")
    else: run(k)
