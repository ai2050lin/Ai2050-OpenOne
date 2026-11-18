"""Phase 2944. Freeze a new State/Note panel for the original Value-write dose."""

from datetime import datetime
from itertools import batched
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

import kv_bf_run as C
import kv_bf_study as B
import kv_address_materials as M


A = C.A
P = A.R.Base.Previous
OLD = C.OUT
ROOT = C.ROOT
OUT = ROOT / "tests/glm5/result/kv_reach_20261115_v1"
TMP = ROOT / "tests/glm5_temp/kv_reach_20261115_v1"
RUN = "KV-REACH-20261115-v1"
SEED = 29442026
PANEL = {"pilot": 12, "discover": 32, "confirm": 96}
WORDS = {
    "pilot": "satin velvet silk linen marble granite turquoise coral ivory emerald sapphire jewel ribbon lace cotton denim silver gold copper quartz",
    "discover": "cello violin harp flute trumpet piano guitar banjo drum cymbal horn tuba oboe organ recorder clarinet accordion lute fiddle bass reed bell shaker whistle keyboard saxophone trombone mandolin",
    "confirm": "rose tulip jasmine daisy violet orchid lily iris poppy lavender mint basil thyme sage dill fennel rosemary parsley coriander cumin clove cinnamon ginger turmeric saffron juniper cedar pine oak maple birch elm ash willow spruce fir walnut chestnut pecan almond hazel bamboo fern moss lotus clover nettle cactus ivy acacia",
}


def setup():
    C.OUT, C.TMP, C.RUN = OUT, TMP, RUN
    C.setup()


def stamp():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def prior(name):
    return P.load(OLD / name)


def words(rows):
    return {r[k] for r in rows for k in ("state", "note", "alt")}


def material(stage, tok, excluded):
    saved = M.WORDS, M.SEED, M.SIZE
    try:
        M.WORDS = {stage: " ".join(w for w in WORDS[stage].split() if w not in excluded)}
        M.SEED, M.SIZE = SEED, PANEL
        return M.material(stage, tok)
    finally:
        M.WORDS, M.SEED, M.SIZE = saved


def plan():
    assert not (OUT / "plan.json").exists()
    tok = A.Base.tok()
    excluded = set()
    for path in (OLD, OLD.with_name("kv_relay_20261115_v1"), OLD.with_name("kv_address_20261115_v1")):
        for f in path.glob("material-*.json"):
            excluded.update(words(P.load(f)))
    binding = {}
    for stage in PANEL:
        rows = material(stage, tok, excluded)
        vocabulary = words(rows)
        assert not vocabulary & excluded, sorted(vocabulary & excluded)
        excluded.update(vocabulary)
        for r in rows:
            A.encoded(tok, r)
        name = f"material-{stage}.json"
        A.save(name, rows)
        binding[stage] = dict(name=name, sha=B.sha(name), n=len(rows), families=PANEL[stage],
                              vocabulary=sorted(vocabulary), words_already_used=sorted(set(WORDS[stage].split())-vocabulary))
    identities = {m: B.model_identity(m) for m in ("q4", "q14")}
    assert identities["q4"] == prior("plan.json")["model_q4"]
    assert identities["q14"] == prior("q14-plan.json")["model"]
    old = prior("q14-selection.json")
    sources = (*B.SOURCES, "tests/glm5/kv_bf_q14.py", "tests/glm5/kv_reach_run.py", "tests/glm5/kv_reach_q14.py",
               "tests/glm5/kv_bf_audit.py", "tests/glm5/kv_address_audit.py", "tests/glm5/own_state_audit.py")
    dose = prior("plan.json")["dose"]
    A.save("plan.json", dict(run=RUN, stamp=stamp(), seed=SEED, materials=binding, identity=identities, dose=dose,
        prior_fixed_q14={k: old[k] for k in ("source", "receiver", "receiver_layer")},
        sources={k: A.digest(ROOT/k) for k in sources}, runtime=A.Base.versions(), provenance=A.Base.git_identity(),
        design="Authored controlled family panel. Eight views = negation/update × emitted order × word identity. All words across splits and past route panels are disjoint. Models compare on exactly these held-out prompts and answer-prefix tokens.",
        readout="Final token of the Code: prefix; both generated slot token identities causally visible. 4B greedy format, words, code and EOS per cell >=90%. 14B forced-prefix Code per cell >=90% plus separate exact greedy prefix and code. 4B-invalid fallback retains a canonical prefix and is flagged.",
        target="At all three nodes, original alpha target min(max(alpha_S, alpha_N), 2*min(alpha_S, alpha_N)); gain capped at 2.05; unchanged full norm, analytic, actual, direction and cross-slot matching checks. Precision arms use recovered natural Q at both receivers. Apply all precision variants only on the second receiver. Dynamic-Q arm is a separate native reference.",
        search=dict(source_layers=list(range(19, 28)), receiver_next_layer=True, max_shortlist=4,
            rules="Natural support across complete discovery families, all heads in adjacent blocks, distinct GQA groups within the receiver block. First two discovery feasibility tests are unrounded target norm >= original .008 and naturally available gain <= cap. Use exact original source and receiver engineering on shortlisted routes. Freeze one route by most late-arm complete discovery families, breaking ties by late-arm views then shortlist order. Fixed Phase 2943 route on same new holdout is reference. No confirmation outputs during 14B discovery."),
        primary="Paired held-out 14B family-level pass differences: reselected late vs old late and reselected late vs reselected native. Report both with 97.5% two-sided family-resampling intervals and paired exact tests (two prespecified endpoints at .025 each); other arms, continuous effects, behavior masks and comparisons are descriptive. A positive improvement requires interval wholly above zero. Never reduce dose gate to reach family count. At 96 independent paired families, idealized binomial power for paired changes .25 vs .05 is recorded; lexical sharing and selected cohort can reduce effective sample size.",
        power=A.Base.check_phase(.25, PANEL["confirm"], p_d=.05, alpha=.025),
        limits="Model-local heads, same RTX 5080, single model at a time. 4B keep original route and all methods. 14B decoder original BF16 weights with validated explicit causal-mask, layer-major forward. Neither repeated words nor prompt views are treated as independent sampling units."))


def verify():
    p = A.load("plan.json")
    assert p["runtime"] == A.Base.versions()
    assert all(B.sha(x["name"]) == x["sha"] for x in p["materials"].values())
    assert all(A.digest(ROOT/k) == h for k, h in p["sources"].items())
    return p


def q4(stage):
    import torch

    p = verify()
    assert B.model_identity("q4") == p["identity"]["q4"]
    if stage == "discover":
        assert A.load("q4-behavior-pilot.json")["passed"]
    if stage == "confirm":
        freeze = A.load("q14-freeze.json")
        assert freeze["plan"] == B.sha("plan.json")
    rows = A.load(p["materials"][stage]["name"])
    k = f"q4-behavior-{stage}.json"

    def work(task):
        if not (OUT/k).exists():
            A.save(k, A.behavioral(task, rows, stage + "-natural"))
        b = A.load(k)
        if not b["passed"] or stage != "confirm":
            return
        allowed = B.good_families(rows, b["rows"])
        written = {}
        packs = list(batched(zip(rows, b["rows"], strict=True), A.BATCH))
        start = time.monotonic()
        for j, pack in enumerate(packs, 1):
            r, o = [list(x) for x in zip(*pack, strict=True)]
            good = np.array([x["family"] in allowed for x in r])
            f = f"q4-confirm-{j:03d}.npz"
            if good.any() and not (OUT/f).exists():
                torch.cuda.empty_cache()
                metadata, results = B.Q.run_batch(task, r, o, good)
                B.write_np(f, B.arms_summary(metadata, results, p["dose"]["limits"]))
            if good.any():
                with np.load(OUT/f) as z:
                    assert np.array_equal(z["ids"], [x["id"] for x in r])
                written[f] = B.sha(f)
            P.stage_log("q4", "confirm-original-route", j, len(packs), start)
        A.save("q4-confirm.json", dict(shards=written, status="finished", plan=B.sha("plan.json"), behavior=B.sha(k)))

    torch.set_float32_matmul_precision("highest")
    A.with_task(work)


if __name__ == "__main__":
    setup()
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd.startswith("q4-"):
        q4(cmd[3:])
    else:
        raise ValueError(cmd)
