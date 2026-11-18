"""Phase 2949: materials, frozen execution contract, and original 14B entry point."""

from pathlib import Path
from statistics import NormalDist
import sys
import time

import kv_early_run as Prev


A, B, P, N, O = Prev.A, Prev.B, Prev.P, Prev.N, Prev.O
ROOT = Prev.ROOT
OUT = ROOT / "tests/glm5/result/kv_prefix_20261116_v1"
TMP = ROOT / "tests/glm5_temp/kv_prefix_20261116_v1"
RUN = "KV-PREFIX-20261116-v1"
SEED = 29492026
SIZES = {"pilot": 8, "discover": 32, "confirm": 104}
BUDGET = 190 * 2**20
SOURCES = ("accepted", "rejected", "role-text", "book-lines", "rule", "format")
SLAB = 64
COORDS = 16
ROLLOUT_CASES = 24
ROLLOUT_TOKENS = 80
ROLLOUT_PASSES = 20
ENDPOINTS = (
    "early-state", "state-route", "early-code", "early-local", "kv-block", "kv-rescue",
    "kv-rescue-specific", "native-full-rescue", "reverse", "state-switch", "state-route-switch",
)
CONFIDENCE = 1 - .05 / len(ENDPOINTS)
VOCAB = {
    "pilot": "violin cello viola piano trumpet trombone clarinet flute oboe bassoon tuba harp guitar mandolin saxophone banjo drum cymbal gong piccolo tambourine accordion harmonica melody rhythm tempo beat octave chord note scale chorus verse anthem sonata opera trio quartet concert symphony stage audience tune song aria fugue organ keyboard music rehearsal composer conductor baton orchestra soprano alto tenor falsetto baritone acoustic string percussion recorder brass duet ensemble solo harmony",
    "discover": "oak maple birch pine cedar spruce elm willow ash beech walnut chestnut hickory cypress fir alder yew poplar aspen sycamore juniper hemlock holly palm bamboo fern moss ivy rose tulip lily orchid violet daisy clover thyme sage mint basil oregano parsley lavender rosemary hollyhock iris lotus jasmine lilac peony dahlia honeysuckle magnolia azalea rhododendron heather olive fig cherry plum peach pear lemon orange lime berry grape mango kiwi papaya coconut avocado blossom pollen petal pistil sepal nectar flower bush shrub bush grass stem root seed shoot leaf bud bark thorn hedge branch trunk",
    "confirm": "antelope badger beaver buffalo camel cheetah coyote dolphin eagle falcon ferret finch gazelle gecko giraffe gopher gorilla hamster hawk heron hippo hyena iguana jaguar kangaroo koala leopard lemur lion lizard llama lobster lynx mackerel manta marmot mongoose moose mouse mule octopus orca ostrich otter owl panda panther parrot peacock pelican penguin pheasant pigeon pike pony porcupine python rabbit raccoon raven reindeer rhino rooster salmon sardine scorpion seal shark sheep shrimp skunk sloth snail sparrow squid squirrel stork swan tiger toad trout turtle viper vole walrus weasel whale wolf wombat zebra caterpillar grasshopper dragonfly beetle cricket locust mantis hornet wasp mosquito butterfly moth bee termite spider tarantula aphid slug worm clam oyster mussel scallop crab prawn carp perch halibut flounder herring anchovy cod tuna bass guppy seahorse eel sturgeon catfish barracuda swordfish albatross bluebird canary crow cuckoo dove duck goose gull ibis kite kingfisher magpie robin swift swallow woodpecker wren yak bison donkey ox donkey calf ram goat hog boar pig antelope deer elk buck doe fox hare rat chipmunk rabbit manatee beluga narwhal pup cub fawn colt foal chick hen drake kitten pup dog cat lynx bobcat puma cougar jaguar monkey macaque baboon gibbon chimp orangutan ape elephant rhino crocodile alligator caiman cobra adder rattlesnake tortoise terrapin newt salamander frog tadpole hydra coral jellyfish plankton starfish sponge",
}


def setup():
    A.OUT, A.TEMP, A.RUN = OUT, TMP, RUN
    A.R.OUT, A.R.TEMP, A.R.RUN = OUT, TMP, RUN
    P.OUT, P.TMP, P.RUN, P.BUDGET = OUT, TMP, RUN, BUDGET
    P.check_write(OUT, BUDGET, operation=RUN)
    P.check_write(TMP, BUDGET, operation=RUN)
    P.setup()
    B.C.OUT = OUT


def excluded():
    seen = Prev.excluded()
    for path in Prev.OUT.glob("material-*.json"):
        seen.update(N.words(P.load(path)))
    return seen


def source_indices(tok, row):
    chat = tok.apply_chat_template([dict(role="user", content=row["prompt"])], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == row["token"]["ids"][:row["token"]["start"]]
    text = row["prompt"]
    base = chat.index(text)

    def indices(begin, end):
        return [i for i, (l, r) in enumerate(enc.offset_mapping)
                if r > base + begin and l < base + end and r > l]

    fact_start = text.index("\n\n") + 2
    fact_stop = text.index("\nNote entry:")
    rstart = text.index("State is")
    rend = text.index(" Copy Note")
    entries = tuple(row["origins"][s][0] for s in ("fact", "other"))
    role_text = [i for i in indices(fact_start, fact_stop) if i not in entries]
    book_start = text.index("\nCodebook:\n") + len("\nCodebook:\n")
    codes = []
    for word in (row["values"]["State"], row["rejected"]):
        a = text.index("\n" + word + " ->", book_start - 1) + 1
        assert a == book_start or text[a - 1] == "\n"
        codes.extend(indices(a, text.index("\n", a)))
    fmt_start = text.index("State: <word>")
    groups = dict(zip(SOURCES, ([entries[0]], [entries[1]], role_text, sorted(set(codes)),
                                indices(rstart, rend), indices(fmt_start, fmt_start + len("State: <word>"))), strict=True))
    flat = [i for x in groups.values() for i in x]
    assert all(groups.values()) and len(flat) == len(set(flat)) and max(flat) < row["token"]["start"]
    return groups


def material(stage, tok, used):
    prior = Prev.VOCAB, Prev.SEED, Prev.SIZES
    try:
        Prev.VOCAB, Prev.SEED, Prev.SIZES = VOCAB, SEED, SIZES
        rows = Prev.material(stage, tok, used)
    finally:
        Prev.VOCAB, Prev.SEED, Prev.SIZES = prior
    for row in rows:
        row["sites"] = source_indices(tok, row)
    return rows


def model_identity():
    identity = B.model_identity("q14")
    assert identity == P.load(Prev.OUT / "plan.json")["snapshot"]
    return identity


def plan():
    assert not (OUT / "plan.json").exists()
    assert A.load("pilot-behavior.json")["passed"] and A.load("pilot-engine.json")["passed"]
    tok = A.Base.tok()
    used = excluded() | N.words(A.load("material-pilot.json"))
    panels = {}
    for stage in ("discover", "confirm"):
        rows = material(stage, tok, used)
        words = N.words(rows)
        assert not used & words
        used |= words
        filename = f"material-{stage}.json"
        A.save(filename, rows)
        panels[stage] = dict(file=filename, sha=B.sha(filename), rows=len(rows), families=SIZES[stage], words=sorted(words))
    prior = P.load(Prev.OUT / "plan.json")
    scripts = {*prior["sources"], "tests/glm5/kv_early_run.py", "tests/glm5/kv_early_audit.py"}
    scripts.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob("kv_prefix_*.py"))
    n = SIZES["confirm"]
    normal = NormalDist()
    z = normal.inv_cdf((1 + CONFIDENCE) / 2)
    power = {k: normal.cdf(.4 * k**.5 - z) + normal.cdf(-.4 * k**.5 - z) for k in (80, n)}
    A.save("plan.json", dict(
        run=RUN, time=O.stamp(), seed=SEED, snapshot=model_identity(), runtime=A.Base.versions(), materials=panels,
        sources={k: A.digest(ROOT / k) for k in sorted(scripts)}, previous_freeze=A.digest(Prev.OUT / "freeze.json"),
        pilot={k: B.sha(k) for k in ("material-pilot.json", "pilot-behavior.json", "pilot-engine.json")},
        storage=dict(peak_bytes=BUDGET, reserve_gib=20, volume="D:"),
        positions="E is the first State emission query, G the State token's position, C the final Code query. Search early E writes at L0..23. Six disjoint prompt masks: accepted and rejected single tokens, remaining role/decision text, both competing codebook lines, rule clause, State format marker. Source Value zeros preserve denominators and all non-target queries. Exact native restoration and same-layer/source head controls.",
        materials_spec="Local generated visible templates, new lexical pools disjoint between pilot/discovery/confirmation and Phase 2937–48 route studies within the exclusion ledger. Eight views per family from role swap with same codebook, field order and distance. Require >=90% per behavior cell and whole families for inference. Preserve any original or replay divergence.",
        adjoint="At native C, compute the derivative of logit(Q)-logit(P) with respect to L25H18 pre-O output by layer-streamed BF16 autodiff of one live query, keeping earlier K/V causal and fixed for later layers. Readout is a local tangent, calibrated against actual full-model branch results. Each live E-branch has its own L25 C query, G key/value, competitor keys and values. Allocate exact FP32 restoration along the secant of alpha_G and the two-order K/V average, then confirm with native BF16 edits.",
        search="Find for each prompt mask the head with strongest mean aligned State-direction contribution and the largest native write dose on another layer. Run 12 single-block arms with live later Q and audit actual State, actual P/Q and downstream-direction native-K/V coordinates. Build three-layer coalitions with distinct source masks from three greedy rankings: signed tangent K/V rescue, actual State decision loss, and mean of both standardized ranks. Exact discovery replay of these coalitions. Freeze the strongest signed native H18 K/V coalition and the strongest State coalition, preferring recognized actual alternative switches then mean margin loss for the latter. For each freeze two same-layer/source controls matched on E native output dose from the closest quartile.",
        coordinates="Under the primary E coalition freeze 16 native post-RoPE K and 16 native V indices, sorted by mean role-signed contribution to restoring the true P/Q margin in discovery. For each K and V separately, draw two non-target sets from nearest quartile using intact-zero and restore dose, without repeated indices within a set. Freeze reverse selected coordinates, whole-slot rescue/zero comparator, exact K-only/V-only/both rescue, selected-coordinate zero and native self restore. Each edit only replaces that G slot at C for H18; other heads, positions and later queries stay live.",
        natural="State change uses full-vocabulary top token, with recognized alternative and unrecognized switch classified separately. Conditions retain fixed State token for propagation calculations, and conditional Code is never counted as natural once any prefix token diverges. For each E group and matched random E control up to 24 switched accepted views in frozen material order, greedily replay actual selected token to a Code line, EOS, or fixed budget of 80 new tokens / 20 speculative passes. Identical control groups share the same actual replay. A speculative token enters the verified history only after an argmax match, or is replaced by the first model-emitted token. Frozen same-lexeme role swaps separately establish natural P/Q. Before freeze a native no-edit generation checks the same replay path. No synthetic token suppression fallback in the primary evidence.",
        gate=dict(confidence=CONFIDENCE, min_family=80, minimum_logit=.10, minimum_switch=.02, labels=ENDPOINTS,
                  power=power, route="Early bridge needs early-code and early-local, full native rescue, coordinate rescue versus sham and controls, reverse transport and negative-dose comparison from kv-block. True lexical switching needs the respective excess full-vocabulary switch frequency above controls. Report recognized and reached C separately. Include family-block bootstrap and leave-one-lexeme sensitivity. In heldout analysis the label is used only to sign an observed contrast."),
        numeric="Preserve packed versus natural P/Q differences <=1 logit with exact full argmax and natural prefix. Exact self native restores and unchanged E State logits for downstream-only interventions, equality of untouched and source coordinates, finite differences audited in physical head-output space. Normalized one-head response error <.05, coordinate sum closure <.001, native Code adjoint replay residual relative error <=.05. Reject entire affected family for sham failure. Endpoints are paired within shape; live K/V donor and applied coordinates are recorded."))


def verify():
    p = A.load("plan.json")
    assert p["runtime"] == A.Base.versions() and p["previous_freeze"] == A.digest(Prev.OUT / "freeze.json")
    assert all(A.digest(ROOT / k) == h for k, h in p["sources"].items())
    assert all(B.sha(v["file"]) == v["sha"] for v in p["materials"].values())
    assert all(B.sha(k) == h for k, h in p["pilot"].items())
    return p


def main(stage):
    setup()
    if stage == "plan":
        plan()
        return
    if stage == "audit":
        from kv_prefix_audit import audit
        audit(verify())
        return
    if stage in ("preview", "close"):
        from kv_prefix_report import main as report
        report(stage, verify())
        return
    import torch
    from kv_prefix_study import pilot, discover, confirm

    p = None if stage == "pilot" else verify()
    identity = model_identity()
    if p is not None:
        assert identity == p["snapshot"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    with P.GpuLease(), torch.no_grad():
        t0 = time.monotonic()
        N.progress("prefix-load", 0, 1, t0)
        m = N.Engine()
        N.progress("prefix-load", 1, 1, t0)
        try:
            {"pilot": pilot, "discover": discover, "confirm": confirm}[stage](m, p)
        finally:
            m.close()


if __name__ == "__main__":
    sys.modules["kv_prefix_run"] = sys.modules[__name__]
    main(sys.argv[1])
