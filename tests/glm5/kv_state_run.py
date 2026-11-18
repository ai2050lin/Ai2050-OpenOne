"""Phase 2950 entry point: direct State reads and true State-discrete interventions."""

from pathlib import Path
import sys
import time

import kv_prefix_run as Prev


A, B, P, N, O, ROOT = Prev.A, Prev.B, Prev.P, Prev.N, Prev.O, Prev.ROOT
OUT = ROOT / "tests/glm5/result/kv_state_20261116_v1"
TMP = ROOT / "tests/glm5_temp/kv_state_20261116_v1"
RUN = "KV-STATE-20261116-v1"
SEED = 29502026
SIZES = {"pilot": 8, "discover": 32, "confirm": 104}
BUDGET = 158 * 2**20
SLAB = 64
G_SOURCES = ("fact", "book")
E_LAYERS = (29, 33, 36, 39)
E_BUDGETS = (256, 768, 1536)
E_CASES = 32
E_TOKENS = 80
E_PASSES = 16
ENDPOINTS = ("g-fact-direction", "g-book-direction", "g-both-direction", "g-both-pq", "g-kv-rescue", "g-kv-reverse",
             "e-switch", "e-native-pq", "e-switch-pq")
CONFIDENCE = 1 - .05 / len(ENDPOINTS)
VOCAB = {
    "pilot": "Alice Bob Charlie David Emma Olivia James Noah Oliver Elijah William Henry Lucas Mason Ethan Alexander Hannah Emily Grace Chloe Lily Sophia Mia Ava Amelia Isabella Harper Evelyn Abigail Elizabeth Eleanor Nora Hazel Violet Ellie Stella Natalie Naomi Lucy Elena Leah Julia Clara Joseph Daniel Matthew Samuel Benjamin Michael Isaac Jacob Logan Luke Peter Aaron Thomas George Adam Andrew Ryan Jason Kevin Robert Charles Richard Edward Arthur Oscar Jack Leo Max Owen Liam Finn Simon Alan Paul Fred Frank Tony Harry Joseph John Jeremy Daniel Aaron",
    "discover": "London Paris Berlin Rome Madrid Lisbon Oslo Stockholm Helsinki Dublin Brussels Vienna Prague Warsaw Athens Cairo Dubai Doha Istanbul Ankara Beijing Shanghai Tokyo Kyoto Osaka Seoul Taipei Bangkok Hanoi Jakarta Manila Sydney Melbourne Perth Auckland Wellington Montreal Toronto Vancouver Calgary Ottawa Boston Chicago Denver Atlanta Dallas Austin Houston Phoenix Miami Orlando Tampa Seattle Portland Sacramento Detroit Pittsburgh Cleveland Columbus Cincinnati Milwaukee Baltimore Buffalo Rochester Richmond Albany Hartford Providence Edinburgh Glasgow Manchester Liverpool Bristol Leeds Newcastle Oxford Cambridge Bath Brighton York Naples Venice Florence Milan Turin Genoa Barcelona Valencia Granada Seville Cordoba Porto Paris Lyon Nice Marseille Lille Bordeaux Toulouse Nantes Bonn Munich Hamburg Cologne Dresden Stuttgart Frankfurt Bremen Hanover Leipzig Zurich Geneva Basel Bern Lausanne Lugano Sofia Belgrade Zagreb Sarajevo Skopje Tirana Budapest Bucharest Odessa Kiev Riga Vilnius Tallinn Minsk Moscow Samara Kazan Tbilisi Baku Yerevan Beirut Damascus Baghdad Tehran Karachi Lahore Mumbai Delhi Pune Chennai Bangalore Kolkata Goa Jaipur Agra ",
    "confirm": "Anderson Anderson Smith Johnson Williams Brown Jones Miller Davis Wilson Moore Taylor Clark Lewis Walker Hall Allen Young King Wright Scott Green Baker Adams Nelson Carter Mitchell Perez Roberts Turner Phillips Campbell Parker Evans Edwards Collins Stewart Morris Murphy Cook Rogers Reed Morgan Bell Bailey Rivera Cooper Richardson Cox Howard Ward Torres Peterson Gray Ramirez James Watson Brooks Kelly Sanders Price Bennett Wood Barnes Ross Henderson Coleman Jenkins Perry Powell Long Patterson Hughes Flores Washington Butler Simmons Foster Bryant Alexander Russell Griffin Diaz Hayes Myers Ford Hamilton Graham Sullivan Wallace Woods Cole West Jordan Owens Reynolds Fisher Ellis Harrison Gibson McDonald Cruz Marshall Ortiz Gomez Murray Freeman Wells Webb Simpson Stevens Tucker Porter Hunter Hicks Crawford Henry Boyd Mason Morales Kennedy Warren Dixon Ramos Reyes Burns Gordon Shaw Holmes Rice Robertson Hunt Black Daniels Palmer Mills Nichols Grant Knight Ferguson Rose Stone Hawkins Dunn Perkins Spencer Gardner Stephens Payne Pierce Berry Matthews Arnold Wagner Willis Ray Watkins Olson Carroll Duncan Snyder Hart Cunningham Bradley Lane Riley Andrews Harper Fox Lawrence Bishop Phelps Welch Hodge Wilkins Farrell Newman Schwartz Pearson Bentley Baxter Hardy Fox Walsh Hardy Baxter Weber Mann Hubbard Hammond Norton Cross Horton Vaughn Baldwin Shelton Walters Meyer Bach Bachman Harvey Pearson Thomas Jackson Douglas Hopkins Howard Chapman Abbott Goodwin Bush Banks Parsons Owen Holland Knight Chandler McKenzie Benson Webster Shaw Boone Hines Bates Maddox Curtis Robbins Ramsey Ford Lambert Brock Rios Newton Jacobs Hughes Hobbs Sutton Morton Pruitt Briggs Kerr Morse Page Cohen Levy Klein Adler Weiss Roth Goldberg Stein Bernstein Kaplan Adler Kramer Fischer Becker Schneider Wagner Kruger Vogel Bauer Zimmermann Braun Frank Krause Schmidt Schulz Richter Werner Walter Berger Hoffman Neumann Beck Busch Mann Weber Fischer Engel Kaiser Schmitt Miller Klein Lang Kramer Koch Roth Dietrich Foster Soto Vega Reyes Rivera Ruiz Luna Santos Silva Castro Vargas Ramos Lopez Mora Rojas Ortiz Rivas Sanchez Navarro Molina Munoz Ortega Medina Vidal Sosa Costa Cruz Garcia Torres Moreno Suarez Mendez Morales Santiago Fuentes Herrera Cabral Perez Gomez Diaz Leclerc Dupont Martin Laurent Petit Leroy Moreau Roux Girard Simon Lefebvre Dubois Mercier Blanc Fontaine Chevalier Bernard Lambert Durand Rousseau Morel Gauthier Robert Lopez Lopez",
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


def material(stage, tok, used):
    original = Prev.VOCAB, Prev.SEED, Prev.SIZES
    try:
        Prev.VOCAB, Prev.SEED, Prev.SIZES = VOCAB, SEED, SIZES
        return Prev.material(stage, tok, used)
    finally:
        Prev.VOCAB, Prev.SEED, Prev.SIZES = original


def model_identity():
    identity = B.model_identity("q14")
    assert identity == P.load(Prev.OUT / "plan.json")["snapshot"]
    return identity


def plan():
    assert not (OUT / "plan.json").exists()
    pilot = A.load("pilot-engine.json")
    assert pilot["passed"]
    tok = A.Base.tok()
    used = excluded() | N.words(A.load("material-pilot.json"))
    panels = {}
    for stage in ("discover", "confirm"):
        rows = material(stage, tok, used)
        words = N.words(rows)
        assert not used & words
        used |= words
        name = f"material-{stage}.json"
        A.save(name, rows)
        panels[stage] = dict(file=name, sha=B.sha(name), rows=len(rows), families=SIZES[stage], words=sorted(words))
    prior = P.load(Prev.OUT / "plan.json")
    src = {*prior["sources"], "tests/glm5/kv_prefix_run.py"}
    src.update(x.relative_to(ROOT).as_posix() for x in Path(__file__).parent.glob("kv_state_*.py"))
    A.save("plan.json", dict(
        run=RUN, time=O.stamp(), seed=SEED, snapshot=model_identity(), runtime=A.Base.versions(), materials=panels,
        previous_freeze=A.digest(Prev.OUT / "freeze.json"), sources={x: A.digest(ROOT / x) for x in sorted(src)},
        pilot={x: B.sha(x) for x in ("material-pilot.json", "pilot-behavior.json", "pilot-engine.json")},
        storage=dict(peak_bytes=BUDGET, reserve_gib=20, volume="D:"),
        behavior="Original new authored template lexical pools, no embeddings from another model. A family is eight role x order x generated-distance views using one codebook. Whole-family behavior and sham checks. At least 90% natural and correct in every cell, timing and greedy correctness separate. Keep full observed Case and Code. n=104 chosen so after modest exclusions n>=80. Continuous standardized effect .4 has roughly .89 power at planned n, before absolute gates and lexical dependence; switches are a separate discrete endpoint.",
        g="At G, the query at the already emitted State token, scan 25x40 original native prompt Value contributions for rule-bearing factual clauses and the two competing codebook lines. Search L0..24 only: direct G block L24 can enter x25(G). Rank per mask signed Code logit-lens projection and physical write dose; pass up to eight distinct layer/head candidates per mask to exact dynamic forward. Freeze one head per mask based on signed role-aligned median/mean (primary mean) native L25H18 G-slot repair projection, with distinct heads. Save discovery all-coordinate projection following Phase2949 original Code adjoint. Control two heads drawn from nearest discovery physical dose quartile at same layer/mask. Test each, both, self restore and controls, then only at C H18 restore original G K, V, both; reverse transplant and whole G V deletion. Leave Q and other slots live.",
        e="At E, before first State, candidate post-block residuals L29/33/36/39. Stream true derivative of logit(rejected)-logit(original) to each position at the native prefix. Rank native residual coordinates by discovery mean derivative times actual matched role-swap donor-minus-own coordinate. Test 256/768/1536 each on discovery, plus an entire L39 donor positive control. Choose smallest budget reaching 50% recognized switch, resolving ties by recognized switch count, mean signed margin loss and earliest layer. Otherwise select the most recognized switches with same tie rules except preferring smaller budget after rate. Freeze two same-size controls from nonselected coordinates matched by absolute displacement. At confirmation test actual donor transplant, self restore, selected-coordinate zero, random donor transplant and full L39 donor control. Use no logit masks and preserve live earlier prefix. E boundaries later than L25 cannot change G in L25 when the emitted token is held fixed. Donor is an explicit exogenous input from the other factual role, never read from future actual generation.",
        natural="Full-vocabulary genuine State argmax scored exact rejected token and unexpected switches separately. If E switch occurs, preserve actual chosen token and verify subsequent actual greedy tokens to Code or EOS, 32 cases per targeted/random/full arm, max 80 emitted tokens or 16 speculative passes. The common Code comparison takes the first 32 technically accepted families and rotates the selected one of eight views by family index, independently of arm outcome. These cases have first priority for replay. Balanced case order by family spread and view; draft tokens never considered generated without exact current-prefix argmax. Record whether P/Q follows original fact or emitted alternative under same codebook, plus full observations and stop reasons. Natural prompt role reversal separately verifies that behavior tracks task roles. No generated code counted after fixed history diverges.",
        gate=dict(labels=ENDPOINTS, confidence=CONFIDENCE, n=80, natural_n=24, logit=.10, switch=.05,
                  g_route="Each source local directed effect compared with matched heads. Joint bridge needs joint local direction, joint actual P/Q loss, whole H18 native restoration plus reverse to pass. The C directed score is local tangent, actual P/Q measured independently.",
                  e_route="Full-vocabulary recognized switch rate vs matched controls on all qualified families. A separate paired outcome in the 32 locked views is a verified recognized switch and P/Q concordant with newly emitted State within the predeclared horizon; unresolved joint events count as unverified, with explicit stop reasons. Positive inference requires at least 24 common views and the same rate/interval rule, without extrapolation. Measure E fixed-history P/Q margin vs matched controls as a separate endpoint."),
        numerics="Packed sham requires full vocabulary equality of prefix choices and raw final P/Q drift <=1. Exact physical checking of local edits and native self equality. C-only and G-only E logits and argmax equality. Post-RoPE K unchanged where required. FP32 local projection reconstruction tolerance .001, normalized head edit error <.05 and E/C adjoint replay relative error <=.05.",
    ))


def verify():
    p = A.load("plan.json")
    assert p["runtime"] == A.Base.versions() and p["previous_freeze"] == A.digest(Prev.OUT / "freeze.json")
    assert all(A.digest(ROOT / k) == v for k, v in p["sources"].items())
    assert all(B.sha(x["file"]) == x["sha"] for x in p["materials"].values())
    assert all(B.sha(k) == v for k, v in p["pilot"].items())
    return p


def main(stage):
    setup()
    if stage == "plan":
        plan()
        return
    if stage == "audit":
        from kv_state_audit import audit
        audit(verify())
        return
    if stage in ("preview", "close"):
        from kv_state_report import render
        render(stage, verify())
        return
    import torch
    from kv_state_study import pilot, discover, confirm

    p = None if stage == "pilot" else verify()
    identity = model_identity()
    if p is not None:
        assert identity == p["snapshot"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    with P.GpuLease(), torch.no_grad():
        t = time.monotonic()
        N.progress("state-load", 0, 1, t)
        m = N.Engine()
        N.progress("state-load", 1, 1, t)
        try:
            {"pilot": pilot, "discover": discover, "confirm": confirm}[stage](m, p)
        finally:
            m.close()


if __name__ == "__main__":
    sys.modules["kv_state_run"] = sys.modules[__name__]
    main(sys.argv[1])
