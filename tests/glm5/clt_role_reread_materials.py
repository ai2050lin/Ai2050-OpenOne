"""Freeze role-versus-reread materials and one archive plan at a time, without model inference."""
import argparse
from itertools import product
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "tests/glm5"
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import checked_dir
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import MODELS, WORDS, read, sha

OUT = HERE / "result/clt_role_reread_v1"
MODEL = MODELS["qwen14"]
ENTRY = HERE / "clt_role_reread.py"
SEED = 202611192322
SIZES = {"pilot": 32, "discovery": 64, "confirm": 128}
FIELDS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
RULES = ("Use the original reserved slot of the requested guest, regardless of its current occupant.",
         "Use the currently occupied slot of the requested guest, regardless of its original reserved slot.")
CRITERIA = (
    "All eight cells require >=85% whole-vocabulary first-token accuracy at each stage. "
    "Natural H36 selected-row over rival within active archive must be correct >=70% in EACH conflict rule. "
    "Pilot, discovery and confirm use 32,64,128 separate families. One frozen rank-one direction per edge. "
    "Confirmation uses complete within-family contrasts. On opposed rows, define E=early-row log mass minus late-row log mass. "
    "Within EACH unchanged final rule require E(control-A)-E(cut-A)>0 and E(cut-B)-E(control-B)>0, "
    "with 99% family-bootstrap lower bounds >0 for each. Require same directions in the code-logit contrasts for an output claim. "
    "A subtracts a matched rank-one component of the B25H18 last query-name token V, weighted by its unchanged attention. "
    "B removes a matched rank-one component of each active record's mean post-RoPE K at the H36 consumer, including code tokens. "
    "Both axes and orthogonal random directions are frozen on discovery. Early row decoding and late match ranking each "
    "require >=80% separately in both active archives on confirm. Natural prefix V and h18 must be exact across future variants "
    "and group pre-O reconstruction max error divided by 1+max absolute native head output must be <=.025. "
    "At least 95% of cases in each arm must have target/control recomputed edit norms within 10% and nonzero. "
    "Opposite-path positive controls: with late rule and fixed early address, flipping late identity after cut-A must retain "
    ">=80% of its baseline crossover strength; reciprocal early-rule test after cut-B. Family bootstrap 99% lower "
    "bounds on retained-minus-0.8*baseline and baseline sensitivity must both exceed zero. "
    "Whole-logit hashes of sham/restore must match baseline; exact earlier name write and bodily K/V controls at their "
    "defined boundaries. H21 and h18 are diagnostics. A failed gate limits inference to its stage; no retuning on holdout."
)


def identities():
    files = [ENTRY, Path(__file__).resolve(), HERE / "source_binding_design.py",
             HERE / "source_binding_runtime.py", HERE / "source_binding_resources.py",
             ROOT / "shared/dist_protocol/mechanism_trace.py",
             ROOT / "shared/dist_protocol/natural_register_runner.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def make_item(tok, family, stage, names, codes, early, late, mode):
    active = family % 2
    early_row = [family // 2 % 2] * 2
    late_row = [family // 4 % 2] * 2
    early_row[active], late_row[active] = early, late
    titles = ("Orion", "Vega")
    target_name = names[family // 8 % 2]
    other_name = next(n for n in names if n != target_name)
    def occupant(index, row):
        return target_name if index == row else other_name
    lines = ["Two archives each have two slots. The first roster lists the original reservations in slot order."]
    for a, title in enumerate(titles):
        lines.append(f"{title} reservations:")
        lines.extend(occupant(i, early_row[a]) for i in range(2))
    lines += [f"Requested guest: {target_name}", "The following ledger lists current occupants and today's code for each slot."]
    for a, title in enumerate(titles):
        lines.append(f"{title} ledger:")
        lines.extend(f"{occupant(i, late_row[a])} | {codes[2*a+i]}" for i in range(2))
    lines += [f"Archive to use: {titles[active]}.", RULES[mode], "Return its current code. Answer with one lowercase word."]
    prompt = tok.apply_chat_template([{"role":"user","content":"\n".join(lines)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) + "Answer:\n"
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    def span(a, b):
        return [i for i, (lo, hi) in enumerate(enc["offset_mapping"]) if lo < b and hi > a]
    records = [span(*m.span()) for m in re.finditer(r"^\w+ \| \w+$", prompt, re.M)]
    query = re.search(r"^Requested guest: (\w+)$", prompt, re.M)
    query_tokens = span(*query.span(1))
    if len(records) != 4 or not query_tokens or not all(records) or min(records[0]) <= max(query_tokens):
        raise ValueError("query and ledger alignment")
    answer_ids = [tok.encode(c, add_special_tokens=False) for c in codes]
    if not all(len(ids) == 1 and tok.encode(prompt+c, add_special_tokens=False) == enc["input_ids"]+ids
               for c, ids in zip(codes, answer_ids)):
        raise ValueError("full answer token boundary")
    return dict(id=f"{stage}-{family:03}-{early}{late}{mode}", family=family, active=active, early=early,
                late=late, mode=mode, prompt=prompt, tokens=enc["input_ids"], query=query_tokens, records=records,
                codes=codes, choice_ids=[ids[0] for ids in answer_ids], target=2*active+(late if mode else early))


def build_rows(tok, stage):
    partition = list(SIZES).index(stage)
    rng = random.Random(SEED + partition)
    syllables = [c+v for c in "bdfgklmnprstvz" for v in "aeiou"]
    used = set()
    rows = []
    for family in range(SIZES[stage]):
        names = []
        while len(names) < 2:
            name = "".join(rng.choices(syllables, k=4)).capitalize()
            if int(digest(name), 16) % 3 != partition or name in used:
                continue
            if len(tok.encode(" "+name, add_special_tokens=False)) != 4 or len(tok.encode(name, add_special_tokens=False)) != 4:
                continue
            names.append(name)
            used.add(name)
        codes = rng.sample([w for w in WORDS if len(tok.encode(w, add_special_tokens=False))==1
                            and len(tok.encode(" "+w, add_special_tokens=False))==1], 4)
        group = [make_item(tok, family, stage, names, codes, e, l, m) for e,l,m in product(range(2), repeat=3)]
        aligned = {(len(r["tokens"]), tuple(map(tuple,r["records"])), tuple(r["query"])) for r in group}
        if len(aligned) != 1:
            raise ValueError(f"position control across all rules and row permutations failed: {family}")
        for early in range(2):
            prefixes = {tuple(r["tokens"][:r["query"][-1]+1]) for r in group if r["early"]==early}
            if len(prefixes)!=1:
                raise ValueError("late cue or rule leaked across causal boundary")
        if len({tuple(r["tokens"]) for r in group})!=8:
            raise ValueError("duplicate factorial cell")
        rows.extend(group)
    return rows


def prepare(stage, previous):
    from transformers import AutoTokenizer
    checked_dir(OUT, 32*2**20, operation="查表冲突预注册")
    frozen = OUT / f"{stage}-plan.json"
    if frozen.exists():
        raise FileExistsError(frozen)
    transfer = {}
    if stage != "pilot":
        prior = "pilot" if stage == "discovery" else "discovery"
        if previous is None:
            raise ValueError("Previous verified run record is required")
        review, receipt = (read(previous / n) for n in ("review.json","server-receipt.json"))
        source = read(previous / "protocol.json")
        if (not receipt["readback_verified"] or receipt["run_id"]!=review["server_run_id"]
                or source!=read(OUT/f"{prior}-plan.json") or not review["conclusion"]["stage_pass"]):
            raise ValueError("Previous gate or receipt mismatch")
        transfer["previous_run"] = receipt["run_id"]
        if stage=="confirm":
            candidate_path = Path(review["scratch"]) / "candidate.json"
            proof = next(a for a in receipt["artifacts"] if a["name"]=="candidate.json")
            if sha(candidate_path)!=proof["sha256"]:
                raise ValueError("Archived candidate differs")
            transfer["candidate"] = read(candidate_path)
            transfer["candidate_sha256"] = sha(candidate_path)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = build_rows(tok, stage)
    questions = OUT / f"{stage}-questions.json"
    save(questions, rows)
    identity = read(HERE/"result/clt_first_record_path_v1/materials/manifest.json")
    tok_sha = digest({k:sha(MODEL/k) for k in FIELDS if (MODEL/k).exists()})
    if tok_sha != identity["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from Qwen snapshot")
    code_sha = digest(identities())
    doc = dict(schema="research_test_plan.v1", kind="real", title=f"Original reservation or current occupant / {stage}",
        objective="Separate a query-name source-specific write from late contextual record-key matching in Qwen3-14B.",
        principle="Independently cross the queried guest's early row and later occupied row. Draw codes after the query boundary.",
        hypothesis="B25H18 writes an early row into the final position while B26H36 contextual keys provide competing late name evidence.",
        alternatives="Direct rescan of original roster, repeated query identity retrieval, other heads, mixed or redundant encodings.",
        criteria=CRITERIA, domains=["KN","RS"], family_ids=["SOURCE-BINDING-20261110","CLT-ROLE-REREAD"],
        entrypoint_sha256=sha(ENTRY), execution_args=[],
        dataset=dict(id="clt-role-reread/"+stage,revision="v1",content_sha256=sha(questions),
            questions_file_sha256=sha(questions), split_manifest_sha256=digest({"seed":SEED,"sizes":SIZES,"method":"name SHA256 canonical JSON mod 3"}),
            case_count=len(rows), exposure="Separate name pools by stage; held-out materials generated only after a verified prior gate."),
        acquisition=dict(model_id="Qwen3-14B",model_snapshot=identity["model_snapshot"],tokenizer_sha256=tok_sha,
            entrypoint_sha256=sha(ENTRY),questions_file_sha256=sha(questions),inputs_sha256=digest([r["tokens"] for r in rows]),
            code_sha256=code_sha,environment_sha256=digest(runtime_identity()),precision="BF16 native, FP32 head accounting",
            seed=SEED,generation=dict(script_args=[],sample=False, stage=stage, previous=transfer.get("candidate_sha256","")),
            capture=dict(layers=[17,25,26,28],readouts=["last requested-name token","final token","active ledger rows"],
                         boundary="h18; B25H18 pre-O name-source contribution; B26H36 QK/V; B28H21 record mass; final logits")),
        analysis=dict(method_version="clt_role_reread.v1",code_sha256=code_sha,scoring_sha256=digest(CRITERIA),
            selection=dict(stage=stage,seed=SEED,sizes=SIZES, **transfer),
            required_artifact_roles=sorted(CORE_ROLES | {"native_field"})))
    validate_plan(doc)
    save(frozen,doc)
    print(f"{stage}: {len(rows)} inputs, {SIZES[stage]} families; {frozen}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SIZES)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    prepare(args.stage, args.previous)
