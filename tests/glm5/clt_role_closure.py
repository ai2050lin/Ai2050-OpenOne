"""Freeze, register and complete the paired old-slot behavior diagnostics."""
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name == "test.py" else Path(__file__).resolve().parents[2]
HERE = ROOT / "tests/glm5"
sys.path[:0] = [str(HERE), str(ROOT)]

from clt_role_reread_materials import FIELDS, MODEL, WORDS, make_item
from deploy.storage_policy import check_write, checked_dir
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import read, sha


FAMILIES = 128
SEED = 202611200002
STAGE = "diagnostic"
OUT = HERE / "result/clt_role_closure_v1"
ENTRY = HERE / "clt_role_closure.py"
PREVIOUS = HERE / "result/clt_role_reread_v1"
PRIOR_RUN = "record-1f7d24bd-8ddf-46d2-854a-20cd97561058"
TASKS = ("old_only", "old_with_ledger", "explicit_slot", "full_old", "full_current")
BATCH = 144
CRITERIA = (
    "Complete all five prompt types on 128 fresh independent name families before interpretation. "
    "Every family crosses both old and current rows and both full rules; use two distinct old-only prompts, "
    "not four duplicate prompts. Extract the full-vocabulary top-1 next token plus candidate logits. "
    "Keep agreement and conflict separate and count paired rule switches. Each of four cells of the "
    "old-with-ledger, explicit-slot, full-old, and full-current tasks and both old-only cells are reliable "
    "at 85% or higher. For the full task require at least 85% of conflict tables to have both answers correct. "
    "Inspect the paired drop from old-only to old-with-ledger, and from explicit-slot to full-old, separately "
    "in agreement and conflict. Draw 10000 family bootstrap replicates with fixed seed for 95% descriptive intervals. "
    "If old-with-ledger and explicit-slot pass while full-old fails, classify the failure at joint instruction and "
    "retrieval in this template. This contrast cannot separate rule binding, use of an address and prompt-form "
    "effects. If either component fails, localize only the earliest observed behavioral failure and report the others. "
    "If the unchanged full task passes, document failed reproduction of the previous behavior gate. "
    "This completion study executes every behavioral contrast without an intermediate scientific stop or "
    "a new candidate search. It makes no claims about source-specific causal interventions."
)


def sources():
    names = [
        "tests/glm5/clt_role_closure.py", "tests/glm5/clt_role_reread_materials.py",
        "tests/glm5/clt_role_reread_desktop.py", "tests/glm5/source_binding_design.py",
        "tests/glm5/source_binding_runtime.py", "tests/glm5/source_binding_resources.py",
        "tests/glm5/run_routine_test.py", "shared/dist_protocol/mechanism_trace.py",
        "shared/dist_protocol/natural_register_runner.py",
    ]
    return {name: sha(ROOT / name) for name in names}


def revise(tok, original, task, body, candidates, target):
    prompt = body + "<|im_end|>" + original["prompt"].split("<|im_end|>", 1)[1]
    tokens = tok.encode(prompt, add_special_tokens=False)
    ids = [tok.encode(answer, add_special_tokens=False) for answer in candidates]
    if not all(len(i) == 1 and tok.encode(prompt + answer, add_special_tokens=False) == tokens + i
               for i, answer in zip(ids, candidates)):
        raise ValueError("Answer boundary is not one model token")
    early, late = original["early"], original["late"]
    return dict(id=f'{original["family"]:03}-{task}-{early}{late}', family=original["family"],
                task=task, early=early, late=None if task == "old_only" else late, active=original["active"],
                prompt=prompt, tokens=tokens, candidates=list(candidates), choice_ids=[i[0] for i in ids], target=target)


def material(tok):
    used = {r["prompt"].split("Requested guest: ", 1)[1].splitlines()[0]
            for r in read(PREVIOUS / "pilot-questions.json")}
    for case in read(PREVIOUS / "pilot-questions.json"):
        for line in case["prompt"].splitlines():
            if line and line.isalpha() and line[0].isupper():
                used.add(line)
    rng = random.Random(SEED)
    syllables = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    row_words = ("top", "bottom")
    titles = ("Orion", "Vega")
    vocabulary = [w for w in WORDS if w not in row_words and len(tok.encode(w, add_special_tokens=False)) == 1
                  and len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    for family in range(FAMILIES):
        names = []
        while len(names) < 2:
            name = "".join(rng.choices(syllables, k=4)).capitalize()
            # Stay outside the V1 discovery/confirm name partitions.
            if name in used or int(digest(name), 16) % 3 != 0:
                continue
            if all(len(tok.encode(form, add_special_tokens=False)) == 4 for form in (name, " " + name)):
                names.append(name)
                used.add(name)
        codes = rng.sample(vocabulary, 4)
        for early in (0, 1):
            for late in (0, 1):
                full = make_item(tok, family, STAGE, names, codes, early, late, 0)
                current = make_item(tok, family, STAGE, names, codes, early, late, 1)
                active = full["active"]
                body = full["prompt"].split("<|im_end|>", 1)[0]
                header = f"Archive to use: {titles[active]}.\n"
                ledger_start = "The following ledger lists current occupants and today's code for each slot."
                row_order = row_words if (family // 16) % 2 else row_words[::-1]
                row_question = ("Return the position of the requested guest's original reserved slot. "
                                f"Answer with exactly one lowercase word: {row_order[0]} or {row_order[1]}.")
                if late == 0:
                    old_body = body.split(ledger_start, 1)[0] + header
                    yield revise(tok, full, "old_only", old_body + row_question, row_words, early)
                with_ledger = body.split(header, 1)[0] + header
                yield revise(tok, full, "old_with_ledger", with_ledger + row_question, row_words, early)
                explicit = f"Use the {row_words[early]} slot in this archive.\nReturn its current code. Answer with one lowercase word."
                yield revise(tok, full, "explicit_slot", with_ledger + explicit, codes, 2 * active + early)
                for original, task in ((full, "full_old"), (current, "full_current")):
                    row = revise(tok, original, task, original["prompt"].split("<|im_end|>", 1)[0],
                                 codes, original["target"])
                    if row["tokens"] != original["tokens"]:
                        raise ValueError("The complete prompt differs from Phase 3003")
                    yield row


def prepare():
    from transformers import AutoTokenizer

    checked_dir(OUT, 32 * 2**20, operation="旧行号对照预注册")
    plan_path = OUT / f"{STAGE}-plan.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = list(material(tokenizer))
    if len(rows) != FAMILIES * 18 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Invalid matched family count")
    questions = OUT / f"{STAGE}-questions.json"
    save(questions, rows)
    identity = read(PREVIOUS / "pilot-plan.json")["acquisition"]
    tokenizer_sha = digest({k: sha(MODEL/k) for k in FIELDS if (MODEL/k).exists()})
    if tokenizer_sha != identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from prior study")
    code_sha = digest(sources())
    question_sha = sha(questions)
    doc = dict(schema="research_test_plan.v1", kind="real",
        title="Old slot completion study / paired calculation and retrieval probes",
        objective="Localize failure of original-slot retrieval and finish the old-slot/current-name behavioral contrasts.",
        principle="Pair old-slot calculation before and after the new ledger, explicit-address retrieval, and both unchanged full prompts.",
        hypothesis="Qwen3-14B may report the old slot and fetch a named slot separately even when the full prompt follows the current name.",
        alternatives="Old-slot calculation error, interference from the later ledger, weak instruction binding, or retrieval error.",
        criteria=CRITERIA, domains=["KN", "RS"], family_ids=["CLT-ROLE-REREAD", "CLT-ROLE-CLOSURE"],
        entrypoint_sha256=sha(ENTRY), execution_args=[],
        dataset=dict(id="clt-role-closure/diagnostic", revision="v1", content_sha256=question_sha,
            questions_file_sha256=question_sha, case_count=len(rows),
            split_manifest_sha256=digest(dict(seed=SEED, families=FAMILIES, prior=PRIOR_RUN)),
            exposure="New pilot-partition names disjoint from Phase 3003. All probes frozen before inference. No fit or tuning."),
        acquisition=dict(model_id="Qwen3-14B", model_snapshot=identity["model_snapshot"], tokenizer_sha256=tokenizer_sha,
            entrypoint_sha256=sha(ENTRY), questions_file_sha256=question_sha, inputs_sha256=digest([r["tokens"] for r in rows]),
            code_sha256=code_sha, environment_sha256=digest(runtime_identity()), precision="BF16 native, FP32 score storage",
            seed=SEED, generation=dict(script_args=[], sample=False, batch=BATCH),
            capture=dict(layers=[39], readouts=["final token"], boundary="After block 39, final RMSNorm and lm_head: full vocabulary top-1 and frozen candidate token logits")),
        analysis=dict(method_version="clt_role_closure.v1", code_sha256=code_sha, scoring_sha256=digest(CRITERIA),
            selection=dict(stage=STAGE, prior_run=PRIOR_RUN, families=FAMILIES, seed=SEED),
            required_artifact_roles=sorted(CORE_ROLES)))
    validate_plan(doc)
    save(plan_path, doc)
    print(f"Frozen {len(rows)} prompts from {FAMILIES} families at {plan_path}", flush=True)


def summarize(records, run):
    import numpy as np

    table = {(r["family"], r["task"], r["early"], r["late"]): r for r in records}
    if len(table) != FAMILIES * 18:
        raise ValueError("Incomplete results")
    cells, totals = {}, {}
    for task in TASKS:
        rows = [r for r in records if r["task"] == task]
        selectors = {"all": rows}
        if task == "old_only":
            selectors.update({str(e): [r for r in rows if r["early"] == e] for e in (0, 1)})
        else:
            selectors.update({f"{e}{l}": [r for r in rows if (r["early"], r["late"]) == (e, l)]
                              for e in (0, 1) for l in (0, 1)})
            selectors.update({label: [r for r in rows if (r["early"] == r["late"]) == agrees]
                              for label, agrees in (("agree", True), ("conflict", False))})
        cells[task] = {k: dict(correct=sum(r["correct"] for r in rs), n=len(rs),
                              constrained_correct=sum(r["constrained_correct"] for r in rs))
                       for k, rs in selectors.items()}
        totals[task] = all(s["correct"] * 100 >= 85 * s["n"] for k, s in cells[task].items() if len(k) <= 2)

    def get(family, task, early, late):
        return table[family, task, early, None if task == "old_only" else late]

    switches = dict(both_correct=0, both_current=0, both_old=0, reverse=0, other=0)
    errors = {task: dict(confirmed_latest_cue=0, other=0) for task in ("old_with_ledger", "full_old")}
    for f in range(FAMILIES):
        for e in (0, 1):
            l = 1 - e
            a, b = (get(f, task, e, l) for task in ("full_old", "full_current"))
            if a["correct"] and b["correct"]:
                switches["both_correct"] += 1
            elif a["top_id"] == b["gold_id"] and b["correct"]:
                switches["both_current"] += 1
            elif a["correct"] and b["top_id"] == a["gold_id"]:
                switches["both_old"] += 1
            elif a["top_id"] == b["gold_id"] and b["top_id"] == a["gold_id"]:
                switches["reverse"] += 1
            else:
                switches["other"] += 1
            row = get(f, "old_with_ledger", e, l)
            for task, r, rival in (("old_with_ledger", row, row["choice_ids"][l]), ("full_old", a, b["gold_id"])):
                if not r["correct"]:
                    errors[task]["confirmed_latest_cue" if r["top_id"] == rival else "other"] += 1

    bootstrap = np.random.default_rng(SEED).integers(FAMILIES, size=(10000, FAMILIES))
    contrasts = {}
    for name, a, b in (("ledger_interference", "old_only", "old_with_ledger"),
                       ("complete_prompt_cost", "explicit_slot", "full_old")):
        for group in ("agree", "conflict"):
            values = np.array([
                np.mean([int(get(f, a, e, l)["correct"]) - int(get(f, b, e, l)["correct"])
                         for e in (0, 1) for l in (0, 1) if (e == l) == (group == "agree")])
                for f in range(FAMILIES)
            ])
            contrasts[f"{name}_{group}"] = dict(diff=float(values.mean()),
                ci95=np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975]).tolist())
    return dict(stage=STAGE, run_id=run, families=FAMILIES, cases=len(records),
                cells=cells, reliable_by_task=totals, conflict_pairs=switches, conflict_errors=errors,
                contrasts=contrasts, full_rule_reliable=totals["full_old"] and totals["full_current"]
                and switches["both_correct"] * 100 >= 85 * 2 * FAMILIES)


def execute():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_resources import resources
    from source_binding_runtime import Runtime

    work = Path(os.environ["AI2050_TEST_WORK_DIR"])
    question_path = Path(os.environ["AI2050_TEST_QUESTIONS"])
    plan, preflight, review = [read(question_path.parent / n) for n in ("protocol.json", "preflight.json", "review.json")]
    run = review["server_run_id"]
    if (preflight["plan_sha256"] != digest(plan) or not preflight["preflight_id"]
            or review["archive_state"] != "pending" or Path(review["scratch"]).resolve() != work.resolve()
            or plan["entrypoint_sha256"] != sha(Path(__file__))
            or plan["dataset"]["questions_file_sha256"] != sha(question_path)
            or plan["acquisition"]["code_sha256"] != digest(sources())):
        raise ValueError("Registered runtime and frozen material must match")
    start = time.monotonic()
    print(json.dumps(dict(step="verify local weight snapshot", completed=0, total=FAMILIES * 18, run=run)), flush=True)
    if snapshot(MODEL) != plan["acquisition"]["model_snapshot"]:
        raise ValueError("Unexpected Qwen3-14B weight snapshot")
    rows = read(question_path)
    if len(rows) != FAMILIES * 18:
        raise ValueError("Expected all paired behavior rows")
    records = []
    with GpuLease(), torch.inference_mode(), (work / "case_results.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        model = Runtime("qwen14")
        torch.backends.cuda.matmul.allow_tf32 = False
        for lo in range(0, len(rows), BATCH):
            check_write(work, 16 * 2**20, operation="旧行号对照运行")
            batch = rows[lo:lo + BATCH]
            def progress(block, layer, x, y, part, length, offset=0, **kwargs):
                if offset == 0 and block in (-1, 9, 19, 29, 39):
                    elapsed = time.monotonic() - start
                    done = lo + len(batch) * (block + 1) / model.cfg.num_hidden_layers
                    print(json.dumps(dict(step="forward", block=block, completed=lo, total=len(rows),
                        elapsed=elapsed, eta=elapsed * (len(rows) - done) / done if done else None,
                        **resources())), flush=True)
            _, logits = model.forward(batch, callback=progress)
            top = logits.argmax(-1).tolist()
            for i, r in enumerate(batch):
                scores = logits[i, r["choice_ids"]].tolist()
                gold = r["target"]
                result = {k: v for k, v in r.items() if k not in ("tokens", "prompt")}
                result.update(run_id=run, top_id=top[i], top_token=model.tok.decode([top[i]]),
                    gold_id=r["choice_ids"][gold], gold_token=r["candidates"][gold],
                    correct=top[i] == r["choice_ids"][gold],
                    constrained_correct=max(range(len(scores)), key=scores.__getitem__) == gold, scores=scores)
                records.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(dict(step="scored", completed=len(records), total=len(rows),
                elapsed=time.monotonic()-start)), flush=True)
    result = summarize(records, run)
    with zipfile.ZipFile(work / "research-scripts.zip", "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sources():
            zf.write(ROOT / path, arcname=path)
    save(work / "artifacts.json", dict(schema="research_test_files.v1",
        files=[dict(path="research-scripts.zip", role="script", retention="core")]))
    save(work / "conclusion.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["launch"]:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch(STAGE, ENTRY, OUT))
    elif not sys.argv[1:]:
        execute()
    else:
        raise SystemExit("use prepare or launch")
