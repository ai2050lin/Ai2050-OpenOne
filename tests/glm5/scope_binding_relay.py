"""Separate direct document access from a pre-question neutral carrier in Qwen3-14B."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path
import random
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

ROOT = (Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name == "test.py"
        else Path(__file__).resolve().parents[2])
HERE = ROOT / "tests/glm5"
sys.path[:0] = [str(ROOT), str(HERE)]
from deploy.storage_policy import checked_dir, check_write
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import MODELS, read, sha


ENTRY = HERE / "scope_binding_relay.py"
OUT = HERE / "result/scope_binding_relay_v2"
MODEL = MODELS["qwen14"]
MODEL_ID = "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
TOKENIZER_ID = "90a2903e36317f2a873560235218be86367fa0ed4209f53096b6d59325fed333"
SOURCE_RUN = "record-0b497c1a-d34f-4628-980c-255a483ae412"
SEED = 202611221031
FAMILIES = (48, 128)
ARMS = ("baseline", "cut_relay", "control_relay", "restore_relay", "cut_document",
        "control_document", "restore_document", "cut_both")
LAYERS = (15, 23, 31)
COLORS = ("red", "blue")
CRITERIA = (
    "Freeze all inputs, source spans, read boundaries and fit rules before running. Use 48 discovery and 128 new "
    "confirm families, 64 per wholly held-out frame, eight cells per family. For every split/frame/scope/object "
    "cell require at least 85% actual whole-vocabulary greedy initial answer token, no ties counted in two-choice "
    "margin. Report natural first token separately from sequence completion (unmeasured). After discovery behavior, "
    "freeze one unnormalized within-family orientation difference axis averaged across runs, using concatenated "
    "native B15/B23/B31 relay pre-question final-token residuals standardized by fixed discovery block norms. "
    "Require identical prefix states for all future questions about each same prefix. On confirm, shared state "
    "plus declared target/scope index must predict actual greedy color in >=75% of all cases per frame and "
    "beat frozen word-bag/location 50% paired baseline by >=15 points, with 99% lower bound above 50%. "
    "Report supplied-truth symbolic reread separately. Failed behavior or predictive gate stops intervention. "
    "Structural candidates frozen before any forward: from all question-region tokens (including output prompt "
    "suffix), cut key access in all blocks/heads separately to literal-document region or neutral-relay region; "
    "cut both as a leakage/control arm. At each cut site compare with same-dtype edit formed by rotating adjacent "
    "coordinates of actual post-O residual-write delta, matched in norm; restore same receiver's original native "
    "write under the cut and require preference logits/argmax bit agreement with baseline. Record applied doses "
    "after BF16 rounding and require >=95% of paired rows to match realized route energy within 10%, nonzero. "
    "Estimate causal effects as held-out family-clustered 99% bootstrap intervals for target minus matched "
    "control loss of correct-vs-other logit margin, separately by frame/scope; claim a contribution only if "
    "mean loss >=0.25 nat and lower bound >0. At least 90% actual greedy accuracy in each single-path positive "
    "control is required to call the opposite route preserved. Cut-both must remove mapping sensitivity "
    "(absolute mean orientation-oriented preference <=0.10 nat in each frame/scope). Role-binding vs phonological "
    "copy and direct reread vs precomputed state at document positions are remaining alternatives. "
    "Do not infer uniqueness or portability outside these frames. Do not infer a mechanism from gate failures."
)


def sources():
    paths = [ENTRY, HERE / "source_binding_design.py", HERE / "source_binding_runtime.py",
             HERE / "source_binding_resources.py", HERE / "run_routine_test.py",
             ROOT / "shared/dist_protocol/natural_register_runner.py",
             ROOT / "shared/dist_protocol/mechanism_trace.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def split_material(tok):
    rng = random.Random(SEED)
    # Fixed digit lengths and a disjoint pool for each cohort. Names are novel in context.
    names = {0: rng.sample(range(1000, 3000), FAMILIES[0] * 2),
             1: rng.sample(range(5000, 9000), FAMILIES[1] * 2)}
    rows = []
    for stage, count in enumerate(FAMILIES):
        for family in range(count):
            a, b = (f"K{x}" for x in names[stage][2 * family:2 * family + 2])
            frame = 0 if stage == 0 else 1 + family % 2
            family_id = f"{'confirm' if stage else 'discovery'}-{family:03}"
            group = []
            for orientation, target, scope in product(range(2), repeat=3):
                c, d = COLORS[orientation], COLORS[1-orientation]
                if frame == 0:
                    doc = f"Archive: crate {a} is {c}; crate {b} is {d}.\nOnsite: crate {a} is {d}; crate {b} is {c}.\n"
                elif frame == 1:
                    doc = f"The archived report puts {c} on container {a} and {d} on container {b}.\nThe onsite inspection puts {d} on container {a} and {c} on container {b}.\n"
                else:
                    doc = f"Recorded inventory\n{a} — {c}\n{b} — {d}\nObserved inventory\n{a} — {d}\n{b} — {c}\n"
                relay = ("End of the entries.\nWait for the next line.\n", "This concludes the report.\nThe next line starts a separate task.\n",
                         "Both lists are complete.\nA request follows below.\n")[frame]
                alias = (a, b)[target]
                source = (("Archive", "Onsite"), ("archived report", "onsite inspection"),
                          ("recorded inventory", "observed inventory"))[frame][scope]
                query = (f"{source} only: what color is crate {alias}? Reply with the lowercase color word.",
                         f"Give the lowercase color of container {alias} as given by the {source}.",
                         f"Using the {source}, state the color assigned to {alias}. Write just the lowercase color word.")[frame]
                rendered = tok.apply_chat_template([{"role": "user", "content": doc+relay+query}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) + "Answer:\n"
                enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
                doc_end = rendered.index(doc) + len(doc)
                query_start = doc_end + len(relay)
                if any(lo < boundary < hi for lo, hi in enc["offset_mapping"] for boundary in (doc_end, query_start)):
                    raise ValueError("document/relay/question cut crosses a token")
                relay_start, query_token = [next(i for i, (lo, _) in enumerate(enc["offset_mapping"]) if lo >= x)
                                            for x in (doc_end, query_start)]
                choices = [tok.encode(color, add_special_tokens=False) for color in COLORS]
                if any(len(ids) != 1 or tok.encode(rendered + color, add_special_tokens=False) != enc["input_ids"] + ids
                       for color, ids in zip(COLORS, choices, strict=True)):
                    raise ValueError("color does not have a clean single-token answer boundary")
                group.append(dict(id=f"{family_id}-{orientation}{target}{scope}", stage=stage, family=family_id,
                                  frame=frame, orientation=orientation, target=target, scope=scope,
                                  prompt=rendered, tokens=enc["input_ids"], relay_start=relay_start, query=query_token,
                                  choice_ids=[ids[0] for ids in choices], gold=orientation ^ target ^ scope))
            if any(len({(r['relay_start'], r['query'], len(r['tokens'])) for r in group if r['scope'] == s}) != 1 for s in (0, 1)):
                raise ValueError("within-scope token alignment changed")
            for orientation in (0, 1):
                if len({tuple(r['tokens'][:r['query']]) for r in group if r['orientation'] == orientation}) != 1:
                    raise ValueError("question changed pre-question tokens")
            for target, scope in product(range(2), repeat=2):
                x, y = [r for r in group if r['target'] == target and r['scope'] == scope]
                if Counter(x['tokens']) != Counter(y['tokens']):
                    raise ValueError("word-bag control failed")
            rows.extend(group)
    return rows


def prepare():
    from transformers import AutoTokenizer

    proof = digest({name: sha(MODEL / name) for name in TOKENIZER_FILES if (MODEL / name).is_file()})
    if proof != TOKENIZER_ID:
        raise ValueError("local tokenizer differs from audited snapshot")
    rows = split_material(AutoTokenizer.from_pretrained(MODEL, local_files_only=True))
    checked_dir(OUT, 32 * 2**20, operation="范围颜色预注册")
    path = OUT / "binding-relay-plan.json"
    if path.exists():
        raise FileExistsError(path)
    questions = OUT / "binding-relay-questions.json"
    save(questions, rows)
    question_sha, entry_sha = sha(questions), sha(ENTRY)
    code = digest(sources())
    plan = dict(schema="research_test_plan.v1", kind="real", title="对象颜色与档案选源：提前保存绑定还是提问后重读",
                objective="区分问题前中立位置传递对象颜色绑定，与问题后直接读取档案或现场记录。",
                principle="固定词袋，交换颜色绑定。查询目标和范围后置。分别隔断原文和中立中继进入问题的全部注意力边。",
                hypothesis="预问题中继可在新句式预测绑定，并在直接原文路径阻断后保留随绑定交换而改变的部分作答。",
                alternatives="问题后重读原文、文档位置本身提前加工过绑定、两路协同或补偿、中继只保存词序。",
                criteria=CRITERIA, domains=["KN", "RS"], family_ids=["CLT-ROLE-REREAD", "COLOR-NAMING-COMPARISON", "COLOR-SCOPE-RELAY"],
                entrypoint_sha256=entry_sha, execution_args=[],
                dataset=dict(id="scope-binding-relay", revision="v2", content_sha256=question_sha,
                             questions_file_sha256=question_sha, split_manifest_sha256=digest([SEED, FAMILIES, [0, 1, 2]]),
                             case_count=len(rows), exposure="New constructed crate IDs and three frames; confirm frames 1 and 2 do not enter discovery. Historical records are design controls only."),
                acquisition=dict(model_id="Qwen3-14B", model_snapshot=MODEL_ID, tokenizer_sha256=proof,
                                 entrypoint_sha256=entry_sha, questions_file_sha256=question_sha,
                                 inputs_sha256=digest([r["tokens"] for r in rows]), code_sha256=code,
                                 environment_sha256=digest(runtime_identity()), precision="native BF16, FP32 dose accounting",
                                 seed=SEED, generation=dict(script_args=[], sample=False, max_new_tokens=1),
                                 capture=dict(layers=list(range(40)), readouts=["last token before question", "answer-prefix final token"],
                                              boundary="Residual after fixed blocks; all-head native attention post-O; full-vocabulary top-1")),
                analysis=dict(method_version="scope_binding_relay.v1", code_sha256=code, scoring_sha256=digest(CRITERIA),
                              selection=dict(stage="binding-relay", layers=LAYERS, families=FAMILIES, arms=ARMS, prior=SOURCE_RUN),
                              required_artifact_roles=sorted(CORE_ROLES | {"native_field"})))
    validate_plan(plan)
    save(path, plan)
    print(f"{len(rows)} aligned questions; {path}; plan {digest(plan)}", flush=True)


def receiver_mask(mask, rows):
    cut = mask.clone()
    for i, row in enumerate(rows):
        n = cut.shape[-1]
        pad = n - len(row["tokens"])
        relay, query = (pad + row[key] for key in ("relay_start", "query"))
        arm = row["arm"]
        if arm.endswith("relay"):
            cut[i, :, query:, relay:query] = False
        elif arm.endswith("document"):
            cut[i, :, query:, :relay] = False
        elif arm == "cut_both":
            cut[i, :, query:, :query] = False
    return cut


def score_batch(rt, rows, stage, stream, artifacts, work, index, start):
    t = rt.t
    native = np.empty((len(rows), len(LAYERS), rt.dim), dtype=np.uint16) if stage != "causal" else None
    doses = np.zeros((len(rows), 40, 2))
    original = rt.block
    state = {"offset": 0}

    def block(b):
        state["offset"] = 0
        layer = original(b)
        if stage != "causal":
            return layer

        def intervene(module, args, kwargs, result):
            output = result[0]
            lo = state["offset"]
            batch = rows[lo:lo+len(output)]
            state["offset"] += len(output)
            if all(row["arm"] == "baseline" for row in batch):
                return result
            modified = dict(kwargs, attention_mask=receiver_mask(kwargs["attention_mask"], batch))
            candidate = module.forward(*args, **modified)[0]
            writes = output.clone()
            for i, row in enumerate(batch):
                q = output.shape[1] - len(row["tokens"]) + row["query"]
                original_write = output[i, q:]
                target_write = candidate[i, q:]
                delta = target_write.float() - original_write.float()
                if row["arm"].startswith("cut_"):
                    writes[i, q:] = target_write
                elif row["arm"].startswith("control_"):
                    pair = delta.reshape(len(delta), -1, 2)
                    noise = t.stack((-pair[..., 1], pair[..., 0]), dim=-1).reshape_as(delta)
                    writes[i, q:] = (original_write.float() + noise).to(output.dtype)
                elif row["arm"].startswith("restore_"):
                    # Execute the masked branch above, then restore the original BF16 receiver write.
                    writes[i, q:] = original_write
                applied = writes[i, q:].float() - original_write.float()
                doses[lo+i, b] = t.stack((delta.square().sum(), applied.square().sum())).cpu().tolist()
            return (writes, *result[1:])

        layer.self_attn.register_forward_hook(intervene, with_kwargs=True)
        return layer

    def capture(b, layer, x, y, batch, length, offset=0, **kwargs):
        if native is not None and b in LAYERS:
            pos = t.tensor([length-len(r['tokens'])+r['query']-1 for r in batch], device=y.device)
            native[offset:offset+len(batch), LAYERS.index(b)] = y[t.arange(len(batch), device=y.device), pos].contiguous().view(t.uint16).cpu().numpy()
        if b >= 0 and (b+1) % 8 == 0 and offset+len(batch) == len(rows):
            from source_binding_resources import resources
            elapsed = time.monotonic() - start
            status = resources()
            print(json.dumps(dict(stage=stage, chunk=index, layers_done=b+1, layers_total=40, elapsed=elapsed,
                                  eta_chunk_seconds=elapsed*(39-b)/(b+1), **status)), flush=True)
            if (status["shared_gpu_bytes"] or 0) > 2**30:
                raise RuntimeError("shared GPU residency exceeded 1 GiB")

    rt.block = block
    try:
        _, logits = rt.forward(rows, capture)
    finally:
        rt.block = original
    top = logits.argmax(-1).cpu().tolist()
    scores = logits[:, rows[0]["choice_ids"]].cpu().numpy()
    records = []
    for i, row in enumerate(rows):
        chosen = int(top[i])
        record = {k: row[k] for k in ("id", "family", "frame", "scope", "target", "orientation", "gold", "arm")}
        record.update(top_id=chosen, actual=rt.tok.decode([chosen]), choices=list(COLORS),
                      correct=chosen == row["choice_ids"][row["gold"]], scores=scores[i].tolist(),
                      id_hash=sha_logits(logits[i]), margin=float(scores[i, row["gold"]]-scores[i, 1-row["gold"]]),
                      dose=np.sqrt(doses[i].sum(axis=0)).tolist())
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
    if native is not None:
        file = work / f"native-{index:04}.npz"
        np.savez_compressed(file, relay_bf16=native, case_id=[r["id"] for r in rows], blocks=LAYERS,
                            token=[r["query"]-1 for r in rows])
        artifacts.append(dict(path=file.name, role="native_field", retention="important"))
        save(work / "artifacts.json", dict(schema="research_test_files.v1", files=artifacts))
    return records, native


def sha_logits(logit):
    import hashlib
    return hashlib.sha256(logit.contiguous().cpu().numpy().tobytes()).hexdigest()


def behavior(records):
    cells = defaultdict(list)
    for r in records:
        cells[f"{r['frame']}/{r['scope']}/{r['target']}"].append(r['correct'])
    stats = {key: dict(correct=sum(a), n=len(a)) for key, a in cells.items()}
    return dict(passed=all(sum(a)/len(a) >= .85 for a in cells.values()), cells=stats)


def interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci99=np.quantile(draws, [.005, .995]).tolist(), families=len(values))


def paired_fields(native):
    groups = native.reshape(-1, 2, 4, len(LAYERS), native.shape[-1])
    if not np.all(groups == groups[:, :, :1]):
        raise ValueError("same causal prefix produced different native states across future questions")
    first = groups[:, :, 0].astype(np.uint32)
    return (first << 16).view(np.float32)


def fit(native):
    states = paired_fields(native)
    diff = states[:, 1] - states[:, 0]
    scales = np.maximum(np.linalg.norm(diff, axis=-1).mean(axis=0), 1e-9)
    axis = (diff / scales[:, None]).mean(axis=0)
    center = states.mean(axis=(0, 1))
    return dict(scales=scales.tolist(), axis=axis.tolist(), center=center.tolist(), layers=LAYERS)


def predictive(records, native, candidate):
    states = paired_fields(native)
    scores = (((states-np.asarray(candidate["center"])) / np.asarray(candidate["scales"])[:, None]) * candidate["axis"]).sum(axis=(2,3))
    cells = defaultdict(lambda: defaultdict(list))
    output = []
    for i, r in enumerate(records):
        score = float(scores[i//8, r["orientation"]])
        orientation = int(score > 0)
        predicted = orientation ^ r["target"] ^ r["scope"]
        correct = score != 0 and COLORS[predicted] == r["actual"]
        cells[r["frame"]][r["family"]].append(correct)
        output.append(dict(r, arm="frozen_prediction", score=score, prediction=COLORS[predicted],
                           prediction_matches_model=correct, symbolic_reread_matches_model=r["correct"]))
    stats = {key: interval([np.mean(row) for row in groups.values()]) for key, groups in cells.items()}
    return dict(passed=all(s["mean"] >= .75 and s["ci99"][0] > .5 for s in stats.values()), by_frame=stats), output


def effects(records):
    groups = defaultdict(dict)
    for row in records:
        groups[row["id"]][row["arm"]] = row
    cells = defaultdict(list)
    for group in groups.values():
        row = group["baseline"]
        cells[f"{row['frame']}/{row['scope']}"].append(group)
    report = {}
    for cell, items in cells.items():
        panel = {}
        for route in ("relay", "document"):
            cut, control, restored = (f"{arm}_{route}" for arm in ("cut", "control", "restore"))
            by_family = defaultdict(list)
            matched = preserved = exact = 0
            for g in items:
                by_family[g[cut]["family"]].append(g[control]["margin"] - g[cut]["margin"])
                target = g[cut]["dose"][1]
                matched += target > 0 and abs(g[control]["dose"][1]-target) <= .10 * target
                preserved += g[cut]["correct"]
                exact += g[restored]["id_hash"] == g["baseline"]["id_hash"]
            stats = interval([np.mean(v) for v in by_family.values()])
            panel[route] = dict(loss=stats, matched=[matched, len(items)], other_route_greedy=[preserved, len(items)],
                                native_recovery=[exact, len(items)], supported=matched/len(items) >= .95 and exact == len(items)
                                and stats["mean"] >= .25 and stats["ci99"][0] > 0)
        panel["double_cut_mean_gold_margin"] = float(np.mean([g["cut_both"]["margin"] for g in items]))
        panel["double_cut_leakage_pass"] = abs(panel["double_cut_mean_gold_margin"]) <= .10
        report[cell] = panel
    return report


def execute():
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime

    work = Path(os.environ["AI2050_TEST_WORK_DIR"])
    questions = Path(os.environ["AI2050_TEST_QUESTIONS"])
    plan, preflight, review = [read(questions.parent / name) for name in ("protocol.json", "preflight.json", "review.json")]
    if (not preflight.get("preflight_id") or preflight["plan_sha256"] != digest(plan)
            or not review.get("server_run_id") or sha(questions) != plan["dataset"]["questions_file_sha256"]
            or sha(Path(__file__)) != plan["entrypoint_sha256"] or digest(sources()) != plan["acquisition"]["code_sha256"]):
        raise ValueError("frozen inputs, code or registered preflight mismatch")
    print(json.dumps(dict(stage="verify_model_snapshot", done=0, total=1)), flush=True)
    if snapshot(MODEL) != MODEL_ID:
        raise ValueError("local model differs from frozen BF16 snapshot")
    material = read(questions)
    parts = [[dict(r, arm="baseline") for r in material if r["stage"] == stage] for stage in (0, 1)]
    if [len(p) for p in parts] != [n*8 for n in FAMILIES]:
        raise ValueError("incomplete family split")
    artifacts = []
    bundle = work / "research-scripts.zip"
    with ZipFile(bundle, "x", compression=ZIP_DEFLATED) as archive:
        for rel in sources():
            archive.write(ROOT / rel, arcname=rel)
    artifacts.append(dict(path=bundle.name, role="script", retention="core"))
    save(work / "artifacts.json", dict(schema="research_test_files.v1", files=artifacts))
    report = dict(stage="binding-relay", run_id=review["server_run_id"], family_counts=list(FAMILIES),
                  historical_sources=[SOURCE_RUN, "MA-H3-v4", "KV-ORTH-20261116-v1", "CLT-ROLE-REREAD-v1"],
                  new_validation="Direct document vs pre-question neutral relay after matched assignment exchange; two new frames; onsite facts explicit.")
    index = 0

    def run_stage(stage, rows, chunk):
        nonlocal index
        result, fields = [], []
        start = time.monotonic()
        for lo in range(0, len(rows), chunk):
            check_write(work, 64 * 2**20, operation="原生源路干预产物")
            print(json.dumps(dict(stage=stage, done=lo, total=len(rows), elapsed=time.monotonic()-start,
                                  eta_seconds=(time.monotonic()-start)*(len(rows)-lo)/lo if lo else None)), flush=True)
            rec, native = score_batch(rt, rows[lo:lo+chunk], stage, stream, artifacts, work, index, time.monotonic())
            result.extend(rec)
            if native is not None:
                fields.append(native)
            index += 1
        return result, np.concatenate(fields) if fields else None

    with GpuLease(), (work / "case_results.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        rt = Runtime("qwen14")
        rt.t.backends.cuda.matmul.allow_tf32 = False
        with rt.t.inference_mode():
            records, native = run_stage("discovery", parts[0], 256)
            report["discovery_behavior"] = behavior(records)
            if report["discovery_behavior"]["passed"]:
                candidate = fit(native)
                report["candidate_sha256"] = digest(candidate)
                save(work / "candidate.json", candidate)
                artifacts.append(dict(path="candidate.json", role="evidence", retention="important"))
                save(work / "artifacts.json", dict(schema="research_test_files.v1", files=artifacts))
                print(json.dumps(dict(stage="frozen_candidate", sha256=digest(candidate))), flush=True)
                heldout, native = run_stage("confirm", parts[1], 256)
                report["confirm_behavior"] = behavior(heldout)
                report["prediction"], predictions = predictive(heldout, native, candidate)
                for row in predictions:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                if report["confirm_behavior"]["passed"] and report["prediction"]["passed"]:
                    groups = [parts[1][i:i+8] for i in range(0, len(parts[1]), 8)]
                    expanded = [dict(r, arm=arm) for group in groups for arm in ARMS for r in group]
                    intervention, _ = run_stage("causal", expanded, 256)
                    report["causal"] = effects(intervention)
                else:
                    report["stop_reason"] = "heldout_behavior_or_frozen_prediction_gate"
            else:
                report["stop_reason"] = "discovery_behavior_gate"
    save(work / "conclusion.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch-desktop", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch_desktop:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch("binding-relay", entry=ENTRY, output=OUT))
    else:
        execute()
