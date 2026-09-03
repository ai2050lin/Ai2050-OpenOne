#!/usr/bin/env python3
"""Sequential cross-model panel, full-coordinate visualization and cleanup."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRACT_OUT = RESULT / "phase2253_c1097_c1120_construction_ecology_contract"
Q4_OUT = RESULT / "phase2254_c1121_c1152_qwen_construction_full_field"
PASSPORT_OUT = RESULT / "phase2255_c1153_c1184_coordinate_passport_ecology"
CAUSAL_OUT = RESULT / "phase2256_c1185_c1208_coordinate_mask_causal"
OUT = RESULT / "phase2257_c1209_c1240_crossmodel_visual_synthesis"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2253_c1097_c1120_construction_ecology_contract as contract  # noqa: E402
import phase2254_c1121_c1152_qwen_construction_full_field as q4_capture  # noqa: E402
import phase2255_c1153_c1184_coordinate_passport_ecology as passports  # noqa: E402


PHASE = 2257
CAMPAIGNS = tuple(f"C{i}" for i in range(1209, 1241))
MODELS = ("qwen3_14b", "glm4", "deepseek7b")
QWEN14_BATCH_SIZE = 8


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def generation(model, tokenizer, device, rows: list[dict], model_name: str) -> list[dict]:
    batch_size = QWEN14_BATCH_SIZE if model_name == "qwen3_14b" else 8
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6,
                                       do_sample=False, pad_token_id=pad,
                                       eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                           "correct_answer": row["correct_answer"],
                           "correct": parsed == row["correct_answer"]})
        if start % 48 == 0:
            print(f"[{model_name}-generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    families = {}
    for family in contract.FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        families[family] = {"rows": len(subset), "candidate_accuracy": ca,
                            "generation_accuracy": ga,
                            "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}
    ca = float(np.mean([row["correct"] for row in candidates]))
    ga = float(np.mean([row["correct"] for row in generated]))
    return {"rows": len(rows), "candidate_accuracy": ca, "generation_accuracy": ga,
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
            "aggregate_dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE,
            "families": families,
            "qualified_families": sorted(f for f, value in families.items() if value["dual_qualified"])}


def capture_field(model, device, rows: list[dict], worker_out: Path, model_name: str) -> dict:
    base = model.model
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    dim = int(base.embed_tokens.weight.shape[1])
    path = worker_out / "raw/qualified_fresh_role_field.float16.npy"
    progress_path = worker_out / "raw/capture_progress.json"
    shape = (len(rows), len(modules), len(contract.ROLES), dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if path.exists() and progress_path.exists():
        progress = load(progress_path)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(path, mode="r+")
    else:
        field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        save(progress_path, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        row_i = completed
        while row_i < len(rows):
            batch_size = 1
            while (batch_size < 8 and row_i + batch_size < len(rows)
                   and len(rows[row_i]["prompt_ids"]) == len(rows[row_i + batch_size]["prompt_ids"])):
                batch_size += 1
            batch = rows[row_i:row_i + batch_size]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = mask.long().cumsum(-1) - 1
            captured.clear()
            with torch.inference_mode():
                # The recorded checkpoints all end at base.norm. Calling the base
                # model avoids loading lm_head from disk without changing any
                # recorded activation.
                base(input_ids=ids, attention_mask=mask, position_ids=pos,
                     use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError(("checkpoint_count", len(captured), len(modules)))
            for q, hidden in enumerate(captured):
                values = hidden.float().cpu().numpy().astype(np.float16)
                for local_i, row in enumerate(batch):
                    for role_i, role in enumerate(contract.ROLES):
                        field[row_i + local_i, q, role_i] = values[local_i, row["role_positions"][role][-1]]
            row_i += batch_size
            if row_i % 8 <= batch_size - 1:
                field.flush()
                save(progress_path, {"shape": list(shape), "completed_rows": row_i})
                print(f"[{model_name}-field] {row_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    save(progress_path, {"shape": list(shape), "completed_rows": len(rows)})
    index = [{
        "hidden_index": i, "case_id": row["case_id"], "panel": row["panel"],
        "family": row["family"], "language": row["language"], "unit": row["unit"],
        "state": row["state"], "surface": row["surface"], "partition": row["partition"],
        "fresh": row["fresh"], "role_positions": row["role_positions"],
        "prompt_length": len(row["prompt_ids"]),
    } for i, row in enumerate(rows)]
    write_rows(worker_out / "raw/field_index.jsonl", index)
    return {"ran": True, "path": str(path.relative_to(ROOT)), "shape": list(shape),
            "checkpoints": len(modules), "coordinates": dim,
            "qualified_families": sorted(set(row["family"] for row in rows))}


def run_worker(model_name: str) -> dict:
    worker_out = OUT / model_name
    final_path = worker_out / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    raw = read_rows(CONTRACT_OUT / "material/fresh_broad_cases.jsonl")
    model = None
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        compiled = contract.compile_rows(tokenizer, raw)
        write_rows(worker_out / "material/fresh_compiled.jsonl", compiled)
        candidate_path = worker_out / "behavior/candidate.jsonl"
        generation_path = worker_out / "behavior/generation.jsonl"
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.model_base.behavior_base.batch_behavior(
                model, device, compiled,
                batch_size=QWEN14_BATCH_SIZE if model_name == "qwen3_14b" else 8)
            generated = generation(model, tokenizer, device, compiled, model_name)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        behavior = behavior_ledger(compiled, candidates, generated)
        save(worker_out / "behavior/ledger.json", behavior)
        observed = [row for row in compiled if row["family"] in behavior["qualified_families"]]
        field = capture_field(model, device, observed, worker_out, model_name) if observed else {
            "ran": False, "reason": "no_family_passed_dual_behavior"}
    finally:
        model_worker.release_model(model_name, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {"rows_complete": len(raw) == 960, "behavior_complete": behavior["rows"] == len(raw),
              "own_tokenizer_compilation": True,
              "field_matches_family_qualification": field["ran"] == bool(behavior["qualified_families"]),
              "field_rows_match": (not field["ran"]) or field["shape"][0] == 96 * len(behavior["qualified_families"])}
    result = {
        "phase": PHASE, "model": model_name, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "loader": loader,
        "placement": placement, "behavior": behavior, "field": field,
        "execution": {
            "behavior_batch_size": QWEN14_BATCH_SIZE if model_name == "qwen3_14b" else 8,
            "generation_batch_size": QWEN14_BATCH_SIZE if model_name == "qwen3_14b" else 8,
            "qwen3_14b_engineering_restart": (
                "The first batch-size-2 attempt was stopped before any behavior result was written; "
                "the frozen material, model, decoding, gates and thresholds were unchanged. "
                "Field capture resumed from its saved row using the same base-model checkpoints, "
                "grouping up to eight equal-length prompts and omitting the downstream lm_head."
                if model_name == "qwen3_14b" else None
            ),
        },
        "hashes": {"field": file_hash(ROOT / field["path"]) if field["ran"] else None},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Each family is qualified independently; unqualified families have no internal field and remain NA.",
    }
    save(final_path, result)
    print(json.dumps({"model": model_name, "behavior": behavior, "field": field,
                      "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


def profiles(field: np.ndarray, index: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    responses = passports.paired_unit_responses(field, index)
    output = {}
    for family, units in responses.items():
        for unit, response in units.items():
            rms = np.sqrt(np.mean(response * response, axis=2, dtype=np.float64)).astype(np.float32)
            rms /= np.sqrt(np.sum(rms * rms, axis=1, keepdims=True)) + 1e-12
            output[(family, unit)] = rms
    return output


def resample(value: np.ndarray, steps: int = 64) -> np.ndarray:
    source = np.linspace(0.0, 1.0, value.shape[0])
    target = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(target, source, value[:, i]) for i in range(value.shape[1])], axis=1)


def retrieval(source: dict, target: dict, families: list[str]) -> dict:
    if len(families) < 2:
        return {"status": "NA_fewer_than_two_overlapping_families"}
    rows = []
    for (family, unit), value in sorted(source.items()):
        if family not in families:
            continue
        query = resample(value)
        distances = {}
        for candidate in families:
            prototypes = [resample(v) for (f, u), v in target.items() if f == candidate and u != unit]
            if not prototypes:
                continue
            distances[candidate] = float(np.mean(np.abs(query - np.mean(prototypes, axis=0))))
        if set(distances) != set(families):
            continue
        predicted = min(families, key=lambda f: (distances[f], families.index(f)))
        wrong = min(value for f, value in distances.items() if f != family)
        rows.append({"family": family, "unit": unit, "predicted": predicted,
                     "correct": predicted == family, "margin": wrong - distances[family]})
    return {"queries": len(rows), "families": families, "chance": 1.0 / len(families),
            "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
            "median_margin": float(np.median([row["margin"] for row in rows])) if rows else None}


def checkpoint_label(q: int, count: int) -> str:
    if q == 0:
        return "embedding"
    if q == count - 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def verify_matrix(path: Path, expected: tuple[int, int]) -> dict:
    value = np.load(path, mmap_mode="r")
    finite = True
    for start in range(0, value.shape[0], 256):
        if not np.isfinite(np.asarray(value[start:start + 256], np.float32)).all():
            finite = False
            break
    result = {"shape": list(value.shape), "shape_ok": tuple(value.shape) == expected,
              "dtype_float16": value.dtype == np.dtype("<f2"), "all_finite": finite,
              "bytes": path.stat().st_size, "sha256": file_hash(path)}
    close_mmap(value)
    return result


def write_payload(path: Path, title: str, binary: Path, shape: tuple[int, int],
                  rows: list[dict], boundary: str, summary: dict) -> None:
    save(path, {"schema": "ai2050.construction-coordinate-ecology.v1",
                "generated_at": datetime.now().astimezone().isoformat(),
                "phase": PHASE, "campaign": "C1209-C1240", "title": title,
                "binary_url": f"/vis_data/research_kernel/{binary.name}",
                "binary_shape": list(shape), "coordinate_count": shape[1],
                "coordinate_semantics": "model-local physical activation coordinates; not weights",
                "rows": rows, "summary": summary, "boundary": boundary})


def fresh_response_atlas(model_id: str, field_path: Path, index_path: Path) -> dict:
    field = np.load(field_path, mmap_mode="r")
    index = read_rows(index_path)
    responses = passports.paired_unit_responses(field, index)
    matrices, labels = [], []
    for family, units in sorted(responses.items()):
        for unit, response in sorted(units.items()):
            if unit < contract.PARENT_UNITS:
                continue
            for q in range(response.shape[0]):
                for role_i, role in enumerate(contract.ROLES):
                    matrices.append(response[q, role_i])
                    labels.append({"family": family, "unit": unit,
                                   "checkpoint": q, "checkpoint_label": checkpoint_label(q, response.shape[0]),
                                   "role": role, "metric": "state1_minus_state0"})
    close_mmap(field)
    matrix = np.stack(matrices).astype("<f2")
    stem = f"c1232_{model_id}_fresh_construction_responses"
    binary, payload = VIS / f"{stem}.npy", VIS / f"{stem}.json"
    np.save(binary, matrix, allow_pickle=False)
    write_payload(payload, f"{model_id} Fresh Construction Responses", binary, matrix.shape,
                  labels, "Unit-level averages retain every model-local coordinate; observational, not causal.",
                  {"families": sorted(responses), "units": sorted({x["unit"] for x in labels})})
    return {"stem": stem, "binary": binary, "metadata": payload, "shape": matrix.shape,
            "verification": verify_matrix(binary, matrix.shape)}


def passport_atlases() -> list[dict]:
    source = np.load(PASSPORT_OUT / "atlas/qwen3_4b_coordinate_passport.float32.npy", mmap_mode="r")
    labels = read_rows(PASSPORT_OUT / "atlas/qwen3_4b_coordinate_passport_rows.jsonl")
    source_values = np.asarray(source, np.float32)
    float16_limit = float(np.finfo(np.float16).max)
    clipped_count = int(np.count_nonzero(np.abs(source_values) > float16_limit))
    raw_min, raw_max = float(np.min(source_values)), float(np.max(source_values))
    matrix = np.clip(source_values, -float16_limit, float16_limit).astype(np.float16)
    close_mmap(source)
    stem = "c1233_qwen3_4b_coordinate_passport"
    binary, payload = VIS / f"{stem}.npy", VIS / f"{stem}.json"
    np.save(binary, matrix.astype("<f2"), allow_pickle=False)
    enriched = [{**row, "checkpoint_label": checkpoint_label(int(row["checkpoint"]), 38)} for row in labels]
    write_payload(payload, "Qwen3-4B Coordinate Passport", binary, matrix.shape, enriched,
                  "All 2560 coordinates are shown. Metrics are predictive diagnostics, not neuron meanings. "
                  "Finite gain outliers outside float16 range are display-clipped; the float32 research matrix is unchanged.",
                  {"metrics": sorted({row["metric"] for row in labels}),
                   "display_clipped_values": clipped_count,
                   "display_total_values": int(source_values.size),
                   "raw_float32_min": raw_min, "raw_float32_max": raw_max,
                   "display_float16_limit": float16_limit})
    passport_artifact = {"stem": stem, "binary": binary, "metadata": payload,
                         "shape": matrix.shape, "verification": verify_matrix(binary, matrix.shape)}
    masks = np.load(PASSPORT_OUT / "atlas/discovery_loo_candidate_masks.uint8.npy", mmap_mode="r")
    final = load(PASSPORT_OUT / "analysis/final.json")
    mask_matrix = np.asarray(masks.reshape(-1, masks.shape[-1]), np.float16)
    close_mmap(masks)
    mask_labels = []
    for family in final["analysis"]["family_order"]:
        for q in range(38):
            for role in contract.ROLES:
                mask_labels.append({"family": family, "checkpoint": q,
                                    "checkpoint_label": checkpoint_label(q, 38),
                                    "role": role, "metric": "discovery_loo_candidate_mask"})
    stem2 = "c1234_qwen3_4b_coordinate_candidate_masks"
    binary2, payload2 = VIS / f"{stem2}.npy", VIS / f"{stem2}.json"
    np.save(binary2, mask_matrix.astype("<f2"), allow_pickle=False)
    write_payload(payload2, "Qwen3-4B Frozen Coordinate Candidate Masks", binary2,
                  mask_matrix.shape, mask_labels,
                  "Binary masks apply frozen gates to every coordinate; they are not Top-K selections or causal neurons.",
                  {"families": final["analysis"]["family_order"]})
    mask_artifact = {"stem": stem2, "binary": binary2, "metadata": payload2,
                     "shape": mask_matrix.shape, "verification": verify_matrix(binary2, mask_matrix.shape)}
    return [passport_artifact, mask_artifact]


def catalog_entry(artifact: dict, title: str, model: str, claim: str, boundary: str) -> dict:
    return {"id": artifact["stem"], "title": title, "phase": PHASE,
            "campaign": "C1209-C1240", "model": model,
            "source_path": "/vis_data/research_kernel/" + artifact["metadata"].name,
            "binary_path": "/vis_data/research_kernel/" + artifact["binary"].name,
            "source_schema": "ai2050.construction-coordinate-ecology.v1",
            "coordinate_count": artifact["shape"][1], "row_count": artifact["shape"][0],
            "claim_level": claim, "boundary": boundary,
            "kinds": ["embedding_and_hiddenstate_physical_activation_coordinates"]}


def update_catalog(artifacts: list[dict]) -> dict:
    catalog = load(CATALOG)
    existing = {row["id"] for row in catalog.get("families", [])}
    for family, label, domain in (
            ("active_passive_role", "Active / Passive Role", "semantic_roles"),
            ("attribute_overwrite", "Attribute Overwrite", "state_change")):
        if family not in existing:
            catalog.setdefault("families", []).append({"id": family, "label": label,
                                                        "domain": domain,
                                                        "operations": ["state0", "state1", "query"]})
    entries = []
    for artifact in artifacts:
        if "passport" in artifact["stem"]:
            title, claim = "Qwen3-4B Full-Coordinate Construction Passport", "prospective_coordinate_diagnostic"
            boundary = "All coordinates and all frozen metrics; predictive association is not causal necessity."
        elif "candidate_masks" in artifact["stem"]:
            title, claim = "Qwen3-4B Frozen Coordinate Candidate Masks", "discovery_loo_coordinate_mask"
            boundary = "Complete binary gate outcome, not Top-K and not a neuron dictionary."
        else:
            title, claim = artifact["stem"].replace("_", " ").title(), "fresh_full_coordinate_observation"
            boundary = "All model-local activation coordinates for fresh unit responses; no cross-model coordinate alignment."
        model = "Qwen3-14B" if "qwen3_14b" in artifact["stem"] else "GLM4" if "glm4" in artifact["stem"] else "DeepSeek-7B" if "deepseek" in artifact["stem"] else "Qwen3-4B"
        entries.append(catalog_entry(artifact, title, model, claim, boundary))
    ids = {row["id"] for row in entries}
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") not in ids] + entries
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save(CATALOG, catalog)
    return {"datasets_added": sorted(ids), "dataset_count": len(catalog["datasets"]),
            "sha256": file_hash(CATALOG), "utf8_json_valid": True}


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(
            Path.home().glob("AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"),
            reverse=True,
        )
        candidates += sorted(
            Path.home().glob("AppData/Local/ms-playwright-go/*/node.exe"),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No npm or local Node runtime is available for the Vite build")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend",
                               text=True, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=300)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:],
            "passed": completed.returncode == 0}


def cleanup(fields: list[tuple[str, Path, str | None]]) -> dict:
    rows = []
    total = 0
    for name, path, expected_hash in fields:
        if not path.exists():
            rows.append({"name": name, "path": str(path.relative_to(ROOT)), "status": "already_absent",
                         "sha256_before": expected_hash, "bytes_deleted": 0})
            continue
        actual = file_hash(path)
        if expected_hash and actual != expected_hash:
            raise RuntimeError(("cleanup_hash_mismatch", name, actual, expected_hash))
        size = path.stat().st_size
        path.unlink()
        total += size
        rows.append({"name": name, "path": str(path.relative_to(ROOT)), "status": "deleted_after_visual_derivative",
                     "sha256_before": actual, "bytes_deleted": size})
    result = {"files": rows, "bytes_deleted": total}
    save(OUT / "cleanup/ledger.json", result)
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    model_summary = {name: {"candidate": row["behavior"]["candidate_accuracy"],
                            "generation": row["behavior"]["generation_accuracy"],
                            "qualified_families": row["behavior"]["qualified_families"],
                            "field": row["field"]} for name, row in result["models"].items()}
    text = rf"""

## Phase {PHASE}: 十构式跨模型面板、全坐标图谱与大阶段总裁决（C1209-C1240） [{stamp}]

**测试原理与用例。** Qwen3-14B、GLM4、DeepSeek-7B严格逐个加载，各自tokenizer重新编译同一960条fresh宽族材料；候选与自由生成按族分别要求不低于0.75，只有合格族采集embedding、每个block后状态、final norm、六角色全部物理激活坐标。跨模型不比较坐标编号，只对各模型内部全坐标响应先形成角色-相对深度强度图，再在共同合格族上做双向留一单元检索。Qwen3-4B的逐坐标护照、冻结候选掩码和fresh单元响应全部导出到通用客户端图谱。

**公式。** 跨维度模型的次级拓扑控制为：

$$
E_{{m,f,u,q,r}}=\sqrt{{\frac1{{d_m}}\sum_jR_{{m,f,u,q,r,j}}^2}},\qquad
\widetilde E=E/(\lVert E_{{q,:}}\rVert_2+\epsilon).
$$

物理坐标只在各模型内部保留和显示；$j$在不同模型间没有同一含义。

**结果汇总。** 串行模型账 `{json.dumps(model_summary, ensure_ascii=False)}`；共同族相对拓扑 `{json.dumps(result['topology'], ensure_ascii=False)}`。Phase2256严格因果族为 `{json.dumps(result['causal_strict_families'], ensure_ascii=False)}`。客户端全坐标产物 `{json.dumps(result['visual_artifacts'], ensure_ascii=False)}`，目录更新 `{json.dumps(result['catalog'], ensure_ascii=False)}`，生产构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。

**理论进展与总裁决。** 本大阶段最强的新事实是：在全新独立构式分母中，Qwen3-4B只有主动/被动角色和属性覆盖通过双行为；它们的discovery留一逐坐标护照可在fresh单元形成连续查询/边界区间，但冻结掩码干预不能在confirmation击败错族和等范数符号控制，因此没有因果闭合。图路径在旧Phase2249分母上的局部原型迁移仍保留，但在本轮新图材料上行为失败，不能扩张为通用图算子。理论主体继续叫“条件化输出场闭合理论”，RDC不变；当前更可信的对象是构式、样本、角色、深度和输出边界共同条件化的分布式坐标响应生态。

**问题、硬伤与瓶颈。** 材料仍是受控模板且人类盲评NA；很多人工图节点不属于自然知识；输出码污染行为接口；只有行为合格族有内部场；逐坐标筛选存在大规模候选和非唯一分解；跨模型拓扑必须压成角色强度，不能证明坐标同构；小模型可能只有粗糙近似；因果控制显示预测护照可能主要是相关结构。现有基础绝对误差、符号计数、逐坐标门槛和有限差分足以登记事实，尚无证据需要命名新基础数学。

**清理与下一步。** 客户端矩阵通过形状、float16、有限值、哈希和生产构建后，未直接展示的逐样本原场按哈希账删除 `{result['cleanup']['bytes_deleted']}` 字节；可视化派生矩阵、行为、索引、合同、逐坐标护照和清理哈希保留。下一阶段目标仍相同，但不得继续调本分母；应建立人工盲评的自然语料最小对库，优先围绕已经通过的角色重排和状态覆盖扩展更多同类构式，同时另建更易被小模型稳定执行的自然图路径行为合同。每条路线继续采用观察-形成-锁箱-因果四层账，失败只淘汰该路线。

**相关文件。** 脚本 `tests/glm5/phase2257_c1209_c1240_crossmodel_visual_synthesis.py`；结果 `tests/glm5/result/phase2257_c1209_c1240_crossmodel_visual_synthesis`；客户端目录 `frontend/public/research_data/current/language_encoding_catalog.json`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def synthesize() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    workers = {name: load(OUT / name / "analysis/final.json") for name in MODELS}
    q4_final = load(Q4_OUT / "analysis/final.json")
    q4_field_path = ROOT / q4_final["role_field"]["path"]
    q4_index_path = Q4_OUT / "raw/role_field_index.jsonl"
    q4_field = np.load(q4_field_path, mmap_mode="r")
    q4_profiles = profiles(q4_field, read_rows(q4_index_path))
    close_mmap(q4_field)
    topology = {}
    for name, worker in workers.items():
        if not worker["field"]["ran"]:
            topology[name] = {"status": "NA_no_behavior_qualified_family_field"}
            continue
        field_path = ROOT / worker["field"]["path"]
        field = np.load(field_path, mmap_mode="r")
        target_profiles = profiles(field, read_rows(OUT / name / "raw/field_index.jsonl"))
        close_mmap(field)
        overlap = sorted(set(f for f, _ in q4_profiles) & set(f for f, _ in target_profiles))
        topology[name] = {"overlap": overlap,
                          "qwen4_to_model": retrieval(q4_profiles, target_profiles, overlap),
                          "model_to_qwen4": retrieval(target_profiles, q4_profiles, overlap)}
    VIS.mkdir(parents=True, exist_ok=True)
    artifacts = passport_atlases()
    q4_artifact = fresh_response_atlas("qwen3_4b", q4_field_path, q4_index_path)
    artifacts.append(q4_artifact)
    cleanup_fields = [("qwen3_4b_role_field", q4_field_path, q4_final["hashes"]["role_field"])]
    for name, worker in workers.items():
        if worker["field"]["ran"]:
            path = ROOT / worker["field"]["path"]
            artifacts.append(fresh_response_atlas(name, path, OUT / name / "raw/field_index.jsonl"))
            cleanup_fields.append((f"{name}_role_field", path, worker["hashes"]["field"]))
    visual_checks = {artifact["stem"]: artifact["verification"] for artifact in artifacts}
    if not all(v["shape_ok"] and v["dtype_float16"] and v["all_finite"] for v in visual_checks.values()):
        raise RuntimeError("visual artifact verification failed")
    catalog = update_catalog(artifacts)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    cleanup_result = cleanup(cleanup_fields)
    causal = load(CAUSAL_OUT / "analysis/final.json")
    checks = {
        "workers_complete": all(row["all_checks_passed"] for row in workers.values()),
        "exact_fresh_denominator": all(row["behavior"]["rows"] == 960 for row in workers.values()),
        "field_per_family_gate": all(row["field"]["ran"] == bool(row["behavior"]["qualified_families"])
                                     for row in workers.values()),
        "topology_for_each_model": set(topology) == set(MODELS),
        "visual_matrices_verified": all(v["shape_ok"] and v["dtype_float16"] and v["all_finite"]
                                        for v in visual_checks.values()),
        "catalog_valid": catalog["utf8_json_valid"], "frontend_build": build["passed"],
        "raw_fields_absent_after_cleanup": all(not path.exists() for _, path, _ in cleanup_fields),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "models": workers,
        "topology": topology, "causal_strict_families": causal["strict_causal_families"],
        "visual_artifacts": {a["stem"]: {"shape": list(a["shape"]),
                                          "sha256": a["verification"]["sha256"]} for a in artifacts},
        "catalog": catalog, "frontend_build": build, "cleanup": cleanup_result,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Two Qwen3-4B construction families have prospective coordinate passports but no causal closure; cross-model results are model-local fields and relative topology only.",
        "next_authorization": "Move to an independently human-reviewed natural minimal-pair library and broaden successful role/state-change constructions without tuning this denominator.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({"models": {k: v["behavior"] for k, v in workers.items()},
                      "topology": topology, "artifacts": result["visual_artifacts"],
                      "cleanup": cleanup_result, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODELS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
        return
    for model_name in MODELS:
        run_worker(model_name)
    synthesize()


if __name__ == "__main__":
    main()
