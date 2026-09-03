#!/usr/bin/env python3
"""Full-coordinate two-token teacher-forced VJP and finite compilation test."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2448 = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
OUT = RESULT / "phase2460_c42001_c42320_two_token_autoregressive_compilation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2460, "C42001-C42320", 2560
QPOINTS, DOSE, SHIFT = (16, 18), 0.02, 791
STEPS = ("first_token", "second_token_path_conditioned", "two_token_total")
FIELDS = ("gradient", "state_times_gradient")
VARIANTS = ("valid", "broken_a", "broken_b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); values = []
    while len(values) < count:
        p = rng.permutation(size)
        if np.all(p != np.arange(size)): values.append(p)
    return np.stack(values)


def selected_rows() -> list[dict]:
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    rows = [row for row in rows if int(row["unit"]) == 5 and row["surface"] == "natural" and int(row["direction"]) == 0]
    rows.sort(key=lambda row: row["case_id"])
    if len(rows) != 96 or any(len(row["target_ids"]) != 2 or len(row["foil_ids"]) != 2 for row in rows):
        raise RuntimeError("frozen two-token row contract failed")
    return rows


def semantic_directions() -> tuple[np.ndarray, list[str]]:
    final = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    passports = np.load(final["analysis"]["passports"], mmap_mode="r")
    raw = np.asarray(passports[0, 0, 0, :, 18], dtype=np.float32).copy()
    close(passports)
    rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=-1, keepdims=True))
    return (raw / np.maximum(rms, 1e-30)).astype(np.float32), final["analysis"]["families"]


def capture(rows: list[dict], directions: np.ndarray, families: list[str], family_permutations: np.ndarray) -> dict:
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "fields": raw / "two_token_query_fields.float32.npy",
        "margins": raw / "step_logprob_margin.float32.npy",
        "path_state_difference": raw / "target_foil_query_state_maxabs.float32.npy",
        "finite_odd": raw / "two_token_total_finite_odd.float32.npy",
        "finite_predicted": raw / "two_token_total_vjp_predicted.float32.npy",
        "finite_signed_margin": raw / "two_token_total_signed_margin.float32.npy",
    }
    controls = 2 + len(family_permutations)
    fields = np.lib.format.open_memmap(paths["fields"], mode="r+" if paths["fields"].exists() else "w+", dtype=np.float32, shape=(len(rows), 3, 2, 2, DIM))
    margins = np.lib.format.open_memmap(paths["margins"], mode="r+" if paths["margins"].exists() else "w+", dtype=np.float32, shape=(len(rows), 3))
    path_difference = np.lib.format.open_memmap(paths["path_state_difference"], mode="r+" if paths["path_state_difference"].exists() else "w+", dtype=np.float32, shape=(len(rows), 2))
    finite = np.lib.format.open_memmap(paths["finite_odd"], mode="r+" if paths["finite_odd"].exists() else "w+", dtype=np.float32, shape=(len(rows), controls))
    finite_pred = np.lib.format.open_memmap(paths["finite_predicted"], mode="r+" if paths["finite_predicted"].exists() else "w+", dtype=np.float32, shape=(len(rows), controls))
    finite_signed = np.lib.format.open_memmap(paths["finite_signed_margin"], mode="r+" if paths["finite_signed_margin"].exists() else "w+", dtype=np.float32, shape=(len(rows), controls, 2))
    vjp_progress_path, finite_progress_path = raw / "vjp_progress.json", raw / "finite_progress.json"
    vjp_completed = int(json.loads(vjp_progress_path.read_text(encoding="utf-8"))["completed"]) if vjp_progress_path.exists() else 0
    finite_completed = int(json.loads(finite_progress_path.read_text(encoding="utf-8"))["completed"]) if finite_progress_path.exists() else 0
    model = tokenizer = None
    if vjp_completed < len(rows) or finite_completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b"); model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        modules = field_utils.modules(model); device = model.get_input_embeddings().weight.device
    else:
        modules = None; device = None
    captures: dict[int, torch.Tensor] = {}; handles = []
    try:
        if vjp_completed < len(rows):
            def leaf_hook(_module, _inputs, result):
                tensor = result[0] if isinstance(result, tuple) else result
                if not tensor.requires_grad: tensor.requires_grad_(True)
            handles.append(modules[0].register_forward_hook(leaf_hook))
            for qpoint in QPOINTS:
                def field_hook(_module, _inputs, result, qpoint=qpoint):
                    captures[qpoint] = result[0] if isinstance(result, tuple) else result
                handles.append(modules[qpoint].register_forward_hook(field_hook))

            def path_fields(row: dict, candidate: list[int]) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
                prompt = row["prompt_ids"]; input_values = prompt + candidate[:-1]
                ids = torch.tensor([input_values], dtype=torch.long, device=device)
                attention = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=device)[None]
                captures.clear()
                with torch.enable_grad():
                    output = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False, return_dict=True)
                    logp = torch.log_softmax(output.logits.float(), dim=-1)
                    scores = [logp[0, len(prompt) - 1 + step, int(candidate[step])] for step in range(2)]
                    capture_list = [captures[qpoint] for qpoint in QPOINTS]
                    grad0 = torch.autograd.grad(scores[0], capture_list, retain_graph=True)
                    grad1 = torch.autograd.grad(scores[1], capture_list)
                token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
                states = [np.asarray(captures[q][0, token_index].detach().float().cpu(), dtype=np.float32) for q in QPOINTS]
                gradients = [np.stack([np.asarray(grad0[s][0, token_index].detach().float().cpu(), dtype=np.float32),
                                       np.asarray(grad1[s][0, token_index].detach().float().cpu(), dtype=np.float32)]) for s in range(2)]
                scalar = [float(score.detach().cpu()) for score in scores]
                del output, logp, scores, ids, attention, positions
                return scalar, states, gradients

            for row_index in range(vjp_completed, len(rows)):
                target_score, target_state, target_grad = path_fields(rows[row_index], rows[row_index]["target_ids"])
                foil_score, foil_state, foil_grad = path_fields(rows[row_index], rows[row_index]["foil_ids"])
                step_gradients = []
                for step in range(2):
                    step_gradients.append([target_grad[slot][step] - foil_grad[slot][step] for slot in range(2)])
                step_gradients.append([step_gradients[0][slot] + step_gradients[1][slot] for slot in range(2)])
                for step in range(3):
                    for slot in range(2):
                        state = (target_state[slot] + foil_state[slot]) / 2.0
                        fields[row_index, step, slot, 0] = step_gradients[step][slot]
                        fields[row_index, step, slot, 1] = state * step_gradients[step][slot]
                margins[row_index, 0] = target_score[0] - foil_score[0]
                margins[row_index, 1] = target_score[1] - foil_score[1]
                margins[row_index, 2] = margins[row_index, 0] + margins[row_index, 1]
                path_difference[row_index] = [float(np.max(np.abs(target_state[slot] - foil_state[slot]))) for slot in range(2)]
                for value in (fields, margins, path_difference): value.flush()
                save(vjp_progress_path, {"completed": row_index + 1, "rows": len(rows)})
                if (row_index + 1) % 8 == 0 or row_index + 1 == len(rows): print(f"[phase2460 VJP] {row_index + 1}/{len(rows)}", flush=True)
            for handle in handles: handle.remove()
            handles.clear()

        if finite_completed < len(rows):
            family_lookup = {family: i for i, family in enumerate(families)}
            language_lookup = {language: i for i, language in enumerate(("en", "zh"))}
            module = modules[18]
            for row_index in range(finite_completed, len(rows)):
                row = rows[row_index]; fi = family_lookup[row["family"]]; li = language_lookup[row["language"]]
                base = directions[li, fi]
                controls_direction = [base, np.roll(base, SHIFT)] + [directions[li, p[fi]] for p in family_permutations]
                prompt = row["prompt_ids"]
                token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
                # Frozen row state RMS from the same q18 query state.
                state_rms = None

                def sequence_score(candidate: list[int], sign: float, direction_tensor: torch.Tensor) -> float:
                    nonlocal state_rms
                    ids = torch.tensor([prompt + candidate[:-1]], dtype=torch.long, device=device)
                    attention = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=device)[None]
                    def intervention(_module, _inputs, result):
                        nonlocal state_rms
                        tensor = result[0] if isinstance(result, tuple) else result; altered = tensor.clone()
                        rms = tensor[0, token_index].detach().float().square().mean().sqrt()
                        if state_rms is None: state_rms = float(rms.cpu())
                        altered[0, token_index] = altered[0, token_index] + (sign * DOSE * rms * direction_tensor).to(altered.dtype)
                        return (altered,) + tuple(result[1:]) if isinstance(result, tuple) else altered
                    handle = module.register_forward_hook(intervention)
                    try:
                        with torch.inference_mode():
                            output = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False, return_dict=True)
                            logp = torch.log_softmax(output.logits.float(), dim=-1)
                            value = sum(logp[0, len(prompt)-1+step, int(candidate[step])] for step in range(2))
                            scalar = float(value.cpu())
                    finally: handle.remove()
                    return scalar

                total_gradient = np.asarray(fields[row_index, 2, 1, 0], dtype=np.float64)
                for control_index, direction in enumerate(controls_direction):
                    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device); signed_values = []
                    for sign_index, sign in enumerate((-1.0, 1.0)):
                        target_value = sequence_score(row["target_ids"], sign, direction_tensor)
                        foil_value = sequence_score(row["foil_ids"], sign, direction_tensor)
                        margin = target_value - foil_value
                        finite_signed[row_index, control_index, sign_index] = margin; signed_values.append(margin)
                    finite[row_index, control_index] = (signed_values[1] - signed_values[0]) / 2.0
                    finite_pred[row_index, control_index] = DOSE * float(state_rms) * float(np.dot(total_gradient, direction.astype(np.float64)))
                for value in (finite, finite_pred, finite_signed): value.flush()
                save(finite_progress_path, {"completed": row_index + 1, "rows": len(rows), "controls": controls})
                if (row_index + 1) % 8 == 0 or row_index + 1 == len(rows): print(f"[phase2460 finite] {row_index + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        del model, tokenizer
        for value in (fields, margins, path_difference, finite, finite_pred, finite_signed): value.flush(); close(value)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    index = OUT / "index/two_token_rows.jsonl"; index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps({key: row[key] for key in ("case_id", "family", "language", "variant", "query_role", "target_ids", "foil_ids")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {**{key: str(path) for key, path in paths.items()}, "field_shape": [len(rows), 3, 2, 2, DIM], "rows": len(rows),
            "steps": list(STEPS), "qpoints": list(QPOINTS), "fields_order": list(FIELDS), "all_physical_coordinates": True,
            "finite_controls": ["matched", "shift791"] + [f"family_derangement_{i}" for i in range(len(family_permutations))],
            "family_derangements": family_permutations.tolist(), "dose": DOSE,
            "vjp_forward_backward_paths": len(rows) * 2, "finite_forward_passes": len(rows) * controls * 4,
            "inference": "Qwen3-4B BF16 CUDA; exact two-token teacher-forced log-probability"}


def build_passports(rows: list[dict], values: np.ndarray) -> np.ndarray:
    lookup = {(row["family"], row["language"], row["variant"], row["query_role"]): i for i, row in enumerate(rows)}
    families = sorted({row["family"] for row in rows})
    output = np.zeros((2, 3, 2, 2, 2, 8, DIM), dtype=np.float32)
    # interaction, step, qslot, field, language, family, coordinate
    for li, language in enumerate(("en", "zh")):
        for fi, family in enumerate(families):
            role = {}
            for variant in VARIANTS:
                source, target = lookup[(family, language, variant, "source")], lookup[(family, language, variant, "target")]
                role[variant] = values[target] - values[source]
            output[0, :, :, :, li, fi] = role["valid"] - role["broken_a"]
            output[1, :, :, :, li, fi] = role["broken_a"] - role["broken_b"]
    return output


def finite_interactions(rows: list[dict], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {(row["family"], row["language"], row["variant"], row["query_role"]): i for i, row in enumerate(rows)}
    semantic, lexical = [], []
    for family in sorted({row["family"] for row in rows}):
        for language in ("en", "zh"):
            role = {}
            for variant in VARIANTS:
                source, target = lookup[(family, language, variant, "source")], lookup[(family, language, variant, "target")]
                role[variant] = values[target] - values[source]
            semantic.append(role["valid"] - role["broken_a"]); lexical.append(role["broken_a"] - role["broken_b"])
    return np.stack(semantic), np.stack(lexical)


def analyze(rows: list[dict], collection: dict, families: list[str]) -> dict:
    fields = np.load(collection["fields"], mmap_mode="r")
    passports = build_passports(rows, fields)
    passport_path = OUT / "derived/two_token_semantic_lexical_passports.float32.npy"; passport_path.parent.mkdir(parents=True, exist_ok=True); np.save(passport_path, passports)
    permutations = derangements(64, 8, 2460)
    crosslanguage = {}
    for interaction, interaction_name in enumerate(("semantic_validity", "lexical_control")):
        crosslanguage[interaction_name] = {}
        for field_index, field_name in enumerate(FIELDS):
            slot = 1 if field_index == 0 else 0
            crosslanguage[interaction_name][field_name] = {}
            for step, step_name in enumerate(STEPS):
                en, zh = passports[interaction, step, slot, field_index, 0], passports[interaction, step, slot, field_index, 1]
                observed = float(np.mean([cosine(en[f], zh[f]) for f in range(8)]))
                shifted = float(np.mean([cosine(en[f], np.roll(zh[f], SHIFT)) for f in range(8)]))
                null = np.asarray([np.mean([cosine(en[f], zh[p[f]]) for f in range(8)]) for p in permutations])
                q95 = float(np.quantile(null, .95))
                crosslanguage[interaction_name][field_name][step_name] = {"qpoint": QPOINTS[slot], "coordinate": observed, "shift791": shifted,
                    "family_null_mean": float(np.mean(null)), "family_null_q95": q95, "physical_advantage": observed-shifted, "family_identity_advantage": observed-q95}
    step_reuse = {}
    for interaction, interaction_name in enumerate(("semantic_validity", "lexical_control")):
        step_reuse[interaction_name] = {}
        for field_index, field_name in enumerate(FIELDS):
            slot = 1 if field_index == 0 else 0
            first, second = passports[interaction, 0, slot, field_index], passports[interaction, 1, slot, field_index]
            observed = float(np.mean([cosine(first[l,f], second[l,f]) for l in range(2) for f in range(8)]))
            null = np.asarray([np.mean([cosine(first[l,f], second[l,p[f]]) for l in range(2) for f in range(8)]) for p in permutations])
            q95 = float(np.quantile(null,.95)); step_reuse[interaction_name][field_name] = {"first_second_same_family": observed,
                "family_null_mean": float(np.mean(null)), "family_null_q95": q95, "family_identity_advantage": observed-q95}
    finite = np.asarray(np.load(collection["finite_odd"], mmap_mode="r"), dtype=np.float64)
    predicted = np.asarray(np.load(collection["finite_predicted"], mmap_mode="r"), dtype=np.float64)
    sem_fin, lex_fin = finite_interactions(rows, finite); sem_pred, lex_pred = finite_interactions(rows, predicted)
    def finite_summary(values: np.ndarray) -> dict:
        means = np.mean(values, axis=0); q95 = float(np.quantile(means[2:], .95))
        return {"matched": float(means[0]), "shift791": float(means[1]), "family_null_mean": float(np.mean(means[2:])),
                "family_null_q95": q95, "matched_minus_shift": float(means[0]-means[1]), "matched_minus_family_q95": float(means[0]-q95),
                "matched_rms": float(np.sqrt(np.mean(values[:,0]**2))), "matched_positive_fraction": float(np.mean(values[:,0] > 0))}
    flat_actual, flat_pred = finite.reshape(-1), predicted.reshape(-1)
    finite_result = {"semantic_actual": finite_summary(sem_fin), "lexical_actual": finite_summary(lex_fin),
                     "semantic_predicted": finite_summary(sem_pred), "lexical_predicted": finite_summary(lex_pred),
                     "row_control_linearity": {"correlation": float(np.corrcoef(flat_actual,flat_pred)[0,1]),
                         "sign_agreement": float(np.mean(np.sign(flat_actual)==np.sign(flat_pred))), "actual_zero_fraction": float(np.mean(flat_actual==0))}}
    path_diff = np.asarray(np.load(collection["path_state_difference"], mmap_mode="r"), dtype=np.float64)
    identity = {"max_target_foil_query_state_difference": float(np.max(path_diff)),
                "gradient_total_sum_max_abs": float(np.max(np.abs(np.asarray(fields[:,2,:,:,],dtype=np.float64) - (np.asarray(fields[:,0,:,:,],dtype=np.float64)+np.asarray(fields[:,1,:,:,],dtype=np.float64)))))}
    close(fields)
    return {"families": families, "passports": str(passport_path), "crosslanguage": crosslanguage, "first_second_step_reuse": step_reuse,
            "finite_two_token_total": finite_result, "exact_identities": identity}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 两token自回归路径的全坐标VJP分解与有限编译（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对fresh unit5的96条natural-direction0八族中英、三variant双角色材料，目标和foil均严格为2 token。分别在各自教师强制前缀上计算第一token与路径条件第二token的对数概率，再把两条路径对共同query-end状态的梯度相减；保存q16/q18全部2560坐标的step1、step2及两步总和gradient与$H\odot g$。另用Phase2448冻结语义方向，在q18以2% RMS对完整两token序列logprob margin作matched、+791及8个family错配的正负有限扰动，共{result['collection']['finite_forward_passes']}次前向。

$$M_{{1:2}}=\sum_{{t=1}}^2\log p(a_t\mid x,a_{{<t}})-\sum_{{t=1}}^2\log p(b_t\mid x,b_{{<t}}),$$
$$g_{{1:2}}=g_1+g_2,\qquad O_{{1:2}}(d)=\frac{{M_{{1:2}}(H+\delta d)-M_{{1:2}}(H-\delta d)}}2.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；两步及总和跨语言全坐标裁决 `{json.dumps(result['analysis']['crosslanguage'], ensure_ascii=False)}`；第一步到第二步复用 `{json.dumps(result['analysis']['first_second_step_reuse'], ensure_ascii=False)}`；完整两token有限效应 `{json.dumps(result['analysis']['finite_two_token_total'], ensure_ascii=False)}`；因果共享状态/求和恒等核对 `{json.dumps(result['analysis']['exact_identities'], ensure_ascii=False)}`；总裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2460_c42001_c42320_two_token_autoregressive_compilation.py`；96×3 step×2 qpoint×2 field×2560全坐标场、逐步logprob margin、目标/foil共同前缀状态差、十控制有限效应、交互护照和final位于同名结果目录。

**分析与理论进展。** 这一步把“最终第一token概率”推进到“第一token选择如何改变第二token条件路径”。若第二步本身保留同family中英坐标身份、并且第一/第二步同family纹理胜family错配，则候选齿轮不是一次性读出；若完整两token有限margin仍胜物理移位与八个family错配，则局部充分性延伸到了路径总分。

**问题硬伤与结论。** 教师强制仍给定正确/错误第一token前缀，不等于模型自由生成；只测2 token实体名、q16/q18与unit5。对数概率margin依赖选定target/foil，VJP仍是分析者求出的输出条件协向量，不是模型显式存储对象。即使有限总分通过，也不能证明必要性、唯一性或语言编码闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows(); directions, families = semantic_directions(); permutations = derangements(8, 8, 2460)
    collection = capture(rows, directions, families, permutations); analysis = analyze(rows, collection, families)
    sem_total = analysis["crosslanguage"]["semantic_validity"]["state_times_gradient"]["two_token_total"]
    sem_second = analysis["crosslanguage"]["semantic_validity"]["state_times_gradient"]["second_token_path_conditioned"]
    fin = analysis["finite_two_token_total"]["semantic_actual"]
    adjudication = {"second_token_semantic_coordinate_lockbox": sem_second["physical_advantage"] > 0 and sem_second["family_identity_advantage"] > 0,
                    "two_token_total_semantic_coordinate_lockbox": sem_total["physical_advantage"] > 0 and sem_total["family_identity_advantage"] > 0,
                    "two_token_finite_semantic_direction_lockbox": fin["matched"] > 0 and fin["matched_minus_shift"] > 0 and fin["matched_minus_family_q95"] > 0,
                    "free_generation_tested": False, "language_encoding_mechanism_closed": False}
    checks = {"rows_96": collection["rows"] == 96, "all_two_token": True, "full_coordinate_shape": collection["field_shape"] == [96,3,2,2,2560],
              "ten_finite_controls": len(collection["finite_controls"]) == 10, "finite_forward_passes_3840": collection["finite_forward_passes"] == 3840,
              "files": all(Path(collection[k]).exists() for k in ("fields","margins","path_state_difference","finite_odd","finite_predicted","finite_signed_margin")),
              "finite": all(math.isfinite(v) for section in analysis["finite_two_token_total"].values() for v in section.values()),
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
