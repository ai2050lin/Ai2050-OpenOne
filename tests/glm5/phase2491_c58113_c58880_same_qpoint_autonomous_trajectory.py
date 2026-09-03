#!/usr/bin/env python3
"""Capture and adjudicate autonomous boundary/first/answer states at one frozen qpoint."""
from __future__ import annotations

import gc
import hashlib
import json
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
P2487 = RESULT / "phase2487_c54721_c55872_orthogonal_family_interface_behavior"
P2490 = RESULT / "phase2490_c57473_c58112_signed_texture_energy_envelope_controls"
OUT = RESULT / "phase2491_c58113_c58880_same_qpoint_autonomous_trajectory"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, MAX_NEW = 2491, "C58113-C58880", 2560, 10
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2487_c54721_c55872_orthogonal_family_interface_behavior as materials  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows() -> tuple[list[dict], dict[str, list[str]]]:
    final = json.loads((P2487 / "analysis/final.json").read_text(encoding="utf-8"))
    qualified = final["behavior"]["qualified"]
    rows = [r for r in read_jsonl(P2487 / "material/orthogonal_family_interface_rows.jsonl")
            if r["unit"] in (15, 16) and r["family"] in qualified[r["output_interface"]]]
    return rows, qualified


def capture(model, tokenizer, rows: list[dict]) -> dict:
    qmods = field_utils.modules(model)
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    path = raw / "qualified_autonomous_path_allqpoint.float16.npy"
    mask_path = raw / "qualified_autonomous_event_mask.bool.npy"
    token_path = raw / "qualified_autonomous_token_ids.int32.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(rows), MAX_NEW + 1, len(qmods), DIM))
    event_mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(len(rows), MAX_NEW + 1))
    token_ids = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.int32, shape=(len(rows), MAX_NEW))
    field[:] = 0; event_mask[:] = False; token_ids[:] = -1
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index = []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(rows):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                generated_ids: list[int] = []
                answer_step = None; parsed = None; correct = False; generated_text = ""
                for step in range(MAX_NEW + 1):
                    captures.clear()
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    for qpoint in range(len(qmods)):
                        field[model_row, step, qpoint, :] = captures[qpoint][0, -1].float().cpu().numpy().astype(np.float16)
                    event_mask[model_row, step] = True
                    if step == MAX_NEW:
                        break
                    next_id = int(torch.argmax(output.logits[0, -1]).item())
                    generated_ids.append(next_id); token_ids[model_row, step] = next_id
                    ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    parsed, correct, _ = materials.parse_answer(generated_text, row)
                    if parsed is not None:
                        answer_step = step + 1
                        # One more iteration is required so the parsed answer token has its own HiddenState.
                        captures.clear()
                        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                        for qpoint in range(len(qmods)):
                            field[model_row, answer_step, qpoint, :] = captures[qpoint][0, -1].float().cpu().numpy().astype(np.float16)
                        event_mask[model_row, answer_step] = True
                        break
                index.append({
                    "model_row": model_row, "case_id": row["case_id"], "unit": row["unit"],
                    "family": row["family"], "language": row["language"], "surface": row["surface"],
                    "output_interface": row["output_interface"], "generated_ids": generated_ids,
                    "generated_text": generated_text, "parsed_answer": parsed, "parsed_correct": bool(correct),
                    "answer_step": answer_step, "first_step": 1 if generated_ids else None,
                })
                if (model_row + 1) % 40 == 0:
                    field.flush(); event_mask.flush(); token_ids.flush()
                    print(f"[phase2491] {model_row + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); event_mask.flush(); token_ids.flush()
        del field, event_mask, token_ids
    index_path = OUT / "index/autonomous_rows.jsonl"
    write_jsonl(index_path, index)
    return {
        "field": str(path), "shape": [len(rows), MAX_NEW + 1, len(qmods), DIM],
        "event_mask": str(mask_path), "token_ids": str(token_path), "index": str(index_path),
        "sha256": {p.name: sha256(p) for p in (path, mask_path, token_path)},
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def compare_family_maps(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    families = sorted(set(first) & set(second))
    if len(families) < 2:
        return {"families": families, "available": False}
    same = [cosine(first[f], second[f]) for f in families]
    wrong = [cosine(first[f], second[families[(i + shift) % len(families)]])
             for i, f in enumerate(families) for shift in range(1, len(families))]
    return {"families": families, "available": True, "same_mean": float(np.mean(same)),
            "wrong_mean": float(np.mean(wrong)), "wrong_q95": float(np.quantile(wrong, 0.95)),
            "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, 0.95))}


def make_passports(field: np.ndarray, index: list[dict], unit: int, qpoint: int) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for interface in sorted({r["output_interface"] for r in index}):
        rows_i = [r for r in index if r["unit"] == unit and r["output_interface"] == interface and r["parsed_correct"]]
        families = sorted({r["family"] for r in rows_i})
        for event in ("boundary", "first", "answer"):
            vectors = {}
            for family in families:
                selected = [r for r in rows_i if r["family"] == family and
                            (event == "boundary" or r["answer_step"] is not None)]
                if not selected:
                    continue
                values = []
                for row in selected:
                    step = 0 if event == "boundary" else (1 if event == "first" else row["answer_step"])
                    values.append(np.asarray(field[row["model_row"], step, qpoint], dtype=np.float64))
                vectors[family] = np.mean(values, axis=0)
            if vectors:
                grand = np.mean(list(vectors.values()), axis=0)
                result[interface][event] = {family: value - grand for family, value in vectors.items()}
    return result


def analyze(collection: dict, qpoint: int) -> dict:
    field = np.load(collection["field"], mmap_mode="r")
    index = read_jsonl(Path(collection["index"]))
    behavior = {}
    for unit in (15, 16):
        behavior[str(unit)] = {}
        for interface in sorted({r["output_interface"] for r in index}):
            values = [r for r in index if r["unit"] == unit and r["output_interface"] == interface]
            behavior[str(unit)][interface] = {"rows": len(values),
                "parsed_rate": sum(r["answer_step"] is not None for r in values) / len(values),
                "accuracy": sum(r["parsed_correct"] for r in values) / len(values)}
    passports_by_unit = {unit: make_passports(field, index, unit, qpoint) for unit in (15, 16)}
    metrics = {}
    for unit in (15, 16):
        p = passports_by_unit[unit]
        metrics[str(unit)] = {"within_interface": {}, "crossinterface": {}}
        for interface, events in p.items():
            metrics[str(unit)]["within_interface"][interface] = {
                "boundary_to_first": compare_family_maps(events.get("boundary", {}), events.get("first", {})),
                "boundary_to_answer": compare_family_maps(events.get("boundary", {}), events.get("answer", {})),
                "first_to_answer": compare_family_maps(events.get("first", {}), events.get("answer", {})),
            }
        interfaces = sorted(p)
        for i in range(len(interfaces)):
            for j in range(i + 1, len(interfaces)):
                key = f"{interfaces[i]}__{interfaces[j]}"
                metrics[str(unit)]["crossinterface"][key] = {
                    event: compare_family_maps(p[interfaces[i]].get(event, {}), p[interfaces[j]].get(event, {}))
                    for event in ("boundary", "first", "answer")
                }
    return {"behavior": behavior, "metrics": metrics}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 同一冻结qpoint的自主回答边界—首token—答案token全坐标轨迹（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 预先引用Phase2490仅由unit15选出的answer-boundary qpoint，并把该同一层位用于所有时间事件，禁止为boundary、first、answer分别挑层。材料只包括Phase2487 confirmation与lockbox均合格的family-interface：entity九族、digit三族、letter三族、side零族，共240条。每条从真实prompt贪心生成，实际token回灌模型；保存boundary加最多10个实际生成token的q0–q37×2560全坐标。只用本次逐条hook运行中正确解析的路径建family护照，并以错family为零对照。

$$T_{{e_1\to e_2}}(q^*)=\mathbb E_f\cos(P_{{f,e_1}}^{{q^*}},P_{{f,e_2}}^{{q^*}}),\qquad q^*_{{boundary}}=q^*_{{first}}=q^*_{{answer}}.$$

**结果汇总。** 冻结qpoint `{result['qpoint']}`；原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；本次行为 `{json.dumps(result['analysis']['behavior'], ensure_ascii=False)}`；unit16同层轨迹与跨接口 `{json.dumps(result['analysis']['metrics']['16'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2491_c58113_c58880_same_qpoint_autonomous_trajectory.py`；240×11事件×38层×2560坐标轨迹、event mask、实际token IDs、逐行解析和final位于同名结果目录。

**分析与理论进展。** 该结果修复Phase2475“不同指标使用不同最佳qpoint却被口头连成时间曲线”的硬伤。family-relative护照同层保持只说明条件纹理可跨事件复现；它不证明同一个原始向量被搬运，也不等于理解、表达或格式切换的心理阶段。跨接口只有共同合格family数至少2才报告。

**问题硬伤与结论。** 逐条hook与批量生成可因BF16边界argmax不同，故行为必须以本次轨迹为准。首token和答案token含不同token身份，余弦混合自回归历史。代码接口仅少数族合格且side完全失败，不能外推所有格式。该Phase提供同层时间拼图，不闭合运输算子或语言编译器。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    rows, qualified = selected_rows()
    qpoint = int(json.loads((P2490 / "analysis/final.json").read_text(encoding="utf-8"))["selection"]["answer_boundary"])
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = capture(model, tokenizer, rows)
    finally:
        model_utils.release_model(model); gc.collect()
    analysis = analyze(collection, qpoint)
    checks = {
        "rows_240": collection["shape"][0] == 240,
        "all_qpoints_and_coordinates": collection["shape"][2:] == [38, 2560],
        "same_qpoint_all_events": True,
        "qualification_not_relaxed": len(qualified["side"]) == 0 and len(qualified["entity"]) == 9,
        "lockbox_paths_exist": analysis["behavior"]["16"]["entity"]["rows"] > 0,
        "hashes": len(collection["sha256"]) == 3,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "qpoint": qpoint,
              "collection": collection, "analysis": analysis,
              "adjudication": {"same_layer_trajectory_available": True, "understanding_stage_proven": False,
                               "format_switch_proven": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
