#!/usr/bin/env python3
"""Calibrate normalized natural/random directions across doses and FP16/BF16."""
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
P2330 = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
OUT = RESULT / "phase2332_c6361_c6480_multidose_natural_direction_calibration"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2330 / "material/typed_language_family_atlas.jsonl"
PHASE = 2332
CAMPAIGN = "C6361-C6480"
FORMATS = (("float16", torch.float16), ("bfloat16", torch.bfloat16))
DOSES = (0.02, 0.01, 0.005, 0.0025)
SOURCES = (10, 20, 30)
TARGETS = {10: (11, 14, 37), 20: (21, 24, 37), 30: (31, 34, 37)}
PROBES = ("natural_state", "random_control", "natural_plus_random")
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def panel(rows: list[dict]) -> list[tuple[dict, dict]]:
    output = []
    for family_index, family in enumerate(contract.FAMILIES):
        language = "en" if family_index % 2 == 0 else "zh"
        pair = [row for row in rows if row["family"] == family and row["language"] == language
                and row["surface"] == "narrative" and int(row["unit"]) == 0]
        pair.sort(key=lambda row: row["state"])
        if len(pair) != 2 or [row["state"] for row in pair] != [0, 1]:
            raise RuntimeError(("panel_pair", family, language, len(pair)))
        output.append((pair[0], pair[1]))
    return output


def random_direction(family: str, source: int, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(f"phase2332|{family}|q{source}|random".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    value = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
    value /= np.linalg.norm(value.astype(np.float64))
    return value.astype(np.float32)


def baseline_states(model, device, module_list: list[Any], row0: dict, row1: dict, source: int) -> tuple[np.ndarray, np.ndarray]:
    capture: list[torch.Tensor] = []

    def hook(_module, _inputs, value):
        capture.append(value[0] if isinstance(value, tuple) else value)

    handle = module_list[source].register_forward_hook(hook)
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        ids, mask, positions = baseline.pad_right([row0["future_prompt_ids"], row1["future_prompt_ids"]], device, pad)
        with torch.inference_mode():
            model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
        tensor = capture[0]
        ends = mask.sum(dim=1) - 1
        values = [tensor[index, ends[index]].float().cpu().numpy() for index in range(2)]
        return values[0], values[1]
    finally:
        handle.remove()


def run_cell(model, device, row0: dict, row1: dict, source: int) -> tuple[np.ndarray, np.ndarray, dict]:
    module_list = modules(model)
    base0, base1 = baseline_states(model, device, module_list, row0, row1, source)
    natural = base1.astype(np.float64) - base0.astype(np.float64)
    natural_norm = float(np.linalg.norm(natural))
    if natural_norm <= EPS:
        raise RuntimeError(("zero_natural_direction", row0["family"], source))
    natural = (natural / natural_norm).astype(np.float32)
    random = random_direction(row0["family"], source, natural.size)
    directions = np.stack([natural, random, natural + random]).astype(np.float32)
    direction_norms = np.linalg.norm(directions.astype(np.float64), axis=1)
    source_norm = float(np.linalg.norm(base0.astype(np.float64)))
    variants = [(None, None, 0)]
    for probe in range(len(PROBES)):
        for dose_index, dose in enumerate(DOSES):
            variants.extend(((probe, dose_index, 1), (probe, dose_index, -1)))
    prompt = row0["future_prompt_ids"]
    ids = torch.tensor([prompt] * len(variants), dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    position_ids = torch.arange(len(prompt), device=device)[None].repeat(len(variants), 1)
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def source_hook(_module, _inputs, value):
        tensor = value[0] if isinstance(value, tuple) else value
        changed = tensor.clone()
        for index, (probe, dose_index, sign) in enumerate(variants):
            if probe is None:
                continue
            direction = torch.tensor(directions[probe], dtype=tensor.dtype, device=tensor.device)
            changed[index, -1] = changed[index, -1] + direction * (sign * DOSES[dose_index] * source_norm)
        return (changed, *value[1:]) if isinstance(value, tuple) else changed

    handles.append(module_list[source].register_forward_hook(source_hook))
    for target in TARGETS[source]:
        def target_hook(_module, _inputs, value, target=target):
            captures[target] = value[0] if isinstance(value, tuple) else value
        handles.append(module_list[target].register_forward_hook(target_hook))
    derivative = np.empty((len(PROBES), len(DOSES), 3, natural.size), dtype=np.float32)
    even = np.empty_like(derivative)
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        for target_index, target in enumerate(TARGETS[source]):
            base = captures[target][0, -1].float().cpu().numpy()
            for probe in range(len(PROBES)):
                for dose_index, dose in enumerate(DOSES):
                    offset = 1 + (probe * len(DOSES) + dose_index) * 2
                    plus = captures[target][offset, -1].float().cpu().numpy()
                    minus = captures[target][offset + 1, -1].float().cpu().numpy()
                    derivative[probe, dose_index, target_index] = (plus - minus) / (2.0 * dose * source_norm)
                    even[probe, dose_index, target_index] = (plus + minus) * 0.5 - base
    finally:
        for handle in handles:
            handle.remove()
    return derivative, even, {
        "family": row0["family"], "language": row0["language"], "source_q": source,
        "state0_case": row0["case_id"], "state1_case": row1["case_id"],
        "source_norm": source_norm, "natural_delta_norm": natural_norm,
        "direction_norms": [float(value) for value in direction_norms],
    }


def collect_format(name: str, dtype: torch.dtype, pairs: list[tuple[dict, dict]]) -> dict:
    worker = OUT / name
    final_path = worker / "analysis/collection.json"
    derivative_path = worker / "raw/directional_derivative.float32.npy"
    even_path = worker / "raw/even_response.float32.npy"
    progress_path = worker / "raw/progress.json"
    index_path = worker / "index/cells.jsonl"
    shape = (len(pairs), len(SOURCES), len(PROBES), len(DOSES), 3, 2560)
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    derivative_path.parent.mkdir(parents=True, exist_ok=True)
    if derivative_path.exists() and even_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed_cells"])
        derivative = np.lib.format.open_memmap(derivative_path, mode="r+")
        even = np.lib.format.open_memmap(even_path, mode="r+")
        index = read_rows(index_path)
    else:
        completed, index = 0, []
        derivative = np.lib.format.open_memmap(derivative_path, mode="w+", dtype=np.float32, shape=shape)
        even = np.lib.format.open_memmap(even_path, mode="w+", dtype=np.float32, shape=shape)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=dtype, use_8bit=False)
        for cell in range(completed, len(pairs) * len(SOURCES)):
            pair_index, source_index = divmod(cell, len(SOURCES))
            row0, row1 = pairs[pair_index]
            d, e, ledger = run_cell(model, device, row0, row1, SOURCES[source_index])
            derivative[pair_index, source_index] = d
            even[pair_index, source_index] = e
            index.append({"pair_index": pair_index, "source_index": source_index, **ledger})
            derivative.flush(); even.flush(); write_rows(index_path, index)
            save(progress_path, {"completed_cells": cell + 1, "shape": list(shape)})
            print(f"[phase2332 {name}] {cell + 1}/{len(pairs) * len(SOURCES)}", flush=True)
        dtypes = defaultdict(int)
        for parameter in model.parameters():
            dtypes[str(parameter.dtype).replace("torch.", "")] += int(parameter.numel())
        model_record = {"format": name, "parameter_dtypes": dict(dtypes), "device": str(device)}
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        derivative.flush(); even.flush(); close_memmap(derivative); close_memmap(even)
    result = {
        "format": name, "shape": list(shape), "doses": list(DOSES), "probes": list(PROBES),
        "sources": list(SOURCES), "targets": {str(k): list(v) for k, v in TARGETS.items()},
        "model": model_record, "derivative_path": str(derivative_path.relative_to(ROOT)),
        "even_path": str(even_path.relative_to(ROOT)),
        "hashes": {"derivative": file_hash(derivative_path), "even": file_hash(even_path)},
    }
    save(final_path, result)
    return result


def relative_mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.square(left - right, dtype=np.float64)) /
                 (np.sum(np.square(left, dtype=np.float64)) + EPS))


def analyze(collections: dict[str, dict], pairs: list[tuple[dict, dict]]) -> dict:
    arrays = {name: np.load(ROOT / row["derivative_path"], mmap_mode="r") for name, row in collections.items()}
    evens = {name: np.load(ROOT / row["even_path"], mmap_mode="r") for name, row in collections.items()}
    source_norms = {}
    for name in collections:
        for row in read_rows(OUT / name / "index/cells.jsonl"):
            source_norms[(name, int(row["pair_index"]), int(row["source_index"]))] = float(row["source_norm"])
    records = []
    for name in arrays:
        d, e = arrays[name], evens[name]
        for pair_index, (row0, _row1) in enumerate(pairs):
            for source_index, source in enumerate(SOURCES):
                for probe in range(2):
                    for target_index, target in enumerate(TARGETS[source]):
                        for dose_index in range(len(DOSES) - 1):
                            larger = d[pair_index, source_index, probe, dose_index, target_index].astype(np.float64)
                            smaller = d[pair_index, source_index, probe, dose_index + 1, target_index].astype(np.float64)
                            odd_effect = smaller * (DOSES[dose_index + 1] * source_norms[(name, pair_index, source_index)])
                            records.append({
                                "format": name, "family": row0["family"], "source_q": source, "target_q": target,
                                "probe": PROBES[probe], "larger_dose": DOSES[dose_index],
                                "smaller_dose": DOSES[dose_index + 1],
                                "convergence_relative_mse": relative_mse(smaller, larger),
                                "sign_agreement": float(np.mean(smaller * larger > 0)),
                                "smaller_nonzero_fraction": float(np.mean(smaller != 0)),
                                "even_to_odd_l2": float(np.linalg.norm(e[pair_index, source_index, probe, dose_index + 1, target_index].astype(np.float64)) /
                                                        (np.linalg.norm(odd_effect) + EPS)),
                            })
                for dose_index, dose in enumerate(DOSES):
                    actual = d[pair_index, source_index, 2, dose_index].astype(np.float64)
                    predicted = (d[pair_index, source_index, 0, dose_index].astype(np.float64) +
                                 d[pair_index, source_index, 1, dose_index].astype(np.float64))
                    for target_index, target in enumerate(TARGETS[source]):
                        records.append({
                            "format": name, "family": row0["family"], "source_q": source, "target_q": target,
                            "probe": "pair_superposition", "dose": dose,
                            "pair_relative_mse": relative_mse(actual[target_index], predicted[target_index]),
                        })
    cross = []
    fp16, bf16 = arrays["float16"], arrays["bfloat16"]
    for pair_index, (row0, _row1) in enumerate(pairs):
        for source_index, source in enumerate(SOURCES):
            for probe in range(2):
                for dose_index, dose in enumerate(DOSES):
                    for target_index, target in enumerate(TARGETS[source]):
                        a = fp16[pair_index, source_index, probe, dose_index, target_index].astype(np.float64)
                        b = bf16[pair_index, source_index, probe, dose_index, target_index].astype(np.float64)
                        cross.append({
                            "family": row0["family"], "source_q": source, "target_q": target,
                            "probe": PROBES[probe], "dose": dose,
                            "symmetric_relative_mse": float(np.sum(np.square(a - b)) /
                                                            ((np.sum(np.square(a)) + np.sum(np.square(b))) / 2 + EPS)),
                            "sign_agreement": float(np.mean(a * b > 0)),
                        })
    write_rows(OUT / "analysis/multidose_records.jsonl", records)
    write_rows(OUT / "analysis/cross_format_records.jsonl", cross)
    summary = {}
    for name in arrays:
        conv = [row for row in records if row["format"] == name and row["probe"] != "pair_superposition"]
        pair_rows = [row for row in records if row["format"] == name and row["probe"] == "pair_superposition"]
        summary[name] = {
            "median_convergence_relative_mse": float(np.median([row["convergence_relative_mse"] for row in conv])),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in conv])),
            "median_nonzero_fraction": float(np.median([row["smaller_nonzero_fraction"] for row in conv])),
            "median_even_to_odd_l2": float(np.median([row["even_to_odd_l2"] for row in conv])),
            "median_pair_superposition_relative_mse": float(np.median([row["pair_relative_mse"] for row in pair_rows])),
            "smallest_step": {
                "relative_mse": float(np.median([row["convergence_relative_mse"] for row in conv if row["smaller_dose"] == 0.0025])),
                "sign_agreement": float(np.median([row["sign_agreement"] for row in conv if row["smaller_dose"] == 0.0025])),
                "nonzero_fraction": float(np.median([row["smaller_nonzero_fraction"] for row in conv if row["smaller_dose"] == 0.0025])),
            },
        }
    cross_summary = {
        "median_symmetric_relative_mse": float(np.median([row["symmetric_relative_mse"] for row in cross])),
        "median_sign_agreement": float(np.median([row["sign_agreement"] for row in cross])),
        "by_dose": {
            str(dose): {
                "median_symmetric_relative_mse": float(np.median([row["symmetric_relative_mse"] for row in cross if row["dose"] == dose])),
                "median_sign_agreement": float(np.median([row["sign_agreement"] for row in cross if row["dose"] == dose])),
            } for dose in DOSES
        },
    }
    for value in list(arrays.values()) + list(evens.values()):
        close_memmap(value)
    return {"formats": summary, "cross_format": cross_summary, "record_count": len(records), "cross_record_count": len(cross)}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 自然语言方向与随机对照的FP16/BF16四剂量校准（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从二十族各取一个 discovery 双状态配对，十族英文、十族中文；在 q10/q20/q30 提取 `state1-state0` 并单位化为自然语言状态方向，同时生成单位化 Rademacher 错方向和未归一化二者和方向。Qwen3-4B 分别以非量化 FP16、BF16 顺序加载释放，在 `0.02/0.01/0.005/0.0025` 四个相对源状态范数剂量做中心差分，读取 `q+1/q+4/final` 的全部2560坐标。自然方向是完整提示差分，仍混合语言操作、词序、tokenizer和边界状态，不被命名为纯语义切向量。

$$
u_{{nat}}=rac{{H_q(x_1)-H_q(x_0)}}{{\lVert H_q(x_1)-H_q(x_0)\rVert_2}},qquad
D_\delta(u)=rac{{H_t(h+\delta\lVert h\rVert u)-H_t(h-\delta\lVert h\rVert u)}}{{2\delta\lVert h\rVert}}.
$$

$$
E_{{conv}}=\frac{{\lVert D_{{\delta/2}}-D_\delta\rVert_2^2}}{{\lVert D_{{\delta/2}}\rVert_2^2+\varepsilon}},\qquad
E_{{pair}}=\frac{{\lVert D(u+v)-D(u)-D(v)\rVert_2^2}}{{\lVert D(u+v)\rVert_2^2+\varepsilon}}.
$$

**结果汇总、相关文件与门槛。** 校准汇总 `{json.dumps(result['analysis'], ensure_ascii=False)}`；采集 `{json.dumps(result['collections'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2332_c6361_c6480_multidose_natural_direction_calibration.py`；结果 `tests/glm5/result/phase2332_c6361_c6480_multidose_natural_direction_calibration`。16.3GB GPU不能容纳Qwen3-4B完整FP32权重加激活，因此没有伪装成FP32 CUDA结论；本期实际回答FP16/BF16及剂量收敛。

**分析、理论进展、问题硬伤与结论。** 多剂量收敛只检验当前有限区间，不等于无穷小Jacobian。FP16/BF16同时改变权重、前向舍入和激活，因此跨格式差异只能叫数值格式敏感性。若最小剂量非零率下降或相邻剂量MSE上升，说明已经触及舍入噪声底；后续图谱应选择收敛更好且非零率足够的格式/剂量，不把更小剂量自动视为更真实。成对和方向保持 `u+v` 原尺度，使叠加公式尺度闭合。本期不证明自然方向是语义齿轮，也不证明流形联络。

**下一阶段。** 目标仍相同，自动继续读取 Phase2331 的 discovery/confirmation 自然状态场，制作二十族全坐标复用、表面/语言差异、层距离与行为正确性护照，冻结候选后才读取 fresh 分区。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2330 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2330 contract invalid")
    pairs = panel(read_rows(MATERIAL))
    write_rows(OUT / "material/calibration_pairs.jsonl", [row for pair in pairs for row in pair])
    collections = {name: collect_format(name, dtype, pairs) for name, dtype in FORMATS}
    analysis = analyze(collections, pairs)
    checks = {
        "twenty_family_pairs": len(pairs) == 20,
        "two_formats": set(collections) == {"float16", "bfloat16"},
        "all_coordinates": all(row["shape"][-1] == 2560 for row in collections.values()),
        "four_doses": all(row["doses"] == list(DOSES) for row in collections.values()),
        "normalized_base_directions": True,
        "pair_sum_scale_preserved": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "collections": collections,
        "analysis": analysis, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2332_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
