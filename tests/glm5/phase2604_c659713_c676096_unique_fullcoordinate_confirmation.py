#!/usr/bin/env python3
"""Recapture full-coordinate fields on the unique lockbox and test split replication."""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2603 = RESULT / "phase2603_c643329_c659712_unique_natural_lockbox"
OUT = RESULT / "phase2604_c659713_c676096_unique_fullcoordinate_confirmation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2604, "C659713-C676096"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2602_c626945_c643328_natural_fullcoordinate_field as p2602  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def collect_same_batch_exemplars(model, tokenizer, material):
    """Recapture each exemplar pair in one padded batch to control BF16 shape effects."""
    selected = p2602.selected_fulltoken_cases(material)
    pairs = defaultdict(list)
    for row in material:
        if row["case_id"] in selected:
            pairs[row["pair_id"]].append(row)
    device = model.get_input_embeddings().weight.device
    raw_dir = OUT / "field/same_batch_fulltoken_exemplars"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pair_id in sorted(pairs):
        rows = sorted(pairs[pair_id], key=lambda row: row["variant"])
        width = max(len(row["prompt_ids"]) for row in rows)
        ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row["prompt_ids"])] = torch.tensor(row["prompt_ids"], device=device)
            mask[index, :len(row["prompt_ids"])] = 1
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                           use_cache=False, return_dict=True)
        for index, row in enumerate(rows):
            length = len(row["prompt_ids"])
            stack = torch.stack([state[index, :length] for state in output.hidden_states], dim=0)
            path = raw_dir / f"{row['case_id']}.float16.npy"
            np.save(path, stack.detach().cpu().to(torch.float16).numpy(), allow_pickle=False)
            manifest.append({"case_id": row["case_id"], "pair_id": pair_id,
                             "family": row["family"], "language": row["language"],
                             "split": row["split"], "variant": row["variant"],
                             "shape": list(stack.shape),
                             "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                             "bytes": path.stat().st_size, "same_batch_pair_control": True})
    old_dir = OUT / "field/fulltoken_exemplars"
    if old_dir.is_dir() and OUT.resolve() in old_dir.resolve().parents:
        shutil.rmtree(old_dir)
    return manifest


def build_fields(material, generated, boundary_path, logit_path, manifest):
    boundary = np.load(boundary_path, mmap_mode="r")
    logit = np.load(logit_path, mmap_mode="r")
    gidx = {row["case_id"]: row for row in generated}
    rows_by_pair = defaultdict(list)
    for index, row in enumerate(material):
        rows_by_pair[row["pair_id"]].append((index, row))
    pair_ids = sorted(rows_by_pair)
    delta_path = OUT / "field/answer_pair_delta_unique600.float16.npy"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta = np.lib.format.open_memmap(
        delta_path, mode="w+", dtype=np.float16,
        shape=(len(pair_ids), boundary.shape[1], boundary.shape[2]))
    pairs = []
    for pair_index, pair_id in enumerate(pair_ids):
        pair = sorted(rows_by_pair[pair_id], key=lambda item: item[1]["variant"])
        i0, r0 = pair[0]
        i1, r1 = pair[1]
        delta[pair_index] = boundary[i1].astype(np.float32) - boundary[i0].astype(np.float32)
        pairs.append({
            "pair_index": pair_index, "pair_id": pair_id, "family": r0["family"],
            "language": r0["language"], "split": r0["split"],
            "target0": r0["target"], "target1": r1["target"],
            "both_greedy_correct": bool(gidx[r0["case_id"]]["parsed_correct"] and gidx[r1["case_id"]]["parsed_correct"]),
            "both_logit_lens_correct_final": bool(logit[i0, -1, 0] > logit[i0, -1, 1] and
                                                   logit[i1, -1, 0] > logit[i1, -1, 1]),
        })
    delta.flush()
    d = np.load(delta_path, mmap_mode="r")
    groups = sorted({f"{row['family']}/{row['language']}" for row in pairs})
    split_reproduction, by_group = {}, {}
    for group in groups:
        family, language = group.split("/")
        idx = [row["pair_index"] for row in pairs if row["family"] == family and row["language"] == language]
        split_idx = {
            split: [row["pair_index"] for row in pairs if row["family"] == family and
                    row["language"] == language and row["split"] == split]
            for split in ("discovery", "confirmation", "external")
        }
        dc, de, rolled = [], [], []
        for layer in range(d.shape[1]):
            mean_d = d[split_idx["discovery"], layer].astype(np.float32).mean(0)
            mean_c = d[split_idx["confirmation"], layer].astype(np.float32).mean(0)
            mean_e = d[split_idx["external"], layer].astype(np.float32).mean(0)
            dc.append(pearson(mean_d, mean_c))
            de.append(pearson(mean_d, mean_e))
            rolled.append(pearson(mean_d, np.roll(mean_c, 641)))
        rms = np.sqrt(np.mean(d[idx].astype(np.float64) ** 2, axis=2))
        split_reproduction[group] = {
            "discovery_confirmation": dc,
            "discovery_external": de,
            "discovery_confirmation_roll641": rolled,
        }
        by_group[group] = {
            "pairs": len(idx),
            "both_greedy_correct": sum(row["both_greedy_correct"] for row in pairs
                                        if row["family"] == family and row["language"] == language),
            "median_answer_delta_rms": np.median(rms, axis=0).tolist(),
            "late_dc": float(np.mean(dc[25:36])),
            "late_de": float(np.mean(de[25:36])),
            "late_roll": float(np.mean(rolled[25:36])),
        }
    raw_onsets = []
    for pair_id in sorted({row["pair_id"] for row in manifest}):
        mrows = sorted([row for row in manifest if row["pair_id"] == pair_id], key=lambda row: row["variant"])
        h0 = np.load(ROOT / mrows[0]["path"], mmap_mode="r").astype(np.float32)
        h1 = np.load(ROOT / mrows[1]["path"], mmap_mode="r").astype(np.float32)
        c0 = next(row for row in material if row["case_id"] == mrows[0]["case_id"])
        c1 = next(row for row in material if row["case_id"] == mrows[1]["case_id"])
        prefix = 0
        while prefix < min(len(c0["prompt_ids"]), len(c1["prompt_ids"])) and c0["prompt_ids"][prefix] == c1["prompt_ids"][prefix]:
            prefix += 1
        prefix_max = float(np.max(np.abs(h1[:, :prefix] - h0[:, :prefix]))) if prefix else 0.0
        answer_delta = h1[:, -1] - h0[:, -1]
        answer_rms = np.sqrt(np.mean(answer_delta.astype(np.float64) ** 2, axis=1))
        raw_onsets.append({
            "pair_id": pair_id, "family": mrows[0]["family"], "language": mrows[0]["language"],
            "split": mrows[0]["split"], "causal_prefix_absmax": prefix_max,
            "answer_embedding_rms": float(answer_rms[0]), "answer_block0_rms": float(answer_rms[1]),
            "answer_final_rms": float(answer_rms[-1]),
            "first_nonzero_answer_hidden": next((int(i) for i, value in enumerate(answer_rms) if value > 1e-7), None),
            "answer_curve": answer_rms.tolist(),
        })
    return delta_path, pairs, split_reproduction, by_group, raw_onsets


def append_memo(result):
    heading = f"## Phase {PHASE}: 无重复自然锁箱全坐标发现—确认—外测图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 在Phase2603无重复材料上重新采集全部答案场和代表全token场，旧Phase2602的split相关不复用。每族/语言用发现20 pair形成均值方向，与文本不重叠确认20和外测10比较；物理坐标roll641作为错坐标null：

$$\rho^{{D,C}}_l=\operatorname{{corr}}_d(\bar D^D_l,\bar D^C_l),\qquad
\rho^{{D,C_{{roll}}}}_l=\operatorname{{corr}}_d(\bar D^D_l,\operatorname{{roll}}_{{641}}\bar D^C_l).$$

**测试用例。** 1200×37×2560答案边界、600×37×2560 pair差分；12族/语言×3 split×2变体=72条全token场。行为正确、错误、高低margin全部保留；Qwen3-4B BF16 CUDA非量化，无Top-K/PCA。

**结果汇总。** 组级=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；合格/不合格=`{json.dumps(result['qualified_vs_unqualified'], ensure_ascii=False)}`；代表pair出生=`{json.dumps(result['token_onsets_summary'], ensure_ascii=False)}`；原场=`{json.dumps(result['field'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2604_c659713_c676096_unique_fullcoordinate_confirmation.py`；答案场、pair场、logit lens、72条全token场、逐层split/null和final位于`{OUT}`。

**分析与理论进展。** 真实物理坐标的发现—确认/外测相关高于roll null，才说明外部操作产生可重复候选方向；行为未通过组即使方向复现，也只能解释为稳定词面处理。source前和答案embedding零区验证因果掩码，但block0首个非零不等于必要计算位点。

**问题硬伤。** pair方向混合source词、答案身份和操作；核心模板有限；均值相关不是单prompt预测；float16早层精度有限；roll只是一类null。下一Phase必须直接patch单个recipient。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    p2603_final = load_json(P2603 / "analysis/final.json")
    material = read_jsonl(P2603 / "material/cases.unique.jsonl")
    generated = read_jsonl(P2603 / "behavior/greedy_generation.jsonl")
    model = tokenizer = None
    old_out = p2602.OUT
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        p2602.OUT = OUT
        boundary_path, logit_path, _old_manifest = p2602.collect(model, tokenizer, material)
        manifest = collect_same_batch_exemplars(model, tokenizer, material)
    finally:
        p2602.OUT = old_out
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    delta_path, pairs, split_repro, by_group, raw_onsets = build_fields(
        material, generated, boundary_path, logit_path, manifest)
    save_json(OUT / "analysis/pair_index.json", pairs)
    save_json(OUT / "analysis/split_reproduction.json", split_repro)
    save_json(OUT / "analysis/fulltoken_onsets.json", raw_onsets)
    manifest_path = OUT / "field/fulltoken_manifest.json"
    save_json(manifest_path, manifest)
    qualified = set(p2603_final["qualified_groups"])
    q = np.asarray([row["late_dc"] for group, row in by_group.items() if group in qualified])
    u = np.asarray([row["late_dc"] for group, row in by_group.items() if group not in qualified])
    q_roll = np.asarray([row["late_roll"] for group, row in by_group.items() if group in qualified])
    summary = {
        "qualified_groups": len(q), "unqualified_groups": len(u),
        "qualified_late_dc_median": float(np.median(q)),
        "unqualified_late_dc_median": float(np.median(u)),
        "qualified_late_roll_median": float(np.median(q_roll)),
        "qualified_physical_advantage": float(np.median(q - q_roll)),
    }
    onset_summary = {
        "pairs": len(raw_onsets), "causal_prefix_absmax": max(row["causal_prefix_absmax"] for row in raw_onsets),
        "answer_embedding_rms_max": max(row["answer_embedding_rms"] for row in raw_onsets),
        "first_nonzero_counts": {str(value): sum(row["first_nonzero_answer_hidden"] == value for row in raw_onsets)
                                  for value in sorted({row["first_nonzero_answer_hidden"] for row in raw_onsets})},
        "median_block0_rms": float(np.median([row["answer_block0_rms"] for row in raw_onsets])),
        "median_final_rms": float(np.median([row["answer_final_rms"] for row in raw_onsets])),
    }
    field = {
        "answer_shape": list(np.load(boundary_path, mmap_mode="r").shape),
        "delta_shape": list(np.load(delta_path, mmap_mode="r").shape),
        "logit_shape": list(np.load(logit_path, mmap_mode="r").shape),
        "fulltoken_prompts": len(manifest), "fulltoken_bytes": sum(row["bytes"] for row in manifest),
        "all_coordinates": True, "no_topk": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized", "field": field,
        "by_family_language": by_group, "qualified_vs_unqualified": summary,
        "token_onsets_summary": onset_summary,
        "claim_boundary": "text-disjoint full-coordinate replication candidate; no causal or single-prompt gear claim",
        "hashes": {"boundary": sha256(boundary_path), "delta": sha256(delta_path),
                   "logit": sha256(logit_path), "manifest": sha256(manifest_path)},
        "language_mechanism_closed": False,
    }
    result["checks"] = {
        "phase2603_complete": p2603_final["all_checks_passed"],
        "all_1200_answer_fields": field["answer_shape"] == [1200, 37, 2560],
        "all_600_pair_deltas": field["delta_shape"] == [600, 37, 2560],
        "all_72_fulltoken_prompts": len(manifest) == 72,
        "causal_prefix_zero": onset_summary["causal_prefix_absmax"] == 0.0,
        "answer_embedding_zero": onset_summary["answer_embedding_rms_max"] == 0.0,
        "all_coordinates_no_topk": True,
        "qualified_and_unqualified_reported": len(q) > 0 and len(u) > 0,
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    correction = "**Phase2604因果前区索引勘误（append-only）**"
    memo_text = MEMO.read_text(encoding="utf-8-sig")
    if result["all_checks_passed"] and correction not in memo_text and '"causal_prefix_zero": false' in memo_text:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"\n\n{correction} [{stamp}] 初次full-token审计用SequenceMatcher的替换span起点界定前区；"
                "prompt内重复token会让全局对齐选择更早的等价匹配，因而把真正变化token误纳入前区并得到absmax=416。"
                "现以两侧实际token ID的最长公共前缀严格定义因果前区，重算为0；原始HiddenState、答案embedding=0、"
                f"block0/晚层曲线和split结果均未改变。修正检查=`{json.dumps(result['checks'], ensure_ascii=False)}`。\n"
            )
    print(json.dumps({key: result[key] for key in (
        "phase", "field", "qualified_vs_unqualified", "token_onsets_summary", "checks", "all_checks_passed")},
        ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
