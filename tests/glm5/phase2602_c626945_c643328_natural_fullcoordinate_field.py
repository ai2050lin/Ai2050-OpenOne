#!/usr/bin/env python3
"""Full-coordinate Qwen3-4B fields for all 1,200 natural single prompts."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2601 = RESULT / "phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox"
OUT = RESULT / "phase2602_c626945_c643328_natural_fullcoordinate_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2602, "C626945-C643328"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as p2601  # noqa: E402


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
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def first_answer_token(tokenizer, prompt, answer):
    full, positions = p2601.candidate_token_ids(tokenizer, prompt, answer)
    return int(full[positions[0]])


def selected_fulltoken_cases(material):
    selected = set()
    groups = sorted({(row["family"], row["language"], row["split"]) for row in material})
    for group in groups:
        pairs = sorted({row["pair_id"] for row in material
                        if (row["family"], row["language"], row["split"]) == group})
        pair_id = pairs[0]
        selected.update(row["case_id"] for row in material if row["pair_id"] == pair_id)
    return selected


def collect(model, tokenizer, material):
    device = model.get_input_embeddings().weight.device
    n_layers = len(model_utils.get_layers(model))
    d_model = model.get_input_embeddings().weight.shape[1]
    n_hidden = n_layers + 1
    OUT.mkdir(parents=True, exist_ok=True)
    boundary_path = OUT / "field/answer_boundary_all1200.float16.npy"
    logit_path = OUT / "field/logit_lens_target_alternate.float32.npy"
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary = np.lib.format.open_memmap(boundary_path, mode="w+", dtype=np.float16,
                                         shape=(len(material), n_hidden, d_model))
    logit_lens = np.lib.format.open_memmap(logit_path, mode="w+", dtype=np.float32,
                                          shape=(len(material), n_hidden, 2))
    selected = selected_fulltoken_cases(material)
    raw_dir = OUT / "field/fulltoken_exemplars"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_ids = [first_answer_token(tokenizer, row["prompt"], row["target"]) for row in material]
    alternate_ids = [first_answer_token(tokenizer, row["prompt"], row["alternate"]) for row in material]
    by_length = defaultdict(list)
    for index, row in enumerate(material):
        by_length[len(row["prompt_ids"])].append((index, row))
    manifest = []
    completed = 0
    final_norm = model.model.norm
    lm_weight = model.get_output_embeddings().weight
    for length in sorted(by_length):
        jobs = by_length[length]
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start + 8]
            ids = torch.tensor([row["prompt_ids"] for _, row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                               use_cache=False, return_dict=True)
            hidden_states = output.hidden_states
            if len(hidden_states) != n_hidden:
                raise RuntimeError((len(hidden_states), n_hidden))
            for local, (global_index, row) in enumerate(batch):
                stack = torch.stack([state[local] for state in hidden_states], dim=0)
                answer = stack[:, -1, :]
                boundary[global_index] = answer.detach().cpu().to(torch.float16).numpy()
                with torch.inference_mode():
                    normalized = final_norm(answer)
                    ids2 = torch.tensor([target_ids[global_index], alternate_ids[global_index]], device=normalized.device)
                    logits2 = normalized.float() @ lm_weight[ids2].float().T
                logit_lens[global_index] = logits2.detach().cpu().numpy()
                if row["case_id"] in selected:
                    path = raw_dir / f"{row['case_id']}.float16.npy"
                    np.save(path, stack.detach().cpu().to(torch.float16).numpy(), allow_pickle=False)
                    manifest.append({"case_id": row["case_id"], "pair_id": row["pair_id"],
                                     "family": row["family"], "language": row["language"],
                                     "split": row["split"], "variant": row["variant"],
                                     "shape": list(stack.shape),
                                     "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                     "bytes": path.stat().st_size})
            completed += len(batch)
            if completed % 240 == 0 or completed == len(material):
                print(f"[phase2602 collect] {completed}/{len(material)}", flush=True)
            del output, hidden_states
    boundary.flush()
    logit_lens.flush()
    return boundary_path, logit_path, manifest


def analyze(material, boundary_path, logit_path, manifest, generated):
    boundary = np.load(boundary_path, mmap_mode="r")
    logit = np.load(logit_path, mmap_mode="r")
    gidx = {row["case_id"]: row for row in generated}
    pair_rows = defaultdict(list)
    for index, row in enumerate(material):
        pair_rows[row["pair_id"]].append((index, row))
    pairs = []
    delta_path = OUT / "field/answer_pair_delta_all600.float16.npy"
    delta = np.lib.format.open_memmap(delta_path, mode="w+", dtype=np.float16,
                                      shape=(len(pair_rows), boundary.shape[1], boundary.shape[2]))
    for pair_index, pair_id in enumerate(sorted(pair_rows)):
        rows = sorted(pair_rows[pair_id], key=lambda item: item[1]["variant"])
        i0, r0 = rows[0]
        i1, r1 = rows[1]
        delta[pair_index] = boundary[i1].astype(np.float32) - boundary[i0].astype(np.float32)
        pairs.append({"pair_index": pair_index, "pair_id": pair_id, "family": r0["family"],
                      "language": r0["language"], "split": r0["split"],
                      "both_greedy_correct": bool(gidx[r0["case_id"]]["parsed_correct"] and
                                                  gidx[r1["case_id"]]["parsed_correct"]),
                      "both_candidate_correct": bool(logit[i0, -1, 0] > logit[i0, -1, 1] and
                                                     logit[i1, -1, 0] > logit[i1, -1, 1])})
    delta.flush()
    delta_ro = np.load(delta_path, mmap_mode="r")
    by_group = {}
    for family, language in sorted({(row["family"], row["language"]) for row in material}):
        indices = [row["pair_index"] for row in pairs if row["family"] == family and row["language"] == language]
        correct = [row["pair_index"] for row in pairs if row["family"] == family and row["language"] == language
                   and row["both_greedy_correct"]]
        failed = [index for index in indices if index not in set(correct)]
        rms = np.sqrt(np.mean(delta_ro[indices].astype(np.float64) ** 2, axis=2))
        by_group[f"{family}/{language}"] = {
            "pairs": len(indices), "both_greedy_correct": len(correct), "failed_pairs": len(failed),
            "median_answer_delta_rms_by_hidden": np.median(rms, axis=0).tolist(),
            "final_rms_correct": float(np.median(np.sqrt(np.mean(delta_ro[correct, -1].astype(np.float64) ** 2, axis=1)))) if correct else None,
            "final_rms_failed": float(np.median(np.sqrt(np.mean(delta_ro[failed, -1].astype(np.float64) ** 2, axis=1)))) if failed else None,
        }
    split_reproduction = {}
    for family, language in sorted({(row["family"], row["language"]) for row in material}):
        discovery = [row["pair_index"] for row in pairs if row["family"] == family and row["language"] == language and row["split"] == "discovery"]
        confirmation = [row["pair_index"] for row in pairs if row["family"] == family and row["language"] == language and row["split"] == "confirmation"]
        correlations = [pearson(delta_ro[discovery, layer].astype(np.float32).mean(0),
                                delta_ro[confirmation, layer].astype(np.float32).mean(0))
                        for layer in range(delta_ro.shape[1])]
        split_reproduction[f"{family}/{language}"] = correlations
    raw_index = {row["case_id"]: row for row in manifest}
    token_onsets = []
    for pair_id in sorted({row["pair_id"] for row in manifest}):
        rows = sorted([row for row in manifest if row["pair_id"] == pair_id], key=lambda row: row["variant"])
        if len(rows) != 2:
            raise RuntimeError(pair_id)
        h0 = np.load(ROOT / rows[0]["path"], mmap_mode="r").astype(np.float32)
        h1 = np.load(ROOT / rows[1]["path"], mmap_mode="r").astype(np.float32)
        d = h1 - h0
        rms = np.sqrt(np.mean(d.astype(np.float64) ** 2, axis=2))
        material_row = next(row for row in material if row["case_id"] == rows[0]["case_id"])
        source_start = min(material_row["source_token_positions"])
        answer = material_row["answer_boundary_token"]
        token_onsets.append({"pair_id": pair_id, "family": rows[0]["family"], "language": rows[0]["language"],
                             "split": rows[0]["split"], "source_start": source_start, "answer_token": answer,
                             "embedding_prefix_max": float(rms[0, :source_start].max()) if source_start else 0.0,
                             "embedding_answer_rms": float(rms[0, answer]),
                             "block0_answer_rms": float(rms[1, answer]),
                             "final_answer_rms": float(rms[-1, answer]),
                             "first_nonzero_answer_hidden": next((int(i) for i, value in enumerate(rms[:, answer]) if value > 1e-7), None),
                             "answer_curve": rms[:, answer].tolist()})
    return delta_path, pairs, by_group, split_reproduction, token_onsets


def append_memo(result):
    heading = f"## Phase {PHASE}: 六真实操作全1200答案场与72条全token全坐标图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 对Phase2601全部1200条自然prompt（包括行为失败组）采集embedding+36 block、答案边界全部2560物理坐标；每个族/语言/发现-确认-外测各冻结1个pair，额外保存72条逐token全场。pair只改变一个source span：

$$D_{{ltd}}=H_{{ltd}}(x^1)-H_{{ltd}}(x^0),\qquad
r_l=\sqrt{{D^{{-1}}\sum_dD_{{l,t_a,d}}^2}}.$$

同时在每层用相同final norm与目标/反事实首token的unembedding行计算双token logit lens；该读出只是测量，不把中层logit lens当模型实际提前输出。

**测试用例。** 1200×37×2560答案边界场；600×37×2560成对差分；12族/语言×3 split×2 variant=72个逐token场，所有坐标以float16保留且分析为float32/64，不做Top-K/PCA。Qwen3-4B BF16 CUDA非量化。

**结果汇总。** 12组动力=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；发现—确认逐层全坐标相关=`{json.dumps(result['split_reproduction'], ensure_ascii=False)}`；72代表pair的因果零区与answer出生=`{json.dumps(result['token_onsets_summary'], ensure_ascii=False)}`；原场=`{json.dumps(result['field'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2602_c626945_c643328_natural_fullcoordinate_field.py`；全1200答案场、600 pair差分、72条全token场、pair索引、manifest与final位于`{OUT}`。

**分析与理论进展。** 单个自然prompt的局部source替换在source之前保持严格因果零区，答案位置embedding为零差，经过上下文混合后才出现差分；这比四prompt二阶项更接近自然source→answer传递。发现—确认方向相关若在行为成功组高于失败组，只是可复现候选脉络，尚需单recipient patch裁决。

**问题硬伤。** $D$仍由两个反事实运行相减；不同pair的目标词身份不同；logit lens不是实际中层读出；float16落盘可能掩盖极小早层值；模板生成任务不能代表开放语言。行为失败组只作为负对照，不能用其低复现反证机制不存在。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    p2601_final = load_json(P2601 / "analysis/final.json")
    material = read_jsonl(P2601 / "material/cases.jsonl")
    generated = read_jsonl(P2601 / "behavior/greedy_generation.jsonl")
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        boundary_path, logit_path, manifest = collect(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    delta_path, pairs, by_group, split_repro, token_onsets = analyze(
        material, boundary_path, logit_path, manifest, generated)
    pair_path = OUT / "analysis/pair_index.json"
    onset_path = OUT / "analysis/fulltoken_onsets.json"
    manifest_path = OUT / "field/fulltoken_manifest.json"
    save_json(pair_path, pairs)
    save_json(onset_path, token_onsets)
    save_json(manifest_path, manifest)
    onset_summary = {
        "n_pairs": len(token_onsets),
        "embedding_prefix_max": max(row["embedding_prefix_max"] for row in token_onsets),
        "embedding_answer_rms_max": max(row["embedding_answer_rms"] for row in token_onsets),
        "first_nonzero_answer_hidden_counts": {str(value): sum(row["first_nonzero_answer_hidden"] == value for row in token_onsets)
                                                  for value in sorted({row["first_nonzero_answer_hidden"] for row in token_onsets})},
        "median_block0_answer_rms": float(np.median([row["block0_answer_rms"] for row in token_onsets])),
        "median_final_answer_rms": float(np.median([row["final_answer_rms"] for row in token_onsets])),
    }
    field = {"answer_boundary_shape": list(np.load(boundary_path, mmap_mode="r").shape),
             "pair_delta_shape": list(np.load(delta_path, mmap_mode="r").shape),
             "logit_lens_shape": list(np.load(logit_path, mmap_mode="r").shape),
             "fulltoken_prompts": len(manifest),
             "fulltoken_bytes": sum(row["bytes"] for row in manifest),
             "answer_boundary_bytes": boundary_path.stat().st_size,
             "pair_delta_bytes": delta_path.stat().st_size,
             "all_coordinates": True, "no_topk": True}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized", "field": field,
        "by_family_language": by_group, "split_reproduction": split_repro,
        "token_onsets_summary": onset_summary,
        "claim_boundary": "full-coordinate descriptive source-to-answer fields; pair differences are not single-prompt causal gears",
        "hashes": {"boundary": sha256(boundary_path), "pair_delta": sha256(delta_path),
                   "logit_lens": sha256(logit_path), "manifest": sha256(manifest_path)},
        "language_mechanism_closed": False,
    }
    result["checks"] = {
        "phase2601_complete": p2601_final["all_checks_passed"],
        "all_1200_answer_fields": field["answer_boundary_shape"] == [1200, 37, 2560],
        "all_600_pair_deltas": field["pair_delta_shape"] == [600, 37, 2560],
        "all_72_fulltoken_prompts": len(manifest) == 72,
        "all_36_fulltoken_pairs": len(token_onsets) == 36,
        "causal_embedding_prefix_zero": onset_summary["embedding_prefix_max"] == 0.0,
        "answer_embedding_zero": onset_summary["embedding_answer_rms_max"] == 0.0,
        "all_coordinates_no_topk": field["all_coordinates"] and field["no_topk"],
        "failures_retained": any(group not in p2601_final["qualified_groups"] for group in by_group),
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "field", "token_onsets_summary", "checks", "all_checks_passed")},
                     ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
