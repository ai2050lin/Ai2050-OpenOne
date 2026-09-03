#!/usr/bin/env python3
"""Repair thinking/truncation confounds in autonomous label-free sentence reordering."""
from __future__ import annotations

import gc
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2380 = RESULT / "phase2380_c16241_c16560_object_slot_progress_adjudication"
OUT = RESULT / "phase2386_c18161_c18480_chat_generation_closure"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
PHASE = 2386
CAMPAIGN = "C18161-C18480"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def normalize_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[一二三四][、.])\s*", "", line.strip())
    return re.sub(r"\s+", " ", line).strip().strip('"“”')


def score(row: dict, text: str, generated_ids: list[int], maximum: int, eos: int | None) -> dict:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    target = [normalize_line(sentence) for sentence in row["target_sentences"]]
    positions = [text.find(sentence) for sentence in row["sentences"]]
    target_positions = [positions[sid] for sid in row["target_order"]]
    all_present = all(position >= 0 for position in positions)
    order_exact = all_present and target_positions == sorted(target_positions)
    first_four = lines[:4]
    return {"generated": text, "generated_tokens": len(generated_ids), "ended_with_eos": bool(generated_ids and eos is not None and generated_ids[-1] == eos),
            "hit_max_new_tokens": len(generated_ids) >= maximum, "sentence_recall": float(np.mean([p >= 0 for p in positions])),
            "all_sentences_present": all_present, "identity_order_exact": order_exact,
            "first_four_lines_exact": first_four == target, "verbatim_full_exact": lines == target,
            "extra_nonempty_lines": max(0, len(lines) - 4), "repeated_target_sentence": any(text.count(sentence) > 1 for sentence in row["sentences"])}


def summarize(rows: list[dict]) -> dict:
    metrics = ("sentence_recall", "all_sentences_present", "identity_order_exact", "first_four_lines_exact",
               "verbatim_full_exact", "ended_with_eos", "hit_max_new_tokens", "repeated_target_sentence")
    result = {"rows": len(rows), **{metric: float(np.mean([row[metric] for row in rows])) for metric in metrics},
              "mean_extra_nonempty_lines": float(np.mean([row["extra_nonempty_lines"] for row in rows]))}
    for dimension in ("family", "language", "surface", "reverse"):
        result[f"by_{dimension}"] = {value: {metric: float(np.mean([row[metric] for row in rows if str(row[dimension]) == value]))
                                                      for metric in metrics[:5]}
                                          for value in sorted({str(row[dimension]) for row in rows})}
    return result


def generate(rows: list[dict], batch_size: int = 8) -> tuple[dict, dict]:
    output_path = OUT / "generation/qwen4b_chat_lockbox.jsonl"
    if output_path.exists():
        existing = read_rows(output_path)
        if len(existing) == len(rows): return summarize(existing), {"rows": len(existing), "resumed": True}
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    tokenizer.padding_side = "left"; device = model.get_input_embeddings().weight.device
    pad = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0); eos = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    compiled = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "keys") and "input_ids" in ids: ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)): ids = ids[0]
        compiled.append((row, [int(x) for x in ids]))
    generated_rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(compiled), batch_size):
                batch = compiled[start:start + batch_size]; width = max(len(ids) for _, ids in batch)
                input_ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(input_ids)
                for local, (_, ids) in enumerate(batch):
                    value = torch.tensor(ids, device=device); input_ids[local, -len(value):] = value; mask[local, -len(value):] = 1
                maximum = 192
                outputs = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=maximum,
                                         pad_token_id=pad, eos_token_id=eos, use_cache=True)
                for local, (row, _ids) in enumerate(batch):
                    new_ids = outputs[local, width:].tolist(); text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    generated_rows.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"],
                                           "language": row["language"], "surface": row["surface"], "reverse": row["reverse"],
                                           "source_index": row["source_index"], **score(row, text, new_ids, maximum, eos)})
                write_rows(output_path, generated_rows)
                if (start + len(batch)) % 64 == 0 or start + len(batch) == len(compiled):
                    print(f"[phase2386 generation] {start + len(batch)}/{len(compiled)}", flush=True)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summarize(generated_rows), {"rows": len(compiled), "prompt_token_range": [min(len(ids) for _, ids in compiled), max(len(ids) for _, ids in compiled)],
                                       "max_new_tokens": 192, "chat_template": True, "enable_thinking": False,
                                       "stopping": "model EOS or non-oracle hard safety cap"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 原生聊天接口下的无标签长句生成闭合复测（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对全部256条fresh unit+来源排列联合锁箱，使用Qwen3原生chat template并显式`enable_thinking=False`，贪心生成最多192 token，以模型EOS作为正常停止，硬上限仅作安全边界且不使用目标答案oracle停止。逐条核对四个完整来源句是否出现、首次出现顺序、前四非空行逐字一致、全文逐字一致、EOS、截断、重复与多余行；与Phase2380直接prompt结果并列，不覆盖旧结果。

$$\mathrm{{closure}}=\mathrm{{order\ exact}}\land\mathrm{{all\ sentences\ present}}\land\neg\mathrm{{truncated}},\qquad
\mathrm{{verbatim}}=\mathbf 1[\widehat y=y].$$

**结果汇总。** 新生成 `{json.dumps(result['generation'], ensure_ascii=False)}`；执行合同 `{json.dumps(result['contract'], ensure_ascii=False)}`；旧接口对照 `{json.dumps(result['direct_baseline'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2386_c18161_c18480_chat_generation_closure.py`；256条逐样本输出与分析位于 `tests/glm5/result/phase2386_c18161_c18480_chat_generation_closure`。

**理论进展、问题硬伤与结论。** 本Phase修复thinking和128-token不足混淆，但不允许把接口优化冒充内部机制。若完整顺序/逐字率显著提高，说明Phase2373/2380的“内容弱”有一部分是解码协议问题；若仍低，则teacher-forced句内容匹配与自主重建之间确有未闭合断点。即使生成通过，也不能反推Attention head 25/10因果负责，仍需后续干预。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    all_rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]
    selected = [row for row in all_rows if row["partition"] == "fresh_joint_lockbox"]
    generation, contract = generate(selected)
    old = json.loads((P2380 / "analysis/final.json").read_text(encoding="utf-8"))["generation"]
    checks = {"all_rows": generation["rows"] == 256, "thinking_disabled": contract["enable_thinking"] is False,
              "finite": all(math.isfinite(generation[key]) for key in ("sentence_recall", "identity_order_exact", "verbatim_full_exact")),
              "non_oracle_stop": contract["stopping"] == "model EOS or non-oracle hard safety cap"}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B-BF16", "generation": generation,
              "contract": contract, "direct_baseline": old, "checks": checks, "all_checks_passed": all(checks.values()),
              "closure": {"behavior_content_closed": generation["first_four_lines_exact"] >= 0.80 and generation["hit_max_new_tokens"] <= 0.05,
                          "mechanism_closed": False}}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
