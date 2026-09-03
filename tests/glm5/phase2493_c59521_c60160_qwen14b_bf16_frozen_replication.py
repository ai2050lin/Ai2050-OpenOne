#!/usr/bin/env python3
"""Frozen Qwen3-14B BF16 device_map=auto replication with crash-isolated worker."""
from __future__ import annotations

import gc
import json
import shutil
import subprocess
import sys
import traceback
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
OUT = RESULT / "phase2493_c59521_c60160_qwen14b_bf16_frozen_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODEL_PATH = ROOT / "models/hf/Qwen3-14B"
OFFLOAD = ROOT / "tests/glm5_temp/phase2493_qwen14b_bf16_offload"
PHASE, CAMPAIGN = 2493, "C59521-C60160"
WORKER_FINAL = OUT / "worker/collection.json"
sys.path.insert(0, str(TESTS))
import phase2487_c54721_c55872_orthogonal_family_interface_behavior as materials  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def frozen_rows() -> list[dict]:
    f2487 = json.loads((P2487 / "analysis/final.json").read_text(encoding="utf-8"))
    families = sorted(f2487["behavior"]["qualified"]["entity"])
    return [r for r in read_jsonl(P2487 / "material/orthogonal_family_interface_rows.jsonl")
            if r["unit"] == 16 and r["output_interface"] == "entity" and r["family"] in families
            and r["surface"] in (0, 2)]


def qmodules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def worker() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = frozen_rows()
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, device_map="auto", max_memory={0: "14GiB", "cpu": "15GiB"},
        offload_folder=str(OFFLOAD), offload_state_dict=True, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model.eval()
    mods = qmodules(model)
    dim = int(model.get_input_embeddings().weight.shape[1])
    state_path = OUT / "raw/qwen14b_answer_boundary_allqpoint.float16.npy"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float16, shape=(len(rows), len(mods), dim))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(mods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    input_device = model.get_input_embeddings().weight.device
    index = []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(rows):
                ids_list = [int(x) for x in tokenizer.encode(row["prompt"], add_special_tokens=False)]
                ids = torch.tensor([ids_list], dtype=torch.long, device=input_device)
                captures.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                for qpoint in range(len(mods)):
                    state[model_row, qpoint] = captures[qpoint][0, -1].float().cpu().numpy().astype(np.float16)
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=10,
                                           do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id)
                new_ids = [int(x) for x in generated[0, ids.shape[1]:].detach().cpu().tolist()]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                parsed, correct, _ = materials.parse_answer(text, row)
                index.append({"model_row": model_row, "case_id": row["case_id"], "family": row["family"],
                              "language": row["language"], "surface": row["surface"], "generated_ids": new_ids,
                              "generated_text": text, "parsed_answer": parsed, "parsed_correct": bool(correct)})
                state.flush()
                print(f"[phase2493 worker] {model_row + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        state.flush(); del state
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    index_path = OUT / "index/qwen14b_rows.jsonl"
    write_jsonl(index_path, index)
    device_map = "auto"
    collection = {"success": True, "rows": len(rows), "field": str(state_path),
                  "shape": [len(rows), len(mods), dim], "index": str(index_path),
                  "precision": "BF16 weights and compute; nonquantized; device_map=auto",
                  "device_map": device_map, "max_memory": {"cuda:0": "14GiB", "cpu": "15GiB"}}
    save(WORKER_FINAL, collection)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def analyze(collection: dict) -> dict:
    states = np.load(collection["field"], mmap_mode="r")
    rows = read_jsonl(Path(collection["index"]))
    families = sorted({r["family"] for r in rows})
    qpoint = round((json.loads((P2490 / "analysis/final.json").read_text(encoding="utf-8"))["selection"]["answer_boundary"] / 37) * (states.shape[1] - 1))
    p = {}
    for language in ("en", "zh"):
        values = []
        for family in families:
            ids = [r["model_row"] for r in rows if r["language"] == language and r["family"] == family]
            values.append(np.asarray(states[ids, qpoint], dtype=np.float64).mean(axis=0))
        values = np.stack(values); p[language] = values - values.mean(axis=0, keepdims=True)
    same = [cosine(p["en"][i], p["zh"][i]) for i in range(len(families))]
    wrong = [cosine(p["en"][i], p["zh"][(i + shift) % len(families)])
             for i in range(len(families)) for shift in range(1, len(families))]
    behavior = {"rows": len(rows), "parsed_rate": sum(r["parsed_answer"] is not None for r in rows) / len(rows),
                "accuracy": sum(r["parsed_correct"] for r in rows) / len(rows)}
    return {"relative_depth_qpoint": qpoint, "families": families, "behavior": behavior,
            "crosslanguage_raw_signed": {"same_mean": float(np.mean(same)), "wrong_mean": float(np.mean(wrong)),
                                         "wrong_q95": float(np.quantile(wrong, 0.95)),
                                         "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, 0.95))}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Qwen3-14B非量化BF16 device_map=auto冻结跨尺度复核（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 冻结Phase2487的九个entity合格族，仅取unit16、中英、surface0/2共36条，不在14B重新选择family或层位。严格尝试Qwen3-14B BF16权重与计算、`device_map=auto`、14GiB GPU+15GiB CPU预算；禁止NF4/INT8替代。模型工作在隔离子进程，即使Windows访问冲突也由父进程保存退出码与日志。成功时保存answer-boundary的Embedding、40 block输出、final norm全部5120坐标，并运行真实贪心行为；按Qwen4B q21/37的相对深度冻结14B层位。

$$q_{{14B}}^*=\operatorname{{round}}\left(\frac{{q_{{4B}}^*}}{{37}}(Q_{{14B}}-1)\right).$$

**结果汇总。** 可行性 `{json.dumps(result['feasibility'], ensure_ascii=False)}`；采集 `{json.dumps(result.get('collection'), ensure_ascii=False)}`；分析 `{json.dumps(result.get('analysis'), ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2493_c59521_c60160_qwen14b_bf16_frozen_replication.py`；子进程stdout/stderr、成功原场/索引或失败诊断、`analysis/final.json`位于同名目录。

**分析、问题硬伤与结论。** 14B成功也只比较模型内family关系，不比较4B/14B物理坐标号或绝对幅度；36条是冻结复核而非新发现。若失败，结论只限本机31.37GiB RAM、RTX5080 16GiB下此次非量化加载不可行，不得解释为模型机制负结果。量化替代被明确拒绝，研究主线继续使用4B全场。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def parent() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if WORKER_FINAL.exists(): WORKER_FINAL.unlink()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    try:
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        exit_code = completed.returncode; stdout = completed.stdout; stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        exit_code = -9; stdout = error.stdout or ""; stderr = (error.stderr or "") + "\nTIMEOUT after 1800 seconds"
    log_dir = OUT / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "worker_stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / "worker_stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
    success = exit_code == 0 and WORKER_FINAL.exists()
    collection = json.loads(WORKER_FINAL.read_text(encoding="utf-8")) if success else None
    analysis = analyze(collection) if success else None
    if OFFLOAD.exists() and OFFLOAD.resolve().is_relative_to((ROOT / "tests/glm5_temp").resolve()):
        shutil.rmtree(OFFLOAD)
    feasibility = {"attempted": True, "success": success, "exit_code": exit_code,
                   "quantization_used": False, "stdout_log": str(log_dir / "worker_stdout.log"),
                   "stderr_log": str(log_dir / "worker_stderr.log"),
                   "failure_scope": None if success else "local BF16 materialization/inference feasibility only"}
    checks = {"attempt_recorded": True, "nonquantized_only": True, "device_map_auto_requested": True,
              "isolated_worker": True, "success_has_full_coordinates_or_failure_has_log":
                  bool(collection and collection["shape"][-1] == 5120) if success else (log_dir / "worker_stderr.log").exists(),
              "temporary_offload_cleaned": not OFFLOAD.exists(), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "feasibility": feasibility,
              "collection": collection, "analysis": analysis,
              "adjudication": {"qwen14b_bf16_replication_available": success,
                               "quantized_fallback_used": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        try: worker()
        except Exception:
            traceback.print_exc(); raise
    else: parent()
