#!/usr/bin/env python3
"""Qualify and collect passive physical traces on the native generation path.

The baseline and instrumented runs use the exact same ``model.generate``
contract.  Hooks only read component outputs.  This avoids treating a manual
cache loop as an equivalent implementation of the model-owned generation API.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase416_dual_track_case_bank import (  # noqa: E402
    MODELS,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase416_real_collector_qualification import (  # noqa: E402
    component_tensor,
    encode_case,
    eos_ids,
    exact_answer,
    js_divergence,
    max_abs,
    neutral_generation_config,
    target_match,
)


PHASE_ID = "Phase417-NativeGenerationPhysicalTrace"
PHASE416 = ROOT / "tests/gpt5/result/phase416_formal_world_physical_atlas"
OUT = ROOT / "tests/gpt5/result/phase417_native_generation_physical_atlas"
CASES = PHASE416 / "phase416_registered_cases.jsonl"
CORE_COMPONENTS = (
    "layer_input",
    "attention_output",
    "mlp_output",
    "residual_increment",
    "layer_output",
)
THRESHOLDS = {
    "same_contract_score_max_abs": 0.0,
    "same_contract_score_js": 0.0,
    "component_ledger_relative_error": 1e-5,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(left.detach().float() - right.detach().float())
    scale = torch.linalg.vector_norm(left.detach().float()).clamp_min(1e-8)
    return float((difference / scale).item())


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def behavior_qualification(model: str) -> dict[str, bool]:
    qualification = read_json(PHASE416 / "phase416_instrument_domain_qualification.json")
    return {
        row["family_id"]: bool(row["formal_behavior_qualified"])
        for row in qualification["behavior_cells"]
        if row["model"] == model
    }


class PassiveGenerationCollector:
    def __init__(self, loaded: Any, qualified_families: dict[str, bool]) -> None:
        self.loaded = loaded
        self.layers = get_layers(loaded.model)
        self.qualified_families = qualified_families
        self.handles: list[Any] = []
        self.active = False
        self.case: dict[str, Any] | None = None
        self.call_index = -1
        self.transient: dict[tuple[int, int], dict[str, torch.Tensor]] = defaultdict(dict)
        self.rows: list[dict[str, Any]] = []
        self.case_rows: list[dict[str, Any]] = []
        self.raw_logits: list[torch.Tensor] = []
        self.case_ledger_errors: list[float] = []
        self.case_call_widths: list[int] = []

    def begin_case(self, case: dict[str, Any]) -> None:
        self.case = case
        self.call_index = -1
        self.transient.clear()
        self.raw_logits = []
        self.case_ledger_errors = []
        self.case_call_widths = []
        self.active = True

    def end_case(self) -> None:
        self.active = False
        self.case = None
        self.transient.clear()

    def record(self, layer: int, component: str, value: Any) -> None:
        if not self.active or self.case is None:
            return
        tensor = component_tensor(value)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            return
        vector = tensor[0, -1].detach().float()
        l2_norm = torch.linalg.vector_norm(vector)
        row = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": self.loaded.key,
            "case_id": self.case["case_id"],
            "semantic_case_id": self.case["semantic_case_id"],
            "family_id": self.case["family_id"],
            "mechanism_id": self.case["mechanism_id"],
            "item_index": int(self.case["item_index"]),
            "split": self.case["split"],
            "template_id": self.case["template_id"],
            "formal_behavior_qualified": self.qualified_families.get(self.case["family_id"], False),
            "native_generation_call_index": self.call_index,
            "prediction_step": self.call_index,
            "execution_phase": "prompt_prefill" if self.call_index == 0 else "cached_incremental",
            "call_token_width": int(tensor.shape[1]),
            "layer": layer,
            "relative_depth": layer / max(1, len(self.layers) - 1),
            "depth_bin": depth_bin(layer, len(self.layers)),
            "component": component,
            "position_role": "current_prediction_input",
            "vector_width": int(vector.numel()),
            "l2_norm": float(l2_norm.item()),
            "rms": float(torch.sqrt(torch.mean(vector.square())).item()),
            "signed_mean": float(torch.mean(vector).item()),
            "max_abs": float(torch.max(vector.abs()).item()),
            "numerically_finite": bool(torch.isfinite(vector).all().item()),
            "observer_id": None,
            "physical": True,
            "natural_generation": True,
            "predictive": False,
            "causal": False,
            "single_neuron_causal": False,
        }
        self.rows.append(row)

    def install(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            def layer_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
                if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                    return
                if index == 0:
                    self.call_index += 1
                    self.case_call_widths.append(int(inputs[0].shape[1]))
                key = (self.call_index, index)
                self.transient[key]["input"] = inputs[0]
                self.record(index, "layer_input", inputs[0])

            def attention_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if not self.active:
                    return
                tensor = component_tensor(output)
                self.transient[(self.call_index, index)]["attention"] = tensor
                self.record(index, "attention_output", tensor)

            def mlp_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if not self.active:
                    return
                tensor = component_tensor(output)
                self.transient[(self.call_index, index)]["mlp"] = tensor
                self.record(index, "mlp_output", tensor)

            def layer_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if not self.active:
                    return
                tensor = component_tensor(output)
                key = (self.call_index, index)
                values = self.transient.pop(key)
                before = values["input"]
                reconstructed = before + values["attention"] + values["mlp"]
                self.case_ledger_errors.append(relative_error(tensor, reconstructed))
                self.record(index, "residual_increment", tensor - before)
                self.record(index, "layer_output", tensor)

            self.handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre),
                    layer.self_attn.register_forward_hook(attention_post),
                    layer.mlp.register_forward_hook(mlp_post),
                    layer.register_forward_hook(layer_post),
                ]
            )

        def lm_head_post(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if self.active:
                tensor = component_tensor(output)
                self.raw_logits.append(tensor[0, -1].detach().float().cpu())

        self.handles.append(self.loaded.model.lm_head.register_forward_hook(lm_head_post))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def generation_output(
    loaded: Any,
    encoded: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> Any:
    return loaded.model.generate(
        **encoded,
        generation_config=neutral_generation_config(loaded),
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )


def generated_ids(output: Any, prompt_length: int) -> list[int]:
    return [int(value) for value in output.sequences[0, prompt_length:].tolist()]


def compare_outputs(baseline: Any, hooked: Any, prompt_length: int) -> dict[str, Any]:
    baseline_ids = generated_ids(baseline, prompt_length)
    hooked_ids = generated_ids(hooked, prompt_length)
    score_count_exact = len(baseline.scores) == len(hooked.scores)
    compared = min(len(baseline.scores), len(hooked.scores))
    score_max_abs = max(
        (max_abs(baseline.scores[index][0], hooked.scores[index][0]) for index in range(compared)),
        default=0.0,
    )
    score_js = max(
        (js_divergence(baseline.scores[index][0], hooked.scores[index][0]) for index in range(compared)),
        default=0.0,
    )
    return {
        "baseline_token_ids": baseline_ids,
        "hooked_token_ids": hooked_ids,
        "token_exact": baseline_ids == hooked_ids,
        "baseline_score_count": len(baseline.scores),
        "hooked_score_count": len(hooked.scores),
        "score_count_exact": score_count_exact,
        "score_max_abs": score_max_abs,
        "score_js": score_js,
    }


@torch.inference_mode()
def run_model(model_key: str, max_new_tokens: int, max_cases: int | None = None) -> dict[str, Any]:
    cases = [row for row in read_jsonl(CASES) if row["model"] == model_key]
    if max_cases is not None:
        cases = cases[:max_cases]
    required = 55 if max_cases is None else len(cases)
    loaded = None
    collector = None
    try:
        print(f"[Phase417] loading {model_key} for {len(cases)} native-generation cases", flush=True)
        loaded = load_probe_model(model_key)
        qualified_families = behavior_qualification(model_key)
        collector = PassiveGenerationCollector(loaded, qualified_families)
        collector.install()
        eos = eos_ids(loaded.tokenizer, loaded.model)
        for case_index, case in enumerate(cases, start=1):
            encoded = encode_case(loaded, case)
            prompt_length = int(encoded["input_ids"].shape[1])
            baseline = generation_output(loaded, encoded, max_new_tokens)
            collector.begin_case(case)
            hooked = generation_output(loaded, encoded, max_new_tokens)
            collector.active = False
            comparison = compare_outputs(baseline, hooked, prompt_length)
            ids = comparison["hooked_token_ids"]
            text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
            emitted_stop = any(token in eos for token in ids)
            physical_call_count = collector.call_index + 1
            call_score_alignment = physical_call_count == comparison["hooked_score_count"]
            raw_logit_alignment = len(collector.raw_logits) == comparison["hooked_score_count"]
            ledger_max = max(collector.case_ledger_errors, default=math.inf)
            physical_finite = all(
                row["numerically_finite"]
                for row in collector.rows
                if row["case_id"] == case["case_id"]
            )
            gates = {
                "same_contract_token_pass": comparison["token_exact"],
                "same_contract_score_count_pass": comparison["score_count_exact"],
                "same_contract_score_max_abs_pass": comparison["score_max_abs"] <= THRESHOLDS["same_contract_score_max_abs"],
                "same_contract_score_js_pass": comparison["score_js"] <= THRESHOLDS["same_contract_score_js"],
                "physical_call_score_alignment_pass": call_score_alignment,
                "raw_logit_call_alignment_pass": raw_logit_alignment,
                "component_ledger_pass": ledger_max <= THRESHOLDS["component_ledger_relative_error"],
                "physical_finite_pass": physical_finite,
            }
            collector.case_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": model_key,
                    "case_id": case["case_id"],
                    "semantic_case_id": case["semantic_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "split": case["split"],
                    "template_id": case["template_id"],
                    "prompt_sha256": case["prompt_sha256"],
                    "prompt_token_count": prompt_length,
                    "generated_text": text,
                    "target_event_match": target_match(text, case["target_aliases"]),
                    "exact_answer_match": exact_answer(text, case["target_aliases"]),
                    "emitted_stop": emitted_stop,
                    "right_censored": not emitted_stop and len(ids) >= max_new_tokens,
                    "native_generation_call_count": physical_call_count,
                    "call_token_widths": collector.case_call_widths,
                    "raw_logit_call_count": len(collector.raw_logits),
                    "component_ledger_max_relative_error": ledger_max,
                    "comparison": comparison,
                    "gates": gates,
                    "native_generation_case_pass": all(gates.values()),
                    "instrument_result_not_mechanism_evidence": True,
                    "causal": False,
                }
            )
            collector.end_case()
            del baseline, hooked, encoded
            gc.collect()
            if case_index % 5 == 0 or case_index == len(cases):
                passed = sum(row["native_generation_case_pass"] for row in collector.case_rows)
                print(f"[Phase417:{model_key}] {case_index}/{len(cases)} pass={passed}", flush=True)

        model_root = OUT / "models" / model_key
        write_jsonl(model_root / "phase417_native_generation_case_rows.jsonl", collector.case_rows)
        write_jsonl(model_root / "phase417_generation_physical_rows.jsonl", collector.rows)
        family_counts = Counter(row["family_id"] for row in collector.case_rows if row["native_generation_case_pass"])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model_key,
            "case_count": len(collector.case_rows),
            "required_case_count": required,
            "native_generation_case_pass_count": sum(row["native_generation_case_pass"] for row in collector.case_rows),
            "native_generation_qualification_pass": bool(
                len(collector.case_rows) == required
                and all(row["native_generation_case_pass"] for row in collector.case_rows)
            ),
            "target_event_match_count": sum(row["target_event_match"] for row in collector.case_rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in collector.case_rows),
            "right_censored_count": sum(row["right_censored"] for row in collector.case_rows),
            "physical_row_count": len(collector.rows),
            "physical_call_count": sum(row["native_generation_call_count"] for row in collector.case_rows),
            "family_pass_count": dict(sorted(family_counts.items())),
            "max_observed_errors": {
                "same_contract_score_max_abs": max(row["comparison"]["score_max_abs"] for row in collector.case_rows),
                "same_contract_score_js": max(row["comparison"]["score_js"] for row in collector.case_rows),
                "component_ledger_relative_error": max(row["component_ledger_max_relative_error"] for row in collector.case_rows),
            },
            "thresholds_frozen_before_denominator": THRESHOLDS,
            "physical_rows_sha256": hash_rows(collector.rows),
            "vram_gb": vram_gb(),
            "native_generation_physical_collection_authorized": bool(
                len(collector.case_rows) == required
                and all(row["native_generation_case_pass"] for row in collector.case_rows)
            ),
            "functional_labels_authorized": False,
            "causal_intervention_authorized": False,
            "neuron_scan_authorized": False,
            "claim_boundary": "same_native_generation_contract_noninterference_and_observer_free_physical_trace_only",
        }
        write_json(model_root / "phase417_native_generation_complete.json", summary)
        return summary
    finally:
        if collector is not None:
            collector.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    summary = run_model(args.model, args.max_new_tokens, args.max_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["native_generation_qualification_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
