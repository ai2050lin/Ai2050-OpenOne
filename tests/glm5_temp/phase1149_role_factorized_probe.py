from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1148_mandatory_mediation_calibration as p1148  # noqa: E402


OUT_PATH = ROOT / "tests" / "glm5_temp" / "phase1149_role_factorized_probe.json"
CONDITIONS = ("answer_boundary", "role_factorized")
MAX_STEPS = 4500
CHECKPOINTS = (0, 500, 1500, 3000, 4500)


class RoleFactorizedModel(p1148.MediationBindingModel):
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, states = self.backbone(input_ids, return_states=True)
        normalized = self.backbone.final_norm(states[-1])
        row_state = normalized[:, -3, :]
        column_state = normalized[:, -2, :]
        return logits, self.row_head(row_state), self.column_head(column_state)


def evaluate(
    model: p1148.MediationBindingModel,
    dataset: dict[str, np.ndarray],
    lexicon: list[int],
) -> dict[str, float]:
    metrics, _ = p1148.evaluate_model(
        model,
        "soft_EF",
        dataset,
        lexicon,
        512,
        "development",
        False,
    )
    return {
        key: float(metrics[key])
        for key in (
            "accuracy",
            "row_address_accuracy",
            "column_address_accuracy",
            "joint_address_accuracy",
            "oracle_accuracy",
        )
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    prereg = p1148.read_json(p1148.PREREG_PATH)
    spec = copy.deepcopy(prereg["replicates"]["discovery_small_r1"])
    spec["training_seed"] = 149911
    spec["sampler_seed"] = 149912
    spec["lexicon_seed"] = 149913
    spec["lexicon"] = p1148.p1147.p1146.make_lexicon(149913)
    spec["data_seeds"]["training"] = 149914
    spec["data_seeds"]["holdout_evaluation"] = 149915
    training_data = p1148.make_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(spec["data_seeds"]["training"]),
        spec["lexicon"],
    )
    evaluation_data = p1148.make_dataset(
        4096,
        prereg["data"]["pairs"]["discovery"],
        int(spec["data_seeds"]["holdout_evaluation"]),
        spec["lexicon"],
    )
    rng = np.random.default_rng(int(spec["sampler_seed"]))
    schedule = rng.integers(
        0,
        len(training_data["inputs"]),
        size=(MAX_STEPS, int(spec["training"]["batch_size"])),
        dtype=np.int64,
    )
    cpu = {
        key: torch.from_numpy(training_data[key])
        for key in ("inputs", "targets", "row_targets", "column_targets", "grid_values")
    }
    device = torch.device("cuda")
    result = {
        "scope": "engineering_only_not_scientific_evidence",
        "conditions": {},
        "paired_training_dataset_digest": p1148.dataset_digest(training_data),
        "paired_schedule_digest": p1148.array_digest(schedule),
    }
    for condition in CONDITIONS:
        p1148.set_seed(int(spec["training_seed"]))
        model_class = (
            p1148.MediationBindingModel
            if condition == "answer_boundary"
            else RoleFactorizedModel
        )
        model = model_class(
            p1148.p1147.p1146.ModelConfig(**spec["architecture"])
        ).to(device)
        initial_digest = p1148.state_digest(model.state_dict())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.003,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        trajectory = [{"step": 0, "metrics": evaluate(model, evaluation_data, spec["lexicon"])}]
        model.train()
        values = p1148.candidate_ids(spec["lexicon"], device)
        for step in range(MAX_STEPS):
            progress = step / max(1, MAX_STEPS - 1)
            lr = 0.0003 + 0.5 * (0.003 - 0.0003) * (1.0 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
            indices = torch.from_numpy(schedule[step])
            ids = cpu["inputs"][indices].to(device)
            targets = cpu["targets"][indices].to(device)
            rows = cpu["row_targets"][indices].to(device)
            columns = cpu["column_targets"][indices].to(device)
            grids = cpu["grid_values"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, row_logits, column_logits = model(ids)
            answer_loss, _, _ = p1148.answer_loss_and_prediction(
                "soft_EF", logits, row_logits, column_logits, grids, targets, values
            )
            loss = (
                answer_loss
                + F.cross_entropy(row_logits.float(), rows)
                + F.cross_entropy(column_logits.float(), columns)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            current = step + 1
            if current in CHECKPOINTS[1:]:
                trajectory.append(
                    {"step": current, "metrics": evaluate(model, evaluation_data, spec["lexicon"])}
                )
                model.train()
        result["conditions"][condition] = {
            "initial_state_digest": initial_digest,
            "parameter_count": p1148.p1147.p1146.model_parameter_count(model),
            "trajectory": trajectory,
        }
        del model
        torch.cuda.empty_cache()
    result["paired_initial_states"] = len(
        {entry["initial_state_digest"] for entry in result["conditions"].values()}
    ) == 1
    result["equal_parameter_counts"] = len(
        {entry["parameter_count"] for entry in result["conditions"].values()}
    ) == 1
    result["probe_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_PATH, result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
