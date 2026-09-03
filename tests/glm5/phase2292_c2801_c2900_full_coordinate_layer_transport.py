#!/usr/bin/env python3
"""Map family-specific adjacent-checkpoint response transport without compression."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
FIELD_OUT = RESULT / "phase2290_c2601_c2700_qwen4b_natural_dynamic_field"
PREDICT_OUT = RESULT / "phase2291_c2701_c2800_sample_conditioned_coordinate_tournament"
OUT = RESULT / "phase2292_c2801_c2900_full_coordinate_layer_transport"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2288_c2501_c2580_natural_sample_condition_contract as contract  # noqa: E402


PHASE = 2292
CAMPAIGN = "C2801-C2900"
EPS = 1e-8
TRANSPORT_GATES = {
    "gain_over_mean": 0.03,
    "gain_over_shared": 0.01,
    "gain_over_identity": 0.01,
    "gain_over_other_controls": 0.01,
    "coordinate_win_fraction": 0.52,
}
ATLAS = OUT / "atlas/layer_transport_error_field.float32.npy"
ATLAS_ROWS = OUT / "atlas/layer_transport_error_rows.jsonl"


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def pairs(index: list[dict]) -> dict[str, list[dict]]:
    cells = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): row
             for row in index}
    output = defaultdict(list)
    for family, language, surface, unit in sorted({key[:4] for key in cells}):
        left = cells.get((family, language, surface, unit, 0))
        right = cells.get((family, language, surface, unit, 1))
        if left and right:
            output[family].append({"partition": left["partition"], "left": int(left["hidden_index"]),
                                   "right": int(right["hidden_index"]), "language": language,
                                   "surface": surface, "unit": unit})
    return dict(output)


def responses(field: np.ndarray, rows: list[dict], q: int, role: int) -> np.ndarray:
    left = np.asarray(field[[row["left"] for row in rows], q, role], np.float32)
    right = np.asarray(field[[row["right"] for row in rows], q, role], np.float32)
    return right - left


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    a = np.mean(xc * (y - my), axis=0) / (np.mean(xc * xc, axis=0) + 1e-5)
    return a.astype(np.float32), (my - a * mx).astype(np.float32)


def predict(model: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    return model[0] * x + model[1]


class Transport:
    def __init__(self, field: np.ndarray, grouped: dict[str, list[dict]]):
        self.field = field
        self.grouped = grouped
        self.families = sorted(grouped)
        self.cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def rows(self, family: str, partition: str) -> list[dict]:
        return [row for row in self.grouped[family] if row["partition"] == partition]

    def fit(self, family: str, q: int, role: int, mode: str = "normal") -> tuple[np.ndarray, np.ndarray]:
        key = (family, q, role, mode)
        if key in self.cache:
            return self.cache[key]
        rows = self.rows(family, "discovery")
        x = responses(self.field, rows, q, role)
        y = responses(self.field, rows, q + 1, role)
        if mode == "shuffled":
            y = y[np.arange(len(y))[::-1]]
        self.cache[key] = fit_affine(x, y)
        return self.cache[key]

    def shared(self, family: str, q: int, role: int) -> tuple[np.ndarray, np.ndarray]:
        key = (family, q, role, "shared")
        if key in self.cache:
            return self.cache[key]
        xs, ys = [], []
        for other in self.families:
            if other == family:
                continue
            rows = self.rows(other, "discovery")
            xs.append(responses(self.field, rows, q, role))
            ys.append(responses(self.field, rows, q + 1, role))
        self.cache[key] = fit_affine(np.concatenate(xs), np.concatenate(ys))
        return self.cache[key]

    def evaluate(self, family: str, partition: str, q: int, role: int) -> tuple[dict, dict[str, np.ndarray]]:
        rows = self.rows(family, partition)
        x = responses(self.field, rows, q, role)
        truth = responses(self.field, rows, q + 1, role)
        wrong = self.families[(self.families.index(family) + 1) % len(self.families)]
        train_truth = responses(self.field, self.rows(family, "discovery"), q + 1, role)
        predictions = {
            "candidate": predict(self.fit(family, q, role), x),
            "identity": x,
            "target_mean": np.broadcast_to(train_truth.mean(0), truth.shape),
            "shared_model": predict(self.shared(family, q, role), x),
            "wrong_family": predict(self.fit(wrong, q, role), x),
            "shuffled_labels": predict(self.fit(family, q, role, "shuffled"), x),
            "zero": np.zeros_like(truth),
        }
        errors = {name: np.mean(np.abs(value - truth), axis=0) for name, value in predictions.items()}
        candidate = errors["candidate"]
        controls = [name for name in predictions if name != "candidate"]
        gains = {name: float((errors[name].mean() - candidate.mean()) /
                             max(float(errors[name].mean()), EPS)) for name in controls}
        wins = {name: float(np.mean(candidate < errors[name])) for name in controls}
        gates = TRANSPORT_GATES
        passes = (gains["target_mean"] >= gates["gain_over_mean"] and
                  gains["shared_model"] >= gates["gain_over_shared"] and
                  gains["identity"] >= gates["gain_over_identity"] and
                  all(gains[name] >= gates["gain_over_other_controls"]
                      for name in ("wrong_family", "shuffled_labels", "zero")) and
                  wins["shared_model"] >= gates["coordinate_win_fraction"] and
                  wins["identity"] >= gates["coordinate_win_fraction"])
        return ({"family": family, "partition": partition, "checkpoint_from": q,
                 "checkpoint_to": q + 1, "role_index": role, "role": contract.ROLES[role],
                 "pairs": len(rows), "candidate_mae": float(candidate.mean()),
                 "gains": gains, "wins": wins, "passes": bool(passes)}, errors)


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 六构式全坐标层间响应传动图（{CAMPAIGN}） [{stamp}]

**测试原理。** 本期不只比较同层响应，而把每个样本在检查点 `q` 的完整状态响应作为输入，逐坐标预测 `q+1`：

$$
widehat R_{{i,q+1,r,j}}=a_{{f,q,r,j}}R_{{i,q,r,j}}+b_{{f,q,r,j}}.
$$

对六个行为合格构式、37条相邻检查点边和六角色全格测试。discovery拟合；confirmation与fresh-confirmation均通过才揭示fresh-lockbox。族特异传播必须同时优于身份复制、目标族均值、跨族共享传播、错族、打乱标签和零模型；全部2560坐标进入误差和胜率，不做PCA、Top-K或余弦筛选。

**测试用例与结果。** 测试涵盖施事、态度事件、比较、位置、持有和关系从句。正式lockbox通过格数 `{result['lockbox_pass_count']}/{result['total_cells']}`；按检查点分布 `{json.dumps(result['checkpoint_distribution'], ensure_ascii=False)}`；按构式分布 `{json.dumps(result['family_distribution'], ensure_ascii=False)}`；深层通过格 `{json.dumps(result['deep_lockbox_cells'], ensure_ascii=False)}`；全坐标传动误差图 `{json.dumps(result['atlas'], ensure_ascii=False)}`；门槛 `{json.dumps(TRANSPORT_GATES, ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 相邻层高可预测性首先可能来自残差式连续传播；只有超过共享传播与身份复制的部分才记为族特异增量。即使某格通过，也不是唯一参数电路或新数学结构。逐坐标独立拟合仍遗漏坐标联盟；角色末token和人工自然材料仍限制外推。脚本 `tests/glm5/phase2292_c2801_c2900_full_coordinate_layer_transport.py`；结果 `tests/glm5/result/phase2292_c2801_c2900_full_coordinate_layer_transport`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    predictor = json.loads((PREDICT_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if not predictor["all_checks_passed"]:
        raise RuntimeError("Phase2291 is not authorized")
    field = np.load(FIELD_OUT / "raw/qwen3_4b_natural_role_field.float16.npy", mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    grouped = pairs(index)
    transport = Transport(field, grouped)
    decisions, atlas_values, atlas_rows = [], [], []
    for family in transport.families:
        for q in range(field.shape[1] - 1):
            for role in range(field.shape[2]):
                confirmation, _ = transport.evaluate(family, "confirmation", q, role)
                fresh = lock = None
                final_errors = None
                if confirmation["passes"]:
                    fresh, fresh_errors = transport.evaluate(family, "fresh_confirmation", q, role)
                    final_errors = fresh_errors
                    if fresh["passes"]:
                        lock, lock_errors = transport.evaluate(family, "fresh_lockbox", q, role)
                        final_errors = lock_errors
                decision = {"family": family, "checkpoint_from": q, "checkpoint_to": q + 1,
                            "role_index": role, "role": contract.ROLES[role],
                            "confirmation": confirmation, "fresh_confirmation": fresh,
                            "fresh_authorized": bool(fresh and fresh["passes"]),
                            "lockbox_revealed": lock is not None, "lockbox": lock,
                            "lockbox_pass": bool(lock and lock["passes"])}
                decisions.append(decision)
                if final_errors is not None:
                    final_partition = "fresh_lockbox" if lock is not None else "fresh_confirmation"
                    for error_name, values in final_errors.items():
                        atlas_rows.append({"row": len(atlas_values), "family": family,
                                           "checkpoint_from": q, "checkpoint_to": q + 1,
                                           "role": contract.ROLES[role], "partition": final_partition,
                                           "error": error_name})
                        atlas_values.append(values.astype(np.float32))
        print(f"[transport] {family} complete", flush=True)
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    atlas = np.stack(atlas_values) if atlas_values else np.empty((0, field.shape[-1]), np.float32)
    np.save(ATLAS, atlas)
    write_rows(ATLAS_ROWS, atlas_rows)
    mmap = getattr(field, "_mmap", None)
    if mmap is not None:
        mmap.close()
    passed = [row for row in decisions if row["lockbox_pass"]]
    checkpoint_distribution = dict(sorted(Counter(row["checkpoint_from"] for row in passed).items()))
    family_distribution = dict(sorted(Counter(row["family"] for row in passed).items()))
    deep = [{"family": row["family"], "checkpoint_from": row["checkpoint_from"], "role": row["role"],
             "gains": row["lockbox"]["gains"]} for row in passed if 6 <= row["checkpoint_from"] <= 30]
    checks = {
        "all_cells": len(decisions) == len(transport.families) * 37 * 6,
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_authorized"] for row in decisions),
        "all_controls": all(set(row["confirmation"]["gains"]) ==
                            {"identity", "target_mean", "shared_model", "wrong_family", "shuffled_labels", "zero"}
                            for row in decisions),
        "atlas_full_coordinates": atlas.shape[-1] == 2560,
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "transport_gates": TRANSPORT_GATES,
        "total_cells": len(decisions), "lockbox_pass_count": len(passed),
        "checkpoint_distribution": checkpoint_distribution,
        "family_distribution": family_distribution, "deep_lockbox_cells": deep,
        "decisions": decisions,
        "atlas": {"path": str(ATLAS.relative_to(ROOT)), "rows": str(ATLAS_ROWS.relative_to(ROOT)),
                  "shape": list(atlas.shape)},
        "hashes": {"atlas": file_hash(ATLAS), "rows": file_hash(ATLAS_ROWS)},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(passed)}/{len(decisions)} adjacent-checkpoint cells prospectively beat identity, "
                              f"mean, shared, wrong-family, shuffled and zero controls; {len(deep)} lie in q6-q30. "
                              "These are family-specific predictive transport increments over shared residual propagation, not causal circuits."),
        "next_authorization": "Run multiscale causal intervention only for Phase2291 middle-layer operator anchors; otherwise record causal NA while preserving transport observations.",
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
