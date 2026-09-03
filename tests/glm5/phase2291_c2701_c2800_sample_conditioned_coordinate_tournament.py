#!/usr/bin/env python3
"""Prospective sample-conditioned full-coordinate predictor tournament."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2289_c2581_c2600_partition_lexicon_repair"
FIELD_OUT = RESULT / "phase2290_c2601_c2700_qwen4b_natural_dynamic_field"
OUT = RESULT / "phase2291_c2701_c2800_sample_conditioned_coordinate_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2288_c2501_c2580_natural_sample_condition_contract as contract  # noqa: E402


PHASE = 2291
CAMPAIGN = "C2701-C2800"
ROUTES = ("pooled", "language_en_to_zh", "surface_narrative_to_dialogue")
MODELS = ("current_affine", "current_previous_affine")
EPS = 1e-8
ATLAS = OUT / "atlas/selected_coordinate_error_passports.float32.npy"
ATLAS_ROWS = OUT / "atlas/selected_coordinate_error_rows.jsonl"


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


def build_pairs(index: list[dict]) -> dict[str, list[dict]]:
    cells = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): row
             for row in index}
    output = defaultdict(list)
    for family, language, surface, unit in sorted({key[:4] for key in cells}):
        left = cells.get((family, language, surface, unit, 0))
        right = cells.get((family, language, surface, unit, 1))
        if left and right:
            output[family].append({"family": family, "language": language, "surface": surface,
                                   "unit": unit, "partition": left["partition"],
                                   "left": int(left["hidden_index"]), "right": int(right["hidden_index"])})
    return dict(output)


def route_filter(rows: list[dict], route: str, source: bool) -> list[dict]:
    if route == "pooled":
        return rows
    if route == "language_en_to_zh":
        return [row for row in rows if row["language"] == ("en" if source else "zh")]
    if route == "surface_narrative_to_dialogue":
        return [row for row in rows if row["surface"] == ("narrative" if source else "dialogue")]
    raise KeyError(route)


def arrays(field: np.ndarray, rows: list[dict], q: int, role: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(field[[row["left"] for row in rows], q, role], np.float32)
    right = np.asarray(field[[row["right"] for row in rows], q, role], np.float32)
    previous = (np.asarray(field[[row["left"] for row in rows], q - 1, role], np.float32)
                if q > 0 else np.zeros_like(left))
    return left, previous, right - left


def fit_current(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx, my = x.mean(0), y.mean(0)
    centered = x - mx
    a = np.mean(centered * (y - my), axis=0) / (np.mean(centered * centered, axis=0) + 1e-5)
    b = my - a * mx
    return a.astype(np.float32), b.astype(np.float32)


def fit_two(x: np.ndarray, previous: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, mp, my = x.mean(0), previous.mean(0), y.mean(0)
    xc, pc, yc = x - mx, previous - mp, y - my
    xx = np.mean(xc * xc, axis=0) + 1e-4
    pp = np.mean(pc * pc, axis=0) + 1e-4
    xp = np.mean(xc * pc, axis=0)
    xy = np.mean(xc * yc, axis=0)
    py = np.mean(pc * yc, axis=0)
    det = xx * pp - xp * xp + 1e-8
    a = (xy * pp - py * xp) / det
    b = (py * xx - xy * xp) / det
    c = my - a * mx - b * mp
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def fit_model(name: str, x: np.ndarray, previous: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    return fit_current(x, y) if name == "current_affine" else fit_two(x, previous, y)


def predict(name: str, model: tuple[np.ndarray, ...], x: np.ndarray, previous: np.ndarray) -> np.ndarray:
    if name == "current_affine":
        return x * model[0] + model[1]
    return x * model[0] + previous * model[1] + model[2]


class Tournament:
    def __init__(self, field: np.ndarray, pairs: dict[str, list[dict]]):
        self.field = field
        self.pairs = pairs
        self.families = sorted(pairs)
        self.cache: dict[tuple, tuple[np.ndarray, ...]] = {}

    def subset(self, family: str, partition: str, route: str, source: bool) -> list[dict]:
        rows = [row for row in self.pairs[family] if row["partition"] == partition]
        return route_filter(rows, route, source)

    def fit(self, family: str, route: str, q: int, role: int, name: str,
            mode: str = "normal") -> tuple[np.ndarray, ...]:
        key = (family, route, q, role, name, mode)
        if key in self.cache:
            return self.cache[key]
        rows = self.subset(family, "discovery", route, True)
        x, previous, y = arrays(self.field, rows, q, role)
        if mode == "shuffled":
            y = y[np.arange(len(y))[::-1]]
        self.cache[key] = fit_model(name, x, previous, y)
        return self.cache[key]

    def fit_shared(self, family: str, route: str, q: int, role: int, name: str) -> tuple[np.ndarray, ...]:
        key = (family, route, q, role, name, "shared")
        if key in self.cache:
            return self.cache[key]
        xs, ps, ys = [], [], []
        for other in self.families:
            if other == family:
                continue
            rows = self.subset(other, "discovery", route, True)
            x, previous, y = arrays(self.field, rows, q, role)
            xs.append(x); ps.append(previous); ys.append(y)
        self.cache[key] = fit_model(name, np.concatenate(xs), np.concatenate(ps), np.concatenate(ys))
        return self.cache[key]

    def evaluate(self, family: str, route: str, partition: str, q: int, role: int,
                 name: str) -> tuple[dict, dict[str, np.ndarray]]:
        rows = self.subset(family, partition, route, False)
        x, previous, truth = arrays(self.field, rows, q, role)
        candidate = predict(name, self.fit(family, route, q, role, name), x, previous)
        wrong = self.families[(self.families.index(family) + 1) % len(self.families)]
        source_rows = self.subset(family, "discovery", route, True)
        _, _, train_y = arrays(self.field, source_rows, q, role)
        target_train = self.subset(family, "discovery", route, False)
        tx, tp, ty = arrays(self.field, target_train, q, role)
        mean_prediction = np.broadcast_to(ty.mean(0), truth.shape)
        shared = predict(name, self.fit_shared(family, route, q, role, name), x, previous)
        shuffled = predict(name, self.fit(family, route, q, role, name, "shuffled"), x, previous)
        wrong_prediction = predict(name, self.fit(wrong, route, q, role, name), x, previous)
        zero = np.zeros_like(truth)
        px_model = fit_current(arrays(self.field, source_rows, q, role)[1], train_y)
        previous_prediction = previous * px_model[0] + px_model[1]
        current_model = fit_current(arrays(self.field, source_rows, q, role)[0], train_y)
        current_prediction = x * current_model[0] + current_model[1]
        oracle = predict(name, fit_model(name, tx, tp, ty), x, previous)
        predictions = {
            "candidate": candidate, "target_mean": mean_prediction, "shared_model": shared,
            "shuffled_labels": shuffled, "wrong_family": wrong_prediction, "zero": zero,
            "previous_checkpoint": previous_prediction, "current_only": current_prediction,
            "target_oracle": oracle,
        }
        errors = {key: np.mean(np.abs(value - truth), axis=0) for key, value in predictions.items()}
        candidate_error = errors["candidate"]
        controls = ["target_mean", "shared_model", "shuffled_labels", "wrong_family", "zero",
                    "previous_checkpoint"]
        if name == "current_previous_affine":
            controls.append("current_only")
        gains = {key: float((errors[key].mean() - candidate_error.mean()) /
                            max(float(errors[key].mean()), EPS)) for key in controls}
        wins = {key: float(np.mean(candidate_error < errors[key])) for key in controls}
        oracle_ratio = float(candidate_error.mean() / max(float(errors["target_oracle"].mean()), EPS))
        gates = contract.PREDICTION_GATES
        passes = (len(rows) >= gates["minimum_pairs"] and
                  gains["target_mean"] >= gates["gain_over_target_mean"] and
                  all(gains[key] >= gates["gain_over_shared_model"] for key in controls if key != "target_mean") and
                  wins["target_mean"] >= gates["coordinate_win_fraction"] and
                  wins["shared_model"] >= gates["coordinate_win_fraction"] and
                  oracle_ratio <= gates["maximum_oracle_ratio"])
        record = {"family": family, "route": route, "partition": partition, "checkpoint": q,
                  "role_index": role, "role": contract.ROLES[role], "model": name,
                  "pairs": len(rows), "candidate_mae": float(candidate_error.mean()),
                  "gains": gains, "wins": wins, "oracle_ratio": oracle_ratio, "passes": bool(passes)}
        return record, errors


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = [{key: row.get(key) for key in ("family", "route", "checkpoint", "role", "model",
                                               "fresh_authorized", "lockbox_revealed", "lockbox_pass")}
               for row in result["decisions"]]
    text = rf"""

## Phase {PHASE}: 自然双语样本条件逐坐标算子竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期只读取Phase2290六个双行为合格构式的完整六角色原场。每个构式测试 pooled、中英 `en→zh`、自然叙述 `narrative→dialogue` 三条路线。discovery 拟合逐坐标当前基态仿射与“当前基态+上一检查点基态”二输入仿射；扫描38个检查点和六角色。confirmation只冻结一个候选；候选必须同时击败目标均值、其他构式共享模型、打乱标签、错族、零模型、上一检查点，二输入模型还必须击败当前基态单输入，并受目标域oracle比例约束。随后依次揭示fresh-confirmation和fresh-lockbox。

**公式。** 对每个物理坐标独立拟合：

$$
widehat R_{{i,q,r,j}}=a_{{q,r,j}}H^0_{{i,q,r,j}}+b_{{q,r,j}},
$$

或：

$$
widehat R_{{i,q,r,j}}=a_{{q,r,j}}H^0_{{i,q,r,j}}+c_{{q,r,j}}H^0_{{i,q-1,r,j}}+b_{{q,r,j}}.
$$

没有PCA、Top-K、余弦筛选或平均差分搬运。

**结果与门槛。** 冻结门 `{json.dumps(contract.PREDICTION_GATES, ensure_ascii=False)}`；18条构式-路线裁决 `{json.dumps(summary, ensure_ascii=False)}`；正式lockbox阳性 `{json.dumps(result['lockbox_passed'], ensure_ascii=False)}`；全坐标误差护照 `{json.dumps(result['atlas'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 即便通过，也只说明当前样本基态可以在该模型、构式、角色与检查点下预测状态响应，不是唯一因果齿轮。逐坐标独立函数忽略坐标间联动；角色末token与人工材料仍是硬伤。脚本 `tests/glm5/phase2291_c2701_c2800_sample_conditioned_coordinate_tournament.py`；结果 `tests/glm5/result/phase2291_c2701_c2800_sample_conditioned_coordinate_tournament`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    source = json.loads((FIELD_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if not source["all_checks_passed"]:
        raise RuntimeError("Phase2290 field is not authorized")
    field = np.load(FIELD_OUT / "raw/qwen3_4b_natural_role_field.float16.npy", mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    pairs = build_pairs(index)
    tournament = Tournament(field, pairs)
    decisions, scans, atlas_values, atlas_rows = [], [], [], []
    for family in tournament.families:
        for route in ROUTES:
            candidates = []
            for q in range(field.shape[1]):
                for role in range(field.shape[2]):
                    for name in MODELS:
                        if name == "current_previous_affine" and q == 0:
                            continue
                        record, _ = tournament.evaluate(family, route, "confirmation", q, role, name)
                        scans.append(record)
                        if record["passes"]:
                            candidates.append(record)
            if candidates:
                chosen = min(candidates, key=lambda row: (row["candidate_mae"], row["checkpoint"],
                                                           row["role_index"], row["model"]))
                fresh, fresh_errors = tournament.evaluate(family, route, "fresh_confirmation",
                                                           chosen["checkpoint"], chosen["role_index"], chosen["model"])
                authorized = bool(fresh["passes"])
                lock = lock_errors = None
                if authorized:
                    lock, lock_errors = tournament.evaluate(family, route, "fresh_lockbox",
                                                             chosen["checkpoint"], chosen["role_index"], chosen["model"])
                decision = {**{key: chosen[key] for key in ("family", "route", "checkpoint", "role_index", "role", "model")},
                            "confirmation": chosen, "fresh_confirmation": fresh,
                            "fresh_authorized": authorized, "lockbox_revealed": authorized,
                            "lockbox": lock, "lockbox_pass": bool(lock and lock["passes"])}
                final_errors = lock_errors if lock_errors is not None else fresh_errors
                final_partition = "fresh_lockbox" if lock_errors is not None else "fresh_confirmation"
            else:
                decision = {"family": family, "route": route, "checkpoint": None, "role_index": None,
                            "role": None, "model": None, "confirmation": None,
                            "fresh_confirmation": None, "fresh_authorized": False,
                            "lockbox_revealed": False, "lockbox": None, "lockbox_pass": False}
                final_errors, final_partition = None, None
            decisions.append(decision)
            if final_errors is not None:
                for key, values in final_errors.items():
                    atlas_rows.append({"row": len(atlas_values), "family": family, "route": route,
                                       "partition": final_partition, "checkpoint": decision["checkpoint"],
                                       "role": decision["role"], "model": decision["model"], "error": key})
                    atlas_values.append(values.astype(np.float32))
            print(f"[tournament] {family}/{route}: q={decision['checkpoint']} role={decision['role']} "
                  f"fresh={decision['fresh_authorized']} lock={decision['lockbox_pass']}", flush=True)
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    atlas = np.stack(atlas_values) if atlas_values else np.empty((0, field.shape[-1]), np.float32)
    np.save(ATLAS, atlas)
    write_rows(ATLAS_ROWS, atlas_rows)
    mmap = getattr(field, "_mmap", None)
    if mmap is not None:
        mmap.close()
    lockbox_passed = [f"{row['family']}|{row['route']}" for row in decisions if row["lockbox_pass"]]
    checks = {
        "six_behavior_families": len(tournament.families) == 6,
        "eighteen_routes": len(decisions) == len(tournament.families) * len(ROUTES),
        "confirmation_scan_complete": len(scans) == len(tournament.families) * len(ROUTES) *
            (38 * 6 + 37 * 6),
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_authorized"] for row in decisions),
        "lockbox_never_selects": all(row["confirmation"] is not None or not row["lockbox_revealed"] for row in decisions),
        "atlas_full_coordinates": atlas.shape[-1] == 2560,
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "routes": list(ROUTES), "models": list(MODELS),
        "decisions": decisions, "lockbox_passed": lockbox_passed,
        "atlas": {"path": str(ATLAS.relative_to(ROOT)), "rows": str(ATLAS_ROWS.relative_to(ROOT)),
                  "shape": list(atlas.shape)},
        "hashes": {"atlas": file_hash(ATLAS), "rows": file_hash(ATLAS_ROWS)},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(lockbox_passed)}/18 natural family-route cells passed the complete prospective "
                              "sample-conditioned full-coordinate tournament. Passing cells are predictive and model-local, not causal gears."),
        "next_authorization": "Map full-coordinate checkpoint transport for every family and causally test only lockbox-positive middle-layer cells.",
    }
    save(OUT / "analysis/confirmation_scan.json", scans)
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
