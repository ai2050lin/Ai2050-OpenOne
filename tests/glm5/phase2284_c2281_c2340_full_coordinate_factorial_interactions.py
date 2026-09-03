#!/usr/bin/env python3
"""Full-coordinate state-by-language and state-by-surface factorial maps."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild"
OPERATOR = RESULT / "phase2283_c2221_c2280_cross_domain_coordinate_operators"
OUT = RESULT / "phase2284_c2281_c2340_full_coordinate_factorial_interactions"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2267_c1505_c1540_coordinate_model_tournament as base  # noqa: E402
import phase2281_c2101_c2160_multilingual_operator_contract as campaign  # noqa: E402
import phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild as source  # noqa: E402


PHASE = 2284
CAMPAIGN = "C2281-C2340"
GAIN_GATE = 0.03
WIN_GATE = 0.55
EPS = 1e-8
ROLES = contract.ROLES


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def lookup(index: list[dict]) -> dict[tuple, int]:
    return {(row["family"], row["language"], int(row["unit"]), row["surface"], int(row["state"])):
            int(row["hidden_index"]) for row in index}


def samples(field: np.ndarray, table: dict, family: str, partition: str, route: str,
            q: int, r: int, dataset: str, shuffled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    units = list(range(0, 12) if partition == "discovery" else
                 range(12, 16) if partition == "confirmation" else
                 range(16, 24) if partition == "fresh_confirmation" else range(24, 32))
    values, mains = [], []
    if route == "state_x_language":
        surfaces = ("direct", "paraphrase")
        for unit in units:
            other = units[(units.index(unit) + 1) % len(units)] if shuffled else unit
            for surface in surfaces:
                en0 = table[(family, "en", other, surface, 0)]
                en1 = table[(family, "en", other, surface, 1)]
                zh0 = table[(family, "zh", unit, surface, 0)]
                zh1 = table[(family, "zh", unit, surface, 1)]
                en_response = np.asarray(field[en1, q, r], dtype=np.float32) - np.asarray(field[en0, q, r], dtype=np.float32)
                zh_response = np.asarray(field[zh1, q, r], dtype=np.float32) - np.asarray(field[zh0, q, r], dtype=np.float32)
                values.append(zh_response - en_response)
                mains.append(0.5 * (np.abs(zh_response) + np.abs(en_response)))
    else:
        languages = ("en", "zh") if dataset == "bilingual" else ("en",)
        targets = ("paraphrase",) if dataset == "bilingual" else ("paraphrase", "context_control")
        for unit in units:
            other = units[(units.index(unit) + 1) % len(units)] if shuffled else unit
            for language in languages:
                for target in targets:
                    d0 = table[(family, language, other, "direct", 0)]
                    d1 = table[(family, language, other, "direct", 1)]
                    t0 = table[(family, language, unit, target, 0)]
                    t1 = table[(family, language, unit, target, 1)]
                    direct_response = np.asarray(field[d1, q, r], dtype=np.float32) - np.asarray(field[d0, q, r], dtype=np.float32)
                    target_response = np.asarray(field[t1, q, r], dtype=np.float32) - np.asarray(field[t0, q, r], dtype=np.float32)
                    values.append(target_response - direct_response)
                    mains.append(0.5 * (np.abs(target_response) + np.abs(direct_response)))
    return np.stack(values), np.stack(mains)


class FactorialEvaluator:
    def __init__(self, field: np.ndarray, index: list[dict], families: list[str], dataset: str):
        self.field = field
        self.table = lookup(index)
        self.families = families
        self.dataset = dataset
        self.cache: dict[tuple, np.ndarray] = {}

    def mean(self, family: str, route: str, q: int, r: int, shuffled: bool = False) -> np.ndarray:
        key = (family, route, q, r, shuffled)
        if key not in self.cache:
            self.cache[key] = samples(self.field, self.table, family, "discovery", route, q, r,
                                      self.dataset, shuffled)[0].mean(axis=0)
        return self.cache[key]

    def shared(self, route: str, q: int, r: int) -> np.ndarray:
        key = ("shared", route, q, r)
        if key not in self.cache:
            self.cache[key] = np.mean([self.mean(family, route, q, r) for family in self.families], axis=0)
        return self.cache[key]

    def evaluate(self, family: str, route: str, partition: str, q: int, r: int) -> tuple[dict, dict[str, np.ndarray]]:
        truth, main = samples(self.field, self.table, family, partition, route, q, r, self.dataset)
        candidate = np.broadcast_to(self.mean(family, route, q, r), truth.shape)
        wrong = self.families[(self.families.index(family) + 1) % len(self.families)]
        controls = {
            "zero": np.zeros_like(truth),
            "shared": np.broadcast_to(self.shared(route, q, r), truth.shape),
            "wrong_family": np.broadcast_to(self.mean(wrong, route, q, r), truth.shape),
            "shuffled_units": np.broadcast_to(self.mean(family, route, q, r, True), truth.shape),
        }
        if q > 0:
            train_prev = samples(self.field, self.table, family, "discovery", route, q - 1, r, self.dataset)[0]
            train_now = samples(self.field, self.table, family, "discovery", route, q, r, self.dataset)[0]
            coeff = base.fit_affine(train_prev, train_now)
            test_prev = samples(self.field, self.table, family, partition, route, q - 1, r, self.dataset)[0]
            controls["previous_checkpoint"] = base.predict(coeff, test_prev)
        else:
            controls["previous_checkpoint"] = np.zeros_like(truth)
        candidate_error = np.mean(np.abs(candidate - truth), axis=0)
        control_errors = {name: np.mean(np.abs(value - truth), axis=0) for name, value in controls.items()}
        candidate_mae = float(candidate_error.mean())
        gains = {name: float((error.mean() - candidate_mae) / max(float(error.mean()), EPS))
                 for name, error in control_errors.items()}
        wins = {name: float(np.mean(candidate_error < error)) for name, error in control_errors.items()}
        mean_abs_interaction = np.mean(np.abs(truth), axis=0)
        mean_abs_main = np.mean(main, axis=0)
        record = {
            "family": family, "dataset": self.dataset, "route": route, "partition": partition,
            "checkpoint": q, "role_index": r, "role": ROLES[r], "samples": len(truth),
            "mae": candidate_mae, "gain_over_controls": gains,
            "coordinate_win_over_controls": wins,
            "minimum_gain": min(gains.values()), "minimum_win": min(wins.values()),
            "interaction_to_main_ratio": float(mean_abs_interaction.mean() / max(float(mean_abs_main.mean()), EPS)),
            "wrong_family": wrong,
        }
        record["passes"] = bool(record["minimum_gain"] >= GAIN_GATE and record["minimum_win"] >= WIN_GATE)
        passport = {
            "discovery_mean_interaction": self.mean(family, route, q, r).astype(np.float32),
            "heldout_mean_interaction": truth.mean(axis=0).astype(np.float32),
            "heldout_mean_abs_interaction": mean_abs_interaction.astype(np.float32),
            "heldout_mean_abs_main": mean_abs_main.astype(np.float32),
            "candidate_abs_error": candidate_error.astype(np.float32),
        }
        passport.update({f"{name}_abs_error": value.astype(np.float32) for name, value in control_errors.items()})
        return record, passport


def decide(evaluator: FactorialEvaluator, family: str, route: str) -> tuple[dict, list[dict]]:
    scan, ranked = [], []
    for q in range(evaluator.field.shape[1]):
        for r in range(evaluator.field.shape[2]):
            record, _ = evaluator.evaluate(family, route, "confirmation", q, r)
            scan.append(record)
            ranked.append((record["passes"], record["minimum_gain"], record["minimum_win"],
                           -record["mae"], q, r, record))
    selected = max(ranked)
    q, r, confirmation = selected[4], selected[5], selected[6]
    fresh, passport = evaluator.evaluate(family, route, "fresh_confirmation", q, r)
    authorized = bool(confirmation["passes"] and fresh["passes"])
    lockbox = None
    if authorized:
        lockbox, passport = evaluator.evaluate(family, route, "fresh_lockbox", q, r)
    return {
        "family": family, "dataset": evaluator.dataset, "route": route,
        "checkpoint": q, "role_index": r, "role": ROLES[r],
        "confirmation": confirmation, "fresh_confirmation": fresh,
        "fresh_authorized": authorized, "lockbox_revealed": authorized,
        "lockbox": lockbox, "lockbox_pass": bool(lockbox and lockbox["passes"]),
        "passport_partition": "fresh_lockbox" if lockbox else "fresh_confirmation",
        "passport": passport,
    }, scan


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = [{k: row[k] for k in ("family", "dataset", "route", "checkpoint", "role",
                                     "fresh_authorized", "lockbox_pass")}
               for row in result["decisions"]]
    text = rf"""

## Phase {PHASE}: 状态乘语言与状态乘表面的全坐标析因图（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期继续读取 Phase2282 原场，不运行模型。对中英十构式计算“真假状态效应在中文与英文之间的差”，并计算“真假状态效应在 direct 与 paraphrase 之间的差”；三个复杂英文构式还加入 context control。它回答的不是两个整向量像不像，而是每个角色、检查点和 2560 个坐标上，状态操作是否会被语言或表面条件系统调制。用例覆盖中英受事、位置、时间、量词，以及英文条件、合取与分类链。

**数学公式与门槛。** 二阶 2×2 析因残差为：

$$
I^{{s\times c}}_{{i,q,r,j}}=
H_{{1,1}}-H_{{0,1}}-H_{{1,0}}+H_{{0,0}},
$$

其中条件 $c$ 分别是语言或表面。discovery 的本构式逐坐标平均交互必须在 confirmation、fresh confirmation 和 lockbox 中同时优于零交互、跨构式共享、循环错族、错配单元和上一检查点五类控制，全部 MAE 增益不低于 `0.03`、逐坐标胜率不低于 `0.55`。所有坐标进入误差，不进行 Top-K、PCA 或余弦筛选。

**结果汇总与门槛。** 裁决 `{json.dumps(summary, ensure_ascii=False)}`；正式通过 `{json.dumps(result['lockbox_passed'], ensure_ascii=False)}`；交互/主效应比 `{json.dumps(result['interaction_to_main'], ensure_ascii=False)}`；图谱 `{json.dumps(result['atlas'], ensure_ascii=False)}`；哈希与检查 `{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2284_c2281_c2340_full_coordinate_factorial_interactions.py`；结果 `tests/glm5/result/phase2284_c2281_c2340_full_coordinate_factorial_interactions`；confirmation 全格扫描和完整逐坐标析因护照均保存。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 稳定二阶残差表示状态与语言/表面的预测性交互，不能直接称为算子不交换、流形曲率、高阶语义超边或因果协同。人工平行材料使语言差异同时包含 token、长度和模板差异；上一检查点预测可能吸收普通传播；均值交互可能掩盖替代实现。理论主体与 RDC 不改名。下一阶段只有预注册的中层因果锚点且通过 Phase2283 才可干预；否则因果分支记为未授权，但跨规模和图谱路线继续。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    if not load(OPERATOR / "analysis/final.json")["all_checks_passed"]:
        raise RuntimeError("Phase2283 operator result failed audit")
    configs = {
        "bilingual": {"field": source.BI_FIELD, "index": source.BI_INDEX,
                       "families": list(campaign.BILINGUAL_FAMILIES),
                       "routes": ("state_x_language", "state_x_surface")},
        "complex": {"field": source.COMPLEX_FIELD, "index": source.COMPLEX_INDEX,
                     "families": list(campaign.COMPLEX_FAMILIES),
                     "routes": ("state_x_surface",)},
    }
    decisions, scans, passport_values, passport_rows = [], [], [], []
    field = None
    try:
        for dataset, config in configs.items():
            field = np.load(config["field"], mmap_mode="r")
            evaluator = FactorialEvaluator(field, read_rows(config["index"]), config["families"], dataset)
            for family in config["families"]:
                for route in config["routes"]:
                    decision, scan = decide(evaluator, family, route)
                    scans.extend(scan)
                    passport = decision.pop("passport")
                    for metric, values in passport.items():
                        passport_rows.append({"row": len(passport_values), "family": family,
                                              "dataset": dataset, "route": route,
                                              "checkpoint": decision["checkpoint"],
                                              "role": decision["role"],
                                              "partition": decision["passport_partition"],
                                              "metric": metric})
                        passport_values.append(values)
                    decisions.append(decision)
                    print(f"[factorial] {dataset}/{family}/{route}: q={decision['checkpoint']} "
                          f"role={decision['role']} lock={decision['lockbox_pass']}", flush=True)
            close_mmap(field)
            field = None
    finally:
        if field is not None:
            close_mmap(field)
    scan_path = OUT / "analysis/confirmation_scan.jsonl"
    decisions_path = OUT / "analysis/decisions.jsonl"
    write_rows(scan_path, scans)
    write_rows(decisions_path, decisions)
    atlas = np.stack(passport_values).astype(np.float32)
    atlas_path = OUT / "atlas/full_coordinate_factorial_passport.float32.npy"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    rows_path = OUT / "atlas/full_coordinate_factorial_passport.rows.jsonl"
    write_rows(rows_path, passport_rows)
    passed = [f"{row['dataset']}:{row['family']}:{row['route']}" for row in decisions if row["lockbox_pass"]]
    interaction_to_main = {f"{row['dataset']}:{row['family']}:{row['route']}":
                           (row["lockbox"] or row["fresh_confirmation"])["interaction_to_main_ratio"]
                           for row in decisions}
    checks = {
        "decision_count": len(decisions) == 23,
        "scan_complete": len(scans) == 23 * 38 * 6,
        "ordered_reveal": all(row["fresh_authorized"] == row["lockbox_revealed"] for row in decisions),
        "atlas_all_coordinates": atlas.shape[1] == 2560,
        "atlas_rows_match": atlas.shape[0] == len(passport_rows),
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    hashes = {"scan": file_hash(scan_path), "decisions": file_hash(decisions_path),
              "atlas": file_hash(atlas_path), "atlas_rows": file_hash(rows_path)}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "gates": {"gain": GAIN_GATE, "coordinate_win": WIN_GATE},
        "decisions": decisions, "lockbox_passed": passed,
        "interaction_to_main": interaction_to_main,
        "atlas": {"path": str(atlas_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)),
                  "shape": list(atlas.shape), "all_coordinates": True},
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(passed)}/23 state-by-language or state-by-surface factorial mean maps "
                              "survived zero, shared, wrong-family, shuffled-unit, and previous-checkpoint controls; "
                              "these are observational interaction maps, not causal hyperedges or curvature."),
        "next_authorization": "Adjudicate the preregistered scale-controlled intervention branch, then run conditional cross-scale replication for Phase2283 positives.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
