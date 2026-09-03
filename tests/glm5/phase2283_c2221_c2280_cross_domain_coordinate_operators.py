#!/usr/bin/env python3
"""Cross-language and cross-surface exact-coordinate operator tournament."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild"
OUT = RESULT / "phase2283_c2221_c2280_cross_domain_coordinate_operators"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2267_c1505_c1540_coordinate_model_tournament as base  # noqa: E402
import phase2281_c2101_c2160_multilingual_operator_contract as campaign  # noqa: E402
import phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild as source  # noqa: E402


PHASE = 2283
CAMPAIGN = "C2221-C2280"
GAIN_MEAN = 0.03
WIN_MEAN = 0.55
GAIN_CONTROL = 0.01
WIN_CONTROL = 0.52
ORACLE_RATIO = 1.25
EPS = 1e-8
MODEL_NAMES = ("affine", "piecewise_quartile")
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


def build_pairs(index: list[dict]) -> dict[str, dict[str, list[tuple[int, int, tuple]]]]:
    return base.build_pairs(index)


def arrays(field: np.ndarray, pairs: list[tuple[int, int, tuple]], q: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return base.arrays(field, pairs, q, r)


def fit_piecewise(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.quantile(x, [0.25, 0.5, 0.75], axis=0).astype(np.float32)
    bins = np.sum(x[:, None, :] > cuts[None, :, :], axis=1)
    means = np.empty((4, x.shape[1]), dtype=np.float32)
    fallback = y.mean(axis=0)
    for k in range(4):
        mask = bins == k
        count = mask.sum(axis=0)
        means[k] = np.where(count > 0, (y * mask).sum(axis=0) / np.maximum(count, 1), fallback)
    return cuts, means


def predict_piecewise(model: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    cuts, means = model
    bins = np.sum(x[:, None, :] > cuts[None, :, :], axis=1)
    columns = np.arange(x.shape[1])[None, :]
    return means[bins, columns]


def fit_model(name: str, x: np.ndarray, y: np.ndarray) -> Any:
    if name == "affine":
        return base.fit_affine(x, y)
    return fit_piecewise(x, y)


def predict_model(name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if name == "affine":
        return base.predict(model, x)
    return predict_piecewise(model, x)


def pair_filter(pairs: list[tuple[int, int, tuple]], position: int, value: str) -> list[tuple[int, int, tuple]]:
    return [pair for pair in pairs if pair[2][position] == value]


def route_directions(dataset: str, route: str) -> list[tuple[int, str, str]]:
    if route == "language":
        return [(0, "en", "zh"), (0, "zh", "en")]
    if dataset == "bilingual":
        return [(2, "direct", "paraphrase"), (2, "paraphrase", "direct")]
    return [(2, "direct", "paraphrase"), (2, "direct", "context_control")]


class Evaluator:
    def __init__(self, field: np.ndarray, pairs: dict, families: list[str], dataset: str):
        self.field = field
        self.pairs = pairs
        self.families = families
        self.dataset = dataset
        self.cache: dict[tuple, Any] = {}

    def data(self, family: str, partition: str, q: int, r: int, pos: int, value: str) -> tuple[np.ndarray, np.ndarray]:
        subset = pair_filter(self.pairs[family][partition], pos, value)
        x, _h1, y = arrays(self.field, subset, q, r)
        return x, y

    def fitted(self, family: str, q: int, r: int, pos: int, value: str, model_name: str,
               x_transform: str = "state") -> Any:
        key = ("fit", family, q, r, pos, value, model_name, x_transform)
        if key not in self.cache:
            x, y = self.data(family, "discovery", q, r, pos, value)
            if x_transform == "previous_response":
                if q == 0:
                    raise ValueError("embedding has no previous response")
                x = self.data(family, "discovery", q - 1, r, pos, value)[1]
            self.cache[key] = fit_model(model_name, x, y)
        return self.cache[key]

    def shared(self, q: int, r: int, pos: int, value: str, model_name: str) -> Any:
        key = ("shared", q, r, pos, value, model_name)
        if key not in self.cache:
            xs, ys = [], []
            for family in self.families:
                x, y = self.data(family, "discovery", q, r, pos, value)
                xs.append(x)
                ys.append(y)
            self.cache[key] = fit_model(model_name, np.concatenate(xs), np.concatenate(ys))
        return self.cache[key]

    def shuffled(self, family: str, q: int, r: int, pos: int, value: str, model_name: str) -> Any:
        key = ("shuffled", family, q, r, pos, value, model_name)
        if key not in self.cache:
            x, y = self.data(family, "discovery", q, r, pos, value)
            self.cache[key] = fit_model(model_name, x, np.roll(y, 1, axis=0))
        return self.cache[key]

    def evaluate(self, family: str, route: str, partition: str, q: int, r: int,
                 model_name: str) -> tuple[dict, dict[str, np.ndarray]]:
        predictions, truths = [], []
        controls: dict[str, list[np.ndarray]] = {
            "target_mean": [], "shared_source": [], "shuffled_source": [],
            "wrong_family": [], "target_oracle": [], "previous_checkpoint": [],
        }
        wrong = self.families[(self.families.index(family) + 1) % len(self.families)]
        for pos, source_value, target_value in route_directions(self.dataset, route):
            target_x, target_y = self.data(family, partition, q, r, pos, target_value)
            source_x, source_y = self.data(family, "discovery", q, r, pos, source_value)
            candidate = predict_model(model_name, self.fitted(family, q, r, pos, source_value, model_name), target_x)
            predictions.append(candidate)
            truths.append(target_y)
            target_train_x, target_train_y = self.data(family, "discovery", q, r, pos, target_value)
            controls["target_mean"].append(np.broadcast_to(target_train_y.mean(axis=0), target_y.shape))
            controls["shared_source"].append(predict_model(model_name, self.shared(q, r, pos, source_value, model_name), target_x))
            controls["shuffled_source"].append(predict_model(model_name, self.shuffled(family, q, r, pos, source_value, model_name), target_x))
            controls["wrong_family"].append(predict_model(model_name, self.fitted(wrong, q, r, pos, source_value, model_name), target_x))
            controls["target_oracle"].append(predict_model(model_name, self.fitted(family, q, r, pos, target_value, model_name), target_x))
            if q > 0:
                target_prev = self.data(family, partition, q - 1, r, pos, target_value)[1]
                prev_model = self.fitted(family, q, r, pos, source_value, "affine", "previous_response")
                controls["previous_checkpoint"].append(base.predict(prev_model, target_prev))
            else:
                controls["previous_checkpoint"].append(np.broadcast_to(target_train_y.mean(axis=0), target_y.shape))
        prediction = np.concatenate(predictions)
        truth = np.concatenate(truths)
        controls_np = {name: np.concatenate(values) for name, values in controls.items()}
        candidate_error = np.mean(np.abs(prediction - truth), axis=0)
        control_errors = {name: np.mean(np.abs(value - truth), axis=0) for name, value in controls_np.items()}
        candidate_mae = float(candidate_error.mean())
        gains = {name: float((error.mean() - candidate_mae) / max(float(error.mean()), EPS))
                 for name, error in control_errors.items()}
        wins = {name: float(np.mean(candidate_error < error)) for name, error in control_errors.items()}
        gated = ("target_mean", "shared_source", "shuffled_source", "wrong_family", "previous_checkpoint")
        oracle_mae = float(control_errors["target_oracle"].mean())
        record = {
            "family": family, "dataset": self.dataset, "route": route, "partition": partition,
            "checkpoint": q, "role_index": r, "role": ROLES[r], "model": model_name,
            "pairs": int(len(truth)), "mae": candidate_mae,
            "control_mae": {name: float(value.mean()) for name, value in control_errors.items()},
            "gain_over_controls": gains, "coordinate_win_over_controls": wins,
            "oracle_ratio": candidate_mae / max(oracle_mae, EPS),
            "minimum_gated_gain": min(gains[name] for name in gated),
            "minimum_gated_win": min(wins[name] for name in gated),
            "wrong_family": wrong,
        }
        record["passes"] = bool(
            gains["target_mean"] >= GAIN_MEAN and wins["target_mean"] >= WIN_MEAN and
            min(gains[name] for name in gated if name != "target_mean") >= GAIN_CONTROL and
            min(wins[name] for name in gated if name != "target_mean") >= WIN_CONTROL and
            record["oracle_ratio"] <= ORACLE_RATIO
        )
        passport = {"candidate_abs_error": candidate_error.astype(np.float32)}
        passport.update({f"{name}_abs_error": value.astype(np.float32) for name, value in control_errors.items()})
        return record, passport


def decide(evaluator: Evaluator, family: str, route: str) -> tuple[dict, list[dict]]:
    scan = []
    ranked = []
    for q in range(evaluator.field.shape[1]):
        for r in range(evaluator.field.shape[2]):
            for model_name in MODEL_NAMES:
                record, _passport = evaluator.evaluate(family, route, "confirmation", q, r, model_name)
                scan.append(record)
                ranked.append((record["passes"], record["minimum_gated_gain"],
                               record["minimum_gated_win"], -record["oracle_ratio"],
                               -record["mae"], q, r, model_name, record))
    selected = max(ranked)
    q, r, model_name = selected[5], selected[6], selected[7]
    confirmation = selected[8]
    fresh, fresh_passport = evaluator.evaluate(family, route, "fresh_confirmation", q, r, model_name)
    fresh_pass = bool(confirmation["passes"] and fresh["passes"])
    lockbox = None
    final_passport = fresh_passport
    if fresh_pass:
        lockbox, final_passport = evaluator.evaluate(family, route, "fresh_lockbox", q, r, model_name)
    decision = {
        "family": family, "dataset": evaluator.dataset, "route": route,
        "checkpoint": q, "role_index": r, "role": ROLES[r], "model": model_name,
        "confirmation": confirmation, "fresh_confirmation": fresh,
        "fresh_authorized": fresh_pass, "lockbox_revealed": fresh_pass,
        "lockbox": lockbox, "lockbox_pass": bool(lockbox and lockbox["passes"]),
        "passport_partition": "fresh_lockbox" if lockbox else "fresh_confirmation",
        "passport": final_passport,
    }
    return decision, scan


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summaries = [{k: row[k] for k in ("family", "dataset", "route", "checkpoint", "role", "model",
                                       "fresh_authorized", "lockbox_revealed", "lockbox_pass")}
                 for row in result["decisions"]]
    text = rf"""

## Phase {PHASE}: 跨语言与跨表面逐坐标算子前瞻竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期读取 Phase2282 的完整六角色原场，不运行模型。中英十构式分别测试英文 discovery 学到的逐坐标状态算子能否预测中文响应、中文能否预测英文；同时测试 direct 能否预测 paraphrase、paraphrase 能否预测 direct。三个复杂英文构式测试 direct 对 paraphrase 与无关上下文表面的迁移。每个构式、路线、38 检查点、六角色和仿射/四分位分段两种基础函数都在 confirmation 扫描；只冻结一个候选，再依次揭示 fresh confirmation 和 fresh lockbox。实际用例覆盖中英受事绑定、位置、时间、量词，以及英文条件后件、合取和分类链。

**数学公式与门槛。** 从源域 $a$ 学习同坐标算子并作用于目标域 $b$ 的基态：

$$
\widehat R^b_j=g^a_j(H^{{0,b}}_j),\qquad
g^a_j\in\left\{{a_jx+b_j,\ \mu_{{j,k(x)}}\right\}}.
$$

候选必须优于目标域 discovery 均值、跨构式共享源域模型、源域错配、循环错族和上一检查点控制；相对目标均值要求增益至少 `0.03`、逐坐标胜率至少 `0.55`，相对其余控制至少 `0.01/0.52`，并且 MAE 不得超过目标域自己拟合模型的 `1.25` 倍。所有误差由完整 2560 坐标计算，不进行 Top-K、PCA 或余弦筛选。

**结果汇总与门槛。** 裁决摘要 `{json.dumps(summaries, ensure_ascii=False)}`；跨语言通过 `{json.dumps(result['cross_language_lockbox'], ensure_ascii=False)}`；跨表面通过 `{json.dumps(result['cross_surface_lockbox'], ensure_ascii=False)}`；全坐标护照 `{json.dumps(result['atlas'], ensure_ascii=False)}`；哈希和检查 `{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2283_c2221_c2280_cross_domain_coordinate_operators.py`；结果 `tests/glm5/result/phase2283_c2221_c2280_cross_domain_coordinate_operators`；逐格 confirmation 扫描、冻结决策和每条路线的完整坐标误差护照均已保存。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 通过只表示同一 Qwen3-4B 物理坐标函数在冻结平行材料之间有预测迁移；它不证明中英共享语义神经元、翻译模块或因果齿轮。上一检查点是很强的一般传播控制；目标域 oracle 只作可达上限，不是机制真值。人工平行模板、角色末 token、float16、同一模型和多格 confirmation 搜索仍是硬伤。理论主体与 RDC 不改名，没有引入新数学。下一阶段无论路线是否通过，都会继续计算完整二阶析因交互图；只有中层通过路线才获得多密度因果锚点资格。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    source_final = load(SOURCE / "analysis/final.json")
    if not source_final["all_checks_passed"]:
        raise RuntimeError("Phase2282 source failed")
    datasets = {
        "bilingual": {
            "field": source.BI_FIELD, "index": source.BI_INDEX,
            "families": list(campaign.BILINGUAL_FAMILIES), "routes": ("language", "surface"),
        },
        "complex": {
            "field": source.COMPLEX_FIELD, "index": source.COMPLEX_INDEX,
            "families": list(campaign.COMPLEX_FAMILIES), "routes": ("surface",),
        },
    }
    decisions, scans, passport_values, passport_rows = [], [], [], []
    try:
        for dataset, config in datasets.items():
            field = np.load(config["field"], mmap_mode="r")
            pairs = build_pairs(read_rows(config["index"]))
            evaluator = Evaluator(field, pairs, config["families"], dataset)
            for family in config["families"]:
                for route in config["routes"]:
                    decision, scan = decide(evaluator, family, route)
                    scans.extend(scan)
                    passport = decision.pop("passport")
                    for metric, values in passport.items():
                        passport_rows.append({
                            "row": len(passport_values), "family": family, "dataset": dataset,
                            "route": route, "checkpoint": decision["checkpoint"],
                            "role": decision["role"], "partition": decision["passport_partition"],
                            "metric": metric,
                        })
                        passport_values.append(values)
                    decisions.append(decision)
                    print(f"[operator] {dataset}/{family}/{route}: q={decision['checkpoint']} "
                          f"role={decision['role']} fresh={decision['fresh_authorized']} "
                          f"lock={decision['lockbox_pass']}", flush=True)
            close_mmap(field)
    finally:
        try:
            close_mmap(field)
        except UnboundLocalError:
            pass
    scan_path = OUT / "analysis/confirmation_scan.jsonl"
    decision_path = OUT / "analysis/decisions.jsonl"
    write_rows(scan_path, scans)
    serializable_decisions = decisions
    write_rows(decision_path, serializable_decisions)
    atlas = np.stack(passport_values).astype(np.float32)
    atlas_path = OUT / "atlas/cross_domain_coordinate_operator_passport.float32.npy"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    rows_path = OUT / "atlas/cross_domain_coordinate_operator_passport.rows.jsonl"
    write_rows(rows_path, passport_rows)
    cross_language = [row["family"] for row in decisions if row["route"] == "language" and row["lockbox_pass"]]
    cross_surface = [f"{row['dataset']}:{row['family']}" for row in decisions
                     if row["route"] == "surface" and row["lockbox_pass"]]
    checks = {
        "decision_count": len(decisions) == 10 * 2 + 3,
        "scan_complete": len(scans) == (10 * 2 + 3) * 38 * 6 * len(MODEL_NAMES),
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_authorized"] for row in decisions),
        "atlas_all_coordinates": atlas.shape[1] == 2560,
        "atlas_rows_match": atlas.shape[0] == len(passport_rows),
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    hashes = {"scan": file_hash(scan_path), "decisions": file_hash(decision_path),
              "atlas": file_hash(atlas_path), "atlas_rows": file_hash(rows_path)}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "gates": {"gain_mean": GAIN_MEAN, "win_mean": WIN_MEAN,
                  "gain_control": GAIN_CONTROL, "win_control": WIN_CONTROL,
                  "oracle_ratio_maximum": ORACLE_RATIO},
        "decisions": decisions, "cross_language_lockbox": cross_language,
        "cross_surface_lockbox": cross_surface,
        "atlas": {"path": str(atlas_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)),
                  "shape": list(atlas.shape), "all_coordinates": True},
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(cross_language)}/10 bilingual families passed bidirectional cross-language "
                              f"coordinate-operator lockbox and {len(cross_surface)}/13 families passed cross-surface "
                              "lockbox under mean, shared, shuffled, wrong-family, previous-checkpoint, and target-oracle controls; results remain observational and model-local."),
        "next_authorization": "Compute full-coordinate factorial interaction maps for all families; authorize causal dose-response only for at most two middle-layer qualified anchors.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
