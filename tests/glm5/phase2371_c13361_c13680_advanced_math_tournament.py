#!/usr/bin/env python3
"""Same-data tournament: S4 Fourier, tensor, OT, CMI, persistence and MDL."""
from __future__ import annotations

import itertools
import json
import math
import sys
import bisect
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
P2369 = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
OUT = RESULT / "phase2371_c13361_c13680_advanced_math_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2368 / "material/long_sentence_permutation.jsonl"
STATES = P2369 / "raw/qwen4b_long_boundary_all_layers.float16.npy"
TOKEN_FIELD = P2369 / "raw/qwen4b_long_reference_all_token_all_layers.float16.npy"
TOKEN_INDEX = P2369 / "index/long_reference_all_token_rows.jsonl"
PHASE = 2371
CAMPAIGN = "C13361-C13680"
PERMS = tuple(itertools.permutations(range(4)))
IRREPS = ("trivial", "standard", "two_dimensional", "standard_sign", "sign")
CHARACTERS = {
    "trivial": (1, {(1, 1, 1, 1): 1, (2, 1, 1): 1, (2, 2): 1, (3, 1): 1, (4,): 1}),
    "standard": (3, {(1, 1, 1, 1): 3, (2, 1, 1): 1, (2, 2): -1, (3, 1): 0, (4,): -1}),
    "two_dimensional": (2, {(1, 1, 1, 1): 2, (2, 1, 1): 0, (2, 2): 2, (3, 1): -1, (4,): 0}),
    "standard_sign": (3, {(1, 1, 1, 1): 3, (2, 1, 1): -1, (2, 2): -1, (3, 1): 0, (4,): 1}),
    "sign": (1, {(1, 1, 1, 1): 1, (2, 1, 1): -1, (2, 2): 1, (3, 1): 1, (4,): -1}),
}

sys.path.insert(0, str(TESTS))
import phase2370_c13041_c13360_pointer_group_operator as basic  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a[b[i]] for i in range(4))


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * 4
    for i, value in enumerate(p): out[value] = i
    return tuple(out)


def cycle_type(p: tuple[int, ...]) -> tuple[int, ...]:
    seen, lengths = set(), []
    for start in range(4):
        if start in seen: continue
        current, length = start, 0
        while current not in seen:
            seen.add(current); length += 1; current = p[current]
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def projectors() -> tuple[dict[str, np.ndarray], dict]:
    out = {}
    for name, (dimension, chars) in CHARACTERS.items():
        matrix = np.zeros((24, 24), dtype=np.float64)
        for pi, p in enumerate(PERMS):
            for g in PERMS:
                source = compose(inverse(g), p)
                matrix[pi, PERMS.index(source)] += dimension * chars[cycle_type(g)] / 24
        out[name] = matrix
    total = sum(out.values())
    audit = {"sum_to_identity_max_error": float(np.max(np.abs(total - np.eye(24)))),
             "idempotence_max_error": {name: float(np.max(np.abs(p @ p - p))) for name, p in out.items()},
             "cross_projector_max_error": max(float(np.max(np.abs(out[a] @ out[b]))) for a in IRREPS for b in IRREPS if a != b)}
    return out, audit


def fourier_spectra(states: np.ndarray, rows: list[dict], projectors_: dict[str, np.ndarray]):
    layer_rows, fields = [], {}
    keys0 = None; splits0 = None
    for qpoint in range(38):
        keys, _, field, _ = basic.build_field(rows, states, qpoint)
        splits = basic.split_groups(keys)
        if keys0 is None: keys0, splits0 = keys, splits
        centered = field - field.mean(axis=1, keepdims=True)
        entry = {"qpoint": qpoint, "splits": {}}
        for split, group_indices in splits.items():
            energies = {}
            for name, projector in projectors_.items():
                projected = np.einsum("ab,gbd->gad", projector, centered[group_indices], optimize=True)
                energies[name] = float(np.square(projected.astype(np.float64)).sum())
            total = sum(energies.values())
            entry["splits"][split] = {name: value / max(total, 1e-12) for name, value in energies.items()}
        layer_rows.append(entry)
        if qpoint in (26, 31, 37): fields[qpoint] = field
        print(f"[phase2371 Fourier] qpoint {qpoint}/37", flush=True)
    return layer_rows, keys0, splits0, fields


def fourier_template_tournament(field: np.ndarray, splits: dict[str, np.ndarray], projectors_: dict[str, np.ndarray]) -> dict:
    train_response = field[splits["train"]] - field[splits["train"], 0][:, None, :]
    mean = train_response.mean(0)
    components = {name: projector @ mean for name, projector in projectors_.items()}
    candidates = []
    actual_c = (field[splits["confirmation"]] - field[splits["confirmation"], 0][:, None, :])[:, 1:]
    actual_l = (field[splits["lockbox"]] - field[splits["lockbox"], 0][:, None, :])[:, 1:]
    for mask in range(1, 1 << len(IRREPS)):
        names = [name for i, name in enumerate(IRREPS) if mask & (1 << i)]
        template = sum((components[name] for name in names), np.zeros_like(mean))
        template = template - template[0]
        pc = np.broadcast_to(template[None, 1:], actual_c.shape)
        pl = np.broadcast_to(template[None, 1:], actual_l.shape)
        rc, _ = basic.response_r2(actual_c, pc); rl, _ = basic.response_r2(actual_l, pl)
        candidates.append({"irreps": names, "confirmation_response_r2": rc, "lockbox_response_r2": rl,
                           "fixed_projector_plus_template_parameters": len(names) * 23 * field.shape[-1]})
    selected = max(candidates, key=lambda x: x["confirmation_response_r2"])
    return {"selected_on_confirmation": selected, "all_irreps": next(x for x in candidates if len(x["irreps"]) == 5),
            "candidate_count": len(candidates), "boundary": "Fourier projection is exact representation analysis; template prediction remains empirical."}


def tensor_tournament(field: np.ndarray, splits: dict[str, np.ndarray]) -> dict:
    train = (field[splits["train"]] - field[splits["train"], 0][:, None, :])[:, 1:].reshape(-1, field.shape[-1]).astype(np.float32)
    confirmation = (field[splits["confirmation"]] - field[splits["confirmation"], 0][:, None, :])[:, 1:].reshape(-1, field.shape[-1]).astype(np.float32)
    lockbox = (field[splits["lockbox"]] - field[splits["lockbox"], 0][:, None, :])[:, 1:].reshape(-1, field.shape[-1]).astype(np.float32)
    max_rank = 128
    _, singular, vt = randomized_svd(train, n_components=max_rank, n_iter=5, random_state=2371)
    rng = np.random.default_rng(2371)
    random_basis, _ = np.linalg.qr(rng.standard_normal((field.shape[-1], max_rank)).astype(np.float32))
    rows = []
    for rank in (8, 32, 64, 128):
        basis = vt[:rank].T
        metrics = {}
        for name, data in (("confirmation", confirmation), ("lockbox", lockbox)):
            reconstructed = (data @ basis) @ basis.T
            random_reconstructed = (data @ random_basis[:, :rank]) @ random_basis[:, :rank].T
            denom = np.square(data - data.mean(0, keepdims=True)).sum()
            metrics[name] = {"trained_basis_reconstruction_r2": 1 - float(np.square(data - reconstructed).sum() / denom),
                             "random_basis_reconstruction_r2": 1 - float(np.square(data - random_reconstructed).sum() / denom)}
        rows.append({"rank": rank, "parameters_basis": rank * field.shape[-1], **metrics})
    return {"rows": rows, "training_singular_values_first16": singular[:16].tolist(),
            "boundary": "Projection uses each lockbox target state and measures compressibility, not target-state prediction."}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def sinkhorn(cost: np.ndarray, epsilon: float = 0.12, iterations: int = 80) -> np.ndarray:
    kernel = np.exp(-cost / epsilon)
    u = np.ones(4); v = np.ones(4)
    for _ in range(iterations):
        u = 0.25 / np.maximum(kernel @ v, 1e-12); v = 0.25 / np.maximum(kernel.T @ u, 1e-12)
    return u[:, None] * kernel * v[None, :]


def ot_tournament(token_field: np.ndarray, token_rows: list[dict]) -> dict:
    layer_rows = []
    for qpoint in range(38):
        correct_hungarian = correct_sinkhorn = total = 0
        residual_hungarian = residual_sinkhorn = 0
        for ri, meta in enumerate(token_rows):
            # The tokenizer often merges the leading space with a marker or splits the marker.
            # Recover occurrences from the exact concatenation of per-token decodes instead of
            # assuming encode(marker) is a subsequence in its whitespace context.
            starts, cursor = [], 0
            for token in meta["tokens"]:
                starts.append(cursor); cursor += len(token)
            decoded = "".join(meta["tokens"])
            occurrence = {}
            for marker in meta["markers"]:
                chars, begin = [], 0
                while (found := decoded.find(marker, begin)) >= 0:
                    chars.append(found); begin = found + len(marker)
                occurrence[marker] = [max(0, bisect.bisect_right(starts, char) - 1) for char in chars]
                if len(occurrence[marker]) < 2: raise RuntimeError((meta["case_id"], marker, occurrence[marker]))
            source_positions = [occurrence[m][0] for m in meta["markers"]]
            target_pairs = sorted((occurrence[m][-1], sid) for sid, m in enumerate(meta["markers"]))
            target_positions = [p for p, _ in target_pairs]
            expected = [sid for _, sid in target_pairs]
            raw_source = np.asarray(token_field[ri, qpoint, source_positions], dtype=np.float32)
            raw_target = np.asarray(token_field[ri, qpoint, target_positions], dtype=np.float32)
            emb_source = np.asarray(token_field[ri, 0, source_positions], dtype=np.float32)
            emb_target = np.asarray(token_field[ri, 0, target_positions], dtype=np.float32)
            for residual, source, target in ((False, raw_source, raw_target), (True, raw_source - emb_source, raw_target - emb_target)):
                cost = 1 - normalize(source) @ normalize(target).T
                rr, cc = linear_sum_assignment(cost)
                correct = sum(int(source_sid == expected[target_slot]) for source_sid, target_slot in zip(rr, cc))
                transport = sinkhorn(cost)
                soft = sum(float(transport[expected[slot], slot]) for slot in range(4)) / max(float(transport.sum()), 1e-12)
                if residual: residual_hungarian += correct; residual_sinkhorn += soft * 4
                else: correct_hungarian += correct; correct_sinkhorn += soft * 4
            total += 4
        layer_rows.append({"qpoint": qpoint, "raw_hungarian_accuracy": correct_hungarian / total,
                           "raw_sinkhorn_soft_accuracy": correct_sinkhorn / total,
                           "context_residual_hungarian_accuracy": residual_hungarian / total,
                           "context_residual_sinkhorn_soft_accuracy": residual_sinkhorn / total})
    best = max(layer_rows, key=lambda r: r["context_residual_hungarian_accuracy"])
    return {"layers": layer_rows, "best_context_residual_layer": best, "chance": 0.25,
            "boundary": "Marker-token OT is confounded by repeated lexical identity; embedding subtraction is only a partial control."}


def discretize(x: np.ndarray, bins: int = 8) -> np.ndarray:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1))[1:-1])
    return np.digitize(x, edges)


def conditional_mutual_information(z: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    nz, ny, nc = int(z.max()) + 1, int(y.max()) + 1, int(control.max()) + 1
    counts = np.bincount((z * ny + y) * nc + control, minlength=nz * ny * nc).reshape(nz, ny, nc).astype(np.float64)
    counts += 1e-9; p = counts / counts.sum()
    pc = p.sum((0, 1)); pzc = p.sum(1); pyc = p.sum(0)
    value = 0.0
    for zi in range(nz):
        for yi in range(ny):
            for ci in range(nc):
                value += p[zi, yi, ci] * math.log((p[zi, yi, ci] * pc[ci]) / (pzc[zi, ci] * pyc[yi, ci]))
    return value / math.log(2)


def cmi_tournament(states: np.ndarray, rows: list[dict], keys: list[tuple], row_index: np.ndarray, splits: dict[str, np.ndarray]) -> dict:
    # CMI(slot; state | identity) requires slot variation inside each identity stratum.
    # The operator lockbox intentionally fixes sigma, so use both frozen source orders
    # for fresh units 4-5 here; otherwise slot is deterministic given identity and CMI=0.
    cmi_groups = np.asarray([i for i, key in enumerate(keys) if key[1] >= 4])
    indices, labels = basic.labels_for(keys, row_index, rows, cmi_groups)
    x = np.asarray(states[indices, 26], dtype=np.float32)
    y, control = labels["first_source_slot"], labels["first_sentence_identity"]
    rng = np.random.default_rng(2371)
    shuffled = y.copy()
    for value in np.unique(control):
        where = np.flatnonzero(control == value); shuffled[where] = rng.permutation(shuffled[where])
    values = np.empty((2, x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        z = discretize(x[:, j])
        values[0, j] = conditional_mutual_information(z, y, control)
        values[1, j] = conditional_mutual_information(z, shuffled, control)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "derived/q26_coordinate_cmi_slot_given_identity.float32.npy", values)
    return {"qpoint": 26, "n": len(x), "actual_bits_median": float(np.median(values[0])),
            "actual_bits_max": float(values[0].max()), "shuffled_bits_median": float(np.median(values[1])),
            "shuffled_bits_max": float(values[1].max()),
            "fraction_actual_above_shuffled_99pct": float((values[0] > np.quantile(values[1], .99)).mean()),
            "boundary": "Binned coordinate CMI is a diagnostic association and has finite-sample upward bias; shuffled labels calibrate it."}


def h0_spectrum(points: np.ndarray) -> np.ndarray:
    points = normalize(points.astype(np.float64)); distance = np.maximum(0, 1 - points @ points.T)
    tree = minimum_spanning_tree(distance).toarray(); edges = np.sort(tree[tree > 0])
    return edges


def topology_tournament(field: np.ndarray, splits: dict[str, np.ndarray]) -> dict:
    train_spectra = np.stack([h0_spectrum(field[g]) for g in splits["train"]])
    lock_spectra = np.stack([h0_spectrum(field[g]) for g in splits["lockbox"]])
    rng = np.random.default_rng(2371)
    random_spectra = np.stack([h0_spectrum(rng.standard_normal(field[g].shape)) for g in splits["lockbox"]])
    coordinate_perm = rng.permutation(field.shape[-1])
    permuted_spectra = np.stack([h0_spectrum(field[g][:, coordinate_perm]) for g in splits["lockbox"]])
    train_mean, lock_mean, random_mean = train_spectra.mean(0), lock_spectra.mean(0), random_spectra.mean(0)
    return {"qpoint": 31, "bars": 23,
            "train_lockbox_spectrum_cosine": float(normalize(train_mean[None])[0] @ normalize(lock_mean[None])[0]),
            "train_lockbox_rmse": float(np.sqrt(np.square(train_mean - lock_mean).mean())),
            "lockbox_random_rmse": float(np.sqrt(np.square(lock_mean - random_mean).mean())),
            "coordinate_permutation_max_difference": float(np.max(np.abs(lock_spectra - permuted_spectra))),
            "label_permutation_difference": 0.0,
            "boundary": "H0 persistence is exactly invariant to point labels and coordinate permutation; it cannot by itself recover sentence identity or order."}


def mdl_tournament(field: np.ndarray, splits: dict[str, np.ndarray]) -> dict:
    translations, slopes, intercepts, direct = basic.fit_operators(field, splits["train"])
    complexities = {"identity": 0, "translation": 3 * 2560, "diagonal_affine": 6 * 2560, "direct_template": 23 * 2560}
    rows = []
    for method in complexities:
        actual, predicted = basic.predict_responses(field, splits["confirmation"], translations, slopes, intercepts, direct, method)
        sse = float(np.square(actual.astype(np.float64) - predicted.astype(np.float64)).sum())
        n = int(actual.size); k = complexities[method]
        bic = n * math.log(max(sse / n, 1e-30)) + k * math.log(n)
        rows.append({"method": method, "sse": sse, "observations": n, "parameters": k, "bic_nats": bic})
    selected = min(rows, key=lambda r: r["bic_nats"])
    return {"confirmation": rows, "selected": selected,
            "boundary": "BIC counts fitted coordinate coefficients but not model-forward cost; it compares these four frozen estimators only."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 同数据高等数学竞赛与Phase2370退化层纠错（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本Phase没有把附件二提出的数学名称当结论，而是在Phase2370完全相同的训练/确认/锁箱上测试：$S_4$五个不可约表示的中心投影、全坐标响应张量低秩基、句标记最优传输、逐坐标条件互信息、24排列轨道的持久$H_0$谱和包含参数惩罚的MDL/BIC。首先纠正Phase2370初次写入时的退化层：q0是相同末token的embedding，排列响应方差为0，旧程序错误报告$R^2=1$；修正后最终选择是q31逐坐标仿射，确认$R^2=0.2102$、锁箱$R^2=0.2130$。

$$
P_\lambda f(\pi)=\frac{{d_\lambda}}{{24}}\sum_{{g\in S_4}}\chi_\lambda(g)f(g^{{-1}}\pi),
\qquad I(Z;Y\mid C)=\sum_{{z,y,c}}p(z,y,c)\log_2\frac{{p(z,y\mid c)}}{{p(z\mid c)p(y\mid c)}}.
$$

**结果汇总。** 群傅里叶投影审计 `{json.dumps(result['projector_audit'], ensure_ascii=False)}`；群傅里叶模板锁箱 `{json.dumps(result['fourier_template'], ensure_ascii=False)}`；张量/HOSVD `{json.dumps(result['tensor'], ensure_ascii=False)}`；OT `{json.dumps(result['ot']['best_context_residual_layer'], ensure_ascii=False)}`；条件信息 `{json.dumps(result['cmi'], ensure_ascii=False)}`；持久拓扑 `{json.dumps(result['topology'], ensure_ascii=False)}`；MDL `{json.dumps(result['mdl'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2371_c13361_c13680_advanced_math_tournament.py`；逐层群谱、OT层表、CMI坐标数组和汇总位于 `tests/glm5/result/phase2371_c13361_c13680_advanced_math_tournament`。

**理论进展、问题硬伤与结论。** 群傅里叶投影是$S_4$实验索引上的精确代数，不证明模型内部显式实现群表示；HOSVD投影读取了锁箱目标，只能说明可压缩性；标记OT受同词复现混淆；CMI有离散化偏差；$H_0$对标签置乱严格不变，不能单独恢复句身份或顺序。只有在fresh-unit预测、简单基线、坐标对照和复杂度惩罚均通过时，某候选才升级为规律拼图，而不是“终极数学机制”。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(MATERIAL); states = np.load(STATES, mmap_mode="r")
    projectors_, projector_audit = projectors()
    spectra, keys, splits, fields = fourier_spectra(states, rows, projectors_)
    fourier_template = fourier_template_tournament(fields[31], splits, projectors_)
    tensor = tensor_tournament(fields[31], splits)
    token_field = np.load(TOKEN_FIELD, mmap_mode="r"); token_rows = read_rows(TOKEN_INDEX)
    ot = ot_tournament(token_field, token_rows)
    _, _, _, row_index = basic.build_field(rows, states, 0)
    cmi = cmi_tournament(states, rows, keys, row_index, splits)
    topology = topology_tournament(fields[31], splits)
    mdl = mdl_tournament(fields[31], splits)
    q31_train = spectra[31]["splits"]["train"]; q31_lock = spectra[31]["splits"]["lockbox"]
    spectrum_cosine = float(normalize(np.asarray(list(q31_train.values()))[None])[0] @ normalize(np.asarray(list(q31_lock.values()))[None])[0])
    result = {"phase": PHASE, "campaign": CAMPAIGN, "phase2370_correction": {"degenerate_qpoint": 0,
              "reason": "same final prompt token gives zero permutation response variance", "correct_selected_qpoint": 31,
              "correct_confirmation_r2": 0.21022252743000125, "correct_lockbox_r2": 0.21303969253213417},
              "projector_audit": projector_audit, "q31_irrep_spectrum_train_lockbox_cosine": spectrum_cosine,
              "fourier_template": fourier_template, "tensor": tensor, "ot": ot, "cmi": cmi,
              "topology": topology, "mdl": mdl,
              "conclusion_boundary": "No tested advanced method is identified with the internal language mechanism solely from this tournament."}
    save(OUT / "analysis/fourier_layer_spectra.json", spectra); save(OUT / "analysis/ot_layers.json", ot["layers"]); save(final_path, result)
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
