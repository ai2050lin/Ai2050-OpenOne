#!/usr/bin/env python3
"""Publish semantic-selection Walsh fields and audit whether partner-recombination must continue."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2501 = RESULT / "phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field"
P2502 = RESULT / "phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox"
P2503 = RESULT / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield"
P2504 = RESULT / "phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox"
P2505 = RESULT / "phase2505_c68865_c70016_semantic_selection_autonomous_output_geometry"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2506_c70017_c71040_semantic_walsh_visualization_stage_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2506, "C70017-C71040", 2560
SOURCE = "phase2506_semantic_selection_walsh"

sys.path.insert(0, str(ROOT / "tests/glm5"))
import model_utils  # noqa: E402
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa: E402
import phase2505_c68865_c70016_semantic_selection_autonomous_output_geometry as trajectory  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def row(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (DIM,) or not np.isfinite(values).all():
        raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **meta, "values": [float(v) for v in values]}


def publish() -> dict:
    f2501, f2503, f2504, f2505 = (load_json(P2501 / "analysis/final.json"), load_json(P2503 / "analysis/final.json"),
                                  load_json(P2504 / "analysis/final.json"), load_json(P2505 / "analysis/final.json"))
    qpoint = int(f2504["contract"]["qpoint"])
    pair_ids = f2504["contract"]["pair_ids"]
    pair_names = [" / ".join(value) for value in f2504["contract"]["pairs"]]
    events = f2501["collection"]["events"]
    added = []

    # Event-level raw q0/q30 values for one confirmation row.
    conf_field = np.load(f2501["collection"]["event_field"], mmap_mode="r")
    conf_rows = walsh.read_jsonl(Path(f2501["collection"]["event_index"]))
    representative = next(r for r in conf_rows if r["unit"] == 21 and r["pair_id"] == pair_ids[0]
                          and r["language"] == "en" and r["surface"] == 0
                          and r["meaning_swap"] == 0 and r["query_marker"] == 0)
    for event_index, event in enumerate(events):
        for qp, kind in ((0, "semantic_selection_event_embedding"), (qpoint, "semantic_selection_event_hiddenstate")):
            added.append(row(conf_field[representative["model_row"], event_index, qp],
                             f"unit21 representative {event} q{qp} raw", kind,
                             phase=2501, unit=21, event=event, layer=qp, pair=pair_names[0],
                             language="en", surface=0, meaning_swap=0, query_marker=0))

    # Corrected pair-relative interaction and all three Walsh terms.
    interaction = np.load(f2504["fields"]["interaction"]["path"], mmap_mode="r")
    terms = np.load(f2504["fields"]["walsh_terms"]["path"], mmap_mode="r")
    term_names = ("meaning_mapping_main", "query_marker_main", "meaning_by_query_interaction")
    for ui, unit in enumerate((21, 23)):
        for event_index, event in enumerate(events):
            for pi, pair_name in enumerate(pair_names):
                added.append(row(interaction[ui, event_index, pi].mean(axis=(0, 1)),
                                 f"unit{unit} {pair_name} {event} q{qpoint} Walsh interaction",
                                 "behavior_necessary_relation_selection_interaction", phase=2504,
                                 unit=unit, pair=pair_name, event=event, layer=qpoint,
                                 averaging="two languages and four surfaces"))
    for ti, term_name in enumerate(term_names):
        for pi, pair_name in enumerate(pair_names):
            added.append(row(terms[1, events.index("answer_boundary"), ti, pi].mean(axis=(0, 1)),
                             f"unit23 {pair_name} answer q{qpoint} {term_name}",
                             "semantic_selection_walsh_term", phase=2504, unit=23, pair=pair_name,
                             event="answer_boundary", layer=qpoint, walsh_term=term_name,
                             averaging="two languages and four surfaces"))

    # A complete prompt's token-by-token q0 embedding and q30 HiddenState.
    token_field = np.load(f2503["collection"]["alltoken_field"], mmap_mode="r")
    token_rows = walsh.read_jsonl(Path(f2503["collection"]["alltoken_index"]))
    token_rep = next(r for r in token_rows if r["pair_id"] == pair_ids[0] and r["language"] == "en" and r["meaning_swap"] == 0)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    tokens = tokenizer.convert_ids_to_tokens(token_rep["prompt_ids"])
    lo, hi = token_rep["offset"]
    for local_index, token in enumerate(tokens):
        for qp, kind in ((0, "semantic_selection_token_embedding"), (qpoint, "semantic_selection_token_hiddenstate")):
            added.append(row(token_field[lo + local_index, qp],
                             f"unit23 token {local_index:03d} {token!r} q{qp}", kind,
                             phase=2503, unit=23, pair=pair_names[0], language="en", surface=0,
                             meaning_swap=0, query_marker=0, token_index=local_index,
                             token_id=int(token_rep["prompt_ids"][local_index]), token=str(token), layer=qp))

    # Mean relation-selection interactions at actual autonomous boundary/first/answer events.
    traj_field = np.load(f2505["collection"]["field"], mmap_mode="r")
    traj_index = trajectory.read_jsonl(Path(f2505["collection"]["index"]))
    for event in ("boundary", "first", "answer"):
        groups = trajectory.group_interactions(traj_field, traj_index, qpoint, event, False)
        for pair_id, pair_name in zip(pair_ids, pair_names):
            values = [value for key, value in groups.items() if key[0] == pair_id]
            added.append(row(np.mean(values, axis=0), f"unit23 {pair_name} autonomous {event} q{qpoint} interaction",
                             "semantic_selection_autonomous_interaction", phase=2505, unit=23, pair=pair_name,
                             event=event, layer=qpoint, averaging=f"{len(values)} language-surface fourcell groups"))

    payload = load_json(ASSET)
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [existing for existing in qwen["rows"] if existing.get("source") != SOURCE] + added
    confirmation_energy = np.square(np.asarray(interaction[0], dtype=np.float64)).sum(axis=(0, 1, 2, 3))
    qwen.setdefault("coordinate_orders", {})["semantic_walsh"] = [int(v) for v in np.argsort(-confirmation_energy)]
    matrix = np.stack([np.asarray(existing["values"], dtype=np.float32) for existing in qwen["rows"]])
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape)
    qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE
    payload["campaign"] = "C39761-C71040"
    payload["summary"]["phase2504_behavior_necessary_semantic_selection_walsh"] = True
    payload["summary"]["phase2505_output_sequence_interaction_all_24_positive"] = True
    payload["summary"]["phase2505_causal_mediator_identified"] = False
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = ("Phase2504-2505 isolate a behavior-necessary relation-selection Walsh interaction with an exact-zero "
                "causal-prefix control and positive candidate-sequence score interactions, but only three pair-relative "
                "contrasts survive the fresh lockbox; no pure semantic code or causal gear is identified.")
    if sentence not in payload["claim_boundary"]:
        payload["claim_boundary"] = payload["claim_boundary"].rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "source": SOURCE, "rows_added": len(added),
            "qwen_shape": list(matrix.shape), "binary": str(binary), "binary_sha256": qwen["binary_sha256"],
            "semantic_order_coordinates": len(confirmation_energy), "alltoken_rows_displayed": len(tokens) * 2,
            "json_bytes": ASSET.stat().st_size}


def cleanup_and_retention() -> dict:
    f2501, f2503, f2504, f2505 = (load_json(P2501 / "analysis/final.json"), load_json(P2503 / "analysis/final.json"),
                                  load_json(P2504 / "analysis/final.json"), load_json(P2505 / "analysis/final.json"))
    delete_paths = [Path(f2501["collection"]["alltoken_field"]),
                    Path(load_json(P2502 / "analysis/final.json")["fields"]["interaction"]["path"]),
                    Path(load_json(P2502 / "analysis/final.json")["fields"]["walsh_terms"]["path"])]
    manifest_path = OUT / "analysis/retention_cleanup_manifest.json"
    previous = load_json(manifest_path) if manifest_path.exists() else {"deleted": []}
    deleted = list(previous.get("deleted", []))
    result_root = RESULT.resolve()
    for path in delete_paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(f"Refuse cleanup outside result root: {resolved}")
        if resolved.exists():
            size = resolved.stat().st_size
            file_hash = digest(resolved)
            resolved.unlink()
            if not any(item["path"] == str(resolved) for item in deleted):
                deleted.append({"path": str(resolved), "bytes": size, "sha256_before_delete": file_hash,
                                "recovery": "re-run the owning phase"})
    kept_paths = [Path(f2501["collection"]["event_field"]), Path(f2503["collection"]["event_field"]),
                  Path(f2503["collection"]["alltoken_field"]), Path(f2504["fields"]["interaction"]["path"]),
                  Path(f2504["fields"]["walsh_terms"]["path"]), Path(f2505["collection"]["field"])]
    kept = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
             "retention": "retained: unique full-coordinate source with parameter-level client slices"} for path in kept_paths]
    manifest = {"deleted": deleted, "kept": kept, "deleted_bytes": sum(v["bytes"] for v in deleted),
                "kept_bytes": sum(v["bytes"] for v in kept)}
    save(manifest_path, manifest)
    return manifest


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 语义选择四格的逐坐标客户端发布、留存清理与自动续研审计（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 在现有c42641研究热力图新增`semantic_walsh`顺序，该顺序只用unit21 confirmation的六事件×三pair×中英文×四surface交互能量冻结，保留完整2560物理坐标顺序作为并列选择。客户端新增：unit21代表样本六事件的q0词嵌入与q30 HiddenState；unit21/unit23三pair×六事件的四格交互；unit23答案处三个Walsh项；一个unit23完整prompt逐token的q0 Embedding和q30 HiddenState；自主boundary/first/answer交互。每行保留unit、event、pair、token ID、token index、layer与平均范围。

$$\Pi_{{Walsh}}=\operatorname{{argsort}}_i\left[-\sum I_{{u21,e,p,\lambda,s,i}}^2\right],\qquad |\Pi_{{Walsh}}|=2560.$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存/清理 `{json.dumps(result['retention'], ensure_ascii=False)}`；阶段裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2506_c70017_c71040_semantic_walsh_visualization_stage_audit.py`；`frontend/src/components/app/ResearchHeatmapRoute.jsx`新增语义选择顺序；c42641 JSON/float32矩阵、生产build和留存清单位于对应目录。

**理论进展。** 参数级客户端现在可直接比较物理坐标、旧family纹理、nonce条件纹理与行为必要四格交互。重要正拼图是：三pair在全新实体/marker锁箱的answer处跨语言/跨surface为正，24/24候选序列输出交互方向正确。重要负拼图是：跨unit的answer-boundary raw pair身份未超过wrong q95，且自主boundary到answer纹理几乎不保持；因此不能画成一条固定语义向量搬运链。

**问题硬伤、清理与结论。** 删除的是首次无效unit22锁箱的两个派生场，以及未发布且由无效unit22构成的旧代表全token场；均有删除前hash并可重跑恢复。有效全qpoint事件场、修复锁箱、三Walsh项与自主轨迹全部留存。当前仍只有三个原配对，无法判断观察到的是family编码还是pair-relative判别；这与本阶段第一目标完全相同，故按用户合同自动进入新配对伙伴复核，不在可视化总结处停止。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    asset = publish()
    retention = cleanup_and_retention()
    dist = ROOT / "frontend/dist/index.html"
    source = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    frontend = {"dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
                "semantic_walsh_order_control": "semantic_walsh" in source}
    sequence = []
    for phase in range(2499, 2506):
        candidates = list(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
        sequence.append(len(candidates) == 1 and load_json(candidates[0])["all_checks_passed"])
    checks = {"rows_added": asset["rows_added"] > 200, "full_coordinate_order": asset["semantic_order_coordinates"] == 2560,
              "parameter_level_alltoken": asset["alltoken_rows_displayed"] > 100,
              "binary_hash": len(asset["binary_sha256"]) == 64,
              "frontend_control": frontend["semantic_walsh_order_control"], "frontend_built": frontend["dist_newer"],
              "retention_hashes": all(len(v["sha256"]) == 64 for v in retention["kept"]),
              "cleanup_scoped": all(Path(v["path"]).suffix == ".npy" for v in retention["deleted"]),
              "phase_sequence": all(sequence), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend,
              "retention": retention,
              "adjudication": {"strongest_piece": "behavior-necessary pair-relative selection interaction reaches candidate-sequence probabilities",
                               "hard_negative": "answer interaction is not a stable vector through autonomous output and raw cross-unit answer identity fails wrong-q95",
                               "next_stage_same_immediate_target": True,
                               "automatic_continuation_required": True,
                               "next_stage_target": "re-pair the same relation families with new partners and fresh strings",
                               "pure_semantic_code_identified": False, "causal_mediator_identified": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
