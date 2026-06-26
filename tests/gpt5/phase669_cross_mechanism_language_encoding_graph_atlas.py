#!/usr/bin/env python3
"""
Phase 669: Cross-Mechanism Language Encoding Graph Atlas.

This phase does not run model inference. It converts the accumulated objective
results into a structured graph atlas so the next work can move from single
mechanism probing to cross-mechanism map completion.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List


OUT_ROOT = Path("results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas")
RESULTS_ROOT = Path("results")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def phase_summary_path(phase: int) -> Path | None:
    matches = sorted(RESULTS_ROOT.glob(f"glm5_phase{phase}_*/phase{phase}_cross_model_summary.md"))
    return matches[0] if matches else None


def confirm_json(phase: int, model: str) -> Dict:
    matches = sorted(RESULTS_ROOT.glob(f"glm5_phase{phase}_*/phase{phase}_{model}_*_confirm.json"))
    return load_json(matches[0]) if matches else {}


def evidence_file(phase: int) -> str:
    path = phase_summary_path(phase)
    return str(path) if path else ""


def has_phase(phase: int) -> bool:
    return phase_summary_path(phase) is not None


def phase_text(phase: int) -> str:
    path = phase_summary_path(phase)
    return read_text(path) if path else ""


def metric_phase665() -> Dict:
    out = {}
    for model in MODELS:
        d = confirm_json(665, model)
        if not d:
            continue
        out[model] = {
            "raw_cases": d.get("n_raw_cases"),
            "selected_items": d.get("n_selected_items"),
            "continuation_failures": d.get("n_continuation_failures"),
            "scan_layers": d.get("scan_layers"),
        }
    return out


def top_specificity(phase: int, model: str, key: str) -> List[Dict]:
    d = confirm_json(phase, model)
    if not d:
        return []
    rows = d.get("summary", {}).get(key, [])
    out = []
    for r in rows[:8]:
        item = {
            "pair_task": r.get("pair_task"),
            "site": r.get("site"),
            "combo": r.get("combo_name"),
        }
        for k in [
            "boundary",
            "writer",
            "ensemble",
            "kind",
            "correct_top1_rate",
            "mismatch_top1_rate",
            "correct_minus_mismatch_margin_delta",
            "correct_minus_zero_margin_delta",
            "n",
        ]:
            if k in r:
                item[k] = r[k]
        out.append(item)
    return out


def phase_exists_list() -> List[int]:
    phases = []
    for p in range(626, 669):
        if has_phase(p):
            phases.append(p)
    return phases


def node(
    node_id: str,
    label: str,
    role: str,
    system: str,
    status: str,
    phases: List[int],
    models: Dict,
    sufficiency: str,
    necessity: str,
    failure_modes: List[str],
    hard_limits: List[str],
    next_tests: List[str],
) -> Dict:
    return {
        "id": node_id,
        "label": label,
        "function_role": role,
        "language_system": system,
        "status": status,
        "evidence_phases": phases,
        "evidence_files": [evidence_file(p) for p in phases if evidence_file(p)],
        "model_evidence": models,
        "sufficiency": sufficiency,
        "necessity": necessity,
        "failure_modes": failure_modes,
        "hard_limits": hard_limits,
        "next_tests": next_tests,
    }


def edge(src: str, dst: str, relation: str, phases: List[int], confidence: str, note: str) -> Dict:
    return {
        "from": src,
        "to": dst,
        "relation": relation,
        "evidence_phases": phases,
        "confidence": confidence,
        "note": note,
    }


def build_atlas() -> Dict:
    p665 = metric_phase665()
    p666 = {m: top_specificity(666, m, "boundary_specificity") for m in MODELS}
    p667 = {m: top_specificity(667, m, "writer_specificity") for m in MODELS}
    p668 = {m: top_specificity(668, m, "ensemble_specificity") for m in MODELS}
    available = phase_exists_list()

    nodes = [
        node(
            "semantic_value_support",
            "Semantic Value Support",
            "Forms candidate value support from object/category/relation/value conditions.",
            "knowledge_network",
            "partial_confirmed",
            [626, 628, 651, 653, 654],
            {"summary": "Value support can be restored into several task contexts but does not guarantee generation."},
            "Restore and protocol patches can raise correct-prefix rank and sometimes exact generation.",
            "Necessary path not fully isolated; support alone is insufficient.",
            ["support_without_generation", "wrong_format_absorption"],
            ["semantic support is mixed with protocol and readout policy", "not yet a clean natural writer map"],
            ["separate same-value wrong-format and different-value same-format controls"],
        ),
        node(
            "task_intent_gate",
            "Task Intent Gate",
            "Permits or suppresses short-value behavior according to instruction/task demand.",
            "reasoning_route",
            "partial_confirmed",
            [651, 652, 653],
            {"summary": "Value-to-task and task-to-value patches show task intent modulates short-answer protocol."},
            "Value-to-task restore can pull non-value prompts toward value answers.",
            "Task-to-value suppression shows intent can block short-answer support, but necessity is position dependent.",
            ["task_intent_suppresses_value", "task_absorbs_value_protocol"],
            ["rank shifts often stronger than exact generation shifts", "position effects are broad"],
            ["build pure intent controls without relation/content changes"],
        ),
        node(
            "protocol_execution_field",
            "Protocol Execution Field",
            "Boundary and label-driven format/protocol trajectory controlling short answer vs explanation/newline behavior.",
            "grammar_format_protocol",
            "strong_partial",
            [638, 639, 640, 641, 642, 643, 645, 647, 648, 649, 650],
            {"summary": "Separator, label_aligned, relation_tail, and L17-L20/L20-L22 trajectories repeatedly control protocol."},
            "Protocol trajectory restores can flip newline/explanation vs short-value generation in several models.",
            "Necessary intervals exist, but side effects and model-specific boundaries remain.",
            ["newline_protocol_takeover", "explanation_protocol_takeover", "template_side_effect"],
            ["protocol and semantic value are mixed at relation_tail/label sites", "template sensitivity remains"],
            ["map protocol nodes across JSON/code/natural explanation formats"],
        ),
        node(
            "first_token_readout_closure",
            "First Token Readout Closure",
            "Makes correct answer prefix become top1 at the first generated token.",
            "readout_competition",
            "confirmed_but_insufficient",
            [631, 632, 633, 635, 636, 659, 660, 661, 662],
            {"summary": "Readout closure can be engineered, but Phase 665 shows it does not close full answer generation."},
            "Readout/last-writer combinations can make correct_prefix top1.",
            "Not sufficient for exact answer sequence.",
            ["space_competitor", "newline_competitor", "word_competitor", "explanation_competitor"],
            ["projection norm and hidden direction are separable", "first-token closure can still fail continuation"],
            ["record multi-competitor margins for every future generation test"],
        ),
        node(
            "multi_competitor_readout",
            "Multi-Competitor Readout",
            "Compares correct prefix against space/newline/word/explanation/punctuation competitors.",
            "readout_competition",
            "confirmed",
            [636, 659, 662, 663, 664],
            {"summary": "Top1 replacement and pairwise margins are insufficient; multi-competitor view better explains failure."},
            "Multi-competitor correction identifies hidden blockers missed by pairwise tests.",
            "Not a standalone generative mechanism.",
            ["top1_barrier", "projection_specific_failure", "competitor_replacement"],
            ["competitor sets must be updated for each task format"],
            ["extend competitor taxonomy beyond ORV values to JSON/code/explanation tasks"],
        ),
        node(
            "format_continuation_state",
            "Format Continuation State",
            "General continuation state after correct_prefix that keeps generation in a format/protocol route.",
            "grammar_format_protocol",
            "confirmed_partial",
            [665, 666],
            {"phase666_top": p666},
            "Correct and mismatch restores can both repair some early boundaries, showing general continuation state.",
            "Zero removal can be destructive but not semantically specific.",
            ["correct_prefix_then_explanation", "correct_prefix_then_newline", "format_reentry"],
            ["mismatch_restore is not a pure semantic control", "format state and value state are entangled"],
            ["construct same-value wrong-format controls"],
        ),
        node(
            "value_specific_token1_transition_state",
            "Value-Specific Token1 Transition State",
            "Controls transition from answer-entry token to first content-discriminative value token.",
            "knowledge_network",
            "strong_model_specific",
            [665, 666, 667, 668],
            {
                "phase665": p665,
                "phase666_top": p666,
                "phase667_top": p667,
                "phase668_top": p668,
            },
            "Correct_restore beats mismatch_restore on key boundaries, especially DS7B and qwen3 L23/L22 boundary.",
            "Zero_remove often destroys token1, but this also affects scale/format/position.",
            ["token0_to_token1_failure", "value_mismatch_degrades_token1"],
            ["small sample counts: qwen3=5, glm4=4, ds7b=12", "not generalized beyond ORV short values yet"],
            ["same-prefix different-continuation controls", "larger continuation failure dataset"],
        ),
        node(
            "writer_topology",
            "Writer Topology",
            "Mechanism topology that writes value-specific token1 transition state.",
            "mechanism_graph",
            "model_specific_partial",
            [667, 668],
            {"phase667_top": p667, "phase668_top": p668},
            "qwen3 head10+head11 ensemble shows strong semantic subchannel; DS7B requires full boundary state.",
            "Single-head closure rejected as a universal explanation.",
            ["single_head_underexplains", "ensemble_undercloses_ds7b"],
            ["o_proj head slice is not full Q/K/V causal path", "layernorm/residual scale untested"],
            ["test layernorm and residual scale controls", "Q/K/V pattern decomposition only on confirmed ensemble candidates"],
        ),
        node(
            "residual_boundary_integrated_state",
            "Residual Boundary Integrated State",
            "Integrated residual state at layer_out/layer_input boundaries that cannot be reduced to small writer sets.",
            "mechanism_graph",
            "strong_in_ds7b_partial_elsewhere",
            [666, 667, 668],
            {"phase668_top": p668},
            "DS7B full L21 layer_out / L22 layer_input restores token1 strongly and specifically.",
            "Small head/component ensembles fail to close DS7B.",
            ["ensemble_gap", "boundary_state_dependency"],
            ["may include normalization/scale state not captured by component restores"],
            ["directly decompose layernorm input/output and residual scale"],
        ),
        node(
            "continuation_controller",
            "Continuation Controller",
            "Stabilizes answer sequence after entry token; Phase 665 shows early token1 bottleneck dominates current failures.",
            "generation_route",
            "partial_confirmed",
            [664, 665],
            {"phase665": p665},
            "If token1 is forced correct, token2 is usually top1 in selected failures.",
            "Later continuation beyond token2 is not yet fully audited.",
            ["token1_transition_failure", "later_sequence_unverified"],
            ["current exact generation tests use short values and small failure sets"],
            ["extend forced continuation audit to token3+ and longer answer formats"],
        ),
    ]

    edges = [
        edge("semantic_value_support", "task_intent_gate", "permission_edge", [651, 652, 653], "medium", "Intent can permit or suppress the value support route."),
        edge("task_intent_gate", "protocol_execution_field", "protocol_edge", [651, 653, 650], "medium", "Intent and protocol jointly choose short answer vs explanation/full sentence behavior."),
        edge("protocol_execution_field", "first_token_readout_closure", "support_edge", [643, 651, 661], "high", "Protocol trajectory can make correct prefix accessible to first-token readout."),
        edge("first_token_readout_closure", "multi_competitor_readout", "competition_edge", [636, 659, 662], "high", "Correct prefix must beat multiple policy competitors."),
        edge("multi_competitor_readout", "format_continuation_state", "transition_edge", [664, 665, 666], "medium", "Top1 prefix does not guarantee continuation route; format continuation state is separate."),
        edge("format_continuation_state", "value_specific_token1_transition_state", "transition_edge", [666], "high", "Phase 666 separates general continuation from value-specific token1 transition."),
        edge("value_specific_token1_transition_state", "writer_topology", "writer_edge", [667, 668], "medium", "Writer topology is model-specific; qwen3 small head ensemble, DS7B residual boundary."),
        edge("writer_topology", "residual_boundary_integrated_state", "integration_edge", [667, 668], "medium", "Full residual boundaries can dominate small writer ensembles."),
        edge("value_specific_token1_transition_state", "continuation_controller", "continuation_edge", [665, 666], "medium", "Once token1 is correct, token2 tends to stabilize in current failures."),
    ]

    phase_index = []
    for p in available:
        txt = phase_text(p)
        phase_index.append({
            "phase": p,
            "summary_file": evidence_file(p),
            "chars": len(txt),
            "has_confirm_json": {m: bool(confirm_json(p, m)) for m in MODELS},
        })

    atlas = {
        "phase": 669,
        "title": "Cross-Mechanism Language Encoding Graph Atlas",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase_range": [626, 668],
        "available_phase_count": len(available),
        "available_phases": available,
        "key_evidence": {
            "phase665_continuation_failures": p665,
            "phase666_token1_boundary_specificity": p666,
            "phase667_writer_specificity": p667,
            "phase668_writer_ensemble_specificity": p668,
        },
        "nodes": nodes,
        "edges": edges,
        "phase_index": phase_index,
        "global_findings": [
            "The token1 transition gate local phase is complete enough to stop single-head chasing.",
            "The current atlas has stable functional nodes but model-specific implementation topology.",
            "The largest open gap is not another local patch, but clean graph-scale controls and purer semantic/format counterfactuals.",
        ],
        "hard_limits": [
            "Several nodes are based on ORV short-value tasks and require format/generalization tests.",
            "Mismatch restore is useful but not a pure semantic intervention.",
            "Zero removal proves necessity only in a broad state-distribution sense.",
            "Natural writer paths are still partly unresolved because head-slice tests do not decompose Q/K/V and attention pattern.",
        ],
        "next_phase": {
            "phase": 670,
            "title": "Graph Atlas Counterfactual Control Set",
            "goal": "Build clean same-value/different-format and different-value/same-format counterfactuals for the atlas nodes before running new model tests.",
            "reason": "The atlas shows the main bottleneck is control purity, not absence of local candidates.",
        },
    }
    return atlas


def write_markdown(atlas: Dict) -> str:
    lines = [
        "# Phase 669 Cross-Mechanism Language Encoding Graph Atlas",
        "",
        f"- generated: `{atlas['timestamp']}`",
        f"- source_phase_range: `{atlas['source_phase_range']}`",
        f"- available_phase_count: `{atlas['available_phase_count']}`",
        f"- available_phases: `{atlas['available_phases']}`",
        "",
        "## Nodes",
        "",
        "| id | label | system | status | phases | sufficiency | necessity |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in atlas["nodes"]:
        phases = ",".join(str(p) for p in n["evidence_phases"])
        lines.append(
            f"| {n['id']} | {n['label']} | {n['language_system']} | {n['status']} | "
            f"{phases} | {n['sufficiency']} | {n['necessity']} |"
        )
    lines += [
        "",
        "## Edges",
        "",
        "| from | to | relation | confidence | phases | note |",
        "|---|---|---|---|---|---|",
    ]
    for e in atlas["edges"]:
        phases = ",".join(str(p) for p in e["evidence_phases"])
        lines.append(f"| {e['from']} | {e['to']} | {e['relation']} | {e['confidence']} | {phases} | {e['note']} |")

    lines += ["", "## Model-Specific Token1 Transition Evidence", ""]
    token_node = next(n for n in atlas["nodes"] if n["id"] == "value_specific_token1_transition_state")
    for model, rows in token_node["model_evidence"].get("phase668_top", {}).items():
        lines += [f"### {model}", "", "| ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows[:6]:
            lines.append(
                f"| {r.get('ensemble')} | {r.get('kind')} | {r.get('n')} | "
                f"{r.get('correct_top1_rate', 0):.3f} | {r.get('mismatch_top1_rate', 0):.3f} | "
                f"{r.get('correct_minus_mismatch_margin_delta', 0):.3f} | {r.get('correct_minus_zero_margin_delta', 0):.3f} |"
            )
        lines.append("")

    lines += [
        "## Global Findings",
        "",
    ]
    for item in atlas["global_findings"]:
        lines.append(f"- {item}")
    lines += ["", "## Hard Limits", ""]
    for item in atlas["hard_limits"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Next Phase",
        "",
        f"- phase: `{atlas['next_phase']['phase']}`",
        f"- title: `{atlas['next_phase']['title']}`",
        f"- goal: {atlas['next_phase']['goal']}",
        f"- reason: {atlas['next_phase']['reason']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas()
    json_path = OUT_ROOT / "phase669_graph_atlas.json"
    md_path = OUT_ROOT / "phase669_graph_atlas.md"
    json_path.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(write_markdown(atlas), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
