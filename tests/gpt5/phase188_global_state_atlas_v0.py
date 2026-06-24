#!/usr/bin/env python3
"""Build Global State-Conditioned Language Mechanism Atlas v0.

This phase is an atlas-infrastructure pass, not a CUDA model test.  It imports
existing GLM5 phase artifacts into a shared schema so later experiments can add
evidence instead of becoming isolated notes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MODELS = ("qwen3", "glm4", "deepseek7b")


CAUSAL_LEVELS: dict[int, dict[str, str]] = {
    1: {
        "name": "correlation",
        "meaning": "Only covariation or behavioral association is observed.",
    },
    2: {
        "name": "decodable_projection",
        "meaning": "A probe/projection can read out the signal; causal status is not proven.",
    },
    3: {
        "name": "transition_evidence",
        "meaning": "Layer-to-layer or component update creates/amplifies the signal.",
    },
    4: {
        "name": "component_path_contribution",
        "meaning": "Attention/MLP/residual/head/channel intervention changes scores or margins.",
    },
    5: {
        "name": "hidden_causal_repair",
        "meaning": "Hidden-state patch repairs failed behavior beyond controls.",
    },
    6: {
        "name": "specific_repair",
        "meaning": "Repair is task-specific rather than a broad distribution perturbation.",
    },
    7: {
        "name": "generation_compositional_closure",
        "meaning": "Mechanism works in real generation and compositional reuse.",
    },
}


@dataclass(frozen=True)
class PhaseSpec:
    phase: int
    title: str
    module: str
    task_family: str
    gate: str
    causal_level: int
    claim: str
    failure_types: tuple[str, ...] = ()
    priority_gap: str = ""
    result_dir: str = ""


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(575, "closed_micro_world_v1", "knowledge_atlas", "OC", "category_gate", 2, "Object/category states are decodable but simple delta swaps have limited causal effect.", ("prompt_retrieval_dominance",), "Need isolated variables.", "results/glm5_phase575_closed_micro_world"),
    PhaseSpec(576, "isolated_category_variable", "knowledge_atlas", "OC", "category_gate", 2, "Object and category subspaces are not cleanly orthogonal; category swaps have near-zero output effect.", ("linear_nonseparability", "prompt_retrieval_dominance"), "Need retrieval circuit closure.", "results/glm5_phase576_isolated_category"),
    PhaseSpec(577, "retrieval_circuit_binding", "retrieval_circuit_atlas", "OC", "retrieval_gate", 4, "Rule retrieval heads exist and category-copy heads are more causal than object-match heads.", ("redundant_retrieval",), "Need true edge closure.", "results/glm5_phase577_retrieval_circuit"),
    PhaseSpec(578, "retrieval_circuit_closure", "retrieval_circuit_atlas", "OC", "retrieval_gate", 4, "Retrieval circuit closure improves specificity but token-level and edge-level evidence differ.", ("edge_specificity_gap",), "Need true attention edge audit.", "results/glm5_phase578_retrieval_closure"),
    PhaseSpec(579, "true_retrieval_edge_closure", "retrieval_circuit_atlas", "OC", "retrieval_gate", 4, "GQA-safe true edge and value attribution audit; evaluation bugs and edge handling clarified.", ("evaluation_fragility",), "Need ORV closure.", "results/glm5_phase579_true_edge_closure"),
    PhaseSpec(580, "orv_retrieval_reasoning", "knowledge_atlas", "ORV", "value_retrieval_gate", 4, "ORV retrieval works but early two-hop result had prompt bug; value token is major source.", ("orv_binding_bias", "template_binding_bias"), "Need composition closure.", "results/glm5_phase580_orv_closure"),
    PhaseSpec(581, "composition_closure", "reasoning_atlas", "Two-hop", "state_bridge_gate", 3, "Two-hop prompt bug fixed; composition reaches 57-90 percent but has state bridge gap.", ("state_bridge_gap", "C_wrong_cat"), "Need failure typing.", "results/glm5_phase581_composition_closure"),
    PhaseSpec(582, "relation_parametric", "failure_atlas", "Two-hop/parametric", "relation_filter_gate", 3, "Relation is used but unstable; main composition failure is wrong intermediate category; GLM4/DS7B show yes-bias.", ("C_wrong_cat", "relation_ignored", "yes_bias", "B_fail", "V_copy_fail"), "Need polarity/readout decomposition.", "results/glm5_phase582_relation_parametric"),
    PhaseSpec(583, "choice_polarity", "gate_atlas", "yes-no", "polarity_format_gate", 3, "Intermediate choice and polarity readout separate state choice from yes/no format calibration.", ("polarity_readout_fail", "yes_bias"), "Need state-conditioned repair.", "results/glm5_phase583_choice_polarity"),
    PhaseSpec(584, "state_conditioned_gate_repair", "gate_atlas", "Two-hop/value/yes-no", "state_gate", 4, "Relation-filter prompt repair and polarity format repair are strong, but wrong-category forcing is not clean path proof.", ("relation_filter_fail", "polarity_fail", "wrong_category_path_ambiguity"), "Need hidden patch.", "results/glm5_phase584_gate_repair"),
    PhaseSpec(585, "state_gate_hidden_patch", "gate_atlas", "value/yes-no", "polarity_format_gate", 5, "Polarity-format hidden patch is strong; value relation-filter hidden patch does not transfer reliably.", ("value_hidden_patch_fail",), "Need distributed value path patch.", "results/glm5_phase585_state_gate_hidden_patch"),
    PhaseSpec(586, "distributed_value_path_patch", "gate_atlas", "value", "value_gate", 4, "Single-position residual patch raises correct logprob but does not flip value winner.", ("candidate_common_only", "ranking_fail"), "Need winner competition audit.", "results/glm5_phase586_distributed_value_path_patch"),
    PhaseSpec(587, "value_winner_competition", "competition_atlas", "value", "candidate_ranking_gate", 3, "Correct and top-wrong often rise together; missing component is winner-margin control.", ("candidate_common_only", "ranking_fail"), "Need candidate-space decomposition.", "results/glm5_phase587_value_winner_competition"),
    PhaseSpec(588, "value_candidate_space_decomposition", "competition_atlas", "value", "candidate_ranking_gate", 3, "Unembedding-based common/specific decomposition is too crude for controllable suppression.", ("decomposition_leakage", "ranking_fail"), "Need component-level attribution.", "results/glm5_phase588_value_candidate_space_decomposition"),
    PhaseSpec(589, "component_value_path_attribution", "component_atlas", "value", "candidate_activation_gate", 4, "Residual/attention carry candidate co-activation; component winner selection remains unresolved.", ("candidate_common_only", "component_margin_fail"), "Need multi-layer patch.", "results/glm5_phase589_component_value_path_attribution"),
    PhaseSpec(590, "value_winner_multilayer_patch", "component_atlas", "value", "candidate_ranking_gate", 4, "Multi-layer patches still fail DS7B winner switch; query_relation gives only small margin gains.", ("ranking_fail", "winner_switch_fail"), "Need internal ranking audit.", "results/glm5_phase590_value_winner_multilayer_patch"),
    PhaseSpec(591, "value_candidate_ranking_audit", "competition_atlas", "value", "candidate_ranking_gate", 3, "Prompt repair creates large specific support; hidden residual+attention patch mostly creates common activation.", ("candidate_common_only", "top_wrong_rule_overlap"), "Need relation-specific ranking atlas.", "results/glm5_phase591_value_candidate_ranking_audit"),
    PhaseSpec(592, "relation_specific_ranking_atlas", "competition_atlas", "value", "candidate_ranking_gate", 2, "Ranking projection peaks are mapped; evidence is projection-level, not causal repair.", ("projection_not_causal",), "Need atlas-guided causal patch.", "results/glm5_phase592_relation_specific_ranking_atlas"),
    PhaseSpec(593, "atlas_guided_causal_patch", "competition_atlas", "value", "candidate_ranking_gate", 4, "Atlas projection nodes do not become reliable residual causal repair nodes.", ("projection_not_causal", "single_vector_patch_fail"), "Need conditional transformation atlas.", "results/glm5_phase593_atlas_guided_causal_patch"),
    PhaseSpec(594, "conditional_transformation_atlas", "state_transition_atlas", "value", "candidate_ranking_gate", 3, "Candidate-specific ranking can strengthen in layer transitions, especially DS7B rule_value L26 MLP update.", ("transition_not_yet_causal",), "Need component causal validation.", "results/glm5_phase594_conditional_transformation_atlas"),
    PhaseSpec(595, "mlp_update_causal_validation", "component_atlas", "value", "candidate_ranking_gate", 4, "MLP update patches still mostly fail winner repair; transition projection does not equal causal patch.", ("component_patch_fail", "winner_switch_fail"), "Need MLP internal path audit.", "results/glm5_phase595_mlp_update_causal_validation"),
    PhaseSpec(596, "mlp_internal_gate_path_audit", "component_atlas", "value", "candidate_ranking_gate", 3, "Internal MLP projections are visible but causal patches remain weak or control-like.", ("internal_projection_not_causal", "channel_patch_fail"), "Need state-conditioned recomputation audit.", "results/glm5_phase596_mlp_internal_gate_path_audit"),
    PhaseSpec(597, "state_conditioned_mlp_generation_audit", "state_transition_atlas", "value", "state_conditioned_mlp_gate", 3, "State-conditioned MLP recomputation generates strong projections, but causal state patches remain weak and controls can match them.", ("state_patch_fail", "control_similarity"), "Need stronger state isolation and generation closure.", "results/glm5_phase597_state_conditioned_mlp_generation_audit"),
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, node_type: str, **attrs: Any) -> None:
    if node_id not in nodes:
        nodes[node_id] = {"node_id": node_id, "node_type": node_type}
    nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})


def add_edge(edges: list[dict[str, Any]], source: str, target: str, edge_type: str, **attrs: Any) -> None:
    row = {"source": source, "target": target, "edge_type": edge_type}
    row.update({k: v for k, v in attrs.items() if v is not None})
    edges.append(row)


def result_jsons(spec: PhaseSpec) -> list[Path]:
    root = Path(spec.result_dir)
    if not root.exists():
        return []
    files = []
    for path in sorted(root.glob("*.json")):
        name = path.name
        if "smoke" in name or "checkpoint" in name:
            continue
        files.append(path)
    return files


def extract_model(path: Path, data: dict[str, Any]) -> str:
    model = data.get("model")
    if isinstance(model, str) and model:
        return model
    for model_name in MODELS:
        if model_name in path.name:
            return model_name
    return "unknown"


def extract_sample_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "n_cases",
        "target_n",
        "n_target_rows",
        "n_rows",
        "n_target_cases_seen",
        "target_cases_seen",
        "total_time_min",
        "alpha",
    ]
    metrics = {k: data[k] for k in metric_keys if k in data}
    if "rows" in data and isinstance(data["rows"], list):
        metrics["row_count"] = len(data["rows"])
    if "summary" in data and isinstance(data["summary"], dict):
        best = data["summary"].get("best")
        if isinstance(best, list):
            metrics["best_count"] = len(best)
            if best and isinstance(best[0], dict):
                metrics["best_top"] = {
                    k: best[0].get(k)
                    for k in [
                        "position",
                        "layer",
                        "bucket",
                        "mean_specific_margin",
                        "margin_gain",
                        "specific_margin_gain",
                        "switch_rate",
                        "positive_specific_rate",
                    ]
                    if k in best[0]
                }
    return metrics


def summary_path(spec: PhaseSpec) -> Path | None:
    root = Path(spec.result_dir)
    if not root.exists():
        return None
    candidates = sorted(root.glob("*cross_model_summary.md"))
    return candidates[0] if candidates else None


def first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.name


def build_atlas(out_dir: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    add_node(nodes, "atlas:global_state_language_mechanism_v0", "atlas", version="v0", phase=188)

    for level, desc in CAUSAL_LEVELS.items():
        level_id = f"causal_level:{level}"
        add_node(nodes, level_id, "causal_level", level=level, **desc)

    for model in MODELS:
        add_node(nodes, f"model:{model}", "model", model=model)

    for spec in PHASES:
        phase_id = f"phase:{spec.phase}"
        module_id = f"module:{spec.module}"
        task_id = f"task:{spec.task_family}"
        gate_id = f"gate:{spec.gate}"
        level_id = f"causal_level:{spec.causal_level}"

        add_node(
            nodes,
            phase_id,
            "phase",
            phase=spec.phase,
            title=spec.title,
            claim=spec.claim,
            priority_gap=spec.priority_gap,
            result_dir=spec.result_dir,
        )
        add_node(nodes, module_id, "module", module=spec.module)
        add_node(nodes, task_id, "task_family", task_family=spec.task_family)
        add_node(nodes, gate_id, "gate", gate=spec.gate)

        add_edge(edges, "atlas:global_state_language_mechanism_v0", phase_id, "contains_phase")
        add_edge(edges, phase_id, module_id, "belongs_to_module", phase=spec.phase)
        add_edge(edges, phase_id, task_id, "studies_task", phase=spec.phase)
        add_edge(edges, phase_id, gate_id, "studies_gate", phase=spec.phase)
        add_edge(edges, phase_id, level_id, "assigned_causal_level", phase=spec.phase, causal_level=spec.causal_level)

        for ft in spec.failure_types:
            failure_id = f"failure:{ft}"
            add_node(nodes, failure_id, "failure_type", failure_type=ft)
            add_edge(edges, phase_id, failure_id, "observes_failure", phase=spec.phase)
            failures.append(
                {
                    "phase": spec.phase,
                    "failure_type": ft,
                    "module": spec.module,
                    "task_family": spec.task_family,
                    "gate": spec.gate,
                    "source": "phase_metadata",
                }
            )

        spath = summary_path(spec)
        if spath:
            evidence_id = f"evidence:phase{spec.phase}:cross_model_summary"
            add_node(
                nodes,
                evidence_id,
                "evidence",
                phase=spec.phase,
                evidence_type="cross_model_summary",
                path=str(spath),
                title=first_heading(spath),
                causal_level=spec.causal_level,
            )
            add_edge(edges, phase_id, evidence_id, "has_evidence", phase=spec.phase)
            evidence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "phase": spec.phase,
                    "path": str(spath),
                    "evidence_type": "cross_model_summary",
                    "causal_level": spec.causal_level,
                }
            )

        for path in result_jsons(spec):
            data = read_json(path)
            if data is None:
                continue
            model = extract_model(path, data)
            sample_id = f"sample_summary:phase{spec.phase}:{model}:{path.stem}"
            add_node(
                nodes,
                sample_id,
                "sample_summary",
                phase=spec.phase,
                model=model,
                path=str(path),
                task_family=spec.task_family,
                gate=spec.gate,
                causal_level=spec.causal_level,
            )
            add_edge(edges, f"model:{model}", sample_id, "has_sample_summary", phase=spec.phase)
            add_edge(edges, phase_id, sample_id, "has_sample_summary", phase=spec.phase)
            add_edge(edges, sample_id, gate_id, "measures_gate", phase=spec.phase, model=model)
            samples.append(
                {
                    "sample_id": sample_id,
                    "phase": spec.phase,
                    "model": model,
                    "path": str(path),
                    "module": spec.module,
                    "task_family": spec.task_family,
                    "gate": spec.gate,
                    "causal_level": spec.causal_level,
                    "metrics": extract_sample_metrics(data),
                }
            )

            atlas = data.get("atlas")
            if isinstance(atlas, dict):
                imported_nodes = atlas.get("nodes")
                imported_edges = atlas.get("edges")
                if isinstance(imported_nodes, list):
                    for idx, node in enumerate(imported_nodes[:500]):
                        imported_id = f"imported:phase{spec.phase}:{model}:node:{idx}"
                        add_node(
                            nodes,
                            imported_id,
                            "imported_atlas_node",
                            phase=spec.phase,
                            model=model,
                            source_path=str(path),
                            raw=node,
                        )
                        add_edge(edges, sample_id, imported_id, "imports_node", phase=spec.phase, model=model)
                if isinstance(imported_edges, list):
                    for idx, edge in enumerate(imported_edges[:500]):
                        edge_id = f"imported:phase{spec.phase}:{model}:edge:{idx}"
                        add_node(
                            nodes,
                            edge_id,
                            "imported_atlas_edge_record",
                            phase=spec.phase,
                            model=model,
                            source_path=str(path),
                            raw=edge,
                        )
                        add_edge(edges, sample_id, edge_id, "imports_edge_record", phase=spec.phase, model=model)

    node_rows = sorted(nodes.values(), key=lambda x: x["node_id"])
    edge_rows = sorted(edges, key=lambda x: (x["source"], x["target"], x["edge_type"]))
    samples = sorted(samples, key=lambda x: (x["phase"], x["model"], x["path"]))
    failures = sorted(failures, key=lambda x: (x["failure_type"], x["phase"]))
    evidence_rows = sorted(evidence_rows, key=lambda x: (x["phase"], x["evidence_id"]))

    write_jsonl(out_dir / "atlas_nodes.jsonl", node_rows)
    write_jsonl(out_dir / "atlas_edges.jsonl", edge_rows)
    write_jsonl(out_dir / "atlas_samples.jsonl", samples)
    write_jsonl(out_dir / "atlas_failures.jsonl", failures)
    write_jsonl(out_dir / "atlas_evidence.jsonl", evidence_rows)
    (out_dir / "atlas_causal_levels.json").write_text(
        json.dumps(CAUSAL_LEVELS, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_schema = {
        "schema_name": "state_trajectory_schema_v0",
        "purpose": "Future activation-trajectory imports; not populated by Phase188.",
        "fields": [
            "sample_id",
            "model",
            "task_family",
            "prompt_id",
            "token_index",
            "token_role",
            "layer",
            "component",
            "position_role",
            "state_features",
            "candidate_scores",
            "common_activation",
            "specific_ranking",
            "margin",
            "failure_type",
            "causal_level",
        ],
    }
    (out_dir / "atlas_state_schema.json").write_text(
        json.dumps(state_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    index = {
        "phase": 188,
        "atlas": "Global State-Conditioned Language Mechanism Atlas v0",
        "phase_range_imported": [min(p.phase for p in PHASES), max(p.phase for p in PHASES)],
        "counts": {
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "sample_summaries": len(samples),
            "failure_records": len(failures),
            "evidence_records": len(evidence_rows),
        },
        "files": {
            "nodes": "atlas_nodes.jsonl",
            "edges": "atlas_edges.jsonl",
            "samples": "atlas_samples.jsonl",
            "failures": "atlas_failures.jsonl",
            "evidence": "atlas_evidence.jsonl",
            "causal_levels": "atlas_causal_levels.json",
            "state_schema": "atlas_state_schema.json",
            "model_summary": "atlas_model_summary.md",
            "task_summary": "atlas_task_summary.md",
            "next_recommendations": "atlas_next_recommendations.md",
        },
    }
    (out_dir / "atlas_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_summaries(out_dir, samples, failures, evidence_rows)
    return index


def write_summaries(
    out_dir: Path,
    samples: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> None:
    model_counts = Counter(row["model"] for row in samples)
    phase_counts = Counter(row["phase"] for row in samples)
    task_counts = Counter(row["task_family"] for row in samples)
    gate_counts = Counter(row["gate"] for row in samples)
    failure_counts = Counter(row["failure_type"] for row in failures)
    level_counts = Counter(row["causal_level"] for row in samples)

    lines = [
        "# Phase188 Atlas Model Summary",
        "",
        "Atlas v0 imports existing GLM5 result files. It does not rerun CUDA models.",
        "",
        "## Imported Sample Summaries by Model",
        "",
        "| model | sample summaries |",
        "|---|---:|",
    ]
    for model in sorted(model_counts):
        lines.append(f"| {model} | {model_counts[model]} |")
    lines.extend(["", "## Imported Sample Summaries by Phase", "", "| phase | count |", "|---:|---:|"])
    for phase, count in sorted(phase_counts.items()):
        lines.append(f"| {phase} | {count} |")
    lines.extend(["", "## Causal Level Coverage", "", "| level | name | sample summaries |", "|---:|---|---:|"])
    for level in sorted(CAUSAL_LEVELS):
        lines.append(f"| {level} | {CAUSAL_LEVELS[level]['name']} | {level_counts[level]} |")
    lines.extend(["", "## Highest-Priority Reading", ""])
    lines.append("- Level 2/3 evidence remains abundant for candidate-specific ranking.")
    lines.append("- Level 4/5 evidence is sparse and often negative for value winner repair.")
    lines.append("- Polarity-format gate is the strongest hidden-repair region; value/ranking gate remains the central gap.")
    (out_dir / "atlas_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    task_lines = [
        "# Phase188 Atlas Task Summary",
        "",
        "## Task Families",
        "",
        "| task family | sample summaries |",
        "|---|---:|",
    ]
    for task, count in sorted(task_counts.items()):
        task_lines.append(f"| {task} | {count} |")
    task_lines.extend(["", "## Gates", "", "| gate | sample summaries |", "|---|---:|"])
    for gate, count in sorted(gate_counts.items()):
        task_lines.append(f"| {gate} | {count} |")
    task_lines.extend(["", "## Failure Types", "", "| failure type | phase records |", "|---|---:|"])
    for failure, count in failure_counts.most_common():
        task_lines.append(f"| {failure} | {count} |")
    task_lines.extend(["", "## Evidence Records", "", f"- Cross-model summaries imported: {len(evidence_rows)}"])
    (out_dir / "atlas_task_summary.md").write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    rec_lines = [
        "# Phase188 Next Recommendations",
        "",
        "## Main Gap",
        "",
        "The atlas points to candidate-specific ranking as the highest-value gap.",
        "",
        "## Recommended Phase189",
        "",
        "Component-Level Candidate Ranking Closure.",
        "",
        "Priority targets:",
        "",
        "1. DS7B rule_value L26 MLP update.",
        "2. DS7B query_relation L19 MLP update.",
        "3. Qwen3 prompt_last/query_category late-layer positive controls.",
        "",
        "Success criteria:",
        "",
        "1. Separate common activation from correct-specific ranking.",
        "2. Find a component/channel intervention that improves correct-vs-old-top-wrong margin.",
        "3. Show random/wrong-relation controls do not match the repair.",
        "4. Promote evidence from Level 3 transition/projection to Level 4 component causal node.",
        "",
        "## Avoid",
        "",
        "- Treating projection peaks as causal nodes.",
        "- Running another single-position residual patch without component/channel isolation.",
        "- Reporting accuracy without failure type and candidate delta matrix.",
    ]
    (out_dir / "atlas_next_recommendations.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/gpt5_phase188_global_state_atlas_v0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_atlas(out_dir)
    print(json.dumps(index["counts"], ensure_ascii=False, sort_keys=True))
    print(f"[ok] wrote atlas v0 to {out_dir}")


if __name__ == "__main__":
    main()
