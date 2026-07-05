from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_demo_bf16 import get_device_for_input, load_model_bf16  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402
from phase941_color_feature_neuron_atlas import (  # noqa: E402
    decode_token,
    first_token_candidates,
    get_lm_head_rows,
    label_score,
    parse_csv,
)


PHASE = 944
RESULT_ROOT = Path("tests/result/phase944_integrated_mechanism_trace")
PUBLIC_ROOT = Path("frontend/public/vis_data/mechanism_trace")

DEFAULT_COLORS = [
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
]

DEFAULT_PROTOCOL_LABELS = ["answer", "word", "color"]
DEFAULT_TERMINATION_LABELS = [".", "\n"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(tensor: torch.Tensor | None) -> float | None:
    if tensor is None:
        return None
    return float(tensor.float().norm().item())


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    af = a.float()
    bf = b.float()
    denom = float(af.norm().item() * bf.norm().item())
    if denom <= 1e-12:
        return None
    return float(torch.dot(af.flatten(), bf.flatten()).item() / denom)


def softmax_dict(scores: dict[str, float]) -> dict[str, float]:
    finite_scores = {key: finite(value) for key, value in scores.items()}
    if not finite_scores:
        return {}
    max_score = max(finite_scores.values())
    exps = {key: math.exp(value - max_score) for key, value in finite_scores.items()}
    total = sum(exps.values())
    if total <= 0:
        return {key: 0.0 for key in finite_scores}
    return {key: float(value / total) for key, value in exps.items()}


def entropy(probs: dict[str, float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs.values()))


def parse_label_list(text: str, default: list[str]) -> list[str]:
    if text is None or str(text) == "":
        return list(default)
    out: list[str] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if raw == "\\n" or item == "\\n":
            out.append("\n")
        elif raw == "\\t" or item == "\\t":
            out.append("\t")
        elif item:
            out.append(item)
    return out or list(default)


def label_token_map(tokenizer, labels: list[str], prefer_spaced: bool = True, exact_labels: set[str] | None = None) -> dict[str, int]:
    exact_labels = exact_labels or set()
    out: dict[str, int] = {}
    for label in labels:
        ids = first_token_candidates(tokenizer, label)
        if ids:
            out[label] = int(ids[0] if label in exact_labels or not prefer_spaced else ids[-1])
    return out


def find_first_position(input_ids: list[int], token_ids: list[int]) -> int | None:
    wanted = {int(x) for x in token_ids}
    for idx, token_id in enumerate(input_ids):
        if int(token_id) in wanted:
            return int(idx)
    return None


def extract_vector(sequence_tensor: torch.Tensor, position: int | None) -> torch.Tensor | None:
    if position is None:
        return None
    if sequence_tensor.dim() == 3:
        sequence_tensor = sequence_tensor[0]
    if position < 0 or position >= int(sequence_tensor.shape[0]):
        return None
    return sequence_tensor[int(position)].detach().float().cpu()


def row_for(label: str, label_to_token: dict[str, int], rows: dict[int, torch.Tensor]) -> torch.Tensor | None:
    token_id = label_to_token.get(label)
    if token_id is None:
        return None
    return rows.get(int(token_id))


def score_rows(vector: torch.Tensor | None, labels: list[str], label_to_token: dict[str, int], rows: dict[int, torch.Tensor]) -> dict[str, float]:
    if vector is None:
        return {}
    out: dict[str, float] = {}
    vf = vector.float()
    for label in labels:
        row = row_for(label, label_to_token, rows)
        if row is None:
            continue
        out[label] = float(torch.dot(vf, row.float()).item())
    return out


def candidate_field(
    vector: torch.Tensor | None,
    target_label: str,
    colors: list[str],
    label_to_token: dict[str, int],
    rows: dict[int, torch.Tensor],
) -> dict[str, Any]:
    scores = score_rows(vector, colors, label_to_token, rows)
    probs = softmax_dict(scores)
    target_score = scores.get(target_label)
    competitors = [(label, value) for label, value in scores.items() if label != target_label]
    competitor_label, competitor_score = (None, None)
    if competitors:
        competitor_label, competitor_score = max(competitors, key=lambda item: item[1])
    margin = None
    if target_score is not None and competitor_score is not None:
        margin = float(target_score - competitor_score)
    target_rank = None
    if target_label in scores:
        target_value = scores[target_label]
        target_rank = int(1 + sum(1 for value in scores.values() if value > target_value))
    field_entropy = entropy(probs)
    gate_open = 0.0
    if len(probs) > 1:
        gate_open = float(1.0 - field_entropy / math.log(len(probs)))
    sorted_scores = sorted(
        [
            {
                "label": label,
                "token_id": label_to_token.get(label),
                "score": float(value),
                "probability": float(probs.get(label, 0.0)),
            }
            for label, value in scores.items()
        ],
        key=lambda row: row["score"],
        reverse=True,
    )
    return {
        "target_label": target_label,
        "target_score": target_score,
        "target_probability": probs.get(target_label),
        "target_rank_within_candidates": target_rank,
        "competitor_label": competitor_label,
        "competitor_score": competitor_score,
        "margin_vs_competitor": margin,
        "candidate_entropy": field_entropy,
        "candidate_gate_open": gate_open,
        "scores": sorted_scores,
    }


def factor_values(
    vector: torch.Tensor | None,
    target_label: str,
    object_label: str,
    relation_label: str,
    category_label: str,
    colors: list[str],
    protocol_labels: list[str],
    termination_labels: list[str],
    label_to_token: dict[str, int],
    rows: dict[int, torch.Tensor],
    previous_margin: float | None,
    component_norm: float | None = None,
    residual_norm: float | None = None,
) -> dict[str, Any]:
    field = candidate_field(vector, target_label, colors, label_to_token, rows)
    color_scores = {row["label"]: row["score"] for row in field["scores"]}
    color_rows = [row_for(label, label_to_token, rows) for label in colors if row_for(label, label_to_token, rows) is not None]
    attribute_projection = None
    if vector is not None and target_label in label_to_token and len(color_rows) >= 2:
        target_row = row_for(target_label, label_to_token, rows)
        other_rows = [
            row_for(label, label_to_token, rows)
            for label in colors
            if label != target_label and row_for(label, label_to_token, rows) is not None
        ]
        if target_row is not None and other_rows:
            direction = target_row.float() - torch.stack([row.float() for row in other_rows]).mean(dim=0)
            attribute_projection = float(torch.dot(vector.float(), direction).item())

    protocol_scores = score_rows(vector, protocol_labels, label_to_token, rows)
    termination_scores = score_rows(vector, termination_labels, label_to_token, rows)
    object_scores = score_rows(vector, [object_label], label_to_token, rows)
    relation_scores = score_rows(vector, [relation_label], label_to_token, rows)
    category_scores = score_rows(vector, [category_label], label_to_token, rows)

    color_mean = float(sum(color_scores.values()) / len(color_scores)) if color_scores else None
    protocol_mean = float(sum(protocol_scores.values()) / len(protocol_scores)) if protocol_scores else None
    termination_max = max(termination_scores.values()) if termination_scores else None
    margin = field.get("margin_vs_competitor")
    boundary_delta = None if margin is None or previous_margin is None else float(margin - previous_margin)
    natural_gate = None
    if component_norm is not None and residual_norm is not None and residual_norm > 1e-12:
        natural_gate = float(component_norm / residual_norm)

    return {
        "O": {
            "label": "object factor",
            "中文": "对象因子",
            "value": object_scores.get(object_label),
            "token_label": object_label,
        },
        "R": {
            "label": "relation factor",
            "中文": "关系因子",
            "value": relation_scores.get(relation_label),
            "token_label": relation_label,
        },
        "A": {
            "label": "attribute factor",
            "中文": "属性因子",
            "value": attribute_projection,
            "token_label": target_label,
        },
        "C": {
            "label": "category factor",
            "中文": "类别因子",
            "value": category_scores.get(category_label) if category_scores else color_mean,
            "token_label": category_label,
            "color_mean_score": color_mean,
        },
        "F": {
            "label": "function factor",
            "中文": "功能因子",
            "value": protocol_mean,
            "protocol_scores": protocol_scores,
        },
        "M": {
            "label": "candidate gate",
            "中文": "候选门控",
            "value": field.get("candidate_gate_open"),
            "entropy": field.get("candidate_entropy"),
        },
        "K": {
            "label": "knowledge path",
            "中文": "知识路径",
            "value": margin,
            "interpretation": "target color margin over strongest color competitor",
        },
        "S_answer": {
            "label": "semantic answer field",
            "中文": "语义答案场",
            "value": field.get("target_score"),
            "probability": field.get("target_probability"),
            "rank": field.get("target_rank_within_candidates"),
        },
        "B": {
            "label": "blocker field",
            "中文": "阻断场",
            "value": field.get("competitor_score"),
            "competitor_label": field.get("competitor_label"),
            "blocker_advantage": None if margin is None else float(-margin),
        },
        "G": {
            "label": "boundary gear",
            "中文": "边界齿轮",
            "value": boundary_delta,
            "margin": margin,
            "previous_margin": previous_margin,
        },
        "N": {
            "label": "natural gate",
            "中文": "自然 gate",
            "value": natural_gate,
            "component_norm": component_norm,
            "residual_norm": residual_norm,
        },
        "P": {
            "label": "output protocol",
            "中文": "输出协议",
            "value": None if protocol_mean is None or color_mean is None else float(protocol_mean - color_mean),
            "protocol_mean_score": protocol_mean,
            "color_mean_score": color_mean,
        },
        "T": {
            "label": "termination action",
            "中文": "终止动作",
            "value": termination_max,
            "termination_scores": termination_scores,
        },
    }


def compact_component(
    vector: torch.Tensor | None,
    target_label: str,
    colors: list[str],
    label_to_token: dict[str, int],
    rows: dict[int, torch.Tensor],
    residual_vector: torch.Tensor | None,
) -> dict[str, Any]:
    field = candidate_field(vector, target_label, colors, label_to_token, rows)
    return {
        "norm": norm(vector),
        "cosine_to_residual": cosine(vector, residual_vector),
        "candidate_field": {
            key: field[key]
            for key in [
                "target_score",
                "target_probability",
                "target_rank_within_candidates",
                "competitor_label",
                "competitor_score",
                "margin_vs_competitor",
                "candidate_gate_open",
            ]
        },
    }


def make_pipeline() -> list[dict[str, Any]]:
    return [
        {
            "id": "context_state",
            "name": "context state",
            "中文": "上文状态",
            "factor_keys": ["O", "R", "A", "C", "F"],
            "meaning": "token embedding and residual stream contain object, relation, attribute and answer-format cues.",
        },
        {
            "id": "conditioned_routing",
            "name": "conditioned routing",
            "中文": "条件化路由",
            "factor_keys": ["N", "G"],
            "meaning": "attention and MLP update the residual stream differently under this context.",
        },
        {
            "id": "candidate_opening",
            "name": "candidate-space opening",
            "中文": "候选空间打开",
            "factor_keys": ["M", "C"],
            "meaning": "the color vocabulary becomes a measurable local candidate field.",
        },
        {
            "id": "knowledge_path",
            "name": "knowledge-path activation",
            "中文": "知识路径激活",
            "factor_keys": ["K", "S_answer"],
            "meaning": "the target color gains margin against the strongest color competitor.",
        },
        {
            "id": "boundary_competition",
            "name": "output-boundary competition",
            "中文": "输出边界竞争",
            "factor_keys": ["B", "G", "P", "T"],
            "meaning": "readout competition decides whether the next token enters the intended answer protocol.",
        },
        {
            "id": "next_token",
            "name": "next token",
            "中文": "下一 token",
            "factor_keys": ["S_answer", "P", "T"],
            "meaning": "the final hidden state is projected through W_U to produce the next-token distribution.",
        },
    ]


def factor_definitions() -> dict[str, dict[str, str]]:
    return {
        "O": {
            "formula": "O_l = h_l · W_U[token(object)]",
            "中文": "对象因子：当前隐状态对对象词的输出读出强度。",
        },
        "R": {
            "formula": "R_l = h_l · W_U[token(relation)]",
            "中文": "关系因子：当前隐状态对关系词的输出读出强度。",
        },
        "A": {
            "formula": "A_l = h_l · (W_U[target attribute] - mean(W_U[other attributes]))",
            "中文": "属性因子：目标颜色相对其它颜色的方向投影。",
        },
        "C": {
            "formula": "C_l = mean_{c in Colors}(h_l · W_U[c])",
            "中文": "类别因子：颜色类别整体是否被打开。",
        },
        "F": {
            "formula": "F_l = mean_{p in Protocol}(h_l · W_U[p])",
            "中文": "功能因子：回答格式、word/color/answer 等协议词的平均读出。",
        },
        "M": {
            "formula": "M_l = 1 - H(softmax(scores_colors)) / log(|Colors|)",
            "中文": "候选门控：颜色候选分布越尖锐，门控越打开。",
        },
        "K": {
            "formula": "K_l = score(target_color) - max score(other_colors)",
            "中文": "知识路径：目标颜色相对最强竞争颜色的边际。",
        },
        "S_answer": {
            "formula": "S_l = h_l · W_U[target_color]",
            "中文": "语义答案场：目标答案 token 的直接输出读出强度。",
        },
        "B": {
            "formula": "B_l = max_{c != target} h_l · W_U[c]",
            "中文": "blocker field：最强错误候选颜色的读出强度。",
        },
        "G": {
            "formula": "G_l = K_l - K_{l-1}",
            "中文": "边界齿轮：本层把目标边际推高还是压低。",
        },
        "N": {
            "formula": "N_l = ||component_l|| / ||residual_l||",
            "中文": "自然 gate：attention/MLP 对当前 residual 的相对更新幅度。",
        },
        "P": {
            "formula": "P_l = mean(score(protocol_tokens)) - mean(score(color_tokens))",
            "中文": "输出协议：格式协议相对颜色候选场的读出偏置。",
        },
        "T": {
            "formula": "T_l = max score(termination_tokens)",
            "中文": "终止动作：句号、换行、EOS 等终止 token 的读出强度。",
        },
    }


def update_manifest(public_path: Path, item: dict[str, Any]) -> None:
    manifest_path = public_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": "mechanism_trace_manifest_v1",
            "updated_at": None,
            "items": [],
        }
    items = [row for row in manifest.get("items", []) if row.get("id") != item.get("id")]
    items.append(item)
    items.sort(key=lambda row: str(row.get("id", "")))
    manifest["schema_version"] = "mechanism_trace_manifest_v1"
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["items"] = items
    write_json(manifest_path, manifest)


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    model_name = args.model
    colors = parse_label_list(args.colors, DEFAULT_COLORS)
    protocol_labels = parse_label_list(args.protocol_labels, DEFAULT_PROTOCOL_LABELS)
    termination_labels = parse_label_list(args.termination_labels, DEFAULT_TERMINATION_LABELS)
    target_label = args.target_label.strip()
    object_label = args.object_label.strip()
    relation_label = args.relation_label.strip()
    category_label = args.category_label.strip()
    prompt = args.prompt

    log(f"Loading {model_name} for Phase {PHASE}")
    model, tokenizer, _device = load_model_bf16(model_name)
    model_device = get_device_for_input(model)
    try:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        input_id_list = [int(x) for x in input_ids[0].detach().cpu().tolist()]
        tokens = [
            {
                "index": int(idx),
                "id": int(token_id),
                "text": decode_token(tokenizer, int(token_id)),
                "role": "context",
            }
            for idx, token_id in enumerate(input_id_list)
        ]

        attribute_pos = find_first_position(input_id_list, first_token_candidates(tokenizer, target_label))
        object_pos = find_first_position(input_id_list, first_token_candidates(tokenizer, object_label))
        relation_pos = find_first_position(input_id_list, first_token_candidates(tokenizer, relation_label))
        answer_pos = int(attention_mask.sum(dim=1).item() - 1)
        for pos, role in [
            (attribute_pos, "attribute_token"),
            (object_pos, "object_token"),
            (relation_pos, "relation_token"),
            (answer_pos, "answer_context"),
        ]:
            if pos is not None and 0 <= pos < len(tokens):
                tokens[pos]["role"] = role if tokens[pos]["role"] == "context" else f"{tokens[pos]['role']},{role}"

        labels_for_rows = sorted(set(colors + protocol_labels + termination_labels + [target_label, object_label, relation_label, category_label]))
        label_to_token = label_token_map(
            tokenizer,
            labels_for_rows,
            prefer_spaced=True,
            exact_labels=set(termination_labels),
        )
        if getattr(tokenizer, "eos_token_id", None) is not None:
            label_to_token["<eos>"] = int(tokenizer.eos_token_id)
            termination_labels = list(dict.fromkeys(termination_labels + ["<eos>"]))
        token_ids = sorted(set(label_to_token.values()))
        rows = get_lm_head_rows(model, token_ids)

        layers = get_layers(model)
        captured: dict[str, dict[int, torch.Tensor]] = {"attention": {}, "mlp": {}}
        hook_handles = []

        def hook_output(kind: str, layer_idx: int):
            def _hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                vector = extract_vector(value.detach(), answer_pos)
                if vector is not None:
                    captured[kind][int(layer_idx)] = vector
            return _hook

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                hook_handles.append(layer.self_attn.register_forward_hook(hook_output("attention", layer_idx)))
            if hasattr(layer, "mlp"):
                hook_handles.append(layer.mlp.register_forward_hook(hook_output("mlp", layer_idx)))

        log(f"Running forward pass, prompt length={len(input_id_list)}, layers={len(layers)}")
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for handle in hook_handles:
            handle.remove()

        final_logits = out.logits[0, answer_pos].detach().float().cpu()
        final_candidate_rows = {
            label: label_score(tokenizer, final_logits, label)
            for label in colors
        }
        target_global = final_candidate_rows.get(target_label, {})
        top_token_values, top_token_ids = torch.topk(final_logits, k=max(1, int(args.topk)))
        top_tokens = [
            {
                "rank": int(idx + 1),
                "token_id": int(token_id),
                "token": decode_token(tokenizer, int(token_id)),
                "logit": float(value),
            }
            for idx, (token_id, value) in enumerate(zip(top_token_ids.tolist(), top_token_values.tolist()))
        ]

        trace_layers: list[dict[str, Any]] = []
        previous_margin: float | None = None
        hidden_states = list(out.hidden_states or [])
        max_hidden = len(hidden_states)
        for hidden_idx, hidden in enumerate(hidden_states):
            layer_number = hidden_idx - 1
            layer_name = "embedding" if layer_number < 0 else f"layer_{layer_number}"
            answer_vec = extract_vector(hidden, answer_pos)
            object_vec = extract_vector(hidden, object_pos)
            relation_vec = extract_vector(hidden, relation_pos)
            attribute_vec = extract_vector(hidden, attribute_pos)

            field = candidate_field(answer_vec, target_label, colors, label_to_token, rows)
            margin = field.get("margin_vs_competitor")
            residual_norm = norm(answer_vec)
            attn_vec = captured["attention"].get(layer_number)
            mlp_vec = captured["mlp"].get(layer_number)
            component_norm = None
            if layer_number >= 0:
                parts = [value for value in [attn_vec, mlp_vec] if value is not None]
                if parts:
                    component_norm = float(math.sqrt(sum(float(part.float().norm().item()) ** 2 for part in parts)))

            layer_entry = {
                "layer": int(layer_number),
                "name": layer_name,
                "residual": {
                    "norm": residual_norm,
                    "delta_norm_from_previous": None,
                },
                "positions": {
                    "answer": {
                        "index": answer_pos,
                        "norm": norm(answer_vec),
                        "candidate_field": field,
                    },
                    "object": {
                        "index": object_pos,
                        "norm": norm(object_vec),
                        "cosine_to_answer": cosine(object_vec, answer_vec),
                    },
                    "relation": {
                        "index": relation_pos,
                        "norm": norm(relation_vec),
                        "cosine_to_answer": cosine(relation_vec, answer_vec),
                    },
                    "attribute": {
                        "index": attribute_pos,
                        "norm": norm(attribute_vec),
                        "cosine_to_answer": cosine(attribute_vec, answer_vec),
                    },
                },
                "components": {
                    "attention": compact_component(attn_vec, target_label, colors, label_to_token, rows, answer_vec)
                    if layer_number >= 0
                    else None,
                    "mlp": compact_component(mlp_vec, target_label, colors, label_to_token, rows, answer_vec)
                    if layer_number >= 0
                    else None,
                },
                "factors": factor_values(
                    answer_vec,
                    target_label,
                    object_label,
                    relation_label,
                    category_label,
                    colors,
                    protocol_labels,
                    termination_labels,
                    label_to_token,
                    rows,
                    previous_margin=previous_margin,
                    component_norm=component_norm,
                    residual_norm=residual_norm,
                ),
            }

            if hidden_idx > 0 and trace_layers:
                prev_vec = extract_vector(hidden_states[hidden_idx - 1], answer_pos)
                if answer_vec is not None and prev_vec is not None:
                    layer_entry["residual"]["delta_norm_from_previous"] = norm(answer_vec - prev_vec)
                    layer_entry["residual"]["cosine_to_previous"] = cosine(answer_vec, prev_vec)

            if args.save_vectors:
                layer_entry["vectors"] = {
                    "answer": answer_vec.tolist() if answer_vec is not None else None,
                    "attention": attn_vec.tolist() if attn_vec is not None else None,
                    "mlp": mlp_vec.tolist() if mlp_vec is not None else None,
                }

            trace_layers.append(layer_entry)
            previous_margin = margin if margin is not None else previous_margin

            if args.max_hidden_states > 0 and hidden_idx + 1 >= int(args.max_hidden_states):
                break

        final_layer = trace_layers[-1] if trace_layers else {}
        final_field = final_layer.get("positions", {}).get("answer", {}).get("candidate_field", {})
        global_closed = bool(target_global.get("rank") == 1)
        candidate_closed = bool(final_field.get("target_rank_within_candidates") == 1)
        summary = {
            "phase": PHASE,
            "model": model_name,
            "prompt": prompt,
            "target_label": target_label,
            "target_global_token_id": target_global.get("token_id"),
            "target_global_rank": target_global.get("rank"),
            "target_global_logit": target_global.get("logit"),
            "target_rank_within_color_candidates": final_field.get("target_rank_within_candidates"),
            "final_margin_vs_color_competitor": final_field.get("margin_vs_competitor"),
            "final_competitor_label": final_field.get("competitor_label"),
            "final_candidate_gate_open": final_field.get("candidate_gate_open"),
            "next_token": top_tokens[0] if top_tokens else None,
            "closed": global_closed,
            "global_closed": global_closed,
            "candidate_closed": candidate_closed,
            "layer_count_recorded": len(trace_layers),
            "raw_hidden_state_count": max_hidden,
        }

        payload = {
            "schema_version": "mechanism_trace_v1",
            "phase": PHASE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "title": "Phase944 Integrated Mechanism Trace",
            "model": model_name,
            "round_name": args.round_name,
            "prompt": prompt,
            "target": {
                "object": object_label,
                "relation": relation_label,
                "attribute": target_label,
                "category": category_label,
            },
            "positions": {
                "attribute": attribute_pos,
                "object": object_pos,
                "relation": relation_pos,
                "answer": answer_pos,
            },
            "tokens": tokens,
            "label_token_map": {
                label: {
                    "token_id": token_id,
                    "token": decode_token(tokenizer, token_id),
                }
                for label, token_id in sorted(label_to_token.items())
            },
            "pipeline": make_pipeline(),
            "factor_definitions": factor_definitions(),
            "summary": summary,
            "top_tokens": top_tokens,
            "final_color_candidates": final_candidate_rows,
            "layers": trace_layers,
            "notes": [
                "Layer readouts before the final model norm are logit-lens style approximations; the final hidden state and final logits are exact for this forward pass.",
                "The trace records the full layer-by-layer computational path and candidate readout over the selected color vocabulary, not only the largest neurons.",
                "Raw vectors are omitted unless --save-vectors is set because frontend JSON would otherwise be very large.",
            ],
        }
        return payload
    finally:
        try:
            release_model(model)
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase944 integrated embedding-to-W_U mechanism trace")
    parser.add_argument("--model", default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--round-name", default="phase944_qwen3_red_cube_trace")
    parser.add_argument("--prompt", default="A red cube is placed on the table. The color of the cube is")
    parser.add_argument("--target-label", default="red")
    parser.add_argument("--object-label", default="cube")
    parser.add_argument("--relation-label", default="color")
    parser.add_argument("--category-label", default="color")
    parser.add_argument("--colors", default=",".join(DEFAULT_COLORS))
    parser.add_argument("--protocol-labels", default=",".join(DEFAULT_PROTOCOL_LABELS))
    parser.add_argument("--termination-labels", default=".,\\n")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--max-hidden-states", type=int, default=0, help="0 records all hidden states")
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument("--export-frontend", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_trace(args)

    round_dir = RESULT_ROOT / args.round_name
    out_path = round_dir / f"phase944_{args.model}_integrated_mechanism_trace.json"
    write_json(out_path, payload)
    log(f"Wrote result: {out_path}")

    if args.export_frontend:
        public_file = PUBLIC_ROOT / f"{args.round_name}.json"
        write_json(public_file, payload)
        update_manifest(
            PUBLIC_ROOT,
            {
                "id": args.round_name,
                "label": f"{args.model} - {payload['target']['attribute']} {payload['target']['object']}",
                "phase": PHASE,
                "model": args.model,
                "prompt": payload["prompt"],
                "target": payload["target"],
                "path": f"/vis_data/mechanism_trace/{args.round_name}.json",
                "created_at": payload["created_at"],
                "summary": payload["summary"],
            },
        )
        log(f"Exported frontend data: {public_file}")

    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "ok",
                "model": args.model,
                "result": str(out_path),
                "frontend_exported": bool(args.export_frontend),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
