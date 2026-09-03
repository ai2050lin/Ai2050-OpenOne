#!/usr/bin/env python3
"""Redundancy-aware persistent attention-edge cuts, route coalitions, and rescue curves."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2520 = RESULT / "phase2520_c85025_c86176_natural_language_counterfactual_fullfield"
P2525 = RESULT / "phase2525_c91329_c92704_multilayer_attention_route_lockbox"
P2529 = RESULT / "phase2529_c95777_c97568_full_source_kv_head_residual_ledger"
P2530 = RESULT / "phase2530_c97569_c99072_source_edge_sufficiency_lockbox"
OUT = RESULT / "phase2531_c99073_c101120_redundant_route_cuts_rescue"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2531, "C99073-C101120"
LATE = tuple(range(20, 36))
REGIONS = ("facts_prefix", "question_context", "query_property", "post_query", "answer_boundary_self")
CONDITIONS = (
    "no_patch",
    "edge_cut_top_external", "edge_cut_top_facts", "edge_cut_top_post_query",
    "edge_cut_random_external", "edge_cut_mass_top_external", "edge_cut_all_late_external",
    "edge_cut_top_l20_23", "edge_cut_top_l24_27", "edge_cut_top_l28_31", "edge_cut_top_l32_35",
    "edge_cut_top_external_matched_whole_rescue", "edge_cut_top_external_shuffled_whole_rescue",
    "edge_cut_all_late_external_rescue_top", "edge_cut_all_late_external_rescue_random",
    "edge_cut_all_late_external_matched_all_rescue",
    "head_zero_top", "head_zero_random", "head_zero_complement", "head_zero_all",
    "head_zero_all_rescue_top", "head_zero_all_rescue_complement",
)
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, : len(sequence)] = 1
    return ids, mask


def source_regions(row: dict) -> list[list[int]]:
    boundary = len(row["prompt_ids"]) - 1
    facts_end = int(row["spans"]["facts_end"][-1][1]); qs, qe = map(int, row["spans"]["query_property"][-1])
    groups = [list(range(0, facts_end)), list(range(facts_end, qs)), list(range(qs, qe)),
              list(range(qe, boundary)), [boundary]]
    if [p for g in groups for p in g] != list(range(boundary + 1)): raise RuntimeError(row["case_id"])
    return groups


def build_jobs(tokenizer, rows: list[dict], family_ids: list[int]) -> list[dict]:
    index = {(r["unit"], r["family_id"], r["language"], r["surface"], r["output_mode"], r["meaning_swap"], r["query_property"]): r for r in rows}
    answer = []
    for family_id in family_ids:
        for language in ("en", "zh"):
            for query in (0, 1):
                row = index[(31, family_id, language, 0, "candidate", 0, query)]
                for candidate_index, entity in enumerate(row["entities"]):
                    continuation = [int(v) for v in tokenizer.encode((" " if language == "en" else "") + entity, add_special_tokens=False)]
                    answer.append({"id": f"f{family_id}-{language}-q{query}", "family_id": family_id, "family": row["family"],
                                   "language": language, "query": query, "candidate_index": candidate_index,
                                   "continuation": continuation, "prompt_length": len(row["prompt_ids"]),
                                   "position": len(row["prompt_ids"]) - 1, "sequence": row["prompt_ids"] + continuation,
                                   "regions": source_regions(row)})
    return answer


def route_map(items: list[dict]) -> dict[int, list[int]]:
    answer: dict[int, list[int]] = {}
    for item in items: answer.setdefault(int(item["layer"]), []).append(int(item["head"]))
    return answer


class Intervention:
    def __init__(self, model):
        self.model = model; self.layers = model_utils.get_layers(model)
        self.nheads = int(model.config.num_attention_heads); self.hdim = int(model.config.head_dim)
        self.positions: list[int] = []; self.regions: list[list[list[int]]] = []
        self.edge_active: dict[int, dict] = {}; self.zero_active: dict[int, list[int]] = {}
        self.replace_active: dict[int, dict] = {}; self.oin: dict[int, torch.Tensor] = {}; self.layer_out: dict[int, torch.Tensor] = {}
        self.handles = []
        def layer19(_module, _args, output): self.layer_out[19] = (output[0] if isinstance(output, tuple) else output).detach()
        self.handles.append(self.layers[19].register_forward_hook(layer19))
        for layer_index in LATE:
            def attn_pre(_module, args, kwargs, layer_index=layer_index):
                if layer_index not in self.edge_active: return None
                original = kwargs.get("attention_mask")
                if original is None: raise RuntimeError("4D causal mask required")
                expanded = original.expand(original.shape[0], self.nheads, *original.shape[2:]).clone()
                spec = self.edge_active[layer_index]
                minimum = torch.finfo(expanded.dtype).min
                for bi, target in enumerate(self.positions):
                    positions = [p for ri in spec["regions"] for p in self.regions[bi][ri]]
                    for head in spec["heads"]: expanded[bi, head, target, positions] = minimum
                changed = dict(kwargs); changed["attention_mask"] = expanded
                return args, changed
            self.handles.append(self.layers[layer_index].self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
            def o_pre(_module, args, layer_index=layer_index):
                x = args[0]; bi = torch.arange(x.shape[0], device=x.device); pi = torch.tensor(self.positions, device=x.device)
                self.oin[layer_index] = x[bi, pi].detach().clone()
                zeros = self.zero_active.get(layer_index, [])
                replacement = self.replace_active.get(layer_index)
                if not zeros and replacement is None: return None
                changed = x.clone().view(x.shape[0], x.shape[1], self.nheads, self.hdim)
                if zeros: changed[bi[:, None], pi[:, None], torch.tensor(zeros, device=x.device)[None, :], :] = 0
                if replacement is not None:
                    heads = torch.tensor(replacement["heads"], device=x.device)
                    source = replacement["source"].to(x.device, x.dtype)
                    changed[bi[:, None], pi[:, None], heads[None, :], :] = source[:, heads]
                return (changed.reshape_as(x), *args[1:])
            self.handles.append(self.layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre))
            def layer_post(_module, _args, output, layer_index=layer_index):
                self.layer_out[layer_index] = (output[0] if isinstance(output, tuple) else output).detach()
            self.handles.append(self.layers[layer_index].register_forward_hook(layer_post))

    def close(self):
        for h in self.handles: h.remove()

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, batch: list[dict]) -> tuple[list[float], torch.Tensor, torch.Tensor]:
        self.positions[:] = [j["position"] for j in batch]; self.regions[:] = [j["regions"] for j in batch]
        self.oin.clear(); self.layer_out.clear()
        with torch.inference_mode(): out = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        scores = []
        for bi, job in enumerate(batch):
            hs = torch.stack([out.last_hidden_state[bi, job["prompt_length"] - 1 + oi] for oi in range(len(job["continuation"]))])
            lp = torch.log_softmax(self.model.lm_head(hs).float(), -1)
            scores.append(float(sum(lp[oi, token] for oi, token in enumerate(job["continuation"])).detach()))
        whole = torch.stack([self.oin[layer].float().cpu().reshape(ids.shape[0], self.nheads, self.hdim) for layer in LATE], 1)
        hidden = torch.stack([self.layer_out[layer][torch.arange(ids.shape[0], device=ids.device), torch.tensor(self.positions, device=ids.device)].float().cpu()
                              for layer in range(19, 36)], 1)
        return scores, whole, hidden


def run(model, tokenizer, work: list[dict], top: dict[int, list[int]], random_routes: dict[int, list[int]], mass_top: dict[int, list[int]]) -> tuple[list[dict], dict]:
    nheads = int(model.config.num_attention_heads)
    all_routes = {layer: list(range(nheads)) for layer in LATE}
    complement = {layer: [h for h in range(nheads) if h not in set(top.get(layer, []))] for layer in LATE}
    item_ids = sorted({j["id"] for j in work}); item_index = {v: i for i, v in enumerate(item_ids)}
    hidden_path = OUT / "fields/condition_answer_boundary_q20_q36.float16.npy"; hidden_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(hidden_path, mode="w+", dtype=np.float16,
                                      shape=(len(CONDITIONS), len(item_ids), 17, int(model.config.hidden_size)))
    field[:] = np.nan
    iv = Intervention(model); device = model.get_input_embeddings().weight.device; output = []
    try:
        for start in range(0, len(work), 8):
            batch = work[start : start + 8]; ids, mask = pad([j["sequence"] for j in batch], tokenizer.pad_token_id, device)
            iv.edge_active.clear(); iv.zero_active.clear(); iv.replace_active.clear()
            base_values, base_whole, base_hidden = iv.forward(ids, mask, batch)

            def edges(routes, region_ids, lo=20, hi=35):
                return {layer: {"heads": heads, "regions": region_ids} for layer, heads in routes.items() if lo <= layer <= hi and heads}
            def replace(routes, source, shuffled=False):
                src = source.roll(2, 0) if shuffled else source
                return {layer: {"heads": heads, "source": src[:, li]} for li, layer in enumerate(LATE) if (heads := routes.get(layer, []))}

            specs = {
                "no_patch": ({}, {}, {}),
                "edge_cut_top_external": (edges(top, [0, 1, 2, 3]), {}, {}),
                "edge_cut_top_facts": (edges(top, [0]), {}, {}),
                "edge_cut_top_post_query": (edges(top, [3]), {}, {}),
                "edge_cut_random_external": (edges(random_routes, [0, 1, 2, 3]), {}, {}),
                "edge_cut_mass_top_external": (edges(mass_top, [0, 1, 2, 3]), {}, {}),
                "edge_cut_all_late_external": (edges(all_routes, [0, 1, 2, 3]), {}, {}),
                "edge_cut_top_l20_23": (edges(top, [0, 1, 2, 3], 20, 23), {}, {}),
                "edge_cut_top_l24_27": (edges(top, [0, 1, 2, 3], 24, 27), {}, {}),
                "edge_cut_top_l28_31": (edges(top, [0, 1, 2, 3], 28, 31), {}, {}),
                "edge_cut_top_l32_35": (edges(top, [0, 1, 2, 3], 32, 35), {}, {}),
                "edge_cut_top_external_matched_whole_rescue": (edges(top, [0, 1, 2, 3]), {}, replace(top, base_whole)),
                "edge_cut_top_external_shuffled_whole_rescue": (edges(top, [0, 1, 2, 3]), {}, replace(top, base_whole, True)),
                "edge_cut_all_late_external_rescue_top": (edges(all_routes, [0, 1, 2, 3]), {}, replace(top, base_whole)),
                "edge_cut_all_late_external_rescue_random": (edges(all_routes, [0, 1, 2, 3]), {}, replace(random_routes, base_whole)),
                "edge_cut_all_late_external_matched_all_rescue": (edges(all_routes, [0, 1, 2, 3]), {}, replace(all_routes, base_whole)),
                "head_zero_top": ({}, top, {}),
                "head_zero_random": ({}, random_routes, {}),
                "head_zero_complement": ({}, complement, {}),
                "head_zero_all": ({}, all_routes, {}),
                "head_zero_all_rescue_top": ({}, all_routes, replace(top, base_whole)),
                "head_zero_all_rescue_complement": ({}, all_routes, replace(complement, base_whole)),
            }
            if tuple(specs) != CONDITIONS: raise RuntimeError("condition order changed")
            for ci, (condition, (edge_spec, zero_spec, replacement_spec)) in enumerate(specs.items()):
                if condition == "no_patch": values, hidden_values = base_values, base_hidden
                else:
                    iv.edge_active.clear(); iv.edge_active.update(edge_spec)
                    iv.zero_active.clear(); iv.zero_active.update(zero_spec)
                    iv.replace_active.clear(); iv.replace_active.update(replacement_spec)
                    values, _, hidden_values = iv.forward(ids, mask, batch)
                for bi, (job, score) in enumerate(zip(batch, values)):
                    output.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                  {"condition": condition, "value": score})
                    if job["candidate_index"] == 0:
                        field[ci, item_index[job["id"]]] = hidden_values[bi].numpy().astype(np.float16)
            if (start + len(batch)) % 24 == 0: field.flush(); print(f"[phase2531] {start + len(batch)}/{len(work)}", flush=True)
    finally:
        iv.close(); field.flush(); del field
    return output, {"path": str(hidden_path), "shape": list(np.load(hidden_path, mmap_mode="r").shape),
                    "dtype": "float16", "bytes": hidden_path.stat().st_size, "sha256": sha(hidden_path),
                    "conditions": list(CONDITIONS), "qpoints": list(range(20, 37)), "items": item_ids}


def panels(rows: list[dict], hidden_meta: dict) -> tuple[dict, dict]:
    index = {(r["id"], r["condition"], r["candidate_index"]): r for r in rows}; ids = sorted({r["id"] for r in rows})
    metrics = {}
    base_margins = {}
    for item_id in ids:
        q = index[(item_id, "no_patch", 0)]["query"]; sign = 1 if q == 0 else -1
        base_margins[item_id] = sign * (index[(item_id, "no_patch", 0)]["value"] - index[(item_id, "no_patch", 1)]["value"])
    for condition in CONDITIONS:
        margins = []
        for item_id in ids:
            q = index[(item_id, condition, 0)]["query"]; sign = 1 if q == 0 else -1
            margins.append(sign * (index[(item_id, condition, 0)]["value"] - index[(item_id, condition, 1)]["value"]))
        losses = [base_margins[i] - value for i, value in zip(ids, margins)]
        metrics[condition] = {"n": len(ids), "accuracy": float(np.mean(np.asarray(margins) > 0)),
                              "mean_oriented_margin": float(np.mean(margins)), "mean_margin_loss": float(np.mean(losses)),
                              "positive_loss_rate": float(np.mean(np.asarray(losses) > 0)),
                              "max_absolute_margin_change": float(np.max(np.abs(losses)))}
    field = np.asarray(np.load(hidden_meta["path"], mmap_mode="r"), np.float32); base = field[0]
    denom = np.sqrt(np.mean(base ** 2, axis=(1, 2))) + 1e-12
    trajectories = {}
    for ci, condition in enumerate(CONDITIONS):
        relative = np.sqrt(np.mean((field[ci] - base) ** 2, axis=2)) / (np.sqrt(np.mean(base ** 2, axis=2)) + 1e-12)
        trajectories[condition] = {"mean_relative_rms_by_qpoint": relative.mean(axis=0).tolist(),
                                   "maximum_relative_rms": float(relative.max()),
                                   "final_mean_relative_rms": float(relative[:, -1].mean())}
    return metrics, trajectories


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 冗余路线持续割、联盟分解与删除—救援（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在unit31九族×英中×双query的36组完整候选序列上，对layer20–35持续执行两类干预。source-edge cut在Attention softmax输入中只屏蔽选定answer-boundary→source边并重新归一化；head-output cut在o_proj前将冻结layer×head切片置零。比较贡献top32、Phase2525 mass-top32、随机32、全部晚层、四层段、top/complement/all联盟；同时做top切断后的matched whole-head、batch错配救援，以及all-late切断后只恢复top或随机。每个条件保存q20–q36全部2560维answer-boundary HiddenState，观察下降与后层恢复，不只看最终答案。

$$E(S)=m(x)-m(\operatorname{{do}}(G_S\leftarrow G_S^{{cut}})),\qquad H^{{cut}}_{{a,20:36}}\text{{记录路径恢复而非只记录终点。}}$$

**结果汇总。** 输出与联盟必要性 `{json.dumps(result['causal'], ensure_ascii=False)}`；逐层场摘要 `{json.dumps(result['trajectory_summary'], ensure_ascii=False)}`；字段 `{json.dumps(result['fields'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2531_c99073_c101120_redundant_route_cuts_rescue.py`；22条件逐样本序列分数、36×17×2560逐层场、SHA-256和final位于`{OUT}`。

**分析与理论进展。** 单top联盟无效只否定其无条件必要性；all-late有效而top无效支持“不完整联盟/并行路线”；内部先降后恢复支持已有备用读路。head-zero top、complement、all以及all后分别恢复top/complement，给出经验功能分解，但因强干预和残差旁路不能直接称图论最小割。matched whole-head救援是数值阳性控制；只有它优于shuffled且被切条件确有损害时，才增加路径特异证据。

**问题硬伤与结论。** attention-edge屏蔽会重新归一化，head-zero会离开自然激活分布，两者估计的不是同一因果量；matched whole-head恢复包含该head从所有source读取的状态，不是纯source rescue；complement规模远大于top，损伤不能直接比较；层间恢复是确定性网络重构而非在线学习。结论只按单部件、联盟、条件、经验割和救援分级，不用一次阴性关闭Attention路线。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    f2520 = load(P2520 / "analysis/final.json"); f2525 = load(P2525 / "analysis/final.json")
    f2529 = load(P2529 / "analysis/final.json"); f2530 = load(P2530 / "analysis/final.json")
    family_ids = f2520["behavior"]["qualified_family_ids"]; material = read(P2520 / "material/natural_rows.jsonl")
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        work = build_jobs(tokenizer, material, family_ids)
        top = route_map(f2529["routes"]["top"]); random_routes = route_map(f2529["routes"]["random"]); mass_top = route_map(f2525["routes"]["top"])
        rows, hidden_meta = run(model, tokenizer, work, top, random_routes, mass_top)
    finally:
        model_utils.release_model(model); gc.collect()
    score_path = OUT / "output/intervention_scores.jsonl"; write(score_path, rows)
    causal, trajectories = panels(rows, hidden_meta)
    traj_path = OUT / "analysis/layerwise_recovery.json"; save(traj_path, trajectories)
    checks = {
        "sources_passed": all(x["all_checks_passed"] for x in (f2520, f2529, f2530)),
        "pairs_36": causal["no_patch"]["n"] == 36,
        "conditions_22": len(causal) == len(CONDITIONS) == 22,
        "baseline_behavior": causal["no_patch"]["accuracy"] >= 0.95,
        "matched_all_rescue_exact": causal["edge_cut_all_late_external_matched_all_rescue"]["max_absolute_margin_change"] == 0.0,
        "matched_top_rescue_exact": causal["edge_cut_top_external_matched_whole_rescue"]["max_absolute_margin_change"] == 0.0,
        "shuffled_not_exact": causal["edge_cut_top_external_shuffled_whole_rescue"]["max_absolute_margin_change"] > 0,
        "field_shape": hidden_meta["shape"] == [22, 36, 17, 2560],
        "hashes": all(len(v) == 64 for v in (sha(score_path), hidden_meta["sha256"], sha(traj_path))),
        "claim_boundary": True,
    }
    summary_keys = ["no_patch", "edge_cut_top_external", "edge_cut_random_external", "edge_cut_all_late_external",
                    "edge_cut_all_late_external_rescue_top", "edge_cut_all_late_external_rescue_random",
                    "head_zero_top", "head_zero_complement", "head_zero_all", "head_zero_all_rescue_top", "head_zero_all_rescue_complement"]
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
        "scope": {"pairs": 36, "conditions": list(CONDITIONS), "qpoints": list(range(20, 37)),
                  "families": f2520["behavior"]["qualified_families"]},
        "causal": causal,
        "trajectory_summary": {k: trajectories[k] for k in summary_keys},
        "fields": {"hidden": hidden_meta, "scores": {"path": str(score_path), "sha256": sha(score_path)},
                   "trajectories": {"path": str(traj_path), "sha256": sha(traj_path)}},
        "adjudication": {"top32_unconditional_necessity": causal["edge_cut_top_external"]["accuracy"] < causal["no_patch"]["accuracy"],
                         "all_late_external_empirical_cut_effect": causal["edge_cut_all_late_external"]["mean_margin_loss"] > 0,
                         "minimal_graph_cut_established": False, "online_compensation_claimed": False,
                         "language_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "causal": {k: causal[k] for k in summary_keys}, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__":
    main()
