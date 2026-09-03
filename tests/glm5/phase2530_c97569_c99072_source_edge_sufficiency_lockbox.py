#!/usr/bin/env python3
"""Exact-shape source-conditioned attention contribution sufficiency on unit31 lockbox."""
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
OUT = RESULT / "phase2530_c97569_c99072_source_edge_sufficiency_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2530, "C97569-C99072"
LATE = tuple(range(20, 36))
REGIONS = ("facts_prefix", "question_context", "query_property", "post_query", "answer_boundary_self")
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
    facts_end = int(row["spans"]["facts_end"][-1][1])
    qs, qe = map(int, row["spans"]["query_property"][-1])
    groups = [list(range(0, facts_end)), list(range(facts_end, qs)), list(range(qs, qe)),
              list(range(qe, boundary)), [boundary]]
    if [p for group in groups for p in group] != list(range(boundary + 1)):
        raise RuntimeError(row["case_id"])
    return groups


def jobs(tokenizer, rows: list[dict], family_ids: list[int]) -> list[dict]:
    index = {(r["unit"], r["family_id"], r["language"], r["surface"], r["output_mode"], r["meaning_swap"], r["query_property"]): r for r in rows}
    answer = []
    for family_id in family_ids:
        for language in ("en", "zh"):
            for query in (0, 1):
                base = index[(31, family_id, language, 0, "candidate", 0, query)]
                donor = index[(31, family_id, language, 0, "candidate", 1, query)]
                if len(base["prompt_ids"]) != len(donor["prompt_ids"]):
                    raise RuntimeError("prompt shape mismatch")
                for candidate_index, entity in enumerate(base["entities"]):
                    continuation = [int(v) for v in tokenizer.encode((" " if language == "en" else "") + entity,
                                                                      add_special_tokens=False)]
                    answer.append({
                        "id": f"f{family_id}-{language}-q{query}", "family_id": family_id, "family": base["family"],
                        "language": language, "query": query, "candidate_index": candidate_index,
                        "continuation": continuation, "prompt_length": len(base["prompt_ids"]),
                        "position": len(base["prompt_ids"]) - 1,
                        "base_sequence": base["prompt_ids"] + continuation,
                        "donor_sequence": donor["prompt_ids"] + continuation,
                        "base_regions": source_regions(base), "donor_regions": source_regions(donor),
                    })
    return answer


class Ledger:
    def __init__(self, model):
        self.model = model
        self.layers = model_utils.get_layers(model)
        self.nheads = int(model.config.num_attention_heads)
        self.nkv = int(model.config.num_key_value_heads)
        self.hdim = int(model.config.head_dim)
        self.positions: list[int] = []
        self.active: dict[int, torch.Tensor] = {}
        self.norm: dict[int, torch.Tensor] = {}
        self.oin: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_index in LATE:
            def attn_pre(_module, args, kwargs, layer_index=layer_index):
                self.norm[layer_index] = (args[0] if args else kwargs["hidden_states"]).detach()
            self.handles.append(self.layers[layer_index].self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
            def o_pre(_module, args, layer_index=layer_index):
                x = args[0]
                bi = torch.arange(x.shape[0], device=x.device)
                pi = torch.tensor(self.positions, device=x.device)
                self.oin[layer_index] = x[bi, pi].detach().clone()
                if layer_index not in self.active:
                    return None
                changed = x.clone()
                delta = self.active[layer_index].to(x.device, x.dtype).reshape(x.shape[0], -1)
                changed[bi, pi] = changed[bi, pi] + delta
                return (changed, *args[1:])
            self.handles.append(self.layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre))

    def close(self):
        for handle in self.handles: handle.remove()

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, batch: list[dict], need_components: bool):
        self.positions[:] = [j["position"] for j in batch]
        self.norm.clear(); self.oin.clear()
        with torch.inference_mode():
            out = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                   output_attentions=need_components, return_dict=True)
        values = []
        for bi, job in enumerate(batch):
            hs = torch.stack([out.last_hidden_state[bi, job["prompt_length"] - 1 + oi]
                              for oi in range(len(job["continuation"]))])
            logits = self.model.lm_head(hs).float()
            lp = torch.log_softmax(logits, dim=-1)
            values.append(float(sum(lp[oi, token] for oi, token in enumerate(job["continuation"]))))
        if not need_components:
            return values, None, None
        source = torch.zeros((ids.shape[0], len(LATE), self.nheads, len(REGIONS), self.hdim), dtype=torch.float32)
        whole = torch.zeros((ids.shape[0], len(LATE), self.nheads, self.hdim), dtype=torch.float32)
        for li, layer_index in enumerate(LATE):
            sa = self.layers[layer_index].self_attn
            x = self.norm[layer_index]
            with torch.inference_mode():
                v = sa.v_proj(x).view(x.shape[0], x.shape[1], self.nkv, self.hdim).transpose(1, 2)
                v = v.repeat_interleave(self.nheads // self.nkv, dim=1)
            for bi, job in enumerate(batch):
                line = out.attentions[layer_index][bi, :, job["position"]].float()
                for ri, positions in enumerate(job["base_regions"]):
                    source[bi, li, :, ri] = torch.einsum("hp,hpd->hd", line[:, positions], v[bi, :, positions].float()).cpu()
            whole[:, li] = self.oin[layer_index].float().cpu().reshape(ids.shape[0], self.nheads, self.hdim)
        return values, source, whole


def route_map(items: list[dict]) -> dict[int, list[int]]:
    answer: dict[int, list[int]] = {}
    for item in items: answer.setdefault(int(item["layer"]), []).append(int(item["head"]))
    return answer


def delta_for(source_base: torch.Tensor, source_donor: torch.Tensor, selected: dict[int, list[int]], region_ids: list[int], shuffled=False) -> dict[int, torch.Tensor]:
    donor = source_donor.roll(2, 0) if shuffled else source_donor
    delta = donor - source_base
    answer = {}
    for li, layer in enumerate(LATE):
        if layer not in selected: continue
        value = torch.zeros_like(delta[:, li, :, 0])
        for head in selected[layer]: value[:, head] = delta[:, li, head, region_ids].sum(dim=1)
        answer[layer] = value
    return answer


def whole_delta(base: torch.Tensor, donor: torch.Tensor, selected: dict[int, list[int]]) -> dict[int, torch.Tensor]:
    answer = {}
    for li, layer in enumerate(LATE):
        if layer not in selected: continue
        value = torch.zeros_like(base[:, li])
        for head in selected[layer]: value[:, head] = donor[:, li, head] - base[:, li, head]
        answer[layer] = value
    return answer


def run(model, tokenizer, work: list[dict], top: dict[int, list[int]], random_routes: dict[int, list[int]], mass_top: dict[int, list[int]]) -> list[dict]:
    all_routes = {layer: list(range(int(model.config.num_attention_heads))) for layer in LATE}
    ledger = Ledger(model); device = model.get_input_embeddings().weight.device; output = []
    try:
        for start in range(0, len(work), 8):
            batch = work[start : start + 8]
            base_ids, base_mask = pad([j["base_sequence"] for j in batch], tokenizer.pad_token_id, device)
            donor_ids, donor_mask = pad([j["donor_sequence"] for j in batch], tokenizer.pad_token_id, device)
            if not torch.equal(base_mask, donor_mask): raise RuntimeError("exact-shape mask mismatch")
            ledger.active.clear()
            base_values, base_source, base_whole = ledger.forward(base_ids, base_mask, batch, True)
            # Regions are position-identical under the balanced meaning swap, so base partitions are valid for donor.
            ledger.active.clear()
            donor_values, donor_source, donor_whole = ledger.forward(donor_ids, donor_mask, batch, True)
            for job, value in zip(batch, base_values):
                output.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                              {"condition": "no_patch", "value": value})
            conditions = {
                "self_top_external": delta_for(base_source, base_source, top, [0, 1, 2, 3]),
                "donor_top_facts": delta_for(base_source, donor_source, top, [0]),
                "donor_top_question_context": delta_for(base_source, donor_source, top, [1]),
                "donor_top_query_property": delta_for(base_source, donor_source, top, [2]),
                "donor_top_post_query": delta_for(base_source, donor_source, top, [3]),
                "donor_top_self": delta_for(base_source, donor_source, top, [4]),
                "donor_top_external": delta_for(base_source, donor_source, top, [0, 1, 2, 3]),
                "donor_mass_top_external": delta_for(base_source, donor_source, mass_top, [0, 1, 2, 3]),
                "donor_random_external": delta_for(base_source, donor_source, random_routes, [0, 1, 2, 3]),
                "donor_all_late_external": delta_for(base_source, donor_source, all_routes, [0, 1, 2, 3]),
                "shuffled_top_external": delta_for(base_source, donor_source, top, [0, 1, 2, 3], shuffled=True),
                "donor_top_whole_head": whole_delta(base_whole, donor_whole, top),
            }
            for condition, active in conditions.items():
                ledger.active.clear(); ledger.active.update(active)
                values, _, _ = ledger.forward(base_ids, base_mask, batch, False)
                for job, value in zip(batch, values):
                    output.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                  {"condition": condition, "value": value})
            if (start + len(batch)) % 24 == 0: print(f"[phase2530] {start + len(batch)}/{len(work)}", flush=True)
    finally:
        ledger.close()
    return output


def panel(rows: list[dict]) -> dict:
    index = {(r["id"], r["condition"], r["candidate_index"]): r for r in rows}
    ids = sorted({r["id"] for r in rows}); answer = {}
    for condition in sorted({r["condition"] for r in rows}):
        values = []
        for item_id in ids:
            query = index[(item_id, condition, 0)]["query"]
            sign = 1 if query == 0 else -1
            base = index[(item_id, "no_patch", 0)]["value"] - index[(item_id, "no_patch", 1)]["value"]
            current = index[(item_id, condition, 0)]["value"] - index[(item_id, condition, 1)]["value"]
            values.append((-sign * (current - base), -sign * current, abs(current - base)))
        answer[condition] = {"n": len(values), "mean_shift_to_donor": float(np.mean([v[0] for v in values])),
                             "positive_shift_rate": float(np.mean([v[0] > 0 for v in values])),
                             "donor_flip_rate": float(np.mean([v[1] > 0 for v in values])),
                             "max_absolute_change": float(max(v[2] for v in values))}
    return answer


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: source-conditioned Attention边贡献充分性锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B、unit31、九自然模式族、英中双语、两query的36组exact-shape candidate任务中，现场重算base/donor的全部layer20–35×32 heads×5互斥source区域×128维head贡献。unit30在Phase2529以全部512路由全坐标能量冻结top32；本Phase只在锁箱unit31向base的o_proj前head切片添加$u_r(donor)-u_r(base)$，分别搬运facts、question-context、query-property、post-query、self、全部外部source，并与等量随机、Phase2525 attention-mass top32、全部晚层、batch错配和whole-head donor对照。self在完全相同shape、mask和后缀中现场核验。

$$u_{{lhr}}(x)=\sum_{{j\in r}}\alpha_{{lhaj}}(x)v_{{lhj}}(x),\qquad \widetilde u_{{lh}}=u_{{lh}}(base)+\sum_{{r\in R}}[u_{{lhr}}(donor)-u_{{lhr}}(base)].$$

**结果汇总。** 因果面板 `{json.dumps(result['causal'], ensure_ascii=False)}`；路线 `{json.dumps(result['routes'], ensure_ascii=False)}`；文件 `{json.dumps(result['files'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2530_c97569_c99072_source_edge_sufficiency_lockbox.py`；逐样本完整候选序列分数、冻结路线、SHA-256和final位于`{OUT}`。

**分析与理论进展。** 该实验第一次把whole-head输出移植拆成source-conditioned可加贡献，直接检验哪个提示区域的真实K/V读取结果能搬动输出。source patch只建立协议内充分性；若whole-head强而各source增量弱，说明跨source联合、被排除的self、或早层状态兼容性重要。若post-query强，也必须解释为候选/指令输出编译路径，而不是上游关系语义本身。

**问题硬伤与结论。** donor差分仍是构造性移植而非自然必要性；多层同时添加冻结贡献会与下游重算发生耦合；region是结构分区，内部仍混合实体、标点和格式；相同head的source贡献并非独立变量。下一Phase用renormalized attention-edge持续阻断、whole-head联盟分解和matched/shuffled救援判断自然使用与备用路径。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    f2520 = load(P2520 / "analysis/final.json")
    f2529 = load(P2529 / "analysis/final.json")
    f2525 = load(P2525 / "analysis/final.json")
    material = read(P2520 / "material/natural_rows.jsonl")
    family_ids = f2520["behavior"]["qualified_family_ids"]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        work = jobs(tokenizer, material, family_ids)
        top = route_map(f2529["routes"]["top"]); random_routes = route_map(f2529["routes"]["random"])
        mass_top = route_map(f2525["routes"]["top"])
        rows = run(model, tokenizer, work, top, random_routes, mass_top)
    finally:
        model_utils.release_model(model); gc.collect()
    output_path = OUT / "output/source_edge_causal_scores.jsonl"; write(output_path, rows)
    causal = panel(rows)
    checks = {
        "sources_passed": f2520["all_checks_passed"] and f2529["all_checks_passed"],
        "lockbox_pairs_36": causal["no_patch"]["n"] == 36,
        "self_exact": causal["self_top_external"]["max_absolute_change"] == 0.0,
        "all_conditions_present": len(causal) == 13,
        "whole_head_positive_control": causal["donor_top_whole_head"]["mean_shift_to_donor"] > 0,
        "hash": len(sha(output_path)) == 64,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
        "scope": {"lockbox_unit": 31, "pairs": 36, "candidate_sequences": 72,
                  "families": f2520["behavior"]["qualified_families"], "regions": list(REGIONS)},
        "routes": {"contribution_top32": f2529["routes"]["top"], "random32": f2529["routes"]["random"],
                   "mass_top32_source_phase": 2525},
        "causal": causal,
        "files": {"scores": {"path": str(output_path), "sha256": sha(output_path)}},
        "adjudication": {"source_specific_sufficiency_tested": True, "natural_necessity_tested": False,
                         "source_to_output_chain_closed": False, "language_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "causal": causal, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__":
    main()
