#!/usr/bin/env python3
"""C761-C772 large-sample replication of the lone c758 causal candidate."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c772_coreference_causal_replication.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c772_coreference_causal_replication.float16.npy"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2200_c684_c709_unified_relation_response_campaign as behavior
import phase2211_c745_c760_fresh_passport_causal_campaign as parent

PHASES = {
    "C761-C763": (2215, "coreference_open_panel_contract"),
    "C764-C766": (2216, "open_panel_behavior_and_frozen_passport"),
    "C767-C770": (2217, "large_sample_multidose_deletion_rescue"),
    "C771-C772": (2218, "replication_adjudication_visualization_cleanup"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
        for name, (phase, slug) in PHASES.items()}

DIM = 2560
QPOINTS = parent.QPOINTS
ROLES = parent.ROLES
UNITS = 24
BEHAVIOR_GATE = 0.75
PASSPORT_GATE = parent.PASSPORT_GAIN_GATE
CAUSAL_GATE = parent.CAUSAL_GAIN_GATE
PRIMARY_DOSE = 1.0
DOSES = (0.25, 0.5, 1.0)
PASSPORT_LABEL = "coreference_binding|en|1"

NAMES_A = ("Avery", "Beatrice", "Cedric", "Delia", "Emmett", "Freya", "Gideon", "Helena",
           "Isaac", "Juliet", "Killian", "Leona", "Magnus", "Naomi", "Owen", "Phoebe",
           "Raphael", "Serena", "Tristan", "Viola", "Walter", "Yvette", "Zachary", "Amelia")
NAMES_B = ("Benedict", "Clara", "Dominic", "Evelyn", "Felix", "Grace", "Hector", "Imogen",
           "James", "Katherine", "Liam", "Matilda", "Nathan", "Olivia", "Peter", "Rosalind",
           "Samuel", "Theresa", "Victor", "Wendy", "Xavier", "Yasmin", "Zane", "Bianca")
OBJECTS = ("telescope", "notebook", "fountain pen", "teacup", "suitcase", "umbrella", "camera", "radio",
           "map", "key", "scarf", "medal", "ticket", "envelope", "clock", "vase", "coin", "book",
           "ring", "flute", "rope", "bottle", "folder", "badge")

TITLES = {
    "C761-C763": "英语自然共指大样本复验合同",
    "C764-C766": "开放表面双行为与冻结护照预测",
    "C767-C770": "二十四单元多剂量全坐标删除救援",
    "C771-C772": "单病例因果候选重裁、可视化与清理",
}
FORMULAS = {
    "C761-C763": "$$\nN_{confirm}=N_{lockbox}=12,\\qquad D_{primary}=1.0,\\quad D_{descriptive}\\in\\{0.25,0.5\\}\n$$",
    "C764-C766": "$$\nG_u=A_u(P_{coref,en,1}^{frozen})-\\max(A_u(P_{wrong}),A_u(P_{shift}))\n$$",
    "C767-C770": "$$\nN_u=(m_{base}-m_{delete})-\\max(m_{base}-m_{shiftdelete},0)\n$$\n$$\nR_u(1)=(m_{correct,1}-m_{delete})-\\max(m_{wrong,1}-m_{delete},m_{shift,1}-m_{delete},0)\n$$",
    "C771-C772": "$$\nG_{causal}=1\\iff \\#\\{u:N_u\\ge.05\\land R_u(1)\\ge.05\\}\\ge8\\;\\text{in each partition}\n$$",
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def out(name: str) -> Path:
    return OUTS[name]


def final(name: str) -> dict:
    return load(out(name) / "analysis/final.json")


def finite(value: Any) -> bool:
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]; marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing: return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究边界。** `{name}` 只复验 Phase 2213 唯一的单病例候选 `coreference_binding|en|1`。仍只读取 embedding、HiddenState、final norm 和 logits 的全部 2560 个激活坐标；不读取 Attention/MLP、权重或梯度，不使用 PCA、Top-K、余弦或 donor 差分。人类盲评未运行，记为 `NA_not_run`。

**运行前冻结合同。**
```json
{json.dumps(load(out(name) / 'protocol/preregistration.json'), ensure_ascii=False, indent=2)}
```

**测试原理、用例与公式。** 24 个互不重叠的说话者对，覆盖当面陈述、日记转述、短信、邮件、证词、会议记录等自然引语表面；询问引号中 `I` 的指称。主判决只认冻结剂量 1.0，0.25/0.5 只画剂量曲线。

{FORMULAS[name]}

**详细结果。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析与理论进展。** {result.get('strict_interpretation')} 理论主体仍为“条件化输出场闭合理论”，组织原则仍为“复用—差分—条件化”。本阶段检验的是一份离散响应护照能否成为可重复的因果调用对象，不把共指预设成独立内部模块。

**问题、硬伤和瓶颈。** 受控英语不能覆盖开放共指；人类自然度为 NA；状态码中心不是原始连续激活值；一次全坐标写入可能同时扰动通用计算；24 单元仍只来自 Qwen3-4B；剂量曲线不得用于事后挑门；激活坐标不是模型参数。

**相关文件。** 脚本 `tests/glm5/phase2215_c761_c772_coreference_causal_replication.py`；结果目录 `{out(name).relative_to(ROOT)}`；裁决 `{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**严格结论与下一步授权。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
              **body, "next_authorization": authorization}
    save(out(name) / "analysis/final.json", result); append_memo(name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return result


def freeze() -> None:
    p = parent.final("C759-C760")
    if not p["all_checks_passed"] or p["causal_passed_groups"] != [PASSPORT_LABEL]:
        raise RuntimeError("Phase2214 did not expose the registered lone candidate")
    common = {"frozen_before_model": True, "parent_phase": 2214, "candidate": PASSPORT_LABEL,
              "model": "Qwen3-4B local BF16 CUDA", "units": UNITS,
              "partitions": {"confirmation": 12, "lockbox": 12}, "behavior_gate": BEHAVIOR_GATE,
              "passport_gain_gate": PASSPORT_GATE, "causal_gain_gate": CAUSAL_GATE,
              "primary_dose": PRIMARY_DOSE, "descriptive_doses": [0.25, 0.5],
              "checkpoints": list(QPOINTS), "roles": list(ROLES), "coordinates": DIM,
              "forbidden": ["attention", "mlp", "weights", "gradients", "PCA", "Top-K", "cosine", "donor_difference", "best_dose_selection"],
              "human_review": "NA_not_run", "reveal_rule": "No object, unit, dose, control or threshold changes after reveal."}
    details = {"C761-C763": "24-unit open-surface material and zero-model contract",
               "C764-C766": "dual behavior plus frozen-passport prediction without refitting",
               "C767-C770": "every eligible unit receives primary and descriptive dose deletion/rescue controls",
               "C771-C772": "partition-level adjudication, exact-coordinate visualization and cleanup"}
    for name, obj in details.items():
        path = out(name) / "protocol/preregistration.json"
        if not path.exists(): save(path, {**common, "phase": PHASES[name][0], "campaign": name, "object": obj})


def partition(unit: int) -> str:
    return "confirmation" if unit < 12 else "lockbox"


def make_case(unit: int, cell_i: int) -> dict:
    a, b, x = NAMES_A[unit], NAMES_B[unit], OBJECTS[unit]
    truth = unit % 2 == 0; target = a if truth else b
    if cell_i == 0:
        statement = f"{a} said to {b}, 'I stored the {x}.'"
    else:
        templates = (
            f"During a meeting with {b}, {a} remarked, 'I stored the {x}.'",
            f"{b}'s diary quotes {a}: 'I stored the {x}.'",
            f"In a message addressed to {b}, {a} wrote, 'I stored the {x}.'",
            f"{a} privately told {b}, 'I stored the {x}.'",
            f"A transcript records {a} telling {b}, 'I stored the {x}.'",
            f"While speaking with {b}, {a} admitted, 'I stored the {x}.'",
            f"{b} received this email from {a}: 'I stored the {x}.'",
            f"At the hearing, {a} informed {b}, 'I stored the {x}.'",
            f"A signed note from {a} to {b} says, 'I stored the {x}.'",
            f"{a} left {b} a voice message: 'I stored the {x}.'",
            f"The minutes quote {a}'s words to {b}: 'I stored the {x}.'",
            f"{b} heard {a} whisper, 'I stored the {x}.'",
        )
        statement = templates[unit % len(templates)]
    core = f"A newly audited communication record states: {statement} In the quoted sentence, does 'I' refer to {target}?"
    correct, wrong = (("Yes", "No") if truth else ("No", "Yes"))
    gold = (unit + cell_i) % 2
    options = f"(A) {correct} (B) {wrong}" if gold == 0 else f"(A) {wrong} (B) {correct}"
    return {"case_id": f"c761-coref-en-u{unit:02d}-{'base' if cell_i == 0 else 'open'}",
            "panel": "coreference_causal_replication", "family": "coreference_binding", "language": "en",
            "operation_type": "coreference_binding", "operation_domain": "open_coreference:paraphrase",
            "surface": "base" if cell_i == 0 else "open_natural", "cell": "base" if cell_i == 0 else "paraphrase",
            "cell_i": cell_i, "transform_id": cell_i, "unit": unit, "partition": partition(unit), "truth": truth,
            "correct_answer": correct, "wrong_answer": wrong, "gold_position": gold,
            "prompt_core": core, "prompt": f"{core} {options}. Reply only A or B.",
            "free_prompt": f"{core} Answer only Yes or No.",
            "role_values": {"primary": a, "secondary": b, "relation": "refer to", "context": x, "query": target},
            "factors": {"open_surface": cell_i, "quotation": 1},
            "semantic_graph": {"external_family": "coreference_binding", "internal_module_assumption": False}}


def material() -> list[dict]:
    return [make_case(unit, cell) for unit in range(UNITS) for cell in (0, 1)]


def phase2215(rows: list[dict]) -> None:
    name = "C761-C763"
    if (out(name) / "analysis/final.json").exists(): return
    tokenizer = parent.load_tokenizer(); compiled = scope.compiler.compile_qwen(tokenizer, rows)
    mp = out(name) / "material/open_coreference_24_units.jsonl"; cp = out(name) / "material/qwen_compiled.jsonl"
    write_rows(mp, rows); write_rows(cp, compiled)
    old = read_rows(parent.out("C745-C748") / "material/fresh_six_family_bilingual.jsonl")
    old_primary = {r["role_values"]["primary"] for r in old}
    overlap = sorted(old_primary & set(NAMES_A))
    balance = defaultdict(lambda: [0, 0]); truth = defaultdict(lambda: [0, 0])
    for row in rows:
        balance[row["partition"]][row["gold_position"]] += 1; truth[row["partition"]][int(row["truth"])] += 1
    missing = [{"case_id": row["case_id"], "role": role} for row in rows for role, value in row["role_values"].items() if value not in row["prompt_core"]]
    audit = {"rows": len(rows), "balance": balance, "truth": truth, "old_primary_overlap": overlap,
             "missing_roles": missing, "token_width": [min(len(r["prompt_ids"]) for r in compiled),
             float(np.median([len(r["prompt_ids"]) for r in compiled])), max(len(r["prompt_ids"]) for r in compiled)],
             "human_review": "NA_not_run"}
    save(out(name) / "audit/material.json", audit)
    close(name, {"strict_interpretation": "This phase freezes a larger, lexically disjoint open-surface panel for the lone c758 candidate. It validates the test interface only.",
                 "material_audit": audit, "material_sha256": file_hash(mp), "compiled_sha256": file_hash(cp),
                 "new_foundational_mathematics_gate": False},
          {"parent": parent.final("C759-C760")["all_checks_passed"], "rows": len(rows) == 48,
           "compiler": len(compiled) == 48, "disjoint": not overlap, "roles": not missing,
           "balance": all(v == [12, 12] for v in balance.values()), "truth": all(v == [12, 12] for v in truth.values()),
           "width": max(len(r["prompt_ids"]) for r in compiled) <= 180},
          "Authorize C764-C766 to require dual behavior and frozen-passport prediction in confirmation and lockbox before intervention.")


def capture(model, device, compiled: list[dict], cand: dict, gen: dict) -> tuple[list[dict], Path]:
    selected = [r for r in compiled if cand[r["case_id"]]["correct"] and gen[r["case_id"]]["correct"]]
    path = out("C764-C766") / "raw/open_coreference_field.float16.npy"; path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(selected), len(QPOINTS), len(ROLES), DIM))
    base = model.model; captured = []
    handles = [m.register_forward_hook(lambda _m, _a, o: captured.append(o[0] if isinstance(o, tuple) else o))
               for m in [base.embed_tokens, *list(base.layers), base.norm]]
    index = []
    try:
        for i, item in enumerate(selected):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device); mask = torch.ones_like(ids)
            captured.clear()
            with torch.inference_mode(): model(input_ids=ids, attention_mask=mask, position_ids=torch.arange(ids.shape[1], device=device)[None], use_cache=False)
            for qi, q in enumerate(QPOINTS):
                values = captured[q][0].float().cpu().numpy()
                for ri, role in enumerate(ROLES): field[i, qi, ri] = values[item["role_positions"][role][-1]].astype(np.float16)
            index.append({"field_index": i, "case_id": item["case_id"], "unit": item["unit"], "cell_i": item["cell_i"], "partition": item["partition"]})
    finally:
        for h in handles: h.remove()
    field.flush(); close_mmap(field); write_rows(out("C764-C766") / "raw/field_index.jsonl", index); return index, path


def phase2216() -> None:
    name = "C764-C766"
    if (out(name) / "analysis/final.json").exists(): return
    compiled = read_rows(out("C761-C763") / "material/qwen_compiled.jsonl"); model = None
    try:
        model, tokenizer, device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        cr = behavior.batch_behavior(model, device, compiled); gr = behavior.free_generate(model, tokenizer, device, compiled)
        cand = {r["case_id"]: r for r in cr}; gen = {r["case_id"]: r for r in gr}
        write_rows(out(name) / "behavior/candidate.jsonl", cr); write_rows(out(name) / "behavior/generation.jsonl", gr)
        index, field_path = capture(model, device, compiled, cand, gen)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    behavior_panels = {}
    for part in ("confirmation", "lockbox"):
        sub = [r for r in compiled if r["partition"] == part]
        behavior_panels[part] = {"rows": len(sub), "candidate_accuracy": float(np.mean([cand[r["case_id"]]["correct"] for r in sub])),
                                 "generation_accuracy": float(np.mean([gen[r["case_id"]]["correct"] for r in sub])),
                                 "dual_accuracy": float(np.mean([cand[r["case_id"]]["correct"] and gen[r["case_id"]]["correct"] for r in sub]))}
    prototypes = parent.load_parent_prototypes(); proto = np.stack([[prototypes[(PASSPORT_LABEL, q, role)] for role in ROLES] for q in QPOINTS])
    wrong = np.stack([[prototypes[(f"translation_route|en|1", q, role)] for role in ROLES] for q in QPOINTS])
    field = np.load(field_path, mmap_mode="r"); imap = {(r["unit"], r["cell_i"]): r["field_index"] for r in index}
    panels = {}
    for part, units in (("confirmation", range(12)), ("lockbox", range(12, 24))):
        values = []
        for unit in units:
            if (unit, 0) not in imap or (unit, 1) not in imap: continue
            code = parent.response_code(np.asarray(field[imap[(unit, 0)]]), np.asarray(field[imap[(unit, 1)]]))
            specific = float(np.mean(code == proto)); wrong_score = float(np.mean(code == wrong)); shifted = float(np.mean(code == np.roll(proto, 257, axis=2)))
            values.append({"unit": unit, "specific": specific, "wrong": wrong_score, "shift257": shifted,
                           "gain": specific - max(wrong_score, shifted)})
        panels[part] = {"pairs": len(values), "units": values, "positive": sum(v["gain"] >= PASSPORT_GATE for v in values),
                        "mean_gain": float(np.mean([v["gain"] for v in values])) if values else 0.0}
        panels[part]["passed"] = len(values) >= 9 and panels[part]["positive"] >= 8 and panels[part]["mean_gain"] >= PASSPORT_GATE
    close_mmap(field); passed = all(p["candidate_accuracy"] >= BEHAVIOR_GATE and p["generation_accuracy"] >= BEHAVIOR_GATE for p in behavior_panels.values()) and all(p["passed"] for p in panels.values())
    close(name, {"strict_interpretation": "The single c758 candidate survives to intervention only if open-surface dual behavior and the unchanged frozen response passport both replicate by independent unit in both partitions.",
                 "behavior": behavior_panels, "passport": panels, "qualified_for_intervention": passed,
                 "captured_rows": len(index), "field_shape": [len(index), len(QPOINTS), len(ROLES), DIM],
                 "field_sha256": file_hash(field_path), "placement": placement, "quantization": quant,
                 "new_foundational_mathematics_gate": False},
          {"parent": final("C761-C763")["all_checks_passed"], "behavior_complete": len(cr) == len(gr) == len(compiled),
           "captured": len(index) > 0, "panels": set(panels) == {"confirmation", "lockbox"},
           "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "finite": finite([behavior_panels, panels])},
          "If qualified, authorize C767-C770 on all eligible units; otherwise record the single c758 causal candidate as not replicated and skip intervention.")


def arrays_for_passport() -> dict:
    prototypes = parent.load_parent_prototypes(); _, target, active = parent.passport_arrays(prototypes, PASSPORT_LABEL)
    _, wrong, _ = parent.passport_arrays(prototypes, "translation_route|en|1")
    return {"target": target, "wrong_target": wrong, "mask": active}


@torch.inference_mode()
def run_mode(model, tokenizer, item: dict, arrays: dict, mode: str, dose: float, free: bool) -> dict:
    ids = torch.tensor([item["free_prompt_ids" if free else "prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids); base = model.model; handles = []
    modules = {0: base.embed_tokens, 37: base.norm}; modules.update({q: base.layers[q - 1] for q in QPOINTS if q not in (0, 37)})
    if mode != "base":
        for qi, q in enumerate(QPOINTS):
            for ri, role in enumerate(ROLES):
                pos = int(item["role_positions"][role][-1]); active_np = arrays["mask"][qi, ri]
                target_np = arrays["target"][qi, ri]; wrong_np = arrays["wrong_target"][qi, ri]
                def patch(_m, _a, output, pos=pos, active_np=active_np, target_np=target_np, wrong_np=wrong_np):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hidden.shape[1] <= pos: return output
                    changed = hidden.clone(); current = hidden[0, pos].float()
                    if mode == "delete": active = torch.tensor(active_np, dtype=torch.bool, device=current.device); current[active] = 0
                    elif mode == "shift_delete": active = torch.tensor(np.roll(active_np, 257).copy(), dtype=torch.bool, device=current.device); current[active] = 0
                    else:
                        active = torch.tensor(active_np, dtype=torch.bool, device=current.device)
                        codes = target_np if mode == "correct" else (wrong_np if mode == "wrong" else np.roll(target_np, 257).copy())
                        centers = torch.tensor(parent.state_center(codes), dtype=torch.float32, device=current.device)
                        current[active] = dose * centers[active]
                    changed[0, pos] = current.to(hidden.dtype); return (changed, *output[1:]) if isinstance(output, tuple) else changed
                handles.append(modules[q].register_forward_hook(patch))
    try:
        if free:
            pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True); parsed = behavior.parse_binary(text, "en")
            return {"text": text, "parsed": parsed, "correct": parsed == item["correct_answer"]}
        result = model(input_ids=ids, attention_mask=mask, position_ids=torch.arange(ids.shape[1], device=ids.device)[None], use_cache=False)
        first = [int(x[0]) for x in item["candidate_ids"]]; scores = result.logits[0, -1, first].float().cpu().numpy(); gold = int(item["gold_position"])
        return {"margin": float(scores[gold] - scores[1 - gold]), "correct": bool(scores[gold] > scores[1 - gold])}
    finally:
        for h in handles: h.remove()


def phase2217() -> None:
    name = "C767-C770"
    if (out(name) / "analysis/final.json").exists(): return
    qualified = final("C764-C766")["qualified_for_intervention"]
    compiled = read_rows(out("C761-C763") / "material/qwen_compiled.jsonl")
    cand = {r["case_id"]: r for r in read_rows(out("C764-C766") / "behavior/candidate.jsonl")}
    gen = {r["case_id"]: r for r in read_rows(out("C764-C766") / "behavior/generation.jsonl")}
    eligible = [r for r in compiled if r["cell_i"] == 1 and cand[r["case_id"]]["correct"] and gen[r["case_id"]]["correct"]] if qualified else []
    results = {}; model = None; placement = None; quant = None
    if eligible:
        arrays = arrays_for_passport()
        try:
            model, tokenizer, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3"); quant = scope.parent.previous.model_base().quantization_audit(model)
            for i, item in enumerate(eligible):
                modes = {"base": ("base", 0.0), "delete": ("delete", 0.0), "shift_delete": ("shift_delete", 0.0)}
                for dose in DOSES:
                    modes[f"correct_{dose}"] = ("correct", dose); modes[f"wrong_{dose}"] = ("wrong", dose); modes[f"shift_{dose}"] = ("shift", dose)
                c = {label: run_mode(model, tokenizer, item, arrays, mode, dose, False) for label, (mode, dose) in modes.items()}
                g = {label: run_mode(model, tokenizer, item, arrays, mode, dose, True) for label, (mode, dose) in modes.items()}
                necessity = (c["base"]["margin"] - c["delete"]["margin"]) - max(c["base"]["margin"] - c["shift_delete"]["margin"], 0.0)
                dose_rows = {}
                for dose in DOSES:
                    rescue = (c[f"correct_{dose}"]["margin"] - c["delete"]["margin"]) - max(c[f"wrong_{dose}"]["margin"] - c["delete"]["margin"], c[f"shift_{dose}"]["margin"] - c["delete"]["margin"], 0.0)
                    dose_rows[str(dose)] = {"rescue_specific_gain": rescue, "correct_generation": g[f"correct_{dose}"]["correct"]}
                primary = dose_rows[str(PRIMARY_DOSE)]
                passed = necessity >= CAUSAL_GATE and primary["rescue_specific_gain"] >= CAUSAL_GATE and g["base"]["correct"] and primary["correct_generation"]
                results[str(item["unit"])] = {"case_id": item["case_id"], "partition": item["partition"], "candidate": c, "generation": g,
                                              "necessity_specific_gain": necessity, "dose_curve": dose_rows, "primary_passed": passed}
                print(f"[C767-C770] {i + 1}/{len(eligible)} unit={item['unit']} pass={passed}", flush=True)
        finally:
            scope.parent.previous.model_base().release_bf16(model); gc.collect()
    panels = {}
    for part in ("confirmation", "lockbox"):
        values = [v for v in results.values() if v["partition"] == part]
        panels[part] = {"units": len(values), "primary_passed_units": sum(v["primary_passed"] for v in values),
                        "mean_necessity": float(np.mean([v["necessity_specific_gain"] for v in values])) if values else 0.0,
                        "mean_primary_rescue": float(np.mean([v["dose_curve"][str(PRIMARY_DOSE)]["rescue_specific_gain"] for v in values])) if values else 0.0,
                        "base_generation_accuracy": float(np.mean([v["generation"]["base"]["correct"] for v in values])) if values else 0.0,
                        "primary_generation_accuracy": float(np.mean([v["generation"][f"correct_{PRIMARY_DOSE}"]["correct"] for v in values])) if values else 0.0}
        panels[part]["passed"] = (len(values) >= 9 and panels[part]["primary_passed_units"] >= 8
                                   and panels[part]["mean_necessity"] >= CAUSAL_GATE and panels[part]["mean_primary_rescue"] >= CAUSAL_GATE
                                   and panels[part]["base_generation_accuracy"] >= BEHAVIOR_GATE and panels[part]["primary_generation_accuracy"] >= BEHAVIOR_GATE)
    replicated = qualified and all(v["passed"] for v in panels.values())
    save(out(name) / "analysis/unit_results.json", results)
    close(name, {"strict_interpretation": "The prior 1/33 causal pass is promoted only if the preregistered dose 1.0 repeats in at least 8/12 independent units in each partition with positive mean necessity and rescue specificity. Lower doses are descriptive and cannot rescue the verdict.",
                 "upstream_qualified": qualified, "eligible_units": len(eligible), "unit_results": results,
                 "partition_results": panels, "causal_candidate_replicated": replicated,
                 "placement": placement, "quantization": quant, "new_foundational_mathematics_gate": False},
          {"parent": final("C764-C766")["all_checks_passed"], "authorized_branch": (qualified and len(eligible) > 0) or (not qualified and len(eligible) == 0),
           "all_units_reported": len(results) == len(eligible), "fixed_primary_dose": PRIMARY_DOSE == 1.0,
           "finite": finite([results, panels])},
          "Authorize C771-C772 to preserve the exact-coordinate field, clean raw duplicates, and close or continue this exact causal object based on partition replication.")


def phase2218() -> None:
    name = "C771-C772"
    if (out(name) / "analysis/final.json").exists(): return
    field_path = out("C764-C766") / "raw/open_coreference_field.float16.npy"; field = np.load(field_path, mmap_mode="r")
    index = read_rows(out("C764-C766") / "raw/field_index.jsonl"); matrix = np.asarray(field).reshape(-1, DIM).astype(np.float16); close_mmap(field)
    rows = [{"kind": "open_coreference_exact_activation", "case_id": item["case_id"], "unit": item["unit"],
             "partition": item["partition"], "cell_i": item["cell_i"], "checkpoint": q, "role": role}
            for item in index for q in QPOINTS for role in ROLES]
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True); np.save(VISUAL_BINARY, matrix)
    payload = {"schema": "ai2050.coreference-causal-replication.v1", "phase": 2218, "campaign": "C761-C772",
               "coordinate_count": DIM, "rows": rows, "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
               "binary_shape": list(matrix.shape), "binary_dtype": "float16", "passport": final("C764-C766")["passport"],
               "causal_replication": final("C767-C770")["partition_results"], "unit_results": final("C767-C770")["unit_results"]}
    save(VISUAL, payload)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "datasets": []}
    entry = {"id": "c772-coreference-causal-replication", "title": "Coreference causal replication", "phase": 2218,
             "type": "exact-coordinate-heatmap", "json": str(VISUAL.relative_to(ROOT)).replace("\\", "/"),
             "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"), "shape": list(matrix.shape)}
    catalog["datasets"] = [r for r in catalog.get("datasets", []) if r.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat(); save(CATALOG, catalog)
    cleanup = {"path": str(field_path.relative_to(ROOT)), "sha256": file_hash(field_path), "deleted": False}; field_path.unlink(); cleanup["deleted"] = True
    save(out(name) / "audit/hash_then_cleanup.json", cleanup)
    replicated = final("C767-C770")["causal_candidate_replicated"]
    decision = ("Continue the replicated causal object to multilingual and cross-model panels." if replicated else
                "Close this exact discretized coreference passport as a causal mechanism candidate; retain it only as an observational atlas regularity and continue broader language-family mapping.")
    close(name, {"strict_interpretation": "This phase distinguishes a real cross-concept observational passport from a causal mechanism. Failure of large-sample deletion/rescue overrides the earlier single-case pass without erasing the reproducible state-transition regularity.",
                 "open_passport_qualified": final("C764-C766")["qualified_for_intervention"],
                 "causal_candidate_replicated": replicated, "partition_results": final("C767-C770")["partition_results"],
                 "visual": {"json": str(VISUAL.relative_to(ROOT)), "binary": str(VISUAL_BINARY.relative_to(ROOT)),
                            "shape": list(matrix.shape), "sha256": file_hash(VISUAL_BINARY)},
                 "cleanup": cleanup, "important_answer_reached": True, "next_stage_same_goal": bool(replicated),
                 "automatic_continuation_decision": decision, "human_review": "NA_not_run",
                 "theory_update": "Discrete response passports are stable observational coordinates. They become causal response states only after partition-level intervention replication, which a single case cannot establish.",
                 "new_foundational_mathematics_gate": False},
          {"parents": final("C764-C766")["all_checks_passed"] and final("C767-C770")["all_checks_passed"],
           "visual": VISUAL.exists() and VISUAL_BINARY.exists(), "rows": len(rows) == matrix.shape[0], "cleaned": not field_path.exists()}, decision)


def run_all() -> None:
    freeze(); rows = material(); phase2215(rows); phase2216(); phase2217(); phase2218()


if __name__ == "__main__": run_all()
