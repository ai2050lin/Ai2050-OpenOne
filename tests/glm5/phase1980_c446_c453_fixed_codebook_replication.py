#!/usr/bin/env python3
"""C446-C453 fixed-codebook replication and conditional HiddenState writer."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c453_fixed_codebook_replication.json"
sys.path.insert(0, str(TESTS))

import phase1968_c434_c445_guarded_response_graph_campaign as prior
import phase1844_c310_c335_dual_axis_common as common
import phase1332_bf16_utils as model_base

PHASES = {
    f"C{campaign}": (1980 + campaign - 446, slug)
    for campaign, slug in (
        (446, "fixed_codebook_confound_adjudication"),
        (447, "fresh_fixed_codebook_material"),
        (448, "qwen_fixed_codebook_behavior"),
        (449, "qualified_fixed_codebook_full_field"),
        (450, "fixed_codebook_guarded_rule_replication"),
        (451, "candidate_pattern_adjudication_and_writer_freeze"),
        (452, "conditional_hiddenstate_writer"),
        (453, "replication_synthesis_visual_cleanup_audit"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}

DIM, CHECKPOINTS, ROLES = prior.DIM, prior.CHECKPOINTS, prior.ROLES
FAMILIES, AXIS_FAMILIES = prior.BROAD_FAMILIES, prior.AXIS_FAMILIES
CONSTRUCTIONS = ("journal", "bulletin", "record")
UNITS = (
    {"p": "Dalen", "s": "Eira", "o": "Faron", "obj": "clementine", "other": "protractor", "node": "vexa", "parent": "seta", "wrong": "seta_alt", "event": "calibration"},
    {"p": "Galen", "s": "Hira", "o": "Isen", "obj": "jicama", "other": "rangefinder", "node": "vexb", "parent": "setb", "wrong": "setb_alt", "event": "mapping"},
    {"p": "Jora", "s": "Kalen", "o": "Luma", "obj": "nectarine", "other": "sextant", "node": "vexc", "parent": "setc", "wrong": "setc_alt", "event": "survey"},
    {"p": "Mira", "s": "Nolan", "o": "Orel", "obj": "parsnip", "other": "tachometer", "node": "vexd", "parent": "setd", "wrong": "setd_alt", "event": "inspection"},
    {"p": "Peren", "s": "Quara", "o": "Rilan", "obj": "quince", "other": "theodolite", "node": "vexe", "parent": "sete", "wrong": "sete_alt", "event": "review"},
    {"p": "Sorin", "s": "Talia", "o": "Uren", "obj": "turnip", "other": "barograph", "node": "vexf", "parent": "setf", "wrong": "setf_alt", "event": "audit"},
    {"p": "Vela", "s": "Warin", "o": "Xara", "obj": "ugli", "other": "planimeter", "node": "vexg", "parent": "setg", "wrong": "setg_alt", "event": "briefing"},
    {"p": "Yalen", "s": "Zira", "o": "Arel", "obj": "yam", "other": "anemometer", "node": "vexh", "parent": "seth", "wrong": "seth_alt", "event": "inventory"},
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((name, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {"phase": PHASES[name][0], "campaign": name, "created_at_utc": datetime.now(timezone.utc).isoformat(), "producer_sha256": producer_hash(), **protocol})
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def finite(value: Any) -> bool:
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)): return math.isfinite(float(value))
    return True


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    if (out / "analysis/final.json").exists(): return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final_checks = {"contract": load(out / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": all(checks.values()), "producer_hash": load(out / "protocol/preregistration.json")["producer_sha256"] == producer_hash()}
    value = {"phase": PHASES[name][0], "campaign": name, "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": headline, "next_authorization": authorization}
    save(out / "analysis/final.json", value); print(json.dumps(value, ensure_ascii=False), flush=True); return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def part(unit: int) -> str:
    return "discovery" if unit < 4 else "confirmation" if unit < 6 else "lockbox"


def fixed_prompt(core: str, truth: bool) -> tuple[str, int, str, str]:
    return f"{core} (A) Yes (B) No. Reply with only A or B.", 0 if truth else 1, "Yes" if truth else "No", "No" if truth else "Yes"


def broad_rows() -> list[dict]:
    rows = []; original = prior.old.UNITS; prior.old.UNITS = UNITS
    try:
        for family, surface, unit, a, b in itertools.product(FAMILIES, CONSTRUCTIONS, range(len(UNITS)), (0, 1), (0, 1)):
            case = prior.old.family_statement(family, unit, a, b)
            core = prior.wrap(surface.replace("journal", "chronicle").replace("bulletin", "dispatch").replace("record", "archive"), case["target"], case["noise"], case["question"])
            prompt, gold, correct, wrong = fixed_prompt(core, bool(case["truth"]))
            rows.append({"case_id": f"c447-broad-{family}-{surface}-u{unit}-{a}{b}", "panel": "fixed_broad", "family": family, "surface": surface, "construction": surface, "unit": unit, "cell": f"{a}{b}", "factor_a": a, "factor_b": b, "order": 0, "partition": part(unit), "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt, "free_prompt": f"{core} Answer with only Yes or No.", "role_values": case["roles"], "semantic_graph": {"statement": a, "query": b, "truth": bool(case["truth"])}})
    finally:
        prior.old.UNITS = original
    return rows


def axis_rows() -> list[dict]:
    rows = []
    for family, surface, unit_i, mask in itertools.product(AXIS_FAMILIES, CONSTRUCTIONS, range(len(UNITS)), range(8)):
        d, a, b = ((mask >> bit) & 1 for bit in range(3)); u = UNITS[unit_i]
        attitude = ("likes", "dislikes")[a]; event = ("ate", "did not eat")[b]
        outer = {"attitude_event": ("reported", "denied"), "nested_attitude": ("believes", "doubts"), "negation_scope": ("affirmed", "denied")}[family][d]
        baseline = "report" if family == "attitude_event" else "believe" if family == "nested_attitude" else "affirm"
        target = f"{u['o']} {outer} that {u['p']} {attitude} the event in which {u['s']} {event} the {u['obj']}."
        question = f"Did {u['o']} {baseline} that {u['p']} {attitude} the event in which {u['s']} {event} the {u['obj']}?"
        noise = f"Separately, {u['s']} catalogued the {u['other']} for {u['p']}."
        mapped = surface.replace("journal", "chronicle").replace("bulletin", "dispatch").replace("record", "archive")
        core = prior.wrap(mapped, target, noise, question); prompt, gold, correct, wrong = fixed_prompt(core, d == 0)
        cell = f"{d}{a}{b}"
        rows.append({"case_id": f"c447-axis-{family}-{surface}-u{unit_i}-{cell}", "panel": "fixed_axis", "family": family, "surface": surface, "construction": surface, "unit": unit_i, "cell": cell, "mask": mask, "query_axis": "outer", "order": 0, "partition": part(unit_i), "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt, "free_prompt": f"{core} Answer with only Yes or No.", "role_values": {"primary": u["p"], "secondary": u["s"], "relation": outer, "context": u["obj"], "query": u["obj"]}, "semantic_graph": {"outer": d, "attitude": a, "event": b, "truth": d == 0}})
    return rows


def material() -> list[dict]: return broad_rows() + axis_rows()


def lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C447"] / "material/cases.jsonl"); return rows, {r["case_id"]: r for r in rows}


def c446() -> None:
    audit = load(prior.OUTS["C445"] / "audit/independent_audit.json")
    out = begin("C446", {"status": "fixed_codebook_replication_contract_frozen", "correction": "C439 broad query pairs also swapped displayed Yes/No option text", "new_interface": "A always means Yes and B always means No", "replication_target": "the identities of 11 candidate groups, not deleted C438 coefficients", "route_policy": "all 22 matched groups are retained"}, {"parent": audit["all_checks_passed"], "continuity": PHASES["C446"][0] == 1980})
    close("C446", {"status": "confound_adjudication_closed", "old_candidates": prior.final("C439")["headline"]["candidate_groups"], "coefficient_replication_possible": False, "pattern_replication_authorized": True, "strict_interpretation": "C439 remains a valid prompt-response regularity but is not yet a pure semantic-query rule."}, {"candidate_count": len(prior.final("C439")["headline"]["candidate_groups"]) == 11}, "C447_material")


def c447() -> None:
    out = begin("C447", {"status": "fresh_fixed_codebook_material_frozen", "design": "8 broad families plus 3 outer-axis families, 3 new wrappers, 8 new lexicons, fixed A=Yes/B=No", "partitions": {"discovery": [0,1,2,3], "confirmation": [4,5], "lockbox": [6,7]}, "naturalness": "controlled grammar only; no independent human panel"}, {"parent": final("C446")["all_checks_passed"]})
    rows = material(); write_rows(out / "material/cases.jsonl", rows)
    frequencies = {key: float(np.mean([r["gold_position"] == 0 for r in rows if r["panel"] == key])) for key in ("fixed_broad", "fixed_axis")}
    roles = all(all(str(v) in r["prompt_core"] for v in r["role_values"].values()) for r in rows)
    headline = {"status": "fresh_fixed_codebook_material_closed", "rows": len(rows), "panel_counts": {p: sum(r["panel"] == p for r in rows) for p in frequencies}, "yes_frequency": frequencies, "role_occurrence": roles, "material_eligible": all(abs(v-0.5)<1e-12 for v in frequencies.values()) and roles, "human_naturalness_review": False, "strict_interpretation": "The fixed codebook removes option permutation but remains controlled English."}
    close("C447", headline, {"rows": len(rows) == 1344, "balance": headline["material_eligible"]}, "C448_behavior")


def c448() -> None:
    out = begin("C448", {"status": "qwen_fixed_codebook_behavior_frozen", "model": "Qwen3-4B BF16 CUDA", "gates": {"heldout": 0.80, "family": 0.65, "construction": 0.70}, "hidden_state_policy": "none"}, {"parent": final("C447")["all_checks_passed"], "material": final("C447")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows, by_id = lookup(); tokenizer = prior.axis_old.base.parent.fresh.tokenizer_qwen(); compiled = prior.family_base.compile_qwen(tokenizer, rows); write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = prior.axis_old.base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12); behavior = read_rows(out / "raw/behavior.jsonl"); held = [r for r in behavior if by_id[r["case_id"]]["partition"] != "discovery"]
    by_family = {f: float(np.mean([r["correct"] for r in held if r["family"] == f])) for f in FAMILIES}
    by_surface = {s: float(np.mean([r["correct"] for r in held if r["surface"] == s])) for s in CONSTRUCTIONS}
    heldout = float(np.mean([r["correct"] for r in held])); eligible = heldout >= .80 and min(by_family.values()) >= .65 and min(by_surface.values()) >= .70
    headline = {"status": "qwen_fixed_codebook_behavior_closed", **run, "heldout_accuracy": heldout, "family_accuracy": by_family, "construction_accuracy": by_surface, "field_authorized": eligible, "strict_interpretation": "The gate qualifies this fixed output interface, not semantic mechanism."}
    close("C448", headline, {"rows": len(behavior)==len(rows), "no_hidden": not (out/"raw/role_states.float16.npy").exists(), "finite": finite(headline)}, "C449_field")


def c449() -> None:
    out = begin("C449", {"status": "qualified_fixed_codebook_full_field_frozen", "archive": "all 1344 cases x 38 checkpoints x six roles x all 2560 coordinates", "no_pca_topk": True}, {"parent": final("C448")["all_checks_passed"]})
    if not final("C448")["headline"]["field_authorized"]:
        close("C449", {"status": "field_not_run_behavior_ineligible", "field_ran": False, "strict_interpretation": "No fixed-codebook internal result."}, {"route_accounted": True}, "C450_replication"); return
    rows, _ = lookup(); compiled = read_rows(OUTS["C448"] / "compiled/qwen3.jsonl")
    selector = lambda r: r["partition"] == "lockbox" and r["surface"] == "record" and ((r["panel"] == "fixed_broad" and r["cell"] in ("00","01")) or (r["panel"] == "fixed_axis" and r["cell"] in ("000","010")))
    run = common.batch_capture_qwen(rows, compiled, out, full_selector=selector, batch_size=8, field_width=prior.FIELD_WIDTH)
    headline = {"status": "qualified_fixed_codebook_full_field_closed", **run, "field_ran": True, "strict_interpretation": "Every coordinate is retained; output-code deconfounding does not make the field causal."}
    close("C449", headline, {"shape": run["role_shape"][1:]==[38,6,2560], "finite": finite(headline)}, "C450_replication")


def response_records(index: list[dict]) -> list[dict]:
    keyed = {r["case_id"]: r for r in index}; records=[]
    for family, surface, unit, op, context in itertools.product(FAMILIES, CONSTRUCTIONS, range(len(UNITS)), ("statement","query"), (0,1)):
        if op == "statement": left,right=(f"0{context}",f"1{context}")
        else: left,right=(f"{context}0",f"{context}1")
        lid=f"c447-broad-{family}-{surface}-u{unit}-{left}"; rid=f"c447-broad-{family}-{surface}-u{unit}-{right}"
        if lid in keyed and rid in keyed: records.append({"group":f"broad:{family}:{op}","family":family,"operation":op,"surface":surface,"unit":unit,"partition":part(unit),"context":context,"left":keyed[lid]["hidden_index"],"right":keyed[rid]["hidden_index"]})
    for family, surface, unit, op, context in itertools.product(AXIS_FAMILIES, CONSTRUCTIONS, range(len(UNITS)), ("outer","attitude"), range(4)):
        bit=0 if op=="outer" else 1; others=[i for i in range(3) if i!=bit]; base=sum(((context>>j)&1)<<ob for j,ob in enumerate(others)); right=base|(1<<bit); lc=f"{base:03b}"[::-1]; rc=f"{right:03b}"[::-1]
        lid=f"c447-axis-{family}-{surface}-u{unit}-{lc}"; rid=f"c447-axis-{family}-{surface}-u{unit}-{rc}"
        if lid in keyed and rid in keyed: records.append({"group":f"axis:{family}:outer:{op}","family":family,"operation":op,"surface":surface,"unit":unit,"partition":part(unit),"context":context,"left":keyed[lid]["hidden_index"],"right":keyed[rid]["hidden_index"]})
    return records


def c450() -> None:
    out=begin("C450", {"status":"fixed_codebook_guarded_rule_replication_frozen","fit":"journal+bulletin discovery units0-3","confirmation":"record units4-5","lockbox":"units6-7 reserved for C452","controls":["identity","zero","mean","wrong group"],"pass":"affine NRMSE beats all controls and active sign accuracy beats controls"}, {"parent":final("C449")["all_checks_passed"]})
    if not final("C449")["headline"]["field_ran"]:
        close("C450", {"status":"replication_not_run_no_field","ran":False,"candidate_groups":[]}, {"route_accounted":True}, "C451_adjudication"); return
    states=np.load(OUTS["C449"]/"raw/role_states.float16.npy",mmap_mode="r"); index=read_rows(OUTS["C449"]/"raw/hidden_index.jsonl"); records=response_records(index); groups=sorted({r["group"] for r in records}); shape=(len(groups),37,6,DIM)
    slope=np.lib.format.open_memmap(out/"analysis/slope.float16.npy",mode="w+",dtype=np.float16,shape=shape); intercept=np.lib.format.open_memmap(out/"analysis/intercept.float16.npy",mode="w+",dtype=np.float16,shape=shape); mean_next=np.lib.format.open_memmap(out/"analysis/mean_next.float16.npy",mode="w+",dtype=np.float16,shape=shape)
    for gi,g in enumerate(groups):
        train=[r for r in records if r["group"]==g and r["partition"]=="discovery" and r["surface"]!="record"]
        for q in range(37):
            for role in range(6):
                x=np.stack([np.asarray(states[r["right"],q,role],np.float32)-np.asarray(states[r["left"],q,role],np.float32) for r in train]); y=np.stack([np.asarray(states[r["right"],q+1,role],np.float32)-np.asarray(states[r["left"],q+1,role],np.float32) for r in train]); xm=x.mean(0); ym=y.mean(0); xc=x-xm; a=np.sum(xc*(y-ym),0)/(np.sum(xc*xc,0)+1e-6); slope[gi,q,role]=np.clip(a,-64,64).astype(np.float16); intercept[gi,q,role]=(ym-a*xm).astype(np.float16); mean_next[gi,q,role]=ym.astype(np.float16)
        slope.flush();intercept.flush();mean_next.flush();print(f"[C450 fit] {gi+1}/{len(groups)} {g}",flush=True)
    results=[]; nodes=[]
    for gi,g in enumerate(groups):
        test=[r for r in records if r["group"]==g and r["partition"]=="confirmation" and r["surface"]=="record"]; wrong=(gi+1)%len(groups); totals={k:0. for k in ("affine","identity","zero","mean","wrong","truth")}; signs={k:[0,0] for k in ("affine","identity","zero","mean","wrong")}
        for q in range(37):
            for role in range(6):
                node={k:0. for k in totals};
                for r in test:
                    x=np.asarray(states[r["right"],q,role],np.float32)-np.asarray(states[r["left"],q,role],np.float32); y=np.asarray(states[r["right"],q+1,role],np.float32)-np.asarray(states[r["left"],q+1,role],np.float32); preds={"affine":np.asarray(slope[gi,q,role],np.float32)*x+np.asarray(intercept[gi,q,role],np.float32),"identity":x,"zero":np.zeros_like(x),"mean":np.asarray(mean_next[gi,q,role],np.float32),"wrong":np.asarray(slope[wrong,q,role],np.float32)*x+np.asarray(intercept[wrong,q,role],np.float32)}; node["truth"]+=float(np.sum(y*y)); totals["truth"]+=float(np.sum(y*y)); active=np.abs(y)>1e-3
                    for name,pred in preds.items(): node[name]+=float(np.sum((pred-y)**2)); totals[name]+=float(np.sum((pred-y)**2)); signs[name][0]+=int(np.sum((pred[active]>=0)==(y[active]>=0)));signs[name][1]+=int(np.sum(active))
                denom=math.sqrt(node["truth"])+1e-8; nodes.append({"group":g,"checkpoint":q,"role":ROLES[role],"affine_nrmse":math.sqrt(node["affine"])/denom,"best_control_nrmse":min(math.sqrt(node[k])/denom for k in ("identity","zero","mean","wrong")),"gain":min(math.sqrt(node[k])/denom for k in ("identity","zero","mean","wrong"))-math.sqrt(node["affine"])/denom})
        denom=math.sqrt(totals["truth"])+1e-8;n={k:math.sqrt(v)/denom for k,v in totals.items() if k!="truth"};s={k:a/max(b,1) for k,(a,b) in signs.items()};passed=bool(test) and n["affine"]<min(n[k] for k in ("identity","zero","mean","wrong")) and s["affine"]>max(s[k] for k in ("identity","zero","mean","wrong"));results.append({"group":g,"test_records":len(test),"nrmse":n,"sign_accuracy":s,"passed":passed})
    write_rows(out/"analysis/group_metrics.jsonl",results);write_rows(out/"analysis/node_metrics.jsonl",nodes);write_rows(out/"analysis/response_records.jsonl",records);save(out/"analysis/groups.json",groups);slope.flush();intercept.flush();mean_next.flush();close_mmap=lambda v:getattr(v,"_mmap",None) and v._mmap.close();close_mmap(slope);close_mmap(intercept);close_mmap(mean_next);close_mmap(states)
    candidates=[r["group"] for r in results if r["passed"]];headline={"status":"fixed_codebook_guarded_rule_replication_closed","ran":True,"groups":len(groups),"candidate_groups":candidates,"candidate_count":len(candidates),"mean_affine_nrmse":float(np.mean([r["nrmse"]["affine"] for r in results])),"strict_interpretation":"Passing identities are independently refit pattern replications, not coefficient replication or causal semantics."};close("C450",headline,{"accounting":len(results)==22,"finite":finite(headline)},"C451_adjudication")


def c451() -> None:
    out=begin("C451", {"status":"candidate_pattern_adjudication_and_writer_freeze","replication":"C439 candidate group identity on fixed-codebook fresh material","stable_gate":"at least 8/11 old candidates replicate and at most 3 old negatives turn positive","writer_selection":"best confirmation gain among replicated broad-query nodes, checkpoints8-30"}, {"parent":final("C450")["all_checks_passed"]})
    old=set(prior.final("C439")["headline"]["candidate_groups"]); new=set(final("C450")["headline"]["candidate_groups"]); replicated=sorted(old&new); false_new=sorted(new-old); stable=len(replicated)>=8 and len(false_new)<=3
    selection=None
    if stable:
        nodes=read_rows(OUTS["C450"]/"analysis/node_metrics.jsonl"); eligible=[r for r in nodes if r["group"] in replicated and r["group"].startswith("broad:") and r["group"].endswith(":query") and 8<=r["checkpoint"]<=30 and r["gain"]>0]; selection=max(eligible,key=lambda r:r["gain"]) if eligible else None
    save(out/"protocol/writer_selection.json",selection)
    headline={"status":"candidate_pattern_adjudication_closed","old_candidates":sorted(old),"new_candidates":sorted(new),"replicated":replicated,"replicated_count":len(replicated),"false_new":false_new,"pattern_stable":stable,"writer_authorized":selection is not None,"writer_selection":selection,"strict_interpretation":"Stability is identity-level recurrence after output-code repair; it is not semantic completeness."};close("C451",headline,{"accounting":len(old)==11,"finite":finite(headline)},"C452_writer")


@torch.inference_mode()
def score_with_patch(model, device, row:dict, layer_index:int|None, positions:list[int], delta:np.ndarray|None)->list[float]:
    hook=None
    if layer_index is not None and delta is not None:
        value=torch.tensor(delta,dtype=torch.float32,device=device)
        def patch(_module,_args,output):
            state=output[0] if isinstance(output,tuple) else output; changed=state.clone(); changed[0,positions]+=value.to(changed.dtype); return (changed,*output[1:]) if isinstance(output,tuple) else changed
        hook=model.model.layers[layer_index].register_forward_hook(patch)
    try:
        ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device); output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True); return [float(output.logits[0,-1,c[0]]) for c in row["candidate_ids"]]
    finally:
        if hook: hook.remove()


def c452() -> None:
    out=begin("C452", {"status":"conditional_hiddenstate_writer_frozen","qualification":"C451 stable pattern and selected broad-query node","test":"record construction lockbox units6-7, both statement contexts","conditions":["natural left","natural right","predicted","actual","wrong group","coordinate roll","wrong role","wrong checkpoint"],"pass":"predicted target-margin shift positive, beats all mismatch shifts, and target choice rate >=0.60","claim":"narrow role/checkpoint sufficiency only"}, {"parent":final("C451")["all_checks_passed"]})
    selection=final("C451")["headline"]["writer_selection"]
    if not final("C451")["headline"]["writer_authorized"]:
        close("C452", {"status":"writer_not_run_pattern_ineligible","writer_ran":False,"specificity_passed":False,"strict_interpretation":"No natural-model causal result."}, {"route_accounted":True}, "C453_synthesis");return
    groups=load(OUTS["C450"]/"analysis/groups.json");gi=groups.index(selection["group"]);wrong=(gi+1)%len(groups);q=selection["checkpoint"];role=ROLES.index(selection["role"]);slope=np.load(OUTS["C450"]/"analysis/slope.float16.npy",mmap_mode="r");intercept=np.load(OUTS["C450"]/"analysis/intercept.float16.npy",mmap_mode="r");states=np.load(OUTS["C449"]/"raw/role_states.float16.npy",mmap_mode="r");index=read_rows(OUTS["C449"]/"raw/hidden_index.jsonl");records=read_rows(OUTS["C450"]/"analysis/response_records.jsonl");compiled={r["case_id"]:r for r in read_rows(OUTS["C448"]/"compiled/qwen3.jsonl")};hidden={r["hidden_index"]:r for r in index};targets=[r for r in records if r["group"]==selection["group"] and r["partition"]=="lockbox" and r["surface"]=="record"]
    model=None;rows=[]
    try:
        model,_tok,device,placement=model_base.load_bf16("qwen3")
        for r in targets:
            left_case=hidden[r["left"]]["case_id"];right_case=hidden[r["right"]]["case_id"];left=compiled[left_case];right=compiled[right_case];x=np.asarray(states[r["right"],q,role],np.float32)-np.asarray(states[r["left"],q,role],np.float32);actual=np.asarray(states[r["right"],q+1,role],np.float32)-np.asarray(states[r["left"],q+1,role],np.float32);pred=np.asarray(slope[gi,q,role],np.float32)*x+np.asarray(intercept[gi,q,role],np.float32);wrong_pred=np.asarray(slope[wrong,q,role],np.float32)*x+np.asarray(intercept[wrong,q,role],np.float32);positions=left["role_positions"][selection["role"]];wrong_role=ROLES[(role+1)%len(ROLES)];conditions={"natural_left":(None,None,positions),"natural_right":(None,None,right["role_positions"][selection["role"]]),"predicted":(q,pred,positions),"actual":(q,actual,positions),"wrong_group":(q,wrong_pred,positions),"coordinate_roll":(q,np.roll(pred,257),positions),"wrong_role":(q,pred,left["role_positions"][wrong_role]),"wrong_checkpoint":(min(q+1,35),pred,positions)};scores={}
            for name,(layer,delta,pos) in conditions.items(): scores[name]=score_with_patch(model,device,right if name=="natural_right" else left,layer,pos,delta)
            base_margin=scores["natural_left"][1]-scores["natural_left"][0]; target={name:value[1]-value[0] for name,value in scores.items()};rows.append({"left_case":left_case,"right_case":right_case,"margins":target,"shifts":{name:margin-base_margin for name,margin in target.items() if name not in ("natural_left","natural_right")},"predicted_target_choice":int(np.argmax(scores["predicted"]))==right["gold_position"]})
    finally:model_base.release_bf16(model)
    write_rows(out/"raw/writer_trials.jsonl",rows);shifts={name:float(np.median([r["shifts"][name] for r in rows])) for name in ("predicted","actual","wrong_group","coordinate_roll","wrong_role","wrong_checkpoint")};rate=float(np.mean([r["predicted_target_choice"] for r in rows]));passed=shifts["predicted"]>0 and shifts["predicted"]>max(shifts[k] for k in ("wrong_group","coordinate_roll","wrong_role","wrong_checkpoint")) and rate>=.60
    headline={"status":"conditional_hiddenstate_writer_closed","writer_ran":True,"selection":selection,"trials":len(rows),"median_margin_shifts":shifts,"predicted_target_choice_rate":rate,"specificity_passed":passed,"strict_interpretation":"A pass is narrow conditional sufficiency at one role/checkpoint; it is not necessity, a unique circuit, or a general semantic writer."};close("C452",headline,{"trials":len(rows)>=4,"finite":finite(headline)},"C453_synthesis")


def hash_remove(paths:list[Path],out:Path)->list[dict]:
    rows=[]
    for path in paths:
        if not path.exists():continue
        h=hashlib.sha256();size=path.stat().st_size
        with path.open("rb") as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        path.unlink();rows.append({"path":str(path.relative_to(ROOT)),"bytes":size,"sha256":h.hexdigest(),"removed":not path.exists()})
    save(out/"audit/cleanup.json",rows);return rows


def c453() -> None:
    out=begin("C453", {"status":"fixed_codebook_replication_synthesis_frozen","visual":"all coordinates for selected fixed-code response means/slopes","cleanup":"checksum bulk fields after export","new_math_gate":"prospective plus causal plus cross-family composition required"}, {"parent":final("C452")["all_checks_passed"]})
    visual=[]
    if final("C450")["headline"].get("ran"):
        groups=load(OUTS["C450"]/"analysis/groups.json");mean=np.load(OUTS["C450"]/"analysis/mean_next.float16.npy",mmap_mode="r");slope=np.load(OUTS["C450"]/"analysis/slope.float16.npy",mmap_mode="r")
        focus=set(final("C451")["headline"]["replicated"] or final("C450")["headline"].get("candidate_groups",[]) or groups)
        for gi,g in enumerate(groups):
            if g not in focus:continue
            for q,role in itertools.product((0,8,16,24,32,36),range(6)):
                visual.append({"id":f"fixed:mean:{g}:q{q}:{ROLES[role]}","source":"fixed_codebook_mean_next","group":g,"checkpoint":q+1,"role":ROLES[role],"values":np.asarray(mean[gi,q,role],np.float32).round(6).tolist()});visual.append({"id":f"fixed:slope:{g}:q{q}:{ROLES[role]}","source":"fixed_codebook_slope","group":g,"checkpoint":q,"role":ROLES[role],"values":np.asarray(slope[gi,q,role],np.float32).round(6).tolist()})
        mean._mmap.close();slope._mmap.close()
    payload={"schema":"c453.fixed-codebook-replication.v1","phase":1987,"campaign":"C453","dimensions":list(range(DIM)),"rows":visual,"summary":{"behavior":final("C448")["headline"],"replication":final("C451")["headline"],"writer":final("C452")["headline"]},"claim_boundary":"Full activation coordinates are observations, not model parameters or semantic neurons."};save(VISUAL,payload)
    cleanup=hash_remove([OUTS["C449"]/"raw/role_states.float16.npy",OUTS["C449"]/"raw/full_fields_holdout.float16.npy",OUTS["C450"]/"analysis/slope.float16.npy",OUTS["C450"]/"analysis/intercept.float16.npy",OUTS["C450"]/"analysis/mean_next.float16.npy"],out)
    headline={"status":"fixed_codebook_replication_campaign_closed","pattern_stable":final("C451")["headline"]["pattern_stable"],"replicated_groups":final("C451")["headline"]["replicated"],"writer_ran":final("C452")["headline"]["writer_ran"],"causal_specificity":final("C452")["headline"]["specificity_passed"],"visual_rows":len(visual),"visual_path":str(VISUAL.relative_to(ROOT)),"cleanup_bytes":sum(r["bytes"] for r in cleanup),"new_math_gate_passed":False,"next_stage_same_goal":False,"strict_interpretation":"This campaign resolves the option-code confound and conditionally tests one writer; the next work changes from pattern replication to breadth/necessity."};close("C453",headline,{"visual":bool(visual) and all(len(r["values"])==DIM for r in visual),"cleanup":all(r["removed"] for r in cleanup),"finite":finite(headline)},"independent_audit_then_new_breadth_stage")


RUNNERS={name:globals()[name.lower()] for name in PHASES}
def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="C446-C453");parser.add_argument("--validate-only",action="store_true");args=parser.parse_args()
    if args.validate_only:
        rows=material();checks={"phase_sequence":[PHASES[f"C{i}"][0] for i in range(446,454)]==list(range(1980,1988)),"rows":len(rows)==1344,"balance":sum(r["gold_position"]==0 for r in rows)==len(rows)//2,"fixed_code":all("(A) Yes (B) No" in r["prompt"] for r in rows),"roles":all(set(r["role_values"])==set(ROLES)-{"boundary"} for r in rows)};print(json.dumps(checks));assert all(checks.values());return
    left,right=args.run.split("-",1) if "-" in args.run else (args.run,args.run)
    for i in range(int(left[1:]),int(right[1:])+1):RUNNERS[f"C{i}"]()
if __name__=="__main__":main()
