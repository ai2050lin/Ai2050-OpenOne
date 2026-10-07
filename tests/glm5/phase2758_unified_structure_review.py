"""Bounded CPU audit and mathematical counterexamples; does not run an LLM."""
import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_unified_structure_review_20260923"
HIST = ROOT / "tests/glm5/result/rdc_query_construction_20260913"
GPT = ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md"
ATTACHMENTS = [
    Path(r"C:\Users\Admin\.codex\attachments\ce8dc325-de55-4f97-a8b6-4da1e8c6586b\已粘贴的文本.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\593c45ec-1886-41d6-a47c-6cdd63736d0c\已粘贴的文本.txt"),
]
SOURCES = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def track(p):
    SOURCES.append({"path": str(p), "bytes": p.stat().st_size,
                    "sha256": sha(p.read_bytes())})
    return p


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


def result(phase):
    paths = list((HIST / f"phase{phase}").glob("*/result.json"))
    assert len(paths) == 1, paths
    return json.loads(track(paths[0]).read_text(encoding="utf-8"))


def auc(y, score):
    pos = score[y == 1]
    neg = score[y == 0]
    delta = pos[:, None] - neg[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def main():
    assert not OUT.exists(), "Keep earlier deliveries immutable."
    OUT.mkdir(parents=True)
    for i, p in enumerate(ATTACHMENTS, 1):
        (OUT / f"attachment_{i}.txt").write_bytes(track(p).read_bytes())
    raw = track(GPT).read_bytes()
    (OUT / "gpt_memo_snapshot.md").write_bytes(raw)
    txt = raw.decode("utf-8")
    heads = list(re.finditer(r"^## Phase (\d+).*", txt, re.M))
    selected = {}
    excerpts = []
    for i, h in enumerate(heads):
        n = int(h.group(1))
        if 3103 <= n <= 3123 or n in (3040, 3072):
            selected[n] = {"line": txt.count("\n", 0, h.start()) + 1,
                           "heading": h.group(0)}
            excerpts.append(txt[h.start():heads[i+1].start() if i+1 < len(heads) else len(txt)])
    write("source_index.json", {"last_heading": heads[-1].group(0), "selected": selected,
                               "snapshot_sha256": sha(raw)})
    (OUT / "selected_history.md").write_text("\n".join(excerpts), encoding="utf-8")

    r10, r11, r12, r16, r18, r20, r23 = [result(n) for n in
                                         (3110, 3111, 3112, 3116, 3118, 3120, 3123)]
    k5 = r10["gates"]["ksweep_truth"]["K5"]
    cv = r16["coverage"]
    audit = {
        "3110": {"reported_metric": "AUC, not accuracy",
                 "K5_seeds": k5["aucs"], "K5_median_saved": k5["median"],
                 "K5_median_from_rounded_scores": float(np.median(k5["aucs"])),
                 "sampled_K": [int(k[1:]) for k in r10["gates"]["ksweep_truth"]],
                 "d_min_truth_saved": r10["d_min_truth"], "d_min_rel_saved": r10["d_min_rel"],
                 "interpretation": "Grid/threshold readout requirement; K<5 unsampled; random-subset refit is not arbitrary random-direction readout or intrinsic dimension."},
        "3116": {"saved": cv, "actual_ratio": cv["sum_single"]/cv["d_all"],
                 "recomputed_relative_discrepancy": abs(cv["sum_single"]-cv["d_all"])/abs(cv["d_all"]),
                 "scope_mismatch": "24 single-layer interventions (L12-35) versus all 36 MLPs jointly removed; unequal sets prevent attributing all discrepancy solely to interaction."},
        "3120": {"part_c_saved": r20.get("part_c"),
                 "interpretation": "Restricted conditional-mean regression, not exact linear full-state dynamics."},
        "3123": {"r_dir": r23["part_a"]["sim"]["r_dir"],
                 "r_anchor": r23["part_a"]["sim"]["r_anchor"],
                 "simulation_final_auc": r23["part_a"]["sim"]["auc_sim_dir"][-1]},
    }
    p18 = list((HIST / "phase3118").glob("*/traj_readout.npz"))
    assert len(p18) == 1
    with np.load(track(p18[0]), allow_pickle=False) as z:
        real = z["auc_curve"].astype(float)
    audit["3118_actual_saved_curve"] = real.tolist()
    audit["3123"]["reference_auc_range"] = [float(real.min()), float(real.max())]
    audit["3123"]["reference_final_auc"] = float(real[-1])
    audit["3123"]["all_reference_steps_ge_0_975"] = bool(np.all(real >= .975))

    # Execute only the isolated historical AUC definition, not its model-loading script.
    p_auc = ROOT / "tests/glm5/phase3111_omega_p109_broadcast_verdict.py"
    tree = ast.parse(track(p_auc).read_text(encoding="utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "auc_score")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = {"np": np}
    exec(compile(module, str(p_auc), "exec"), env)
    yy = np.r_[np.zeros(16, dtype=int), np.ones(16, dtype=int)]
    ss = np.zeros(32)
    audit["historical_auc_tie_counterexample"] = {
        "synthetic_only": True, "all_scores_constant": True,
        "historical_function_auc": env["auc_score"](yy, ss),
        "tie_aware_auc": auc(yy, ss),
        "scope": "Demonstrates a tie-handling implementation error, not the size of bias in actual captures; those metrics were not recomputed here.",
    }
    write("numerical_audit.json", audit)

    # Analytic construction h=y*a: rank-one signal with zero row mean and fixed norm.
    y = np.tile([-1., 1.], 32)
    a = np.repeat(np.arange(1., 1281.), 2) * np.tile([1., -1.], 1280)
    a /= np.linalg.norm(a)
    H = y[:, None] * a[None, :]
    assert np.max(np.abs(H.mean(1))) < 1e-14
    eig = np.linalg.eigvalsh(H @ H.T)
    rank = int(np.count_nonzero(eig > eig[-1] * 1e-10))
    rng = np.random.default_rng(2758)
    scores = []
    for _ in range(10):
        ids = rng.choice(2560, 5, replace=False)
        scores.append(auc((y > 0).astype(int), H[:, ids] @ a[ids]))
    synthetic = {
        "status": "Constructed mathematical counterexample; not LLM evidence",
        "definition": "h(y)=y*a, y in {-1,+1}; a proportional to (1,-1,2,-2,...,1280,-1280)",
        "samples": 64, "coordinates": 2560, "numerical_rank": rank,
        "row_mean_max_abs": float(np.max(np.abs(H.mean(1)))),
        "row_norm_range": [float(np.linalg.norm(H, axis=1).min()),
                           float(np.linalg.norm(H, axis=1).max())],
        "ten_random_5_coordinate_auc": scores,
        "single_coordinate_oriented_auc": auc((y > 0).astype(int), H[:, 0]),
        "orthogonal_readouts": {"directions": "e_1 and -e_2",
                                "dot_product": 0, "score_correlation": 1},
        "conclusion": "Distributed high-AUC coordinate readout and orthogonal predictive ports do not rule out a rank-one latent signal. Centering and norm control do not resolve this example.",
    }
    assert rank == 1 and min(scores) == 1
    write("rank_one_counterexample.json", synthetic)
    write("research_contract.json", {
        "status": "PROPOSED_NOT_EXECUTED", "no_gpu_run_in_this_phase": True,
        "question": "Can frozen rules map typed external transformations and their unseen compositions to internal response changes and output changes?",
        "milestones": [
            {"id": "A", "deliverable": "versioned claim ledger and reusable materials; correct metric definitions and invalidate downstream overclaims"},
            {"id": "B", "deliverable": "typed external relation/role/condition graph and full-coordinate internal response atlas with source provenance"},
            {"id": "C", "deliverable": "frozen simple/additive/bilinear/conditional-graph operator comparison on unseen entities, forms, graphs, compositions and depths"},
            {"id": "D", "deliverable": "targeted natural-source and matched intervention checks allowing redundancy and reconstruction; complete free generation"},
            {"id": "E", "deliverable": "conditional cross-model/scale study; evaluate geometry/category candidates only where added predictions justify them"},
        ],
        "prediction_contracts": {
            "descriptive": "Target-state probes and logit decompositions; do not count as forecasting.",
            "condition_transfer": "Base prompt state plus known transformation; changed-prompt target state is withheld.",
            "early_forecast": "Only declared early boundary state and externally available conditions; late target state and future tokens withheld.",
        },
        "controls": ["word identity", "typed relation and role", "token/position alignment", "negation scope",
                     "near-duplicate world families", "truth balance", "attention/cache/precision config",
                     "model competence vs intended answer"],
        "acceptance": "Use validation to fix realistic error budgets, baselines and complexity; require group-level held-out gains, calibrated uncertainty and recorded failure domains. No universal perfect accuracy requirement.",
        "resource_plan": "Cost pilot before freezing sample counts; sequential local CUDA 4B/14B/GLM4 if later authorized; chunk full-coordinate fields without blind full factorial expansion.",
    })
    write("evidence_manifest.json", {"phase": 2758, "created_local": datetime.now().astimezone().isoformat(),
                                    "sources": SOURCES, "model_runs": 0,
                                    "script_sha256": sha(Path(__file__).read_bytes())})
    print(json.dumps({"ratio": audit["3116"]["actual_ratio"],
                      "relative_discrepancy": cv["cov"],
                      "reference_auc": real.tolist(),
                      "tie_counterexample": audit["historical_auc_tie_counterexample"],
                      "rank_one": synthetic["numerical_rank"], "output": str(OUT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
