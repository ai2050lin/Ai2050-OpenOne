"""Read-only source review and arithmetic checks; never loads an LLM.

Run from the repository root with its local Python. These checks validate saved
reports, not historical forward passes or the scientific hypotheses themselves.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2760_system_review_20261103"


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        "AGENTS.md", "README.md", "design/platform_spec_v1.md",
        "research/claude/docs/AGI_CLAUDE_MEMO.md",
        "research/gpt5/docs/AGI_GPT5_MEMO.md",
        "research/glm5/docs/AGI_GLM5_MEMO.md",
        "research/MainAnalysis/20261101_01_注意力机制提取了模式.md",
        "research/gpt5/docs/PAT_Q4R_FINDINGS_20261101.md",
        "research/gpt5/docs/PAT_Q9B_PREREG_v1.md",
        "research/gpt5/docs/PAT_Q9B_WSP_ADDENDUM_v1.md",
        "data/lang_bench/q9b_capgate_manifest.v1.json",
        "research/neurons/census-qwen3-4b-2ca5f0a756d36c8a-20M.json",
        "research/interventions/EXPL-v1-results.json",
        "research/interventions/EXPL-v2-results.json",
        "research/transcoder/TC-v1-qwen3-4b-L30/report.json",
        "research/transcoder/GQ-ATOM-v1/report.json",
        "research/calibration/C1-ioi-gpt2/report.json",
        "research/entity_atlas/EA-v2-qwen3-4b/report.json",
        "research/assembly/AM-v1-qwen3-4b/report.json",
    ]
    for tree in ["assembly", "fusion", "mode_field", "weight_space"]:
        sources += [p.relative_to(ROOT).as_posix()
                    for p in sorted((ROOT / "research" / tree).rglob("report.json"))]
    for name in ["grammar", "runner_v3", "analysis", "atlas", "neurons",
                 "census", "census_job", "goodfire_probe", "goodfire_runner",
                 "transcoder", "assembly", "weight_read", "mode_field", "fusion"]:
        sources.append(f"shared/dist_protocol/{name}.py")
    sources = list(dict.fromkeys(sources))
    inventory, headings = [], {}
    for rel in sources:
        p = ROOT / rel
        row = {"path": rel, "exists": p.is_file()}
        if p.is_file():
            row.update(bytes=p.stat().st_size, sha256=digest(p))
            if p.suffix == ".md":
                lines = p.read_text(encoding="utf-8-sig").splitlines()
                headings[rel] = [{"line": i, "title": line}
                                 for i, line in enumerate(lines, 1)
                                 if re.match(r"^#{1,3} ", line)]
            if p.suffix == ".json":
                d = read(rel)
                row["top_level_keys"] = list(d) if isinstance(d, dict) else None
                if isinstance(d, dict):
                    for key in ["model_snapshot", "precision", "prereg_sha256",
                                "verdict", "criteria"]:
                        if key in d:
                            row[key] = d[key]
        inventory.append(row)

    checks = []

    def check(name, got, expected, tolerance=1e-10):
        ok = (math.isclose(got, expected, rel_tol=tolerance, abs_tol=tolerance)
              if isinstance(got, (int, float)) and not isinstance(got, bool)
              else got == expected)
        checks.append({"name": name, "computed": got, "stored_or_expected": expected,
                       "pass": ok})

    am1 = read("research/assembly/AM-v1-qwen3-4b/report.json")
    for lang, row in am1["pooled"].items():
        check(f"AM1/{lang}/recovered", 1 - row["atlas"] / row["uniform"],
              row["recovered_atlas"])
    check("AM1/passing_cells", sum(am1["verdict"]["H1_cells"].values()), 12)
    am2_summary = {}
    for model in ["qwen3-4b", "gemma-3-4b-it"]:
        d = read(f"research/assembly/AM-v2-{model}/report.json")
        am2_summary[model] = {"verdict": d["verdict"], "by_language": {}}
        for lang, result in d["results"].items():
            r = result["P2"]
            check(f"AM2/{model}/{lang}/recovered", 1-r["kl"]["poe_slot"]/r["kl"]["base"],
                  r["recovered_vs_base"])
            check(f"AM2/{model}/{lang}/slot_gain", 1-r["kl"]["poe_slot"]/r["kl"]["poe_noslot"],
                  r["slot_gain"])
            am2_summary[model]["by_language"][lang] = {
                k: r[k] for k in ["recovered_vs_base", "recovered_vs_uniform", "slot_gain"]}
        check(f"AM2/{model}/H1", d["verdict"]["H1"], False)
        check(f"AM2/{model}/H2", d["verdict"]["H2"], False)

    tc = read("research/transcoder/TC-v1-qwen3-4b-L30/report.json")
    for name, row in tc["conditions"].items():
        check(f"TC/{name}/loss_recovery", (tc["ce_zero"]-row["ce"])/(tc["ce_zero"]-tc["ce_clean"]),
              row["loss_recovered_vs_zero"])
    feature = tc["conditions"]["tc-m16384-k64-s0"]
    neuron = tc["conditions"]["neuron_k64"]
    wins = (feature["fvu"] < neuron["fvu"] and
            feature["loss_recovered_vs_zero"] > neuron["loss_recovered_vs_zero"])
    check("TC/feature_wins_k64", wins, tc["atom_compare"]["tc-m16384-k64-s0"]["feature_wins"])

    expl = read("research/interventions/EXPL-v1-results.json")["summary"]
    check("EXPL1/win_share", expl["outcomes"]["win"]/expl["n_units"], expl["win_share"])
    gl = read("research/weight_space/GL-v1-glm4-9b-chat-hf/report.json")
    check("GL/edit_fraction_of_patch", gl["H4"]["edit_restore"]/gl["H1"]["value"], gl["H4"]["frac_of_patch"])
    check("GL/natural_direction_threshold", gl["H2"]["ratio"] >= gl["criteria"]["H2_read_ratio_min"], False)
    check("GL/key_threshold", gl["H7"]["ratio"] <= gl["criteria"]["H7_key_ratio_max"], False)
    for model in ["qwen3-4b", "gemma-3-4b-it"]:
        d = read(f"research/assembly/AM-v3-{model}/report.json")
        check(f"AM3/{model}/all_preregistered_fail", all(v is False for v in d["verdict"].values()), True)

    z = 1.959963984540054
    wilson_all_success_lower = 144/(144+z*z)
    notes = {
        "EXPL1": expl,
        "EXPL2": read("research/interventions/EXPL-v2-results.json")["summary"],
        "census": {k: v for k, v in read("research/neurons/census-qwen3-4b-2ca5f0a756d36c8a-20M.json").items() if k != "layers"},
        "AM1": am1["pooled"], "AM2": am2_summary,
        "TC_k64": {"feature": feature, "neuron": neuron, "feature_wins": wins},
        "GL": {k: gl[k] for k in ["precision", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "verdict"]},
        "WS6": {k: read("research/weight_space/WS-v6-qwen3-4b/report.json")[k] for k in ["H1", "H2", "H3", "H4", "verdict"]},
        "statistical_cautions": {
            "Q4R_144_of_144_naive_independent_Wilson_lower": wilson_all_success_lower,
            "Q4R_note": "24 triplets are clusters; the independent-binomial interval is illustrative, not a valid cluster-adjusted interval.",
            "20_permutations_minimum_p": 1/21,
            "noninferiority_planning_n_q_point1_margin_point02": math.ceil((1.6448536269514722+0.8416212335729143)**2*.1/.02**2),
            "noninferiority_note": "Approximation for planning a directional paired accuracy noninferiority test, alpha .05, power .8; q is total discordance. Not a power calculation for the historical 440 questions."
        },
        "q9b_named_results_found": [p.relative_to(ROOT).as_posix() for p in (ROOT/"results").rglob("*q9b*")],
    }
    git = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = {
        "schema": "research_source_review.v1", "phase": 2760,
        "created_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "git_commit": git.stdout.strip(), "scope": "source inspection and arithmetic recomputation only",
        "llm_forward_passes": 0, "gpu_experiments": 0,
        "not_done": ["historical activation re-execution", "complete raw tensor audit", "protocol test suite rerun", "live production center inspection"],
        "external_primary_sources": [
            "https://already-clean.goodfire.com/",
            "https://rikiyat.github.io/blog/weight-space-programming/",
            "https://github.com/RikiyaT/lm_understand-logs",
            "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/README.md",
            "https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        ], "sources": inventory,
    }
    for name, obj in [("source_manifest.json", manifest), ("section_index.json", headings),
                      ("saved_metrics.json", notes), ("arithmetic_checks.json", checks)]:
        (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    failed = [row["name"] for row in checks if not row["pass"]]
    print(json.dumps({"sources": len(inventory), "missing": [r["path"] for r in inventory if not r["exists"]],
                      "checks": len(checks), "failed": failed, "output": str(OUT)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
