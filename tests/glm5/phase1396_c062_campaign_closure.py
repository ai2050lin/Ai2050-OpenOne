#!/usr/bin/env python3
"""Phase1396: audited closure of the C062 route-factorized full-field campaign."""
from __future__ import annotations

import json, py_compile, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core

PHASE,CAMPAIGN=1396,"C062"
OUT=TESTS/"result/phase1396_c062_campaign_closure"
PHASE_DIRS={
 1390:TESTS/"result/phase1390_c062_route_factorized_field_campaign_contract",
 1391:TESTS/"result/phase1391_c062_family_factorized_behavior",
 1392:TESTS/"result/phase1392_c062_full_field_camera",
 1393:TESTS/"result/phase1393_c062_discovery_full_field",
 1394:TESTS/"result/phase1394_c062_coordinate_curves",
 1395:TESTS/"result/phase1395_c062_event_bundle_mediation",
}


def main():
 if (OUT/"analysis/final.json").exists() and core.load(OUT/"analysis/final.json").get("all_checks_passed"):
  raise RuntimeError("Phase1396 already exists and passed")
 last=core.load(PHASE_DIRS[1395]/"analysis/final.json")
 if last["authorization"]!="run_phase1396_c062_campaign_closure":raise RuntimeError("closure not authorized")
 audits={p:core.load(d/"audit/independent_final_audit.json") for p,d in PHASE_DIRS.items()}
 behavior=core.load(PHASE_DIRS[1391]/"analysis/qwen3_family_behavior_summary.json")
 camera=core.load(PHASE_DIRS[1392]/"analysis/camera_summary.json")
 field=core.load(PHASE_DIRS[1393]/"analysis/full_field_summary.json")
 coord=core.load(PHASE_DIRS[1394]/"analysis/coordinate_summary.json")
 med=core.load(PHASE_DIRS[1395]/"analysis/event_mediation_summary.json")
 scripts=sorted(TESTS.glob("phase139[0-6]_c062_*.py"))
 for script in scripts:py_compile.compile(str(script),doraise=True)
 patterns=(r"\.self_attn",r"\.mlp",r"output_attentions\s*=\s*True",r"named_parameters\(",r"\.backward\(",r"torch\.pca")
 hits=[]
 runtime_scripts=[s for s in scripts if 1391<=int(s.name[5:9])<=1395 and not s.name.endswith("_audit.py")]
 for script in runtime_scripts:
  text=script.read_text(encoding="utf-8")
  for pat in patterns:
   if re.search(pat,text,re.I):hits.append({"script":script.name,"pattern":pat})
 checks={"all_phase_audits":all(v["all_checks_passed"] for v in audits.values()),
         "behavior_breadth":behavior["behavior_qualified"] and len(behavior["qualified_families"])==4,
         "camera":camera["camera_qualified"],"field_complete":all(field["checks"].values()),
         "coordinate_complete":all(coord["checks"].values()),"mediation_complete":all(med["checks"].values()),
         "c060_family_transfer_primary":coord["primary"]["c060_family_transfer_suff"],
         "c062_rediscovery_primary":coord["primary"]["c062_family_rediscovery_suff"],
         "matched_deletion_not_overclaimed":not coord["primary"]["c060_family_transfer_reverse"] and not coord["primary"]["c062_family_rediscovery_reverse"],
         "top1_boundary_not_query":med["top1_qualified"] and med["boundary_reference_qualified"] and not med["query_reference_qualified"],
         "forbidden_runtime_access_absent":not hits,"scripts_compile":True}
 result={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed_after_all_frozen_eligible_routes",
         "checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),
         "phase_audits":{str(p):{"passed":a["passed"],"total":a["total"],"all_checks_passed":a["all_checks_passed"]} for p,a in audits.items()},
         "formal_results":{"qualified_families":behavior["qualified_families"],
            "discovery_endpoint_gain_median":field["endpoint"]["correct_gain_median"],
            "c060_family_transfer_512_suff":coord["primary"]["c060_family_transfer_suff"],
            "c060_family_transfer_512_reverse":coord["primary"]["c060_family_transfer_reverse"],
            "c062_family_512_suff":coord["primary"]["c062_family_rediscovery_suff"],
            "c062_family_512_reverse":coord["primary"]["c062_family_rediscovery_reverse"],
            "first_tested_c060_family_k":coord["first_tested_qualified_size"]["c060_family_fixed"],
            "first_tested_c062_family_k":coord["first_tested_qualified_size"]["c062_family_discovery"],
            "top1_block_fraction":med["metrics"]["top1"]["splits"]["pooled"]["block_fraction_median"],
            "query_block_fraction":med["metrics"]["query_reference"]["splits"]["pooled"]["block_fraction_median"],
            "boundary_block_fraction":med["metrics"]["boundary_reference"]["splits"]["pooled"]["block_fraction_median"]},
         "claim_boundary":{"supported":[
            "Qwen-specific C060 family-coordinate sufficiency transfer to new animal/building vocabulary",
            "Qwen-specific C062 discovery-family coordinate sufficiency on both holdouts with 3/4 family breadth",
            "discovery full hidden response field with early family-position and late boundary-position candidates",
            "single late whole-state decision checkpoint can block or preserve artificial early rescue according to false/true donor state"],
            "not_supported":["minimal or necessary coordinate set","fixed relation vector or semantic neurons",
            "late checkpoint is a family-identity state rather than answer-polarity state","unique natural serial path",
            "attention, MLP, parameter, cross-model, or open-language mechanism"]},
         "forbidden_hits":hits,"automatic_next_phase":False,
         "next_required_action":"new C063 contract must separate family identity from answer polarity at the late checkpoint and test natural-state necessity; do not extend C062",
         "finished_at_utc":datetime.now(timezone.utc).isoformat()}
 core.save(OUT/"analysis/final.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
