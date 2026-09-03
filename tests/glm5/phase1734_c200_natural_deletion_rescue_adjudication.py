#!/usr/bin/env python3
"""C200: adjudicate eligibility for natural deletion/rescue without post-hoc coordinate selection."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; OUT=RESULT/"phase1734_c200_natural_deletion_rescue_adjudication"; C194=RESULT/"phase1728_c194_signed_operator_campaign_contract"; C197=RESULT/"phase1731_c197_structure_model_tournament"; C198=RESULT/"phase1732_c198_broad_natural_program_trajectory"; C199=RESULT/"phase1733_c199_unseen_composition_prediction"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
PHASE,CAMPAIGN=1734,"C200"


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(C199/"audit/independent_final_audit.json"); tournament=core.load(C197/"analysis/tournament.json"); composition=core.load(C199/"analysis/composition_prediction.json"); behavior=core.load(C198/"protocol/behavior_lock.json")
    eligible=tournament["primary_gate_passed"] and composition["primary_gate_passed"] and len(behavior["eligible_programs"])>=8
    checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="C200_natural_deletion_and_typed_rescue","behavior_qualified":len(behavior["eligible_programs"])==9,"candidate_not_qualified":not tournament["primary_gate_passed"] and not composition["primary_gate_passed"],"causal_eligible_false":eligible is False}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"natural_deletion_rescue_eligibility_frozen","prerequisite":"both C197 checkpoint-transform prediction and C199 unseen-composition prediction must pass before deriving a natural deletion target","behavior_qualified_programs":behavior["eligible_programs"],"candidate_transform_qualified":tournament["primary_gate_passed"],"unseen_composition_qualified":composition["primary_gate_passed"],"scientific_causal_eligible":eligible,"registered_controls":["delete predicted path contribution","correct model-predicted rescue","wrong-family rescue","wrong-role rescue","wrong-checkpoint rescue"],"decision":"typed_not_tested because there is no prospectively qualified path contribution to delete; running a post-hoc Top-K intervention would test a newly invented object","claim_boundary":"This is a prerequisite adjudication, not evidence that natural necessity or rescue fails.","producer_sha256":core.sha(Path(__file__)),"authorization":"C201_cross_model_model_specific_interfaces_and_relative_topology"}; core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks,"scientific_causal_eligible":eligible},indent=2))


def close():
    p=core.load(OUT/"protocol/preregistration.json"); checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"typed_not_tested":p["scientific_causal_eligible"] is False,"hash":core.sha(Path(__file__))==p["producer_sha256"]}; final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed_typed_not_tested","checks":checks,"all_checks_passed":all(checks.values()),"headline":{"behavior_qualified":True,"natural_deletion_rescue_tested":False,"reason":p["decision"],"inference":"No claim about natural necessity, causal insufficiency, or rescue failure is licensed."},"next_authorization":p["authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    a=argparse.ArgumentParser(); a.add_argument("command",choices=("contract","close")); x=a.parse_args(); contract() if x.command=="contract" else close()
if __name__=="__main__":main()
