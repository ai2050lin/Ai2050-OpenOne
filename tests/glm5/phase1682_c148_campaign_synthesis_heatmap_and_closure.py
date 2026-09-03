#!/usr/bin/env python3
"""C148: C140-C147 synthesis, causal audit, and coordinate heatmap export."""
from __future__ import annotations
import json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1682_c148_campaign_synthesis_heatmap_and_closure";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1676_c142_mobius_output_code_separation as c142
PHASE,CAMPAIGN=1682,"C148";CHECKPOINTS=c127.CHECKPOINTS
C141=R/"phase1675_c141_multifamily_full_coordinate_atlas";C142=R/"phase1676_c142_mobius_output_code_separation";C143=R/"phase1677_c143_transition_model_competition";C144=R/"phase1678_c144_dual_graph_composition_reconstruction";C145=R/"phase1679_c145_correct_error_depth_trajectory_atlas";C146=R/"phase1680_c146_cross_model_interface_sweep";C147=R/"phase1681_c147_cross_model_relative_topology_eligibility"
def now():return datetime.now(timezone.utc).isoformat()
def c141_raw_rows():
 raw=np.load(C141/"raw/qwen3_six_role_field.bf16.npy",mmap_mode="r");rows=core.rows(C141/"compiled/qwen3.jsonl");out=[]
 for arm in c141.ARMS:
  i=next(j for j,r in enumerate(rows) if r["arm"]==arm and r["partition"]=="confirmation")
  for role in ("primary","boundary"):
   ri=c141.ROLES.index(role)
   for q in (0,16,27,36,37):out.append({"dataset":"C141","kind":"representative_raw_state","arm":arm,"case_id":rows[i]["case_id"],"partition":"confirmation","role":role,"checkpoint_index":q,"checkpoint":CHECKPOINTS[q],"token_positions":rows[i]["role_positions"][role],"values":c127.decode(raw[i,ri,q]).tolist()})
 return out
def c142_rows():
 freeze=core.load(C142/"protocol/frozen_nominees.json");report=core.load(C142/"analysis/confirmation.json");disc=np.load(C142/"analysis/discovery_mobius.float32.npy",mmap_mode="r");conf=np.load(C142/"analysis/confirmation_mobius.float32.npy",mmap_mode="r");code_d=np.load(C142/"analysis/discovery_code.float32.npy",mmap_mode="r");code_c=np.load(C142/"analysis/confirmation_code.float32.npy",mmap_mode="r");out=[]
 for ai,arm in enumerate(c141.ARMS):
  for si,name in enumerate(c142.SUBSET_NAMES):
   if not report["semantic_results"][arm][name]["passed"]:continue
   n=freeze["nominees"][arm][name]
   for partition,field in (("discovery",disc),("confirmation",conf)):out.append({"dataset":"C142","kind":"mobius_response","arm":arm,"effect":name,"partition":partition,"role":n["role"],"checkpoint_index":n["checkpoint_index"],"checkpoint":n["checkpoint"],"values":np.asarray(field[ai,:,si,n["role_index"],n["checkpoint_index"]].mean(0),np.float32).tolist()})
  n=freeze["nominees"][arm]["output_code"]
  for partition,field in (("discovery",code_d),("confirmation",code_c)):out.append({"dataset":"C142","kind":"output_code_response","arm":arm,"effect":"output_code","partition":partition,"role":n["role"],"checkpoint_index":n["checkpoint_index"],"checkpoint":n["checkpoint"],"values":np.asarray(field[ai,:,n["role_index"],n["checkpoint_index"]].mean(0),np.float32).tolist()})
 return out
def c145_rows():
 freeze=core.load(C145/"protocol/frozen_error_nominee.json");d=np.load(C145/"analysis/discovery_matched_error_residuals.float32.npy",mmap_mode="r");c=np.load(C145/"analysis/confirmation_exploratory_all_error_residuals.float32.npy",mmap_mode="r");ri,q=freeze["role_index"],freeze["checkpoint_index"]
 return [{"dataset":"C145","kind":"matched_error_residual","partition":"discovery","role":freeze["role"],"checkpoint_index":q,"checkpoint":freeze["checkpoint"],"values":np.asarray(d[:,ri,q].mean(0),np.float32).tolist()},{"dataset":"C145","kind":"exploratory_cross_partition_fallback_error_residual","partition":"confirmation_exploratory","role":freeze["role"],"checkpoint_index":q,"checkpoint":freeze["checkpoint"],"values":np.asarray(c[:,ri,q].mean(0),np.float32).tolist()}]
def main():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C147/"audit/independent_closure_audit.json")
 c143=core.load(C143/"analysis/confirmation.json");c144=core.load(C144/"analysis/confirmation_reconstruction.json");c145=core.load(C145/"analysis/confirmation.json");c146=core.load(C146/"analysis/confirmation.json");c147=core.load(C147/"analysis/eligibility_and_missingness.json")
 causal_authorized=bool(c143["prediction_gate_passed"])
 summary={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"C140_C148_campaign_closed","results":{"C140":"identifiability and typed status contract passed","C141":"1280-case five-arm all-token/all-coordinate Qwen atlas captured","C142":"9/35 output-code-averaged Mobius nominees replicated","C143":{"prediction_gate_passed":False,"winner":c143["frozen_winner"],"candidate_cosine":c143["derived"]["candidate_median_cosine"],"rollout_ratio":c143["derived"]["rollout_relative_error_ratio_vs_zero"]},"C144":{"aggregate_composition_gate_passed":c144["composition_gate_passed"],"frozen_order":c144["frozen_order"],"median_cosine":c144["frozen_order_median_cosine"],"per_arm_not_universal":True},"C145":{"formal_error_replication":False,"eligibility_passed":c145["eligibility_passed"],"exploratory_cosine":c145["nominee"]["cosine"],"missing_exact_support":c145["missing_exact_support_count"]},"C146":{"common_interface_gate_passed":c146["common_interface_gate_passed"],"winner":c146["winner"]},"C147":"cross-model internal topology not-tested"},"causal_intervention_run":False,"causal_status":"not-tested","causal_reason":"C143 frozen held-out transition predictor failed error, cosine, and rollout gates","theory_update":"The strongest repeated object is a typed, context-conditioned response field with transferable first-order components and task-specific interactions; no universal transition law, cross-model topology, or causal coordinate graph is established.","claim_boundary":"activation-coordinate observation and bounded prediction only; no Attention/MLP/weight inspection, unique circuit, semantic neuron, complete language mechanism, or new mathematics","next_stage":"continue observation with larger independent lexical units; prioritize arm-specific response laws and nonlinear cross-token transition candidates before causal patching"}
 OUT.mkdir(parents=True);(OUT/"analysis").mkdir();(OUT/"audit").mkdir();(OUT/"visualization").mkdir();core.save(OUT/"analysis/campaign_synthesis.json",summary)
 raw_rows=c141_raw_rows();response_rows=c142_rows();error_rows=c145_rows();payload=core.load(PUBLIC)
 payload["c140_c148_observation_batch"]={"campaign_summary":summary,"c141":{"behavior":core.load(C141/"analysis/authoritative_run.json")["behavior"],"representative_raw_rows":raw_rows},"c142":{"confirmation":core.load(C142/"analysis/confirmation.json"),"response_rows":response_rows},"c143":{"confirmation":c143},"c144":{"confirmation":c144},"c145":{"confirmation":c145,"error_rows":error_rows},"c146":{"selection":core.load(C146/"protocol/frozen_interface.json"),"confirmation":c146},"c147":{"eligibility":c147}}
 payload.update({"phase":PHASE,"campaign":"C109-C148","title":"Role-State Atlas + Multi-Family Response and Transition Audit","created_at_utc":now(),"claim_boundary":"C140-C148 add a 1280-case five-family atlas, output-code-separated Mobius effects, bounded transition competition, low-order centered reconstruction, typed error trajectories, and three-model interface results. Every displayed 2560-column row is an embedding or HiddenState activation coordinate, never a model weight. C143 prediction failed, C144 passed only an aggregate first-order gate, C145 formal error replication lacked support, and no common cross-model behavior interface existed; causal and cross-model internal claims remain not-tested."})
 canonical=OUT/"visualization/c109_c148_observation_atlas.json";core.save(canonical,payload);shutil.copyfile(canonical,PUBLIC)
 all_rows=raw_rows+response_rows+error_rows
 checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="start_C148","causal_closed":not causal_authorized and not summary["causal_intervention_run"],"raw_rows":len(raw_rows)==50,"response_rows":len(response_rows)==28,"error_rows":len(error_rows)==2,"full_coordinates":all(len(r["values"])==2560 for r in all_rows),"embedding":any(r["checkpoint_index"]==0 for r in raw_rows),"hidden":any(r["checkpoint_index"]==36 for r in raw_rows),"asset_match":core.sha(canonical)==core.sha(PUBLIC),"typed_claims":summary["causal_status"]=="not-tested" and c147["status"].startswith("cross_model_topology_not_tested")}
 audit={"phase":PHASE,"campaign":CAMPAIGN,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_causal_status":"not-tested","asset_sha256":core.sha(PUBLIC),"asset_bytes":PUBLIC.stat().st_size,"authorization":"independent_final_and_memo"};core.save(OUT/"audit/internal_closure_audit.json",audit);print(json.dumps({"summary":summary,"audit":audit},indent=2))
if __name__=="__main__":main()
