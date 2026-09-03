#!/usr/bin/env python3
"""Phase1491: capture embeddings and all role-aligned Hidden States for every C085 case."""
from __future__ import annotations
import inspect,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; C=TESTS/"result/phase1489_c085_prospective_layered_contract"; B=TESTS/"result/phase1490_c085_behavior_stratification"; OUT=TESTS/"result/phase1491_c085_all_case_field_capture"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
from phase1392_c062_full_field_camera import make_batch
BATCH=12

@torch.inference_mode()
def capture(cases,protocol,case_strata):
    raw=OUT/"raw/all_role_field.float16.npy"; raw.parent.mkdir(parents=True,exist_ok=True); model=None
    try:
        model,tok,device,placement=load_bf16("qwen3"); quant=quantization_audit(model); dim=int(model.config.hidden_size); shape=(len(cases),37,9,dim); field=np.lib.format.open_memmap(raw,mode="w+",dtype=np.float16,shape=shape); pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id); roles=protocol["roles"]; supports="logits_to_keep" in inspect.signature(model.forward).parameters; index=[]; finite=True
        for start in range(0,len(cases),BATCH):
            batch=cases[start:start+BATCH]; ids,mask,pos,offsets=make_batch(batch,pad,device); kwargs={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"output_hidden_states":True,"return_dict":True}
            if supports: kwargs["logits_to_keep"]=1
            out=model(**kwargs)
            if len(out.hidden_states)!=37: raise RuntimeError(("state_count",len(out.hidden_states)))
            role_index=torch.tensor([[offsets[i]+batch[i]["role_positions"][role][0] for role in roles] for i in range(len(batch))],dtype=torch.long,device=device); bi=torch.arange(len(batch),device=device)[:,None]; block=np.empty((len(batch),37,9,dim),dtype=np.float16)
            for state,h in enumerate(out.hidden_states):
                gathered=h[bi,role_index]; finite=finite and bool(torch.isfinite(gathered).all()); block[:,state]=gathered.to(dtype=torch.float16,device="cpu").numpy()
            field[start:start+len(batch)]=block; logits=out.logits[:,-1].float()
            for i,row in enumerate(batch):
                scores=[float(logits[i,values[0]]) for values in row["candidate_ids"]]
                index.append({"row_index":start+i,"case_id":row["case_id"],"set_id":case_strata[row["case_id"]][0],"stratum":case_strata[row["case_id"]][1],"partition":row["partition"],"family":row["family"],"index":row["index"],"surface":row["surface"],"cell":row["cell"],"record_relation_id":row["record_relation_id"],"query_relation_id":row["query_relation_id"],"entity_match":row["entity_match"],"object_match":row["object_match"],"relation_match":row["relation_match"],"truth":row["truth"],"gold_position":row["gold_position"],"capture_scores":scores,"capture_prediction":int(np.argmax(scores)),"role_positions":row["role_positions"]})
            del out,ids,mask,pos,role_index,bi,block
        field.flush(); del field
        return index,{"placement":placement,"quantization":quant,"shape":list(shape),"hidden_dim":dim,"finite_during_capture":finite}
    finally:
        if model is not None: release_bf16(model)

def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1491 exists")
    bf=core.load(B/"analysis/final.json"); ba=core.load(B/"audit/independent_final_audit.json"); protocol=core.load(C/"protocol/preregistration.json")
    if bf["authorization"]!="run_phase1491_c085_all_case_field_capture" or not ba["all_checks_passed"]: raise RuntimeError("Phase1490 authorization missing")
    compiled=core.rows(C/"compiled/qwen3_active.jsonl"); sets=core.rows(B/"material/stratified_composition_sets.jsonl"); keys=tuple(f"{s}_{c}" for s in protocol["surfaces"] for c in protocol["cells"]); case_strata={group[k]:(group["set_id"],group["stratum"]) for group in sets for k in keys}; index,runtime=capture(compiled,protocol,case_strata)
    raw=OUT/"raw/all_role_field.float16.npy"; core.write_rows(OUT/"raw/all_role_field_index.jsonl",index); behavior={r["case_id"]:r for r in core.rows(B/"raw/behavior.jsonl")}; maxdiff=max(abs(v-behavior[r["case_id"]]["scores"][i]) for r in index for i,v in enumerate(r["capture_scores"])); arr=np.load(raw,mmap_mode="r")
    # Hidden-state capture uses output_hidden_states plus logits_to_keep, whereas
    # Phase1490 used the behavior-only forward.  BF16 logits need not be bitwise
    # identical across those kernels; the frozen integrity object is the complete
    # candidate ordering, while the observed score delta remains fully reported.
    checks={"count":len(index)==3456,"shape":list(arr.shape)==[3456,37,9,2560],"dtype":arr.dtype==np.float16,"all_strata":set(r["stratum"] for r in index)==set(bf["stratum_counts"]),"behavior_prediction_identity":all(r["capture_prediction"]==behavior[r["case_id"]]["prediction"] for r in index),"finite":runtime["finite_during_capture"] and all(math.isfinite(v) for r in index for v in r["capture_scores"]),"bf16":runtime["quantization"]["has_bf16_parameters"],"not_quantized":not runtime["quantization"]["has_quantized_modules"]}
    if not all(checks.values()): raise RuntimeError(checks)
    meta={"phase":1491,"campaign":"C085","shape":runtime["shape"],"dtype":"float16","axis_order":["case","state","role","coordinate"],"roles":protocol["roles"],"states":"state0 embedding output plus state1-state36 block outputs","file_size_bytes":raw.stat().st_size,"raw_sha256":core.sha(raw),"index_sha256":core.sha(OUT/"raw/all_role_field_index.jsonl"),"behavior_score_max_abs_diff":maxdiff,"checks":checks,"runtime":runtime,"finished_at_utc":datetime.now(timezone.utc).isoformat()}
    core.save(OUT/"analysis/capture_metadata.json",meta); core.save(OUT/"analysis/final.json",{"phase":1491,"campaign":"C085","status":"all_case_field_capture_complete","raw_sha256":meta["raw_sha256"],"authorization":"run_phase1492_c085_stratified_factorial_atlas"}); print(json.dumps({k:v for k,v in meta.items() if k!="runtime"},indent=2))
if __name__=="__main__": main()
