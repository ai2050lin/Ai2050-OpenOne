#!/usr/bin/env python3
"""Isolated GLM4 all-coordinate worker for Phase2100/C566."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2095_c561_c568_fresh_voice_response_replication as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output = args.output.resolve()
    model = None
    try:
        rows = [row for row in campaign.rows_read(campaign.material_path()) if row["surface"] == "record" and row["unit"] in (0,1,2,3,9,10,11)]
        model, tokenizer, device, placement = campaign.parent.previous.model_base().load_bf16("glm4")
        compile_base = campaign.parent.previous.prior.previous.parent.previous.prior.compile_base
        compiled = compile_base.compile_qwen(tokenizer, rows)
        base = model.model; checkpoint_count = len(base.layers) + 2; dim = int(model.get_input_embeddings().weight.shape[1]); width = max(len(row["prompt_ids"]) for row in compiled); n = len(rows)
        raw = args.output.parent.parent / "raw"; raw.mkdir(parents=True, exist_ok=True)
        last_path = raw / "glm4_role_last.npy"; full_path = raw / "glm4_full_token.npy"
        last = np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float16, shape=(n, checkpoint_count, len(campaign.ROLES), dim))
        full = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16, shape=(n, checkpoint_count, width, dim))
        captured = []
        def hook(_module, _args, output): captured.append(output[0] if isinstance(output, tuple) else output)
        hooks = [base.embed_tokens.register_forward_hook(hook)] + [layer.register_forward_hook(hook) for layer in base.layers] + [base.norm.register_forward_hook(hook)]
        index = []; correct = 0
        try:
            for i, row in enumerate(compiled):
                values = row["prompt_ids"]; ids = torch.tensor([values], dtype=torch.long, device=device); mask = torch.ones_like(ids); pos = torch.arange(len(values), device=device)[None]
                captured.clear()
                with torch.inference_mode(): output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != checkpoint_count: raise RuntimeError(("checkpoints", len(captured), checkpoint_count))
                positions = [int(row["role_positions"][role][-1]) for role in campaign.ROLES]
                for q, state in enumerate(captured):
                    state32 = state[0].float().cpu().numpy(); last[i,q] = state32[positions].astype(np.float16); full[i,q,:len(values)] = state32.astype(np.float16)
                scores = [float(output.logits[0,-1,candidate[0]]) for candidate in row["candidate_ids"]]; prediction = int(scores[1] > scores[0]); is_correct = prediction == row["gold_position"]; correct += int(is_correct)
                index.append({"hidden_index": i, "case_id": row["case_id"], "operation_domain": row["operation_domain"], "query_contract": row["query_contract"],
                    "unit": row["unit"], "variant": row["variant"], "pair_id": row["pair_id"], "partition": row["partition"], "length": len(values),
                    "role_positions": row["role_positions"], "gold_position": row["gold_position"], "prediction": prediction, "correct": is_correct})
                if i % 24 == 0 or i+1 == n: print(f"[C566 GLM4] {i+1}/{n}", flush=True)
            last.flush(); full.flush(); campaign.rows_write(raw / "glm4_index.jsonl", index)
        finally:
            for item in hooks: item.remove()
        pairs = defaultdict(dict)
        for row in index: pairs[row["pair_id"]][row["variant"]] = row["hidden_index"]
        metrics = {}; gates = {}; qpoints = (round((checkpoint_count-1)*0.65), checkpoint_count-1)
        for domain in campaign.DOMAINS:
            for contract in campaign.CONTRACTS:
                discovery = [(v[0],v[1]) for key,v in pairs.items() if set(v)=={0,1} and index[v[0]]["operation_domain"]==domain and index[v[0]]["query_contract"]==contract and index[v[0]]["partition"]=="discovery"]
                lockbox = [(v[0],v[1]) for key,v in pairs.items() if set(v)=={0,1} and index[v[0]]["operation_domain"]==domain and index[v[0]]["query_contract"]==contract and index[v[0]]["partition"]=="lockbox"]
                wrong_domain = next(value for value in campaign.DOMAINS if value != domain)
                wrong_pairs = [(v[0],v[1]) for key,v in pairs.items() if set(v)=={0,1} and index[v[0]]["operation_domain"]==wrong_domain and index[v[0]]["query_contract"]==contract and index[v[0]]["partition"]=="discovery"]
                for q in qpoints:
                    proto = np.mean([np.asarray(last[b,q],np.float32)-np.asarray(last[a,q],np.float32) for a,b in discovery],axis=0)
                    wrong = np.mean([np.asarray(last[b,q],np.float32)-np.asarray(last[a,q],np.float32) for a,b in wrong_pairs],axis=0); wrong=campaign.scaled_like(wrong,proto)
                    target = np.stack([np.asarray(last[b,q],np.float32)-np.asarray(last[a,q],np.float32) for a,b in lockbox]); pred=np.broadcast_to(proto,target.shape); bad=np.broadcast_to(wrong,target.shape); zero=np.zeros_like(target)
                    key=f"{domain}|{contract}|q{q}"; values={"pairs":len(lockbox),"correct":campaign.metric(pred,target),"zero":campaign.metric(zero,target),"wrong":campaign.metric(bad,target)}; metrics[key]=values
                    gates[key]=values["correct"]["nrmse"]<=values["zero"]["nrmse"]-campaign.CONTROL_MARGIN and values["correct"]["nrmse"]<=values["wrong"]["nrmse"]-campaign.CONTROL_MARGIN
        passed=int(sum(gates.values())); total=len(gates)
        del last, full
        save(args.output, {"status":"closed","model":"glm4","rows":n,"behavior_accuracy":correct/n,"checkpoint_count":checkpoint_count,"coordinate_count":dim,
            "full_shape":[n,checkpoint_count,width,dim],"last_shape":[n,checkpoint_count,len(campaign.ROLES),dim],"qpoints":list(qpoints),"metrics":metrics,"gates":gates,
            "gate_summary":{"passed":passed,"total":total,"pass_rate":passed/total},"functional_candidate":correct/n>=0.9 and passed/total>=0.75,
            "last_path":str(last_path.relative_to(ROOT)),"full_path":str(full_path.relative_to(ROOT)),"placement":placement})
    except Exception as exc:
        save(args.output,{"status":"worker_exception","exception_type":type(exc).__name__,"exception":str(exc),"traceback":traceback.format_exc()}); raise
    finally:
        campaign.parent.previous.model_base().release_bf16(model)


if __name__ == "__main__": main()
