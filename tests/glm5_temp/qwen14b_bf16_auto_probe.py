#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.core_model_loading as core_loading

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/hf/Qwen3-14B"
ROW = ROOT / "tests/glm5/result/phase2405_c24241_c24560_deconfounded_operation_contract/material/selection_rows.jsonl"
OFFLOAD = ROOT / "tests/glm5_temp/qwen14b_bf16_offload"
OFFLOAD.mkdir(parents=True, exist_ok=True)
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
torch.set_num_threads(1)
core_loading.GLOBAL_WORKERS = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True, use_fast=False)
print("[bf16-probe] loading", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", max_memory={0: "13GiB", "cpu": "14GiB"},
    offload_folder=OFFLOAD, offload_state_dict=True, offload_buffers=True, low_cpu_mem_usage=True,
    trust_remote_code=True, local_files_only=True, attn_implementation="eager",
)
model.eval()
row = json.loads(ROW.read_text(encoding="utf-8").splitlines()[0])
messages = [{"role": "user", "content": row["prompt"]}]
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
device = model.get_input_embeddings().weight.device
input_ids = torch.tensor([ids], dtype=torch.long, device=device)
with torch.inference_mode():
    output = model(input_ids=input_ids, use_cache=False, return_dict=True)
print(json.dumps({"class": model.__class__.__name__, "dtype": str(next(model.parameters()).dtype),
                  "device_map": model.hf_device_map, "logits_shape": list(output.logits.shape),
                  "finite": bool(torch.isfinite(output.logits).all().item()),
                  "cuda_gib": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0}, default=str), flush=True)
