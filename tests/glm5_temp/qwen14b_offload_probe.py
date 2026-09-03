#!/usr/bin/env python3
"""Temporary probe for a Windows torch_cpu.dll-safe Qwen14B load path."""
import os
from pathlib import Path

os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/hf/Qwen3-14B"
OFFLOAD = ROOT / "tests/glm5_temp/qwen14b_offload"
OFFLOAD.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True, use_fast=False)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=quant, device_map="auto", max_memory={0: "9GiB", "cpu": "22GiB"},
    offload_folder=OFFLOAD, offload_state_dict=True, trust_remote_code=True, local_files_only=True,
    low_cpu_mem_usage=True, attn_implementation="eager",
)
model.eval()
ids = tokenizer.encode("Hello", return_tensors="pt").to(model.get_input_embeddings().weight.device)
with torch.inference_mode(): output = model(input_ids=ids, use_cache=False, return_dict=True)
print({"loaded": True, "logits": list(output.logits.shape), "device_map": model.hf_device_map})
