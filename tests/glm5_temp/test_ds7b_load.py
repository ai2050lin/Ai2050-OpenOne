"""Quick test: Load DeepSeek7B and extract a few residuals."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully!")

# Quick test
text = "将以下中文翻译成英文：猫"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[13][0, -1, :].cpu().float().numpy()
print(f"Hidden state at L12: shape={h.shape}, norm={sum(x**2 for x in h)**0.5:.4f}")
print("DeepSeek7B works!")
