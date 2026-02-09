import os

print("Script started, setting env vars...", flush=True)

# 修改模型下载目录到 D:\develop\model
os.environ["HF_HOME"] = r"D:\develop\model"

# 修复 HF_ENDPOINT 缺少 https:// 的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Importing numpy...", flush=True)
import numpy as np

print("Importing plotly...", flush=True)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Importing torch...", flush=True)
import torch

print("Importing transformer_lens...", flush=True)
import transformer_lens

print("Imports finished, starting to load model...", flush=True)

# 加载模型 - 使用 Qwen3-4B (约8GB)
model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
print("Model loaded!")

# 要分析的文本
prompt = "The quick brown fox jumps over the lazy dog"

# 运行模型并缓存所有中间激活值
logits, cache = model.run_with_cache(prompt)
tokens = model.to_str_tokens(prompt)
print(f"Prompt: {prompt}")
print(f"Tokens: {tokens}")

# ==================== 预测结构可视化 ====================
print("\n" + "="*60)
print("预测结构 (Prediction Structure)")
print("="*60)

# 获取 logits 的概率分布
probs = torch.softmax(logits[0], dim=-1)  # [seq_len, vocab_size]

# 对于每个位置，显示 top-5 预测
top_k = 5
print(f"\n每个 token 位置的 Top-{top_k} 预测:")
print("-" * 60)

prediction_data = []
for pos in range(len(tokens)):
    # 获取该位置的 top-k 预测
    top_probs, top_indices = torch.topk(probs[pos], top_k)
    top_tokens = [model.to_string(idx.item()) for idx in top_indices]
    
    current_token = tokens[pos]
    # 下一个真实 token（如果存在）
    next_token = tokens[pos + 1] if pos + 1 < len(tokens) else "[END]"
    
    print(f"\n位置 {pos}: '{current_token}' → 预测下一个 token:")
    print(f"  真实下一个: '{next_token}'")
    print(f"  Top-{top_k} 预测:")
    for i, (tok, prob) in enumerate(zip(top_tokens, top_probs)):
        marker = "✓" if tok.strip() == next_token.strip() else " "
        print(f"    {marker} {i+1}. '{tok}' ({prob.item()*100:.2f}%)")
    
    prediction_data.append({
        'position': pos,
        'current': current_token,
        'next_actual': next_token,
        'predictions': list(zip(top_tokens, top_probs.tolist()))
    })

# ==================== 可视化预测概率 ====================
# 创建热力图显示每个位置的 top-k 预测概率
fig2 = go.Figure()

# 准备数据
z_data = []
hover_text = []
for data in prediction_data:
    row = [p[1] for p in data['predictions']]
    z_data.append(row)
    hover_row = [f"位置: {data['position']}<br>当前: '{data['current']}'<br>预测: '{p[0]}'<br>概率: {p[1]*100:.2f}%" 
                 for p in data['predictions']]
    hover_text.append(hover_row)

fig2.add_trace(go.Heatmap(
    z=z_data,
    x=[f"Top-{i+1}" for i in range(top_k)],
    y=[f"{i}: {t}" for i, t in enumerate(tokens)],
    colorscale="Viridis",
    hoverinfo="text",
    text=hover_text,
    colorbar=dict(title="概率")
))

fig2.update_layout(
    title=dict(text="Qwen 模型预测结构 - 每个位置的 Top-5 预测概率", font=dict(size=18)),
    xaxis_title="预测排名",
    yaxis_title="Token 位置",
    width=700,
    height=600,
    yaxis=dict(autorange="reversed")  # 让第一个 token 在上面
)

fig2.write_html("prediction_visualization.html", include_plotlyjs=True, full_html=True)

# ==================== 文本生成示例 ====================
print("\n" + "="*60)
print("文本生成示例")
print("="*60)
generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
print(f"\n输入: {prompt}")
print(f"生成: {generated}")

# ==================== 每层神经网络状态 (Logit Lens) ====================
print("\n" + "="*60)
print("每层神经网络状态 (Logit Lens)")
print("="*60)

n_layers = model.cfg.n_layers
seq_len = len(tokens)
print(f"\n模型层数: {n_layers}")
print(f"序列长度: {seq_len}")

# 对每一层的残差流进行 unembed，得到每层的预测
# 这就是著名的 "Logit Lens" 技术
layer_predictions = []
layer_top_probs = []

for layer in range(n_layers):
    # 获取该层之后的残差流
    resid = cache[f"blocks.{layer}.hook_resid_post"]  # [batch, seq, d_model]
    
    # 应用 LayerNorm 和 Unembed 得到 logits
    scaled_resid = model.ln_final(resid)
    layer_logits = model.unembed(scaled_resid)  # [batch, seq, vocab]
    
    # 获取最后一个位置的 top-1 预测
    layer_probs = torch.softmax(layer_logits[0, -1], dim=-1)
    top_prob, top_idx = torch.max(layer_probs, dim=-1)
    top_token = model.to_string(top_idx.item())
    
    layer_predictions.append(top_token)
    layer_top_probs.append(top_prob.item())
    
    print(f"Layer {layer:2d}: 预测 '{top_token}' (概率: {top_prob.item()*100:.2f}%)")

# ==================== 每层每个位置的预测热力图 ====================
print("\n创建每层预测热力图...")

# 收集每层每个位置的 top-1 预测概率
layer_pos_probs = np.zeros((n_layers, seq_len))
layer_pos_tokens = [['' for _ in range(seq_len)] for _ in range(n_layers)]

for layer in range(n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    scaled_resid = model.ln_final(resid)
    layer_logits = model.unembed(scaled_resid)
    layer_probs = torch.softmax(layer_logits[0], dim=-1)
    
    for pos in range(seq_len):
        top_prob, top_idx = torch.max(layer_probs[pos], dim=-1)
        layer_pos_probs[layer, pos] = top_prob.item()
        layer_pos_tokens[layer][pos] = model.to_string(top_idx.item())

# 创建可视化
fig3 = go.Figure()

# 创建悬停文本
hover_texts = []
for layer in range(n_layers):
    row = []
    for pos in range(seq_len):
        text = f"Layer: {layer}<br>位置: {pos}<br>当前token: '{tokens[pos]}'<br>预测: '{layer_pos_tokens[layer][pos]}'<br>概率: {layer_pos_probs[layer, pos]*100:.2f}%"
        row.append(text)
    hover_texts.append(row)

fig3.add_trace(go.Heatmap(
    z=layer_pos_probs,
    x=[f"{i}: {t}" for i, t in enumerate(tokens)],
    y=[f"Layer {i}" for i in range(n_layers)],
    colorscale="Viridis",
    hoverinfo="text",
    text=hover_texts,
    colorbar=dict(title="Top-1 概率"),
    zmin=0,
    zmax=1
))

fig3.update_layout(
    title=dict(
        text="Qwen 模型每层预测状态 (Logit Lens)",
        font=dict(size=18)
    ),
    xaxis_title="Token 位置",
    yaxis_title="层",
    width=1000,
    height=700,
    yaxis=dict(autorange="reversed")  # 第0层在上面
)

fig3.write_html("layer_visualization.html", include_plotlyjs=True, full_html=True)

print("\n✅ 可视化文件已保存:")
print("   - prediction_visualization.html (预测结构)")
print("   - layer_visualization.html (每层神经网络状态)")
