"""
生成完整的36层前向传播演示数据
每层包含: attention norm, ffn gate/norm, residual norm, neuron_activations
神经元激活值根据层的功能区域变化:
  L0:   嵌入层 - 低激活为主, 少量语义信号
  L1-5: 词法加工 - 激活逐渐增强
  L6:   断裂层 - MLP norm暴增, 语法神经元>0.8
  L7-11: 语法加工 - 语法方向(w_u_perp)强激活
  L12-17: 语义+语法混合 - 双方向激活
  L18-23: 语义提取 - w_u方向增强
  L24-29: 逻辑注入 - logic方向增强
  L30-34: 输出决策 - w_u主导, 高激活
  L35:  输出映射 - 最终层, 极高激活
"""
import json
import math
import random

random.seed(42)

def gen_neurons(layer, n_neurons=7):
    """根据层号生成神经元, 激活值随层功能变化"""
    neurons = []
    
    # 确定各子空间的基础激活概率
    if layer <= 1:
        # 嵌入/词法: w_u略强, 其他弱
        base = {'w_u': 0.45, 'w_u_perp': 0.25, 'dark_matter': 0.20, 'logic': 0.10}
    elif layer <= 5:
        # 词法加工: 逐渐增强
        t = (layer - 2) / 4
        base = {'w_u': 0.50 + t*0.15, 'w_u_perp': 0.30 + t*0.10, 'dark_matter': 0.25 + t*0.10, 'logic': 0.12 + t*0.05}
    elif layer == 6:
        # 断裂层: w_u_perp暴增!
        base = {'w_u': 0.55, 'w_u_perp': 0.85, 'dark_matter': 0.50, 'logic': 0.25}
    elif layer <= 11:
        # 语法加工: w_u_perp主导
        t = (layer - 7) / 5
        base = {'w_u': 0.45 + t*0.10, 'w_u_perp': 0.75 - t*0.10, 'dark_matter': 0.40 + t*0.10, 'logic': 0.20 + t*0.05}
    elif layer <= 17:
        # 语义+语法混合
        t = (layer - 12) / 6
        base = {'w_u': 0.60 + t*0.15, 'w_u_perp': 0.60 - t*0.10, 'dark_matter': 0.35 + t*0.10, 'logic': 0.25 + t*0.05}
    elif layer <= 23:
        # 语义提取
        t = (layer - 18) / 6
        base = {'w_u': 0.75 + t*0.10, 'w_u_perp': 0.45 - t*0.05, 'dark_matter': 0.40 + t*0.05, 'logic': 0.30 + t*0.10}
    elif layer <= 29:
        # 逻辑注入
        t = (layer - 24) / 6
        base = {'w_u': 0.70 + t*0.15, 'w_u_perp': 0.35 - t*0.05, 'dark_matter': 0.35 - t*0.05, 'logic': 0.40 + t*0.10}
    else:
        # 输出决策/映射
        t = (layer - 30) / 5
        base = {'w_u': 0.85 + t*0.10, 'w_u_perp': 0.30 - t*0.05, 'dark_matter': 0.25 - t*0.05, 'logic': 0.35 - t*0.05}
    
    subspaces = ['w_u', 'w_u_perp', 'dark_matter', 'logic']
    # 分配神经元到各子空间
    counts = [3, 2, 1, 1]  # w_u多, 其他少
    idx = 0
    for si, ss in enumerate(subspaces):
        for _ in range(counts[si]):
            b = base[ss]
            # 激活值围绕基础值波动
            activation = max(0.02, min(0.99, b + random.gauss(0, 0.12)))
            # 位置: 在圆盘内随机分布
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(1.5, 6.0)
            x = r * math.cos(angle)
            z = r * math.sin(angle)
            neurons.append({
                'x': round(x, 1),
                'z': round(z, 1),
                'activation': round(activation, 2),
                'subspace': ss,
            })
            idx += 1
    
    return neurons

def get_layer_label(layer):
    if layer == 0: return "嵌入层"
    if layer <= 5: return "词法加工"
    if layer == 6: return "语法加工(断裂层)"
    if layer <= 11: return "语法加工"
    if layer <= 17: return "语义+语法混合"
    if layer <= 23: return "语义提取"
    if layer <= 29: return "逻辑注入"
    return "输出决策"

def get_attn_norm(layer):
    """Attention norm: 浅层高, 逐层递减"""
    base = 14.0 - layer * 0.1
    return round(base + random.gauss(0, 0.5), 1)

def get_ffn_gate(layer):
    """FFN gate: 断裂层高, 其他中等"""
    if layer == 6: return round(0.45 + random.gauss(0, 0.03), 2)
    if layer <= 5: return round(0.25 + layer * 0.02 + random.gauss(0, 0.03), 2)
    if layer <= 17: return round(0.30 + random.gauss(0, 0.03), 2)
    return round(0.25 - (layer - 18) * 0.005 + random.gauss(0, 0.03), 2)

def get_ffn_norm(layer):
    """FFN norm: 断裂层暴增, 深层增大"""
    if layer == 6: return round(45.0 + random.gauss(0, 2), 1)
    if layer <= 5: return round(8 + layer * 3 + random.gauss(0, 1), 1)
    if layer <= 17: return round(25 + (layer - 12) * 2.5 + random.gauss(0, 2), 1)
    return round(40 + (layer - 18) * 3 + random.gauss(0, 2), 1)

def get_residual_norm(layer):
    """Residual norm: 逐层增大, 断裂层跳跃"""
    if layer <= 5: return round(15 + layer * 4 + random.gauss(0, 1), 1)
    if layer == 6: return round(85 + random.gauss(0, 3), 1)  # 断裂层!
    if layer <= 17: return round(50 + (layer - 7) * 3 + random.gauss(0, 2), 1)
    return round(60 + (layer - 18) * 4 + random.gauss(0, 2), 1)

# 生成36层数据
layers_data = []
for l in range(36):
    layer_entry = {
        "layer": l,
        "label": get_layer_label(l),
        "attention": {
            "norm": get_attn_norm(l),
        },
        "ffn": {
            "gate_activation": get_ffn_gate(l),
            "norm": get_ffn_norm(l),
        },
        "residual_norm": get_residual_norm(l),
        "neuron_activations": gen_neurons(l, 7),
    }
    # L0额外添加attention pattern
    if l == 0:
        layer_entry["attention"]["pattern"] = [
            [0.33, 0.20, 0.15, 0.12, 0.10, 0.10],
            [0.25, 0.30, 0.18, 0.10, 0.09, 0.08],
            [0.18, 0.22, 0.28, 0.12, 0.11, 0.09],
            [0.15, 0.18, 0.20, 0.25, 0.12, 0.10],
            [0.12, 0.15, 0.18, 0.20, 0.22, 0.13],
            [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        ]
    # 为FFN添加top_neurons
    neurons = layer_entry["neuron_activations"]
    top = sorted(neurons, key=lambda n: n['activation'], reverse=True)[:5]
    layer_entry["ffn"]["top_neurons"] = [
        {"id": 1000 + i * 100 + l * 10, "activation": n["activation"], "subspace": n["subspace"]}
        for i, n in enumerate(top)
    ]
    layers_data.append(layer_entry)

output = {
    "schema_version": "2.0",
    "model": "qwen3-4b",
    "model_info": {
        "n_layers": 36,
        "d_model": 2560,
        "n_heads": 20,
        "vocab_size": 151936,
    },
    "sentence": "The cat sat on the mat",
    "tokens": ["The", " cat", " sat", " on", " the", " mat"],
    "token_ids": [791, 3795, 7672, 389, 279, 3682],
    "layers": layers_data,
    "visualizations": [],
    "forward_pass_config": {
        "models": [
            {"id": "qwen3-4b", "label": "Qwen3-4B", "n_layers": 36, "d_model": 2560},
            {"id": "glm4-9b", "label": "GLM4-9B-Chat", "n_layers": 40, "d_model": 4096},
            {"id": "ds7b", "label": "DeepSeek-R1-7B", "n_layers": 28, "d_model": 3584},
        ],
        "sentences": [
            {"id": "s1", "text": "The cat sat on the mat", "tokens": ["The", " cat", " sat", " on", " the", " mat"]},
            {"id": "s2", "text": "She didn't like the movie", "tokens": ["She", " didn", "'t", " like", " the", " movie"]},
            {"id": "s3", "text": "The students have been studying", "tokens": ["The", " students", " have", " been", " studying"]},
        ],
    },
}

with open("d:/ai2050/TransformerLens-Project/frontend/public/data/forward_pass_demo.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Generated {len(layers_data)} layers with neuron data")
