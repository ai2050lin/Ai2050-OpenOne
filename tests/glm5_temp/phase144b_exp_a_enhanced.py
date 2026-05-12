"""
Phase 144b: Exp A 大数据量增强版 — 更多的跨领域句子对
====================================================
目的: Phase 144的Exp A每个类别只有1-2对句子, 数据量太少。
增加每个类别到8-10对, 确保结论可靠。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from model_utils import (load_model, get_layers, get_model_info, release_model, get_sample_layers)

TEMP_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
TEMP_DIR.mkdir(exist_ok=True)

# 大幅增加句子对数量
CROSS_DOMAIN_PAIRS = [
    # NOT (否定): 10对
    ("The cat sat on the mat", "The cat did not sit on the mat", "NOT"),
    ("She is happy", "She is not happy", "NOT"),
    ("He can swim", "He cannot swim", "NOT"),
    ("The door is open", "The door is not open", "NOT"),
    ("They will come", "They will not come", "NOT"),
    ("I like coffee", "I do not like coffee", "NOT"),
    ("The sky is blue", "The sky is not blue", "NOT"),
    ("She was running", "She was not running", "NOT"),
    ("We have finished", "We have not finished", "NOT"),
    ("The bird can sing", "The bird cannot sing", "NOT"),
    
    # TENSE (时态): 10对
    ("I went to the store", "I will go to the store", "TENSE"),
    ("He was running", "He is running", "TENSE"),
    ("She has finished", "She had finished", "TENSE"),
    ("They played soccer", "They are playing soccer", "TENSE"),
    ("We ate dinner", "We will eat dinner", "TENSE"),
    ("The dog barked", "The dog barks", "TENSE"),
    ("I wrote a letter", "I write a letter", "TENSE"),
    ("She sang beautifully", "She sings beautifully", "TENSE"),
    ("He drove the car", "He drives the car", "TENSE"),
    ("The children laughed", "The children laugh", "TENSE"),
    
    # SYN (同范畴替换): 10对
    ("The cat sat on the mat", "The dog sat on the mat", "SYN"),
    ("She loves music", "He loves music", "SYN"),
    ("The apple is red", "The sky is blue", "SYN"),
    ("The man walked home", "The woman walked home", "SYN"),
    ("A bird flew over", "A plane flew over", "SYN"),
    ("The book is heavy", "The stone is heavy", "SYN"),
    ("She cooked dinner", "He cooked dinner", "SYN"),
    ("The river flows east", "The wind blows east", "SYN"),
    ("I read the letter", "I wrote the letter", "SYN"),
    ("The sun rises early", "The moon rises early", "SYN"),
    
    # SCOPE (辖域): 10对
    ("All birds can fly", "Not all birds can fly", "SCOPE"),
    ("Every student passed", "Not every student passed", "SCOPE"),
    ("Some people agree", "Not some people agree", "SCOPE"),
    ("Both options work", "Not both options work", "SCOPE"),
    ("All cats like fish", "Not all cats like fish", "SCOPE"),
    ("Every door is locked", "Not every door is locked", "SCOPE"),
    ("All children play", "Not all children play", "SCOPE"),
    ("Each student passed", "Not each student passed", "SCOPE"),
    ("Both methods work", "Not both methods work", "SCOPE"),
    ("Everyone was invited", "Not everyone was invited", "SCOPE"),
    
    # CROSS (跨领域): 10对
    ("The cat sat on the mat", "Quantum physics describes particles", "CROSS"),
    ("I love classical music", "The stock market crashed today", "CROSS"),
    ("She baked a chocolate cake", "The algorithm sorts the array", "CROSS"),
    ("The river flows through the valley", "The function returns an integer", "CROSS"),
    ("The children played in the park", "The compiler optimizes the code", "CROSS"),
    ("He painted a beautiful landscape", "The database stores the records", "CROSS"),
    ("The sun set over the ocean", "The server processes the request", "CROSS"),
    ("She danced at the concert", "The model trains on the dataset", "CROSS"),
    ("The flowers bloomed in spring", "The program executes the command", "CROSS"),
    ("We climbed the mountain", "The network routes the packet", "CROSS"),
    
    # NONSENSE (无意义): 8对
    ("The cat sat on the mat", "Colorless green ideas sleep furiously", "NONSENSE"),
    ("She walked to the store", "The procedural abstraction crystallizes", "NONSENSE"),
    ("The sun is shining", "Furiously sleep ideas green colorless", "NONSENSE"),
    ("I ate breakfast today", "The ontological vacuum oscillates backward", "NONSENSE"),
    ("The dog barked loudly", "Transparent concepts breathe silently", "NONSENSE"),
    ("We went to school", "The epistemological granite dreams softly", "NONSENSE"),
    ("She read the book", "Abstract paradoxes vibrate gracefully", "NONSENSE"),
    ("He played the guitar", "Metaphorical wavelengths dissolve utterly", "NONSENSE"),
]

def collect_layer_outputs_at(model, inputs_embeds, position_ids, toks, sample_layers):
    layers = get_layers(model)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    hooks = []
    for li in sample_layers:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    with torch.no_grad():
        try:
            attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
            if attention_mask is not None:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask)
            else:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    return captured


def run_exp_a_enhanced(model_name):
    print(f"\nPhase 144b: Exp A 大数据量增强 — {model_name}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    sample_layers = get_sample_layers(n_layers, 6)
    eps = 2.0
    
    # 只测Exp A, 但每对用5个随机方向
    results_by_category = defaultdict(list)
    
    for pair_idx, (s1, s2, category) in enumerate(CROSS_DOMAIN_PAIRS):
        if pair_idx % 10 == 0:
            print(f"  Processing pair {pair_idx}/{len(CROSS_DOMAIN_PAIRS)}...")
        
        with torch.no_grad():
            toks1 = tokenizer(s1, return_tensors="pt").to(device)
            toks2 = tokenizer(s2, return_tensors="pt").to(device)
            
            embed_layer = model.get_input_embeddings()
            emb1_base = embed_layer(toks1.input_ids).detach().clone()
            emb2_base = embed_layer(toks2.input_ids).detach().clone()
        
        n_directions = 3
        direction_cosines = {li: [] for li in sample_layers}
        
        for di in range(n_directions):
            v = torch.randn(d_model, device=device, dtype=emb1_base.dtype)
            v = v / v.norm()
            
            with torch.no_grad():
                emb1_pert = emb1_base.clone()
                emb1_pert[0, -1, :] += (eps * v).to(emb1_base.dtype)
                
                emb2_pert = emb2_base.clone()
                emb2_pert[0, -1, :] += (eps * v).to(emb2_base.dtype)
                
                pos1 = torch.arange(emb1_base.shape[1], device=device).unsqueeze(0)
                pos2 = torch.arange(emb2_base.shape[1], device=device).unsqueeze(0)
                
                out1_base = collect_layer_outputs_at(model, emb1_base, pos1, toks1, sample_layers)
                out1_pert = collect_layer_outputs_at(model, emb1_pert, pos1, toks1, sample_layers)
                out2_base = collect_layer_outputs_at(model, emb2_base, pos2, toks2, sample_layers)
                out2_pert = collect_layer_outputs_at(model, emb2_pert, pos2, toks2, sample_layers)
            
            for li in sample_layers:
                key = f"L{li}"
                if key not in out1_base or key not in out1_pert:
                    continue
                if key not in out2_base or key not in out2_pert:
                    continue
                
                delta1 = (out1_pert[key] - out1_base[key])[0, -1, :].float().numpy()
                delta2 = (out2_pert[key] - out2_base[key])[0, -1, :].float().numpy()
                
                n1 = np.linalg.norm(delta1)
                n2 = np.linalg.norm(delta2)
                
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_val = float(np.dot(delta1, delta2) / (n1 * n2))
                    direction_cosines[li].append(cos_val)
        
        layer_means = {}
        for li in sample_layers:
            if direction_cosines[li]:
                layer_means[li] = float(np.mean(direction_cosines[li]))
        
        if layer_means:
            all_cos = list(layer_means.values())
            results_by_category[category].append(float(np.mean(all_cos)))
    
    # 聚合
    category_stats = {}
    for cat, vals in results_by_category.items():
        category_stats[cat] = {
            "mean_cos": float(np.mean(vals)),
            "std_cos": float(np.std(vals)),
            "median_cos": float(np.median(vals)),
            "min_cos": float(np.min(vals)),
            "max_cos": float(np.max(vals)),
            "n_pairs": len(vals),
        }
    
    # 打印
    print(f"\n跨领域Jacobian一致性 (cos, ε={eps}):")
    for cat in ["NOT", "TENSE", "SYN", "SCOPE", "CROSS", "NONSENSE"]:
        if cat in category_stats:
            s = category_stats[cat]
            print(f"  {cat:12s}: cos = {s['mean_cos']:.4f} ± {s['std_cos']:.4f} "
                  f"[{s['min_cos']:.4f}, {s['max_cos']:.4f}] (n={s['n_pairs']})")
    
    # 保存
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    out_path = TEMP_DIR / f"phase144b_{model_name}_expa_enhanced.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "exp": "A_enhanced", "by_category": category_stats,
                   "n_pairs_total": len(CROSS_DOMAIN_PAIRS)}, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n保存到: {out_path}")
    
    release_model(model)
    return category_stats


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_exp_a_enhanced(model_name)
