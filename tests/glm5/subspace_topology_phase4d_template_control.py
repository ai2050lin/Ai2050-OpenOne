"""
Phase 58d: 控制模板相似度验证
=====================================
关键问题: 
  synonym overlap=0.42 > hyponym overlap=0.24
  这是否因为synonym模板(big/large)几乎只差一个词, 导致结构相似度膨胀?

验证方案:
  1. 用语法不同的模板重新测试synonym对
  2. 对比同义词在"相似模板"vs"不同模板"下的overlap
  3. 测试: 如果模板完全不同, synonym overlap是否仍然>hyponym?
"""

import sys, json, numpy as np
from pathlib import Path
from datetime import datetime

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import load_model, get_model_info, release_model, safe_decode
from subspace_topology_phase4b_backbone_decode import WORD_TEMPLATES
import torch

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)

# 完全不同语境的同义词模板
SYNONYM_DIFF_CONTEXT = {
    "big": [
        "Size matters when the big project started",
        "Everyone noticed the big difference immediately",
        "The big question remains unanswered",
        "She has a big personality that fills rooms",
        "The big picture reveals hidden patterns",
        "He took a big risk with that investment",
        "The big picture window overlooked the valley",
        "There is a big gap between theory and practice",
        "The big moment finally arrived",
        "She made a big impression on the committee",
        "The big story broke at midnight",
        "His big idea changed everything",
        "The big challenge is yet to come",
        "She scored a big victory in court",
        "The big reveal surprised everyone",
    ],
    "large": [
        "Scale matters when the large project started",
        "Everyone noticed the large difference immediately",
        "The large question remains unanswered",
        "She has a large personality that fills rooms",
        "The large picture reveals hidden patterns",
        "He took a large risk with that investment",
        "The large picture window overlooked the valley",
        "There is a large gap between theory and practice",
        "The large moment finally arrived",
        "She made a large impression on the committee",
        "The large story broke at midnight",
        "His large idea changed everything",
        "The large challenge is yet to come",
        "She scored a large victory in court",
        "The large reveal surprised everyone",
    ],
}

# 上下位不同语境 — 确保目标词出现在句子中
HYPONYM_DIFF_CONTEXT = {
    "apple": [
        "The apple company released a new product today",
        "She bit into the apple from the autumn harvest",
        "The apple orchard stretched across the hillside",
        "He cored the apple and sliced it for the pie",
        "The apple blossom appeared in early spring",
        "This apple variety was both sweet and tart",
        "She juiced the apple for breakfast this morning",
        "The worm ruined the apple she had just bought",
        "The apple season began with early varieties",
        "The apple cider was warm and comforting",
        "The apple tree was heavy with ripe fruit",
        "He grew an apple tree on the balcony",
        "The apple harvest festival was last weekend",
        "She picked the apple from the highest branch",
        "The apple peel was shiny and red today",
    ],
    "fruit": [
        "The fruit market had a wide selection today",
        "She enjoyed the fruit from the summer harvest",
        "The fruit grove stretched across the hillside",
        "He sorted the fruit carefully before packing",
        "The fruit blossom appeared in early spring",
        "This fruit selection was sweet and varied",
        "She blended the fruit for breakfast this morning",
        "The mold ruined the fruit she had just bought",
        "The fruit season began with early harvests",
        "The fruit punch was refreshing and sweet",
        "The fruit trees were heavy with ripe crops",
        "He grew organic fruit on the farm",
        "The fruit harvest festival was last weekend",
        "She picked the fruit from the highest branch",
        "The fruit skin was smooth and colorful today",
    ],
}

TEST_PAIRS = [
    {"key": "synonym_original", "w_a": "big", "w_b": "large", 
     "templates_a": None, "templates_b": None,
     "desc": "synonym(original templates)"},
    {"key": "synonym_diff_ctx", "w_a": "big", "w_b": "large",
     "templates_a": SYNONYM_DIFF_CONTEXT["big"], 
     "templates_b": SYNONYM_DIFF_CONTEXT["large"],
     "desc": "synonym(different context)"},
    {"key": "hyponym_original", "w_a": "apple", "w_b": "fruit",
     "templates_a": None, "templates_b": None,
     "desc": "hyponym(original templates)"},
    {"key": "hyponym_diff_ctx", "w_a": "apple", "w_b": "fruit",
     "templates_a": HYPONYM_DIFF_CONTEXT["apple"],
     "templates_b": HYPONYM_DIFF_CONTEXT["fruit"],
     "desc": "hyponym(different context)"},
]


def find_target_pos_in_full(tokenizer, input_ids, target_word):
    tokens_list = input_ids[0].tolist()
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped == target_word.lower():
                return i, j - i
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 3:
                return i, j - i
    return None, None


def collect_word_activations(model, tokenizer, device, word, templates, target_layers, n_layers):
    activations = {li: [] for li in target_layers}
    found = 0
    with torch.no_grad():
        for tmpl in templates:
            inputs = tokenizer(tmpl, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            pos, tlen = find_target_pos_in_full(tokenizer, input_ids, word)
            if pos is None or pos >= seq_len:
                continue
            actual_pos = min(pos + (tlen // 2), seq_len - 1)
            found += 1
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states
            for li in target_layers:
                activations[li].append(hidden[li + 1][0, actual_pos].detach().cpu().float().numpy())
    return activations, found


def pca_subspace(vectors, n_dims=10):
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_c = X - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n = min(n_dims, Vt.shape[0])
    return Vt[:n].T, (S ** 2) / len(X_c), mean


def subspace_overlap(basis_a, basis_b):
    if basis_a is None or basis_b is None:
        return 0.0
    proj = basis_b.T @ basis_a @ basis_a.T @ basis_b
    k = min(basis_a.shape[1], basis_b.shape[1])
    return float(np.trace(proj) / k)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    model_name = args.model
    n_dims = 10
    
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    target_layers = sorted(set([0, 1] + list(range(0, n_layers, max(1, n_layers // 6)))))
    log_time(f"{model_name}: layers={target_layers}")
    
    results = {}
    
    for pair in TEST_PAIRS:
        key = pair["key"]
        w_a = pair["w_a"]
        w_b = pair["w_b"]
        
        templates_a = pair["templates_a"] if pair["templates_a"] else WORD_TEMPLATES[w_a]
        templates_b = pair["templates_b"] if pair["templates_b"] else WORD_TEMPLATES[w_b]
        
        log_time(f"Testing {key}: {pair['desc']}")
        
        acts_a, found_a = collect_word_activations(model, tokenizer, device, w_a, templates_a, target_layers, n_layers)
        acts_b, found_b = collect_word_activations(model, tokenizer, device, w_b, templates_b, target_layers, n_layers)
        
        log_time(f"  Found: {w_a}={found_a}, {w_b}={found_b}")
        
        layer_results = {}
        for li in target_layers:
            if len(acts_a[li]) >= 2 and len(acts_b[li]) >= 2:
                basis_a, _, _ = pca_subspace(acts_a[li], n_dims)
                basis_b, _, _ = pca_subspace(acts_b[li], n_dims)
                overlap = subspace_overlap(basis_a, basis_b)
                layer_results[str(li)] = {"overlap": float(overlap)}
        
        results[key] = {"desc": pair["desc"], "layers": layer_results}
    
    log_time("")
    log_time("=" * 70)
    log_time(f"PHASE 58d: Template Control - {model_name}")
    log_time("=" * 70)
    
    log_time("\n--- Overlap by Template Type (mid layer) ---")
    mid_li = target_layers[len(target_layers) // 2]
    for key, data in results.items():
        mid_overlap = data["layers"].get(str(mid_li), {}).get("overlap", 0)
        log_time(f"  {key:30s}: L{mid_li} overlap={mid_overlap:.3f}")
    
    log_time("\n--- Full Layer Evolution ---")
    for key, data in results.items():
        parts = []
        for lk in sorted(data["layers"].keys(), key=int):
            parts.append("L{}={:.3f}".format(lk, data["layers"][lk]["overlap"]))
        log_time("  {}: {}".format(key, " | ".join(parts)))
    
    # 关键对比
    syn_orig = results.get("synonym_original", {}).get("layers", {}).get(str(mid_li), {}).get("overlap", 0)
    syn_diff = results.get("synonym_diff_ctx", {}).get("layers", {}).get(str(mid_li), {}).get("overlap", 0)
    hypo_orig = results.get("hyponym_original", {}).get("layers", {}).get(str(mid_li), {}).get("overlap", 0)
    hypo_diff = results.get("hyponym_diff_ctx", {}).get("layers", {}).get(str(mid_li), {}).get("overlap", 0)
    
    log_time("\n--- KEY COMPARISON ---")
    log_time(f"  synonym(original)={syn_orig:.3f} vs synonym(diff_ctx)={syn_diff:.3f}  delta={syn_orig-syn_diff:+.3f}")
    log_time(f"  hyponym(original)={hypo_orig:.3f} vs hyponym(diff_ctx)={hypo_diff:.3f}  delta={hypo_orig-hypo_diff:+.3f}")
    log_time(f"  synonym-hyponym gap(original)={syn_orig-hypo_orig:.3f}")
    log_time(f"  synonym-hyponym gap(diff_ctx)  ={syn_diff-hypo_diff:.3f}")
    
    if syn_diff > hypo_diff:
        log_time("  CONCLUSION: synonym>hyponym overlap is ROBUST across template types!")
    else:
        log_time("  CONCLUSION: synonym>hyponym overlap may be template artifact")
    
    release_model(model)
    log_time("Done!")


if __name__ == "__main__":
    main()
