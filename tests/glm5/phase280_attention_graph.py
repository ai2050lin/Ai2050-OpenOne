"""
Phase 280: Attention Graph Dynamics (AGD)
===========================================
From hidden state trajectories → attention graph topology.
Directly measure how language reconfigures the computation graph.

Core hypotheses (from Phase 279 + theoretical analyses):
  H1: Attention communities form hierarchically through layers
  H2: Operators (not/if/because/must) induce distinct graph rewiring patterns
  H3: SVO role binding changes attention topology, not just hidden states
  H4: Cross-lingual attention communities are structurally similar (graph isomorphism)

Block A: Attention Community Evolution
  - Build token-level attention graph at each layer
  - Apply community detection (modularity-based spectral clustering)
  - Track community count, size, persistence across layers

Block B: Operator-Induced Graph Rewiring
  - Compare attention graphs for: "happy" vs "not happy", "go" vs "must go", etc.
  - Measure: edge turnover rate, community structure change, hub node shifts

Block C: Role Binding Graph Patterns  
  - "dog chases cat" vs "cat chases dog" — same tokens, different attention topology
  - Track: how subject/object/verb form distinct attention clusters

Block D: Cross-Lingual Graph Comparison
  - EN: "the dog chases the cat" → ZH: "狗追猫" → FR: "le chien chasse le chat"
  - Compare: community structure, modularity, hub alignment

Usage:
  python tests/glm5/phase280_attention_graph.py qwen3
  python tests/glm5/phase280_attention_graph.py glm4
  python tests/glm5/phase280_attention_graph.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers

RESULT_DIR = Path("results/phase280_attention_graph")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Stimulus Definitions =====

# Block A: Attention Community Evolution
# Multi-token sentences to observe community formation
COMMUNITY_SENTENCES = [
    # Short: 3-4 tokens
    "the dog runs",
    "a cat sleeps",
    "birds can fly",
    # Medium: 5-7 tokens  
    "the big red dog runs fast",
    "a small black cat sleeps quietly",
    "the old wise man speaks slowly",
    # SVA: subject-verb agreement
    "the cat eats the fish",
    "the cats eat the fish",
    # Transitive: SVO
    "the dog chases the cat",
    "the man reads the book",
    "the woman writes a letter",
    "the king rules the city",
    # Dative
    "the girl gives the boy a gift",
    "the teacher shows the student the answer",
    # Adjective phrases
    "the red apple is sweet",
    "the cold water feels good",
]

# Block B: Operator-Induced Graph Rewiring
# Use carrier sentences to ensure multi-token context for graph analysis
OPERATOR_GRAPH_TEST = {
    "not": [
        ("he is happy", "he is not happy"),
        ("this is true", "this is not true"),
        ("it is possible", "it is not possible"),
    ],
    "if": [
        ("it will rain", "if it will rain"),
        ("she is ready", "if she is ready"),
        ("it is possible", "if it is possible"),
    ],
    "because": [
        ("he was tired", "because he was tired"),
        ("she was late", "because she was late"),
    ],
    "must": [
        ("you go now", "you must go now"),
        ("they stay here", "they must stay here"),
        ("we leave today", "we must leave today"),
    ],
    "no": [
        ("there is reason", "there is no reason"),
        ("there is a way", "there is no way"),
        ("there is time", "there is no time"),
    ],
    "every": [
        ("a person came", "every person came"),
        ("a day passes", "every day passes"),
    ],
}

# Block C: Role Binding Graph Patterns
ROLE_BINDING_PAIRS = {
    "svo_dog_cat": ("the dog chases the cat", "the cat chases the dog"),
    "svo_man_woman": ("the man loves the woman", "the woman loves the man"),
    "svo_king_city": ("the king rules the city", "the city surrounds the king"),
}

# Block D: Cross-Lingual Graph Comparison
CROSSLINGUAL_TEST = {
    "dog_chase_cat": {
        "en": "the dog chases the cat",
        "zh": "狗追猫",
        "fr": "le chien chasse le chat",
    },
    "man_read_book": {
        "en": "the man reads the book",
        "zh": "男人读书",
        "fr": "l'homme lit le livre",
    },
    "child_eat_apple": {
        "en": "the child eats the apple",
        "zh": "小孩吃苹果",
        "fr": "l'enfant mange la pomme",
    },
}


# ===== Model Loading =====

def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flash attention compatibility check
    attn_impl = "eager"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        log_time(f"  flash_attn available, using {attn_impl}")
    except ImportError:
        log_time(f"  flash_attn not available, using {attn_impl}")

    # output_attentions not supported with flash_attention_2 — use eager for this phase
    log_time("  Note: Phase 280 requires output_attentions, using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # Must use eager for attention output
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== Attention Graph Construction =====

def extract_attention_graph(model, tokenizer, device, prompt, n_layers, max_len=64):
    """
    Extract per-layer attention matrices and tokenized info.
    Returns:
      attentions: {layer_idx: [n_heads, seq_len, seq_len] numpy}
      tokens: list of decoded token strings
      token_ids: list of token ids
    """
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    
    seq_len = input_ids.shape[1]
    decoded = [tokenizer.decode([tid.item()]).strip() for tid in input_ids[0]]

    with torch.no_grad():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        except Exception as e:
            log_time(f"  WARNING: forward failed for '{prompt[:60]}': {e}")
            return None, None, None

    # outputs.attentions: tuple of (layer,) each [batch, n_heads, seq_len, seq_len]
    attn_dict = {}
    for layer_idx, attn_tensor in enumerate(outputs.attentions):
        if attn_tensor is not None:
            attn_dict[layer_idx] = attn_tensor[0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]

    return attn_dict, decoded, input_ids[0].cpu().tolist()


def build_token_graph(attn_matrix, threshold=None):
    """
    Build a token-level graph from attention matrix.
    attn_matrix: [n_heads, seq_len, seq_len] or [seq_len, seq_len]
    threshold: if None, use top-3 connections per token
    
    Returns:
      adj: [seq_len, seq_len] binary adjacency matrix
      weights: [seq_len, seq_len] float weight matrix
    """
    # Average across heads if 3D input
    if len(attn_matrix.shape) == 3:
        attn = np.mean(attn_matrix, axis=0)  # [seq_len, seq_len]
    else:
        attn = attn_matrix
    
    seq_len = attn.shape[0]
    
    if threshold is None:
        # Adaptive threshold: keep top-3 attention targets per source token
        adj = np.zeros((seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            if i == 0:  # skip BOS/first token self-connections
                continue
            top_k = min(3, seq_len)
            top_indices = np.argsort(attn[i])[-top_k:]
            adj[i, top_indices] = 1
    else:
        adj = (attn > threshold).astype(np.float32)
    
    return adj, attn


def compute_graph_metrics(adj, weights, tokens):
    """
    Compute graph-level metrics.
    
    Returns dict with:
      - density: edge count / possible edges
      - avg_degree: mean degree
      - modularity_estimate: approximate modularity score
      - hub_tokens: tokens with highest weighted degree
      - n_components: number of weakly connected components
    """
    seq_len = adj.shape[0]
    n_possible = seq_len * (seq_len - 1)
    n_edges = int(np.sum(adj > 0))
    density = n_edges / max(n_possible, 1)
    
    degrees = np.sum(adj, axis=1) + np.sum(adj, axis=0)
    avg_degree = np.mean(degrees)
    
    # Weighted degree for hub detection
    weighted_degree = np.sum(weights, axis=1) + np.sum(weights, axis=0)
    hub_indices = np.argsort(weighted_degree)[-3:][::-1]
    hub_tokens = [tokens[i] for i in hub_indices if i < len(tokens)]
    
    # Simple modularity estimate using degree-based partition
    # Q ≈ (edges within clusters - expected edges) / total edges
    modularity = _estimate_modularity(adj, weights)
    
    # Connected components (simple BFS)
    visited = set()
    n_components = 0
    for i in range(seq_len):
        if i not in visited and np.sum(adj[i]) > 0:
            stack = [i]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for u in range(seq_len):
                    if adj[v, u] > 0 and u not in visited:
                        stack.append(u)
            n_components += 1
    
    return {
        "density": float(density),
        "n_edges": n_edges,
        "avg_degree": float(avg_degree),
        "modularity_estimate": float(modularity),
        "hub_tokens": hub_tokens,
        "n_components": n_components,
    }


def _estimate_modularity(adj, weights):
    """Simple modularity estimate."""
    seq_len = adj.shape[0]
    total_w = np.sum(weights)
    if total_w < 1e-10:
        return 0.0
    
    # Use spectral clustering to get approximate communities
    # Build Laplacian and get 2nd eigenvector for bipartition
    deg = np.sum(weights, axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-10)))
    L_norm = np.eye(seq_len) - D_sqrt_inv @ weights @ D_sqrt_inv
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        # Fiedler vector (2nd smallest eigenvector)
        if seq_len >= 2:
            fiedler = eigenvectors[:, 1]
            partition = (fiedler > 0).astype(int)
            
            # Compute modularity Q
            m = total_w
            Q = 0.0
            for i in range(seq_len):
                for j in range(seq_len):
                    if partition[i] == partition[j]:
                        expected = deg[i] * deg[j] / m
                        Q += weights[i, j] - expected
            Q /= m
            return max(0.0, float(Q))
    except Exception:
        pass
    
    return 0.0


# ===== Block A: Attention Community Evolution =====

def block_a_community_evolution(model, tokenizer, device, model_name, n_layers):
    """
    Track how attention communities form and evolve across layers.
    Core question: Do tokens form stable attention clusters? How do they change?
    """
    log_time("=" * 50)
    log_time("Block A: Attention Community Evolution")
    log_time("=" * 50)
    
    # Sample layers for detailed analysis
    step = max(1, n_layers // 8)
    sample_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    all_results = {}
    
    for sent_idx, sent in enumerate(COMMUNITY_SENTENCES):
        if sent_idx > 0 and sent_idx % 4 == 0:
            log_time(f"  Community sentences: {sent_idx}/{len(COMMUNITY_SENTENCES)}")
        
        attn_dict, tokens, _ = extract_attention_graph(
            model, tokenizer, device, sent, n_layers, max_len=64
        )
        if attn_dict is None:
            continue
        
        sent_metrics = {}
        for layer_idx in sample_layers:
            if layer_idx not in attn_dict:
                continue
            
            attn = attn_dict[layer_idx]  # [n_heads, seq_len, seq_len]
            adj, weights = build_token_graph(attn)
            metrics = compute_graph_metrics(adj, weights, tokens)
            
            sent_metrics[str(layer_idx)] = metrics
        
        key = sent.replace(" ", "_")[:40]
        all_results[key] = {
            "sentence": sent,
            "n_tokens": len(tokens),
            "tokens": tokens,
            "per_layer_metrics": sent_metrics,
        }
    
    # Aggregate statistics per layer
    layer_aggregates = {}
    for layer_idx in sample_layers:
        densities, modularities, n_comps, densities_nonzero = [], [], [], []
        for key, data in all_results.items():
            lk = str(layer_idx)
            if lk in data["per_layer_metrics"]:
                m = data["per_layer_metrics"][lk]
                densities.append(m["density"])
                modularities.append(m["modularity_estimate"])
                n_comps.append(m["n_components"])
        
        if densities:
            layer_aggregates[str(layer_idx)] = {
                "n_sentences": len(densities),
                "mean_density": float(np.mean(densities)),
                "mean_modularity": float(np.mean(modularities)),
                "mean_n_components": float(np.mean(n_comps)),
            }
    
    # Identify "community formation layers" where modularity peaks
    mod_by_layer = [(int(l), d["mean_modularity"]) for l, d in layer_aggregates.items()]
    mod_by_layer.sort(key=lambda x: x[0])
    
    log_time(f"  Analyzed {len(all_results)} sentences across {len(sample_layers)} layers")
    for l, mod_val in mod_by_layer:
        if l in sample_layers:
            dens = layer_aggregates[str(l)].get("mean_density", 0)
            ncomp = layer_aggregates[str(l)].get("mean_n_components", 0)
            log_time(f"    L{l:2d}: modularity={mod_val:.4f}, density={dens:.4f}, n_components={ncomp:.1f}")
    
    results = {
        "model": model_name,
        "n_sentences": len(all_results),
        "sample_layers": sample_layers,
        "per_sentence": all_results,
        "layer_aggregates": layer_aggregates,
        "modularity_by_layer": {str(l): v for l, v in mod_by_layer},
    }
    
    out_path = RESULT_DIR / f"{model_name}_block_a_community.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===== Block B: Operator-Induced Graph Rewiring =====

def block_b_operator_rewiring(model, tokenizer, device, model_name, n_layers):
    """
    How do operators change the attention graph structure?
    Compare graph before/after operator application.
    """
    log_time("=" * 50)
    log_time("Block B: Operator-Induced Graph Rewiring")
    log_time("=" * 50)
    
    step = max(1, n_layers // 5)
    sample_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    rewiring_results = {}
    
    total_pairs = sum(len(v) for v in OPERATOR_GRAPH_TEST.values())
    pair_count = 0
    
    for op_name, pairs in OPERATOR_GRAPH_TEST.items():
        for base, combined in pairs:
            pair_count += 1
            if pair_count % 4 == 0:
                log_time(f"  Operator pairs: {pair_count}/{total_pairs}")
            
            # Extract graphs for base and combined
            attn_base, tokens_base, _ = extract_attention_graph(
                model, tokenizer, device, base, n_layers, max_len=32
            )
            attn_comb, tokens_comb, _ = extract_attention_graph(
                model, tokenizer, device, combined, n_layers, max_len=32
            )
            
            if attn_base is None or attn_comb is None:
                continue
            
            pair_metrics = {}
            for layer_idx in sample_layers:
                if layer_idx not in attn_base or layer_idx not in attn_comb:
                    continue
                
                # Build graphs
                adj_base, w_base = build_token_graph(attn_base[layer_idx])
                adj_comb, w_comb = build_token_graph(attn_comb[layer_idx])
                
                # Edge turnover: how many edges changed?
                # Min size for comparison
                min_len = min(adj_base.shape[0], adj_comb.shape[0])
                
                # Align on first min_len tokens
                adj_b = adj_base[:min_len, :min_len]
                adj_c = adj_comb[:min_len, :min_len]
                
                n_possible = min_len * (min_len - 1)
                edge_turnover = np.sum(np.abs(adj_b - adj_c)) / max(n_possible, 1)
                
                # Community structure change
                metrics_base = compute_graph_metrics(adj_b, w_base[:min_len, :min_len], tokens_base[:min_len])
                metrics_comb = compute_graph_metrics(adj_c, w_comb[:min_len, :min_len], tokens_comb[:min_len])
                
                modularity_delta = metrics_comb["modularity_estimate"] - metrics_base["modularity_estimate"]
                density_ratio = metrics_comb["density"] / max(metrics_base["density"], 1e-6)
                
                # Hub shift: overlap of hub tokens
                hubs_base = set(metrics_base["hub_tokens"])
                hubs_comb = set(metrics_comb["hub_tokens"])
                hub_jaccard = len(hubs_base & hubs_comb) / max(len(hubs_base | hubs_comb), 1)
                
                pair_metrics[str(layer_idx)] = {
                    "edge_turnover": float(edge_turnover),
                    "modularity_delta": float(modularity_delta),
                    "density_ratio": float(density_ratio),
                    "hub_jaccard": float(hub_jaccard),
                    "n_edges_base": metrics_base["n_edges"],
                    "n_edges_combined": metrics_comb["n_edges"],
                }
            
            rewiring_results[f"{op_name}:{base}->{combined}"] = {
                "operator": op_name,
                "base": base,
                "combined": combined,
                "n_tokens_base": len(tokens_base),
                "n_tokens_combined": len(tokens_comb),
                "per_layer": pair_metrics,
            }
    
    # Aggregate by operator type
    operator_summary = defaultdict(lambda: defaultdict(list))
    for key, data in rewiring_results.items():
        op = data["operator"]
        for lk, metrics in data["per_layer"].items():
            operator_summary[op][lk].append(metrics["edge_turnover"])
    
    operator_profile = {}
    for op, layer_data in operator_summary.items():
        op_profile = {}
        for lk, turnovers in layer_data.items():
            op_profile[lk] = {
                "mean_edge_turnover": float(np.mean(turnovers)),
                "n_pairs": len(turnovers),
            }
        operator_profile[op] = op_profile
    
    # Log key findings
    log_time(f"  Analyzed {len(rewiring_results)} operator pairs")
    closest_mid = str(min(sample_layers, key=lambda x: abs(x - n_layers // 2)))
    for op, profile in sorted(operator_profile.items()):
        if closest_mid in profile:
            log_time(f"  {op}: L{closest_mid} edge_turnover={profile[closest_mid]['mean_edge_turnover']:.4f}")
    
    results = {
        "model": model_name,
        "n_pairs": len(rewiring_results),
        "sample_layers": sample_layers,
        "per_pair": rewiring_results,
        "operator_profile": operator_profile,
    }
    
    out_path = RESULT_DIR / f"{model_name}_block_b_operator_rewiring.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===== Block C: Role Binding Graph Patterns =====

def block_c_role_binding(model, tokenizer, device, model_name, n_layers):
    """
    Test if SVO role binding changes attention graph topology.
    "dog chases cat" vs "cat chases dog" — same tokens, different graph.
    """
    log_time("=" * 50)
    log_time("Block C: Role Binding Graph Patterns")
    log_time("=" * 50)
    
    step = max(1, n_layers // 5)
    sample_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    role_results = {}
    
    for pair_name, (sent_a, sent_b) in ROLE_BINDING_PAIRS.items():
        log_time(f"  Analyzing: {pair_name}")
        
        attn_a, tokens_a, _ = extract_attention_graph(
            model, tokenizer, device, sent_a, n_layers, max_len=64
        )
        attn_b, tokens_b, _ = extract_attention_graph(
            model, tokenizer, device, sent_b, n_layers, max_len=64
        )
        
        if attn_a is None or attn_b is None:
            log_time(f"  WARNING: Failed to extract for {pair_name}")
            continue
        
        pair_metrics = {}
        for layer_idx in sample_layers:
            if layer_idx not in attn_a or layer_idx not in attn_b:
                continue
            
            attn_matrix_a = np.mean(attn_a[layer_idx], axis=0)  # [seq, seq]
            attn_matrix_b = np.mean(attn_b[layer_idx], axis=0)
            
            min_len = min(attn_matrix_a.shape[0], attn_matrix_b.shape[0])
            a_ = attn_matrix_a[:min_len, :min_len]
            b_ = attn_matrix_b[:min_len, :min_len]
            
            # Frobenius distance between attention matrices
            frob_dist = float(np.linalg.norm(a_ - b_) / np.sqrt(min_len * min_len))
            
            # Max attention difference element
            max_diff = float(np.max(np.abs(a_ - b_)))
            
            # Correlation between attention patterns
            a_flat = a_.flatten()
            b_flat = b_.flatten()
            corr = float(np.corrcoef(a_flat, b_flat)[0, 1]) if min_len > 1 else 0
            
            adj_a, w_a = build_token_graph(attn_a[layer_idx])
            adj_b, w_b = build_token_graph(attn_b[layer_idx])
            metrics_a = compute_graph_metrics(adj_a, w_a, tokens_a)
            metrics_b = compute_graph_metrics(adj_b, w_b, tokens_b)
            
            modularity_delta = metrics_b["modularity_estimate"] - metrics_a["modularity_estimate"]
            hubs_a = set(metrics_a["hub_tokens"])
            hubs_b = set(metrics_b["hub_tokens"])
            hub_jaccard = len(hubs_a & hubs_b) / max(len(hubs_a | hubs_b), 1)
            
            pair_metrics[str(layer_idx)] = {
                "frobenius_distance": frob_dist,
                "max_attn_diff": max_diff,
                "attention_correlation": corr,
                "modularity_delta": modularity_delta,
                "hub_jaccard": hub_jaccard,
                "tokens_a": tokens_a[:min_len],
                "tokens_b": tokens_b[:min_len],
            }
        
        role_results[pair_name] = {
            "sent_a": sent_a,
            "sent_b": sent_b,
            "per_layer": pair_metrics,
        }
        
        # Log
        closest_l = str(min(sample_layers, key=lambda x: abs(x - n_layers // 2)))
        if closest_l in pair_metrics:
            pm = pair_metrics[closest_l]
            log_time(f"    L{closest_l}: frob={pm['frobenius_distance']:.4f}, "
                     f"corr={pm['attention_correlation']:.4f}, hub_jac={pm['hub_jaccard']:.2f}")
    
    results = {
        "model": model_name,
        "n_pairs": len(role_results),
        "sample_layers": sample_layers,
        "per_pair": role_results,
    }
    
    out_path = RESULT_DIR / f"{model_name}_block_c_role_binding.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===== Block D: Cross-Lingual Graph Comparison =====

def block_d_crosslingual(model, tokenizer, device, model_name, n_layers):
    """
    Test if attention graph structure is language-invariant.
    Compare attention communities across EN/ZH/FR for same meaning.
    """
    log_time("=" * 50)
    log_time("Block D: Cross-Lingual Attention Graph Comparison")
    log_time("=" * 50)
    
    step = max(1, n_layers // 5)
    sample_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    crosslingual_results = {}
    
    for concept, langs in CROSSLINGUAL_TEST.items():
        log_time(f"  Concept: {concept}")
        
        # Extract graphs for all languages
        lang_graphs = {}
        for lang, sent in langs.items():
            attn, tokens, _ = extract_attention_graph(
                model, tokenizer, device, sent, n_layers, max_len=64
            )
            if attn is not None:
                lang_graphs[lang] = {
                    "attn": attn,
                    "tokens": tokens,
                    "sentence": sent,
                    "n_tokens": len(tokens),
                }
        
        if len(lang_graphs) < 2:
            continue
        
        # Pairwise comparison
        lang_list = sorted(lang_graphs.keys())
        pair_comparisons = {}
        
        for i in range(len(lang_list)):
            for j in range(i + 1, len(lang_list)):
                l1, l2 = lang_list[i], lang_list[j]
                g1, g2 = lang_graphs[l1], lang_graphs[l2]
                
                layer_metrics = {}
                for layer_idx in sample_layers:
                    if layer_idx not in g1["attn"] or layer_idx not in g2["attn"]:
                        continue
                    
                    a1 = np.mean(g1["attn"][layer_idx], axis=0)
                    a2 = np.mean(g2["attn"][layer_idx], axis=0)
                    
                    # Normalize to compare graph structure regardless of seq_len
                    min_len = min(a1.shape[0], a2.shape[0])
                    a1_norm = a1[:min_len, :min_len]
                    a2_norm = a2[:min_len, :min_len]
                    
                    frob_dist = float(np.linalg.norm(a1_norm - a2_norm) / np.sqrt(min_len * min_len))
                    
                    # Build graphs and compare properties
                    adj1, w1 = build_token_graph(g1["attn"][layer_idx])
                    adj2, w2 = build_token_graph(g2["attn"][layer_idx])
                    m1 = compute_graph_metrics(adj1, w1, g1["tokens"])
                    m2 = compute_graph_metrics(adj2, w2, g2["tokens"])
                    
                    layer_metrics[str(layer_idx)] = {
                        "frobenius_distance": frob_dist,
                        "modularity_1": m1["modularity_estimate"],
                        "modularity_2": m2["modularity_estimate"],
                        "modularity_diff": abs(m1["modularity_estimate"] - m2["modularity_estimate"]),
                        "density_ratio": m2["density"] / max(m1["density"], 1e-6),
                    }
                
                pair_key = f"{l1}_vs_{l2}"
                pair_comparisons[pair_key] = {
                    "lang_1": l1,
                    "lang_2": l2,
                    "sent_1": g1["sentence"],
                    "sent_2": g2["sentence"],
                    "n_tokens_1": g1["n_tokens"],
                    "n_tokens_2": g2["n_tokens"],
                    "per_layer": layer_metrics,
                }
        
        crosslingual_results[concept] = {
            "languages": lang_list,
            "pairwise": pair_comparisons,
        }
        
        # Log
        for pair_key, pc in pair_comparisons.items():
            closest_l = str(min(sample_layers, key=lambda x: abs(x - n_layers // 2)))
            if closest_l in pc["per_layer"]:
                m = pc["per_layer"][closest_l]
                log_time(f"    {pair_key}: L{closest_l} frob={m['frobenius_distance']:.4f}, "
                         f"mod_diff={m['modularity_diff']:.4f}")
    
    results = {
        "model": model_name,
        "n_concepts": len(crosslingual_results),
        "sample_layers": sample_layers,
        "per_concept": crosslingual_results,
    }
    
    out_path = RESULT_DIR / f"{model_name}_block_d_crosslingual.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===== Main =====

def main():
    global _log_file
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    
    log_path = RESULT_DIR / f"{model_name}_phase280.log"
    _log_file = str(log_path)
    
    log_time(f"Phase 280: Attention Graph Dynamics")
    log_time(f"Model: {model_name}")
    t_start = time.time()
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    log_time(f"Model info: n_layers={n_layers}, d_model={info.d_model}, class={info.model_class}")
    
    all_results = {}
    
    # Block A: Community Evolution
    t_a = time.time()
    results_a = block_a_community_evolution(model, tokenizer, device, model_name, n_layers)
    all_results["block_a"] = {"elapsed": round(time.time() - t_a, 1)}
    log_time(f"Block A completed in {time.time() - t_a:.1f}s")
    
    # Block B: Operator Rewiring
    t_b = time.time()
    results_b = block_b_operator_rewiring(model, tokenizer, device, model_name, n_layers)
    all_results["block_b"] = {"elapsed": round(time.time() - t_b, 1)}
    log_time(f"Block B completed in {time.time() - t_b:.1f}s")
    
    # Block C: Role Binding
    t_c = time.time()
    results_c = block_c_role_binding(model, tokenizer, device, model_name, n_layers)
    all_results["block_c"] = {"elapsed": round(time.time() - t_c, 1)}
    log_time(f"Block C completed in {time.time() - t_c:.1f}s")
    
    # Block D: Cross-Lingual
    t_d = time.time()
    results_d = block_d_crosslingual(model, tokenizer, device, model_name, n_layers)
    all_results["block_d"] = {"elapsed": round(time.time() - t_d, 1)}
    log_time(f"Block D completed in {time.time() - t_d:.1f}s")
    
    # Summary
    total_time = time.time() - t_start
    log_time("=" * 50)
    log_time(f"PHASE 280 COMPLETE — Total: {total_time:.1f}s")
    log_time(f"  A-Community: {all_results['block_a']['elapsed']}s")
    log_time(f"  B-Operator:  {all_results['block_b']['elapsed']}s")
    log_time(f"  C-Role:      {all_results['block_c']['elapsed']}s")
    log_time(f"  D-XLingual:  {all_results['block_d']['elapsed']}s")
    
    # Release
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("GPU memory released.")


if __name__ == "__main__":
    main()
