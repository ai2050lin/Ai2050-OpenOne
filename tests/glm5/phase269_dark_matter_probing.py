"""
Phase 269: Transformer计算暗物质 — W_U不可见空间的语义编码探测
================================================================

Phase 268的决定性发现：W_U只能读取4-14%的hidden state方差。
86-96%的hidden state方差存在于W_U不可见的方向上。

核心假设：语言的语法和语义编码主要存在于W_U不可见的子空间中。

验证方法：
1. 将hidden state空间分解为W_U可见和W_U不可见两个子空间
2. 在两个子空间中分别做语义特征probing（number, animacy, tense）
3. 如果W_U不可见空间的probing准确率 >> W_U可见空间 → 假设正确

关键指标：
- W_U可见空间 (V_vis): W_U的前k个右奇异向量张成的空间
- W_U不可见空间 (V_inv): 正交补空间
- 在V_vis和V_inv中分别做logistic regression probing
- 比较两个空间对语义特征的分类准确率

=== 用法 ===
  python tests/glm5/phase269_dark_matter_probing.py qwen3
  python tests/glm5/phase269_dark_matter_probing.py glm4
  python tests/glm5/phase269_dark_matter_probing.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase269_dark_matter_probing")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Prompt Generation with Clear Semantic Labels =====

def generate_labeled_prompts():
    """Generate prompts with clear semantic labels for probing.
    
    Returns: list of (prompt, label_dict) tuples
    label_dict keys: number (singular/plural), animacy (animate/inanimate),
                     tense (past/present/future), category (animal/tool/clothing/body)
    """
    labeled = []
    
    # === Number feature: singular vs plural ===
    SING_NOUNS = [
        "cat", "dog", "bird", "fish", "tree", "house", "car", "book",
        "phone", "chair", "table", "door", "river", "mountain", "cloud",
        "king", "queen", "child", "woman", "man", "doctor", "teacher",
        "soldier", "artist", "writer", "farmer", "driver", "baker",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "eagle",
    ]
    
    for noun in SING_NOUNS:
        labeled.append((f"The {noun} sits", {"number": "singular", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun} runs", {"number": "singular", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun} thinks", {"number": "singular", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun} will go", {"number": "singular", "animacy": "", "tense": "future", "category": ""}))
        labeled.append((f"The {noun} went home", {"number": "singular", "animacy": "", "tense": "past", "category": ""}))
    
    for noun in SING_NOUNS[:25]:
        labeled.append((f"The {noun}s sit", {"number": "plural", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun}s run", {"number": "plural", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun}s think", {"number": "plural", "animacy": "", "tense": "present", "category": ""}))
        labeled.append((f"The {noun}s will go", {"number": "plural", "animacy": "", "tense": "future", "category": ""}))
        labeled.append((f"The {noun}s went home", {"number": "plural", "animacy": "", "tense": "past", "category": ""}))
    
    # === Animacy feature: animate vs inanimate ===
    ANIMATE = ["cat", "dog", "bird", "horse", "cow", "child", "woman", "man",
               "boy", "girl", "baby", "friend", "teacher", "doctor", "king",
               "queen", "prince", "soldier", "artist", "writer", "rabbit",
               "deer", "bear", "wolf", "fox", "lion", "tiger", "elephant"]
    INANIMATE = ["rock", "chair", "table", "door", "wall", "road", "bridge",
                 "tower", "boat", "ship", "train", "car", "phone", "book",
                 "pen", "clock", "shirt", "pants", "cup", "plate", "hammer",
                 "knife", "ball", "key", "lamp", "mirror", "stone", "wood"]
    
    for word in ANIMATE:
        labeled.append((f"The {word} thinks about tomorrow", {"number": "", "animacy": "animate", "tense": "", "category": ""}))
        labeled.append((f"The {word} is alive", {"number": "", "animacy": "animate", "tense": "", "category": ""}))
    for word in INANIMATE:
        labeled.append((f"The {word} sits on the shelf", {"number": "", "animacy": "inanimate", "tense": "", "category": ""}))
        labeled.append((f"The {word} is heavy", {"number": "", "animacy": "inanimate", "tense": "", "category": ""}))
    
    # === Category feature: animal vs tool vs clothing vs body ===
    ANIMALS = ["cat", "dog", "bird", "horse", "fish", "lion", "tiger", "bear",
               "wolf", "deer", "rabbit", "mouse", "elephant", "monkey", "whale",
               "dolphin", "snake", "frog", "duck", "goose"]
    TOOLS = ["hammer", "wrench", "saw", "drill", "screwdriver", "pliers",
             "chisel", "axe", "shovel", "rake", "knife", "scissors", "needle",
             "rope", "wire", "nail", "screw", "bolt", "glue", "tape"]
    CLOTHING = ["shirt", "pants", "dress", "skirt", "jacket", "coat", "hat",
                "scarf", "gloves", "socks", "shoes", "boots", "belt", "tie",
                "vest", "sweater", "hoodie", "blouse", "shorts", "jeans"]
    BODY_PARTS = ["head", "face", "eye", "ear", "nose", "mouth", "neck",
                  "arm", "hand", "finger", "chest", "back", "leg", "foot",
                  "shoulder", "elbow", "knee", "wrist", "thumb", "tooth"]
    
    for word in ANIMALS:
        labeled.append((f"The {word} is interesting", {"number": "", "animacy": "", "tense": "", "category": "animal"}))
        labeled.append((f"I saw a {word} today", {"number": "", "animacy": "", "tense": "", "category": "animal"}))
    for word in TOOLS:
        labeled.append((f"The {word} is useful", {"number": "", "animacy": "", "tense": "", "category": "tool"}))
        labeled.append((f"I need a {word} now", {"number": "", "animacy": "", "tense": "", "category": "tool"}))
    for word in CLOTHING:
        labeled.append((f"The {word} is clean", {"number": "", "animacy": "", "tense": "", "category": "clothing"}))
        labeled.append((f"She wore the {word} today", {"number": "", "animacy": "", "tense": "", "category": "clothing"}))
    for word in BODY_PARTS:
        labeled.append((f"My {word} hurts", {"number": "", "animacy": "", "tense": "", "category": "body"}))
        labeled.append((f"The {word} was injured", {"number": "", "animacy": "", "tense": "", "category": "body"}))
    
    # === Tense feature: past vs present vs future ===
    for noun in SING_NOUNS[:15]:
        labeled.append((f"The {noun} will come", {"number": "", "animacy": "", "tense": "future", "category": ""}))
        labeled.append((f"The {noun} went away", {"number": "", "animacy": "", "tense": "past", "category": ""}))
        labeled.append((f"The {noun} is here", {"number": "", "animacy": "", "tense": "present", "category": ""}))
    
    # Filter: only keep prompts with at least one label
    labeled = [(p, l) for p, l in labeled if any(v for v in l.values())]
    
    random.seed(42)
    random.shuffle(labeled)
    return labeled


# ===== Model Loading =====

def load_model_bf16(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (BF16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn_impl in ["flash_attention_2", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"  Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:120]}, trying next...")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    info = get_model_info(model, model_name)
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, info


def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core Analysis =====

def collect_hidden_states_labeled(model, tokenizer, input_device, labeled_prompts, n_layers, model_name):
    """Collect hidden states with labels for probing."""
    import torch
    
    n_total = n_layers + 1
    # Store hidden states per layer
    hidden_per_layer = {l: [] for l in range(n_total)}
    labels_list = []
    
    log_time(f"Collecting hidden states for {len(labeled_prompts)} labeled prompts...")
    t_start = time.time()
    
    for i, (prompt, label) in enumerate(labeled_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        last_pos = int(attn_mask.sum().item()) - 1
        for l in range(n_total):
            h = out.hidden_states[l][0, last_pos, :].detach().float().cpu().numpy()
            hidden_per_layer[l].append(h)
        
        labels_list.append(label)
        
        del out
        torch.cuda.empty_cache()
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            log_time(f"  {i+1}/{len(labeled_prompts)} done ({rate:.1f}/s)")
        
        if (i + 1) % 100 == 0:
            gc.collect()
    
    # Convert to arrays
    for l in range(n_total):
        hidden_per_layer[l] = np.array(hidden_per_layer[l])
    
    return hidden_per_layer, labels_list


def compute_wu_subspaces(W_U, n_components=200):
    """Compute W_U visible and invisible subspaces.
    
    Returns:
        V_vis: [d_model, n_components] — W_U visible basis
        V_inv: [d_model, d_model - n_components] — W_U invisible basis
    """
    from sklearn.utils.extmath import randomized_svd
    
    log_time(f"Computing W_U subspaces (n_components={n_components})...")
    U, S, Vt = randomized_svd(W_U.astype(np.float32), n_components=n_components, random_state=42)
    
    # Vt: [n_components, d_model] — right singular vectors = W_U row space basis
    V_vis = Vt.T  # [d_model, n_components]
    
    # Compute orthogonal complement for V_inv
    d_model = W_U.shape[1]
    log_time(f"  Computing orthogonal complement ({d_model - n_components} dims)...")
    
    # Use QR decomposition to find orthogonal complement
    # Start with V_vis and extend to full orthonormal basis
    Q, R = np.linalg.qr(np.hstack([V_vis, np.eye(d_model, d_model - n_components)]))
    V_inv = Q[:, n_components:]  # [d_model, d_model - n_components]
    
    # Verify orthogonality
    overlap = np.sum(V_vis.T @ V_inv)
    log_time(f"  V_vis shape: {V_vis.shape}, V_inv shape: {V_inv.shape}, overlap: {overlap:.2e}")
    
    return V_vis, V_inv


def project_to_subspace(H, V):
    """Project hidden states H to subspace V.
    
    H: [N, d_model]
    V: [d_model, k]
    Returns: [N, k]
    """
    return H @ V


def probe_feature(H_projected, labels, feature_name, feature_values):
    """Probe a semantic feature using logistic regression.
    
    H_projected: [N, k] — hidden states in subspace
    labels: list of label dicts
    feature_name: e.g. "number"
    feature_values: e.g. ["singular", "plural"]
    
    Returns: dict with accuracy, n_samples, etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # Filter to samples with this label
    indices = [i for i, l in enumerate(labels) if l.get(feature_name) in feature_values]
    if len(indices) < 10:
        return {"accuracy": 0.0, "n_samples": 0, "feature": feature_name, "note": "too few samples"}
    
    H_sub = H_projected[indices]
    y = np.array([feature_values.index(labels[i][feature_name]) for i in indices])
    
    # Check class balance
    n_classes = len(set(y))
    if n_classes < 2:
        return {"accuracy": 0.0, "n_samples": len(indices), "feature": feature_name, "note": "single class"}
    
    # Cross-validated accuracy
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    try:
        scores = cross_val_score(clf, H_sub, y, cv=min(5, min(np.bincount(y))), scoring='accuracy')
        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
    except Exception as e:
        # Fallback: train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(H_sub, y, test_size=0.2, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        mean_acc = float(clf.score(X_test, y_test))
        std_acc = 0.0
    
    return {
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "n_samples": len(indices),
        "feature": feature_name,
        "feature_values": feature_values,
        "class_balance": [int(c) for c in np.bincount(y)],
    }


def run_probing_analysis(hidden_per_layer, labels_list, V_vis, V_inv, n_total_layers, model_name):
    """Run probing in both W_U visible and invisible subspaces."""
    
    log_time(f"\n{'='*60}")
    log_time(f"Dark Matter Probing — {model_name}")
    log_time(f"{'='*60}")
    
    features = {
        "number": ["singular", "plural"],
        "animacy": ["animate", "inanimate"],
        "tense": ["past", "present", "future"],
        "category": ["animal", "tool", "clothing", "body"],
    }
    
    results = {}
    
    # Sample layers: early, middle, late
    n_layers = n_total_layers - 1  # exclude embedding
    sample_layers = [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]
    # Add more layers around the middle
    mid = n_layers // 2
    for l in [mid-2, mid-1, mid, mid+1, mid+2]:
        if 1 <= l <= n_layers and l not in sample_layers:
            sample_layers.append(l)
    sample_layers = sorted(set(sample_layers))
    
    log_time(f"Probing layers: {sample_layers}")
    
    for l in sample_layers:
        H = hidden_per_layer[l]  # [N, d_model]
        
        # Project to W_U visible subspace
        H_vis = H @ V_vis   # [N, n_components]
        # Project to W_U invisible subspace  
        H_inv = H @ V_inv   # [N, d_model - n_components]
        # Also test full space
        H_full = H
        
        layer_results = {}
        for feat_name, feat_values in features.items():
            # Probe in W_U visible space
            res_vis = probe_feature(H_vis, labels_list, feat_name, feat_values)
            # Probe in W_U invisible space
            res_inv = probe_feature(H_inv, labels_list, feat_name, feat_values)
            # Probe in full space
            res_full = probe_feature(H_full, labels_list, feat_name, feat_values)
            
            layer_results[feat_name] = {
                "visible_accuracy": res_vis["accuracy"],
                "visible_std": res_vis.get("accuracy_std", 0),
                "invisible_accuracy": res_inv["accuracy"],
                "invisible_std": res_inv.get("accuracy_std", 0),
                "full_accuracy": res_full["accuracy"],
                "full_std": res_full.get("accuracy_std", 0),
                "n_samples": res_vis["n_samples"],
                "advantage": res_inv["accuracy"] - res_vis["accuracy"],
                "n_vis_dims": V_vis.shape[1],
                "n_inv_dims": V_inv.shape[1],
            }
            
            adv = res_inv["accuracy"] - res_vis["accuracy"]
            direction = "INV>" if adv > 0.05 else ("VIS>" if adv < -0.05 else "SAME")
            log_time(f"  L{l} {feat_name:>15}: vis={res_vis['accuracy']:.3f}, "
                     f"inv={res_inv['accuracy']:.3f}, full={res_full['accuracy']:.3f}, "
                     f"adv={adv:+.3f} [{direction}]")
        
        results[f"L{l}"] = layer_results
    
    return results


# ===== Main =====

def run_model(model_name):
    global _log_file
    _log_file = RESULT_DIR / f"{model_name}_log.txt"
    
    import torch
    from model_utils import get_W_U, release_model
    
    log_time(f"\n{'='*70}")
    log_time(f"Phase 269: Dark Matter Probing — {model_name}")
    log_time(f"{'='*70}")
    
    # Generate labeled prompts
    labeled = generate_labeled_prompts()
    log_time(f"Generated {len(labeled)} labeled prompts")
    
    # Count labels per feature
    for feat in ["number", "animacy", "tense", "category"]:
        vals = [l[1].get(feat, "") for l in labeled if l[1].get(feat, "")]
        from collections import Counter
        counts = Counter(vals)
        log_time(f"  {feat}: {dict(counts)}")
    
    # Load model
    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model
    n_total = n_layers + 1
    
    # Step 1: Collect hidden states
    t0 = time.time()
    hidden_per_layer, labels_list = collect_hidden_states_labeled(
        model, tokenizer, input_device, labeled, n_layers, model_name)
    t_collect = time.time() - t0
    log_time(f"Hidden state collection: {t_collect:.1f}s")
    
    # Step 2: Get W_U
    W_U = get_W_U(model, model_name)
    log_time(f"W_U shape={W_U.shape}")
    
    # Step 3: Compute W_U subspaces
    n_components = min(200, d_model - 1)
    V_vis, V_inv = compute_wu_subspaces(W_U, n_components=n_components)
    
    # Step 4: Verify Phase 268 alignment
    for l in [1, n_layers//2, n_layers]:
        H = hidden_per_layer[l]
        H_centered = H - H.mean(axis=0, keepdims=True)
        total_var = np.sum(H_centered ** 2)
        vis_var = np.sum((H_centered @ V_vis) ** 2)
        inv_var = np.sum((H_centered @ V_inv) ** 2)
        alignment = vis_var / total_var if total_var > 0 else 0
        log_time(f"  L{l} alignment check: vis={alignment:.4f}, inv={1-alignment:.4f}")
    
    # Step 5: Probing analysis
    probing_results = run_probing_analysis(
        hidden_per_layer, labels_list, V_vis, V_inv, n_total, model_name)
    
    # Step 6: Summary
    log_time(f"\n{'='*70}")
    log_time(f"=== SUMMARY: Dark Matter Probing — {model_name} ===")
    log_time(f"{'='*70}")
    
    log_time(f"{'Layer':>6} {'Feature':>15} {'Vis_Acc':>8} {'Inv_Acc':>8} {'Full_Acc':>8} {'Advantage':>10} {'Winner':>8}")
    log_time("-" * 75)
    
    dark_matter_wins = 0
    visible_wins = 0
    
    for layer_key in sorted(probing_results.keys(), key=lambda x: int(x[1:])):
        for feat_name, res in probing_results[layer_key].items():
            adv = res["advantage"]
            winner = "DARK" if adv > 0.05 else ("VIS" if adv < -0.05 else "SAME")
            if adv > 0.05:
                dark_matter_wins += 1
            elif adv < -0.05:
                visible_wins += 1
            
            log_time(f"{layer_key:>6} {feat_name:>15} {res['visible_accuracy']:>8.3f} "
                     f"{res['invisible_accuracy']:>8.3f} {res['full_accuracy']:>8.3f} "
                     f"{adv:>+10.3f} {winner:>8}")
    
    log_time(f"\nDark matter wins: {dark_matter_wins}, Visible wins: {visible_wins}")
    
    if dark_matter_wins > visible_wins:
        conclusion = "DARK MATTER DOMINANT: Semantic features are encoded primarily in W_U-invisible space"
    elif visible_wins > dark_matter_wins:
        conclusion = "VISIBLE DOMINANT: Semantic features are encoded primarily in W_U-visible space"
    else:
        conclusion = "MIXED: Semantic features distributed across both spaces"
    
    log_time(f"\nCONCLUSION: {conclusion}")
    
    # Save results
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_components": n_components,
        "conclusion": conclusion,
        "dark_matter_wins": dark_matter_wins,
        "visible_wins": visible_wins,
        "probing_results": probing_results,
        "timing": {"collect_s": round(t_collect, 1)},
    }
    
    result_file = RESULT_DIR / f"{model_name}_dark_matter_probing.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log_time(f"Results saved: {result_file}")
    
    # Release model
    release_model(model)
    del hidden_per_layer
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released")
    
    _log_file = None
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase269_dark_matter_probing.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1]
    if model_name not in ("qwen3", "glm4", "deepseek7b"):
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    result = run_model(model_name)
    log_time(f"\nPhase 269 complete for {model_name}")
