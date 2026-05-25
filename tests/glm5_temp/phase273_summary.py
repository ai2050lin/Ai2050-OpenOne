"""Phase 273 Summary: Cross-model comparison of causal path decomposition."""
import sys, json, os
import numpy as np
from pathlib import Path

def load_results(model_name):
    base = Path("results/phase273_causal_path")
    results = {}
    for exp in ["exp_a", "exp_b", "exp_c", "exp_d"]:
        path = base / f"{model_name}_{exp}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results[exp] = json.load(f)
    return results

def summarize_exp_a(data, model_name):
    """Summarize causal importance map."""
    meta = data.get("_meta", {})
    sample_layers = meta.get("sample_layers", [])

    # For each category, compute mean impact curve
    categories = {
        "fruits": ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "lemon", "lime"],
        "animals": ["dog", "cat", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow"],
        "vehicles": ["car", "bus", "train", "plane", "bike", "truck", "boat", "ship", "taxi", "van"],
    }

    cat_curves = {}
    for cat_name, words in categories.items():
        curves = []
        for w in words:
            if w in data and not w.startswith("_"):
                impacts = [abs(data[w].get(str(l), {}).get("mean_logit_change", 0)) for l in sample_layers]
                curves.append(impacts)
        if curves:
            cat_curves[cat_name] = {
                "mean": [float(np.mean([c[i] for c in curves])) for i in range(len(sample_layers))],
                "std": [float(np.std([c[i] for c in curves])) for i in range(len(sample_layers))],
            }

    # Find peak impact layer per category
    peak_layers = {}
    for cat_name, curve in cat_curves.items():
        mean_impacts = curve["mean"]
        peak_idx = int(np.argmax(mean_impacts))
        peak_layers[cat_name] = {
            "layer": sample_layers[peak_idx],
            "impact": mean_impacts[peak_idx],
        }

    # Print summary
    print(f"\n=== Exp A: Causal Importance Map — {model_name} ===")
    print(f"  Sample layers: {sample_layers}")
    for cat_name, curve in cat_curves.items():
        mean = curve["mean"]
        # Find top-3 most impactful layers
        top3_idx = np.argsort(mean)[-3:][::-1]
        top3 = [(sample_layers[i], round(mean[i], 2)) for i in top3_idx]
        print(f"  {cat_name}: peak=L{peak_layers[cat_name]['layer']}({peak_layers[cat_name]['impact']:.2f}), "
              f"top3={top3}")

    # Compute shallow vs deep impact ratio
    shallow_layers = sample_layers[:len(sample_layers)//3]
    deep_layers = sample_layers[2*len(sample_layers)//3:]
    for cat_name, curve in cat_curves.items():
        mean = curve["mean"]
        shallow_mean = np.mean([mean[sample_layers.index(l)] for l in shallow_layers if l in sample_layers])
        deep_mean = np.mean([mean[sample_layers.index(l)] for l in deep_layers if l in sample_layers])
        ratio = shallow_mean / max(deep_mean, 0.01)
        print(f"  {cat_name}: shallow/deep ratio = {ratio:.2f}")

    return cat_curves, peak_layers


def summarize_exp_b(data, model_name):
    """Summarize MLP causal tracing."""
    meta = data.get("_meta", {})
    sample_layers = meta.get("sample_layers", [])
    test_words = meta.get("test_words", [])

    print(f"\n=== Exp B: MLP Causal Tracing — {model_name} ===")

    # Compute average MLP impact vs residual impact per layer
    for w in test_words[:3]:  # Show first 3 words
        if w in data and not w.startswith("_"):
            impacts = [abs(data[w].get(str(l), {}).get("mean_logit_change", 0)) for l in sample_layers]
            peak_idx = int(np.argmax(impacts))
            print(f"  {w}: MLP peak=L{sample_layers[peak_idx]}({impacts[peak_idx]:.2f}), "
                  f"mean={np.mean(impacts):.2f}")


def summarize_exp_c(data, model_name):
    """Summarize cross-concept causal overlap."""
    within = data.get("within_summary", {})
    between = data.get("between_summary", {})

    print(f"\n=== Exp C: Cross-concept Causal Overlap — {model_name} ===")
    for label, summary in [("Within", within), ("Between", between)]:
        print(f"  {label}: spearman_r={summary.get('spearman_r_mean', 0):.4f}, "
              f"cosine={summary.get('cosine_sim_mean', 0):.4f}, "
              f"jaccard={summary.get('top_k_jaccard_mean', 0):.4f}")

    # Compute delta
    delta_spearman = within.get('spearman_r_mean', 0) - between.get('spearman_r_mean', 0)
    delta_cosine = within.get('cosine_sim_mean', 0) - between.get('cosine_sim_mean', 0)
    delta_jaccard = within.get('top_k_jaccard_mean', 0) - between.get('top_k_jaccard_mean', 0)
    print(f"  Δ(Within-Between): spearman={delta_spearman:+.4f}, "
          f"cosine={delta_cosine:+.4f}, jaccard={delta_jaccard:+.4f}")

    return within, between


def summarize_exp_d(data, model_name):
    """Summarize causal divergence points."""
    pairs = data.get("pairs", [])

    within_pairs = [p for p in pairs if p.get("is_within", False)]
    between_pairs = [p for p in pairs if not p.get("is_within", False)]

    print(f"\n=== Exp D: Causal Divergence Point — {model_name} ===")

    for pair in pairs:
        label = "WITHIN" if pair.get("is_within", False) else "BETWEEN"
        pair_name = pair.get("pair", "")
        a_ctx = pair.get("max_diff_layer_a_context", "N/A")
        b_ctx = pair.get("max_diff_layer_b_context", "N/A")
        print(f"  [{label}] {pair_name}: max_diff_layer = L{a_ctx} (A ctx), L{b_ctx} (B ctx)")

    # Compute differential impact curves for within vs between
    meta = data.get("_meta", {})
    sample_layers = meta.get("sample_layers", [])

    within_diffs = []
    between_diffs = []
    for pair in pairs:
        a_ctx = pair.get("a_context_impacts", {})
        diffs = []
        for l in sample_layers:
            d = a_ctx.get(str(l), {}).get("differential", 0)
            diffs.append(abs(d))
        if pair.get("is_within", False):
            within_diffs.append(diffs)
        else:
            between_diffs.append(diffs)

    if within_diffs:
        mean_within = np.mean(within_diffs, axis=0)
        print(f"  Within mean max differential: {np.max(mean_within):.4f} at L{sample_layers[int(np.argmax(mean_within))]}")
    if between_diffs:
        mean_between = np.mean(between_diffs, axis=0)
        print(f"  Between mean max differential: {np.max(mean_between):.4f} at L{sample_layers[int(np.argmax(mean_between))]}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    results = load_results(model_name)

    if "exp_a" in results:
        summarize_exp_a(results["exp_a"], model_name)
    if "exp_b" in results:
        summarize_exp_b(results["exp_b"], model_name)
    if "exp_c" in results:
        summarize_exp_c(results["exp_c"], model_name)
    if "exp_d" in results:
        summarize_exp_d(results["exp_d"], model_name)

    # Cross-model comparison
    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON: Exp C Overlap")
    print("="*60)

    all_within = {}
    all_between = {}
    for mn in ["qwen3", "glm4", "deepseek7b"]:
        r = load_results(mn)
        if "exp_c" in r:
            w = r["exp_c"].get("within_summary", {})
            b = r["exp_c"].get("between_summary", {})
            all_within[mn] = w
            all_between[mn] = b

    print(f"\n{'Model':<12} {'Within Spearman':>16} {'Between Spearman':>17} {'Δ':>8} {'Within Jaccard':>15} {'Between Jaccard':>16} {'Δ':>8}")
    print("-"*95)
    for mn in all_within:
        ws = all_within[mn].get('spearman_r_mean', 0)
        bs = all_between[mn].get('spearman_r_mean', 0)
        wj = all_within[mn].get('top_k_jaccard_mean', 0)
        bj = all_between[mn].get('top_k_jaccard_mean', 0)
        print(f"{mn:<12} {ws:>16.4f} {bs:>17.4f} {ws-bs:>+8.4f} {wj:>15.4f} {bj:>16.4f} {wj-bj:>+8.4f}")


if __name__ == "__main__":
    main()
