"""
Phase 367: Δh Orthogonality Baseline — Is angle≈89° a high-dimensional artifact?
================================================================================

CRITICAL VALIDATION: Phase 365 found that Δh at last_token position is nearly
orthogonal to the W_U direction (angle ≈ 89°) across all layers and models.

But in a high-dimensional space (d_model = 2560-4096), a random vector is
almost always nearly orthogonal to any fixed direction. The expected angle
between a random vector and a fixed direction in d-dimensional space is:

  E[angle] = arccos(0) ≈ 90°  (concentrates around 90° for large d)

So angle ≈ 89° might be trivially expected and NOT evidence of a special mechanism.

This script tests:
1. Generate random Gaussian vectors in d_model dimensions
2. Compute their angle with the W_U direction for each test pair
3. Compare with the actual Δh angles from Phase 365
4. If Δh angles are significantly different from random (closer to 0° or 180°),
   then the orthogonality is meaningful
5. If Δh angles match random angles, the "hidden channel" interpretation is wrong

Also tests:
- Whether Δh's cosine similarity across layers (0.83-0.96) is significantly
  higher than random (which would be ~0)
- Whether Δh's norm growth pattern matches what we'd expect from random walk

No model inference needed — uses Phase 365 results directly.
"""

import sys, os, json, gc
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {"n_layers": 36, "d_model": 2560},
    "glm4": {"n_layers": 40, "d_model": 4096},
    "deepseek7b": {"n_layers": 28, "d_model": 3584},
}


def analyze_orthogonality_baseline(model_name):
    log(f"\n  === {model_name} ===")
    cfg = MODEL_CONFIGS[model_name]
    d_model = cfg["d_model"]
    n_layers = cfg["n_layers"]

    # Load Phase 365 results
    p365_path = f"results/phase365_dual_component/{model_name}_phase365.json"
    if not os.path.exists(p365_path):
        log(f"  Phase 365 data not found: {p365_path}")
        return None

    with open(p365_path, "r") as f:
        p365 = json.load(f)

    # Load Phase 364 results for W_U direction info
    p364_path = f"results/phase364_layer_role/{model_name}_phase364.json"
    # We don't actually need W_U — we just need the angle statistics

    rt_last = p365.get("rigid_transfer_last_summary", {})
    rt_obj = p365.get("rigid_transfer_obj_summary", {})

    # ===== Test 1: Random angle distribution =====
    log(f"  Test 1: Random angle distribution in d={d_model}")
    np.random.seed(42)
    n_random = 100000
    # Random unit vector vs fixed direction
    random_angles = []
    for _ in range(n_random):
        v = np.random.randn(d_model)
        v = v / np.linalg.norm(v)
        # Angle with e_1 (arbitrary fixed direction)
        cos_angle = abs(v[0])  # dot with e_1
        angle = np.arccos(np.clip(cos_angle, 0, 1)) * 180 / np.pi
        random_angles.append(angle)

    random_angles = np.array(random_angles)
    log(f"    Random angles: mean={np.mean(random_angles):.2f}°, "
        f"std={np.std(random_angles):.2f}°, "
        f"median={np.median(random_angles):.2f}°")
    log(f"    P(angle < 89°) = {np.mean(random_angles < 89):.4f}")
    log(f"    P(angle < 85°) = {np.mean(random_angles < 85):.4f}")
    log(f"    P(angle < 80°) = {np.mean(random_angles < 80):.4f}")

    # The expected angle for random vectors in d dimensions:
    # E[|cos(angle)|] ≈ 1/sqrt(d)
    # So E[angle] ≈ arccos(1/sqrt(d))
    expected_angle = np.arccos(1.0 / np.sqrt(d_model)) * 180 / np.pi
    log(f"    Theoretical E[angle] for random vectors: {expected_angle:.2f}°")

    # ===== Test 2: Compare actual Δh angles with random =====
    log(f"  Test 2: Actual Δh angles vs random baseline")
    actual_angles = []
    actual_cos_sims = []
    actual_norms = []

    for li_str, data in rt_last.items():
        angle = data.get("angle_wu_mean")
        cos_sim = data.get("cos_sim_mean")
        norm = data.get("delta_norm_mean")
        if angle is not None:
            actual_angles.append(angle)
        if cos_sim is not None:
            actual_cos_sims.append(cos_sim)
        if norm is not None:
            actual_norms.append(norm)

    if len(actual_angles) > 0:
        actual_angles = np.array(actual_angles)
        log(f"    Actual angles: mean={np.mean(actual_angles):.2f}°, "
            f"std={np.std(actual_angles):.2f}°, "
            f"min={np.min(actual_angles):.2f}°, max={np.max(actual_angles):.2f}°")
        
        # Z-test: is actual angle significantly different from random?
        random_mean = np.mean(random_angles)
        random_std = np.std(random_angles)
        n_actual = len(actual_angles)
        z_score = (np.mean(actual_angles) - random_mean) / (random_std / np.sqrt(n_actual))
        log(f"    Z-score (actual vs random): {z_score:.2f}")
        log(f"    → {'SIGNIFICANTLY DIFFERENT' if abs(z_score) > 2 else 'NOT significantly different'}")
    else:
        z_score = 0.0

    if len(actual_cos_sims) > 0:
        actual_cos_sims = np.array(actual_cos_sims)
        # Random cosine similarity in d dimensions: E[|cos|] ≈ 1/sqrt(d)
        expected_cos = 1.0 / np.sqrt(d_model)
        log(f"    Actual cos_sim(prev): mean={np.mean(actual_cos_sims):.4f}, "
            f"expected random={expected_cos:.4f}")
        log(f"    → {'SIGNIFICANTLY higher than random' if np.mean(actual_cos_sims) > 5*expected_cos else 'Within random range'}")
    else:
        expected_cos = 1.0 / np.sqrt(d_model)

    # ===== Test 3: Norm growth pattern =====
    log(f"  Test 3: Norm growth pattern")
    if len(actual_norms) > 0:
        actual_norms_arr = np.array(actual_norms)
        log(f"    ||Δh|| range: {actual_norms_arr[0]:.2f} → {actual_norms_arr[-1]:.2f} "
            f"(ratio={actual_norms_arr[-1]/max(actual_norms_arr[0], 0.01):.0f}x)")
        
        # For a random walk in d_model dimensions, ||Δh|| grows as sqrt(n_steps)
        # For a directed process, ||Δh|| grows linearly
        # Check growth rate
        if len(actual_norms_arr) > 5:
            # Fit log(norm) vs log(layer) to estimate growth exponent
            layers_arr = np.arange(1, len(actual_norms_arr) + 1, dtype=float)
            valid = actual_norms_arr > 0
            if np.sum(valid) > 3:
                log_norms = np.log(actual_norms_arr[valid])
                log_layers = np.log(layers_arr[valid])
                # Linear fit: log(norm) = alpha * log(layer) + c
                alpha = np.polyfit(log_layers, log_norms, 1)[0]
                log(f"    Growth exponent: α={alpha:.2f} (1.0=linear, 0.5=random walk)")

    # ===== Test 4: Layer-wise angle comparison =====
    log(f"  Test 4: Layer-wise angle comparison (early vs late)")
    early_angles = []
    late_angles = []
    for li_str, data in rt_last.items():
        li = int(li_str)
        angle = data.get("angle_wu_mean")
        if angle is not None:
            if li <= n_layers // 3:
                early_angles.append(angle)
            elif li >= 2 * n_layers // 3:
                late_angles.append(angle)

    if len(early_angles) > 0 and len(late_angles) > 0:
        log(f"    Early layers (L0-L{n_layers//3}): mean angle={np.mean(early_angles):.2f}°")
        log(f"    Late layers (L{2*n_layers//3}-L{n_layers-1}): mean angle={np.mean(late_angles):.2f}°")
        diff = np.mean(early_angles) - np.mean(late_angles)
        log(f"    Difference: {diff:+.2f}° ({'early more orthogonal' if diff > 0 else 'late more orthogonal'})")

    # ===== Test 5: Object position vs last_token position comparison =====
    log(f"  Test 5: Object position Δh angles")
    obj_angles = []
    for li_str, data in rt_obj.items():
        angle = data.get("angle_wu_mean")
        if angle is not None:
            obj_angles.append(angle)
    
    if len(obj_angles) > 0:
        obj_angles = np.array(obj_angles)
        log(f"    Object position angles: mean={np.mean(obj_angles):.2f}°")
        if len(actual_angles) > 0:
            log(f"    Last_token angles: mean={np.mean(actual_angles):.2f}°")
            log(f"    Difference: {np.mean(obj_angles) - np.mean(actual_angles):+.2f}°")

    return {
        "model": model_name,
        "d_model": d_model,
        "random_angle_mean": float(np.mean(random_angles)),
        "random_angle_std": float(np.std(random_angles)),
        "actual_angle_mean": float(np.mean(actual_angles)) if len(actual_angles) > 0 else None,
        "actual_angle_std": float(np.std(actual_angles)) if len(actual_angles) > 0 else None,
        "z_score": float(z_score) if len(actual_angles) > 0 else None,
        "actual_cos_sim_mean": float(np.mean(actual_cos_sims)) if len(actual_cos_sims) > 0 else None,
        "expected_random_cos_sim": float(expected_cos),
        "growth_exponent": float(alpha) if len(actual_norms_arr) > 5 else None,
    }


def main():
    log("=" * 60)
    log("Phase 367: Δh Orthogonality Baseline Analysis")
    log("=" * 60)

    results = {}
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        results[model_name] = analyze_orthogonality_baseline(model_name)

    # ===== Summary =====
    log("\n" + "=" * 60)
    log("SUMMARY: Is angle≈89° a high-dimensional artifact?")
    log("=" * 60)

    for model_name, r in results.items():
        if r is None:
            continue
        log(f"\n  {model_name} (d={r['d_model']}):")
        log(f"    Random angle: {r['random_angle_mean']:.2f}° ± {r['random_angle_std']:.2f}°")
        if r['actual_angle_mean'] is not None:
            log(f"    Actual angle: {r['actual_angle_mean']:.2f}° ± {r['actual_angle_std']:.2f}°")
            log(f"    Z-score: {r['z_score']:.2f}")
            log(f"    Cos_sim(prev): actual={r['actual_cos_sim_mean']:.4f} vs random≈{r['expected_random_cos_sim']:.4f}")
        if r['growth_exponent'] is not None:
            log(f"    ||Δh|| growth exponent: {r['growth_exponent']:.2f}")

    # Save results
    os.makedirs("results/phase367_orthogonality_baseline", exist_ok=True)
    out_path = "results/phase367_orthogonality_baseline/summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
