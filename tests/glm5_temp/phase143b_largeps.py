"""
Phase 143b: Quick Large-Epsilon Verification
Tests whether the cos≈0 result in Exp 1 is due to numerical noise (ε too small)
or genuine nonlinearity.

Uses ε = 0.5 and ε = 2.0 (vs original ε = 0.005) with just 3 sentences.
If cos increases significantly with larger ε → noise was the issue.
If cos stays ≈ 0 → genuine piecewise dynamics.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, time, json, numpy as np, torch
from datetime import datetime
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, get_sample_layers)

SENTENCES = [
    ("The cat sat on the mat", "The cat did not sit on the mat"),
    ("A dog runs in the park", "A dog does not run in the park"),
    ("The student passed the exam", "The student did not pass the exam"),
]

EPSILONS = [0.005, 0.05, 0.5, 2.0]
TEST_LAYERS = [0, 12, 24]  # For Qwen3; will adapt for other models
N_PROBES = 5

def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")

def capture_hs(model, tokenizer, device, sentence, n_layers):
    input_device = get_device_for_input(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = outputs.hidden_states
    logits = outputs.logits[0, -1].float().cpu().numpy()
    hidden = {l: hs[l][0, -1, :].float().cpu().numpy() for l in range(len(hs))}
    return hidden, logits

def inject_and_capture(model, tokenizer, device, sentence, inject_layer, direction_np, epsilon, n_layers):
    input_device = get_device_for_input(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    layers = get_layers(model)
    injected = [False]
    direction_tensor = torch.tensor(direction_np, dtype=torch.float32, device=input_device)

    def inject_hook(module, input, output):
        if not injected[0]:
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += epsilon * direction_tensor.to(h.dtype)
                injected[0] = True
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += epsilon * direction_tensor.to(h.dtype)
                injected[0] = True
                return h
        return output

    hook = layers[inject_layer].register_forward_hook(inject_hook)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hook.remove()
    hs = outputs.hidden_states
    logits = outputs.logits[0, -1].float().cpu().numpy()
    hidden = {l: hs[l][0, -1, :].float().cpu().numpy() for l in range(len(hs))}
    return hidden, logits

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"Phase 143b: Large-Epsilon Verification — {model_name}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, L={info.n_layers}, d={info.d_model}")

    # Adapt test layers
    n_l = info.n_layers
    test_layers = [0, n_l // 3, 2 * n_l // 3]

    np.random.seed(42)
    probes = [np.random.randn(info.d_model) for _ in range(N_PROBES)]
    probes = [v / np.linalg.norm(v) for v in probes]

    results = {}

    for eps in EPSILONS:
        print(f"\n--- ε = {eps} ---")
        all_cos = []

        for s_base, s_neg in SENTENCES:
            h_base, _ = capture_hs(model, tokenizer, device, s_base, n_l)
            h_neg, _ = capture_hs(model, tokenizer, device, s_neg, n_l)

            for inject_l in test_layers:
                response_l = inject_l + 2
                if response_l >= n_l + 1:
                    response_l = inject_l + 1

                for probe in probes:
                    h_inj_base, _ = inject_and_capture(model, tokenizer, device, s_base, inject_l, probe, eps, n_l)
                    h_inj_neg, _ = inject_and_capture(model, tokenizer, device, s_neg, inject_l, probe, eps, n_l)

                    jv_base = (h_inj_base[response_l] - h_base[response_l]) / eps
                    jv_neg = (h_inj_neg[response_l] - h_neg[response_l]) / eps

                    n_b = np.linalg.norm(jv_base)
                    n_n = np.linalg.norm(jv_neg)

                    if n_b > 1e-8 and n_n > 1e-8:
                        cos = float(np.dot(jv_base, jv_neg) / (n_b * n_n))
                    else:
                        cos = 0.0
                    all_cos.append(cos)

        mean_cos = np.mean(all_cos)
        std_cos = np.std(all_cos)
        median_cos = np.median(all_cos)
        print(f"  ε={eps}: mean_cos={mean_cos:.4f}, std={std_cos:.4f}, median={median_cos:.4f}, n={len(all_cos)}")
        print(f"  ||Jv||_base: {n_b:.4f}, ||Jv||_neg: {n_n:.4f}")
        results[str(eps)] = {
            "mean_cos": float(mean_cos),
            "std_cos": float(std_cos),
            "median_cos": float(median_cos),
            "n_samples": len(all_cos),
        }

    release_model(model)

    # Verdict
    print(f"\n{'='*50}")
    print("VERDICT:")
    cos_small = results.get("0.005", {}).get("mean_cos", 0)
    cos_large = results.get("2.0", {}).get("mean_cos", 0)
    if cos_large > 0.5:
        print(f"  Small ε noise: cos(ε=0.005)={cos_small:.4f}, cos(ε=2.0)={cos_large:.4f}")
        print(f"  → Previous cos≈0 was due to numerical noise!")
        print(f"  → System may be locally linear at larger scale")
    elif cos_large > 0.2:
        print(f"  Moderate improvement: cos(ε=0.005)={cos_small:.4f}, cos(ε=2.0)={cos_large:.4f}")
        print(f"  → Partially piecewise with some local structure")
    else:
        print(f"  No improvement: cos(ε=0.005)={cos_small:.4f}, cos(ε=2.0)={cos_large:.4f}")
        print(f"  → GENUINELY piecewise: Jacobian varies strongly between points")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase143b_{model_name}_largeps_{timestamp}.json"
    with open(out_path, 'w') as f:
        json.dump({"model": model_name, "results": results, "timestamp": timestamp}, f, indent=2)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
