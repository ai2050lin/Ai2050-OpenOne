"""Phase 149c: DS7B精简实验 — 仅Exp2(三层次方向分解)
用bfloat16+CPU offload绕过8bit加载卡住问题
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, numpy as np, torch, time
from datetime import datetime
from pathlib import Path
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, get_W_U, release_model

OUTPUT_DIR = Path("tests/glm5_temp")
TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
]
N_SENTENCES = 15
EPSILON = 2.0


def get_row_null_bases(W_U, n_components=200):
    d = W_U.shape[1]
    k = min(n_components, d - 2)
    np.random.seed(42)
    n_samples = k + 10
    Omega = np.random.randn(W_U.shape[0], n_samples).astype(np.float32)
    batch_size = 50000
    Y = np.zeros((d, n_samples), dtype=np.float32)
    for i in range(0, W_U.shape[0], batch_size):
        end = min(i + batch_size, W_U.shape[0])
        Y += W_U[i:end].T.astype(np.float32) @ Omega[i:end]
    del Omega
    Q, R = np.linalg.qr(Y)
    del Y, R
    effective_k = min(k, Q.shape[1])
    return Q[:, :effective_k].T, effective_k


def project_to_null(vec, row_basis):
    return vec - row_basis.T @ (row_basis @ vec)


def project_to_row(vec, row_basis):
    return row_basis.T @ (row_basis @ vec)


def compute_row_energy(delta, row_basis):
    delta_norm_sq = np.sum(delta ** 2)
    if delta_norm_sq < 1e-16:
        return 0.0, 1.0
    row_coeffs = row_basis @ delta
    row_energy = np.sum(row_coeffs ** 2) / delta_norm_sq
    return float(row_energy), float(1.0 - row_energy)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 149c: Three-Level Direction (simplified) for {model_name}", flush=True)

    # 加载模型: 使用8bit但分步加载
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = MODEL_CONFIGS[model_name]

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (8bit, this may take a while)...", flush=True)
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['path'], quantization_config=bnb_config, device_map='auto',
        trust_remote_code=True, local_files_only=True, attn_implementation='eager')
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded in {time.time()-t0:.1f}s, device={device}", flush=True)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    d_model = info.d_model
    n_layers = info.n_layers
    print(f"Info: {info.model_class}, {info.n_layers}L, d={info.d_model}", flush=True)

    # W_U
    print("Loading W_U...", flush=True)
    t0 = time.time()
    W_U = get_W_U(model, model_name)
    print(f"W_U: shape={W_U.shape}, time={time.time()-t0:.1f}s", flush=True)
    row_basis, k = get_row_null_bases(W_U)
    print(f"Row basis: {k} components", flush=True)

    # Exp 2: 三层次方向分解
    sample_layers = sorted(set([0, 1] + list(range(1, n_layers, max(1, n_layers // 8))) + [n_layers - 1, n_layers]))
    sample_layers = [l for l in sample_layers if l <= n_layers]

    results = {}

    for sent_idx in range(N_SENTENCES):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        clean_hs = out_clean.hidden_states

        for dir_type in ["null", "row"]:
            np.random.seed(sent_idx * 100 + (0 if dir_type == "null" else 50))
            raw_vec = np.random.randn(d_model)
            delta = project_to_null(raw_vec, row_basis) if dir_type == "null" else project_to_row(raw_vec, row_basis)
            norm = np.linalg.norm(delta)
            if norm < 1e-8:
                continue
            delta = delta / norm * EPSILON

            delta_null_original = project_to_null(delta, row_basis)
            delta_row_original = project_to_row(delta, row_basis)
            delta_null_norm = np.linalg.norm(delta_null_original)
            delta_row_norm = np.linalg.norm(delta_row_original)

            # 注入
            def make_inject_hook(dt):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_new = h.clone()
                    h_new[0, -1, :] += torch.tensor(dt, dtype=h.dtype, device=h.device)
                    return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
                return hook

            hooks = [layers[0].register_forward_hook(make_inject_hook(delta))]
            try:
                with torch.no_grad():
                    out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            finally:
                for h in hooks:
                    h.remove()

            layer_data = []
            for li in sample_layers:
                pv = out_perturbed.hidden_states[li][0, -1, :].float().cpu().numpy()
                cv = clean_hs[li][0, -1, :].float().cpu().numpy()
                dp = pv - cv
                dp_norm = np.linalg.norm(dp)

                if dp_norm < 1e-10:
                    layer_data.append({'layer': li, 'global_cos': 0, 'null_ratio': 1, 'null_int_cos': 0, 'row_int_cos': 0, 'delta_norm': 0})
                    continue

                global_cos = float(np.dot(delta, dp) / (np.linalg.norm(delta) * dp_norm))
                row_e, null_r = compute_row_energy(dp, row_basis)
                dp_null = project_to_null(dp, row_basis)
                dp_row = project_to_row(dp, row_basis)

                null_int_cos = 0.0
                if delta_null_norm > 1e-8 and np.linalg.norm(dp_null) > 1e-8:
                    null_int_cos = float(np.dot(delta_null_original, dp_null) / (delta_null_norm * np.linalg.norm(dp_null)))
                row_int_cos = 0.0
                if delta_row_norm > 1e-8 and np.linalg.norm(dp_row) > 1e-8:
                    row_int_cos = float(np.dot(delta_row_original, dp_row) / (delta_row_norm * np.linalg.norm(dp_row)))

                layer_data.append({
                    'layer': li, 'global_cos': global_cos, 'null_ratio': null_r,
                    'null_int_cos': null_int_cos, 'row_int_cos': row_int_cos,
                    'delta_norm': float(dp_norm),
                })

            key = f"sent{sent_idx}_{dir_type}"
            results[key] = layer_data
            print(f"  Sent{sent_idx} ({dir_type}): L1 gc={layer_data[1]['global_cos']:.4f} L5 gc={layer_data[min(5,len(layer_data)-1)]['global_cos']:.4f}", flush=True)

    # 汇总
    print("\n=== Summary ===", flush=True)
    for dir_type in ["null", "row"]:
        layer_avgs = {}
        for key, ld in results.items():
            if f"_{dir_type}" not in key:
                continue
            for entry in ld:
                li = entry['layer']
                if li not in layer_avgs:
                    layer_avgs[li] = {'gc': [], 'nr': [], 'nic': [], 'ric': []}
                layer_avgs[li]['gc'].append(entry['global_cos'])
                layer_avgs[li]['nr'].append(entry['null_ratio'])
                layer_avgs[li]['nic'].append(entry['null_int_cos'])
                layer_avgs[li]['ric'].append(entry['row_int_cos'])

        print(f"\n--- {dir_type.upper()}-space input ---")
        print(f"  Layer  global_cos  null_ratio  null_int_cos  row_int_cos")
        for li in sorted(layer_avgs.keys()):
            a = layer_avgs[li]
            print(f"  L{li:>4d}  {np.mean(a['gc']):>10.4f}  {np.mean(a['nr']):>10.4f}  {np.mean(a['nic']):>13.4f}  {np.mean(a['ric']):>12.4f}")

    # 保存
    all_results = {
        "phase": "149c", "model": model_name, "timestamp": timestamp,
        "exp2_three_level_direction": {k: {'layers': [e['layer'] for e in v], 'global_cos': [e['global_cos'] for e in v], 'null_ratio': [e['null_ratio'] for e in v], 'null_int_cos': [e['null_int_cos'] for e in v], 'row_int_cos': [e['row_int_cos'] for e in v]} for k, v in results.items()}
    }
    result_file = OUTPUT_DIR / f"phase149c_{model_name}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {result_file}", flush=True)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
