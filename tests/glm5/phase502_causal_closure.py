"""
Phase 502 R1: Gain-Support 因果闭环 + 多目标词 + g⊙w_D投影干预
=================================================================
从Phase501几何对齐推进到因果验证:
  Exp1: v_cat ± direction 干预 → DCF变化 (因果语义方向)
  Exp2: 投影分解: v_cat∥g⊙w_D vs v_cat⊥g⊙w_D 对DCF的贡献
  Exp3: 多目标词paraphrase w_D 排除token artifact

加载策略: 参考 model_demo_bf16.py
  Qwen3:   bfloat16 + flash_attention_2 + 全GPU
  GLM4:    bfloat16 + flash_attention_2 + device_map="auto"
  DS7B:    bfloat16 + flash_attention_2 + device_map="auto"

Usage:
  python tests/glm5/phase502_causal_closure.py qwen3 1
  python tests/glm5/phase502_causal_closure.py glm4 1
  python tests/glm5/phase502_causal_closure.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, math, time, traceback
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)
from safetensors import safe_open

# ===== 类别(R1用10对象, R2用20对象) =====
CATEGORIES = {
    "fruit": {
        "objects_r1": ["apple","banana","orange","grape","pear","peach","mango","plum","cherry","lemon"],
        "objects_r2": ["apple","banana","orange","grape","pear","peach","mango","plum",
                      "cherry","lemon","apricot","kiwi","pineapple","melon","coconut",
                      "lime","fig","pomegranate","papaya","avocado"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
        "paraphrase_targets": ["fruit","food","produce","edible fruit"],
    },
    "clothing": {
        "objects_r1": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse","scarf","vest"],
        "objects_r2": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
                      "scarf","vest","hat","glove","sock","boot","belt",
                      "tie","jeans","shorts","hoodie","raincoat"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
        "paraphrase_targets": ["clothing","garment","attire","clothes"],
    },
    "emotion": {
        "objects_r1": ["joy","anger","fear","sadness","surprise","disgust","pride","shame","guilt","envy"],
        "objects_r2": ["joy","anger","fear","sadness","surprise","disgust","pride","shame",
                      "guilt","envy","hope","love","hate","boredom","anxiety",
                      "jealousy","gratitude","regret","curiosity","embarrassment"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
        "paraphrase_targets": ["emotion","feeling","sentiment","affect"],
    },
    "action": {
        "objects_r1": ["run","eat","build","throw","buy","learn","measure","communicate","swim","write"],
        "objects_r2": ["run","eat","build","throw","buy","learn","measure","communicate",
                      "swim","write","sing","draw","fly","climb","teach",
                      "drive","cook","dance","fight","sleep"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
        "paraphrase_targets": ["action","activity","deed","behavior"],
    },
    "animal": {
        "objects_r1": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake","rabbit","whale"],
        "objects_r2": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
                      "rabbit","whale","lion","bear","fox","wolf","deer",
                      "monkey","shark","frog","penguin","owl"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
        "paraphrase_targets": ["animal","creature","beast","critter"],
    },
}

ALL_CLASS_TOKENS = ["fruit","animal","clothing","emotion","action"]
OUTPUT_DIR = Path("results/glm5")
LOG_INTERVAL = 3  # 每处理3个对象输出一次日志


def load_model_bf16(model_name: str):
    """BF16 + flash_attention_2 + device_map=auto"""
    cfg = MODEL_CONFIGS[model_name]
    path = cfg["path"]
    print(f"[load] {model_name} bf16+flash+auto from {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_n = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_n = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[load] GPU:{gpu_n} CPU:{cpu_n} {gpu_mem:.1f}GB class={type(model).__name__}")
    else:
        print(f"[load] all-GPU {gpu_mem:.1f}GB class={type(model).__name__}")
    return model, tokenizer, device


def get_rmsnorm_weight(model):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        parts = attr.split('.')
        obj = model
        ok = True
        for p in parts:
            if hasattr(obj, p): obj = getattr(obj, p)
            else: ok = False; break
        if ok and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta':
                return w.float().cpu().numpy()
    return None


def get_token_ids(tokenizer, tokens):
    ids = []
    for t in tokens:
        tid = tokenizer.encode(t, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def extract_hidden(model, tokenizer, prompt, device):
    """Forward pass, return last-token hidden state"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, 'model'):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        else:
            outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy().astype(np.float64)


def dcf_from_hidden(h_np, W_U, target_ids, comp_ids):
    logits = h_np @ W_U.T
    t = np.mean([logits[i] for i in target_ids if i < len(logits)])
    c = np.mean([logits[i] for i in comp_ids if i < len(logits)])
    return float(t - c)


def unit_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def projection(v, onto):
    """projection of v onto 'onto' direction"""
    onto_u = unit_vec(onto)
    return np.dot(v, onto_u) * onto_u


# ==================== Main Experiment ====================
def run_phase502(model, tokenizer, model_name, device, round_str="1"):
    info = get_model_info(model, model_name)
    print(f"[info] layers={info.n_layers} d_model={info.d_model}")
    t_start = time.time()
    obj_key = f"objects_r{round_str}" if round_str in ("1","2") else "objects_r1"

    # --- Get W_U and g ---
    W_U = get_W_U(model, model_name)
    if W_U is None:
        print("ERROR: get_W_U returned None")
        return None
    W_U = W_U.astype(np.float64)
    g_vec = get_rmsnorm_weight(model)
    if g_vec is None:
        # Try alternate: load from safetensors directly
        print("WARNING: get_rmsnorm_weight returned None, trying direct safetensors load...")
        from safetensors import safe_open
        cfg = MODEL_CONFIGS[model_name]
        model_dir = Path(cfg["path"])
        sf_files = sorted(model_dir.glob("*.safetensors"))
        for sf in sf_files:
            with safe_open(str(sf), framework="pt") as f:
                for key in f.keys():
                    if 'norm' in key.lower() and 'weight' in key.lower() and not any(
                        x in key.lower() for x in ['layer', 'input', 'post']):
                        g_vec = f.get_tensor(key).float().cpu().numpy().astype(np.float64)
                        print(f"[g] loaded {key} from {sf.name} shape={g_vec.shape}")
                        break
                if g_vec is not None:
                    break
        if g_vec is None:
            print("ERROR: Cannot get RMSNorm gain weight")
            return None
    if g_vec is None:
        print("ERROR: Cannot get RMSNorm gain weight after all attempts")
        return None
    print(f"[W_U] {W_U.shape}  [g] {g_vec.shape}")

    # Token IDs
    cat_target_ids = {}
    cat_target_para_ids = {}
    all_comp_ids = []
    for cat, cfg in CATEGORIES.items():
        cat_target_ids[cat] = get_token_ids(tokenizer, cfg["target_tokens"])
        cat_target_para_ids[cat] = get_token_ids(tokenizer, cfg["paraphrase_targets"])
    for cat in CATEGORIES:
        others = [c for c in ALL_CLASS_TOKENS if c not in CATEGORIES[cat]["target_tokens"]]
        all_comp_ids = list(set(all_comp_ids + get_token_ids(tokenizer, others)))

    results = {}

    for cat_name, cfg in CATEGORIES.items():
        t_cat = time.time()
        print(f"\n--- {cat_name} ---")
        target_ids = cat_target_ids[cat_name]
        para_ids = cat_target_para_ids[cat_name]

        # w_D (single-target) and g⊙w_D
        wD_single_t = np.mean([W_U[i] for i in target_ids if i < len(W_U)], axis=0)
        wD_single_c = np.mean([W_U[i] for i in all_comp_ids if i < len(W_U)], axis=0)
        wD_single = wD_single_t - wD_single_c
        gwD_single = wD_single * g_vec

        # w_D (multi-paraphrase)
        wD_para_t = np.mean([W_U[i] for i in para_ids if i < len(W_U)], axis=0)
        wD_para = wD_para_t - wD_single_c

        # Per-object data
        v_cat_objs = []    # category direction per object
        D_base_list = []   # baseline DCF
        D_rich_list = []
        D_neutral_list = []

        for oi, obj in enumerate(cfg[obj_key]):
            prompt_rich = f"The {obj} {cfg['relation']}"
            prompt_neutral = f"The {obj} is a thing"
            h_rich = extract_hidden(model, tokenizer, prompt_rich, device)
            h_neutral = extract_hidden(model, tokenizer, prompt_neutral, device)
            v_cat_objs.append(h_rich - h_neutral)
            D_rich_list.append(dcf_from_hidden(h_rich, W_U, target_ids, all_comp_ids))
            D_neutral_list.append(dcf_from_hidden(h_neutral, W_U, target_ids, all_comp_ids))
            D_base_list.append(dcf_from_hidden(h_rich, W_U, target_ids, all_comp_ids))

            if (oi + 1) % LOG_INTERVAL == 0:
                print(f"  [{oi+1}/{len(cfg[obj_key])}] D_rich={D_rich_list[-1]:.2f} D_neutral={D_neutral_list[-1]:.2f}")

        v_cat_mean = np.mean(v_cat_objs, axis=0)
        v_cat_unit = unit_vec(v_cat_mean)
        gwD_unit = unit_vec(gwD_single)
        wD_unit = unit_vec(wD_single)

        # ===============================
        # Exp1: Intervention Effect
        # ===============================
        # For each object, do interventions on v_cat direction
        scales = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # ablate/reverse/restore/double
        intervention_results = {s: {"D_delta": [], "D_abs": []} for s in scales}
        intervention_results_base = {}  # baseline (no intervention)

        for oi, (h_base, v_obj) in enumerate(zip(
            [extract_hidden(model, tokenizer, f"The {obj} {cfg['relation']}", device) for obj in cfg[obj_key]],
            v_cat_objs
        )):
            D_base = dcf_from_hidden(h_base, W_U, target_ids, all_comp_ids)
            for s in scales:
                h_mod = h_base + s * v_cat_mean
                D_mod = dcf_from_hidden(h_mod, W_U, target_ids, all_comp_ids)
                intervention_results[s]["D_delta"].append(D_mod - D_base)
                intervention_results[s]["D_abs"].append(D_mod)

        # ===============================
        # Exp2: Projection Decomposition
        # ===============================
        # Decompose v_cat into ∥g⊙w_D and ⊥g⊙w_D
        v_para = projection(v_cat_mean, gwD_single)
        v_perp = v_cat_mean - v_para

        proj_results = {"para": {"D_delta": []}, "perp": {"D_delta": []}, "both": {"D_delta": []}}
        for oi, h_base in enumerate(
            [extract_hidden(model, tokenizer, f"The {obj} {cfg['relation']}", device) for obj in cfg[obj_key]]
        ):
            D_base = dcf_from_hidden(h_base, W_U, target_ids, all_comp_ids)
            # Inject ∥ component only
            h_para = h_base + 1.0 * v_para
            proj_results["para"]["D_delta"].append(dcf_from_hidden(h_para, W_U, target_ids, all_comp_ids) - D_base)
            # Inject ⊥ component only
            h_perp = h_base + 1.0 * v_perp
            proj_results["perp"]["D_delta"].append(dcf_from_hidden(h_perp, W_U, target_ids, all_comp_ids) - D_base)
            # Inject both (= full v_cat)
            h_both = h_base + 1.0 * v_cat_mean
            proj_results["both"]["D_delta"].append(dcf_from_hidden(h_both, W_U, target_ids, all_comp_ids) - D_base)

        # ===============================
        # Exp3: Multi-paraphrase comparison
        # ===============================
        cos_single_to_para = float(np.dot(wD_single, wD_para) / (np.linalg.norm(wD_single) * np.linalg.norm(wD_para)))
        cos_para_v = float(np.dot(v_cat_mean, wD_para) / (np.linalg.norm(v_cat_mean) * np.linalg.norm(wD_para)))
        cos_para_gwD = float(np.dot(v_cat_mean, wD_para * g_vec) / (np.linalg.norm(v_cat_mean) * np.linalg.norm(wD_para * g_vec)))

        results[cat_name] = {
            # Exp1: Intervention summary
            "intervention": {
                str(s): {
                    "mean_D_delta": round(float(np.mean(intervention_results[s]["D_delta"])), 4),
                    "std_D_delta": round(float(np.std(intervention_results[s]["D_delta"])), 4),
                } for s in scales
            },
            # Exp1 dose-response slope
            "dose_slope": round(
                float(np.polyfit(scales, [np.mean(intervention_results[s]["D_delta"]) for s in scales], 1)[0]), 4
            ),

            # Exp2: Projection decomposition
            "projection": {
                "para": {
                    "mean_D_delta": round(float(np.mean(proj_results["para"]["D_delta"])), 4),
                    "std_D_delta": round(float(np.std(proj_results["para"]["D_delta"])), 4),
                },
                "perp": {
                    "mean_D_delta": round(float(np.mean(proj_results["perp"]["D_delta"])), 4),
                    "std_D_delta": round(float(np.std(proj_results["perp"]["D_delta"])), 4),
                },
                "both": {
                    "mean_D_delta": round(float(np.mean(proj_results["both"]["D_delta"])), 4),
                    "std_D_delta": round(float(np.std(proj_results["both"]["D_delta"])), 4),
                },
                "para_frac": round(float(np.linalg.norm(v_para) / max(np.linalg.norm(v_cat_mean), 1e-10)), 4),
            },

            # Exp3: Multi-paraphrase
            "paraphrase": {
                "cos_single_to_para": round(cos_single_to_para, 6),
                "cos_para_v": round(cos_para_v, 6),
                "cos_para_gwD": round(cos_para_gwD, 6),
                "gain_boost": round(cos_para_gwD - cos_para_v, 6),
            },

            # Baseline stats
            "baseline": {
                "mean_D_rich": round(float(np.mean(D_rich_list)), 4),
                "mean_D_neutral": round(float(np.mean(D_neutral_list)), 4),
                "norm_v_cat": round(float(np.linalg.norm(v_cat_mean)), 2),
            },
        }

        dt = time.time() - t_cat
        print(f"  [{cat_name}] done in {dt:.0f}s  dose_slope={results[cat_name]['dose_slope']:.3f}"
              f"  para_frac={results[cat_name]['projection']['para_frac']:.3f}"
              f"  cos_para_to_single={cos_single_to_para:.4f}")

    # ===== Summary =====
    slopes = [results[c]["dose_slope"] for c in results]
    para_fracs = [results[c]["projection"]["para_frac"] for c in results]
    para_boosts = [results[c]["paraphrase"]["gain_boost"] for c in results]

    summary = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "total_time": round(time.time() - t_start, 1),
        "mean_dose_slope": round(np.mean(slopes), 4),
        "positive_slope_cats": sum(1 for s in slopes if s > 0.01),
        "mean_para_frac": round(np.mean(para_fracs), 4),
        "mean_para_gain_boost": round(np.mean(para_boosts), 6),
        "categories": results,
    }

    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase502_causal_closure.py <model> [round]")
        print("  model: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    rd = sys.argv[2] if len(sys.argv) > 2 else "1"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"Phase 502 R{rd}: Causal Closure — {model_name}")
    print("Exp1: v_cat ± direction intervention")
    print("Exp2: g⊙w_D ∥ vs ⊥ projection")
    print("Exp3: Multi-paraphrase w_D")
    print("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    print(f"[load] {time.time() - t0:.0f}s")

    try:
        results = run_phase502(model, tokenizer, model_name, device, rd)
        total = time.time() - t0

        if results is None:
            print("ERROR")
            return

        # Print
        print(f"\n{'=' * 70}")
        print(f"RESULTS — {model_name}")
        print(f"{'=' * 70}")
        print(f"  mean dose_slope:     {results['mean_dose_slope']:.3f}")
        print(f"  +slope cats (>0.01): {results['positive_slope_cats']}/{len(CATEGORIES)}")
        print(f"  mean para_frac:      {results['mean_para_frac']:.3f}")
        print(f"  total time:          {total:.0f}s")

        print(f"\n  Exp1: Intervention dose-response")
        for cat, r in results['categories'].items():
            print(f"    {cat:10s}: slope={r['dose_slope']:+.3f}  "
                  f"[-1: {r['intervention']['-1.0']['mean_D_delta']:+.2f}  "
                  f"+1: {r['intervention']['1.0']['mean_D_delta']:+.2f}  "
                  f"+2: {r['intervention']['2.0']['mean_D_delta']:+.2f}]")

        print(f"\n  Exp2: Projection ∥ vs ⊥ g⊙w_D")
        for cat, r in results['categories'].items():
            p = r['projection']
            print(f"    {cat:10s}: ∥={p['para']['mean_D_delta']:+.3f}  ⊥={p['perp']['mean_D_delta']:+.3f}  "
                  f"para_frac={p['para_frac']:.3f}")

        print(f"\n  Exp3: Multi-paraphrase gain boost")
        for cat, r in results['categories'].items():
            print(f"    {cat:10s}: gain_boost={r['paraphrase']['gain_boost']:+.4f}  "
                  f"cos_single->para={r['paraphrase']['cos_single_to_para']:.4f}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"phase502_{model_name}_r{rd}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

    finally:
        release_model(model)
        print("  Released.")


if __name__ == "__main__":
    main()
