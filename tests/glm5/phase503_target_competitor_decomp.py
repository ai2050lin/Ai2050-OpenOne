"""
Phase 503: Target/Competitor 因果分解 + matched-norm 投影 + 异常类机制
========================================================================
Exp1: 4种 v_cat 对照(rich-neutral, rich-wrong, matched-neutral, no-catword)
Exp2: target/competitor 分别追踪
Exp3: matched-norm parallel/orthogonal 干预
Exp4: clothing 多 competitor set 分析
Exp5: action 类型分解
Exp6: 随机方向对照

加载: bf16 + device_map="auto" + sdpa
Usage: python tests/glm5/phase503_target_competitor_decomp.py qwen3
       python tests/glm5/phase503_target_competitor_decomp.py glm4
       python tests/glm5/phase503_target_competitor_decomp.py deepseek7b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, time, traceback
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from model_utils import (get_model_info, release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("results/glm5")
LOG_INTERVAL = 5

# ===== 类别(R1用15对象) =====
CATEGORIES = {
    "fruit": {
        "objects": ["apple","banana","orange","grape","pear","peach","mango","plum",
                    "cherry","lemon","apricot","kiwi","pineapple","melon","coconut"],
        "relation": "is a type of fruit",
        "wrong_cat": "animal",
        "target": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt","dress","jacket","pants","coat","skirt","sweater",
                    "blouse","scarf","vest","hat","glove","sock","boot","belt"],
        "relation": "is a type of clothing",
        "wrong_cat": "food",
        "target": ["clothing"],
    },
    "emotion": {
        "objects": ["joy","anger","fear","sadness","surprise","disgust","pride",
                    "shame","guilt","envy","hope","love","hate","boredom","anxiety"],
        "relation": "is a type of emotion",
        "wrong_cat": "action",
        "target": ["emotion"],
    },
    "action": {
        "objects": ["run","eat","build","throw","buy","learn","measure",
                    "communicate","swim","write","sing","draw","fly","climb","teach"],
        "relation": "is a type of action",
        "wrong_cat": "emotion",
        "target": ["action"],
    },
    "animal": {
        "objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle",
                    "snake","rabbit","whale","lion","bear","fox","wolf","deer"],
        "relation": "is a type of animal",
        "wrong_cat": "fruit",
        "target": ["animal"],
    },
}

# 扩展clothing competitor sets
CLOTHING_COMP_SETS = {
    "standard": ["fruit","animal","emotion","action"],
    "tool": ["hammer","knife","scissors","wrench","saw","axe","drill","pliers"],
    "object": ["rock","stick","paper","stone","brick","log","rope","wire"],
    "fabric": ["cotton","wool","silk","leather","linen","cloth","fabric","textile"],
    "artifact": ["car","house","phone","chair","door","window","lamp","wheel"],
}

# 扩展action子类型
ACTION_SUBTYPES = {
    "physical": ["run","swim","throw","climb","fly","dance","fight","jump"],
    "creation": ["build","draw","write","sing","cook","paint","carve","sew"],
    "exchange": ["buy","sell","trade","give","pay","lend","borrow","offer"],
    "cognitive": ["learn","read","think","plan","solve","remember","decide","judge"],
}

ALL_CLASS_TOKENS = ["fruit","animal","clothing","emotion","action"]


def load_model_bf16(model_name):
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] {model_name} bf16+auto from {cfg['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    gpu_n = cpu_n = 0
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_n = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_n = sum(1 for v in dmap.values() if 'cpu' in str(v))
    print(f"[load] GPU:{gpu_n} CPU:{cpu_n} {gpu:.1f}GB {type(model).__name__}")
    return model, tokenizer, device


def get_norm_weight(model):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj, p, None)
            if obj is None: break
        if obj and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta': return w.float().cpu().numpy()
    return None


def load_weight_from_safetensors(model_name, key_contains):
    cfg = MODEL_CONFIGS[model_name]
    model_dir = Path(cfg["path"])
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if all(k in key.lower() for k in key_contains):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_ids(tokenizer, words):
    ids = []
    for w in words:
        tid = tokenizer.encode(w, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def forward(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy().astype(np.float64)


def dcf(h, W_U, tid, cid):
    l = h @ W_U.T
    t = np.mean([l[i] for i in tid if i < len(l)]) if tid else 0
    c = np.mean([l[i] for i in cid if i < len(l)]) if cid else 0
    return float(t - c)


def dcf_split(h, W_U, tid, cid):
    """Returns (D, target, competitor)"""
    l = h @ W_U.T
    t = np.mean([l[i] for i in tid if i < len(l)]) if tid else 0
    c = np.mean([l[i] for i in cid if i < len(l)]) if cid else 0
    return float(t - c), float(t), float(c)


def unit(v):
    n = np.linalg.norm(v)
    return (v/n) if n > 1e-10 else v


def proj(v, onto):
    u = unit(onto)
    return np.dot(v, u) * u


# ==================== MAIN ====================
def run_phase503(model, tokenizer, model_name, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    print(f"[info] layers={info.n_layers} d_model={d_model}")
    t0 = time.time()

    # --- Weights ---
    W_U = get_W_U(model, model_name)
    if W_U is None: return None
    W_U = W_U.astype(np.float64)
    g_vec = get_norm_weight(model)
    if g_vec is None:
        g_vec = load_weight_from_safetensors(model_name, ["norm","weight"])
        if g_vec is not None: g_vec = g_vec.astype(np.float64)
    if g_vec is None: return None
    print(f"[W_U] {W_U.shape} [g] {g_vec.shape}")

    # --- Token IDs ---
    cat_ids = {c: get_ids(tokenizer, cfg["target"]) for c, cfg in CATEGORIES.items()}
    std_comp_ids = get_ids(tokenizer, ALL_CLASS_TOKENS)
    # Remove own class from competitors
    cat_comp_ids = {}
    for c in CATEGORIES:
        cat_comp_ids[c] = [i for i in std_comp_ids if i not in cat_ids[c]]

    # --- Clothing competitor sets ---
    cloth_comp_ids = {}
    for set_name, tokens in CLOTHING_COMP_SETS.items():
        if set_name == "standard":
            cloth_comp_ids[set_name] = cat_comp_ids["clothing"]
        else:
            cloth_comp_ids[set_name] = get_ids(tokenizer, tokens)

    # --- Per-category results ---
    all_results = {}

    for cat_name, cfg in CATEGORIES.items():
        tc = time.time()
        print(f"\n--- {cat_name} ({len(cfg['objects'])} obj) ---")
        tid = cat_ids[cat_name]
        cid = cat_comp_ids[cat_name]
        objs = cfg["objects"]
        n = len(objs)

        # Build 4 v_cat variants
        # v1: rich - neutral, v2: rich - wrong_cat, v3: matched_neutral, v4: no_catword
        h_rich = [None] * n
        h_neutral = [None] * n
        h_wrong = [None] * n
        h_matched = [None] * n
        h_nocat = [None] * n

        for i, obj in enumerate(objs):
            h_rich[i] = forward(model, tokenizer, f"The {obj} {cfg['relation']}", device)
            h_neutral[i] = forward(model, tokenizer, f"The {obj} is a thing", device)
            h_wrong[i] = forward(model, tokenizer, f"The {obj} is a type of {cfg['wrong_cat']}", device)
            h_matched[i] = forward(model, tokenizer, f"The {obj} belongs to a category", device)
            # no-catword: strip the category word but keep relation structure
            nocat_rel = cfg['relation'].replace(cat_name, "it")
            h_nocat[i] = forward(model, tokenizer, f"The {obj} {nocat_rel}", device)

            if (i+1) % LOG_INTERVAL == 0:
                D = dcf(h_rich[i], W_U, tid, cid)
                print(f"  [{i+1}/{n}] D={D:.2f}")

        # 4 v_cat variants
        v1 = np.mean([h_rich[i]-h_neutral[i] for i in range(n)], axis=0)  # rich-neutral
        v2 = np.mean([h_rich[i]-h_wrong[i] for i in range(n)], axis=0)    # rich-wrong
        v3 = np.mean([h_rich[i]-h_matched[i] for i in range(n)], axis=0)  # rich-matched_neutral
        v4 = np.mean([h_rich[i]-h_nocat[i] for i in range(n)], axis=0)    # rich-no_catword

        # w_D and q_c (= g⊙w_D)
        wD_t = np.mean([W_U[i] for i in tid if i < len(W_U)], axis=0)
        wD_c = np.mean([W_U[i] for i in cid if i < len(W_U)], axis=0)
        wD = wD_t - wD_c
        qc = wD * g_vec

        # ========== Exp1: 4-variant v_cat slope test ==========
        exp1 = {}
        for vname, vvec in [("rich-neutral",v1), ("rich-wrong",v2), ("matched-neutral",v3), ("no-catword",v4)]:
            slopes = []
            for i in range(n):
                D0 = dcf(h_rich[i], W_U, tid, cid)
                h_mod = h_rich[i] + 1.0 * vvec
                D1 = dcf(h_mod, W_U, tid, cid)
                slopes.append(D1 - D0)
            exp1[vname] = {"mean_delta": round(float(np.mean(slopes)), 4), "std": round(float(np.std(slopes)), 4)}

        # ========== Exp2: target/competitor decomposition ==========
        exp2 = {"v1": [], "v2": [], "v3": [], "v4": []}
        for i in range(n):
            for vname, vvec in [("v1",v1), ("v2",v2), ("v3",v3), ("v4",v4)]:
                D0, t0, c0 = dcf_split(h_rich[i], W_U, tid, cid)
                D1, t1, c1 = dcf_split(h_rich[i] + 1.0 * vvec, W_U, tid, cid)
                exp2[vname].append({
                    "dD": D1-D0, "dT": t1-t0, "dC": c1-c0,
                    "T_frac": (t1-t0)/(abs(D1-D0)+1e-10), "C_frac": -(c1-c0)/(abs(D1-D0)+1e-10),
                })

        exp2_summary = {}
        for vname in ["v1","v2","v3","v4"]:
            arr = exp2[vname]
            exp2_summary[vname] = {
                "mean_dD": round(float(np.mean([a["dD"] for a in arr])), 4),
                "mean_dT": round(float(np.mean([a["dT"] for a in arr])), 4),
                "mean_dC": round(float(np.mean([a["dC"] for a in arr])), 4),
                "T_dominant": sum(1 for a in arr if abs(a["T_frac"]) > abs(a["C_frac"])),
                "C_dominant": sum(1 for a in arr if abs(a["C_frac"]) > abs(a["T_frac"])),
            }

        # ========== Exp3: matched-norm parallel/orthogonal ==========
        v_para = proj(v1, qc)
        v_perp = v1 - v_para
        para_norm = np.linalg.norm(v_para)
        perp_norm = np.linalg.norm(v_perp)
        # Match norms: scale both to 1.0
        v_para_unit = unit(v_para)
        v_perp_unit = unit(v_perp)

        exp3 = {"para": [], "perp": [], "both": [], "random": []}
        for i in range(n):
            h0 = h_rich[i]
            D0 = dcf(h0, W_U, tid, cid)
            # para: matched-norm injection
            Dp = dcf(h0 + 1.0 * v_para_unit, W_U, tid, cid)
            # perp: same norm injection
            Dx = dcf(h0 + 1.0 * v_perp_unit, W_U, tid, cid)
            # both: full v1
            Db = dcf(h0 + 1.0 * v1, W_U, tid, cid)
            # random direction control
            rnd = np.random.randn(d_model).astype(np.float64)
            rnd_u = unit(rnd)
            Dr = dcf(h0 + 1.0 * rnd_u, W_U, tid, cid)
            exp3["para"].append(Dp-D0)
            exp3["perp"].append(Dx-D0)
            exp3["both"].append(Db-D0)
            exp3["random"].append(Dr-D0)

        exp3_summary = {
            k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for k, v in exp3.items()
        }
        if para_norm > 1e-10:
            exp3_summary["para_frac"] = round(float(para_norm**2 / (para_norm**2 + perp_norm**2)), 4)
        else:
            exp3_summary["para_frac"] = 0.0

        # ========== Exp4: clothing multi-competitor (only for clothing) ==========
        exp4 = {}
        if cat_name == "clothing":
            for set_name, cids in cloth_comp_ids.items():
                if not cids: continue
                slopes = []
                for i in range(n):
                    D0 = dcf(h_rich[i], W_U, tid, cids)
                    D1 = dcf(h_rich[i] + 1.0 * v1, W_U, tid, cids)
                    slopes.append(D1 - D0)
                exp4[set_name] = {
                    "mean_delta": round(float(np.mean(slopes)), 4),
                    "n_comp": len(cids),
                }

        # ========== Exp5: action subtype decomposition ==========
        exp5 = {}
        if cat_name == "action":
            for stype, stype_objs in ACTION_SUBTYPES.items():
                # Find objects that are in this subtype and in our test set
                stype_test = [o for o in stype_objs if o in objs]
                if len(stype_test) < 3: continue
                stype_idx = [objs.index(o) for o in stype_test]
                slopes = []
                for i in stype_idx:
                    D0 = dcf(h_rich[i], W_U, tid, cid)
                    D1 = dcf(h_rich[i] + 1.0 * v1, W_U, tid, cid)
                    slopes.append(D1 - D0)
                exp5[stype] = {
                    "mean_delta": round(float(np.mean(slopes)), 4),
                    "n": len(stype_test),
                    "objects": stype_test[:5],
                }

        # Store
        all_results[cat_name] = {
            "exp1_variants": exp1,
            "exp2_target_competitor": exp2_summary,
            "exp3_matched_norm": exp3_summary,
            "exp4_clothing_multicomp": exp4,
            "exp5_action_subtypes": exp5,
            "baseline": {
                "mean_D_rich": round(float(np.mean([dcf(h_rich[i], W_U, tid, cid) for i in range(n)])), 4),
                "mean_D_neutral": round(float(np.mean([dcf(h_neutral[i], W_U, tid, cid) for i in range(n)])), 4),
                "cos_v1_qc": round(float(np.dot(unit(v1), unit(qc))) if np.linalg.norm(v1)>1e-10 else 0, 6),
                "cos_v1_wD": round(float(np.dot(unit(v1), unit(wD))) if np.linalg.norm(v1)>1e-10 else 0, 6),
                "gain_ratio": round(float(np.linalg.norm(qc)/max(np.linalg.norm(wD),1e-10)), 2),
            },
        }

        dt = time.time() - tc
        b = all_results[cat_name]["baseline"]
        print(f"  [{cat_name}] {dt:.0f}s D={b['mean_D_rich']:.1f} "
              f"cos_v_qc={b['cos_v1_qc']:.4f} ratio={b['gain_ratio']:.1f} "
              f"para={exp3_summary['para']['mean']:.3f} perp={exp3_summary['perp']['mean']:.3f}")

    # ===== Summary =====
    para_means = [all_results[c]["exp3_matched_norm"]["para"]["mean"] for c in all_results]
    perp_means = [all_results[c]["exp3_matched_norm"]["perp"]["mean"] for c in all_results]
    random_means = [all_results[c]["exp3_matched_norm"]["random"]["mean"] for c in all_results]
    cos_v_qc = [all_results[c]["baseline"]["cos_v1_qc"] for c in all_results]

    summary = {
        "model": model_name, "n_layers": info.n_layers, "d_model": d_model,
        "total_time": round(time.time()-t0, 1),
        "mean_para_effect": round(np.mean(para_means), 4),
        "mean_perp_effect": round(np.mean(perp_means), 4),
        "mean_random_effect": round(np.mean(random_means), 4),
        "mean_cos_v_qc": round(np.mean(cos_v_qc), 6),
        "categories": all_results,
    }

    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase503_target_competitor_decomp.py <model>")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown: {model_name}")
        sys.exit(1)

    print("="*70)
    print(f"Phase 503: Target/Competitor Decomp — {model_name}")
    print("Exp1: 4-variant v_cat | Exp2: T/C split | Exp3: matched-norm")
    print("Exp4: clothing multicomp | Exp5: action subtypes | Exp6: random ctrl")
    print("="*70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    print(f"[load] {time.time()-t0:.0f}s")

    try:
        results = run_phase503(model, tokenizer, model_name, device)
        total = time.time() - t0

        if results is None: return

        # Print
        print(f"\n{'='*70}\nRESULTS — {model_name}\n{'='*70}")
        print(f"  mean para_effect:    {results['mean_para_effect']:+.4f}")
        print(f"  mean perp_effect:    {results['mean_perp_effect']:+.4f}")
        print(f"  mean random_effect:  {results['mean_random_effect']:+.4f}")
        print(f"  mean cos_v_qc:       {results['mean_cos_v_qc']:.5f}")

        print(f"\n  Exp1: 4-variant v_cat slopes")
        for cat, r in results['categories'].items():
            e1 = r['exp1_variants']
            print(f"  {cat:10s}: rich-neutral={e1['rich-neutral']['mean_delta']:+.3f}  "
                  f"rich-wrong={e1['rich-wrong']['mean_delta']:+.3f}  "
                  f"matched={e1['matched-neutral']['mean_delta']:+.3f}  "
                  f"nocat={e1['no-catword']['mean_delta']:+.3f}")

        print(f"\n  Exp2: Target/Competitor split (v1=rich-neutral)")
        for cat, r in results['categories'].items():
            e2 = r['exp2_target_competitor']['v1']
            dom = "T" if e2['T_dominant'] > e2['C_dominant'] else "C"
            print(f"  {cat:10s}: dT={e2['mean_dT']:+.3f} dC={e2['mean_dC']:+.3f}  [{dom}-dom {e2['T_dominant']}T/{e2['C_dominant']}C]")

        print(f"\n  Exp3: Matched-norm para/perp/random")
        for cat, r in results['categories'].items():
            e3 = r['exp3_matched_norm']
            print(f"  {cat:10s}: para={e3['para']['mean']:+.3f} perp={e3['perp']['mean']:+.3f} "
                  f"rand={e3['random']['mean']:+.3f}  frac={e3['para_frac']:.3f}")

        if results['categories'].get('clothing',{}).get('exp4_clothing_multicomp'):
            print(f"\n  Exp4: Clothing multi-competitor")
            for k, v in results['categories']['clothing']['exp4_clothing_multicomp'].items():
                print(f"    {k:12s}: Δ={v['mean_delta']:+.3f} (n={v['n_comp']})")

        if results['categories'].get('action',{}).get('exp5_action_subtypes'):
            print(f"\n  Exp5: Action subtypes")
            for k, v in results['categories']['action']['exp5_action_subtypes'].items():
                print(f"    {k:12s}: Δ={v['mean_delta']:+.3f} (n={v['n']})  {v['objects']}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"phase503_{model_name}_r1.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

    finally:
        release_model(model)
        print("  Released.")

if __name__ == "__main__":
    main()
