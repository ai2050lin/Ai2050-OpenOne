"""
Phase 504: W·R_pre vs g⊙w_D 对齐 — GPT5×GLM5 路线统一终测
============================================================
核心测试: cos(W·R_pre, g⊙w_D) — GPT5的转移映射是否与GLM5的Gain读出方向一致?

方法:
  train: 12对象/类 拟合 ridge W
  test:  8对象/类 计算 W·R_pre
  对比: cos(W·R_pre, g⊙w_D) vs cos(W·R_pre, w_D) vs cos(W·R_pre, v_c)

加载: bf16 + device_map="auto" + sdpa
Usage: python tests/glm5/phase504_W_Rpre_alignment.py qwen3
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from model_utils import (get_model_info, release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("results/glm5")
LOG_INTERVAL = 3

CATEGORIES = {
    "fruit":   {"objects": ["apple","banana","orange","grape","pear","peach","mango","plum",
                            "cherry","lemon","apricot","kiwi","pineapple","melon","coconut","lime","fig","pomegranate","papaya","avocado"],
                "relation": "is a type of fruit"},
    "clothing":{"objects": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
                            "scarf","vest","hat","glove","sock","boot","belt","tie","jeans","shorts","hoodie","raincoat"],
                "relation": "is a type of clothing"},
    "emotion": {"objects": ["joy","anger","fear","sadness","surprise","disgust","pride","shame",
                            "guilt","envy","hope","love","hate","boredom","anxiety","jealousy","gratitude","regret","curiosity","embarrassment"],
                "relation": "is a type of emotion"},
    "action":  {"objects": ["run","eat","build","throw","buy","learn","measure","communicate",
                            "swim","write","sing","draw","fly","climb","teach","drive","cook","dance","fight","sleep"],
                "relation": "is a type of action"},
    "animal":  {"objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
                            "rabbit","whale","lion","bear","fox","wolf","deer","monkey","shark","frog","penguin","owl"],
                "relation": "is a type of animal"},
}

ALL_CLASS_TOKENS = ["fruit","animal","clothing","emotion","action"]
TRAIN_N = 12
TEST_N = 8


def load_model_bf16(model_name):
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device
    gpu = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    gpu_n = cpu_n = 0
    if hasattr(model,'hf_device_map'):
        d = model.hf_device_map
        gpu_n = sum(1 for v in d.values() if 'cuda' in str(v))
        cpu_n = sum(1 for v in d.values() if 'cpu' in str(v))
    print(f"[load] GPU:{gpu_n} CPU:{cpu_n} {gpu:.1f}GB {type(model).__name__}")
    return model, tokenizer, device


def get_norm_g(model, model_name=""):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj,p,None)
            if obj is None: break
        if obj and hasattr(obj,'weight'):
            w = obj.weight.detach()
            if str(w.device)!='meta': return w.float().cpu().numpy()
    # fallback: safetensors (last file, typically contains model.norm.weight)
    cfg = MODEL_CONFIGS[model_name]
    sfs = sorted(Path(cfg["path"]).glob("*.safetensors"))
    for sf in sfs:
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if ('norm' in key.lower() and 'weight' in key.lower()
                    and not any(x in key for x in ['layer','input','post','attention','rms','self'])):
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
    hs = outputs.hidden_states
    last = hs[-1][0, -1, :].float().cpu().numpy().astype(np.float64)
    # pre-answer = 倒数第二个token
    seq_len = hs[-1].shape[1]
    if seq_len >= 2:
        pre = hs[-1][0, -2, :].float().cpu().numpy().astype(np.float64)
    else:
        pre = last.copy()
    return last, pre


def ridge_map(X, Y, ridge=0.1):
    """X=[n,d], Y=[n,d] -> W=[d,d]"""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    XtX = X.T @ X
    reg = ridge * np.eye(XtX.shape[0], dtype=np.float64)
    return np.linalg.solve(XtX + reg, X.T @ Y).astype(np.float64)


def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-10 else v


def run_phase504(model, tokenizer, model_name, device):
    info = get_model_info(model, model_name)
    print(f"[info] layers={info.n_layers} d_model={info.d_model}")
    t0 = time.time()

    W_U = get_W_U(model, model_name)
    if W_U is None: return None
    W_U = W_U.astype(np.float64)
    g = get_norm_g(model, model_name)
    if g is None: return None
    g = g.astype(np.float64)
    print(f"[W_U] {W_U.shape} [g] {g.shape}")

    cat_ids = {}
    comp_ids_all = []
    for cat, cfg in CATEGORIES.items():
        cat_ids[cat] = get_ids(tokenizer, [cat])
    for cat in CATEGORIES:
        others = [c for c in ALL_CLASS_TOKENS if c != cat]
        comp_ids_all = list(set(comp_ids_all + get_ids(tokenizer, others)))

    results = {}

    for cat_name, cfg in CATEGORIES.items():
        tc = time.time()
        print(f"\n--- {cat_name} (train={TRAIN_N}, test={TEST_N}) ---")
        objs = cfg["objects"]
        tid = cat_ids[cat_name]
        cid = comp_ids_all

        # Compute w_D and g⊙w_D once (global to category)
        wD_t = np.mean([W_U[i] for i in tid if i < len(W_U)], axis=0)
        wD_c = np.mean([W_U[i] for i in cid if i < len(W_U)], axis=0)
        wD = wD_t - wD_c
        qc = wD * g

        # --- Collect train/test data ---
        train_objs = objs[:TRAIN_N]
        test_objs = objs[TRAIN_N:TRAIN_N+TEST_N]

        X_train = []  # pre-answer states (R_pre)
        Y_train = []  # answer states (A_answer)
        v_cat_test = []  # v_c = answer_rich - answer_neutral

        for i, obj in enumerate(train_objs):
            prom = f"The {obj} {cfg['relation']}"
            ans, pre = forward(model, tokenizer, prom, device)
            X_train.append(pre)
            Y_train.append(ans)
            if (i+1) % LOG_INTERVAL == 0:
                print(f"  train [{i+1}/{TRAIN_N}] ||pre||={np.linalg.norm(pre):.0f}")

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Fit W via ridge regression
        W_map = ridge_map(X_train, Y_train, ridge=0.1)

        # Predict on test
        WR_pre_list = []
        v_cat_list = []
        D_base_list = []

        for i, obj in enumerate(test_objs):
            prom_rich = f"The {obj} {cfg['relation']}"
            prom_neutral = f"The {obj} is a thing"

            ans_rich, pre_rich = forward(model, tokenizer, prom_rich, device)
            ans_neutral, _ = forward(model, tokenizer, prom_neutral, device)

            # W·R_pre: predicted answer from pre-answer
            wr = W_map @ pre_rich
            WR_pre_list.append(wr)

            # v_cat: observed answer difference
            vc = ans_rich - ans_neutral
            v_cat_list.append(vc)

            # D baseline
            logits = ans_rich @ W_U.T
            t = np.mean([logits[j] for j in tid if j < len(logits)])
            c = np.mean([logits[j] for j in cid if j < len(logits)])
            D_base_list.append(float(t-c))

            if (i+1) % LOG_INTERVAL == 0:
                print(f"  test [{i+1}/{TEST_N}] D={D_base_list[-1]:.1f}")

        # --- Compute alignments ---
        WR_mean = np.mean(WR_pre_list, axis=0)
        vc_mean = np.mean(v_cat_list, axis=0)

        cos_WR_wD = float(np.dot(unit(WR_mean), unit(wD))) if np.linalg.norm(WR_mean)>1e-10 else 0
        cos_WR_qc = float(np.dot(unit(WR_mean), unit(qc))) if np.linalg.norm(WR_mean)>1e-10 else 0
        cos_WR_vc = float(np.dot(unit(WR_mean), unit(vc_mean))) if np.linalg.norm(WR_mean)>1e-10 else 0
        cos_vc_qc = float(np.dot(unit(vc_mean), unit(qc))) if np.linalg.norm(vc_mean)>1e-10 else 0

        # R2 of transfer
        Y_pred = np.array([W_map @ x for x in X_train])
        Y_test_pred = np.array([W_map @ fwd[1] for fwd in
            [forward(model, tokenizer, f"The {obj} {cfg['relation']}", device) for obj in train_objs]])
        ss_res = np.sum((Y_train - Y_pred)**2)
        ss_tot = np.sum((Y_train - Y_train.mean(axis=0))**2)
        r2_train = float(1 - ss_res/(ss_tot+1e-10))

        results[cat_name] = {
            "cos_WR_wD": round(cos_WR_wD, 6),
            "cos_WR_qc": round(cos_WR_qc, 6),
            "cos_WR_vc": round(cos_WR_vc, 6),
            "cos_vc_qc": round(cos_vc_qc, 6),
            "gain_boost_WR": round(cos_WR_qc - cos_WR_wD, 6),
            "gain_boost_vc": round(cos_vc_qc - cos_WR_vc, 6),
            "r2_train": round(r2_train, 6),
            "norm_WR": round(float(np.linalg.norm(WR_mean)), 2),
            "norm_vc": round(float(np.linalg.norm(vc_mean)), 2),
            "mean_D_base": round(float(np.mean(D_base_list)), 2),
        }

        dt = time.time()-tc
        r = results[cat_name]
        print(f"  [{cat_name}] {dt:.0f}s WR→qc={r['cos_WR_qc']:.4f} "
              f"WR→wD={r['cos_WR_wD']:.4f} Δ={r['gain_boost_WR']:+.4f} "
              f"vc→qc={r['cos_vc_qc']:.4f} r2={r['r2_train']:.3f}")

    # Summary
    all_gb = [results[c]["gain_boost_WR"] for c in results]
    all_cos_qc = [results[c]["cos_WR_qc"] for c in results]
    all_r2 = [results[c]["r2_train"] for c in results]

    summary = {
        "model": model_name,
        "mean_gain_boost_WR": round(np.mean(all_gb), 6),
        "mean_cos_WR_qc": round(np.mean(all_cos_qc), 6),
        "mean_r2_train": round(np.mean(all_r2), 6),
        "positive_gain_cats": sum(1 for gb in all_gb if gb > 0.001),
        "total_cats": len(all_gb),
        "categories": results,
    }
    return summary


def main():
    if len(sys.argv)<2:
        print("Usage: python phase504_W_Rpre_alignment.py <model>")
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS: sys.exit(1)

    print("="*70)
    print(f"Phase 504: W·R_pre vs g⊙w_D — {model_name}")
    print(f"Train={TRAIN_N} Test={TEST_N} per category")
    print("="*70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    print(f"[load] {time.time()-t0:.0f}s")
    try:
        res = run_phase504(model, tokenizer, model_name, device)
        tot = time.time()-t0
        if res is None: return

        print(f"\n{'='*70}\nRESULTS — {model_name}\n{'='*70}")
        print(f"  mean cos(WR,qc):      {res['mean_cos_WR_qc']:+.5f}")
        print(f"  mean gain_boost WR:   {res['mean_gain_boost_WR']:+.5f}")
        print(f"  mean r2_train:        {res['mean_r2_train']:.4f}")
        print(f"  +gain cats:           {res['positive_gain_cats']}/{res['total_cats']}")
        print(f"  total time:           {tot:.0f}s")

        print(f"\n  Per-category:")
        for cat, r in res['categories'].items():
            sig = "✅" if r['gain_boost_WR']>0.001 else ("⚠️" if r['gain_boost_WR']>-0.001 else "❌")
            print(f"  {sig} {cat:10s}: WR→qc={r['cos_WR_qc']:+.5f} WR→wD={r['cos_WR_wD']:+.5f} "
                  f"Δ={r['gain_boost_WR']:+.5f} vc→qc={r['cos_vc_qc']:+.5f} r2={r['r2_train']:.3f}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR/f"phase504_{model_name}_r1.json"
        with open(out,'w',encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out}")
    finally:
        release_model(model)
        print("  Released.")

if __name__=="__main__":
    main()
