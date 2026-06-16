"""
Phase 505 R1: W·R_pre 全层扫描 — 找正R²工作窗口
===============================================
Phase 504证明W·R_pre在末层失败(R²<0)。R1扫描7个关键层找到正R²窗口。
每层: ridge fit pre→ans, 报告R² + cos(WR, qc) + cos(WR, vc)

加载: model_demo_bf16.py 模式 (bf16+auto+sdpa)
Usage: python tests/glm5/phase505_layer_sweep_WRpre.py qwen3 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'tests/glm5')

import gc, json, time, traceback
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from model_utils import (get_model_info, release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("results/glm5")
TRAIN_N = 12
TEST_N = 8

CATEGORIES = {
    "fruit":{"objects":["apple","banana","orange","grape","pear","peach","mango","plum",
              "cherry","lemon","apricot","kiwi","pineapple","melon","coconut","lime","fig","pomegranate","papaya","avocado"],
              "relation":"is a type of fruit"},
    "emotion":{"objects":["joy","anger","fear","sadness","surprise","disgust","pride","shame",
              "guilt","envy","hope","love","hate","boredom","anxiety","jealousy","gratitude","regret","curiosity","embarrassment"],
              "relation":"is a type of emotion"},
    "animal":{"objects":["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
              "rabbit","whale","lion","bear","fox","wolf","deer","monkey","shark","frog","penguin","owl"],
              "relation":"is a type of animal"},
    "action":{"objects":["run","eat","build","throw","buy","learn","measure","communicate",
              "swim","write","sing","draw","fly","climb","teach","drive","cook","dance","fight","sleep"],
              "relation":"is a type of action"},
    "clothing":{"objects":["shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
              "scarf","vest","hat","glove","sock","boot","belt","tie","jeans","shorts","hoodie","raincoat"],
              "relation":"is a type of clothing"},
}
ALL_CLASS = ["fruit","animal","clothing","emotion","action"]


def load_bf16(name):
    cfg = MODEL_CONFIGS[name]
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    m.eval()
    dev = next(m.parameters()).device
    g = torch.cuda.memory_allocated()/1e9
    gn=cn=0
    if hasattr(m,'hf_device_map'):
        dm=m.hf_device_map; gn=sum(1 for v in dm.values() if 'cuda' in str(v)); cn=sum(1 for v in dm.values() if 'cpu' in str(v))
    print(f"[load] GPU:{gn} CPU:{cn} {g:.1f}GB {type(m).__name__}")
    return m, tok, dev


def get_norm_g(model, model_name):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj,p,None); 
            if obj is None: break
        if obj and hasattr(obj,'weight'):
            w = obj.weight.detach()
            if str(w.device)!='meta': return w.float().cpu().numpy()
    cfg = MODEL_CONFIGS[model_name]
    for sf in sorted(Path(cfg["path"]).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if 'norm' in key.lower() and 'weight' in key.lower() and not any(x in key for x in ['layer','input','post']):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_ids(tokenizer, words):
    ids = []
    for w in words:
        tid = tokenizer.encode(w, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def forward_layer(model, tokenizer, prompt, device, target_layer):
    """Forward pass, extract hidden at target_layer for last token and pre-last token"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states
    L = len(hs)-1
    layer_idx = min(target_layer, L-1)
    seq_len = hs[layer_idx].shape[1]
    ans = hs[layer_idx][0, -1, :].float().cpu().numpy().astype(np.float64)
    pre = hs[layer_idx][0, -2, :].float().cpu().numpy().astype(np.float64) if seq_len>=2 else ans.copy()
    return ans, pre


def ridge_map(X, Y, ridge=0.1):
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge*np.eye(XtX.shape[0]), X.T @ Y).astype(np.float64)


def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-10 else v


def run_phase505(model, tokenizer, model_name, device):
    info = get_model_info(model, model_name)
    L = info.n_layers
    # 7 key layers
    scan_layers = [0, L//4, L//2, 3*L//4, L-3, L-2, L-1]
    scan_layers = [min(l, L-1) for l in scan_layers]
    scan_layers = sorted(set(scan_layers))
    print(f"[info] L={L}, scan_layers={scan_layers}")

    # Weights
    W_U = get_W_U(model, model_name)
    if W_U is None: return None
    W_U = W_U.astype(np.float64)
    g = get_norm_g(model, model_name)
    if g is None: return None
    g = g.astype(np.float64)

    cat_ids = {}
    comp_ids = []
    for cat in CATEGORIES:
        cat_ids[cat] = get_ids(tokenizer, [cat])
    for cat in CATEGORIES:
        others = [c for c in ALL_CLASS if c!=cat]
        comp_ids = list(set(comp_ids + get_ids(tokenizer, others)))

    results = {}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        objs = cfg["objects"]
        tid = cat_ids[cat_name]
        cid = comp_ids
        train_objs = objs[:TRAIN_N]
        test_objs = objs[TRAIN_N:TRAIN_N+TEST_N]

        # Global qc for this category
        wD_t = np.mean([W_U[i] for i in tid if i<len(W_U)], axis=0)
        wD_c = np.mean([W_U[i] for i in cid if i<len(W_U)], axis=0)
        qc = (wD_t - wD_c) * g
        wD = wD_t - wD_c

        layer_results = {}

        for li, layer in enumerate(scan_layers):
            # Collect train data
            X, Y = [], []
            for obj in train_objs:
                ans, pre = forward_layer(model, tokenizer, f"The {obj} {cfg['relation']}", device, layer)
                X.append(pre)
                Y.append(ans)
            X = np.array(X); Y = np.array(Y)
            W = ridge_map(X, Y)

            # Test
            WR_list, vc_list = [], []
            for obj in test_objs:
                ans_r, pre_r = forward_layer(model, tokenizer, f"The {obj} {cfg['relation']}", device, layer)
                ans_n, _ = forward_layer(model, tokenizer, f"The {obj} is a thing", device, layer)
                WR_list.append(W @ pre_r)
                vc_list.append(ans_r - ans_n)

            WR_mean = np.mean(WR_list, axis=0)
            vc_mean = np.mean(vc_list, axis=0)

            # R2 on train
            Y_pred = np.array([W @ x for x in X])
            ss_res = np.sum((Y - Y_pred)**2)
            ss_tot = np.sum((Y - Y.mean(axis=0))**2)
            r2 = float(1 - ss_res/(ss_tot+1e-10))

            # Cosines
            wrm = np.linalg.norm(WR_mean)
            vcm = np.linalg.norm(vc_mean)
            cos_WR_qc = float(np.dot(unit(WR_mean), unit(qc))) if wrm>1e-10 else 0
            cos_WR_wD = float(np.dot(unit(WR_mean), unit(wD))) if wrm>1e-10 else 0
            cos_WR_vc = float(np.dot(unit(WR_mean), unit(vc_mean))) if wrm>1e-10 and vcm>1e-10 else 0
            cos_vc_qc = float(np.dot(unit(vc_mean), unit(qc))) if vcm>1e-10 else 0

            layer_results[str(layer)] = {
                "r2": round(r2, 4),
                "cos_WR_qc": round(cos_WR_qc, 6),
                "cos_WR_wD": round(cos_WR_wD, 6),
                "cos_WR_vc": round(cos_WR_vc, 6),
                "cos_vc_qc": round(cos_vc_qc, 6),
                "gain_boost": round(cos_WR_qc - cos_WR_wD, 6),
                "norm_WR": round(float(wrm), 2),
                "norm_vc": round(float(vcm), 2),
            }

            sig = "✅" if r2 > 0 else "❌"
            print(f"  {sig} L{layer:>3}: R²={r2:+.3f} cosWR→qc={cos_WR_qc:+.4f} "
                  f"cosWR→vc={cos_WR_vc:+.4f} cosvc→qc={cos_vc_qc:+.4f}")

        results[cat_name] = layer_results

    # Best layer per category
    summary = {"model": model_name, "L": L, "scan_layers": scan_layers, "categories": results}
    return summary


def main():
    if len(sys.argv)<2:
        print("Usage: python phase505_layer_sweep_WRpre.py <model> [1|2]")
        sys.exit(1)
    mn = sys.argv[1]; rd = sys.argv[2] if len(sys.argv)>2 else "1"
    if mn not in MODEL_CONFIGS: sys.exit(1)

    print("="*70)
    print(f"Phase 505 R{rd}: Layer Sweep W·R_pre — {mn}")
    print(f"train={TRAIN_N} test={TEST_N} 3 cats=5")
    print("="*70)

    t0=time.time()
    model, tokenizer, device = load_bf16(mn)
    print(f"[load] {time.time()-t0:.0f}s")
    try:
        res = run_phase505(model, tokenizer, mn, device)
        tot = time.time()-t0
        if res is None: return

        print(f"\n{'='*70}\nRESULTS — {mn}\n{'='*70}")
        # Find best layer per category
        for cat, lr in res["categories"].items():
            best = max(lr.items(), key=lambda x: x[1]["r2"])
            print(f"  {cat:10s}: best L={best[0]:>3} R²={best[1]['r2']:+.3f} "
                  f"cosWR→qc={best[1]['cos_WR_qc']:+.4f} cosWR→vc={best[1]['cos_WR_vc']:+.4f}")

        # Layer-average R² across categories
        print(f"\n  Layer-level avg R²:")
        for l in res["scan_layers"]:
            avg_r2 = np.mean([res["categories"][c][str(l)]["r2"] for c in CATEGORIES])
            n_pos = sum(1 for c in CATEGORIES if res["categories"][c][str(l)]["r2"]>0)
            print(f"    L{l:>3}: avg R²={avg_r2:+.3f}  +cats={n_pos}/{len(CATEGORIES)}")

        out = OUTPUT_DIR/f"phase505_{mn}_r{rd}.json"
        with open(out,'w',encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out}")
    finally:
        release_model(model)
        print("  Released.")

if __name__=="__main__":
    main()
