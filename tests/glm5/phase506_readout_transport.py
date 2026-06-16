"""
Phase 506 R1: 读出相关转移映射 — 预测标量替代完整向量
=======================================================
Phase 505证明: 预测完整answer hidden state失败(全层负R²)
Phase 506修正: 预测读出相关标量 <ans,qc>, D, T, C

扫5层×5类, 12train/8test, ridge拟合pre→ans后计算标量预测R²

加载: model_demo_bf16.py (bf16+auto+sdpa)
Usage: python tests/glm5/phase506_readout_transport.py qwen3 1
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
TRAIN_N = 12; TEST_N = 8

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


def get_norm_g(model, name):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj,p,None)
            if obj is None: break
        if obj and hasattr(obj,'weight'):
            w = obj.weight.detach()
            if str(w.device)!='meta': return w.float().cpu().numpy()
    cfg = MODEL_CONFIGS[name]
    for sf in sorted(Path(cfg["path"]).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if 'norm' in key.lower() and 'weight' in key.lower() and not any(x in key for x in ['layer','input','post']):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_ids(tokenizer, words):
    ids=[]
    for w in words:
        tid=tokenizer.encode(w,add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def forward_layer(model, tokenizer, prompt, device, target_layer):
    """Forward and return (ans, pre) at target_layer for last token"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states
    L = len(hs)-1
    li = min(target_layer, L-1)
    seq = hs[li].shape[1]
    ans = hs[li][0, -1, :].float().cpu().numpy().astype(np.float64)
    pre = hs[li][0, -2, :].float().cpu().numpy().astype(np.float64) if seq>=2 else ans.copy()
    return ans, pre


def ridge_map(X, Y, ridge=0.1):
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge*np.eye(XtX.shape[0]), X.T @ Y).astype(np.float64)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-y_true.mean())**2)
    return float(1 - ss_res/(ss_tot+1e-10))


def run_phase506(model, tokenizer, model_name, device):
    info = get_model_info(model, model_name)
    L = info.n_layers
    # 5 key layers: mid→late
    scan_layers = [L//2, 2*L//3, L-5, L-3, L-1]
    scan_layers = sorted(set(min(l, L-1) for l in scan_layers))
    print(f"[info] L={L}, scan={scan_layers}")

    W_U = get_W_U(model, model_name)
    if W_U is None: return None
    W_U = W_U.astype(np.float64)
    g = get_norm_g(model, model_name)
    if g is None: return None
    g = g.astype(np.float64)

    cat_ids = {}; comp_ids = []
    for cat in CATEGORIES: cat_ids[cat] = get_ids(tokenizer, [cat])
    for cat in CATEGORIES:
        others = [c for c in ALL_CLASS if c!=cat]
        comp_ids = list(set(comp_ids + get_ids(tokenizer, others)))

    # Global qc per category
    cat_qc = {}
    for cat, cfg in CATEGORIES.items():
        tid = cat_ids[cat]
        wDt = np.mean([W_U[i] for i in tid if i<len(W_U)], axis=0)
        wDc = np.mean([W_U[i] for i in comp_ids if i<len(W_U)], axis=0)
        cat_qc[cat] = (wDt - wDc) * g

    results = {}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        objs = cfg["objects"]
        tid = cat_ids[cat_name]
        cid = comp_ids
        qc = cat_qc[cat_name]
        train_objs = objs[:TRAIN_N]
        test_objs = objs[TRAIN_N:TRAIN_N+TEST_N]
        layer_results = {}

        for layer in scan_layers:
            # Collect train
            X, Y = [], []
            for obj in train_objs:
                ans, pre = forward_layer(model, tokenizer, f"The {obj} {cfg['relation']}", device, layer)
                X.append(pre); Y.append(ans)
            X = np.array(X); Y = np.array(Y)
            W = ridge_map(X, Y)

            # Train scalar targets
            Y_pred = np.array([W@x for x in X])
            y_D_true = np.array([np.dot(y, qc) for y in Y])
            y_D_pred = np.array([np.dot(yp, qc) for yp in Y_pred])

            # Test
            WR_test = []; ans_test = []; vc_test = []
            D_test, T_test, C_test = [], [], []
            for obj in test_objs:
                ans_r, pre_r = forward_layer(model, tokenizer, f"The {obj} {cfg['relation']}", device, layer)
                ans_n, _ = forward_layer(model, tokenizer, f"The {obj} is a thing", device, layer)
                wr = W @ pre_r
                WR_test.append(wr); ans_test.append(ans_r); vc_test.append(ans_r-ans_n)

                # D, T, C from answer (ground truth at answer position)
                l_r = ans_r @ W_U.T
                t_r = np.mean([l_r[i] for i in tid if i<len(l_r)])
                c_r = np.mean([l_r[i] for i in cid if i<len(l_r)])
                D_test.append(float(t_r-c_r)); T_test.append(float(t_r)); C_test.append(float(c_r))

            # Scalar predictions from WR
            D_pred_WR = np.array([np.dot(wr, qc) for wr in WR_test])
            T_pred_WR = np.array([np.dot(wr, g*W_U[t] if t<len(W_U) else 0) for wr,t in [(w,tid[0]) for w in WR_test]])
            # Actually compute T/C from logits of WR
            D_pred = []; T_pred = []; C_pred = []
            for wr in WR_test:
                l = wr @ W_U.T
                tp = np.mean([l[i] for i in tid if i<len(l)])
                cp = np.mean([l[i] for i in cid if i<len(l)])
                D_pred.append(float(tp-cp)); T_pred.append(float(tp)); C_pred.append(float(cp))

            # R² scores
            r2_vec = r2_score(np.array(ans_test).flatten(), np.array(WR_test).flatten())
            r2_D = r2_score(np.array(D_test), np.array(D_pred))
            r2_T = r2_score(np.array(T_test), np.array(T_pred))
            r2_C = r2_score(np.array(C_test), np.array(C_pred))

            # Cosine (direction alignment only)
            WR_m = np.mean(WR_test, axis=0)
            vc_m = np.mean(vc_test, axis=0)
            wrm = np.linalg.norm(WR_m); vcm = np.linalg.norm(vc_m); qcn = np.linalg.norm(qc)
            cos_WR_qc = float(np.dot(WR_m,qc)/(wrm*qcn+1e-10)) if wrm>1e-10 else 0
            cos_vc_qc = float(np.dot(vc_m,qc)/(vcm*qcn+1e-10)) if vcm>1e-10 else 0
            cos_WR_vc = float(np.dot(WR_m,vc_m)/(wrm*vcm+1e-10)) if wrm>1e-10 and vcm>1e-10 else 0

            layer_results[str(layer)] = {
                "r2_vector": round(r2_vec, 4),
                "r2_D": round(r2_D, 4),
                "r2_T": round(r2_T, 4),
                "r2_C": round(r2_C, 4),
                "cos_WR_qc": round(cos_WR_qc, 6),
                "cos_vc_qc": round(cos_vc_qc, 6),
                "cos_WR_vc": round(cos_WR_vc, 6),
                "mean_D_true": round(float(np.mean(D_test)), 2),
                "mean_D_pred": round(float(np.mean(D_pred)), 2),
            }

            sig = "✅" if r2_D>0 else "❌"
            print(f"  {sig} L{layer:>3}: R²_D={r2_D:+.3f} R²_T={r2_T:+.3f} R²_C={r2_C:+.3f} "
                  f"R²_vec={r2_vec:+.2f} cosWR→qc={cos_WR_qc:+.3f} cosvc→qc={cos_vc_qc:+.3f}")

        results[cat_name] = layer_results

    summary = {"model": model_name, "L": L, "scan_layers": scan_layers, "categories": results}
    return summary


def main():
    if len(sys.argv)<2: sys.exit(1)
    mn = sys.argv[1]; rd = sys.argv[2] if len(sys.argv)>2 else "1"
    if mn not in MODEL_CONFIGS: sys.exit(1)

    print("="*70)
    print(f"Phase 506 R{rd}: Readout Transport — {mn}")
    print(f"Predicting <ans,qc>/D/T/C  |  train={TRAIN_N} test={TEST_N}")
    print("="*70)

    t0=time.time()
    model, tokenizer, device = load_bf16(mn)
    print(f"[load] {time.time()-t0:.0f}s")
    try:
        res = run_phase506(model, tokenizer, mn, device)
        if res is None: return

        print(f"\n{'='*70}\nRESULTS — {mn}\n{'='*70}")
        # Best per category
        for cat, lr in res["categories"].items():
            best = max(lr.items(), key=lambda x: x[1]["r2_D"])
            print(f"  {cat:10s}: best L={best[0]:>3} R²_D={best[1]['r2_D']:+.3f} "
                  f"R²_T={best[1]['r2_T']:+.3f} R²_C={best[1]['r2_C']:+.3f} "
                  f"cosWR→qc={best[1]['cos_WR_qc']:+.3f}")

        # Layer avg
        print(f"\n  Layer-level avg R²_D:")
        for l in res["scan_layers"]:
            avg = np.mean([res["categories"][c][str(l)]["r2_D"] for c in CATEGORIES])
            n_pos = sum(1 for c in CATEGORIES if res["categories"][c][str(l)]["r2_D"]>0)
            print(f"    L{l:>3}: avg R²_D={avg:+.3f} +cats={n_pos}/{len(CATEGORIES)}")

        out = OUTPUT_DIR/f"phase506_{mn}_r{rd}.json"
        with open(out,'w',encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out}")
    finally:
        release_model(model)
        print("  Released.")

if __name__=="__main__":
    main()
