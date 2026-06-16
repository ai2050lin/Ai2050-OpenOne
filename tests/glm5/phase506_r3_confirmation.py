"""
Phase 506 R3: 确认测试 — Tokenizer差异 + R²_D_qc稳健性
=====================================================
1. 检查三模型的tokenizer如何处理类别词
2. 用更多数据(20train/10test)验证R²_D_qc正值的稳健性
3. 去除类别词污染: 用 "belongs to the same group as apple" 替代 "is a type of fruit"

Usage:
  python tests/glm5/phase506_r3_confirmation.py qwen3
  python tests/glm5/phase506_r3_confirmation.py glm4
  python tests/glm5/phase506_r3_confirmation.py deepseek7b
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
TRAIN_N = 20; TEST_N = 10  # 增大数据量

CATEGORIES = {
    "fruit": {
        "objects": ["apple","banana","orange","grape","pear","peach","mango","plum",
                    "cherry","lemon","apricot","kiwi","pineapple","melon","coconut","lime",
                    "fig","pomegranate","papaya","avocado","strawberry","blueberry",
                    "raspberry","blackberry","tangerine","watermelon","guava","lychee",
                    "persimmon","nectarine"],
        "relation": "is a type of fruit",
        "clean_relation": "belongs to the same category as apple",  # 无类别词
    },
    "animal": {
        "objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
                    "rabbit","whale","lion","bear","fox","wolf","deer","monkey",
                    "shark","frog","penguin","owl","giraffe","zebra","parrot","turtle",
                    "flamingo","otter","cheetah","gorilla","hedgehog","pelican"],
        "relation": "is a type of animal",
        "clean_relation": "belongs to the same category as dog",
    },
    "action": {
        "objects": ["run","eat","build","throw","buy","learn","measure","communicate",
                    "swim","write","sing","draw","fly","climb","teach","drive",
                    "cook","dance","fight","sleep","read","listen","paint","explore",
                    "analyze","create","discover","observe","practice","investigate"],
        "relation": "is a type of action",
        "clean_relation": "belongs to the same category as run",
    },
}
ALL_CLASS = list(CATEGORIES.keys())


def load_bf16_auto(name):
    cfg = MODEL_CONFIGS[name]
    print(f"[load] Loading {name} (bf16 + auto + sdpa)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    m.eval()
    dev = next(m.parameters()).device
    gmem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[load] {name}: mem={gmem:.1f}GB class={type(m).__name__} ({time.time()-t0:.0f}s)")
    return m, tok, dev


def get_norm_g(model, name):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj, p, None)
            if obj is None: break
        if obj and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta': return w.float().cpu().numpy()
    cfg = MODEL_CONFIGS[name]
    for sf in sorted(Path(cfg["path"]).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if 'norm' in key.lower() and 'weight' in key.lower() and not any(x in key for x in ['layer','input','post']):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_token_ids(tokenizer, words):
    ids = []
    for w in words:
        tid = tokenizer.encode(w, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def forward_with_all_hidden(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states
    n_layers = len(hs)
    result = {}
    for l in range(n_layers):
        seq_len = hs[l].shape[1]
        ans = hs[l][0, -1, :].float().cpu().numpy().astype(np.float64)
        pre = hs[l][0, -2, :].float().cpu().numpy().astype(np.float64) if seq_len >= 2 else ans.copy()
        result[l] = {"ans": ans, "pre": pre}
    return result


def ridge_map(X, Y, ridge=0.1):
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), X.T @ Y).astype(np.float64)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res/(ss_tot+1e-10))


def rms_norm(vec):
    return float(np.sqrt(np.mean(vec**2)))


def run_phase506_r3(model, tokenizer, model_name, device):
    info = get_model_info(model, model_name)
    L = info.n_layers
    d = info.d_model
    print(f"[info] L={L}, d={d}")

    # === Part 1: Tokenizer差异分析 ===
    print(f"\n{'='*60}")
    print("Part 1: Tokenizer差异 — 类别词tokenization")
    print(f"{'='*60}")

    for cat in CATEGORIES:
        tokens = tokenizer.encode(cat, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  {cat}: ids={tokens}, decoded={decoded}, n_tokens={len(tokens)}")

    # 测试两种prompt的tokenization
    for cat, cfg in CATEGORIES.items():
        obj = cfg["objects"][0]
        rich = f"The {obj} {cfg['relation']}"
        clean = f"The {obj} {cfg['clean_relation']}"
        neutral = f"The {obj} is a thing"

        rich_ids = tokenizer.encode(rich, add_special_tokens=False)
        clean_ids = tokenizer.encode(clean, add_special_tokens=False)
        neutral_ids = tokenizer.encode(neutral, add_special_tokens=False)

        print(f"  {cat}: rich={len(rich_ids)}tok, clean={len(clean_ids)}tok, neutral={len(neutral_ids)}tok")

    # === Part 2: L0 cos(vc,qc) with clean prompt (no category word) ===
    print(f"\n{'='*60}")
    print("Part 2: L0 cos(vc,qc) — rich vs clean vs neutral")
    print(f"{'='*60}")

    W_U = get_W_U(model, model_name).astype(np.float64)
    g = get_norm_g(model, model_name)
    if g is None: return None
    g = g.astype(np.float64)

    cat_meta = {}
    for cat_name, cfg in CATEGORIES.items():
        target_ids = get_token_ids(tokenizer, [cat_name])
        other_cats = [c for c in ALL_CLASS if c != cat_name]
        competitor_ids = get_token_ids(tokenizer, other_cats)
        wDt = np.mean([W_U[i] for i in target_ids if i < len(W_U)], axis=0) if target_ids else np.zeros(d)
        wDc = np.mean([W_U[i] for i in competitor_ids if i < len(W_U)], axis=0) if competitor_ids else np.zeros(d)
        qc = (wDt - wDc) * g
        cat_meta[cat_name] = {"target_ids": target_ids, "competitor_ids": competitor_ids, "qc": qc}

    # Compare three prompt types at L0 and L-1
    prompt_comparison = {}
    for cat_name, cfg in CATEGORIES.items():
        obj = cfg["objects"][0]
        qc = cat_meta[cat_name]["qc"]
        qcn = np.linalg.norm(qc)

        # Rich prompt (has category word)
        rich_states = forward_with_all_hidden(model, tokenizer,
            f"The {obj} {cfg['relation']}", device)
        # Clean prompt (no category word)
        clean_states = forward_with_all_hidden(model, tokenizer,
            f"The {obj} {cfg['clean_relation']}", device)
        # Neutral prompt
        neutral_states = forward_with_all_hidden(model, tokenizer,
            f"The {obj} is a thing", device)

        for l_key, l in [("L0", 0), ("L-1", L)]:
            h_rich = rich_states[l]["ans"]
            h_clean = clean_states[l]["ans"]
            h_neutral = neutral_states[l]["ans"]

            vc_rich = h_rich - h_neutral
            vc_clean = h_clean - h_neutral

            cos_rich = float(np.dot(vc_rich, qc) / (np.linalg.norm(vc_rich) * qcn + 1e-10))
            cos_clean = float(np.dot(vc_clean, qc) / (np.linalg.norm(vc_clean) * qcn + 1e-10))
            norm_rich = float(np.linalg.norm(vc_rich))
            norm_clean = float(np.linalg.norm(vc_clean))

            key = f"{cat_name}_{l_key}"
            prompt_comparison[key] = {
                "cos_vc_qc_rich": round(cos_rich, 6),
                "cos_vc_qc_clean": round(cos_clean, 6),
                "vc_norm_rich": round(norm_rich, 2),
                "vc_norm_clean": round(norm_clean, 2),
            }
            print(f"  {cat_name} {l_key}: cos(rich)={cos_rich:+.4f} cos(clean)={cos_clean:+.4f} "
                  f"|vc_rich|={norm_rich:.1f} |vc_clean|={norm_clean:.1f}")

    # === Part 3: Ridge R²_D_qc with more data ===
    print(f"\n{'='*60}")
    print(f"Part 3: Ridge R²_D_qc — TRAIN={TRAIN_N} TEST={TEST_N}")
    print(f"{'='*60}")

    # Key layers from R2
    ridge_layers = sorted(set([0, L//4, L//3, L//2, 2*L//3, L-5, L-3, L-1]))
    ridge_layers = [min(l, L) for l in ridge_layers]
    print(f"  Layers: {ridge_layers}")

    ridge_results = {}
    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        objs = cfg["objects"]
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        qc = cat_meta[cat_name]["qc"]

        train_objs = objs[:TRAIN_N]
        test_objs = objs[TRAIN_N:TRAIN_N+TEST_N]

        # Collect data
        train_data = {l: {"pre": [], "ans": []} for l in ridge_layers}
        test_data = {l: {"pre": [], "ans_rich": [], "ans_neutral": []} for l in ridge_layers}

        for obj in train_objs:
            states = forward_with_all_hidden(model, tokenizer,
                f"The {obj} {cfg['relation']}", device)
            for l in ridge_layers:
                if l < len(states):
                    train_data[l]["pre"].append(states[l]["pre"])
                    train_data[l]["ans"].append(states[l]["ans"])

        for obj in test_objs:
            rich_states = forward_with_all_hidden(model, tokenizer,
                f"The {obj} {cfg['relation']}", device)
            neutral_states = forward_with_all_hidden(model, tokenizer,
                f"The {obj} is a thing", device)
            for l in ridge_layers:
                if l < len(rich_states):
                    test_data[l]["pre"].append(rich_states[l]["pre"])
                    test_data[l]["ans_rich"].append(rich_states[l]["ans"])
                    test_data[l]["ans_neutral"].append(neutral_states[l]["ans"])

        cat_ridge = {}
        for l in ridge_layers:
            if len(train_data[l]["pre"]) < 2: continue

            X_train = np.array(train_data[l]["pre"])
            Y_train = np.array(train_data[l]["ans"])
            W = ridge_map(X_train, Y_train, ridge=0.1)

            X_test = np.array(test_data[l]["pre"])
            Y_test = np.array(test_data[l]["ans_rich"])
            WR_test = np.array([W @ x for x in X_test])

            # R²_vector
            r2_vec = r2_score(Y_test.flatten(), WR_test.flatten())

            # R²_D_qc (predicting <ans, qc>)
            D_true_qc = np.array([np.dot(y, qc) for y in Y_test])
            D_pred_qc = np.array([np.dot(wr, qc) for wr in WR_test])
            r2_D_qc = r2_score(D_true_qc, D_pred_qc)

            # R²_D (predicting D = target_logit - competitor_logit from logits)
            D_true, D_pred = [], []
            for y, wr in zip(Y_test, WR_test):
                l_true = y @ W_U.T
                t_true = np.mean([l_true[i] for i in target_ids if i < len(l_true)])
                c_true = np.mean([l_true[i] for i in competitor_ids if i < len(l_true)])
                D_true.append(t_true - c_true)
                l_pred = wr @ W_U.T
                t_pred = np.mean([l_pred[i] for i in target_ids if i < len(l_pred)])
                c_pred = np.mean([l_pred[i] for i in competitor_ids if i < len(l_pred)])
                D_pred.append(t_pred - c_pred)
            r2_D = r2_score(np.array(D_true), np.array(D_pred))

            # vc and WR alignment
            vc_test = np.array(test_data[l]["ans_rich"]) - np.array(test_data[l]["ans_neutral"])
            vc_mean = np.mean(vc_test, axis=0)
            WR_mean = np.mean(WR_test, axis=0)
            qcn = np.linalg.norm(qc)
            vc_norm = np.linalg.norm(vc_mean)
            wr_norm = np.linalg.norm(WR_mean)
            cos_vc_qc = float(np.dot(vc_mean, qc)/(vc_norm*qcn+1e-10)) if vc_norm > 1e-10 else 0
            cos_WR_qc = float(np.dot(WR_mean, qc)/(wr_norm*qcn+1e-10)) if wr_norm > 1e-10 else 0

            cat_ridge[l] = {
                "r2_vector": round(r2_vec, 4),
                "r2_D": round(r2_D, 4),
                "r2_D_qc": round(r2_D_qc, 4),
                "cos_vc_qc": round(cos_vc_qc, 6),
                "cos_WR_qc": round(cos_WR_qc, 6),
                "n_train": len(train_data[l]["pre"]),
                "n_test": len(test_data[l]["pre"]),
            }

            sig = "✓" if r2_D_qc > 0 else "✗"
            print(f"  {sig} L{l:>3}: R²_vec={r2_vec:+.3f} R²_D={r2_D:+.3f} R²_D_qc={r2_D_qc:+.3f} "
                  f"cos(vc→qc)={cos_vc_qc:+.4f} cos(WR→qc)={cos_WR_qc:+.4f} "
                  f"n={len(train_data[l]['pre'])}/{len(test_data[l]['pre'])}")

        ridge_results[cat_name] = cat_ridge

    return {
        "model": model_name, "L": L, "d_model": d,
        "train_n": TRAIN_N, "test_n": TEST_N,
        "tokenizer_analysis": {cat: {
            "token_ids": tokenizer.encode(cat, add_special_tokens=False),
            "decoded": [tokenizer.decode([t]) for t in tokenizer.encode(cat, add_special_tokens=False)],
        } for cat in CATEGORIES},
        "prompt_comparison": prompt_comparison,
        "ridge": ridge_results,
        "ridge_layers": ridge_layers,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/glm5/phase506_r3_confirmation.py <model_name>")
        sys.exit(1)
    mn = sys.argv[1]
    if mn not in MODEL_CONFIGS:
        print(f"Unknown model: {mn}")
        sys.exit(1)

    print("="*70)
    print(f"Phase 506 R3: Confirmation — {mn}")
    print(f"train={TRAIN_N} test={TEST_N} | bf16+auto+sdpa")
    print("="*70)

    t0 = time.time()
    model, tokenizer, device = load_bf16_auto(mn)
    try:
        results = run_phase506_r3(model, tokenizer, mn, device)
        if results is None: return

        # Summary
        print(f"\n{'='*70}")
        print(f"Phase 506 R3 Summary — {mn}")
        print(f"{'='*70}")

        # Tokenizer
        print("\n[Tokenizer] Category word tokenization:")
        for cat, info in results["tokenizer_analysis"].items():
            print(f"  {cat}: {info['decoded']} ({len(info['token_ids'])} tokens)")

        # Prompt comparison
        print("\n[Prompt] L0 cos(vc,qc): rich vs clean (no category word)")
        for cat in CATEGORIES:
            l0_rich = results["prompt_comparison"].get(f"{cat}_L0", {})
            l0_clean = results["prompt_comparison"].get(f"{cat}_L0", {})
            print(f"  {cat}: cos_rich={l0_rich.get('cos_vc_qc_rich','N/A'):+.4f} "
                  f"cos_clean={l0_clean.get('cos_vc_qc_clean','N/A'):+.4f} "
                  f"|vc_rich|={l0_rich.get('vc_norm_rich','N/A')} "
                  f"|vc_clean|={l0_clean.get('vc_norm_clean','N/A')}")

        # Ridge summary
        print("\n[Ridge] R²_D_qc with more data:")
        for cat, lr in results["ridge"].items():
            best = max(lr.items(), key=lambda x: x[1]["r2_D_qc"])
            print(f"  {cat}: best L{best[0]} R²_D_qc={best[1]['r2_D_qc']:+.3f} "
                  f"(n={best[1]['n_train']}/{best[1]['n_test']})")

        out = OUTPUT_DIR / f"phase506_r3_{mn}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)
        print(f"\nSaved: {out}")
        print(f"Total: {time.time()-t0:.0f}s")
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
