"""
Phase 298c: Normalized Direction Causal Test
=============================================
Goal: Test whether NORMALIZED role directions have causal effectiveness
for DS7B (where raw directions failed).

Key Insight from Phase 298b: DS7B's "1D" was a norm artifact.
Normalized PCA shows dim50=2, so there IS structure, but raw PCA
missed it due to norm dominance.

This script tests:
1. Causal effect of normalized role direction
2. Causal effect of per-role-pair specific direction
3. Compare with Phase 298 raw direction results

Usage:
  python tests/glm5/phase298c_norm_causal.py qwen3
  python tests/glm5/phase298c_norm_causal.py glm4
  python tests/glm5/phase298c_norm_causal.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase298_role_subspace")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass

# Stimuli (same as Phase 297)
def build_stimuli():
    stimuli = []
    adj_verb_tokens = {
        "open": {"adj": {"P1": ["the door is open", "the gate is open"], "P2": ["the door remains open", "the gate remains open"], "P3": ["the open door", "the open gate"], "P4": ["the shop seemed open", "the road seemed open"]},
                 "verb": {"P1": ["they open the door", "they open the gate"], "P2": ["we open the door", "we open the gate"], "P3": ["the door will open", "the gate will open"], "P4": ["they began to open the shop", "they began to open the road"]}},
        "clear": {"adj": {"P1": ["the path is clear", "the road is clear"], "P2": ["the path remains clear", "the road remains clear"], "P3": ["the clear path", "the clear road"], "P4": ["the desk seemed clear", "the table seemed clear"]},
                  "verb": {"P1": ["they clear the path", "they clear the road"], "P2": ["we clear the path", "we clear the road"], "P3": ["the path will clear", "the road will clear"], "P4": ["they began to clear the desk", "they began to clear the table"]}},
        "warm": {"adj": {"P1": ["the room is warm", "the house is warm"], "P2": ["the room remains warm", "the house remains warm"], "P3": ["the warm room", "the warm house"], "P4": ["the water seemed warm", "the food seemed warm"]},
                 "verb": {"P1": ["they warm the room", "they warm the house"], "P2": ["we warm the room", "we warm the house"], "P3": ["the room will warm", "the house will warm"], "P4": ["they began to warm the water", "they began to warm the food"]}},
        "clean": {"adj": {"P1": ["the floor is clean", "the table is clean"], "P2": ["the floor remains clean", "the table remains clean"], "P3": ["the clean floor", "the clean table"], "P4": ["the room seemed clean", "the house seemed clean"]},
                  "verb": {"P1": ["they clean the floor", "they clean the table"], "P2": ["we clean the floor", "we clean the table"], "P3": ["the floor will clean", "the table will clean"], "P4": ["they began to clean the room", "they began to clean the house"]}},
    }
    adj_noun_tokens = {
        "light": {"adj": {"P1": ["the bag is light", "the box is light"], "P2": ["the bag remains light", "the box remains light"], "P3": ["the light bag", "the light box"], "P4": ["the load seemed light", "the dress seemed light"]},
                  "noun": {"P1": ["the light is bright", "the light is warm"], "P2": ["that light is bright", "that light is warm"], "P3": ["near the light", "by the light"], "P4": ["they saw the light", "they found the light"]}},
        "cold": {"adj": {"P1": ["the water is cold", "the wind is cold"], "P2": ["the water remains cold", "the wind remains cold"], "P3": ["the cold water", "the cold wind"], "P4": ["the room seemed cold", "the air seemed cold"]},
                 "noun": {"P1": ["the cold is severe", "the cold is bitter"], "P2": ["that cold is severe", "that cold is bitter"], "P3": ["in the cold", "despite the cold"], "P4": ["they felt the cold", "they noticed the cold"]}},
    }
    noun_verb_tokens = {
        "fire": {"noun": {"P1": ["the fire is hot", "the fire is big"], "P2": ["that fire is hot", "that fire is big"], "P3": ["near the fire", "by the fire"], "P4": ["they saw the fire", "they started the fire"]},
                 "verb": {"P1": ["they fire the gun", "they fire the worker"], "P2": ["they will fire the gun", "they will fire the worker"], "P3": ["the gun will fire", "the engine will fire"], "P4": ["they began to fire the gun", "they began to fire the worker"]}},
        "record": {"noun": {"P1": ["the record is old", "the record is broken"], "P2": ["that record is old", "that record is broken"], "P3": ["on the record", "for the record"], "P4": ["they broke the record", "they set the record"]},
                   "verb": {"P1": ["they record music", "they record data"], "P2": ["they will record music", "they will record data"], "P3": ["the device will record", "the system will record"], "P4": ["they began to record music", "they began to record data"]}},
    }
    all_tokens = {}; all_tokens.update(adj_verb_tokens); all_tokens.update(adj_noun_tokens); all_tokens.update(noun_verb_tokens)
    for token, roles in all_tokens.items():
        rp = "adj_verb" if token in adj_verb_tokens else ("adj_noun" if token in adj_noun_tokens else "noun_verb")
        for role, pairs in roles.items():
            for pair_label, sentences in pairs.items():
                for sent in sentences:
                    stimuli.append({"sentence": sent, "target_word": token, "token_label": token,
                                    "role_label": role, "pair_label": pair_label, "role_pair": rp})
    return stimuli

def build_causal_stimuli():
    test_pairs = [
        ("the window is open", "open", "adj", "adj_verb"), ("they open the window", "open", "verb", "adj_verb"),
        ("the market is open", "open", "adj", "adj_verb"), ("they open the market", "open", "verb", "adj_verb"),
        ("the field is clear", "clear", "adj", "adj_verb"), ("they clear the field", "clear", "verb", "adj_verb"),
        ("the meal is warm", "warm", "adj", "adj_verb"), ("they warm the meal", "warm", "verb", "adj_verb"),
        ("the shirt is clean", "clean", "adj", "adj_verb"), ("they clean the shirt", "clean", "verb", "adj_verb"),
        ("the feather is light", "light", "adj", "adj_noun"), ("the light is on", "light", "noun", "adj_noun"),
        ("the drink is cold", "cold", "adj", "adj_noun"), ("the cold is harsh", "cold", "noun", "adj_noun"),
        ("the fire is bright", "fire", "noun", "noun_verb"), ("they fire the employee", "fire", "verb", "noun_verb"),
        ("the record is famous", "record", "noun", "noun_verb"), ("they record the song", "record", "verb", "noun_verb"),
    ]
    stimuli = []
    for sent, target, role, rp in test_pairs:
        stimuli.append({"sentence": sent, "target_word": target, "token_label": target,
                        "role_label": role, "pair_label": "test", "role_pair": rp, "group": "causal_test"})
    return stimuli

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            break
        except: pass
    if model is None: raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    log(f"  Loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

def _capture_single(model, tokenizer, sent, n_layers, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    return {"hidden": hs}

def _find_token_pos(decoded_tokens, target):
    target_lower = target.lower()
    for i, t in enumerate(decoded_tokens):
        if t == target_lower: return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower: return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower: return i
    return None

def resolve_positions(stimuli, tokenizer):
    resolved = []
    for stim in stimuli:
        toks = tokenizer.encode(stim["sentence"], add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, stim["target_word"])
        if pos is not None:
            new_stim = dict(stim); new_stim["target_pos"] = pos; resolved.append(new_stim)
    return resolved


def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase298c_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 298c: Normalized Direction Causal Test -- {model_name}")

    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers; d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")

    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)

    # Deduplicate and capture
    all_sentences = []; sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences); all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]

    log(f"Capturing {len(all_sentences)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent, nl)
        if (i + 1) % 20 == 0:
            el = time.time() - t0; rate = (i + 1) / max(el, 1)
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={(len(all_sentences)-i-1)/rate:.0f}s")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Done in {time.time()-t0:.0f}s")

    # ---- Extract role directions ----
    log(f"\n--- Extracting role directions (raw, normalized, per-role-pair) ---")
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set); token_pairs = defaultdict(set); token_rp = {}
    for stim in sub_stimuli:
        token_roles[stim["token_label"]].add(stim["role_label"])
        token_pairs[stim["token_label"]].add(stim["pair_label"])
        token_rp[stim["token_label"]] = stim.get("role_pair", "")
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])

    # For each layer, compute 3 types of direction:
    # 1. raw_mean_delta (same as Phase 298)
    # 2. norm_mean_delta (normalized increments, then averaged)
    # 3. per_rp_directions (per role-pair normalized mean)
    layers_obj = get_layers(model)
    input_device = next(model.parameters()).device

    sample_layers = sorted(set(
        list(range(0, nl, max(1, nl // 6))) + [nl - 2, nl - 1]
    ) & set(range(n_layers if (n_layers := nl) else nl)))

    # Organize causal test pairs
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]; role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim
    dual_test = [(t, sorted(rs.keys())) for t, rs in test_pairs.items() if len(rs) >= 2]

    results = {}
    for li in sample_layers:
        if li == 0: continue

        # Compute cell means
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Per-token pair-averaged deltas
        token_deltas = {}
        token_norm_deltas = {}  # normalized version
        token_rp_deltas = defaultdict(list)  # per role-pair
        token_rp_norm_deltas = defaultdict(list)

        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            r1_means = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
            r2_means = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
            r1_means = [m for m in r1_means if m is not None]
            r2_means = [m for m in r2_means if m is not None]
            if r1_means and r2_means:
                delta = np.mean(r2_means, axis=0) - np.mean(r1_means, axis=0)
                token_deltas[token] = delta
                nrm = np.linalg.norm(delta)
                if nrm > 1e-8:
                    token_norm_deltas[token] = delta / nrm
                rp = token_rp.get(token, "")
                token_rp_deltas[rp].append(delta)
                if nrm > 1e-8:
                    token_rp_norm_deltas[rp].append(delta / nrm)

        if len(token_deltas) < 2: continue

        # Compute 3 direction types
        raw_dir = np.mean(list(token_deltas.values()), axis=0)
        norm_dir = np.mean(list(token_norm_deltas.values()), axis=0)
        # Scale norm_dir to match raw_dir norm
        norm_dir_scaled = norm_dir * np.linalg.norm(raw_dir)

        # Per-role-pair directions
        rp_dirs = {}
        for rp, deltas in token_rp_deltas.items():
            if len(deltas) >= 1:
                rp_dirs[rp] = np.mean(deltas, axis=0)
        rp_norm_dirs = {}
        for rp, deltas in token_rp_norm_deltas.items():
            if len(deltas) >= 1:
                mean_norm = np.mean(deltas, axis=0)
                nrm = np.linalg.norm(mean_norm)
                if nrm > 1e-8:
                    # Scale to same norm as raw_dir
                    rp_norm_dirs[rp] = mean_norm / nrm * np.linalg.norm(raw_dir)

        # ---- Causal test ----
        layer_results = {}
        for token, roles_list in dual_test:
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            stim_r1 = test_pairs[token].get(r1)
            stim_r2 = test_pairs[token].get(r2)
            if not stim_r1 or not stim_r2: continue
            if stim_r1.get("target_pos") is None or stim_r2.get("target_pos") is None: continue

            pos_r1 = stim_r1["target_pos"]
            pos_r2 = stim_r2["target_pos"]
            sent_r1 = stim_r1["sentence"]
            sent_r2 = stim_r2["sentence"]

            # Baseline forward passes
            inputs_r1 = tok(sent_r1, return_tensors="pt", truncation=True, max_length=64)
            inputs_r1 = {k: v.to(input_device) for k, v in inputs_r1.items()}
            inputs_r2 = tok(sent_r2, return_tensors="pt", truncation=True, max_length=64)
            inputs_r2 = {k: v.to(input_device) for k, v in inputs_r2.items()}

            with torch.no_grad():
                out_r1 = model(**inputs_r1, output_hidden_states=True)
                out_r2 = model(**inputs_r2, output_hidden_states=True)

            logits_r1 = out_r1.logits[0, pos_r1].float().cpu().numpy()
            logits_r2 = out_r2.logits[0, pos_r2].float().cpu().numpy()

            rp = stim_r1.get("role_pair", "")

            # Test each direction type
            for dir_name, direction in [
                ("raw", raw_dir),
                ("norm_scaled", norm_dir_scaled),
                ("rp_specific", rp_dirs.get(rp, raw_dir)),
                ("rp_norm_scaled", rp_norm_dirs.get(rp, norm_dir_scaled)),
            ]:
                if direction is None: continue

                dir_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=input_device)

                # Add direction to r1 sentence
                intervened = [False]
                def make_hook(target_pos, dt, flag):
                    def hook_fn(module, input, output):
                        if not flag[0] and isinstance(output, tuple):
                            h = output[0].clone()
                            h[0, target_pos, :] += dt.to(dtype=h.dtype, device=h.device)
                            flag[0] = True
                            return (h,) + output[1:]
                        return output
                    return hook_fn

                hook = layers_obj[li].register_forward_hook(make_hook(pos_r1, dir_tensor, intervened))
                with torch.no_grad():
                    out_patched = model(**inputs_r1)
                hook.remove()

                logits_patched = out_patched.logits[0, pos_r1].float().cpu().numpy()

                # Measure shift
                cos_base_r2 = float(np.dot(logits_r1, logits_r2) / (np.linalg.norm(logits_r1) * np.linalg.norm(logits_r2) + 1e-10))
                cos_patch_r2 = float(np.dot(logits_patched, logits_r2) / (np.linalg.norm(logits_patched) * np.linalg.norm(logits_r2) + 1e-10))
                shift = cos_patch_r2 - cos_base_r2

                # Random control
                random_dir = np.random.randn(len(direction))
                random_dir = random_dir / np.linalg.norm(random_dir) * np.linalg.norm(direction)
                random_tensor = torch.tensor(random_dir, dtype=torch.bfloat16, device=input_device)

                intervened2 = [False]
                hook2 = layers_obj[li].register_forward_hook(make_hook(pos_r1, random_tensor, intervened2))
                with torch.no_grad():
                    out_random = model(**inputs_r1)
                hook2.remove()

                logits_random = out_random.logits[0, pos_r1].float().cpu().numpy()
                cos_random_r2 = float(np.dot(logits_random, logits_r2) / (np.linalg.norm(logits_random) * np.linalg.norm(logits_r2) + 1e-10))
                random_shift = cos_random_r2 - cos_base_r2

                key = f"{token}_{r1}->{r2}"
                if key not in layer_results:
                    layer_results[key] = {}
                layer_results[key][f"L{li}_{dir_name}"] = {
                    "cos_shift": round(shift, 6),
                    "random_shift": round(random_shift, 6),
                    "specificity": round(shift / max(abs(random_shift), 1e-10), 2),
                }

        results[li] = layer_results
        n_tested = sum(len(v) for v in layer_results.values())
        log(f"  L{li}: tested {n_tested} direction-pair combinations")

    # ---- Save and summarize ----
    log(f"\n--- Saving and Summarizing ---")

    def convert_keys(d):
        if isinstance(d, defaultdict): d = dict(d)
        if isinstance(d, dict): return {str(k): convert_keys(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return d.tolist()
        return d

    output = {
        "model": model_name, "n_layers": nl,
        "causal_test_normalized": convert_keys(results),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULT_DIR / f"{model_name}_norm_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # Summary
    log(f"\n{'='*60}")
    log(f"PHASE 298c SUMMARY -- {model_name}")
    log(f"{'='*60}")

    mid = nl // 2
    for dir_type in ["raw", "norm_scaled", "rp_specific", "rp_norm_scaled"]:
        cos_shifts = []
        random_shifts = []
        specifics = []
        for li_str, layer_res in results.items():
            li = int(li_str)
            if abs(li - mid) > 5 or li == 0: continue
            for key, dir_res in layer_res.items():
                for dir_key, r in dir_res.items():
                    if dir_type in dir_key:
                        cos_shifts.append(r["cos_shift"])
                        random_shifts.append(r["random_shift"])
                        specifics.append(r["specificity"])

        if cos_shifts:
            n_pos = sum(1 for s in cos_shifts if s > 0)
            log(f"\n  Direction type: {dir_type} (mid-layer +/-5):")
            log(f"    avg cos_shift: {np.mean(cos_shifts):+.6f}")
            log(f"    positive rate: {n_pos}/{len(cos_shifts)} = {n_pos/len(cos_shifts)*100:.0f}%")
            log(f"    avg specificity: {np.mean(specifics):.1f}x")

    release_model(model)
    gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 298c complete for {model_name}!")


if __name__ == "__main__":
    main()
