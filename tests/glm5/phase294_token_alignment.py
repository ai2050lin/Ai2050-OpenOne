"""
Phase 294: Token Alignment Fix
================================
Core fix: Patch by semantic role positions, not absolute positions.

Phase 293 bug: When A and B have different token lengths, position-specific
patching uses the SAME absolute position for source (B) and destination (A).
This causes misalignment: e.g., B[4]="not" gets patched into A[4]="happy",
or B[last] position doesn't exist in A's shorter sequence.

Phase 294 fix: Build token-level alignment mapping semantic roles to their
ACTUAL positions in both A and B. When patching, use B[src_pos] → A[dst_pos]
where src_pos and dst_pos refer to the SAME semantic role in different sequences.

Experiments:
  Exp A: B→A aligned (inject negation) — operand, last_token
  Exp B: B→A misaligned (Phase 293 replication) — for direct comparison
  Exp C: A→B aligned (remove negation) — operator, operand, last_token
  Exp D: Component contract — attn+MLP synergy at key layers

Usage:
  python tests/glm5/phase294_token_alignment.py qwen3
  python tests/glm5/phase294_token_alignment.py glm4
  python tests/glm5/phase294_token_alignment.py deepseek7b
"""
import sys, os, gc, time, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase294_token_alignment")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f: f.write(line + "\n")
        except: pass


# =====================================================================
# NEGATION PAIRS (same as Phase 293)
# =====================================================================
def build_negation_pairs():
    pairs = []
    # lexical_not_adj
    for n,p,ne,adj in [("happy","she is happy","she is not happy","happy"),
        ("open","the door is open","the door is not open","open"),
        ("possible","victory is possible","victory is not possible","possible"),
        ("ready","they are ready","they are not ready","ready"),
        ("important","this is important","this is not important","important"),
        ("clear","the answer is clear","the answer is not clear","clear"),
        ("safe","the area is safe","the area is not safe","safe"),
        ("fair","the decision is fair","the decision is not fair","fair"),
        ("simple","the problem is simple","the problem is not simple","simple"),
        ("correct","your answer is correct","your answer is not correct","correct")]:
        pairs.append({"name":f"adj_{n}","A":p,"B":ne,"subtype":"lexical_not_adj",
                      "op":"not","operand":adj})
    # syntactic_do_not
    for n,p,ne,v in [("agree","they agree with the proposal","they do not agree with the proposal","agree"),
        ("remember","i remember the meeting","i do not remember the meeting","remember"),
        ("understand","we understand the problem","we do not understand the problem","understand"),
        ("know","she knows the answer","she does not know the answer","knows"),
        ("believe","he believes the story","he does not believe the story","believes"),
        ("support","they support the plan","they do not support the plan","support"),
        ("accept","she accepts the offer","she does not accept the offer","accepts"),
        ("expect","we expect rain","we do not expect rain","expect"),
        ("trust","he trusts the source","he does not trust the source","trusts"),
        ("follow","they follow the rules","they do not follow the rules","follow")]:
        pairs.append({"name":f"verb_{n}","A":p,"B":ne,"subtype":"syntactic_do_not",
                      "op":"not","operand":v})
    # existential_no
    for n,p,ne in [("nothing","he found something interesting","he found nothing interesting"),
        ("no_one","someone came to the party","no one came to the party"),
        ("no_food","there was some food left","there was no food left"),
        ("no_idea","she had some idea what to do","she had no idea what to do"),
        ("no_reason","there is a reason to worry","there is no reason to worry"),
        ("no_choice","they had a choice in the matter","they had no choice in the matter"),
        ("no_doubt","there is some doubt about it","there is no doubt about it"),
        ("no_evidence","there is evidence of fraud","there is no evidence of fraud"),
        ("no_sign","there is a sign of life","there is no sign of life"),
        ("no_animal","an animal crossed the road","no animal crossed the road")]:
        pairs.append({"name":f"no_{n}","A":p,"B":ne,"subtype":"existential_no",
                      "op":"no","operand":n})
    # never
    for n,p,ne,v in [("seen","i have seen it before","i have never seen it before","seen"),
        ("been","she has been to Paris","she has never been to Paris","been"),
        ("told","he told someone the secret","he never told anyone the secret","told"),
        ("gives_up","she sometimes gives up","she never gives up","gives"),
        ("forgets","he sometimes forgets names","he never forgets a face","forgets"),
        ("complains","she often complains","she never complains","complains"),
        ("tells_truth","he sometimes tells the truth","he never tells the truth","tells"),
        ("late","she is sometimes late","she is never late","late"),
        ("apologizes","he sometimes apologizes","he never apologizes","apologizes"),
        ("admits","she sometimes admits mistakes","she never admits mistakes","admits")]:
        pairs.append({"name":f"never_{n}","A":p,"B":ne,"subtype":"never",
                      "op":"never","operand":v})
    # morphological_neg
    for n,p,ne,root in [("impossible","the task is possible","the task is impossible","possible"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable","acceptable"),
        ("incomplete","the report is complete","the report is incomplete","complete"),
        ("irrelevant","the comment is relevant","the comment is irrelevant","relevant"),
        ("dishonest","the person is honest","the person is dishonest","honest"),
        ("unfair","the treatment was fair","the treatment was unfair","fair"),
        ("unlikely","the outcome is likely","the outcome is unlikely","likely"),
        ("incorrect","the assumption is correct","the assumption is incorrect","correct"),
        ("uncertain","the result is certain","the result is uncertain","certain"),
        ("disobey","the soldiers obey orders","the soldiers disobey orders","obey")]:
        pairs.append({"name":f"prefix_{n}","A":p,"B":ne,"subtype":"morphological_neg",
                      "op":n,"operand":root})
    # scope_quantifier
    for n,p,ne,q in [("not_all","all birds can fly","not all birds can fly","all"),
        ("not_everyone","everyone agreed","not everyone agreed","everyone"),
        ("not_always","she always tells the truth","she does not always tell the truth","always"),
        ("not_entirely","the plan is entirely successful","the plan is not entirely successful","entirely"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness","necessarily"),
        ("not_exactly","that is exactly what i meant","that is not exactly what i meant","exactly"),
        ("not_quite","the work is finished","the work is not quite finished","quite"),
        ("not_completely","the glass is full","the glass is not completely full","completely"),
        ("not_if","she will come if invited","she will not come if invited","if"),
        ("not_any","there are some problems","there are not any problems","any")]:
        pairs.append({"name":f"scope_{n}","A":p,"B":ne,"subtype":"scope_quantifier",
                      "op":"not","operand":q})
    return pairs


# =====================================================================
# TOKEN ALIGNMENT — the core fix
# =====================================================================
def build_alignment(tokenizer, sent_a, sent_b, op_text, operand_text):
    """
    Build token-level semantic role alignment between A (positive) and B (negated).

    Key insight: When A and B have different lengths, the same semantic role
    (e.g., "operand") appears at DIFFERENT absolute positions. Phase 293
    mistakenly used the same absolute position for both source and destination.

    Returns dict mapping semantic roles to (pos_in_A, pos_in_B) pairs.
    """
    toks_a = tokenizer.encode(sent_a, add_special_tokens=True)
    toks_b = tokenizer.encode(sent_b, add_special_tokens=True)
    dec_a = [tokenizer.decode([t]).strip().lower() for t in toks_a]
    dec_b = [tokenizer.decode([t]).strip().lower() for t in toks_b]

    len_a, len_b = len(toks_a), len(toks_b)

    # Find common prefix (by exact token ID match)
    prefix_len = 0
    for i in range(min(len_a, len_b)):
        if toks_a[i] == toks_b[i]:
            prefix_len = i + 1
        else:
            break

    # Find common suffix (by exact token ID match, from end)
    suffix_len = 0
    max_suffix = min(len_a, len_b) - prefix_len  # don't overlap with prefix
    for i in range(1, max_suffix + 1):
        if toks_a[len_a - i] == toks_b[len_b - i]:
            suffix_len = i
        else:
            break

    # Find operator position in B
    operator_pos_b = None
    op_lower = op_text.lower()
    for i, t in enumerate(dec_b):
        if op_lower in t or t in op_lower:
            operator_pos_b = i
            break
    # Fallback: search by first 3 chars
    if operator_pos_b is None and len(op_lower) >= 2:
        for i, t in enumerate(dec_b):
            if op_lower[:3] in t or t[:3] in op_lower:
                operator_pos_b = i
                break

    # Find operand position in A and B
    operand_pos_a = None
    operand_pos_b = None
    opnd_lower = operand_text.lower()
    for i, t in enumerate(dec_a):
        if opnd_lower in t or t in opnd_lower:
            operand_pos_a = i
            break
    if operand_pos_a is None and len(opnd_lower) >= 2:
        for i, t in enumerate(dec_a):
            if opnd_lower[:3] in t:
                operand_pos_a = i
                break

    for i, t in enumerate(dec_b):
        if opnd_lower in t or t in opnd_lower:
            operand_pos_b = i
            break
    if operand_pos_b is None and len(opnd_lower) >= 2:
        for i, t in enumerate(dec_b):
            if opnd_lower[:3] in t:
                operand_pos_b = i
                break

    # For morphological_neg: operator and operand share the same position in B
    if operand_pos_b is None and operator_pos_b is not None:
        operand_pos_b = operator_pos_b

    # Last token positions (the token whose next-token prediction we care about)
    last_pos_a = len_a - 1
    last_pos_b = len_b - 1

    # For A→B operator patching: position in A corresponding to operator in B
    # This is the position just after the shared prefix (where "not" would go)
    operator_corresponding_a = prefix_len if prefix_len < len_a else None

    # Length difference: how many extra tokens B has vs A
    len_diff = len_b - len_a

    return {
        "toks_a": toks_a, "toks_b": toks_b,
        "dec_a": dec_a, "dec_b": dec_b,
        "len_a": len_a, "len_b": len_b,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "len_diff": len_diff,
        "operator_pos_b": operator_pos_b,
        "operand_pos_a": operand_pos_a,
        "operand_pos_b": operand_pos_b,
        "last_pos_a": last_pos_a,
        "last_pos_b": last_pos_b,
        "operator_corresponding_a": operator_corresponding_a,
        # Debug: show alignment for verification
        "debug": f"A={dec_a} B={dec_b} prefix={prefix_len} suffix={suffix_len} "
                 f"opB={operator_pos_b} opndA={operand_pos_a} opndB={operand_pos_b} "
                 f"lastA={last_pos_a} lastB={last_pos_b} diff={len_diff}",
    }


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_phase294(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto)...")

    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try flash_attention_2 → sdpa → eager (fallback chain)
    model = None
    used_attn = "eager"
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            used_attn = attn_impl
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:80]}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")

    model.eval()
    gpu = torch.cuda.memory_allocated() / 1e9
    log(f"  Loaded with attn={used_attn}, GPU={gpu:.1f}GB")

    layers = get_layers(model); nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU: {len(gpu_l)}{' ('+str(gpu_l[0])+'-'+str(gpu_l[-1])+')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' ('+str(cpu_l[0])+'-'+str(cpu_l[-1])+')' if cpu_l else ''}")

    return model, tok


# =====================================================================
# CAPTURE
# =====================================================================
def capture_all(model, tokenizer, sent, n_layers, max_len=64):
    """Capture attn_out, mlp_out, hidden_states for all layers."""
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    cap = {"attn": {}, "mlp": {}, "hidden": {}}
    hooks = []
    for li in range(n_layers):
        def mah(li):
            def hook(m, i, o):
                cap["attn"][li] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float().clone()
            return hook
        def mmh(li):
            def hook(m, i, o):
                cap["mlp"][li] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try:
            out = model(**inputs, output_hidden_states=True)
        except Exception as e:
            log(f"  capture ERR: {e}")
            cap["hidden"] = {}; cap["logits"] = None
        else:
            for li, hs in enumerate(out.hidden_states):
                cap["hidden"][li] = hs.detach().cpu().float().clone()
            cap["logits"] = out.logits[0, -1, :].detach().cpu().float().clone()
    for h in hooks: h.remove()
    return cap


# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(logits, la, lb, kab):
    """Compute NP = PROG/(1+KR). la=base, lb=target."""
    if logits is None:
        return None
    kp = float(F.kl_div(F.log_softmax(logits, -1), F.softmax(lb, -1), reduction='sum'))
    kr = min(kp / max(kab, 1e-6), 100.0)
    db = lb - la  # direction from base to target
    dp = logits - la  # direction from base to patched
    nb, np_ = float(torch.norm(db)), float(torch.norm(dp))
    cd = float(torch.dot(dp, db) / (nb * np_)) if nb > 1e-8 and np_ > 1e-8 else 0
    prog = cd * min(np_ / nb, 2.0) if nb > 1e-8 else 0
    np_metric = prog / (1.0 + kr)
    return {"kl_ratio": round(kr, 4), "progress": round(prog, 5),
            "cos_dir": round(cd, 5), "natural_prog": round(np_metric, 6)}


# =====================================================================
# ALIGNED PATCH HOOKS
# =====================================================================
def _aligned_pos_hook(pv_cpu, src_pos, dst_pos):
    """
    Hook: take pv[:, src_pos, :] and place at ref[:, dst_pos, :].
    This is the KEY FIX: src_pos is in B's token space, dst_pos is in A's.
    """
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            if src_pos < pv.shape[1] and dst_pos < no[0].shape[1]:
                no[0][:, dst_pos, :] = pv[:, src_pos, :]
            return no
        no = o.clone()
        if src_pos < pv.shape[1] and dst_pos < no.shape[1]:
            no[:, dst_pos, :] = pv[:, src_pos, :]
        return no
    return hook


def _aligned_multi_pos_hook(pv_cpu, src_dst_pairs):
    """Hook: patch multiple (src_pos, dst_pos) pairs at once."""
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            for sp, dp in src_dst_pairs:
                if sp < pv.shape[1] and dp < no[0].shape[1]:
                    no[0][:, dp, :] = pv[:, sp, :]
            return no
        no = o.clone()
        for sp, dp in src_dst_pairs:
            if sp < pv.shape[1] and dp < no.shape[1]:
                no[:, dp, :] = pv[:, sp, :]
        return no
    return hook


def forward_aligned_patch(model, tokenizer, sent_run, n_layers, max_len, patches):
    """
    Run model on sent_run with aligned position patches.

    patches: list of (module, pv_cpu, src_pos, dst_pos)
             or (module, pv_cpu, [(src_pos, dst_pos), ...])  for multi-position
    """
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent_run, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    hooks = []
    for patch in patches:
        mod = patch[0]
        pv = patch[1]
        if len(patch) == 4:
            # Single position: (module, pv, src_pos, dst_pos)
            _, _, sp, dp = patch
            hooks.append(mod.register_forward_hook(_aligned_pos_hook(pv, sp, dp)))
        elif len(patch) == 3:
            # Multi position: (module, pv, [(src, dst), ...])
            _, _, pairs = patch
            hooks.append(mod.register_forward_hook(_aligned_multi_pos_hook(pv, pairs)))

    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    PATCH ERR: {e}")
        logits = None

    for h in hooks:
        h.remove()
    return logits


# =====================================================================
# EXPERIMENT A: B→A Aligned Position Patching (inject negation)
# =====================================================================
def run_exp_A(model, tok, pair_data, pair_metrics, pair_align, nl, max_len):
    """
    B→A direction: run on A (positive), inject B's (negated) activations.
    Aligned: B[operand_pos_B] → A[operand_pos_A], B[last_pos_B] → A[last_pos_A]
    Misaligned: B[pos_B] → A[pos_B] (Phase 293 bug — same absolute position)
    """
    log("\n=== EXP A: B→A Position Patching (aligned + misaligned) ===")
    layers = get_layers(model)
    results = []
    total = nl * 3 * 4 * len(pair_data)  # 3 comps × 4 roles
    done = 0; t0 = time.time()

    for li in range(nl):
        for comp in ['attn', 'mlp', 'resid_post']:
            for pn, pd in pair_data.items():
                pm = pair_metrics.get(pn)
                al = pair_align.get(pn)
                if not pm or not al: continue

                la, lb, kab = pm["la"], pm["lb"], pm["kab"]

                # Get B's activation for this layer×component
                if comp == 'attn':
                    pv = pd["B"]["attn"].get(li)
                    mod = layers[li].self_attn
                elif comp == 'mlp':
                    pv = pd["B"]["mlp"].get(li)
                    mod = layers[li].mlp
                elif comp == 'resid_post':
                    pv = pd["B"]["hidden"].get(li + 1)
                    mod = layers[li]
                if pv is None: continue

                # 1. ALIGNED operand: B[operand_pos_B] → A[operand_pos_A]
                if al["operand_pos_a"] is not None and al["operand_pos_b"] is not None:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(mod, pv, al["operand_pos_b"], al["operand_pos_a"])])
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "role": "operand_aligned",
                                        "dir": "B2A", "name": pn, "subtype": pm["subtype"], **m})

                # 2. ALIGNED last: B[last_pos_B] → A[last_pos_a]
                logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                    [(mod, pv, al["last_pos_b"], al["last_pos_a"])])
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({"layer": li, "component": comp, "role": "last_aligned",
                                    "dir": "B2A", "name": pn, "subtype": pm["subtype"], **m})

                # 3. MISALIGNED operand: B[operand_pos_B] → A[operand_pos_B]
                #    (Phase 293 bug: uses B's position as destination in A)
                if al["operand_pos_b"] is not None and al["operand_pos_b"] < al["len_a"]:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(mod, pv, al["operand_pos_b"], al["operand_pos_b"])])
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "role": "operand_misaligned",
                                        "dir": "B2A", "name": pn, "subtype": pm["subtype"], **m})

                # 4. MISALIGNED last: B[last_pos_B] → A[last_pos_B]
                #    (Phase 293 bug: uses B's last position as destination in A)
                if al["last_pos_b"] < al["len_a"]:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(mod, pv, al["last_pos_b"], al["last_pos_b"])])
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "role": "last_misaligned",
                                        "dir": "B2A", "name": pn, "subtype": pm["subtype"], **m})

                done += 1

        # Progress logging every 2 layers
        if (li + 1) % 2 == 0 or li == nl - 1:
            el = time.time() - t0; rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  ExpA L{li+1}/{nl} {done}/{total} ({rate:.1f}/s) ETA={eta:.0f}s")

    log(f"  Exp A: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# =====================================================================
# EXPERIMENT B: A→B Aligned Position Patching (remove negation)
# =====================================================================
def run_exp_B(model, tok, pair_data, pair_metrics, pair_align, nl, max_len):
    """
    A→B direction: run on B (negated), inject A's (positive) activations.
    This tests "removing" the negation signal.

    For A→B, la=B_logits (base), lb=A_logits (target).
    """
    log("\n=== EXP B: A→B Position Patching (remove negation) ===")
    layers = get_layers(model)
    results = []
    total = nl * 3 * 3 * len(pair_data)  # 3 comps × 3 roles
    done = 0; t0 = time.time()

    for li in range(nl):
        for comp in ['attn', 'mlp', 'resid_post']:
            for pn, pd in pair_data.items():
                pm = pair_metrics.get(pn)
                al = pair_align.get(pn)
                if not pm or not al: continue

                # For A→B: base=B, target=A
                la_b2a, lb_b2a, kab = pm["lb"], pm["la"], pm["kab"]

                # Get A's activation for this layer×component
                if comp == 'attn':
                    pv = pd["A"]["attn"].get(li)
                    mod = layers[li].self_attn
                elif comp == 'mlp':
                    pv = pd["A"]["mlp"].get(li)
                    mod = layers[li].mlp
                elif comp == 'resid_post':
                    pv = pd["A"]["hidden"].get(li + 1)
                    mod = layers[li]
                if pv is None: continue

                # 1. OPERATOR: A[operator_corresponding_a] → B[operator_pos_b]
                #    "Remove" the negation operator by restoring the pre-negation context
                if (al["operator_pos_b"] is not None and
                    al["operator_corresponding_a"] is not None):
                    logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                        [(mod, pv, al["operator_corresponding_a"], al["operator_pos_b"])])
                    m = compute_metrics(logits, la_b2a, lb_b2a, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "role": "operator",
                                        "dir": "A2B", "name": pn, "subtype": pm["subtype"], **m})

                # 2. OPERAND: A[operand_pos_a] → B[operand_pos_b]
                if al["operand_pos_a"] is not None and al["operand_pos_b"] is not None:
                    logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                        [(mod, pv, al["operand_pos_a"], al["operand_pos_b"])])
                    m = compute_metrics(logits, la_b2a, lb_b2a, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "role": "operand",
                                        "dir": "A2B", "name": pn, "subtype": pm["subtype"], **m})

                # 3. LAST: A[last_pos_a] → B[last_pos_b]
                logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                    [(mod, pv, al["last_pos_a"], al["last_pos_b"])])
                m = compute_metrics(logits, la_b2a, lb_b2a, kab)
                if m:
                    results.append({"layer": li, "component": comp, "role": "last",
                                    "dir": "A2B", "name": pn, "subtype": pm["subtype"], **m})

                done += 1

        if (li + 1) % 2 == 0 or li == nl - 1:
            el = time.time() - t0; rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  ExpB L{li+1}/{nl} {done}/{total} ({rate:.1f}/s) ETA={eta:.0f}s")

    log(f"  Exp B: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# =====================================================================
# EXPERIMENT C: Component Contract (attn+MLP synergy)
# =====================================================================
def run_exp_C(model, tok, pair_data, pair_metrics, pair_align, nl, max_len, key_layers):
    """
    Test component contracts: attn only, MLP only, attn+MLP together.
    Compute synergy = NP(attn+mlp) - NP(attn) - NP(mlp).

    Key layers: early (0-4), mid, late (last-4 to last).
    """
    log(f"\n=== EXP C: Component Contract ({len(key_layers)} key layers) ===")
    layers = get_layers(model)
    results = []
    configs = ['attn_only', 'mlp_only', 'attn_mlp']
    roles = ['operand', 'last']
    total = len(key_layers) * len(configs) * len(roles) * len(pair_data)
    done = 0; t0 = time.time()

    for li in key_layers:
        for role in roles:
            for cfg_name in configs:
                for pn, pd in pair_data.items():
                    pm = pair_metrics.get(pn)
                    al = pair_align.get(pn)
                    if not pm or not al: continue

                    la, lb, kab = pm["la"], pm["lb"], pm["kab"]

                    # Determine src/dst positions based on role
                    if role == "operand":
                        if al["operand_pos_a"] is None or al["operand_pos_b"] is None:
                            continue
                        src_pos = al["operand_pos_b"]
                        dst_pos = al["operand_pos_a"]
                    else:  # last
                        src_pos = al["last_pos_b"]
                        dst_pos = al["last_pos_a"]

                    patches = []

                    if cfg_name == 'attn_only':
                        pv_attn = pd["B"]["attn"].get(li)
                        if pv_attn is None: continue
                        patches.append((layers[li].self_attn, pv_attn, src_pos, dst_pos))

                    elif cfg_name == 'mlp_only':
                        pv_mlp = pd["B"]["mlp"].get(li)
                        if pv_mlp is None: continue
                        patches.append((layers[li].mlp, pv_mlp, src_pos, dst_pos))

                    elif cfg_name == 'attn_mlp':
                        pv_attn = pd["B"]["attn"].get(li)
                        pv_mlp = pd["B"]["mlp"].get(li)
                        if pv_attn is None or pv_mlp is None: continue
                        patches.append((layers[li].self_attn, pv_attn, src_pos, dst_pos))
                        patches.append((layers[li].mlp, pv_mlp, src_pos, dst_pos))

                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len, patches)
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": cfg_name, "role": role,
                                        "dir": "B2A", "name": pn, "subtype": pm["subtype"], **m})

                    done += 1

        if (li + 1) % 2 == 0 or li == key_layers[-1]:
            el = time.time() - t0; rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  ExpC L{li+1} {done}/{total} ({rate:.1f}/s) ETA={eta:.0f}s")

    log(f"  Exp C: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# =====================================================================
# ANALYSIS
# =====================================================================
def analyze_results(resA, resB, resC, pair_align, nl, model_name):
    """Compute summary statistics and key comparisons."""
    log("\n=== ANALYSIS ===")

    # ---- A: Aligned vs Misaligned comparison ----
    log("\n  --- Aligned vs Misaligned (B→A, resid_post) ---")
    for role_aligned, role_mis in [("operand_aligned", "operand_misaligned"),
                                    ("last_aligned", "last_misaligned")]:
        aligned_nps = defaultdict(list)
        mis_nps = defaultdict(list)
        for r in resA:
            if r["component"] == "resid_post" and r["role"] == role_aligned:
                aligned_nps[r["layer"]].append(r["natural_prog"])
            elif r["component"] == "resid_post" and r["role"] == role_mis:
                mis_nps[r["layer"]].append(r["natural_prog"])

        log(f"\n  {role_aligned} vs {role_mis}:")
        for li in range(nl):
            a_vals = aligned_nps.get(li, [])
            m_vals = mis_nps.get(li, [])
            if a_vals and m_vals:
                a_mean = np.mean(a_vals)
                m_mean = np.mean(m_vals)
                diff = a_mean - m_mean
                if abs(diff) > 0.005 or li < 5 or li >= nl - 3:
                    log(f"    L{li}: aligned={a_mean:.5f} mis={m_mean:.5f} diff={diff:+.5f}")

    # ---- B: Last layer focus ----
    log("\n  --- Last Layer Focus (resid_post) ---")
    for li in [0, 1, nl - 3, nl - 2, nl - 1]:
        for role in ["operand_aligned", "last_aligned", "operand_misaligned", "last_misaligned"]:
            vals = [r["natural_prog"] for r in resA
                    if r["component"] == "resid_post" and r["layer"] == li and r["role"] == role]
            if vals:
                log(f"    L{li} {role}: NP={np.mean(vals):.5f} (n={len(vals)})")

    # ---- C: A→B operator effect ----
    log("\n  --- A→B Operator Effect ---")
    for comp in ['attn', 'mlp', 'resid_post']:
        comp_nps = defaultdict(list)
        for r in resB:
            if r["component"] == comp and r["role"] == "operator":
                comp_nps[r["layer"]].append(r["natural_prog"])
        # Find best layer
        if comp_nps:
            best_li = max(comp_nps, key=lambda l: np.mean(comp_nps[l]))
            log(f"    {comp}: best L{best_li} NP={np.mean(comp_nps[best_li]):.5f}")

    # ---- D: Component synergy ----
    log("\n  --- Component Synergy (attn+mlp vs attn+mlp separate) ---")
    for role in ['operand', 'last']:
        for li in set(r["layer"] for r in resC):
            attn_np = [r["natural_prog"] for r in resC
                       if r["layer"] == li and r["component"] == "attn_only" and r["role"] == role]
            mlp_np = [r["natural_prog"] for r in resC
                      if r["layer"] == li and r["component"] == "mlp_only" and r["role"] == role]
            both_np = [r["natural_prog"] for r in resC
                       if r["layer"] == li and r["component"] == "attn_mlp" and r["role"] == role]
            if attn_np and mlp_np and both_np:
                synergy = np.mean(both_np) - np.mean(attn_np) - np.mean(mlp_np)
                if abs(synergy) > 0.005 or li < 3 or li >= nl - 3:
                    log(f"    L{li} {role}: attn={np.mean(attn_np):.5f} mlp={np.mean(mlp_np):.5f} "
                        f"both={np.mean(both_np):.5f} synergy={synergy:+.5f}")

    # ---- E: Subtype breakdown ----
    log("\n  --- Subtype Breakdown (B→A, resid_post, L0, operand_aligned) ---")
    subtype_nps = defaultdict(list)
    for r in resA:
        if r["component"] == "resid_post" and r["layer"] == 0 and r["role"] == "operand_aligned":
            subtype_nps[r["subtype"]].append(r["natural_prog"])
    for st, nps in sorted(subtype_nps.items()):
        log(f"    {st}: NP={np.mean(nps):.5f} (n={len(nps)})")

    # ---- F: Alignment statistics ----
    log("\n  --- Alignment Statistics ---")
    len_diffs = [al["len_diff"] for al in pair_align.values()]
    op_valid = sum(1 for al in pair_align.values() if al["operator_pos_b"] is not None)
    opnd_valid = sum(1 for al in pair_align.values()
                     if al["operand_pos_a"] is not None and al["operand_pos_b"] is not None)
    same_pos_count = sum(1 for al in pair_align.values()
                         if al["operand_pos_a"] == al["operand_pos_b"]
                         and al["operand_pos_a"] is not None)
    log(f"    Pairs: {len(pair_align)}")
    log(f"    Len diff: mean={np.mean(len_diffs):.1f}, range=[{min(len_diffs)}, {max(len_diffs)}]")
    log(f"    Operator pos valid: {op_valid}/{len(pair_align)}")
    log(f"    Operand pos valid: {opnd_valid}/{len(pair_align)}")
    log(f"    Same operand pos in A&B: {same_pos_count}/{opnd_valid}")


# =====================================================================
# MAIN
# =====================================================================
def run_phase294(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase294_{model_name}.txt")
    log(f"{'='*60}")
    log(f"Phase 294: Token Alignment Fix — {model_name}")
    log(f"{'='*60}")

    t0_total = time.time()

    # 1. Load model
    model, tok = load_model_phase294(model_name)
    info = get_model_info(model, model_name); nl = info.n_layers
    log(f"Model: {info.model_class}, L={nl}, d={info.d_model}")

    # Warmup
    input_device = next(model.parameters()).device
    with torch.no_grad():
        try: model(**tok("warmup", return_tensors="pt").to(input_device).data)
        except: pass

    # 2. Build pairs
    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)} negation (6 subtypes)")

    MAX_LEN = 64

    # 3. Capture + Build alignment
    log("\n=== CAPTURE + ALIGNMENT ===")
    t0_cap = time.time()
    pair_data = {}
    pair_metrics = {}
    pair_align = {}

    for pi, pr in enumerate(pairs):
        pn, sa, sb, st = pr["name"], pr["A"], pr["B"], pr["subtype"]

        cap_a = capture_all(model, tok, sa, nl, MAX_LEN)
        cap_b = capture_all(model, tok, sb, nl, MAX_LEN)
        pair_data[pn] = {"A": cap_a, "B": cap_b, "subtype": st}

        la = cap_a.get("logits")
        lb = cap_b.get("logits")
        if la is None or lb is None:
            log(f"  SKIP {pn}: no logits")
            continue
        kab = float(F.kl_div(F.log_softmax(la, -1), F.softmax(lb, -1), reduction='sum'))

        # Build alignment
        al = build_alignment(tok, sa, sb, pr["op"], pr["operand"])
        pair_align[pn] = al

        pair_metrics[pn] = {"la": la, "lb": lb, "kab": kab,
                            "sa": sa, "sb": sb, "subtype": st}

        # Log first 3 pairs for verification
        if pi < 3:
            log(f"  [{pn}] {al['debug']}")

        # Free GPU memory from capture
        if (pi + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()
            log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0_cap:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    log(f"  Capture+Align: {len(pair_metrics)} pairs, {time.time()-t0_cap:.0f}s")

    # Print alignment summary
    log("\n  Alignment summary:")
    for st in ["lexical_not_adj", "syntactic_do_not", "existential_no",
               "never", "morphological_neg", "scope_quantifier"]:
        st_pairs = [(pn, al) for pn, al in pair_align.items()
                    if pair_metrics.get(pn, {}).get("subtype") == st]
        if st_pairs:
            mean_diff = np.mean([al["len_diff"] for _, al in st_pairs])
            same_pos = sum(1 for _, al in st_pairs
                          if al["operand_pos_a"] == al["operand_pos_b"]
                          and al["operand_pos_a"] is not None)
            log(f"    {st}: n={len(st_pairs)}, mean_len_diff={mean_diff:.1f}, "
                f"same_operand_pos={same_pos}/{len(st_pairs)}")

    # 4. Run experiments
    # Exp A: B→A aligned + misaligned
    resA = run_exp_A(model, tok, pair_data, pair_metrics, pair_align, nl, MAX_LEN)

    # Exp B: A→B aligned
    resB = run_exp_B(model, tok, pair_data, pair_metrics, pair_align, nl, MAX_LEN)

    # Exp C: Component contract at key layers
    # Key layers: early (0-4), every 4th mid, late (last-4 to last)
    key_layers = sorted(set(
        list(range(min(5, nl))) +                          # early: 0-4
        list(range(5, nl - 4, 4)) +                        # mid: every 4th
        list(range(max(nl - 4, 5), nl))                    # late: last-3 to last
    ))
    log(f"\n  Key layers for Exp C: {key_layers}")
    resC = run_exp_C(model, tok, pair_data, pair_metrics, pair_align, nl, MAX_LEN, key_layers)

    # 5. Analysis
    analyze_results(resA, resB, resC, pair_align, nl, model_name)

    # 6. Save results
    out_path = RESULT_DIR / f"{model_name}_alignment.json"

    # Aggregate results
    def aggregate(results, group_keys):
        agg = defaultdict(list)
        for r in results:
            key = tuple(r[k] for k in group_keys)
            agg[key].append(r["natural_prog"])
        return {",".join(str(k) for k in key): {
            "mean_np": round(float(np.mean(vals)), 6),
            "std_np": round(float(np.std(vals)), 6),
            "n": len(vals)
        } for key, vals in sorted(agg.items())}

    save_data = {
        "model": model_name, "n_layers": nl, "n_pairs": len(pair_metrics),
        "key_layers": key_layers,
        "exp_A_count": len(resA),
        "exp_B_count": len(resB),
        "exp_C_count": len(resC),
        # Aggregated summaries
        "exp_A_summary": aggregate(resA, ["layer", "component", "role"]),
        "exp_B_summary": aggregate(resB, ["layer", "component", "role"]),
        "exp_C_summary": aggregate(resC, ["layer", "component", "role"]),
        # Raw results (sample: first 200 per experiment for storage efficiency)
        "exp_A_raw": resA[:500],
        "exp_B_raw": resB[:500],
        "exp_C_raw": resC[:500],
        # Alignment info
        "alignment_info": {pn: {
            "len_a": al["len_a"], "len_b": al["len_b"], "len_diff": al["len_diff"],
            "prefix_len": al["prefix_len"], "suffix_len": al["suffix_len"],
            "operator_pos_b": al["operator_pos_b"],
            "operand_pos_a": al["operand_pos_a"], "operand_pos_b": al["operand_pos_b"],
            "last_pos_a": al["last_pos_a"], "last_pos_b": al["last_pos_b"],
            "dec_a": al["dec_a"], "dec_b": al["dec_b"],
        } for pn, al in pair_align.items()},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)
    log(f"\n  SAVED: {out_path}")

    # 7. Release model
    release_model(model); del model; gc.collect(); torch.cuda.empty_cache()

    total_time = time.time() - t0_total
    log(f"\nPhase 294 {model_name} DONE. Total: {total_time:.0f}s ({total_time/60:.1f}min)")
    return save_data


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            run_phase294(name)
            gc.collect(); torch.cuda.empty_cache(); time.sleep(5)
    else:
        run_phase294(model_name)


if __name__ == "__main__":
    main()
