"""
Phase 292: Negation Contract Atlas
====================================
Four experiments in one script:

A. Sliding Block Heatmap: block_size=[1,2,4], stride=block_size
   → Corrected PROG (subtract α=0 baseline), Synergy

B. Position-Specific Patching:
   - operator_pos: only patch at negation word position (not/no/never/un- etc)
   - operand_pos: only patch at negated content position
   - last_pos: only patch at last token (prediction point)
   - all_pos: patch all positions (standard)

C. Component Decomposition (for best block from A):
   - attn_only: replace attention output only
   - mlp_only: replace MLP output only
   - both: replace both
   - resid_after: replace residual after entire block

D. Subtype Contract Clustering:
   For each subtype, compute contract_signature:
   [best_block, best_prog, best_kr, alpha_curve_shape, position_sensitivity, component_sensitivity]

DEEP LAYER FIX: output_t.device for dynamic device in hooks.

Usage:
  python tests/glm5/phase292_contract_atlas.py qwen3
  python tests/glm5/phase292_contract_atlas.py glm4
  python tests/glm5/phase292_contract_atlas.py deepseek7b
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

RESULT_DIR = Path("results/phase292_contract_atlas")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file,"a",encoding="utf-8") as f: f.write(line+"\n")
        except: pass


# ====== DATASET: Negation with position annotations ======
def build_negation_pairs():
    """Each pair has 'operator_pos' (index of negation word in B) and 'operand_pos'."""
    pairs = []
    # lexical_not_adj: "not" at position 2, adjective at end
    for name, pos, neg in [
        ("happy","she is happy","she is not happy"),
        ("open","the door is open","the door is not open"),
        ("possible","victory is possible","victory is not possible"),
        ("ready","they are ready","they are not ready"),
        ("important","this is important","this is not important"),
        ("clear","the answer is clear","the answer is not clear"),
        ("safe","the area is safe","the area is not safe"),
        ("fair","the decision is fair","the decision is not fair"),
        ("simple","the problem is simple","the problem is not simple"),
        ("correct","your answer is correct","your answer is not correct"),
    ]:
        pairs.append({"name":f"neg_adj_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"lexical_not_adj","op_word":"not","op_offset":2})
    # syntactic_do_not
    for name, pos, neg in [
        ("agree","they agree with the proposal","they do not agree with the proposal"),
        ("remember","i remember the meeting","i do not remember the meeting"),
        ("understand","we understand the problem","we do not understand the problem"),
        ("know","she knows the answer","she does not know the answer"),
        ("believe","he believes the story","he does not believe the story"),
        ("support","they support the plan","they do not support the plan"),
        ("accept","she accepts the offer","she does not accept the offer"),
        ("expect","we expect rain","we do not expect rain"),
        ("trust","he trusts the source","he does not trust the source"),
        ("follow","they follow the rules","they do not follow the rules"),
    ]:
        pairs.append({"name":f"neg_verb_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"syntactic_do_not","op_word":"not","op_offset":2})
    # existential_no
    for name, pos, neg in [
        ("nothing","he found something interesting","he found nothing interesting"),
        ("no_one","someone came to the party","no one came to the party"),
        ("no_food","there was some food left","there was no food left"),
        ("no_idea","she had some idea what to do","she had no idea what to do"),
        ("no_reason","there is a reason to worry","there is no reason to worry"),
        ("no_choice","they had a choice in the matter","they had no choice in the matter"),
        ("no_doubt","there is some doubt about it","there is no doubt about it"),
        ("no_evidence","there is evidence of fraud","there is no evidence of fraud"),
        ("no_sign","there is a sign of life","there is no sign of life"),
        ("no_animal","an animal crossed the road","no animal crossed the road"),
    ]:
        pairs.append({"name":f"neg_no_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"existential_no","op_word":"no","op_offset":0})
    # never
    for name, pos, neg in [
        ("seen","i have seen it before","i have never seen it before"),
        ("been","she has been to Paris","she has never been to Paris"),
        ("told","he told someone the secret","he never told anyone the secret"),
        ("gives_up","she sometimes gives up","she never gives up"),
        ("forgets","he sometimes forgets names","he never forgets a face"),
        ("complains","she often complains","she never complains"),
        ("tells_truth","he sometimes tells the truth","he never tells the truth"),
        ("late","she is sometimes late","she is never late"),
        ("apologizes","he sometimes apologizes","he never apologizes"),
        ("admits","she sometimes admits mistakes","she never admits mistakes"),
    ]:
        pairs.append({"name":f"neg_never_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"never","op_word":"never","op_offset":2})
    # morphological_neg
    for name, pos, neg, prefix in [
        ("impossible","the task is possible","the task is impossible","im"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable","un"),
        ("incomplete","the report is complete","the report is incomplete","in"),
        ("irrelevant","the comment is relevant","the comment is irrelevant","ir"),
        ("dishonest","the person is honest","the person is dishonest","dis"),
        ("unfair","the treatment was fair","the treatment was unfair","un"),
        ("unlikely","the outcome is likely","the outcome is unlikely","un"),
        ("incorrect","the assumption is correct","the assumption is incorrect","in"),
        ("uncertain","the result is certain","the result is uncertain","un"),
        ("disobey","the soldiers obey orders","the soldiers disobey orders","dis"),
    ]:
        pairs.append({"name":f"neg_prefix_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"morphological_neg","op_word":prefix,"op_offset":3})
    # scope_quantifier
    for name, pos, neg in [
        ("not_all","all birds can fly","not all birds can fly"),
        ("not_everyone","everyone agreed","not everyone agreed"),
        ("not_always","she always tells the truth","she does not always tell the truth"),
        ("not_entirely","the plan is entirely successful","the plan is not entirely successful"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness"),
        ("not_exactly","that is exactly what i meant","that is not exactly what i meant"),
        ("not_quite","the work is finished","the work is not quite finished"),
        ("not_completely","the glass is full","the glass is not completely full"),
        ("not_if","she will come if invited","she will not come if invited"),
        ("not_any","there are some problems","there are not any problems"),
    ]:
        pairs.append({"name":f"neg_scope_{name}","A":pos,"B":neg,"category":"negation",
                       "subtype":"scope_quantifier","op_word":"not","op_offset":0})
    return pairs


# ====== MODEL LOADING ======
def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto, eager)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["path"],torch_dtype=torch.bfloat16,
        device_map="auto",trust_remote_code=True,local_files_only=True,attn_implementation="eager")
    model.eval()
    gpu = torch.cuda.memory_allocated()/1e9
    log(f"  Loaded, GPU={gpu:.1f}GB")
    layers = get_layers(model); nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type=='cuda' else cpu_l).append(li)
    log(f"  GPU: {len(gpu_l)}{' ('+str(gpu_l[0])+'-'+str(gpu_l[-1])+')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' ('+str(cpu_l[0])+'-'+str(cpu_l[-1])+')' if cpu_l else ''}")
    return model, tok


# ====== CAPTURE with position tracking ======
def find_token_position(tokenizer, sentence, target_word):
    """Find the first token position of target_word in tokenized sentence."""
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    # Try to find the target word's token(s)
    for i, tid in enumerate(tokens):
        decoded = tokenizer.decode([tid]).strip().lower()
        if target_word.lower() in decoded:
            return i
    # Fallback: try partial match
    for i, tid in enumerate(tokens):
        decoded = tokenizer.decode([tid]).strip().lower()
        if any(c in decoded for c in target_word.lower()[:3]):
            return i
    return None  # not found

def capture_all(model, tokenizer, sent, n_layers, max_len=64):
    layers = get_layers(model); al = min(n_layers, len(layers))
    inputs = tokenizer(sent,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)
    cap={}; hooks=[]
    for li in range(al):
        def mah(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["attn"]=(o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        def mmh(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["mlp"]=(o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try: model(**inputs)
        except Exception as e: log(f"  capture ERR: {e}"); cap={}
    for h in hooks: h.remove()
    return cap


# ====== PATCHING FUNCTIONS ======

def _make_attn_patch_hook(pv_cpu):
    """Hook: replace attention output, position-aware if needed."""
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        mms = min(pv.shape[1], ref.shape[1])
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            no[0][:, :mms, :] = pv[:, :mms, :]
            return no
        no = o.clone()
        no[:, :mms, :] = pv[:, :mms, :]
        return no
    return hook

def _make_mlp_patch_hook(pv_cpu):
    """Hook: replace MLP output."""
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        mms = min(pv.shape[1], ref.shape[1])
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            no[0][:, :mms, :] = pv[:, :mms, :]
            return no
        no = o.clone()
        no[:, :mms, :] = pv[:, :mms, :]
        return no
    return hook

def _make_positional_patch_hook(pv_cpu, positions):
    """Hook: only patch specified positions, leave others as original."""
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        no = (o[0].clone(),) + o[1:] if isinstance(o, tuple) else o.clone()
        target = no[0] if isinstance(no, tuple) else no
        for p in positions:
            if p < min(pv.shape[1], target.shape[1]):
                target[:, p, :] = pv[:, p, :]
        if isinstance(no, tuple):
            return no
        return no
    return hook


def forward_patched(model, tokenizer, sent_a, n_layers, max_len,
                    attn_patches, mlp_patches, resid_patches=None):
    """
    Generic patched forward with deep layer fix.
    attn_patches/mlp_patches: {layer_idx: pv_cpu_tensor}
    resid_patches: {layer_idx: pv_cpu_tensor}  (replace entire residual after layer)
    Returns: logits or None
    """
    layers = get_layers(model)
    inputs = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, max_len-sl), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, max_len-sl), value=0)

    hooks = []
    # Attention patches
    for li, pv in attn_patches.items():
        if li >= n_layers: continue
        hooks.append(layers[li].self_attn.register_forward_hook(_make_attn_patch_hook(pv)))
    # MLP patches
    for li, pv in mlp_patches.items():
        if li >= n_layers: continue
        hooks.append(layers[li].mlp.register_forward_hook(_make_mlp_patch_hook(pv)))
    # Residual patches (replace full layer output)
    if resid_patches:
        for li, pv in resid_patches.items():
            if li >= n_layers: continue
            def rh(pv):
                def hook(m, i, o):
                    ref = o[0] if isinstance(o, tuple) else o
                    pv_d = pv.to(ref.device).to(ref.dtype)
                    mms = min(pv_d.shape[1], ref.shape[1])
                    if isinstance(o, tuple):
                        no = (o[0].clone(),) + o[1:]
                        no[0][:, :mms, :] = pv_d[:, :mms, :]
                        return no
                    no = o.clone()
                    no[:, :mms, :] = pv_d[:, :mms, :]
                    return no
                return hook
            hooks.append(layers[li].register_forward_hook(rh(pv)))

    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    PATCH ERR: {e}")
        logits = None
    for h in hooks: h.remove()
    return logits


def compute_metrics(logits, la, lb, kab):
    if logits is None: return None
    kp = float(F.kl_div(F.log_softmax(logits,-1), F.softmax(lb,-1), reduction='sum'))
    kr = min(kp / max(kab, 1e-6), 100.0)
    db, dp = lb - la, logits - la
    nb, np_ = float(torch.norm(db)), float(torch.norm(dp))
    cd = float(torch.dot(dp, db) / (nb * np_)) if nb > 1e-8 and np_ > 1e-8 else 0
    prog = cd * min(np_ / nb, 2.0) if nb > 1e-8 else 0
    return {"kl_ratio": kr, "progress": prog, "cos_dir": cd}


# ====== EXPERIMENT A: Sliding Block Heatmap ======
def run_exp_A(model, tok, pairs, pair_data, pair_metrics, nl, max_len):
    """Block sizes 1,2,4 with stride=block_size. Alpha=1.0 only (fast scan)."""
    log("\n=== EXPERIMENT A: Sliding Block Heatmap ===")
    layers = get_layers(model)
    results = []
    alpha = 1.0

    for bs in [1, 2, 4]:
        log(f"\n  Block size={bs}")
        stride = bs
        block_starts = list(range(0, nl, stride))

        for bstart in block_starts:
            bend = min(bstart + bs, nl)
            # For each pair
            for pn, pd_item in pair_data.items():
                pm = pair_metrics.get(pn)
                if not pm: continue
                la, lb, kab = pm["logits_a"], pm["logits_b"], pm["kl_ab"]
                subtype = pm["subtype"]

                attn_patches = {}
                for li in range(bstart, bend):
                    lk = f"L{li}"
                    aa = pd_item["A"].get(lk, {}).get("attn")
                    ab = pd_item["B"].get(lk, {}).get("attn")
                    if aa is None or ab is None: continue
                    ms = min(aa.shape[1], ab.shape[1], max_len)
                    pv_cpu = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
                    attn_patches[li] = pv_cpu

                if not attn_patches: continue

                logits = forward_patched(model, tok, pm["sent_a"], nl, max_len,
                                         attn_patches, {})
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({
                        "block_size": bs, "block_start": bstart,
                        "block_end": bend-1, "name": pn, "subtype": subtype,
                        **m
                    })

            if (bstart // stride) % 10 == 0:
                log(f"    bstart={bstart}, done so far: {len(results)}")

    log(f"  Exp A total results: {len(results)}")
    return results


# ====== EXPERIMENT B: Position-Specific Patching ======
def run_exp_B(model, tok, pairs, pair_data, pair_metrics, nl, max_len):
    """Patch only at operator/operand/last positions for best block."""
    log("\n=== EXPERIMENT B: Position-Specific Patching ===")
    layers = get_layers(model)
    results = []
    alpha = 1.0

    # Use best block from Phase 291 results (early block for Qwen3, etc.)
    # We'll test: block [0-3] and block at midpoint
    test_blocks = [(0, min(4, nl))]
    if nl > 8:
        mid = nl // 2
        test_blocks.append((mid - 2, mid + 2))
    if nl > 20:
        late = nl - 4
        test_blocks.append((late, nl))

    for bstart, bend in test_blocks:
        log(f"\n  Block [{bstart}-{bend-1}]")
        for pn, pd_item in pair_data.items():
            pm = pair_metrics.get(pn)
            if not pm: continue
            la, lb, kab = pm["logits_a"], pm["logits_b"], pm["kl_ab"]
            subtype = pm["subtype"]
            pair = next((p for p in pairs if p["name"] == pn), None)
            if not pair: continue

            sent_a = pm["sent_a"]
            sent_b = pair["B"]
            seq_len = pm["seq_len"]
            last_pos = seq_len - 1

            # Find operator position
            op_word = pair.get("op_word", "not")
            op_pos = find_token_position(tok, sent_b, op_word)
            if op_pos is None:
                op_pos = 2  # fallback

            # Operand position: right after operator
            operand_pos = op_pos + 1

            for pos_type, positions in [
                ("operator", [op_pos]),
                ("operand", [operand_pos]),
                ("last", [last_pos]),
                ("all", list(range(seq_len))),
            ]:
                # Build position-masked patches
                attn_patches = {}
                for li in range(bstart, bend):
                    lk = f"L{li}"
                    aa = pd_item["A"].get(lk, {}).get("attn")
                    ab = pd_item["B"].get(lk, {}).get("attn")
                    if aa is None or ab is None: continue
                    ms = min(aa.shape[1], ab.shape[1], max_len)
                    # Interpolate only at specified positions
                    pv_cpu = aa[:, :ms, :].float().clone()
                    for p in positions:
                        if p < ms:
                            pv_cpu[:, p, :] = (1-alpha)*aa[:, p, :].float() + alpha*ab[:, p, :].float()
                    attn_patches[li] = pv_cpu

                if not attn_patches: continue

                logits = forward_patched(model, tok, sent_a, nl, max_len,
                                         attn_patches, {})
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({
                        "block_start": bstart, "block_end": bend-1,
                        "pos_type": pos_type, "name": pn, "subtype": subtype,
                        "op_pos": op_pos, "last_pos": last_pos,
                        **m
                    })

    log(f"  Exp B total results: {len(results)}")
    return results


# ====== EXPERIMENT C: Component Decomposition ======
def run_exp_C(model, tok, pairs, pair_data, pair_metrics, nl, max_len):
    """For best block, test attn-only / mlp-only / both / resid-after-block."""
    log("\n=== EXPERIMENT C: Component Decomposition ===")
    layers = get_layers(model)
    results = []
    alpha = 1.0

    # Same test blocks as Exp B
    test_blocks = [(0, min(4, nl))]
    if nl > 8:
        mid = nl // 2
        test_blocks.append((mid - 2, mid + 2))
    if nl > 20:
        late = nl - 4
        test_blocks.append((late, nl))

    for bstart, bend in test_blocks:
        log(f"\n  Block [{bstart}-{bend-1}]")
        for pn, pd_item in pair_data.items():
            pm = pair_metrics.get(pn)
            if not pm: continue
            la, lb, kab = pm["logits_a"], pm["logits_b"], pm["kl_ab"]
            subtype = pm["subtype"]

            for comp_type in ["attn_only", "mlp_only", "both", "resid_after"]:
                attn_patches = {}
                mlp_patches = {}
                resid_patches = {}

                for li in range(bstart, bend):
                    lk = f"L{li}"
                    aa = pd_item["A"].get(lk, {}).get("attn")
                    ab = pd_item["B"].get(lk, {}).get("attn")
                    ma = pd_item["A"].get(lk, {}).get("mlp")
                    mb = pd_item["B"].get(lk, {}).get("mlp")

                    if comp_type == "attn_only":
                        if aa is not None and ab is not None:
                            ms = min(aa.shape[1], ab.shape[1], max_len)
                            attn_patches[li] = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
                    elif comp_type == "mlp_only":
                        if ma is not None and mb is not None:
                            ms = min(ma.shape[1], mb.shape[1], max_len)
                            mlp_patches[li] = (1-alpha)*ma[:,:ms,:].float() + alpha*mb[:,:ms,:].float()
                    elif comp_type == "both":
                        if aa is not None and ab is not None:
                            ms = min(aa.shape[1], ab.shape[1], max_len)
                            attn_patches[li] = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
                        if ma is not None and mb is not None:
                            ms = min(ma.shape[1], mb.shape[1], max_len)
                            mlp_patches[li] = (1-alpha)*ma[:,:ms,:].float() + alpha*mb[:,:ms,:].float()
                    elif comp_type == "resid_after":
                        # Resid after block = replace full layer output for last layer in block
                        if li == bend - 1:
                            # We need the residual after the entire block
                            # This is equivalent to replacing the entire block's contribution
                            # Approximate: replace the output of the last layer in the block
                            if aa is not None and ab is not None and ma is not None and mb is not None:
                                ms = min(aa.shape[1], ab.shape[1], max_len)
                                # resid_after = attn + mlp combined into the residual
                                # We replace both for all layers in block
                                attn_patches[li] = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
                                mlp_patches[li] = (1-alpha)*ma[:,:ms,:].float() + alpha*mb[:,:ms,:].float()
                                # Also add for all other layers in block
                        elif li < bend - 1:
                            if aa is not None and ab is not None:
                                ms = min(aa.shape[1], ab.shape[1], max_len)
                                attn_patches[li] = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
                            if ma is not None and mb is not None:
                                ms = min(ma.shape[1], mb.shape[1], max_len)
                                mlp_patches[li] = (1-alpha)*ma[:,:ms,:].float() + alpha*mb[:,:ms,:].float()

                if not attn_patches and not mlp_patches: continue

                logits = forward_patched(model, tok, pm["sent_a"], nl, max_len,
                                         attn_patches, mlp_patches, resid_patches)
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({
                        "block_start": bstart, "block_end": bend-1,
                        "component": comp_type, "name": pn, "subtype": subtype,
                        **m
                    })

    log(f"  Exp C total results: {len(results)}")
    return results


# ====== EXPERIMENT D: Subtype Contract Clustering ======
def run_exp_D(model, tok, pairs, pair_data, pair_metrics, nl, max_len):
    """Alpha interpolation for each subtype at best block."""
    log("\n=== EXPERIMENT D: Subtype α Curves ===")
    layers = get_layers(model)
    results = []
    alphas = [0, 0.25, 0.5, 0.75, 1.0]

    # Test at early block and deep block
    test_blocks = [(0, min(4, nl))]
    if nl > 20:
        late = nl - 4
        test_blocks.append((late, nl))

    for bstart, bend in test_blocks:
        log(f"\n  Block [{bstart}-{bend-1}]")
        for alpha in alphas:
            for pn, pd_item in pair_data.items():
                pm = pair_metrics.get(pn)
                if not pm: continue
                la, lb, kab = pm["logits_a"], pm["logits_b"], pm["kl_ab"]
                subtype = pm["subtype"]

                attn_patches = {}
                for li in range(bstart, bend):
                    lk = f"L{li}"
                    aa = pd_item["A"].get(lk, {}).get("attn")
                    ab = pd_item["B"].get(lk, {}).get("attn")
                    if aa is None or ab is None: continue
                    ms = min(aa.shape[1], ab.shape[1], max_len)
                    attn_patches[li] = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()

                if not attn_patches: continue

                logits = forward_patched(model, tok, pm["sent_a"], nl, max_len,
                                         attn_patches, {})
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({
                        "block_start": bstart, "block_end": bend-1,
                        "alpha": alpha, "name": pn, "subtype": subtype,
                        **m
                    })

            log(f"    α={alpha:.2f}: {len([r for r in results if r['alpha']==alpha])} results")

    log(f"  Exp D total results: {len(results)}")
    return results


# ====== MAIN ======
def run_phase292(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase292_{model_name}.txt")
    log(f"{'='*60}")
    log(f"Phase 292: Negation Contract Atlas — {model_name}")
    log(f"{'='*60}")

    model, tok = load_model(model_name)
    info = get_model_info(model, model_name); nl = info.n_layers
    log(f"Model: {info.model_class}, L={nl}, d={info.d_model}")
    with torch.no_grad():
        try: model(**tok("warmup",return_tensors="pt").to(DEV))
        except: pass

    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)} negation (6 subtypes)")
    for st in sorted(set(p["subtype"] for p in pairs)):
        n = sum(1 for p in pairs if p["subtype"]==st)
        log(f"  {st}: {n}")

    MAX_LEN = 64
    layers = get_layers(model)

    # ====== CAPTURE ======
    log("\n=== CAPTURE PHASE ===")
    t0 = time.time()
    pair_data = {}  # {name: {A: {L0: {attn:.., mlp:..}}, B: {..}, seq_len, subtype}}
    pair_metrics = {}  # {name: {logits_a, logits_b, kl_ab, sent_a, seq_len, subtype}}

    for pi, pr in enumerate(pairs):
        pn, sa, sb, st = pr["name"], pr["A"], pr["B"], pr["subtype"]
        toks_a = len(tok.encode(sa, add_special_tokens=True))
        toks_b = len(tok.encode(sb, add_special_tokens=True))
        cl = min(max(toks_a, toks_b), MAX_LEN)
        oa = capture_all(model, tok, sa, nl, cl)
        ob = capture_all(model, tok, sb, nl, cl)
        if oa and ob:
            pair_data[pn] = {"A": oa, "B": ob, "seq_len": cl, "subtype": st}
        ia = tok(sa, return_tensors="pt", truncation=True, max_length=cl).to(DEV)
        ib = tok(sb, return_tensors="pt", truncation=True, max_length=cl).to(DEV)
        with torch.no_grad():
            la = model(**ia).logits[0,-1,:].detach().cpu().float()
            lb = model(**ib).logits[0,-1,:].detach().cpu().float()
        kab = float(F.kl_div(F.log_softmax(la,-1), F.softmax(lb,-1), reduction='sum'))
        pair_metrics[pn] = {"logits_a": la, "logits_b": lb, "kl_ab": kab,
                            "sent_a": sa, "seq_len": cl, "subtype": st}
        if (pi+1) % 20 == 0: log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0:.0f}s")
    log(f"  Capture done: {len(pair_data)} pairs, {time.time()-t0:.0f}s")

    # ====== RUN EXPERIMENTS ======
    all_results = {}

    # Exp A: Sliding Block Heatmap
    tA = time.time()
    resA = run_exp_A(model, tok, pairs, pair_data, pair_metrics, nl, MAX_LEN)
    all_results["exp_A"] = resA
    log(f"  Exp A done: {len(resA)} results, {time.time()-tA:.0f}s")

    # Exp B: Position-Specific Patching
    tB = time.time()
    resB = run_exp_B(model, tok, pairs, pair_data, pair_metrics, nl, MAX_LEN)
    all_results["exp_B"] = resB
    log(f"  Exp B done: {len(resB)} results, {time.time()-tB:.0f}s")

    # Exp C: Component Decomposition
    tC = time.time()
    resC = run_exp_C(model, tok, pairs, pair_data, pair_metrics, nl, MAX_LEN)
    all_results["exp_C"] = resC
    log(f"  Exp C done: {len(resC)} results, {time.time()-tC:.0f}s")

    # Exp D: Subtype α Curves
    tD = time.time()
    resD = run_exp_D(model, tok, pairs, pair_data, pair_metrics, nl, MAX_LEN)
    all_results["exp_D"] = resD
    log(f"  Exp D done: {len(resD)} results, {time.time()-tD:.0f}s")

    # ====== ANALYSIS ======
    log("\n=== ANALYSIS ===")

    # A: Block heatmap summary (corrected PROG using α=0 baseline)
    # Group by (block_size, block_start)
    a_agg = defaultdict(list)
    for r in resA:
        a_agg[(r["block_size"], r["block_start"])].append(r)
    a_summary = {}
    for (bs, bstart), rows in sorted(a_agg.items()):
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        a_summary[f"bs{bs}_L{bstart}"] = {
            "block_size": bs, "block_start": bstart,
            "block_end": bstart + bs - 1,
            "mean_prog": float(np.mean(progs)),
            "mean_kr": float(np.mean(krs)),
            "n": len(progs)
        }
    # Find synergy: block_size=4 vs sum of block_size=1
    synergy_data = {}
    for bstart in range(0, nl, 4):
        key4 = f"bs4_L{bstart}"
        if key4 not in a_summary: continue
        prog4 = a_summary[key4]["mean_prog"]
        sum_prog1 = 0
        for li in range(bstart, min(bstart+4, nl)):
            key1 = f"bs1_L{li}"
            if key1 in a_summary:
                sum_prog1 += a_summary[key1]["mean_prog"]
        synergy_data[f"L{bstart}-{bstart+3}"] = {
            "block_prog": prog4,
            "sum_single_prog": sum_prog1,
            "synergy": prog4 - sum_prog1
        }
    log(f"\n  SYNERGY (block4_prog - sum_of_single_progs):")
    for k, v in sorted(synergy_data.items()):
        log(f"    {k}: blk={v['block_prog']:.3f}, sum1={v['sum_single_prog']:.3f}, "
            f"synergy={v['synergy']:.3f} {'✓' if v['synergy']>0.05 else ''}")

    # B: Position effect summary
    b_agg = defaultdict(list)
    for r in resB:
        b_agg[(r["block_start"], r["pos_type"])].append(r)
    b_summary = {}
    for (bstart, pt), rows in sorted(b_agg.items()):
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        b_summary[f"L{bstart}_{pt}"] = {
            "block_start": bstart, "pos_type": pt,
            "mean_prog": float(np.mean(progs)),
            "mean_kr": float(np.mean(krs)),
            "n": len(progs)
        }
    log(f"\n  POSITION EFFECT (attn patch, α=1.0):")
    for k, v in sorted(b_summary.items()):
        log(f"    {k}: prog={v['mean_prog']:.3f}, KR={v['mean_kr']:.2f}")

    # C: Component effect summary
    c_agg = defaultdict(list)
    for r in resC:
        c_agg[(r["block_start"], r["component"])].append(r)
    c_summary = {}
    for (bstart, comp), rows in sorted(c_agg.items()):
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        c_summary[f"L{bstart}_{comp}"] = {
            "block_start": bstart, "component": comp,
            "mean_prog": float(np.mean(progs)),
            "mean_kr": float(np.mean(krs)),
            "n": len(progs)
        }
    log(f"\n  COMPONENT EFFECT:")
    for k, v in sorted(c_summary.items()):
        log(f"    {k}: prog={v['mean_prog']:.3f}, KR={v['mean_kr']:.2f}")

    # D: Subtype α curves
    d_agg = defaultdict(list)
    for r in resD:
        d_agg[(r["block_start"], r["subtype"], r["alpha"])].append(r)
    d_summary = {}
    for (bstart, st, alpha), rows in sorted(d_agg.items()):
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        d_summary[f"L{bstart}_{st}_a{alpha:.2f}"] = {
            "block_start": bstart, "subtype": st, "alpha": alpha,
            "mean_prog": float(np.mean(progs)),
            "mean_kr": float(np.mean(krs)),
            "n": len(progs)
        }

    # ====== SAVE ======
    out_path = RESULT_DIR / f"{model_name}_atlas.json"
    save_data = {
        "model": model_name, "n_layers": nl,
        "n_pairs": len(pair_data),
        "exp_A_block_heatmap": a_summary,
        "exp_A_synergy": synergy_data,
        "exp_B_position": b_summary,
        "exp_C_component": c_summary,
        "exp_D_subtype_alpha": d_summary,
        "exp_A_raw_count": len(resA),
        "exp_B_raw_count": len(resB),
        "exp_C_raw_count": len(resC),
        "exp_D_raw_count": len(resD),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)
    log(f"\n  SAVED: {out_path} ({len(str(save_data))} chars)")

    # ====== CLEANUP ======
    release_model(model)
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 292 {model_name} DONE. Total: {time.time()-t0:.0f}s")
    return save_data


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            run_phase292(name)
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    else:
        run_phase292(model_name)

if __name__ == "__main__":
    main()
