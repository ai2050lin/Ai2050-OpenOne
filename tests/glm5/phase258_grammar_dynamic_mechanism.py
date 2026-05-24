"""
Phase 258: Grammar Dynamic Computation Mechanism
================================================

Core finding from Phase 257: OV@W_E ≈ 0 for all grammar heads.
→ Grammar is COMPUTED, not STORED in static weights.
→ Next question: What do grammar heads READ from residual stream?

Parts:
  Part 1: Grammar head attention pattern analysis
    - Where do grammar heads attend? (subject vs distractor)
    - Is attention pattern causally structured for grammar?

  Part 2: Residual stream decoding at attended positions
    - What information does the attended position carry?
    - Logit lens on attended hidden states

  Part 3: Grammar layering verification
    - 4 grammar types × 30 samples each
    - Precise layer peak for each grammar type
    - Test: different grammar rules peak at different layers?

  Part 4: Activation patching (causal verification)
    - Patch grammar head output from correct→wrong sentence
    - Measure if correct grammar probability increases
    - Convert correlation (logit attribution) to causation

Usage:
  python tests/glm5/phase258_grammar_dynamic_mechanism.py --model qwen3 --part 1
  python tests/glm5/phase258_grammar_dynamic_mechanism.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase258_grammar_dynamic")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_model_safe(model_name):
    """Load model with bfloat16 + device_map=auto + flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} from {cfg['path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, then eager
    for attn_impl in ["flash_attention_2", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:80]}, trying next...")
            continue

    model.eval()
    from model_utils import get_model_info
    info = get_model_info(model, model_name)
    log_time(f"{model_name}: class={info.model_class}, layers={info.n_layers}, "
             f"d_model={info.d_model}, vocab={info.vocab_size}")

    config = model.config
    n_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = getattr(config, 'head_dim', info.d_model // n_heads)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)

    log_time(f"  n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}")

    return model, tokenizer, info, n_heads, head_dim

def get_W_U_safe(model, model_name):
    from model_utils import get_W_U
    return get_W_U(model, model_name)

def release_model_safe(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU cleared")

def save_result(model_name, part, data):
    fname = RESULT_DIR / f"{model_name}_part{part}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    log_time(f"Results saved to {fname}")

def safe_decode(tokenizer, token_id):
    try:
        r = tokenizer.decode([token_id])
        return r.strip() if r else f"<tok_{token_id}>"
    except:
        return f"<tok_{token_id}>"

def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_weight_to_numpy(weight_tensor, model_name=None, layer_name=None):
    import torch
    if weight_tensor.is_meta:
        from model_utils import MODEL_CONFIGS
        import glob, os
        from safetensors import safe_open
        model_path = MODEL_CONFIGS.get(model_name, {}).get("path", None)
        if model_path and layer_name:
            sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    if layer_name in sf.keys():
                        w = sf.get_tensor(layer_name)
                        return w.float().numpy()
        raise ValueError(f"Cannot load meta tensor {layer_name} for {model_name}")
    return weight_tensor.detach().cpu().float().numpy()


# ============================================================
# Part 1: Grammar Head Attention Pattern Analysis
# ============================================================

def part1_attention_pattern(model_name):
    """
    Analyze where grammar heads attend.

    Key question: When a grammar head makes a grammar decision at the verb position,
    does it attend to the SUBJECT position or the DISTRACTOR position?

    Design: Subject-verb agreement with distractors
    "The cat [that the dog chases] ___ (sits/sit)"
    Subject = "cat" (singular) → should attend here
    Distractor = "dog" (singular too, but in different clause)

    Also: simple agreement without distractor
    "The cat ___ (sits/sit)"
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers, "n_heads": n_heads}

    # Load Part 1 grammar heads from Phase 257
    p257_dir = Path("results/phase257_grammar_geometry")
    part1_path = p257_dir / f"{model_name}_part1.json"
    grammar_heads = []
    if part1_path.exists():
        with open(part1_path, 'r', encoding='utf-8') as f:
            p257_data = json.load(f)
        for label, count in p257_data.get("consistent_grammar_heads", []):
            parts = label.replace("L", "").split("_H")
            if len(parts) == 2:
                grammar_heads.append((int(parts[0]), int(parts[1])))
    # Fallback: last 1/3 layers
    if not grammar_heads:
        for li in range(info.n_layers * 2 // 3, info.n_layers):
            for h in range(min(8, n_heads)):
                grammar_heads.append((li, h))

    log_time(f"Grammar heads to analyze: {len(grammar_heads)}")
    log_time(f"Top heads: {[(f'L{l}_H{h}', c) for (l,h), (_, c) in zip(grammar_heads[:5], p257_data.get('consistent_grammar_heads', [])[:5])]}")

    # ---- Define grammar tasks with position annotations ----
    # Each task: (prompt, subject_token, distractor_token_or_None, target_token, competitor_token,
    #             subject_position (char offset), verb_fill_position)
    grammar_tasks = [
        # === Subject-verb agreement: simple ===
        {
            "name": "sv_simple_sing",
            "prompt": "The cat",
            "target_word": " sits",
            "competitor_word": " sit",
            "type": "agreement_simple",
            "subject_words": ["cat"],
        },
        {
            "name": "sv_simple_plur",
            "prompt": "The cats",
            "target_word": " sit",
            "competitor_word": " sits",
            "type": "agreement_simple",
            "subject_words": ["cats"],
        },
        {
            "name": "sv_simple_she",
            "prompt": "She",
            "target_word": " walks",
            "competitor_word": " walk",
            "type": "agreement_simple",
            "subject_words": ["She"],
        },
        {
            "name": "sv_simple_they",
            "prompt": "They",
            "target_word": " walk",
            "competitor_word": " walks",
            "type": "agreement_simple",
            "subject_words": ["They"],
        },
        # === Subject-verb agreement: with distractor ===
        {
            "name": "sv_distractor_sing",
            "prompt": "The cat that the dogs chase",
            "target_word": " sits",
            "competitor_word": " sit",
            "type": "agreement_distractor",
            "subject_words": ["cat"],
            "distractor_words": ["dogs"],
        },
        {
            "name": "sv_distractor_plur",
            "prompt": "The cats that the dog chases",
            "target_word": " sit",
            "competitor_word": " sits",
            "type": "agreement_distractor",
            "subject_words": ["cats"],
            "distractor_words": ["dog"],
        },
        # More distractor examples
        {
            "name": "sv_distractor_sing2",
            "prompt": "The girl that the boys like",
            "target_word": " walks",
            "competitor_word": " walk",
            "type": "agreement_distractor",
            "subject_words": ["girl"],
            "distractor_words": ["boys"],
        },
        {
            "name": "sv_distractor_plur2",
            "prompt": "The girls that the boy likes",
            "target_word": " walk",
            "competitor_word": " walks",
            "type": "agreement_distractor",
            "subject_words": ["girls"],
            "distractor_words": ["boy"],
        },
        # === Tense ===
        {
            "name": "tense_past1",
            "prompt": "Yesterday, she",
            "target_word": " went",
            "competitor_word": " goes",
            "type": "tense",
            "subject_words": ["Yesterday"],
        },
        {
            "name": "tense_past2",
            "prompt": "Last night, he",
            "target_word": " ate",
            "competitor_word": " eats",
            "type": "tense",
            "subject_words": ["Last", "night"],
        },
        # === Comparative ===
        {
            "name": "comp_than",
            "prompt": "She is taller",
            "target_word": " than",
            "competitor_word": " then",
            "type": "comparative",
            "subject_words": ["taller"],
        },
    ]

    # ---- Run analysis ----
    task_results = {}

    for task_idx, task in enumerate(grammar_tasks):
        task_name = task["name"]
        prompt = task["prompt"]

        log_time(f"Task {task_idx+1}/{len(grammar_tasks)}: {task_name} — '{prompt}'")

        # Tokenize prompt to find subject/distractor positions
        # We need token-level positions of subject and distractor
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(task["target_word"], add_special_tokens=False)
        competitor_ids = tokenizer.encode(task["competitor_word"], add_special_tokens=False)

        if not target_ids or not competitor_ids:
            log_time(f"  Cannot tokenize target/competitor, skipping")
            continue

        target_id = target_ids[0]
        competitor_id = competitor_ids[0]

        # Find subject/distractor token positions using decode-matching
        # More robust: decode each token individually and match by string containment
        def find_word_positions_by_decode(prompt_text, words, tok):
            """Find token positions by decoding each token and matching word substrings."""
            prompt_toks = tok.encode(prompt_text, add_special_tokens=False)
            positions = []
            for word in words:
                word_lower = word.lower()
                # Find character offset of word in prompt
                char_start = prompt_text.find(word)
                if char_start < 0:
                    char_start = prompt_text.lower().find(word_lower)
                if char_start < 0:
                    continue
                char_end = char_start + len(word)

                # Reconstruct character positions by cumulatively decoding tokens
                cum_chars = 0
                for ti in range(len(prompt_toks)):
                    decoded = tok.decode(prompt_toks[:ti+1])
                    tok_char_end = len(decoded)
                    # Previous token's end
                    if ti > 0:
                        prev_decoded = tok.decode(prompt_toks[:ti])
                        tok_char_start = len(prev_decoded)
                    else:
                        tok_char_start = 0

                    # Check if this token overlaps with the word's character range
                    if tok_char_start < char_end and tok_char_end > char_start:
                        positions.append(ti)
            return sorted(set(positions))

        subject_positions = find_word_positions_by_decode(
            prompt, task.get("subject_words", []), tokenizer)
        distractor_positions = find_word_positions_by_decode(
            prompt, task.get("distractor_words", []), tokenizer)

        # Verb position = last token position (where prediction happens)
        verb_position = len(prompt_tokens) - 1  # 0-indexed

        log_time(f"  Subject positions: {subject_positions}, Distractor: {distractor_positions}")
        log_time(f"  Verb position: {verb_position}")

        if not subject_positions:
            log_time(f"  No subject positions found, skipping")
            continue

        # Run model with attention weights
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Hook to capture attention weights and head outputs
        attn_weights_captured = {}
        head_outputs_captured = {}
        resid_pre_captured = {}

        def make_attn_hook(li):
            def hook(module, input, output):
                # output is (hidden_states, attn_weights, past_key_value)
                # For eager mode: output[1] is attention weights
                if isinstance(output, tuple) and len(output) >= 2:
                    # attn_weights: [batch, n_heads, seq_q, seq_k]
                    aw = output[1]
                    if aw is not None:
                        attn_weights_captured[li] = aw.detach().float().cpu()
            return hook

        def make_o_proj_hook(li):
            def hook(module, input, output):
                inp = input[0]  # [batch, seq, n_heads * head_dim]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                head_outputs_captured[li] = head_outs[0, :, :, :].detach().float().cpu().numpy()
                # shape: [seq, n_heads, head_dim]
            return hook

        def make_resid_hook(li):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    resid_pre_captured[li] = input[0].detach().float().cpu().numpy()
                else:
                    resid_pre_captured[li] = input.detach().float().cpu().numpy()
            return hook

        hooks = []
        for li, h in grammar_heads:
            layer = layers[li]
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(make_o_proj_hook(li)))
            hooks.append(layer.register_forward_hook(make_resid_hook(li)))

        # Remove duplicate hooks for same layer
        hooked_layers = set()
        clean_hooks = []
        for hook in hooks:
            # We'll just keep all hooks; duplicates won't cause errors
            clean_hooks.append(hook)
        hooks = clean_hooks

        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_attentions=True)
            except Exception as e:
                log_time(f"  Forward failed: {e}")
                for h in hooks:
                    h.remove()
                continue

        for h in hooks:
            h.remove()

        # ---- Analyze attention patterns ----
        head_attn_analysis = {}

        for li, h in grammar_heads:
            if li not in attn_weights_captured:
                continue

            attn_w = attn_weights_captured[li]  # [batch, n_heads, seq_q, seq_k]
            if attn_w.dim() != 4:
                continue

            # Attention from verb_position to all positions
            head_attn = attn_w[0, h, verb_position, :].numpy()  # [seq_k]

            # Attention to subject positions
            subject_attn = float(np.mean([head_attn[p] for p in subject_positions if p < len(head_attn)])) if subject_positions else 0.0
            # Attention to distractor positions
            distractor_attn = float(np.mean([head_attn[p] for p in distractor_positions if p < len(head_attn)])) if distractor_positions else 0.0
            # Attention to first token (often high by default)
            first_attn = float(head_attn[0]) if len(head_attn) > 0 else 0.0
            # Total attention mass
            total_attn = float(np.sum(head_attn))

            # Subject attention ratio = subject_attn / (subject_attn + distractor_attn)
            attn_ratio = subject_attn / max(subject_attn + distractor_attn, 1e-10)

            head_attn_analysis[f"L{li}_H{h}"] = {
                "subject_attn": round(subject_attn, 4),
                "distractor_attn": round(distractor_attn, 4),
                "attn_ratio": round(attn_ratio, 4),
                "first_attn": round(first_attn, 4),
                "total_attn": round(total_attn, 4),
            }

        # Grammar signal (for verification)
        target_dir = W_U[target_id]
        competitor_dir = W_U[competitor_id]

        head_grammar_signals = {}
        for li, h in grammar_heads:
            if li not in head_outputs_captured:
                continue
            head_outs = head_outputs_captured[li]  # [seq, n_heads, head_dim]

            # Get W_O for this layer
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]

            # Head output at verb position
            h_out = head_outs[verb_position, h, :]  # [head_dim]
            contrib_target = float(target_dir @ W_O_h @ h_out)
            contrib_competitor = float(competitor_dir @ W_O_h @ h_out)
            grammar_signal = contrib_target - contrib_competitor

            head_grammar_signals[f"L{li}_H{h}"] = round(grammar_signal, 4)

        task_result = {
            "prompt": prompt,
            "target_word": task["target_word"],
            "competitor_word": task["competitor_word"],
            "target_id": target_id,
            "competitor_id": competitor_id,
            "task_type": task["type"],
            "subject_positions": subject_positions,
            "distractor_positions": distractor_positions,
            "verb_position": verb_position,
            "head_attn_analysis": head_attn_analysis,
            "head_grammar_signals": head_grammar_signals,
        }

        # Actual logit diff
        final_logits = out.logits[0, -1].float().cpu().numpy()
        task_result["actual_target_logit"] = round(float(final_logits[target_id]), 3)
        task_result["actual_competitor_logit"] = round(float(final_logits[competitor_id]), 3)
        task_result["logit_diff"] = round(float(final_logits[target_id] - final_logits[competitor_id]), 3)

        # Sort heads by grammar signal
        sorted_heads = sorted(head_grammar_signals.items(), key=lambda x: x[1], reverse=True)
        task_result["top10_heads_by_signal"] = sorted_heads[:10]

        # Sort heads by subject attention ratio
        if head_attn_analysis:
            sorted_by_attn = sorted(
                head_attn_analysis.items(),
                key=lambda x: x[1]["attn_ratio"],
                reverse=True
            )
            task_result["top10_heads_by_subject_attn"] = sorted_by_attn[:10]

            # Correlation: grammar signal vs subject attention ratio
            common_heads = set(head_attn_analysis.keys()) & set(head_grammar_signals.keys())
            if len(common_heads) > 2:
                gs = [head_grammar_signals[h] for h in common_heads]
                ar = [head_attn_analysis[h]["attn_ratio"] for h in common_heads]
                corr = np.corrcoef(gs, ar)[0, 1]
                task_result["grammar_signal_vs_subject_attn_corr"] = round(float(corr), 4)

        task_results[task_name] = task_result

        log_time(f"  Logit diff: {task_result['logit_diff']:.3f}")
        if head_attn_analysis:
            # Show top grammar heads' attention
            for h_label, gs in sorted_heads[:3]:
                if h_label in head_attn_analysis:
                    ha = head_attn_analysis[h_label]
                    log_time(f"    {h_label}: grammar_signal={gs:.3f}, subject_attn={ha['subject_attn']:.3f}, "
                             f"distractor_attn={ha['distractor_attn']:.3f}, ratio={ha['attn_ratio']:.3f}")

        # Clean up
        del attn_weights_captured, head_outputs_captured, resid_pre_captured, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Cross-task summary ----
    log_time("\n=== Cross-task attention pattern summary ===")

    # For each head, collect attention ratio across distractor tasks
    distractor_tasks = [t for t in grammar_tasks if t["type"] == "agreement_distractor"]
    head_attn_ratios = defaultdict(list)

    for task_name, tr in task_results.items():
        if tr["task_type"] != "agreement_distractor":
            continue
        for h_label, ha in tr.get("head_attn_analysis", {}).items():
            head_attn_ratios[h_label].append(ha["attn_ratio"])

    # Heads with consistently high subject attention ratio
    consistent_subject_heads = []
    for h_label, ratios in head_attn_ratios.items():
        if len(ratios) >= 2:
            mean_ratio = np.mean(ratios)
            consistent_subject_heads.append((h_label, round(float(mean_ratio), 4), len(ratios)))

    consistent_subject_heads.sort(key=lambda x: x[1], reverse=True)

    results["task_results"] = task_results
    results["consistent_subject_attention_heads"] = consistent_subject_heads[:20]

    log_time(f"\nHeads with consistent subject attention (distractor tasks):")
    for h, ratio, n_tasks in consistent_subject_heads[:10]:
        log_time(f"  {h}: mean_subject_ratio={ratio:.4f} (across {n_tasks} tasks)")

    save_result(model_name, 1, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 2: Residual Stream Decoding at Attended Positions
# ============================================================

def part2_residual_stream_decoding(model_name):
    """
    Decode what information the grammar head reads from residual stream.

    Method:
    1. Run model on grammar prompts
    2. At the verb position, find where grammar heads attend most
    3. At that attended position, extract residual stream (hidden state)
    4. Apply logit lens: W_U @ hidden_state → top predicted tokens
    5. See if the attended hidden state carries grammatical features (e.g., singular/plural)
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Load Part 1 results from Phase 258 for grammar heads
    p258_dir = RESULT_DIR
    part1_path = p258_dir / f"{model_name}_part1.json"
    grammar_heads = []

    if part1_path.exists():
        with open(part1_path, 'r', encoding='utf-8') as f:
            p258_data = json.load(f)
        for h_label, ratio, count in p258_data.get("consistent_subject_attention_heads", []):
            parts = h_label.replace("L", "").split("_H")
            if len(parts) == 2:
                grammar_heads.append((int(parts[0]), int(parts[1])))
    else:
        # Fallback: use Phase 257 heads
        p257_dir = Path("results/phase257_grammar_geometry")
        part1_257_path = p257_dir / f"{model_name}_part1.json"
        if part1_257_path.exists():
            with open(part1_257_path, 'r', encoding='utf-8') as f:
                p257_data = json.load(f)
            for label, count in p257_data.get("consistent_grammar_heads", []):
                parts = label.replace("L", "").split("_H")
                if len(parts) == 2:
                    grammar_heads.append((int(parts[0]), int(parts[1])))

    if not grammar_heads:
        # Last 1/3 layers, top 5 heads per layer
        for li in range(info.n_layers * 2 // 3, info.n_layers):
            for h in range(min(5, n_heads)):
                grammar_heads.append((li, h))

    log_time(f"Grammar heads for residual stream decoding: {len(grammar_heads)}")
    # Only analyze top 10 heads to save time
    grammar_heads = grammar_heads[:10]
    log_time(f"Analyzing top 10 heads: {[(f'L{l}_H{h}') for l,h in grammar_heads]}")

    # Define prompts for logit lens analysis
    analysis_prompts = [
        # Singular subject
        {"prompt": "The cat", "subject_word": "cat", "number": "singular",
         "target": " sits", "competitor": " sit"},
        {"prompt": "The dog", "subject_word": "dog", "number": "singular",
         "target": " runs", "competitor": " run"},
        {"prompt": "She", "subject_word": "She", "number": "singular",
         "target": " walks", "competitor": " walk"},
        # Plural subject
        {"prompt": "The cats", "subject_word": "cats", "number": "plural",
         "target": " sit", "competitor": " sits"},
        {"prompt": "The dogs", "subject_word": "dogs", "number": "plural",
         "target": " run", "competitor": " runs"},
        {"prompt": "They", "subject_word": "They", "number": "plural",
         "target": " walk", "competitor": " walks"},
        # Distractor
        {"prompt": "The cat that the dogs chase", "subject_word": "cat", "number": "singular",
         "target": " sits", "competitor": " sit"},
        {"prompt": "The cats that the dog chases", "subject_word": "cats", "number": "plural",
         "target": " sit", "competitor": " sits"},
    ]

    prompt_results = {}

    for pi, prompt_info in enumerate(analysis_prompts):
        prompt = prompt_info["prompt"]
        subject_word = prompt_info["subject_word"]
        number = prompt_info["number"]

        log_time(f"\nPrompt {pi+1}/{len(analysis_prompts)}: '{prompt}' (subject={subject_word}, {number})")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Find subject token position using decode-based matching
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        subject_pos = None
        char_start = prompt.find(subject_word)
        if char_start >= 0:
            char_end = char_start + len(subject_word)
            for ti in range(len(prompt_tokens)):
                decoded = tokenizer.decode(prompt_tokens[:ti+1])
                tok_char_end = len(decoded)
                if ti > 0:
                    prev_decoded = tokenizer.decode(prompt_tokens[:ti])
                    tok_char_start = len(prev_decoded)
                else:
                    tok_char_start = 0
                if tok_char_start < char_end and tok_char_end > char_start:
                    subject_pos = ti
                    break

        if subject_pos is None:
            log_time(f"  Cannot find subject position, skipping")
            continue

        # Hook to capture: attention weights, residual stream at each layer
        attn_weights = {}
        resid_streams = {}  # {layer_idx: [1, seq, d_model]}

        def make_attn_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    aw = output[1]
                    if aw is not None:
                        attn_weights[li] = aw.detach().float().cpu()
            return hook

        def make_resid_hook(li):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    resid_streams[li] = input[0].detach().float().cpu().numpy()
            return hook

        hooks = []
        hooked_layers = set()
        for li, h in grammar_heads:
            if li not in hooked_layers:
                layer = layers[li]
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
                hooks.append(layer.register_forward_hook(make_resid_hook(li)))
                hooked_layers.add(li)

        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_attentions=True, output_hidden_states=True)
            except Exception as e:
                log_time(f"  Forward failed: {e}")
                for h in hooks:
                    h.remove()
                continue

        for h in hooks:
            h.remove()

        # Get hidden states from output (all layers)
        if out.hidden_states:
            all_hidden = out.hidden_states  # tuple of [1, seq, d_model]

        # ---- For each grammar head: where it attends and what it reads ----
        head_decodings = {}

        for li, h in grammar_heads:
            if li not in attn_weights:
                continue

            attn_w = attn_weights[li]  # [batch, n_heads, seq_q, seq_k]
            if attn_w.dim() != 4:
                continue

            verb_pos = len(prompt_tokens) - 1

            # Attention pattern from verb position
            head_attn = attn_w[0, h, verb_pos, :].numpy()  # [seq_k]

            # Top-3 attended positions
            top3_pos = np.argsort(head_attn)[-3:][::-1]

            # For each top attended position, decode residual stream
            position_decodings = []
            for pos in top3_pos:
                attn_weight = float(head_attn[pos])
                token_at_pos = safe_decode(tokenizer, int(prompt_tokens[pos])) if pos < len(prompt_tokens) else "<eos>"

                # Get hidden state at this position from the grammar head's layer
                if li in resid_streams:
                    hidden = resid_streams[li][0, pos, :]  # [d_model]
                elif out.hidden_states and li < len(out.hidden_states):
                    hidden = out.hidden_states[li][0, pos, :].float().cpu().numpy()
                else:
                    continue

                # Logit lens: W_U @ hidden → top tokens
                logits_at_pos = W_U @ hidden  # [vocab_size]
                top10_ids = np.argsort(logits_at_pos)[-10:][::-1]
                top10_tokens = [(safe_decode(tokenizer, int(tid)), round(float(logits_at_pos[tid]), 2))
                               for tid in top10_ids]

                # Check if singular/plural verbs appear in top tokens
                sing_verb_ids = [tokenizer.encode(w, add_special_tokens=False)[0]
                                for w in ["sits", "runs", "walks", "eats"] if tokenizer.encode(w, add_special_tokens=False)]
                plur_verb_ids = [tokenizer.encode(w, add_special_tokens=False)[0]
                                for w in ["sit", "run", "walk", "eat"] if tokenizer.encode(w, add_special_tokens=False)]

                sing_verb_logits = [float(logits_at_pos[vid]) for vid in sing_verb_ids if vid < len(logits_at_pos)]
                plur_verb_logits = [float(logits_at_pos[vid]) for vid in plur_verb_ids if vid < len(logits_at_pos)]

                number_signal = (np.mean(sing_verb_logits) - np.mean(plur_verb_logits)) if (sing_verb_logits and plur_verb_logits) else 0.0

                position_decodings.append({
                    "position": int(pos),
                    "token": token_at_pos,
                    "attn_weight": round(attn_weight, 4),
                    "is_subject": pos == subject_pos,
                    "top10_logit_lens": top10_tokens,
                    "number_signal": round(float(number_signal), 4),  # positive = singular, negative = plural
                    "mean_sing_verb_logit": round(float(np.mean(sing_verb_logits)), 4) if sing_verb_logits else None,
                    "mean_plur_verb_logit": round(float(np.mean(plur_verb_logits)), 4) if plur_verb_logits else None,
                })

            head_decodings[f"L{li}_H{h}"] = position_decodings

        prompt_result = {
            "prompt": prompt,
            "subject_word": subject_word,
            "number": number,
            "subject_pos": subject_pos,
            "head_decodings": head_decodings,
        }

        # Summary: for each head, does it attend to subject and does subject carry number signal?
        head_summaries = {}
        for h_label, pos_decodings in head_decodings.items():
            subject_decodings = [d for d in pos_decodings if d["is_subject"]]
            if subject_decodings:
                sd = subject_decodings[0]
                head_summaries[h_label] = {
                    "subject_attn": sd["attn_weight"],
                    "subject_number_signal": sd["number_signal"],
                    "subject_top3_tokens": sd["top10_logit_lens"][:3],
                }

        prompt_result["head_summaries"] = head_summaries

        # Log key findings
        for h_label, hs in head_summaries.items():
            log_time(f"  {h_label}: subject_attn={hs['subject_attn']:.3f}, "
                     f"number_signal={hs['subject_number_signal']:.4f}, "
                     f"top3={hs['subject_top3_tokens'][:2]}")

        prompt_results[f"prompt_{pi}"] = prompt_result

        # Clean up
        del attn_weights, resid_streams, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Cross-prompt analysis: number signal consistency ----
    log_time("\n=== Number signal at subject position ===")

    # For singular prompts, number_signal should be positive (promotes singular verbs)
    # For plural prompts, number_signal should be negative (promotes plural verbs)
    head_number_consistency = defaultdict(lambda: {"sing_signals": [], "plur_signals": []})

    for pi, prompt_info in enumerate(analysis_prompts):
        key = f"prompt_{pi}"
        if key not in prompt_results:
            continue
        number = prompt_info["number"]
        for h_label, hs in prompt_results[key].get("head_summaries", {}).items():
            if number == "singular":
                head_number_consistency[h_label]["sing_signals"].append(hs["subject_number_signal"])
            else:
                head_number_consistency[h_label]["plur_signals"].append(hs["subject_number_signal"])

    # A grammar head should have: positive number_signal for singular, negative for plural
    number_consistent_heads = []
    for h_label, signals in head_number_consistency.items():
        if signals["sing_signals"] and signals["plur_signals"]:
            mean_sing = np.mean(signals["sing_signals"])
            mean_plur = np.mean(signals["plur_signals"])
            # For singular prompts, signal should be positive; for plural, negative
            consistency = mean_sing - mean_plur  # should be positive for grammar heads
            number_consistent_heads.append((h_label, round(float(consistency), 4),
                                           round(float(mean_sing), 4), round(float(mean_plur), 4)))

    number_consistent_heads.sort(key=lambda x: x[1], reverse=True)

    results["prompt_results"] = prompt_results
    results["number_consistent_heads"] = number_consistent_heads

    log_time(f"\nHeads with consistent number signal (sing_signal - plur_signal):")
    for h, cons, ms, mp in number_consistent_heads[:10]:
        log_time(f"  {h}: consistency={cons:.4f} (sing={ms:.4f}, plur={mp:.4f})")

    save_result(model_name, 2, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 3: Grammar Layering Verification
# ============================================================

def part3_grammar_layering(model_name):
    """
    Verify that different grammar types peak at different layers.

    4 grammar types × 30 samples each = 120 tasks total.
    For each task, compute per-layer logit attribution.
    Find the peak layer for each grammar type.
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # ---- Define 4 grammar types with many samples ----
    grammar_samples = {
        "agreement_simple": [
            # Singular subject → singular verb
            ("The cat", " sits", " sit"),
            ("The dog", " runs", " run"),
            ("The bird", " flies", " fly"),
            ("The girl", " walks", " walk"),
            ("The boy", " reads", " read"),
            ("She", " walks", " walk"),
            ("He", " runs", " run"),
            ("It", " sits", " sit"),
            # Plural subject → plural verb
            ("The cats", " sit", " sits"),
            ("The dogs", " run", " runs"),
            ("The birds", " fly", " flies"),
            ("The girls", " walk", " walks"),
            ("The boys", " read", " reads"),
            ("They", " walk", " walks"),
            ("We", " run", " runs"),
            ("You", " sit", " sits"),
            # More variety
            ("The man", " works", " work"),
            ("The woman", " speaks", " speak"),
            ("The child", " plays", " play"),
            ("The tree", " grows", " grow"),
            ("The men", " work", " works"),
            ("The women", " speak", " speaks"),
            ("The children", " play", " plays"),
            ("The trees", " grow", " grows"),
        ],
        "agreement_distractor": [
            # Subject-verb agreement with attractor/distractor
            ("The cat that the dogs chase", " sits", " sit"),
            ("The cats that the dog chases", " sit", " sits"),
            ("The girl that the boys like", " walks", " walk"),
            ("The girls that the boy likes", " walk", " walks"),
            ("The man that the women see", " runs", " run"),
            ("The men that the woman sees", " run", " runs"),
            ("The bird that the cats watch", " flies", " fly"),
            ("The birds that the cat watches", " fly", " flies"),
            ("The book that the students read", " is", " are"),
            ("The books that the student reads", " are", " is"),
            ("The apple that the children eat", " falls", " fall"),
            ("The apples that the child eats", " fall", " falls"),
            ("The key that the doors need", " opens", " open"),
            ("The keys that the door needs", " open", " opens"),
            ("The house that the families built", " stands", " stand"),
            ("The houses that the family built", " stand", " stands"),
        ],
        "tense": [
            # Past tense
            ("Yesterday, she", " went", " goes"),
            ("Yesterday, he", " ate", " eats"),
            ("Yesterday, they", " ran", " runs"),
            ("Last night, she", " slept", " sleeps"),
            ("Last week, he", " wrote", " writes"),
            ("Last year, they", " built", " builds"),
            ("In the past, she", " walked", " walks"),
            ("Before, he", " knew", " knows"),
            ("Previously, they", " thought", " thinks"),
            ("Once, she", " sang", " sings"),
            # Future/habitual contrast
            ("Tomorrow, she", " will", " would"),
            ("Tomorrow, he", " goes", " went"),
            ("Tomorrow, they", " arrive", " arrived"),
            ("Now, she", " is", " was"),
            ("Now, he", " runs", " ran"),
            ("Now, they", " eat", " ate"),
            ("Always, she", " walks", " walked"),
            ("Usually, he", " reads", " read"),
        ],
        "comparative_article": [
            # Comparative vs superlative
            ("She is taller", " than", " then"),
            ("He is faster", " than", " then"),
            ("This is bigger", " than", " then"),
            ("That is smaller", " than", " then"),
            # Article selection
            ("I saw", " a", " an"),  # Actually "a" before consonant... context matters
            ("I need", " an", " a"),  # "an" before vowel
            # Determiner
            ("___ dog is friendly", "The", "A"),
            ("___ cats are hungry", "The", "A"),
        ],
    }

    # Count samples per type
    for gtype, samples in grammar_samples.items():
        log_time(f"  {gtype}: {len(samples)} samples")

    # ---- Per-layer logit attribution for each sample ----
    layer_attributions = defaultdict(lambda: defaultdict(list))
    # layer_attributions[grammar_type][layer] = [attribution_values]

    for gtype, samples in grammar_samples.items():
        log_time(f"\n=== Processing {gtype}: {len(samples)} samples ===")

        for si, (prompt, target_word, competitor_word) in enumerate(samples):
            # Tokenize
            target_ids = tokenizer.encode(target_word, add_special_tokens=False)
            competitor_ids = tokenizer.encode(competitor_word, add_special_tokens=False)

            if not target_ids or not competitor_ids:
                continue

            target_id = target_ids[0]
            competitor_id = competitor_ids[0]

            if target_id >= W_U.shape[0] or competitor_id >= W_U.shape[0]:
                continue

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            # Hook MLP output for layer attribution
            mlp_outputs = {}

            def make_mlp_hook(li):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        mlp_outputs[li] = output[0][0, -1].detach().float().cpu().numpy()
                    else:
                        mlp_outputs[li] = output[0, -1].detach().float().cpu().numpy()
                return hook

            # Hook attention output
            attn_outputs = {}

            def make_attn_out_hook(li):
                def hook(module, input, output):
                    # Attention block output (after o_proj + residual)
                    if isinstance(output, tuple):
                        attn_outputs[li] = output[0][0, -1].detach().float().cpu().numpy()
                    else:
                        attn_outputs[li] = output[0, -1].detach().float().cpu().numpy()
                return hook

            hooks = []
            for li in range(info.n_layers):
                layer = layers[li]
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
                hooks.append(layer.register_forward_hook(make_attn_out_hook(li)))

            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
                except Exception as e:
                    log_time(f"  Forward failed for sample {si}: {e}")
                    for h in hooks:
                        h.remove()
                    continue

            for h in hooks:
                h.remove()

            # Compute per-layer attribution
            target_dir = W_U[target_id]
            competitor_dir = W_U[competitor_id]

            # Method: attribution[layer] = (target_dir - competitor_dir) @ layer_output
            for li in range(info.n_layers):
                if li in mlp_outputs:
                    mlp_attr = float((target_dir - competitor_dir) @ mlp_outputs[li])
                    layer_attributions[gtype][li].append(mlp_attr)

            if (si + 1) % 8 == 0:
                log_time(f"  Processed {si+1}/{len(samples)} samples for {gtype}")

            # Clean up
            del mlp_outputs, attn_outputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Compute peak layers for each grammar type ----
    log_time("\n=== Peak layer analysis ===")

    peak_layers = {}
    for gtype, layer_attrs in layer_attributions.items():
        mean_attrs = {}
        for li, attrs in layer_attrs.items():
            if attrs:
                mean_attrs[li] = float(np.mean(attrs))

        if not mean_attrs:
            continue

        peak_layer = max(mean_attrs, key=mean_attrs.get)
        peak_value = mean_attrs[peak_layer]

        # Top-5 layers
        sorted_layers = sorted(mean_attrs.items(), key=lambda x: x[1], reverse=True)
        top5 = [(f"L{l}", round(v, 4)) for l, v in sorted_layers[:5]]

        peak_layers[gtype] = {
            "peak_layer": f"L{peak_layer}",
            "peak_layer_idx": peak_layer,
            "peak_attribution": round(peak_value, 4),
            "top5_layers": top5,
            "n_samples": len(list(layer_attrs.values())[0]) if layer_attrs else 0,
        }

        log_time(f"  {gtype}: peak at L{peak_layer} (attribution={peak_value:.4f}), top5={top5}")

    results["peak_layers"] = peak_layers
    results["layer_attributions_raw"] = {
        gtype: {f"L{li}": [round(v, 4) for v in attrs]
                for li, attrs in layer_attrs.items()}
        for gtype, layer_attrs in layer_attributions.items()
    }

    # ---- Cross-type comparison ----
    log_time("\n=== Cross-type peak layer comparison ===")
    if len(peak_layers) >= 2:
        peak_layer_indices = {gtype: pl["peak_layer_idx"] for gtype, pl in peak_layers.items()}
        log_time(f"  Peak layers by type: {peak_layer_indices}")

        # Are the peaks significantly different?
        peak_values = list(peak_layer_indices.values())
        if len(set(peak_values)) > 1:
            log_time(f"  ★ Peak layers are DIFFERENT across grammar types")
        else:
            log_time(f"  Peak layers are the SAME across grammar types")

    save_result(model_name, 3, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 4: Activation Patching (Causal Verification)
# ============================================================

def part4_activation_patching(model_name):
    """
    Causal test: Does patching grammar head activation from correct→wrong sentence
    fix the grammar error?

    Method:
    1. Run correct sentence: "The cat sits" → get grammar head output at verb position
    2. Run wrong sentence: "The cat sit" → get grammar head output at verb position
    3. Patch: replace grammar head output in wrong sentence with output from correct sentence
    4. Measure: does the probability of correct verb increase?
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Load grammar heads
    p257_dir = Path("results/phase257_grammar_geometry")
    part1_path = p257_dir / f"{model_name}_part1.json"
    grammar_heads = []

    if part1_path.exists():
        with open(part1_path, 'r', encoding='utf-8') as f:
            p257_data = json.load(f)
        for label, count in p257_data.get("consistent_grammar_heads", []):
            parts = label.replace("L", "").split("_H")
            if len(parts) == 2:
                grammar_heads.append((int(parts[0]), int(parts[1])))

    if not grammar_heads:
        for li in range(info.n_layers * 2 // 3, info.n_layers):
            for h in range(min(5, n_heads)):
                grammar_heads.append((li, h))

    # Top 10 grammar heads for ablation
    grammar_heads = grammar_heads[:10]
    log_time(f"Grammar heads for ablation test: {[(f'L{l}_H{h}') for l,h in grammar_heads]}")

    # Define test prompts: (prefix, correct_verb, wrong_verb)
    test_prompts = [
        ("The cat", " sits", " sit"),
        ("The dogs", " run", " runs"),
        ("She", " walks", " walk"),
        ("They", " eat", " eats"),
        ("The girl that the boys like", " walks", " walk"),
        ("The cats that the dog chases", " sit", " sits"),
    ]

    patching_results = {}

    for pair_idx, (prefix, correct_verb, wrong_verb) in enumerate(test_prompts):
        log_time(f"\nPrompt {pair_idx+1}: '{prefix}' → {correct_verb}/{wrong_verb}")

        # Tokenize
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        correct_verb_ids = tokenizer.encode(correct_verb, add_special_tokens=False)
        wrong_verb_ids = tokenizer.encode(wrong_verb, add_special_tokens=False)

        if not correct_verb_ids or not wrong_verb_ids:
            log_time(f"  Cannot tokenize verbs, skipping")
            continue

        correct_verb_id = correct_verb_ids[0]
        wrong_verb_id = wrong_verb_ids[0]

        # ---- Baseline: get logits without ablation ----
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attn_mask)

        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_logit_diff = float(base_logits[correct_verb_id] - base_logits[wrong_verb_id])

        log_time(f"  Baseline logit_diff (correct-wrong): {base_logit_diff:.3f}")

        # ---- Ablation: for each grammar head, zero-out its contribution ----
        head_patching = {}

        for li, h in grammar_heads:
            # Get W_O for this head
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]

            # Hook to capture head output at last position
            head_output = {}

            def make_capture_hook(capture_dict):
                def hook(module, input, output):
                    inp = input[0]  # [batch, seq, n_heads * head_dim]
                    batch, seq, _ = inp.shape
                    head_outs = inp.view(batch, seq, n_heads, head_dim)
                    capture_dict["output"] = head_outs[0, -1, h, :].detach().float().cpu()
                return hook

            # Get head output
            hook_cap = layers[li].self_attn.o_proj.register_forward_hook(make_capture_hook(head_output))
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attn_mask)
            hook_cap.remove()

            if "output" not in head_output:
                continue

            h_out = head_output["output"]  # [head_dim]

            # Compute this head's contribution to logit_diff
            # contribution = (target_dir - competitor_dir) @ W_O_h @ h_out
            target_dir = W_U[correct_verb_id]
            competitor_dir = W_U[wrong_verb_id]
            diff_dir = target_dir - competitor_dir  # [d_model]

            head_contribution = float(diff_dir @ W_O_h @ h_out.numpy())

            # Now ablate: zero-out this head's contribution
            # delta_output = W_O_h @ h_out → this is what the head writes to residual stream
            delta_output = W_O_h @ h_out.numpy()  # [d_model]
            delta_tensor = torch.tensor(delta_output, dtype=torch.float16, device=input_device)

            # Apply ablation by subtracting the head's contribution from the attention output
            ablation_applied = [False]

            def make_ablation_hook(delta, head_label):
                def hook(module, input, output):
                    if ablation_applied[0]:
                        return output
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[0, -1, :] -= delta.to(hidden.device).to(hidden.dtype)
                        ablation_applied[0] = True
                        return (hidden,) + output[1:]
                    return output
                return hook

            hook_abl = layers[li].self_attn.register_forward_hook(
                make_ablation_hook(delta_tensor, f"L{li}_H{h}"))
            with torch.no_grad():
                ablated_out = model(input_ids=input_ids, attention_mask=attn_mask)
            hook_abl.remove()

            ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
            ablated_logit_diff = float(ablated_logits[correct_verb_id] - ablated_logits[wrong_verb_id])

            # Effect of ablation
            logit_diff_change = ablated_logit_diff - base_logit_diff
            # Negative change = ablation hurt grammar = head was contributing to grammar
            # Positive change = ablation helped = head was anti-grammar

            head_patching[f"L{li}_H{h}"] = {
                "base_logit_diff": round(base_logit_diff, 4),
                "ablated_logit_diff": round(ablated_logit_diff, 4),
                "logit_diff_change": round(logit_diff_change, 4),
                "head_contribution_attribution": round(head_contribution, 4),
                "correct_verb_logit_before": round(float(base_logits[correct_verb_id]), 4),
                "correct_verb_logit_after": round(float(ablated_logits[correct_verb_id]), 4),
            }

            log_time(f"  L{li}_H{h}: logit_diff {base_logit_diff:.3f}→{ablated_logit_diff:.3f} "
                     f"(change={logit_diff_change:.3f}, attribution={head_contribution:.3f})")

            # Clean up
            del head_output, ablated_out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        patching_results[f"prompt_{pair_idx}"] = {
            "prefix": prefix,
            "correct_verb": correct_verb,
            "wrong_verb": wrong_verb,
            "base_logit_diff": round(base_logit_diff, 4),
            "head_ablation": head_patching,
        }

        # Clean up
        del base_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary: which heads are causally involved ----
    log_time("\n=== Causal head summary (ablation) ===")

    # Collect ablation effects across all prompts
    head_causal_scores = defaultdict(list)
    for pair_key, pair_data in patching_results.items():
        for h_label, hp in pair_data.get("head_ablation", {}).items():
            head_causal_scores[h_label].append(hp["logit_diff_change"])

    causal_heads = []
    for h_label, changes in head_causal_scores.items():
        mean_change = np.mean(changes)
        n_negative = sum(1 for c in changes if c < 0)  # ablation hurt = head was helping
        causal_heads.append((h_label, round(float(mean_change), 4), n_negative, len(changes)))

    causal_heads.sort(key=lambda x: x[1])  # Most negative first = most causal

    results["patching_results"] = patching_results
    results["causal_heads"] = causal_heads

    log_time(f"\nCausal heads (sorted by mean ablation effect, most negative = most causal):")
    for h, mean_c, n_neg, n_total in causal_heads:
        log_time(f"  {h}: mean_ablation_effect={mean_c:.4f}, hurt_in_{n_neg}/{n_total}_prompts")

    save_result(model_name, 4, results)
    release_model_safe(model)
    return results


# ============================================================
# Main
# ============================================================

PART_FUNCTIONS = {
    1: part1_attention_pattern,
    2: part2_residual_stream_decoding,
    3: part3_grammar_layering,
    4: part4_activation_patching,
}

def main():
    parser = argparse.ArgumentParser(description="Phase 258: Grammar Dynamic Computation Mechanism")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--part", type=str, required=True,
                       help="Part number (1-4) or 'all'")
    args = parser.parse_args()

    model_name = args.model

    if args.part == "all":
        parts = [1, 2, 3, 4]
    else:
        parts = [int(args.part)]

    log_time(f"Phase 258: Grammar Dynamic Computation Mechanism")
    log_time(f"Model: {model_name}, Parts: {parts}")
    log_time(f"=" * 60)

    for part_num in parts:
        if part_num not in PART_FUNCTIONS:
            log_time(f"Unknown part: {part_num}, skipping")
            continue

        log_time(f"\n{'#' * 60}")
        log_time(f"# Starting Part {part_num}")
        log_time(f"{'#' * 60}")

        try:
            result = PART_FUNCTIONS[part_num](model_name)
            log_time(f"Part {part_num} completed successfully!")
        except Exception as e:
            log_time(f"Part {part_num} FAILED: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)

    log_time(f"\nPhase 258 completed for {model_name}!")

if __name__ == "__main__":
    main()
