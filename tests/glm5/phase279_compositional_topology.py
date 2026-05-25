"""
Phase 279: Compositional Computational Topology
================================================
Shift from token-level dynamics to RELATION/COMPOSITION/OPERATOR dynamics.

Key question: How does language CHANGE the computation graph?

Block A: Relation Dynamics
  - Compare "dog chases cat" vs "cat chases dog" vs "dog" vs "cat"
  - Measure: does relation change Jacobian? Subspace? Trajectory divergence?
  - FIND: Is relation encoded differently than entity?

Block B: Composition Test
  - f(A+B) vs f(A)+f(B): "red apple" vs "red"+"apple"
  - Measure: trajectory of compound vs sum of trajectories
  - FIND: Is composition linear or dynamic routing?

Block C: Operator Dynamics
  - not, if, because, every, some — true computational operators
  - Compare "happy" vs "not happy", "run" vs "if run"
  - Measure: how operator changes subsequent reachable states

Block D: Recursive Closure
  - "the dog" vs "the dog that chased the cat" vs nested
  - Measure: trajectory curvature, subspace dimension, Jacobian rank
  - FIND: Does recursion increase computational dimension?

Usage:
  python tests/glm5/phase279_compositional_topology.py qwen3
  python tests/glm5/phase279_compositional_topology.py glm4
  python tests/glm5/phase279_compositional_topology.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import MODEL_CONFIGS, get_model_info, get_layers

RESULT_DIR = Path("results/phase279_compositional_topology")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Stimulus Definitions =====

# Block A: Relation dynamics — same entities, different relations
RELATION_TRIPLES = [
    # (subject, relation, object) — SVO order
    ("dog", "chases", "cat"),
    ("cat", "chases", "dog"),       # reversed
    ("dog", "sees", "cat"),         # different relation
    ("dog", "bites", "cat"),        # different relation
    ("man", "loves", "woman"),
    ("woman", "loves", "man"),      # reversed
    ("king", "rules", "city"),
    ("city", "surrounds", "king"),  # reversed + different relation
]

# Control: same entities without relation
RELATION_CONTROLS = [
    "dog", "cat", "man", "woman", "king", "city",
]

# Block B: Composition test
COMPOSITION_PAIRS = [
    # (modifier, noun) — adjective-noun
    ("red", "apple"), ("big", "dog"), ("angry", "man"), ("cold", "water"),
    ("happy", "child"), ("sharp", "knife"), ("old", "city"), ("dark", "night"),
    # (modifier, verb) — adverb-verb
    ("slowly", "walk"), ("quickly", "run"), ("carefully", "think"),
    ("never", "stop"), ("always", "win"), ("barely", "see"),
]

# Block C: Operator dynamics
OPERATOR_PAIRS = [
    # (operator, operand) — how operator transforms operand
    ("not", "happy"), ("not", "true"), ("not", "possible"),
    ("if", "rain"), ("if", "possible"), ("if", "ready"),
    ("because", "tired"), ("because", "late"), ("because", "cold"),
    ("every", "person"), ("every", "cat"), ("every", "day"),
    ("some", "people"), ("some", "cats"), ("some", "days"),
    ("no", "reason"), ("no", "way"), ("no", "time"),
    # Tense operators
    ("will", "go"), ("will", "eat"), ("will", "think"),
    ("can", "fly"), ("can", "swim"), ("can", "read"),
    ("must", "leave"), ("must", "stay"), ("must", "go"),
]

# Operands alone (for comparison)
OPERAND_ALONE = list(set([op for _, op in OPERATOR_PAIRS]))

# Operators alone
OPERATOR_ALONE = list(set([op for op, _ in OPERATOR_PAIRS]))

# Block D: Recursive closure
RECURSIVE_SENTENCES = [
    # Level 0: simple
    "the dog",
    # Level 1: one relative clause
    "the dog that chased the cat",
    # Level 2: two relative clauses
    "the dog that chased the cat that ate the fish",
    # Level 3: three relative clauses
    "the dog that chased the cat that ate the fish that swam in the river",
    # Also: prepositional recursion
    "the man",
    "the man in the house",
    "the man in the house on the hill",
    "the man in the house on the hill by the river",
    # Also: complement recursion
    "I think",
    "I think that she knows",
    "I think that she knows that he left",
    "I think that she knows that he left that we forgot",
]


# ===== Model Loading =====

def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash attention first, fallback to eager
    attn_impl = "eager"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        log_time(f"  flash_attn available, using {attn_impl}")
    except ImportError:
        log_time(f"  flash_attn not available, using {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# ===== Trajectory Extraction =====

def extract_trajectory(model, tokenizer, device, prompt):
    """Extract h_l at ALL layers for a prompt. Returns dict or None."""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            log_time(f"  WARNING: Forward failed for '{prompt[:50]}': {e}")
            return None

    hs = outputs.hidden_states
    h_dict = {}
    for l in range(len(hs)):
        h_dict[l] = hs[l][0, -1, :].float().cpu().numpy()
    return h_dict


def extract_trajectory_all_positions(model, tokenizer, device, prompt):
    """Extract h_l at ALL layers, ALL token positions. Returns {l: [seq_len, d_model]} or None."""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            log_time(f"  WARNING: Forward failed for '{prompt[:50]}': {e}")
            return None

    hs = outputs.hidden_states
    h_dict = {}
    for l in range(len(hs)):
        h_dict[l] = hs[l][0, :, :].float().cpu().numpy()  # [seq_len, d_model]
    return h_dict


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def subspace_overlap(A, B, k=10):
    """Compute subspace overlap between two sets of vectors using top-k SVD."""
    # A: [d, n1], B: [d, n2]
    if A.shape[1] < k or B.shape[1] < k:
        k = min(A.shape[1], B.shape[1])
    if k < 1:
        return 0.0
    U_a, _, _ = np.linalg.svd(A, full_matrices=False)
    U_b, _, _ = np.linalg.svd(B, full_matrices=False)
    # Overlap = sum of (cos between columns)^2
    overlap = np.sum((U_a[:, :k].T @ U_b[:, :k]) ** 2) / k
    return float(overlap)


# ===== Block A: Relation Dynamics =====

def block_a_relation(model, tokenizer, device, model_name, n_layers, d_model):
    """
    Compare SVO sentences with reversed/computed controls.
    Core question: Does relation change computation differently than entity swap?
    """
    log_time("=== Block A: Relation Dynamics ===")

    # Extract trajectories for all SVO sentences and controls
    svo_trajs = {}  # (s, r, o) -> {l: h}
    control_trajs = {}  # word -> {l: h}
    svo_allpos = {}  # (s, r, o) -> {l: [seq_len, d_model]}

    log_time(f"  Extracting {len(RELATION_TRIPLES)} SVO trajectories...")
    t0 = time.time()
    for i, (s, r, o) in enumerate(RELATION_TRIPLES):
        prompt = f"The {s} {r} the {o}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            svo_trajs[(s, r, o)] = traj

        # Also extract all positions for key pairs
        if r == "chases":
            allpos = extract_trajectory_all_positions(model, tokenizer, device, prompt)
            if allpos is not None:
                svo_allpos[(s, r, o)] = allpos

        if (i + 1) % 4 == 0:
            elapsed = time.time() - t0
            log_time(f"    SVO progress: {i+1}/{len(RELATION_TRIPLES)}, elapsed={elapsed:.1f}s")

    log_time(f"  Extracting {len(RELATION_CONTROLS)} control trajectories...")
    for i, word in enumerate(RELATION_CONTROLS):
        prompt = f"The {word}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            control_trajs[word] = traj

    # === Analysis 1: Relation swap effect ===
    # "dog chases cat" vs "cat chases dog" — same entities, different order
    # Compare trajectory divergence at each layer
    log_time("  Analysis 1: Relation swap (S→O reversal)")

    swap_pairs = [
        (("dog", "chases", "cat"), ("cat", "chases", "dog")),
        (("man", "loves", "woman"), ("woman", "loves", "man")),
    ]

    swap_results = {}
    for pair_a, pair_b in swap_pairs:
        if pair_a in svo_trajs and pair_b in svo_trajs:
            h_a = svo_trajs[pair_a]
            h_b = svo_trajs[pair_b]
            per_layer_cos = {}
            per_layer_norm_diff = {}
            for l in range(n_layers + 1):
                if l in h_a and l in h_b:
                    per_layer_cos[l] = cosine_sim(h_a[l], h_b[l])
                    per_layer_norm_diff[l] = float(np.linalg.norm(h_a[l] - h_b[l]))

            key = f"{pair_a[0]}_{pair_a[1]}_{pair_a[2]}_vs_{pair_b[0]}_{pair_b[1]}_{pair_b[2]}"
            swap_results[key] = {
                "per_layer_cosine": {str(k): v for k, v in per_layer_cos.items()},
                "per_layer_norm_diff": {str(k): v for k, v in per_layer_norm_diff.items()},
            }

    # === Analysis 2: Relation type effect ===
    # "dog chases cat" vs "dog sees cat" vs "dog bites cat" — same S/O, different relation
    log_time("  Analysis 2: Relation type (same S/O, different R)")

    relation_type_groups = defaultdict(list)
    for (s, r, o) in svo_trajs.keys():
        relation_type_groups[(s, o)].append((s, r, o))

    relation_type_results = {}
    for (s, o), triples in relation_type_groups.items():
        if len(triples) < 2:
            continue
        # Pairwise cosine between different relations on same S/O
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                ta, tb = svo_trajs[triples[i]], svo_trajs[triples[j]]
                per_layer_cos = {}
                for l in range(n_layers + 1):
                    if l in ta and l in tb:
                        per_layer_cos[l] = cosine_sim(ta[l], tb[l])
                key = f"{triples[i][1]}_vs_{triples[j][1]}_on_{s}_{o}"
                relation_type_results[key] = {
                    "per_layer_cosine": {str(k): v for k, v in per_layer_cos.items()},
                }

    # === Analysis 3: SVO vs entity-only ===
    # How much does adding relation change the trajectory from entity alone?
    log_time("  Analysis 3: SVO vs entity-only")

    svo_vs_entity_results = {}
    for (s, r, o), traj in svo_trajs.items():
        if s in control_trajs:
            ctrl = control_trajs[s]
            per_layer_cos = {}
            for l in range(n_layers + 1):
                if l in traj and l in ctrl:
                    per_layer_cos[l] = cosine_sim(traj[l], ctrl[l])
            svo_vs_entity_results[f"{s}_{r}_{o}_vs_{s}"] = {
                "per_layer_cosine": {str(k): v for k, v in per_layer_cos.items()},
            }

    # === Analysis 4: Subspace structure ===
    # Compare subspace of all SVO deltas vs entity deltas
    log_time("  Analysis 4: Subspace structure (SVO deltas vs entity deltas)")

    # Build delta matrices for SVO and entity-only
    svo_deltas = {}  # layer -> [d_model, n_svo]
    entity_deltas = {}  # layer -> [d_model, n_entity]

    for l in range(n_layers):
        svo_cols = []
        for (s, r, o), traj in svo_trajs.items():
            if l in traj and l + 1 in traj:
                svo_cols.append(traj[l + 1] - traj[l])
        if svo_cols:
            svo_deltas[l] = np.column_stack(svo_cols)

        entity_cols = []
        for word, traj in control_trajs.items():
            if l in traj and l + 1 in traj:
                entity_cols.append(traj[l + 1] - traj[l])
        if entity_cols:
            entity_deltas[l] = np.column_stack(entity_cols)

    subspace_overlaps = {}
    for l in range(n_layers):
        if l in svo_deltas and l in entity_deltas:
            k = min(5, svo_deltas[l].shape[1], entity_deltas[l].shape[1])
            if k >= 2:
                ov = subspace_overlap(svo_deltas[l], entity_deltas[l], k=k)
                subspace_overlaps[str(l)] = ov

    # === Analysis 5: Increment rank comparison ===
    # SVO increment matrix vs entity increment matrix — are they same rank structure?
    svo_ranks = {}
    entity_ranks = {}
    for l in range(n_layers):
        if l in svo_deltas:
            try:
                U, s, _ = np.linalg.svd(svo_deltas[l], full_matrices=False)
                total = np.sum(s ** 2)
                if total > 1e-20:
                    cumvar = np.cumsum(s ** 2) / total
                    svo_ranks[str(l)] = {
                        "var_top1": float(cumvar[0]),
                        "n_sig": int(np.sum(s ** 2 / total > 0.01)),
                    }
            except Exception:
                pass
        if l in entity_deltas:
            try:
                U, s, _ = np.linalg.svd(entity_deltas[l], full_matrices=False)
                total = np.sum(s ** 2)
                if total > 1e-20:
                    cumvar = np.cumsum(s ** 2) / total
                    entity_ranks[str(l)] = {
                        "var_top1": float(cumvar[0]),
                        "n_sig": int(np.sum(s ** 2 / total > 0.01)),
                    }
            except Exception:
                pass

    results = {
        "model": model_name,
        "n_svo_triples": len(svo_trajs),
        "n_controls": len(control_trajs),
        "swap_analysis": swap_results,
        "relation_type_analysis": relation_type_results,
        "svo_vs_entity": svo_vs_entity_results,
        "subspace_overlap": subspace_overlaps,
        "svo_increment_ranks": svo_ranks,
        "entity_increment_ranks": entity_ranks,
    }

    out_path = RESULT_DIR / f"{model_name}_block_a_relation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log_time(f"  SVO trajectories: {len(svo_trajs)}")
    log_time(f"  Swap pairs analyzed: {len(swap_results)}")
    log_time(f"  Relation type pairs: {len(relation_type_results)}")
    log_time(f"  SVO vs entity: {len(svo_vs_entity_results)}")

    if swap_results:
        for key, data in list(swap_results.items())[:2]:
            cos_vals = list(data["per_layer_cosine"].values())
            if cos_vals:
                log_time(f"  Swap '{key}': min_cos={min(cos_vals):.4f}, "
                         f"mid_cos={cos_vals[len(cos_vals)//2]:.4f}, "
                         f"last_cos={cos_vals[-1]:.4f}")

    if subspace_overlaps:
        for l in sorted(subspace_overlaps.keys(), key=int)[:5]:
            log_time(f"  Subspace overlap L{l}: {subspace_overlaps[l]:.4f}")

    return results


# ===== Block B: Composition Test =====

def block_b_composition(model, tokenizer, device, model_name, n_layers, d_model):
    """
    f(A+B) vs f(A)+f(B): Is composition linear or dynamic routing?
    Compare trajectory of "red apple" vs trajectory("red") + trajectory("apple")
    """
    log_time("=== Block B: Composition Test ===")

    # Extract trajectories for:
    # 1. Compound: "the red apple"
    # 2. Modifier alone: "the red"
    # 3. Noun alone: "the apple"

    compound_trajs = {}  # (mod, noun) -> {l: h}
    modifier_trajs = {}  # mod -> {l: h}
    noun_trajs = {}      # noun -> {l: h}

    log_time(f"  Extracting compound trajectories...")
    t0 = time.time()
    for i, (mod, noun) in enumerate(COMPOSITION_PAIRS):
        # Compound
        if mod in ["slowly", "quickly", "carefully", "never", "always", "barely"]:
            prompt = f"the {mod} {noun}s"  # adverb-verb
        else:
            prompt = f"the {mod} {noun}"  # adjective-noun
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            compound_trajs[(mod, noun)] = traj

        if (i + 1) % 6 == 0:
            elapsed = time.time() - t0
            log_time(f"    Compound progress: {i+1}/{len(COMPOSITION_PAIRS)}, elapsed={elapsed:.1f}s")

    log_time(f"  Extracting modifier trajectories...")
    for mod in set(m for m, _ in COMPOSITION_PAIRS):
        prompt = f"the {mod}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            modifier_trajs[mod] = traj

    log_time(f"  Extracting noun trajectories...")
    for noun in set(n for _, n in COMPOSITION_PAIRS):
        prompt = f"the {noun}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            noun_trajs[noun] = traj

    # === Analysis 1: Composition linearity test ===
    # f(compound) vs (f(modifier) + f(noun)) / 2
    # Use cosine to measure similarity
    log_time("  Analysis 1: Composition linearity")

    linearity_results = {}
    for (mod, noun), traj_c in compound_trajs.items():
        if mod in modifier_trajs and noun in noun_trajs:
            traj_m = modifier_trajs[mod]
            traj_n = noun_trajs[noun]

            per_layer = {}
            for l in range(n_layers + 1):
                if l in traj_c and l in traj_m and l in traj_n:
                    h_c = traj_c[l]
                    h_m = traj_m[l]
                    h_n = traj_n[l]

                    # Linear combination (equal weight)
                    h_linear = (h_m + h_n) / 2.0

                    cos_linear = cosine_sim(h_c, h_linear)
                    cos_mod = cosine_sim(h_c, h_m)
                    cos_noun = cosine_sim(h_c, h_n)

                    # Delta: compound - linear
                    delta = h_c - h_linear
                    delta_norm = float(np.linalg.norm(delta))
                    c_norm = float(np.linalg.norm(h_c))
                    relative_delta = delta_norm / max(c_norm, 1e-10)

                    per_layer[l] = {
                        "cos_vs_linear": cos_linear,
                        "cos_vs_modifier": cos_mod,
                        "cos_vs_noun": cos_noun,
                        "relative_delta_from_linear": relative_delta,
                    }

            linearity_results[f"{mod}_{noun}"] = per_layer

    # === Analysis 2: Nonlinearity profile ===
    # At which layers is composition most nonlinear?
    log_time("  Analysis 2: Nonlinearity profile")

    nonlinearity_profile = {}
    for l in range(n_layers + 1):
        rel_deltas = []
        for key, pl in linearity_results.items():
            if l in pl:
                rel_deltas.append(pl[l]["relative_delta_from_linear"])

        if rel_deltas:
            nonlinearity_profile[str(l)] = {
                "mean_rel_delta": float(np.mean(rel_deltas)),
                "std_rel_delta": float(np.std(rel_deltas)),
                "max_rel_delta": float(np.max(rel_deltas)),
                "n_pairs": len(rel_deltas),
            }

    # === Analysis 3: Direction of nonlinearity ===
    # At each layer, SVD the nonlinear residuals to see structure
    log_time("  Analysis 3: Nonlinearity direction structure")

    nonlinearity_dirs = {}
    for l in range(n_layers):
        residuals = []
        for (mod, noun), traj_c in compound_trajs.items():
            if mod in modifier_trajs and noun in noun_trajs:
                if l in traj_c and l in modifier_trajs[mod] and l in noun_trajs[noun]:
                    h_c = traj_c[l + 1] - traj_c[l]
                    h_m_delta = modifier_trajs[mod][l + 1] - modifier_trajs[mod][l]
                    h_n_delta = noun_trajs[noun][l + 1] - noun_trajs[noun][l]
                    residual = h_c - (h_m_delta + h_n_delta) / 2.0
                    residuals.append(residual)

        if len(residuals) >= 5:
            R = np.column_stack(residuals)  # [d_model, n]
            try:
                _, s, _ = np.linalg.svd(R, full_matrices=False)
                total = np.sum(s ** 2)
                if total > 1e-20:
                    cumvar = np.cumsum(s ** 2) / total
                    nonlinearity_dirs[str(l)] = {
                        "var_top1": float(cumvar[0]) if len(cumvar) > 0 else 0,
                        "var_top3": float(cumvar[2]) if len(cumvar) > 2 else 0,
                        "n_sig": int(np.sum(s ** 2 / total > 0.01)),
                        "total_residual_energy": float(total),
                    }
            except Exception:
                pass

    results = {
        "model": model_name,
        "n_compounds": len(compound_trajs),
        "n_modifiers": len(modifier_trajs),
        "n_nouns": len(noun_trajs),
        "linearity_per_pair": linearity_results,
        "nonlinearity_profile": nonlinearity_profile,
        "nonlinearity_directions": nonlinearity_dirs,
    }

    out_path = RESULT_DIR / f"{model_name}_block_b_composition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log_time(f"  Compounds: {len(compound_trajs)}, Linearity pairs: {len(linearity_results)}")

    if nonlinearity_profile:
        early = [l for l in range(min(5, n_layers + 1)) if str(l) in nonlinearity_profile]
        mid = [l for l in range(n_layers // 2 - 2, n_layers // 2 + 3) if str(l) in nonlinearity_profile]
        late = [l for l in range(max(0, n_layers - 4), n_layers + 1) if str(l) in nonlinearity_profile]

        e_nl = np.mean([nonlinearity_profile[str(l)]["mean_rel_delta"] for l in early]) if early else 0
        m_nl = np.mean([nonlinearity_profile[str(l)]["mean_rel_delta"] for l in mid]) if mid else 0
        l_nl = np.mean([nonlinearity_profile[str(l)]["mean_rel_delta"] for l in late]) if late else 0
        log_time(f"  Nonlinearity: early={e_nl:.4f}, mid={m_nl:.4f}, late={l_nl:.4f}")

    return results


# ===== Block C: Operator Dynamics =====

def block_c_operator(model, tokenizer, device, model_name, n_layers, d_model):
    """
    How do operators (not, if, because, every, some, will, can, must)
    change the computation compared to operand alone?
    """
    log_time("=== Block C: Operator Dynamics ===")

    # Extract trajectories for:
    # 1. Operator + operand: "not happy"
    # 2. Operand alone: "happy"
    # 3. Operator alone: "not"

    op_combined_trajs = {}  # (op, operand) -> {l: h}
    op_operand_trajs = {}   # operand -> {l: h}
    op_operator_trajs = {}  # operator -> {l: h}

    log_time(f"  Extracting operator+operand trajectories...")
    t0 = time.time()
    for i, (op, operand) in enumerate(OPERATOR_PAIRS):
        prompt = f"{op} {operand}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            op_combined_trajs[(op, operand)] = traj

        if (i + 1) % 6 == 0:
            elapsed = time.time() - t0
            log_time(f"    Op+Operand progress: {i+1}/{len(OPERATOR_PAIRS)}, elapsed={elapsed:.1f}s")

    log_time(f"  Extracting operand-alone trajectories...")
    for operand in OPERAND_ALONE:
        prompt = f"the {operand}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            op_operand_trajs[operand] = traj

    log_time(f"  Extracting operator-alone trajectories...")
    for op in OPERATOR_ALONE:
        prompt = f"{op}"
        traj = extract_trajectory(model, tokenizer, device, prompt)
        if traj is not None:
            op_operator_trajs[op] = traj

    # === Analysis 1: Operator effect on trajectory ===
    # Compare (op + operand) vs (operand alone)
    log_time("  Analysis 1: Operator effect")

    operator_effect = {}
    for (op, operand), traj_comb in op_combined_trajs.items():
        if operand in op_operand_trajs:
            traj_base = op_operand_trajs[operand]

            per_layer = {}
            for l in range(n_layers + 1):
                if l in traj_comb and l in traj_base:
                    h_c = traj_comb[l]
                    h_b = traj_base[l]
                    cos = cosine_sim(h_c, h_b)
                    norm_diff = float(np.linalg.norm(h_c - h_b))
                    relative_diff = norm_diff / max(np.linalg.norm(h_b), 1e-10)
                    per_layer[l] = {
                        "cosine": cos,
                        "relative_diff": relative_diff,
                        "norm_diff": norm_diff,
                    }

            operator_effect[f"{op}_{operand}"] = per_layer

    # === Analysis 2: Operator type signature ===
    # Group by operator type, compare effect profiles
    log_time("  Analysis 2: Operator type signatures")

    operator_signatures = defaultdict(list)
    for key, pl in operator_effect.items():
        op = key.split("_")[0]
        operator_signatures[op].append(pl)

    operator_summary = {}
    for op, all_pls in operator_signatures.items():
        # Mean relative_diff at key layers
        for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
            rel_diffs = []
            for pl in all_pls:
                if l in pl:
                    rel_diffs.append(pl[l]["relative_diff"])
            if rel_diffs:
                if op not in operator_summary:
                    operator_summary[op] = {}
                operator_summary[op][str(l)] = {
                    "mean_rel_diff": float(np.mean(rel_diffs)),
                    "n_instances": len(rel_diffs),
                }

    # === Analysis 3: Does operator negate or transform? ===
    # Compare "not happy" vs "sad" — is negation semantically equivalent?
    log_time("  Analysis 3: Negation-as-inversion test")

    negation_test = [
        (("not", "happy"), "sad"),
        (("not", "good"), "bad"),
        (("not", "true"), "false"),
    ]

    negation_results = {}
    for (op, operand), antonym in negation_test:
        key = f"{op}_{operand}"
        if key in operator_effect and antonym in op_operand_trajs:
            traj_comb = op_combined_trajs.get((op, operand))
            traj_anti = op_operand_trajs.get(antonym)

            if traj_comb and traj_anti:
                per_layer = {}
                for l in range(n_layers + 1):
                    if l in traj_comb and l in traj_anti:
                        cos = cosine_sim(traj_comb[l], traj_anti[l])
                        per_layer[l] = {"cosine_not_vs_antonym": cos}

                negation_results[key] = per_layer

    # === Analysis 4: Operator delta subspace ===
    # Delta = (op + operand) - (operand alone)
    # SVD of these deltas to see if operators share subspace
    log_time("  Analysis 4: Operator delta subspace")

    # Group deltas by operator
    operator_deltas = defaultdict(lambda: defaultdict(list))
    for (op, operand), traj_comb in op_combined_trajs.items():
        if operand in op_operand_trajs:
            traj_base = op_operand_trajs[operand]
            for l in range(n_layers):
                if l in traj_comb and l in traj_base and l + 1 in traj_comb and l + 1 in traj_base:
                    # Delta of increments
                    delta_comb = traj_comb[l + 1] - traj_comb[l]
                    delta_base = traj_base[l + 1] - traj_base[l]
                    # Residual = how operator changes the increment
                    residual = delta_comb - delta_base
                    operator_deltas[op][l].append(residual)

    # Compute subspace overlap between different operators at mid-layer
    mid_layer = n_layers // 2
    operator_overlap = {}
    op_names = sorted(operator_deltas.keys())
    for i in range(len(op_names)):
        for j in range(i + 1, len(op_names)):
            op_a, op_b = op_names[i], op_names[j]
            if mid_layer in operator_deltas[op_a] and mid_layer in operator_deltas[op_b]:
                cols_a = np.column_stack(operator_deltas[op_a][mid_layer])
                cols_b = np.column_stack(operator_deltas[op_b][mid_layer])
                k = min(3, cols_a.shape[1], cols_b.shape[1])
                if k >= 2:
                    ov = subspace_overlap(cols_a, cols_b, k=k)
                    operator_overlap[f"{op_a}_vs_{op_b}"] = ov

    results = {
        "model": model_name,
        "n_operator_pairs": len(op_combined_trajs),
        "n_operands": len(op_operand_trajs),
        "n_operators": len(op_operator_trajs),
        "operator_effect": operator_effect,
        "operator_signatures": operator_summary,
        "negation_test": negation_results,
        "operator_subspace_overlap": operator_overlap,
    }

    out_path = RESULT_DIR / f"{model_name}_block_c_operator.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"  Operator effects: {len(operator_effect)}")
    log_time(f"  Operator types: {list(operator_summary.keys())}")
    log_time(f"  Negation tests: {len(negation_results)}")
    log_time(f"  Operator overlaps: {len(operator_overlap)}")

    if operator_summary:
        for op in sorted(operator_summary.keys()):
            mid_l = str(n_layers // 2)
            if mid_l in operator_summary[op]:
                log_time(f"  {op} mid-layer rel_diff: {operator_summary[op][mid_l]['mean_rel_diff']:.4f}")

    if negation_results:
        for key, pl in negation_results.items():
            cos_vals = list(pl.values())
            if cos_vals:
                mean_cos = np.mean([v["cosine_not_vs_antonym"] for v in cos_vals])
                log_time(f"  Negation test '{key}': mean cos(not_X, antonym)={mean_cos:.4f}")

    return results


# ===== Block D: Recursive Closure =====

def block_d_recursion(model, tokenizer, device, model_name, n_layers, d_model):
    """
    Track trajectory changes as recursive depth increases.
    "the dog" → "the dog that chased the cat" → nested
    """
    log_time("=== Block D: Recursive Closure ===")

    # Extract trajectories for all recursive sentences
    recursive_trajs = {}  # sentence -> {l: h}

    log_time(f"  Extracting {len(RECURSIVE_SENTENCES)} recursive trajectories...")
    t0 = time.time()
    for i, sent in enumerate(RECURSIVE_SENTENCES):
        traj = extract_trajectory(model, tokenizer, device, sent)
        if traj is not None:
            recursive_trajs[sent] = traj

        if (i + 1) % 3 == 0:
            elapsed = time.time() - t0
            log_time(f"    Recursive progress: {i+1}/{len(RECURSIVE_SENTENCES)}, elapsed={elapsed:.1f}s")

    # === Analysis 1: Trajectory divergence with depth ===
    # Compare base vs each level of recursion
    log_time("  Analysis 1: Trajectory divergence vs recursion depth")

    # Group by type
    relative_groups = [
        ["the dog", "the dog that chased the cat",
         "the dog that chased the cat that ate the fish",
         "the dog that chased the cat that ate the fish that swam in the river"],
    ]
    prep_groups = [
        ["the man", "the man in the house",
         "the man in the house on the hill",
         "the man in the house on the hill by the river"],
    ]
    comp_groups = [
        ["I think", "I think that she knows",
         "I think that she knows that he left",
         "I think that she knows that he left that we forgot"],
    ]

    all_groups = [("relative", relative_groups), ("prepositional", prep_groups), ("complement", comp_groups)]

    recursion_divergence = {}
    for gtype, groups in all_groups:
        for group in groups:
            base_sent = group[0]
            if base_sent not in recursive_trajs:
                continue

            base_traj = recursive_trajs[base_sent]

            for depth, sent in enumerate(group):
                if sent not in recursive_trajs:
                    continue
                traj = recursive_trajs[sent]

                per_layer = {}
                for l in range(n_layers + 1):
                    if l in base_traj and l in traj:
                        cos = cosine_sim(base_traj[l], traj[l])
                        norm_diff = float(np.linalg.norm(traj[l] - base_traj[l]))
                        relative_diff = norm_diff / max(np.linalg.norm(base_traj[l]), 1e-10)
                        per_layer[l] = {
                            "cosine": cos,
                            "relative_diff": relative_diff,
                        }

                recursion_divergence[f"{gtype}_d{depth}"] = {
                    "depth": depth,
                    "sentence": sent[:60],
                    "per_layer": per_layer,
                }

    # === Analysis 2: Increment dimension vs depth ===
    # Does deeper recursion increase the rank of the increment?
    log_time("  Analysis 2: Increment dimension vs depth")

    increment_ranks = {}
    for gtype, groups in all_groups:
        for group in groups:
            for depth, sent in enumerate(group):
                if sent not in recursive_trajs:
                    continue
                traj = recursive_trajs[sent]

                per_layer_rank = {}
                for l in range(n_layers):
                    if l in traj and l + 1 in traj:
                        delta = traj[l + 1] - traj[l]
                        per_layer_rank[l] = float(np.linalg.norm(delta))

                increment_ranks[f"{gtype}_d{depth}"] = {
                    "depth": depth,
                    "per_layer_delta_norm": {str(k): v for k, v in per_layer_rank.items()},
                }

    # === Analysis 3: Consecutive depth delta ===
    # Compare each level to the previous level (not just base)
    log_time("  Analysis 3: Consecutive depth delta")

    consecutive_delta = {}
    for gtype, groups in all_groups:
        for group in groups:
            for depth in range(1, len(group)):
                sent_prev = group[depth - 1]
                sent_curr = group[depth]
                if sent_prev not in recursive_trajs or sent_curr not in recursive_trajs:
                    continue

                traj_prev = recursive_trajs[sent_prev]
                traj_curr = recursive_trajs[sent_curr]

                per_layer = {}
                for l in range(n_layers + 1):
                    if l in traj_prev and l in traj_curr:
                        cos = cosine_sim(traj_prev[l], traj_curr[l])
                        relative_diff = float(np.linalg.norm(traj_curr[l] - traj_prev[l])) / \
                                       max(np.linalg.norm(traj_prev[l]), 1e-10)
                        per_layer[l] = {
                            "cosine": cos,
                            "relative_diff": relative_diff,
                        }

                consecutive_delta[f"{gtype}_d{depth-1}_to_d{depth}"] = {
                    "depth_transition": f"d{depth-1}->d{depth}",
                    "per_layer": per_layer,
                }

    # === Analysis 4: Trajectory curvature ===
    # Does deeper recursion increase trajectory curvature (change in direction)?
    log_time("  Analysis 4: Trajectory curvature")

    trajectory_curvature = {}
    for gtype, groups in all_groups:
        for group in groups:
            for depth, sent in enumerate(group):
                if sent not in recursive_trajs:
                    continue
                traj = recursive_trajs[sent]

                curvatures = {}
                for l in range(1, n_layers):
                    if l - 1 in traj and l in traj and l + 1 in traj:
                        d1 = traj[l] - traj[l - 1]
                        d2 = traj[l + 1] - traj[l]
                        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos_turn = cosine_sim(d1, d2)
                            curvatures[l] = 1.0 - cos_turn  # 0 = straight, 2 = reversal

                trajectory_curvature[f"{gtype}_d{depth}"] = {
                    "depth": depth,
                    "mean_curvature": float(np.mean(list(curvatures.values()))) if curvatures else 0,
                    "max_curvature": float(max(curvatures.values())) if curvatures else 0,
                    "per_layer_curvature": {str(k): v for k, v in curvatures.items()},
                }

    results = {
        "model": model_name,
        "n_sentences": len(recursive_trajs),
        "recursion_divergence": recursion_divergence,
        "increment_ranks": increment_ranks,
        "consecutive_delta": consecutive_delta,
        "trajectory_curvature": trajectory_curvature,
    }

    out_path = RESULT_DIR / f"{model_name}_block_d_recursion.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log_time(f"  Sentences: {len(recursive_trajs)}")

    for key, data in trajectory_curvature.items():
        log_time(f"  Curvature '{key}': mean={data['mean_curvature']:.4f}, max={data['max_curvature']:.4f}")

    # Show divergence from base at mid and late layers
    for key, data in recursion_divergence.items():
        mid_l = str(n_layers // 2)
        last_l = str(n_layers)
        if mid_l in data.get("per_layer", {}):
            log_time(f"  Divergence '{key}': mid_cos={data['per_layer'][mid_l]['cosine']:.4f}, "
                     f"last_cos={data['per_layer'].get(last_l, {}).get('cosine', 'N/A')}")

    return results


# ===== Main =====

def main():
    global _log_file

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    log_path = RESULT_DIR / f"{model_name}_phase279.log"
    _log_file = str(log_path)

    log_time(f"Phase 279: Compositional Computational Topology")
    log_time(f"Model: {model_name}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    log_time(f"Model info: n_layers={n_layers}, d_model={d_model}, class={info.model_class}")

    # Block A: Relation Dynamics
    log_time("=" * 60)
    results_a = block_a_relation(model, tokenizer, device, model_name, n_layers, d_model)

    # Block B: Composition Test
    log_time("=" * 60)
    results_b = block_b_composition(model, tokenizer, device, model_name, n_layers, d_model)

    # Block C: Operator Dynamics
    log_time("=" * 60)
    results_c = block_c_operator(model, tokenizer, device, model_name, n_layers, d_model)

    # Block D: Recursive Closure
    log_time("=" * 60)
    results_d = block_d_recursion(model, tokenizer, device, model_name, n_layers, d_model)

    # Final Summary
    log_time("=" * 60)
    log_time("PHASE 279 OBJECTIVE RESULTS")
    log_time("=" * 60)

    # Block A summary
    log_time("Block A — Relation:")
    swap = results_a.get("swap_analysis", {})
    for key, data in list(swap.items())[:2]:
        cos_vals = [v for v in data["per_layer_cosine"].values()]
        if cos_vals:
            log_time(f"  Swap '{key}': range=[{min(cos_vals):.4f}, {max(cos_vals):.4f}]")

    # Block B summary
    log_time("Block B — Composition:")
    nl = results_b.get("nonlinearity_profile", {})
    if nl:
        key_layers = ["0", str(n_layers // 4), str(n_layers // 2), str(3 * n_layers // 4), str(n_layers)]
        for kl in key_layers:
            if kl in nl:
                log_time(f"  L{kl}: mean_rel_delta={nl[kl]['mean_rel_delta']:.4f}")

    # Block C summary
    log_time("Block C — Operator:")
    sigs = results_c.get("operator_signatures", {})
    for op in sorted(sigs.keys()):
        mid = str(n_layers // 2)
        if mid in sigs[op]:
            log_time(f"  {op} mid rel_diff: {sigs[op][mid]['mean_rel_diff']:.4f}")

    # Block D summary
    log_time("Block D — Recursion:")
    for key, data in results_d.get("trajectory_curvature", {}).items():
        log_time(f"  {key}: curvature={data['mean_curvature']:.4f}")

    # Release
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released. Phase 279 complete.")


if __name__ == "__main__":
    main()
