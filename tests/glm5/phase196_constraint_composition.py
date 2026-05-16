"""
Phase 196: Constraint Composition Analysis
============================================

核心转向: 从"单约束描述"到"约束组合的数学结构"

用户关键批评:
1. 仍在"对象化ΔS" — 应停止寻找"语义原子"
2. 把训练统计误认为语言本体 — 需要更严格的因果推理
3. Logit Lens不可靠 — 只使用最终输出分布
4. 默认"功能可局部化" — 可能是全息分布式计算

关键实验: 约束组合 (Constraint Composition)
- 如果语义 = 约束, 那么约束如何组合?
- 加性: KL(AB) ≈ KL(A) + KL(B)?
- 支配性: KL(AB) ≈ max(KL(A), KL(B))?
- 超加性: KL(AB) > KL(A) + KL(B)? (交互放大)
- 亚加性: KL(AB) < KL(A) + KL(B)? (冗余)

4个实验:
Exp1: Single Constraint Profile (最终输出分布, 含Renyi熵)
      - KL散度, Shannon熵, Renyi熵(α=0,2,∞), 有效分支因子
      - 无logit lens, 只用最终输出
Exp2: Constraint Composition
      - negation+question: "Does the cat not chase the dog?"
      - question+role_binding: "Does the dog chase the cat?"
      - negation+role_binding: "The dog does not chase the cat"
      - 测试: KL(AB) vs KL(A)+KL(B), token-level组合
Exp3: Branching Structure (Renyi熵谱)
      - H_0 = log(|support|) — 可能的续写数
      - H_1 = Shannon熵 — 平均不确定性
      - H_2 = collision熵 — 有效可能结果数
      - H_∞ = min-entropy — 最可能结果
      - 有效分支因子 = exp(H_2)
Exp4: Constraint Independence (逐样本)
      - 每个样本上的约束强度
      - 约束之间的相关性
      - 是否存在"约束类型"的聚类

数据量: 30句对 (Qwen3), 20句对 (GLM4/DS7B)
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from model_utils import get_model_info, get_layers, release_model, MODEL_CONFIGS

# ===== 30个基础句 (transitive, 用于所有变换+组合) =====
# 格式: (base, negation, question, role_binding, negation+question, question+role_binding, negation+role_binding)
SENTENCE_SEXTETS = [
    # (base, negation, question, role_binding, neg+q, q+rb, neg+rb)
    ("The cat chases the dog",
     "The cat does not chase the dog",
     "Does the cat chase the dog?",
     "The dog chases the cat",
     "Does the cat not chase the dog?",
     "Does the dog chase the cat?",
     "The dog does not chase the cat"),
    ("The teacher helps the student",
     "The teacher does not help the student",
     "Does the teacher help the student?",
     "The student helps the teacher",
     "Does the teacher not help the student?",
     "Does the student help the teacher?",
     "The student does not help the teacher"),
    ("The leader guides the team",
     "The leader does not guide the team",
     "Does the leader guide the team?",
     "The team guides the leader",
     "Does the leader not guide the team?",
     "Does the team guide the leader?",
     "The team does not guide the leader"),
    ("The parent protects the child",
     "The parent does not protect the child",
     "Does the parent protect the child?",
     "The child protects the parent",
     "Does the parent not protect the child?",
     "Does the child protect the parent?",
     "The child does not protect the parent"),
    ("The writer inspires the reader",
     "The writer does not inspire the reader",
     "Does the writer inspire the reader?",
     "The reader inspires the writer",
     "Does the writer not inspire the reader?",
     "Does the reader inspire the writer?",
     "The reader does not inspire the writer"),
    ("The coach trains the athlete",
     "The coach does not train the athlete",
     "Does the coach train the athlete?",
     "The athlete trains the coach",
     "Does the coach not train the athlete?",
     "Does the athlete train the coach?",
     "The athlete does not train the coach"),
    ("The manager evaluates the employee",
     "The manager does not evaluate the employee",
     "Does the manager evaluate the employee?",
     "The employee evaluates the manager",
     "Does the manager not evaluate the employee?",
     "Does the employee evaluate the manager?",
     "The employee does not evaluate the manager"),
    ("The doctor advises the patient",
     "The doctor does not advise the patient",
     "Does the doctor advise the patient?",
     "The patient advises the doctor",
     "Does the doctor not advise the patient?",
     "Does the patient advise the doctor?",
     "The patient does not advise the doctor"),
    ("The buyer pays the seller",
     "The buyer does not pay the seller",
     "Does the buyer pay the seller?",
     "The seller pays the buyer",
     "Does the buyer not pay the seller?",
     "Does the seller pay the buyer?",
     "The seller does not pay the buyer"),
    ("The host welcomes the guest",
     "The host does not welcome the guest",
     "Does the host welcome the guest?",
     "The guest welcomes the host",
     "Does the host not welcome the guest?",
     "Does the guest welcome the host?",
     "The guest does not welcome the host"),
    ("The soldier follows the general",
     "The soldier does not follow the general",
     "Does the soldier follow the general?",
     "The general follows the soldier",
     "Does the soldier not follow the general?",
     "Does the general follow the soldier?",
     "The general does not follow the soldier"),
    ("The artist influences the critic",
     "The artist does not influence the critic",
     "Does the artist influence the critic?",
     "The critic influences the artist",
     "Does the artist not influence the critic?",
     "Does the critic influence the artist?",
     "The critic does not influence the artist"),
    ("The driver transports the passenger",
     "The driver does not transport the passenger",
     "Does the driver transport the passenger?",
     "The passenger transports the driver",
     "Does the driver not transport the passenger?",
     "Does the passenger transport the driver?",
     "The passenger does not transport the driver"),
    ("The chef feeds the guest",
     "The chef does not feed the guest",
     "Does the chef feed the guest?",
     "The guest feeds the chef",
     "Does the chef not feed the guest?",
     "Does the guest feed the chef?",
     "The guest does not feed the chef"),
    ("The singer entertains the crowd",
     "The singer does not entertain the crowd",
     "Does the singer entertain the crowd?",
     "The crowd entertains the singer",
     "Does the singer not entertain the crowd?",
     "Does the crowd entertain the singer?",
     "The crowd does not entertain the singer"),
    ("The judge questions the witness",
     "The judge does not question the witness",
     "Does the judge question the witness?",
     "The witness questions the judge",
     "Does the judge not question the witness?",
     "Does the witness question the judge?",
     "The witness does not question the judge"),
    ("The editor corrects the author",
     "The editor does not correct the author",
     "Does the editor correct the author?",
     "The author corrects the editor",
     "Does the editor not correct the author?",
     "Does the author correct the editor?",
     "The author does not correct the editor"),
    ("The therapist helps the client",
     "The therapist does not help the client",
     "Does the therapist help the client?",
     "The client helps the therapist",
     "Does the therapist not help the client?",
     "Does the client help the therapist?",
     "The client does not help the therapist"),
    ("The company hires the worker",
     "The company does not hire the worker",
     "Does the company hire the worker?",
     "The worker hires the company",
     "Does the company not hire the worker?",
     "Does the worker hire the company?",
     "The worker does not hire the company"),
    ("The government regulates the market",
     "The government does not regulate the market",
     "Does the government regulate the market?",
     "The market regulates the government",
     "Does the government not regulate the market?",
     "Does the market regulate the government?",
     "The market does not regulate the government"),
    ("The engine drives the machine",
     "The engine does not drive the machine",
     "Does the engine drive the machine?",
     "The machine drives the engine",
     "Does the engine not drive the machine?",
     "Does the machine drive the engine?",
     "The machine does not drive the engine"),
    ("The pilot flies the plane",
     "The pilot does not fly the plane",
     "Does the pilot fly the plane?",
     "The plane flies the pilot",
     "Does the pilot not fly the plane?",
     "Does the plane fly the pilot?",
     "The plane does not fly the pilot"),
    ("The farmer grows the crop",
     "The farmer does not grow the crop",
     "Does the farmer grow the crop?",
     "The crop grows the farmer",
     "Does the farmer not grow the crop?",
     "Does the crop grow the farmer?",
     "The crop does not grow the farmer"),
    ("The builder constructs the house",
     "The builder does not construct the house",
     "Does the builder construct the house?",
     "The house constructs the builder",
     "Does the builder not construct the house?",
     "Does the house construct the builder?",
     "The house does not construct the builder"),
    ("The scientist discovers the truth",
     "The scientist does not discover the truth",
     "Does the scientist discover the truth?",
     "The truth discovers the scientist",
     "Does the scientist not discover the truth?",
     "Does the truth discover the scientist?",
     "The truth does not discover the scientist"),
    ("The teacher encourages the student",
     "The teacher does not encourage the student",
     "Does the teacher encourage the student?",
     "The student encourages the teacher",
     "Does the teacher not encourage the student?",
     "Does the student encourage the teacher?",
     "The student does not encourage the teacher"),
    ("The river nourishes the valley",
     "The river does not nourish the valley",
     "Does the river nourish the valley?",
     "The valley nourishes the river",
     "Does the river not nourish the valley?",
     "Does the valley nourish the river?",
     "The valley does not nourish the river"),
    ("The system controls the process",
     "The system does not control the process",
     "Does the system control the process?",
     "The process controls the system",
     "Does the system not control the process?",
     "Does the process control the system?",
     "The process does not control the system"),
    ("The author writes the book",
     "The author does not write the book",
     "Does the author write the book?",
     "The book writes the author",
     "Does the author not write the book?",
     "Does the book write the author?",
     "The book does not write the author"),
    ("The musician plays the instrument",
     "The musician does not play the instrument",
     "Does the musician play the instrument?",
     "The instrument plays the musician",
     "Does the musician not play the instrument?",
     "Does the instrument play the musician?",
     "The instrument does not play the musician"),
]

# 功能标签
FUNC_LABELS = {
    "base": "Base",
    "negation": "Negation",
    "question": "Question",
    "role_binding": "RoleBinding",
    "neg_plus_q": "Neg+Q",
    "q_plus_rb": "Q+RoleBind",
    "neg_plus_rb": "Neg+RoleBind",
}

# 功能索引 (对应SENTENCE_SEXTETS中的位置)
FUNC_INDICES = {
    "base": 0,
    "negation": 1,
    "question": 2,
    "role_binding": 3,
    "neg_plus_q": 4,
    "q_plus_rb": 5,
    "neg_plus_rb": 6,
}

# 组合定义: (funcA, funcB, combined)
COMBINATIONS = [
    ("negation", "question", "neg_plus_q"),
    ("question", "role_binding", "q_plus_rb"),
    ("negation", "role_binding", "neg_plus_rb"),
]

TOP_K = 20  # top-k tokens to analyze
PROB_THRESHOLD = 1e-4  # minimum probability for "significant" tokens


def load_model_with_flash(model_name):
    """加载模型, GLM4/DS7B用device_map=auto, 尝试flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash attention, 失败则fallback
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        print(f"[load] {model_name} loaded with flash_attention_2")
    except Exception as e:
        print(f"[load] flash_attention_2 failed ({e}), falling back to eager")
        attn_impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


def get_output_distribution(model, tokenizer, text, device):
    """
    获取最终输出分布 — 只用最后一层, 不用logit lens

    Returns:
        probs: np.array [vocab_size]
        entropy: float (Shannon entropy)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False)
        # outputs.logits: [1, seq_len, vocab_size]
        logits = outputs.logits[0, -1, :].float()  # last token position

    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()  # [vocab_size]

    # Shannon entropy
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))

    return probs, entropy


def compute_renyi_entropy(probs, alpha):
    """
    Renyi entropy: H_α = (1/(1-α)) * log(Σ p_i^α)

    Special cases:
    - α→0: H_0 = log(|support|) — number of possible outcomes
    - α=1: H_1 = Shannon entropy (use limit)
    - α=2: H_2 = -log(Σ p_i²) — collision entropy
    - α→∞: H_∞ = -log(max(p_i)) — min-entropy
    """
    if alpha == 0:
        # H_0 = log(|support|) — number of tokens with p > threshold
        support = np.sum(probs > PROB_THRESHOLD)
        return float(np.log(max(support, 1)))
    elif alpha == 1:
        # Shannon entropy
        p = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))
    elif alpha == float('inf'):
        # Min-entropy
        return float(-np.log(max(np.max(probs), 1e-10)))
    else:
        # General Renyi entropy
        p_alpha = np.sum(np.power(np.clip(probs, 1e-20, 1.0), alpha))
        if p_alpha <= 0:
            return 0.0
        return float((1.0 / (1.0 - alpha)) * np.log(p_alpha))


def compute_kl_divergence(p, q):
    """KL(p || q) — p是reference分布, q是query分布"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def compute_js_divergence(p, q):
    """JS divergence (symmetric)"""
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def get_promoted_suppressed_tokens(base_probs, trans_probs, tokenizer, top_k=TOP_K):
    """
    获取promoted和suppressed tokens

    Returns:
        promoted: list of (token_text, delta_prob, prob_in_trans)
        suppressed: list of (token_text, delta_prob, prob_in_base)
    """
    delta = trans_probs - base_probs  # [vocab_size]

    # Promoted: delta > 0, sorted by delta descending
    promoted_ids = np.where(delta > 0)[0]
    promoted_deltas = delta[promoted_ids]
    top_promoted_idx = np.argsort(promoted_deltas)[-top_k:][::-1]

    promoted = []
    for idx in top_promoted_idx:
        tid = int(promoted_ids[idx])
        text = tokenizer.decode([tid]).strip()
        promoted.append({
            "text": text,
            "delta": float(delta[tid]),
            "prob_trans": float(trans_probs[tid]),
        })

    # Suppressed: delta < 0, sorted by |delta| descending
    suppressed_ids = np.where(delta < 0)[0]
    suppressed_deltas = delta[suppressed_ids]
    top_suppressed_idx = np.argsort(np.abs(suppressed_deltas))[-top_k:][::-1]

    suppressed = []
    for idx in top_suppressed_idx:
        tid = int(suppressed_ids[idx])
        text = tokenizer.decode([tid]).strip()
        suppressed.append({
            "text": text,
            "delta": float(delta[tid]),
            "prob_base": float(base_probs[tid]),
        })

    return promoted, suppressed


def compute_jaccard(set_a, set_b):
    """Jaccard overlap of two sets"""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def run_experiment(model_name):
    """运行Phase 196完整实验"""
    t0_total = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 196: Constraint Composition Analysis — {model_name}")
    print(f"{'='*70}")

    # 加载模型
    model, tokenizer, device = load_model_with_flash(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    vocab_size = info.vocab_size
    print(f"  n_layers={n_layers}, d_model={info.d_model}, vocab={vocab_size}")
    sys.stdout.flush()

    # 数据量
    if model_name == 'qwen3':
        n_pairs = 30
    else:
        n_pairs = 20
    n_pairs = min(n_pairs, len(SENTENCE_SEXTETS))
    print(f"  n_pairs={n_pairs}")
    sys.stdout.flush()

    # ===== 收集所有分布 =====
    # dist_storage[func_type][pair_idx] = probs (np.array [vocab_size])
    dist_storage = defaultdict(dict)
    entropy_storage = defaultdict(dict)  # Shannon entropy
    renyi_storage = defaultdict(dict)     # Renyi entropies

    all_funcs = list(FUNC_INDICES.keys())
    total_texts = n_pairs * len(all_funcs)
    text_count = 0
    t0_collect = time.time()

    print(f"\n--- Collecting output distributions ({total_texts} texts) ---")
    sys.stdout.flush()

    for pair_idx in range(n_pairs):
        sextet = SENTENCE_SEXTETS[pair_idx]
        for func_type in all_funcs:
            text_idx = FUNC_INDICES[func_type]
            text = sextet[text_idx]

            probs, entropy = get_output_distribution(model, tokenizer, text, device)
            dist_storage[func_type][pair_idx] = probs
            entropy_storage[func_type][pair_idx] = entropy

            # Renyi entropies
            renyi_storage[func_type][pair_idx] = {
                "H0": compute_renyi_entropy(probs, 0),
                "H1": entropy,  # Shannon
                "H2": compute_renyi_entropy(probs, 2),
                "H_inf": compute_renyi_entropy(probs, float('inf')),
            }

            text_count += 1
            if text_count % 20 == 0:
                elapsed = time.time() - t0_collect
                rate = text_count / elapsed
                remaining = (total_texts - text_count) / rate
                print(f"  [{text_count}/{total_texts}] {rate:.1f} texts/s, "
                      f"~{remaining:.0f}s remaining, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
                sys.stdout.flush()

    elapsed_collect = time.time() - t0_collect
    print(f"\n  Collection done in {elapsed_collect:.1f}s")
    sys.stdout.flush()

    # ===== Exp1: Single Constraint Profile =====
    print(f"\n{'='*70}")
    print(f"Exp1: Single Constraint Profile (Final Output Only)")
    print(f"{'='*70}")

    single_funcs = ["negation", "question", "role_binding"]
    exp1_results = {}

    for func in single_funcs:
        kl_values = []
        delta_entropy = []
        renyi_deltas = {"H0": [], "H1": [], "H2": [], "H_inf": []}
        effective_branching = []
        all_promoted_texts = set()
        all_suppressed_texts = set()

        for pair_idx in range(n_pairs):
            base_probs = dist_storage["base"][pair_idx]
            trans_probs = dist_storage[func][pair_idx]

            # KL divergence
            kl = compute_kl_divergence(base_probs, trans_probs)
            kl_values.append(kl)

            # Entropy change
            base_e = entropy_storage["base"][pair_idx]
            trans_e = entropy_storage[func][pair_idx]
            delta_entropy.append(trans_e - base_e)

            # Renyi entropy changes
            for rkey in ["H0", "H1", "H2", "H_inf"]:
                base_r = renyi_storage["base"][pair_idx][rkey]
                trans_r = renyi_storage[func][pair_idx][rkey]
                renyi_deltas[rkey].append(trans_r - base_r)

            # Effective branching factor = exp(H_2)
            trans_h2 = renyi_storage[func][pair_idx]["H2"]
            effective_branching.append(float(np.exp(trans_h2)))

            # Top promoted/suppressed
            promoted, suppressed = get_promoted_suppressed_tokens(
                base_probs, trans_probs, tokenizer)
            all_promoted_texts.update([t["text"] for t in promoted[:5]])
            all_suppressed_texts.update([t["text"] for t in suppressed[:5]])

        exp1_results[func] = {
            "mean_KL": float(np.mean(kl_values)),
            "std_KL": float(np.std(kl_values)),
            "median_KL": float(np.median(kl_values)),
            "mean_delta_entropy": float(np.mean(delta_entropy)),
            "std_delta_entropy": float(np.std(delta_entropy)),
            "renyi_deltas": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                           for k, v in renyi_deltas.items()},
            "mean_effective_branching": float(np.mean(effective_branching)),
            "top_promoted": list(all_promoted_texts),
            "top_suppressed": list(all_suppressed_texts),
        }

        print(f"\n  {FUNC_LABELS[func]}:")
        print(f"    KL(base||{func}): mean={exp1_results[func]['mean_KL']:.4f}, "
              f"std={exp1_results[func]['std_KL']:.4f}, "
              f"median={exp1_results[func]['median_KL']:.4f}")
        print(f"    ΔH (Shannon): mean={exp1_results[func]['mean_delta_entropy']:.4f}, "
              f"std={exp1_results[func]['std_delta_entropy']:.4f}")
        print(f"    ΔH0 (support): mean={exp1_results[func]['renyi_deltas']['H0']['mean']:.4f}")
        print(f"    ΔH2 (collision): mean={exp1_results[func]['renyi_deltas']['H2']['mean']:.4f}")
        print(f"    ΔH∞ (min-ent): mean={exp1_results[func]['renyi_deltas']['H_inf']['mean']:.4f}")
        print(f"    Effective branching: {exp1_results[func]['mean_effective_branching']:.1f}")
        print(f"    Top promoted: {', '.join(list(all_promoted_texts)[:8])}")
        print(f"    Top suppressed: {', '.join(list(all_suppressed_texts)[:8])}")

    sys.stdout.flush()

    # ===== Exp2: Constraint Composition =====
    print(f"\n{'='*70}")
    print(f"Exp2: Constraint Composition")
    print(f"{'='*70}")

    exp2_results = {}

    for funcA, funcB, funcAB in COMBINATIONS:
        kl_A_values = []
        kl_B_values = []
        kl_AB_values = []
        interaction_values = []  # KL(AB) - KL(A) - KL(B)
        interaction_ratio = []  # KL(AB) / (KL(A) + KL(B))

        # Token-level composition
        promoted_A_all = []
        promoted_B_all = []
        promoted_AB_all = []
        jaccard_promoted_AB_vs_union = []

        for pair_idx in range(n_pairs):
            base_probs = dist_storage["base"][pair_idx]
            probs_A = dist_storage[funcA][pair_idx]
            probs_B = dist_storage[funcB][pair_idx]
            probs_AB = dist_storage[funcAB][pair_idx]

            kl_A = compute_kl_divergence(base_probs, probs_A)
            kl_B = compute_kl_divergence(base_probs, probs_B)
            kl_AB = compute_kl_divergence(base_probs, probs_AB)

            kl_A_values.append(kl_A)
            kl_B_values.append(kl_B)
            kl_AB_values.append(kl_AB)

            # Interaction
            interaction = kl_AB - (kl_A + kl_B)
            interaction_values.append(interaction)

            denom = kl_A + kl_B
            if denom > 0.01:
                interaction_ratio.append(kl_AB / denom)
            else:
                interaction_ratio.append(float('nan'))

            # Token-level: promoted tokens
            delta_A = probs_A - base_probs
            delta_B = probs_B - base_probs
            delta_AB = probs_AB - base_probs

            # Significant promotion: delta > threshold
            promoted_A_set = set(np.where(delta_A > PROB_THRESHOLD)[0])
            promoted_B_set = set(np.where(delta_B > PROB_THRESHOLD)[0])
            promoted_AB_set = set(np.where(delta_AB > PROB_THRESHOLD)[0])

            promoted_A_all.append(promoted_A_set)
            promoted_B_all.append(promoted_B_set)
            promoted_AB_all.append(promoted_AB_set)

            # Jaccard: promoted_AB vs union(A, B)
            union_AB = promoted_A_set | promoted_B_set
            jacc = compute_jaccard(promoted_AB_set, union_AB)
            jaccard_promoted_AB_vs_union.append(jacc)

        # Aggregate
        mean_kl_A = float(np.mean(kl_A_values))
        mean_kl_B = float(np.mean(kl_B_values))
        mean_kl_AB = float(np.mean(kl_AB_values))
        mean_interaction = float(np.mean(interaction_values))
        mean_ratio = float(np.nanmean(interaction_ratio))

        # Classification
        if abs(mean_interaction) < 0.1 * (mean_kl_A + mean_kl_B):
            comp_type = "ADDITIVE"
        elif mean_interaction > 0.1 * (mean_kl_A + mean_kl_B):
            comp_type = "SUPER-ADDITIVE (interaction amplifies)"
        else:
            comp_type = "SUB-ADDITIVE (redundant)"

        # Token-level overlap
        mean_jaccard = float(np.mean(jaccard_promoted_AB_vs_union))

        # Per-sample analysis: what fraction of samples are super/additive/sub?
        n_super = sum(1 for i, iv in enumerate(interaction_values)
                      if iv > 0.1 * (kl_A_values[i] + kl_B_values[i]))
        n_sub = sum(1 for i, iv in enumerate(interaction_values)
                     if iv < -0.1 * (kl_A_values[i] + kl_B_values[i]))
        n_add = len(interaction_values) - n_super - n_sub

        exp2_results[f"{funcA}+{funcB}"] = {
            "funcA": funcA,
            "funcB": funcB,
            "funcAB": funcAB,
            "mean_KL_A": mean_kl_A,
            "mean_KL_B": mean_kl_B,
            "mean_KL_AB": mean_kl_AB,
            "mean_interaction": mean_interaction,
            "mean_interaction_ratio": mean_ratio,
            "composition_type": comp_type,
            "mean_jaccard_promoted_vs_union": mean_jaccard,
            "per_sample_KL_A": [float(x) for x in kl_A_values],
            "per_sample_KL_B": [float(x) for x in kl_B_values],
            "per_sample_KL_AB": [float(x) for x in kl_AB_values],
            "per_sample_interaction": [float(x) for x in interaction_values],
        }

        print(f"\n  {FUNC_LABELS[funcA]} + {FUNC_LABELS[funcB]}:")
        print(f"    KL(A) = {mean_kl_A:.4f}")
        print(f"    KL(B) = {mean_kl_B:.4f}")
        print(f"    KL(AB) = {mean_kl_AB:.4f}")
        print(f"    Expected (additive) = {mean_kl_A + mean_kl_B:.4f}")
        print(f"    Interaction = {mean_interaction:.4f} ({'+' if mean_interaction > 0 else ''}{mean_interaction/(mean_kl_A + mean_kl_B)*100:.1f}% of additive)")
        print(f"    Ratio KL(AB)/(KL(A)+KL(B)) = {mean_ratio:.4f}")
        print(f"    ★ Composition: {comp_type}")
        print(f"    Token Jaccard(AB, A∪B) = {mean_jaccard:.4f}")

    sys.stdout.flush()

    # ===== Exp3: Branching Structure (Renyi Entropy Spectrum) =====
    print(f"\n{'='*70}")
    print(f"Exp3: Branching Structure (Renyi Entropy Spectrum)")
    print(f"{'='*70}")

    all_funcs_for_branching = ["base", "negation", "question", "role_binding",
                                "neg_plus_q", "q_plus_rb", "neg_plus_rb"]
    exp3_results = {}

    for func in all_funcs_for_branching:
        h0_vals = []
        h1_vals = []
        h2_vals = []
        hinf_vals = []
        branch_vals = []

        for pair_idx in range(n_pairs):
            r = renyi_storage[func][pair_idx]
            h0_vals.append(r["H0"])
            h1_vals.append(r["H1"])
            h2_vals.append(r["H2"])
            hinf_vals.append(r["H_inf"])
            branch_vals.append(float(np.exp(r["H2"])))  # effective branching

        exp3_results[func] = {
            "H0_mean": float(np.mean(h0_vals)),
            "H1_mean": float(np.mean(h1_vals)),
            "H2_mean": float(np.mean(h2_vals)),
            "H_inf_mean": float(np.mean(hinf_vals)),
            "effective_branching_mean": float(np.mean(branch_vals)),
            "H0_std": float(np.std(h0_vals)),
            "H1_std": float(np.std(h1_vals)),
            "H2_std": float(np.std(h2_vals)),
            "H_inf_std": float(np.std(hinf_vals)),
        }

        print(f"\n  {FUNC_LABELS.get(func, func)}:")
        print(f"    H_0 (support):      {exp3_results[func]['H0_mean']:.2f} ± {exp3_results[func]['H0_std']:.2f}")
        print(f"    H_1 (Shannon):      {exp3_results[func]['H1_mean']:.2f} ± {exp3_results[func]['H1_std']:.2f}")
        print(f"    H_2 (collision):    {exp3_results[func]['H2_mean']:.2f} ± {exp3_results[func]['H2_std']:.2f}")
        print(f"    H_∞ (min-entropy):  {exp3_results[func]['H_inf_mean']:.2f} ± {exp3_results[func]['H_inf_std']:.2f}")
        print(f"    Effective branching: {exp3_results[func]['effective_branching_mean']:.1f}")

    sys.stdout.flush()

    # ===== Exp4: Constraint Independence (Per-Sample Correlation) =====
    print(f"\n{'='*70}")
    print(f"Exp4: Constraint Independence (Per-Sample Correlation)")
    print(f"{'='*70}")

    exp4_results = {}
    single_funcs_all = ["negation", "question", "role_binding"]

    # Per-sample KL values
    per_sample_kl = {}
    for func in single_funcs_all:
        kl_vals = []
        for pair_idx in range(n_pairs):
            base_probs = dist_storage["base"][pair_idx]
            trans_probs = dist_storage[func][pair_idx]
            kl_vals.append(compute_kl_divergence(base_probs, trans_probs))
        per_sample_kl[func] = kl_vals

    # Correlation matrix
    print(f"\n  Per-sample KL correlation matrix:")
    corr_matrix = {}
    for i, funcA in enumerate(single_funcs_all):
        corr_matrix[funcA] = {}
        for j, funcB in enumerate(single_funcs_all):
            if i <= j:
                valsA = per_sample_kl[funcA]
                valsB = per_sample_kl[funcB]
                r = np.corrcoef(valsA, valsB)[0, 1] if len(valsA) > 2 else 0.0
                corr_matrix[funcA][funcB] = float(r)
                if i < j:
                    print(f"    r({FUNC_LABELS[funcA]}, {FUNC_LABELS[funcB]}) = {r:.4f}")

    # JS divergence between constraint patterns
    print(f"\n  JS divergence between constraint patterns (mean):")
    js_results = {}
    for i, funcA in enumerate(single_funcs_all):
        js_results[funcA] = {}
        for j, funcB in enumerate(single_funcs_all):
            if i < j:
                js_vals = []
                for pair_idx in range(n_pairs):
                    probs_A = dist_storage[funcA][pair_idx]
                    probs_B = dist_storage[funcB][pair_idx]
                    js_vals.append(compute_js_divergence(probs_A, probs_B))
                mean_js = float(np.mean(js_vals))
                js_results[funcA][funcB] = mean_js
                print(f"    JS({FUNC_LABELS[funcA]}, {FUNC_LABELS[funcB]}) = {mean_js:.4f}")

    exp4_results = {
        "per_sample_kl": {k: [float(x) for x in v] for k, v in per_sample_kl.items()},
        "correlation_matrix": corr_matrix,
        "js_divergence": js_results,
    }

    sys.stdout.flush()

    # ===== Composition Deep Dive: Token-Level Analysis =====
    print(f"\n{'='*70}")
    print(f"Deep Dive: Token-Level Composition Analysis")
    print(f"{'='*70}")

    for funcA, funcB, funcAB in COMBINATIONS:
        print(f"\n  --- {FUNC_LABELS[funcA]} + {FUNC_LABELS[funcB]} ---")

        # Aggregate promoted tokens across all pairs
        promoted_A_counter = defaultdict(float)
        promoted_B_counter = defaultdict(float)
        promoted_AB_counter = defaultdict(float)

        for pair_idx in range(n_pairs):
            base_probs = dist_storage["base"][pair_idx]
            probs_A = dist_storage[funcA][pair_idx]
            probs_B = dist_storage[funcB][pair_idx]
            probs_AB = dist_storage[funcAB][pair_idx]

            promoted_A, _ = get_promoted_suppressed_tokens(base_probs, probs_A, tokenizer, top_k=10)
            promoted_B, _ = get_promoted_suppressed_tokens(base_probs, probs_B, tokenizer, top_k=10)
            promoted_AB, _ = get_promoted_suppressed_tokens(base_probs, probs_AB, tokenizer, top_k=10)

            for t in promoted_A[:5]:
                promoted_A_counter[t["text"]] += t["delta"]
            for t in promoted_B[:5]:
                promoted_B_counter[t["text"]] += t["delta"]
            for t in promoted_AB[:5]:
                promoted_AB_counter[t["text"]] += t["delta"]

        # Top promoted for each
        top_A = sorted(promoted_A_counter.items(), key=lambda x: -x[1])[:8]
        top_B = sorted(promoted_B_counter.items(), key=lambda x: -x[1])[:8]
        top_AB = sorted(promoted_AB_counter.items(), key=lambda x: -x[1])[:8]

        print(f"    Top promoted by {FUNC_LABELS[funcA]}: {', '.join([f'{t}({d:.4f})' for t, d in top_A])}")
        print(f"    Top promoted by {FUNC_LABELS[funcB]}: {', '.join([f'{t}({d:.4f})' for t, d in top_B])}")
        print(f"    Top promoted by {FUNC_LABELS[funcAB]}: {', '.join([f'{t}({d:.4f})' for t, d in top_AB])}")

        # Check: Are AB's promoted tokens = A's ∪ B's?
        set_A = set([t for t, _ in top_A])
        set_B = set([t for t, _ in top_B])
        set_AB = set([t for t, _ in top_AB])
        union_AB = set_A | set_B

        overlap_with_union = set_AB & union_AB
        novel_in_AB = set_AB - union_AB
        missing_from_AB = union_AB - set_AB

        print(f"    AB tokens in A∪B: {len(overlap_with_union)}/{len(set_AB)} "
              f"({', '.join(list(overlap_with_union)[:5])})")
        if novel_in_AB:
            print(f"    Novel in AB (not in A or B): {', '.join(list(novel_in_AB)[:5])}")
        if missing_from_AB:
            print(f"    Missing from AB (in A∪B but not AB): {', '.join(list(missing_from_AB)[:5])}")

    sys.stdout.flush()

    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"SUMMARY: Constraint Composition Analysis — {model_name}")
    print(f"{'='*70}")

    print(f"\n  Exp1: Single Constraint KL (base→transformed)")
    for func in single_funcs:
        print(f"    {FUNC_LABELS[func]:>12}: KL = {exp1_results[func]['mean_KL']:.4f} ± {exp1_results[func]['std_KL']:.4f}")

    print(f"\n  Exp2: Constraint Composition")
    for funcA, funcB, funcAB in COMBINATIONS:
        key = f"{funcA}+{funcB}"
        r = exp2_results[key]
        print(f"    {FUNC_LABELS[funcA]:>12} + {FUNC_LABELS[funcB]:>12}: "
              f"KL(AB)={r['mean_KL_AB']:.4f}, "
              f"KL(A)+KL(B)={r['mean_KL_A']+r['mean_KL_B']:.4f}, "
              f"interaction={r['mean_interaction']:+.4f} → {r['composition_type']}")

    print(f"\n  Exp3: Effective Branching Factor")
    for func in all_funcs_for_branching:
        r = exp3_results[func]
        label = FUNC_LABELS.get(func, func)
        base_branch = exp3_results["base"]["effective_branching_mean"]
        delta_branch = r["effective_branching_mean"] - base_branch
        print(f"    {label:>12}: {r['effective_branching_mean']:.1f} "
              f"(Δ={delta_branch:+.1f} vs base)")

    print(f"\n  Exp4: Constraint Correlations")
    for funcA in single_funcs_all:
        for funcB in single_funcs_all:
            if funcA < funcB and funcA in corr_matrix and funcB in corr_matrix[funcA]:
                print(f"    r({FUNC_LABELS[funcA]}, {FUNC_LABELS[funcB]}) = "
                      f"{corr_matrix[funcA][funcB]:.4f}")

    total_time = time.time() - t0_total
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  PHASE 196 {model_name.upper()} COMPLETE")
    sys.stdout.flush()

    # ===== 保存结果 =====
    output = {
        "model": model_name,
        "n_pairs": n_pairs,
        "exp1_single_constraint": exp1_results,
        "exp2_composition": exp2_results,
        "exp3_branching": exp3_results,
        "exp4_independence": exp4_results,
        "total_time_s": total_time,
    }

    # 转换numpy类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj

    output = convert_numpy(output)

    out_dir = Path(__file__).parent.parent / "glm5_temp"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"phase196_{model_name}_{time.strftime('%Y%m%d_%H%M')}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_file}")

    # 释放模型
    release_model(model)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase196_constraint_composition.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1].lower().replace("-", "")
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"  Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_experiment(model_name)
