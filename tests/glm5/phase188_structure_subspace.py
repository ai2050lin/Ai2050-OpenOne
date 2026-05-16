"""
Phase 188: Structure Subspace Decoding — What Lives in the Low-PC Directions?
============================================================================

★★★ 理论基础 ★★★

Phase 187发现表示空间有内容/结构子空间分离:
- 内容子空间(高PC, ~85-90%方差): 编码"句子关于什么"
- 结构子空间(低PC, ~10-15%方差): 编码"句子如何表达"

但"结构子空间"到底编码了什么? 这是理解语言编码数学结构的关键!

★★★ 核心假设 ★★★

低PC方向编码的是以下信息之一(或组合):
1. 语法特征(时态、数、格)
2. 逻辑关系(否定、条件、量词)
3. 关系结构(主谓宾关系、依存关系)
4. 语义角色(施事、受事、工具)
5. 语用信息(语气、态度、焦点)

★★★ 实验设计 ★★★

Exp1: 低PC探针分类 (Low-PC Probing)
  核心: 在低PC方向上能线性分类哪些语言学特征?
  方法:
    a. 提取大量句子的hidden states
    b. 做PCA, 分离高PC(>50)和低PC(≤50)投影
    c. 分别在高PC和低PC投影上训练探针, 分类:
       - 时态 (past vs present)
       - 数 (singular vs plural)
       - 肯定/否定
       - 句型 (陈述/疑问/祈使)
       - 语义角色 (施事/受事)
    d. 比较: 低PC上的分类准确率 vs 高PC上的分类准确率

Exp2: 子空间交叉验证 (Cross-Subspace Validation)
  核心: 内容和结构信息是否真的正交?
  方法:
    a. 在高PC投影上分类内容特征(主题类别)
    b. 在低PC投影上分类结构特征(语法、逻辑)
    c. 做双因子交叉验证: 内容特征在低PC上是否可分?
       结构特征在高PC上是否可分?
  预期: 内容在低PC不可分, 结构在高PC不可分 → 真正正交

Exp3: 子空间对齐的跨层动态 (Dynamic Subspace Tracking)
  核心: 内容/结构分离何时出现?如何演化?
  方法:
    a. 在每一层做PCA
    b. 计算相邻层主方向的旋转角 (subspace angle)
    c. 追踪E_pc(结构差异在主子空间中的能量)随层的变化
    d. 找出分离的"涌现层" — E_pc突然下降的层
  预期: 存在某个临界层, 在该层内容/结构分离突然增强

Exp4: 语义场探测 (Semantic Field Probing)
  核心: 极性对比在低PC方向中是否有特定结构?
  方法:
    a. 收集大量极性词对 (hot/cold, love/hate, big/small等)
    b. 计算每个极性词对的差异向量
    c. 在低PC子空间中投影这些差异向量
    d. 分析: 极性差异向量是否形成低维结构?
    e. 计算极性差异向量之间的角度 → 是否有"语义场"几何结构?
  预期: 极性差异向量在低PC子空间中形成低维流形

★★★ 数据量 ★★★
- 200+句子, 覆盖多种语法结构和语义类别
- 需要足够数据做可靠的PCA和探针训练

Usage: python tests/glm5/phase188_structure_subspace.py <model_name>
       python tests/glm5/phase188_structure_subspace.py qwen3
"""

import sys, os, time, json, gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[P188] Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[P188] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# DATA: Carefully constructed sentences with linguistic annotations
# =====================================================================

# ★ Exp1: Sentences with clear linguistic features for probing
PROBE_SENTENCES = {
    # Tense: past vs present (20 each)
    "tense_present": [
        "The cat sleeps on the mat",
        "She walks to the store",
        "He reads a book every day",
        "The dog runs in the park",
        "I eat breakfast at seven",
        "They play soccer on weekends",
        "We study English at school",
        "The bird sings in the morning",
        "She drinks coffee every day",
        "He drives to work each day",
        "The sun rises in the east",
        "Rain falls from the sky",
        "Fish swim in the river",
        "Babies cry when hungry",
        "The wind blows through trees",
        "Students learn new things",
        "Rivers flow to the sea",
        "The clock ticks steadily",
        "Plants grow toward light",
        "Stars shine at night",
    ],
    "tense_past": [
        "The cat slept on the mat",
        "She walked to the store",
        "He read a book yesterday",
        "The dog ran in the park",
        "I ate breakfast at seven",
        "They played soccer yesterday",
        "We studied English last year",
        "The bird sang in the morning",
        "She drank coffee yesterday",
        "He drove to work last week",
        "The sun rose early today",
        "Rain fell from the sky",
        "Fish swam in the river",
        "Babies cried all night",
        "The wind blew through trees",
        "Students learned new things",
        "Rivers flowed to the sea",
        "The clock ticked steadily",
        "Plants grew toward light",
        "Stars shone bright last night",
    ],
    # Number: singular vs plural (20 each)
    "number_singular": [
        "The cat sleeps quietly",
        "A dog barks loudly",
        "The child plays outside",
        "A bird flies high",
        "The tree grows tall",
        "A flower blooms bright",
        "The house stands firm",
        "A car drives fast",
        "The star shines bright",
        "A river flows deep",
        "The book lies open",
        "A door closes slowly",
        "The wind blows gently",
        "A clock ticks steadily",
        "The baby cries softly",
        "A fish swims upstream",
        "The student reads carefully",
        "A teacher speaks clearly",
        "The fire burns hot",
        "A stone sits heavy",
    ],
    "number_plural": [
        "The cats sleep quietly",
        "Dogs bark loudly",
        "Children play outside",
        "Birds fly high",
        "Trees grow tall",
        "Flowers bloom bright",
        "Houses stand firm",
        "Cars drive fast",
        "Stars shine bright",
        "Rivers flow deep",
        "Books lie open",
        "Doors close slowly",
        "Winds blow gently",
        "Clocks tick steadily",
        "Babies cry softly",
        "Fish swim upstream",
        "Students read carefully",
        "Teachers speak clearly",
        "Fires burn hot",
        "Stones sit heavy",
    ],
    # Polarity: affirmative vs negative (20 each)
    "polarity_affirm": [
        "The cat is sleeping",
        "She likes the movie",
        "He can swim well",
        "They will come tomorrow",
        "The door is open",
        "She has finished work",
        "He did eat breakfast",
        "The water is warm",
        "I do understand you",
        "She was happy then",
        "He could run fast",
        "They would agree soon",
        "The answer is correct",
        "She will pass the test",
        "He must leave now",
        "The sky is blue",
        "I have seen this before",
        "She does know the truth",
        "He is coming today",
        "They are ready now",
    ],
    "polarity_negate": [
        "The cat is not sleeping",
        "She does not like the movie",
        "He cannot swim well",
        "They will not come tomorrow",
        "The door is not open",
        "She has not finished work",
        "He did not eat breakfast",
        "The water is not warm",
        "I do not understand you",
        "She was not happy then",
        "He could not run fast",
        "They would not agree soon",
        "The answer is not correct",
        "She will not pass the test",
        "He must not leave now",
        "The sky is not blue",
        "I have not seen this before",
        "She does not know the truth",
        "He is not coming today",
        "They are not ready now",
    ],
    # Sentence type: declarative vs question (20 each)
    "sentype_declare": [
        "The cat sleeps on the mat",
        "She walks to the store",
        "He reads a book every day",
        "The dog runs in the park",
        "I eat breakfast at seven",
        "They play soccer on weekends",
        "We study English at school",
        "The bird sings in the morning",
        "She drinks coffee every day",
        "He drives to work each day",
        "The sun rises in the east",
        "Rain falls from the sky",
        "Fish swim in the river",
        "Babies cry when hungry",
        "The wind blows through trees",
        "Students learn new things",
        "Rivers flow to the sea",
        "The clock ticks steadily",
        "Plants grow toward light",
        "Stars shine at night",
    ],
    "sentype_question": [
        "Does the cat sleep on the mat",
        "Does she walk to the store",
        "Does he read a book every day",
        "Does the dog run in the park",
        "Do I eat breakfast at seven",
        "Do they play soccer on weekends",
        "Do we study English at school",
        "Does the bird sing in the morning",
        "Does she drink coffee every day",
        "Does he drive to work each day",
        "Does the sun rise in the east",
        "Does rain fall from the sky",
        "Do fish swim in the river",
        "Do babies cry when hungry",
        "Does the wind blow through trees",
        "Do students learn new things",
        "Do rivers flow to the sea",
        "Does the clock tick steadily",
        "Do plants grow toward light",
        "Do stars shine at night",
    ],
    # Semantic category: animals vs objects vs nature vs people (20 each)
    "category_animal": [
        "The cat sleeps on the mat",
        "A dog barks at the door",
        "The bird sings a song",
        "Fish swim in the pond",
        "The horse runs fast",
        "A bee flies to the flower",
        "The lion roars loudly",
        "Eagles soar above clouds",
        "The rabbit hops away",
        "Whales dive deep below",
        "The snake slides quietly",
        "A frog jumps into water",
        "The wolf howls at night",
        "Butterflies flutter gently",
        "The bear catches salmon",
        "A deer runs through forest",
        "The owl hoots in darkness",
        "Penguins waddle on ice",
        "The fox sneaks around",
        "Dolphins play in waves",
    ],
    "category_object": [
        "The chair stands in the corner",
        "A book lies on the table",
        "The door opens slowly",
        "The clock ticks on the wall",
        "A pen writes on paper",
        "The lamp shines brightly",
        "The cup holds hot tea",
        "A key opens the lock",
        "The mirror reflects light",
        "The wheel turns smoothly",
        "A bridge crosses the river",
        "The bell rings loudly",
        "The knife cuts the bread",
        "A box holds many toys",
        "The rope ties the boat",
        "The stove heats the room",
        "A window lets in light",
        "The pipe carries water",
        "The shield blocks the blow",
        "A candle burns slowly",
    ],
}

# ★ Exp4: Polar contrast pairs for semantic field probing
POLAR_CONTRASTS = [
    # (positive_pole, negative_pole, contrast_name, dimension)
    ("hot", "cold", "hot_cold", "temperature"),
    ("big", "small", "big_small", "size"),
    ("fast", "slow", "fast_slow", "speed"),
    ("happy", "sad", "happy_sad", "emotion"),
    ("love", "hate", "love_hate", "sentiment"),
    ("light", "dark", "light_dark", "brightness"),
    ("strong", "weak", "strong_weak", "strength"),
    ("rich", "poor", "rich_poor", "wealth"),
    ("good", "bad", "good_bad", "quality"),
    ("high", "low", "high_low", "elevation"),
    ("new", "old", "new_old", "age"),
    ("open", "closed", "open_closed", "state"),
    ("soft", "hard", "soft_hard", "texture"),
    ("wet", "dry", "wet_dry", "moisture"),
    ("alive", "dead", "alive_dead", "vitality"),
    # More abstract contrasts
    ("truth", "lie", "truth_lie", "honesty"),
    ("peace", "war", "peace_war", "conflict"),
    ("create", "destroy", "create_destroy", "action"),
    ("remember", "forget", "remember_forget", "memory"),
    ("accept", "reject", "accept_reject", "decision"),
]


def get_hidden_states(model, tokenizer, device, sentence, n_layers):
    """Get hidden states at all layers for last token position"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    # Use last token position
    last_pos = attention_mask.sum().item() - 1
    result = {}
    for l in range(n_layers):
        h = hidden_states[l][0, last_pos, :].float().cpu().numpy()
        result[l] = h

    return result


def simple_logistic_probe(X_train, y_train, X_test, y_test):
    """Simple logistic regression probe using numpy"""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    return train_acc, test_acc


def run_exp1_probing(model, tokenizer, device, n_layers, d_model):
    """Exp1: Low-PC Probing — what linguistic features live in low-PC directions?"""
    print("\n" + "=" * 70)
    print("Exp1: LOW-PC PROBING")
    print("  ★★★ What linguistic features live in low-PC directions? ★★★")
    print("=" * 70)

    # Sample layers for probing
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f"  Sample layers: {sample_layers}")

    # Collect hidden states for all sentences
    feature_groups = {
        "tense": ("tense_present", "tense_past"),
        "number": ("number_singular", "number_plural"),
        "polarity": ("polarity_affirm", "polarity_negate"),
        "sen_type": ("sentype_declare", "sentype_question"),
    }

    # Also add category pairs
    category_groups = {
        "category": ("category_animal", "category_object"),
    }

    all_sentences = []
    all_labels = {}
    for feat_name, (pos_key, neg_key) in feature_groups.items():
        pos_sents = PROBE_SENTENCES[pos_key]
        neg_sents = PROBE_SENTENCES[neg_key]
        for s in pos_sents:
            all_sentences.append(s)
            all_labels[s] = (feat_name, 1)
        for s in neg_sents:
            all_sentences.append(s)
            all_labels[s] = (feat_name, 0)

    for feat_name, (pos_key, neg_key) in category_groups.items():
        pos_sents = PROBE_SENTENCES[pos_key]
        neg_sents = PROBE_SENTENCES[neg_key]
        for s in pos_sents:
            if s not in all_labels:
                all_sentences.append(s)
                all_labels[s] = (feat_name, 1)
        for s in neg_sents:
            if s not in all_labels:
                all_sentences.append(s)
                all_labels[s] = (feat_name, 0)

    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    all_sentences = unique_sentences
    print(f"  Total unique sentences: {len(all_sentences)}")

    # Get hidden states
    print("  Collecting hidden states...")
    hs_dict = {}  # sentence -> {layer: hidden_state}
    for i, s in enumerate(all_sentences):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Sentence {i+1}/{len(all_sentences)}")
        hs = get_hidden_states(model, tokenizer, device, s, n_layers)
        hs_dict[s] = hs

    results = {}

    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")

        # Build data matrix
        H = np.array([hs_dict[s][li] for s in all_sentences])  # (N, d_model)
        N = H.shape[0]

        # Center and PCA
        H_centered = H - H.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        # Vt: (min(N,d), d_model) — rows are principal directions

        n_high = min(50, Vt.shape[0])
        n_low_start = min(50, Vt.shape[0])

        # Project onto high-PC and low-PC subspaces
        # High-PC: first n_high directions
        V_high = Vt[:n_high, :]  # (50, d_model)
        V_low = Vt[n_low_start:, :]  # (remaining, d_model)

        if V_low.shape[0] == 0:
            print(f"    Not enough PCs for low subspace, skipping")
            continue

        H_high = H_centered @ V_high.T  # (N, 50) — projections onto high PCs
        H_low = H_centered @ V_low.T    # (N, remaining) — projections onto low PCs

        # For efficiency, use top 50 low-PC directions
        n_low_use = min(50, V_low.shape[0])
        H_low = H_low[:, :n_low_use]

        # Also compute variance explained
        total_var = np.sum(S ** 2)
        high_var = np.sum(S[:n_high] ** 2) / total_var
        low_var = np.sum(S[n_low_start:] ** 2) / total_var

        print(f"    High-PC variance: {high_var:.4f}, Low-PC variance: {low_var:.4f}")

        layer_result = {
            "high_pc_var": float(high_var),
            "low_pc_var": float(low_var),
            "n_high": n_high,
            "n_low": n_low_use,
        }

        # Probe each feature in both subspaces
        for feat_name, (pos_key, neg_key) in {**feature_groups, **category_groups}.items():
            pos_sents = PROBE_SENTENCES[pos_key]
            neg_sents = PROBE_SENTENCES[neg_key]

            # Build indices
            pos_set = set(pos_sents)
            neg_set = set(neg_sents)

            indices = []
            labels = []
            for i, s in enumerate(all_sentences):
                if s in pos_set:
                    indices.append(i)
                    labels.append(1)
                elif s in neg_set:
                    indices.append(i)
                    labels.append(0)

            if len(indices) < 10:
                continue

            indices = np.array(indices)
            labels = np.array(labels)

            # Shuffle and split
            perm = np.random.RandomState(42).permutation(len(indices))
            n_train = int(0.8 * len(indices))
            train_idx = indices[perm[:n_train]]
            test_idx = indices[perm[n_train:]]
            y_train = labels[perm[:n_train]]
            y_test = labels[perm[n_train:]]

            # Probe on high-PC
            try:
                train_acc_h, test_acc_h = simple_logistic_probe(
                    H_high[train_idx], y_train, H_high[test_idx], y_test)
            except Exception:
                train_acc_h, test_acc_h = 0.5, 0.5

            # Probe on low-PC
            try:
                train_acc_l, test_acc_l = simple_logistic_probe(
                    H_low[train_idx], y_train, H_low[test_idx], y_test)
            except Exception:
                train_acc_l, test_acc_l = 0.5, 0.5

            # Also probe on FULL space
            try:
                train_acc_f, test_acc_f = simple_logistic_probe(
                    H[train_idx], y_train, H[test_idx], y_test)
            except Exception:
                train_acc_f, test_acc_f = 0.5, 0.5

            print(f"    {feat_name:15s}: high-PC={test_acc_h:.3f}, low-PC={test_acc_l:.3f}, "
                  f"full={test_acc_f:.3f}")

            layer_result[f"{feat_name}_high"] = float(test_acc_h)
            layer_result[f"{feat_name}_low"] = float(test_acc_l)
            layer_result[f"{feat_name}_full"] = float(test_acc_f)

        results[str(li)] = layer_result

    return results


def run_exp2_cross_subspace(model, tokenizer, device, n_layers, d_model):
    """Exp2: Cross-Subspace Validation — are content and structure truly orthogonal?"""
    print("\n" + "=" * 70)
    print("Exp2: CROSS-SUBSPACE VALIDATION")
    print("  ★★★ Are content and structure truly orthogonal? ★★★")
    print("=" * 70)

    # Use a subset of layers
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set([max(0, l) for l in sample_layers]))
    print(f"  Sample layers: {sample_layers}")

    # Content feature: category (animal vs object)
    # Structure features: tense, number, polarity, sentence type
    content_pos = PROBE_SENTENCES["category_animal"]
    content_neg = PROBE_SENTENCES["category_object"]

    structure_groups = {
        "tense": ("tense_present", "tense_past"),
        "number": ("number_singular", "number_plural"),
        "polarity": ("polarity_affirm", "polarity_negate"),
    }

    # Build sentences with BOTH content and structure labels
    # We need sentences that have BOTH labels
    # Strategy: use structure sentences that are NOT content-labeled, and vice versa

    # Simpler approach: use category sentences that also have tense/number info
    # But our sentences don't overlap. Let's use the structure sentences for structure probing
    # and category sentences for content probing, independently.

    # Actually, let's do a proper cross-validation:
    # 1. Compute PCA on ALL sentences
    # 2. Project category difference vectors onto high/low PCs
    # 3. Project structure difference vectors onto high/low PCs
    # 4. Check: does category info live in high PCs? Structure in low PCs?

    all_sentences = []
    for key in PROBE_SENTENCES:
        all_sentences.extend(PROBE_SENTENCES[key])
    all_sentences = list(set(all_sentences))
    print(f"  Total unique sentences: {len(all_sentences)}")

    # Get hidden states
    print("  Collecting hidden states...")
    hs_dict = {}
    for i, s in enumerate(all_sentences):
        if (i + 1) % 20 == 0:
            print(f"    Sentence {i+1}/{len(all_sentences)}")
        hs_dict[s] = get_hidden_states(model, tokenizer, device, s, n_layers)

    results = {}

    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")

        H = np.array([hs_dict[s][li] for s in all_sentences])
        H_centered = H - H.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)

        n_high = min(50, Vt.shape[0])
        n_low_start = min(50, Vt.shape[0])
        V_high = Vt[:n_high, :]
        V_low = Vt[n_low_start:, :]

        if V_low.shape[0] == 0:
            continue

        # Compute category difference vectors
        cat_pos_hs = np.array([hs_dict[s][li] for s in content_pos if s in hs_dict])
        cat_neg_hs = np.array([hs_dict[s][li] for s in content_neg if s in hs_dict])

        # Category centroid difference
        delta_cat = cat_pos_hs.mean(axis=0) - cat_neg_hs.mean(axis=0)
        delta_cat_centered = delta_cat - H.mean(axis=0)

        # Project onto high and low PCs
        proj_high = np.dot(V_high, delta_cat_centered)
        proj_low = np.dot(V_low, delta_cat_centered)
        energy_high = np.sum(proj_high ** 2)
        energy_low = np.sum(proj_low ** 2)
        energy_total = energy_high + energy_low
        frac_high = energy_high / max(energy_total, 1e-10)

        print(f"    Category Δ: E_high={energy_high:.4f}, E_low={energy_low:.4f}, "
              f"frac_high={frac_high:.4f}")

        layer_result = {"category_frac_high": float(frac_high)}

        # Compute structure difference vectors
        for struct_name, (pos_key, neg_key) in structure_groups.items():
            struct_pos = PROBE_SENTENCES[pos_key]
            struct_neg = PROBE_SENTENCES[neg_key]
            struct_pos_hs = np.array([hs_dict[s][li] for s in struct_pos if s in hs_dict])
            struct_neg_hs = np.array([hs_dict[s][li] for s in struct_neg if s in hs_dict])

            delta_struct = struct_pos_hs.mean(axis=0) - struct_neg_hs.mean(axis=0)
            delta_struct_centered = delta_struct - H.mean(axis=0)

            proj_high_s = np.dot(V_high, delta_struct_centered)
            proj_low_s = np.dot(V_low, delta_struct_centered)
            energy_high_s = np.sum(proj_high_s ** 2)
            energy_low_s = np.sum(proj_low_s ** 2)
            energy_total_s = energy_high_s + energy_low_s
            frac_high_s = energy_high_s / max(energy_total_s, 1e-10)

            print(f"    {struct_name:15s} Δ: E_high={energy_high_s:.4f}, E_low={energy_low_s:.4f}, "
                  f"frac_high={frac_high_s:.4f}")

            layer_result[f"{struct_name}_frac_high"] = float(frac_high_s)

        results[str(li)] = layer_result

    return results


def run_exp3_subspace_dynamics(model, tokenizer, device, n_layers, d_model):
    """Exp3: Dynamic Subspace Tracking — when does content/structure separation emerge?"""
    print("\n" + "=" * 70)
    print("Exp3: DYNAMIC SUBSPACE TRACKING")
    print("  ★★★ When does content/structure separation emerge? ★★★")
    print("=" * 70)

    # Use category and syntactic sentences
    cat_sents = PROBE_SENTENCES["category_animal"] + PROBE_SENTENCES["category_object"]
    syn_sents = PROBE_SENTENCES["tense_present"] + PROBE_SENTENCES["tense_past"]
    num_sents = PROBE_SENTENCES["number_singular"] + PROBE_SENTENCES["number_plural"]

    all_sents = list(set(cat_sents + syn_sents + num_sents))
    print(f"  Total sentences: {len(all_sents)}")

    # Get hidden states at ALL layers
    print("  Collecting hidden states at all layers...")
    hs_dict = {}
    for i, s in enumerate(all_sents):
        if (i + 1) % 20 == 0:
            print(f"    Sentence {i+1}/{len(all_sents)}")
        hs_dict[s] = get_hidden_states(model, tokenizer, device, s, n_layers)

    results = {}

    for li in range(n_layers):
        H = np.array([hs_dict[s][li] for s in all_sents])
        H_centered = H - H.mean(axis=0, keepdims=True)

        # PCA
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)

        # Compute E_pc for category and syntactic differences
        # Category
        cat_animal_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["category_animal"] if s in hs_dict])
        cat_object_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["category_object"] if s in hs_dict])
        delta_cat = cat_animal_hs.mean(axis=0) - cat_object_hs.mean(axis=0)
        delta_cat_c = delta_cat - H.mean(axis=0)

        n_high = min(50, Vt.shape[0])
        V_high = Vt[:n_high, :]
        proj_h = V_high @ delta_cat_c
        E_pc_cat = np.sum(proj_h ** 2) / max(np.sum(delta_cat_c ** 2), 1e-10)

        # Syntactic (tense)
        tense_pres_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["tense_present"] if s in hs_dict])
        tense_past_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["tense_past"] if s in hs_dict])
        delta_syn = tense_pres_hs.mean(axis=0) - tense_past_hs.mean(axis=0)
        delta_syn_c = delta_syn - H.mean(axis=0)
        proj_h_s = V_high @ delta_syn_c
        E_pc_syn = np.sum(proj_h_s ** 2) / max(np.sum(delta_syn_c ** 2), 1e-10)

        # Number
        num_sg_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["number_singular"] if s in hs_dict])
        num_pl_hs = np.array([hs_dict[s][li] for s in PROBE_SENTENCES["number_plural"] if s in hs_dict])
        delta_num = num_sg_hs.mean(axis=0) - num_pl_hs.mean(axis=0)
        delta_num_c = delta_num - H.mean(axis=0)
        proj_h_n = V_high @ delta_num_c
        E_pc_num = np.sum(proj_h_n ** 2) / max(np.sum(delta_num_c ** 2), 1e-10)

        # Separation index = E_pc(category) - E_pc(syntactic)
        # Higher = more separation (category in high PCs, syntactic in low PCs)
        separation = E_pc_cat - E_pc_syn

        # Subspace rotation: compare PC directions with previous layer
        if li > 0:
            H_prev = np.array([hs_dict[s][li - 1] for s in all_sents])
            H_prev_c = H_prev - H_prev.mean(axis=0, keepdims=True)
            _, _, Vt_prev = np.linalg.svd(H_prev_c, full_matrices=False)
            # Grassmann distance between top-50 subspaces
            n_sub = min(50, Vt.shape[0], Vt_prev.shape[0])
            Q1 = Vt[:n_sub, :].T  # (d, 50)
            Q2 = Vt_prev[:n_sub, :].T  # (d, 50)
            M = Q1.T @ Q2  # (50, 50)
            _, sigma, _ = np.linalg.svd(M)
            # Principal angles
            cos_angles = np.clip(sigma, 0, 1)
            mean_angle = np.arccos(np.mean(cos_angles))
        else:
            mean_angle = 0.0

        results[str(li)] = {
            "E_pc_category": float(E_pc_cat),
            "E_pc_syntactic": float(E_pc_syn),
            "E_pc_number": float(E_pc_num),
            "separation_index": float(separation),
            "subspace_rotation_deg": float(np.degrees(mean_angle)),
            "cumvar_top10": float(np.sum(S[:10] ** 2) / np.sum(S ** 2)),
            "cumvar_top50": float(np.sum(S[:min(50, len(S))] ** 2) / np.sum(S ** 2)),
        }

    # Print summary
    print("\n  Subspace dynamics summary:")
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        li = max(0, min(li, n_layers - 1))
        r = results[str(li)]
        print(f"    L{li:2d}: E_pc(cat)={r['E_pc_category']:.3f}, "
              f"E_pc(syn)={r['E_pc_syntactic']:.3f}, "
              f"sep={r['separation_index']:.3f}, "
              f"rot={r['subspace_rotation_deg']:.2f}°")

    return results


def run_exp4_semantic_field(model, tokenizer, device, n_layers, d_model):
    """Exp4: Semantic Field Probing — do polar contrasts have geometric structure?"""
    print("\n" + "=" * 70)
    print("Exp4: SEMANTIC FIELD PROBING")
    print("  ★★★ Do polar contrasts have geometric structure in low-PC space? ★★★")
    print("=" * 70)

    # Build sentences from polar contrasts
    # Use template: "The [concept] is [adjective]"
    polar_sentences = {}
    for pos_word, neg_word, name, dim in POLAR_CONTRASTS:
        # Create simple sentences with these words
        pos_sent = f"Things that are {pos_word}"
        neg_sent = f"Things that are {neg_word}"
        polar_sentences[name] = (pos_sent, neg_sent, dim)

    # Also use the PROBE_SENTENCES polarity pairs directly
    polarity_affirm = PROBE_SENTENCES["polarity_affirm"]
    polarity_negate = PROBE_SENTENCES["polarity_negate"]

    # Get hidden states for polar sentences
    all_sents = []
    for name, (pos_s, neg_s, dim) in polar_sentences.items():
        all_sents.extend([pos_s, neg_s])

    # Remove duplicates
    all_sents = list(set(all_sents))
    print(f"  Total polar sentences: {len(all_sents)}")

    # Also need reference sentences for PCA
    ref_sents = []
    for key in ["category_animal", "category_object", "tense_present", "tense_past",
                 "number_singular", "number_plural"]:
        ref_sents.extend(PROBE_SENTENCES[key])
    ref_sents = list(set(ref_sents))

    all_sents_with_ref = list(set(all_sents + ref_sents))
    print(f"  Total sentences (with reference): {len(all_sents_with_ref)}")

    # Get hidden states
    print("  Collecting hidden states...")
    hs_dict = {}
    for i, s in enumerate(all_sents_with_ref):
        if (i + 1) % 20 == 0:
            print(f"    Sentence {i+1}/{len(all_sents_with_ref)}")
        hs_dict[s] = get_hidden_states(model, tokenizer, device, s, n_layers)

    results = {}

    # Analyze at last layer
    last_li = n_layers - 1
    mid_li = n_layers // 2

    for li in [last_li]:
        print(f"\n  --- Layer {li} ---")

        # PCA on reference sentences
        H_ref = np.array([hs_dict[s][li] for s in ref_sents if s in hs_dict])
        H_ref_c = H_ref - H_ref.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_ref_c, full_matrices=False)

        n_high = min(50, Vt.shape[0])
        V_high = Vt[:n_high, :]
        V_low = Vt[n_high:, :]

        # Compute difference vectors for each polar contrast
        delta_vectors_high = {}
        delta_vectors_low = {}
        delta_vectors_full = {}

        for name, (pos_s, neg_s, dim) in polar_sentences.items():
            if pos_s not in hs_dict or neg_s not in hs_dict:
                continue
            h_pos = hs_dict[pos_s][li]
            h_neg = hs_dict[neg_s][li]
            delta = h_pos - h_neg

            # Project onto high and low PCs
            delta_c = delta - H_ref.mean(axis=0)
            proj_high = V_high @ delta_c
            proj_low = V_low @ delta_c if V_low.shape[0] > 0 else np.array([])

            E_high = np.sum(proj_high ** 2) / max(np.sum(delta_c ** 2), 1e-10)
            E_low = 1.0 - E_high

            delta_vectors_high[name] = proj_high
            delta_vectors_low[name] = proj_low if len(proj_low) > 0 else np.array([0])
            delta_vectors_full[name] = delta_c

            print(f"    {name:20s} (dim={dim:12s}): E_high={E_high:.3f}, E_low={E_low:.3f}")

        # Compute pairwise angles between polar contrast vectors
        print("\n  Pairwise angles between polar contrasts (in low-PC space):")
        contrast_names = list(delta_vectors_low.keys())
        n_contrasts = len(contrast_names)

        if n_contrasts >= 2:
            # Build matrix of low-PC difference vectors
            min_dim = min(len(delta_vectors_low[c]) for c in contrast_names)
            D_low = np.array([delta_vectors_low[c][:min_dim] for c in contrast_names])

            # Normalize
            norms = np.linalg.norm(D_low, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            D_low_norm = D_low / norms

            # Pairwise cosine similarities
            cos_sim = D_low_norm @ D_low_norm.T

            # Also compute in high-PC space
            D_high = np.array([delta_vectors_high[c] for c in contrast_names])
            norms_h = np.linalg.norm(D_high, axis=1, keepdims=True)
            norms_h = np.maximum(norms_h, 1e-10)
            D_high_norm = D_high / norms_h
            cos_sim_high = D_high_norm @ D_high_norm.T

            # Print interesting pairs
            for i in range(n_contrasts):
                for j in range(i + 1, n_contrasts):
                    c_low = cos_sim[i, j]
                    c_high = cos_sim_high[i, j]
                    if abs(c_low) > 0.3:  # Only print significant correlations
                        dim_i = dict(POLAR_CONTRASTS)[contrast_names[i] if contrast_names[i] in
                                     dict((n, d) for _, _, n, d in POLAR_CONTRASTS) else ""][""] \
                            if False else ""
                        print(f"      {contrast_names[i]:20s} vs {contrast_names[j]:20s}: "
                              f"cos_low={c_low:.3f}, cos_high={c_high:.3f}")

            # Check if same-dimension contrasts are more aligned
            dim_map = {}
            for pos_w, neg_w, name, dim in POLAR_CONTRASTS:
                if name in delta_vectors_low:
                    dim_map[name] = dim

            print("\n  Within-dimension vs cross-dimension alignment:")
            within_cos = []
            cross_cos = []
            for i in range(n_contrasts):
                for j in range(i + 1, n_contrasts):
                    d_i = dim_map.get(contrast_names[i], "unknown")
                    d_j = dim_map.get(contrast_names[j], "unknown")
                    if d_i == d_j and d_i != "unknown":
                        within_cos.append(cos_sim[i, j])
                    else:
                        cross_cos.append(cos_sim[i, j])

            if within_cos and cross_cos:
                print(f"    Within-dimension mean cos: {np.mean(within_cos):.4f} (n={len(within_cos)})")
                print(f"    Cross-dimension mean cos:  {np.mean(cross_cos):.4f} (n={len(cross_cos)})")

            # Also check: do polar contrasts form anti-aligned pairs in high-PC space?
            # (i.e., opposite contrasts should be anti-correlated)
            print("\n  Anti-alignment check (high-PC space):")
            for i in range(n_contrasts):
                for j in range(i + 1, n_contrasts):
                    if cos_sim_high[i, j] < -0.3:
                        print(f"    ANTI-ALIGNED: {contrast_names[i]:20s} vs {contrast_names[j]:20s}: "
                              f"cos_high={cos_sim_high[i, j]:.3f}")

        layer_result = {
            "n_contrasts": n_contrasts,
            "mean_within_cos_low": float(np.mean(within_cos)) if within_cos else 0,
            "mean_cross_cos_low": float(np.mean(cross_cos)) if cross_cos else 0,
        }

        # Per-contrast E_low
        for name, (pos_s, neg_s, dim) in polar_sentences.items():
            if name in delta_vectors_full:
                delta_c = delta_vectors_full[name]
                proj_h = V_high @ delta_c
                E_h = np.sum(proj_h ** 2) / max(np.sum(delta_c ** 2), 1e-10)
                layer_result[f"{name}_E_low"] = float(1.0 - E_h)

        results[str(li)] = layer_result

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase188_structure_subspace.py <model_name>")
        print("       model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {type(model).__name__}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")

    # Run all experiments
    print("\n" + "=" * 70)
    print("Running Exp1: Low-PC Probing...")
    print("  ★★★ What linguistic features live in low-PC directions? ★★★")
    exp1_results = run_exp1_probing(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp2: Cross-Subspace Validation...")
    print("  ★★★ Are content and structure truly orthogonal? ★★★")
    exp2_results = run_exp2_cross_subspace(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp3: Dynamic Subspace Tracking...")
    print("  ★★★ When does content/structure separation emerge? ★★★")
    exp3_results = run_exp3_subspace_dynamics(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp4: Semantic Field Probing...")
    print("  ★★★ Do polar contrasts have geometric structure? ★★★")
    exp4_results = run_exp4_semantic_field(model, tokenizer, device, n_layers, d_model)

    # Save results
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "exp1_low_pc_probing": exp1_results,
        "exp2_cross_subspace": exp2_results,
        "exp3_subspace_dynamics": exp3_results,
        "exp4_semantic_field": exp4_results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase188_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # ===== SUMMARY =====
    print("\n" + "#" * 70)
    print(f"PHASE 188 SUMMARY")
    print("#" * 70)

    print("\n★★★ Exp1: Low-PC Probing ★★★")
    # Find best layer for each feature
    for feat in ["tense", "number", "polarity", "sen_type", "category"]:
        best_high = 0
        best_low = 0
        best_layer_h = ""
        best_layer_l = ""
        for li, r in exp1_results.items():
            k_h = f"{feat}_high"
            k_l = f"{feat}_low"
            if k_h in r and r[k_h] > best_high:
                best_high = r[k_h]
                best_layer_h = li
            if k_l in r and r[k_l] > best_low:
                best_low = r[k_l]
                best_layer_l = li
        if best_high > 0 or best_low > 0:
            print(f"  {feat:15s}: best high-PC={best_high:.3f} (L{best_layer_h}), "
                  f"best low-PC={best_low:.3f} (L{best_layer_l})")

    print("\n★★★ Exp2: Cross-Subspace Validation ★★★")
    for li, r in exp2_results.items():
        cat_frac = r.get("category_frac_high", 0)
        print(f"  L{li}: category_frac_high={cat_frac:.3f}", end="")
        for feat in ["tense", "number", "polarity"]:
            frac = r.get(f"{feat}_frac_high", 0)
            print(f", {feat}_frac_high={frac:.3f}", end="")
        print()

    print("\n★★★ Exp3: Subspace Dynamics ★★★")
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        li = max(0, min(li, n_layers - 1))
        if str(li) in exp3_results:
            r = exp3_results[str(li)]
            print(f"  L{li:2d}: E_pc(cat)={r['E_pc_category']:.3f}, "
                  f"E_pc(syn)={r['E_pc_syntactic']:.3f}, sep={r['separation_index']:.3f}")

    print("\n★★★ Exp4: Semantic Field ★★★")
    for li, r in exp4_results.items():
        print(f"  L{li}: within_dim_cos={r.get('mean_within_cos_low', 0):.4f}, "
              f"cross_dim_cos={r.get('mean_cross_cos_low', 0):.4f}")

    # Release model
    release_model(model)
    force_cleanup()

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"Phase 188 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
