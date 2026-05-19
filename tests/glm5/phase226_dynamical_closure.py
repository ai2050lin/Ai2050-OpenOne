"""
Phase 226: 动力学闭包测试 — 决定性实验
========================================

核心问题:
  Phase 225 发现 top5(J_l) ≡ top5(PCA(Δh_l)) ⊥ W_U
  但这到底是"稳定观测投影"还是"真实动力学变量"？

  用户分析的核心洞察:
    1. "5维核心"可能只是谱间隙(spectral gap)现象, 不是本体结构
    2. 真正需要的不是"继续测子空间角度", 而是"找到动力学闭包"
    3. 闭包 = 存在 z_{l+1} = F(z_l), F低维、跨句子稳定、跨模型稳定

本Phase四个实验:

  Exp1: 谱间隙分析 (Spectral Gap Analysis)
    - PCA解释方差比是否有gap?
    - 还是平滑衰减? (→ 维度=5只是谱间隙假象)
    - 不同k值的子空间稳定性对比

  Exp2: 线性动力学闭包 (Linear Dynamical Closure) ★★★★★
    - z_l = P_k (h_l - mean_l) ∈ R^k
    - 拟合 z_{l+1} = A_l z_l + b_l (逐层线性映射)
    - R² vs k 曲线: R²在k=5饱和→5D闭包; 持续增长→更高维
    - 随机投影对照: R²_random vs R²_PCA

  Exp3: 跨约束稳定性 (Cross-Constraint Stability) ★★★★
    - 在SVA约束对上训练F
    - 在tense/voice/negation约束对上测试
    - 如果F跨约束通用 → 强闭包

  Exp4: Markov性测试 (Markov Order Test) ★★★
    - F_1(z_l) vs F_2(z_l, z_{l-1})
    - ΔR² < 5% → 一阶Markov (层动力学是真实动力学)
    - ΔR² >> 5% → 需要记忆 (层≠时间, 至少不是一阶)

跨模型: Qwen3 → GLM4 → DS7B
BF16 + device_map="auto" + sdpa(flash) + 定期GC
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

# ============================================================
# 模型加载 (BF16 + device_map="auto" + SDPA/Flash)
# ============================================================

def load_model_bf16_sdpa(model_name: str):
    """BF16 + device_map="auto" + SDPA(flash) 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bf16 + auto + sdpa)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试SDPA (flash attention), 失败则回退eager
    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            model.eval()
            print(f"[load] {model_name} loaded with attn_impl={attn_impl}")
            break
        except Exception as e:
            print(f"[load] attn_impl={attn_impl} failed: {e}")
            if attn_impl == "eager":
                raise

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


# ============================================================
# 句子生成
# ============================================================

def generate_sentences(n_per_type=40):
    """生成约束对 + 一般句子"""
    sentences = {}

    # === SVA ===
    singular = ["cat", "dog", "bird", "girl", "boy", "tree", "car", "child",
                "man", "woman", "fish", "horse", "student", "teacher",
                "flower", "river", "star", "moon", "sun", "cloud",
                "book", "table", "chair", "window", "door", "wall",
                "king", "queen", "prince", "princess", "knight",
                "doctor", "nurse", "soldier", "farmer", "artist",
                "river", "mountain", "forest", "desert", "ocean"]
    verbs_s = ["chases", "runs", "sings", "reads", "walks", "falls", "moves",
               "plays", "works", "dances", "swims", "sleeps", "barks", "flies",
               "grows", "blooms", "flows", "shines", "rises", "blows",
               "sits", "stands", "breaks", "opens", "closes", "cracks",
               "rules", "commands", "fights", "smiles", "rides",
               "heals", "helps", "marches", "plants", "paints",
               "drifts", "towers", "whispers", "burns", "crashes"]

    sva_pairs = []
    for i in range(min(n_per_type, len(singular))):
        subj = singular[i]
        verb = verbs_s[i % len(verbs_s)]
        sva_pairs.append({
            "A": f"The {subj} {verb}",
            "B": f"The {subj}s {verb.rstrip('s')}",
            "type": "number_sva"
        })
    sentences["number_sva"] = sva_pairs

    # === Tense ===
    tense_pairs = [
        ("The cat sleeps", "The cat slept"),
        ("The dog runs", "The dog ran"),
        ("The bird sings", "The bird sang"),
        ("The girl reads", "The girl read"),
        ("The boy walks", "The boy walked"),
        ("The tree grows", "The tree grew"),
        ("The car moves", "The car moved"),
        ("The child plays", "The child played"),
        ("The man works", "The man worked"),
        ("The woman dances", "The woman danced"),
        ("The fish swims", "The fish swam"),
        ("The student studies", "The student studied"),
        ("The teacher speaks", "The teacher spoke"),
        ("The river flows", "The river flowed"),
        ("The wind blows", "The wind blew"),
        ("The sun shines", "The sun shone"),
        ("The rain falls", "The rain fell"),
        ("The fire burns", "The fire burned"),
        ("The snow melts", "The snow melted"),
        ("The bell rings", "The bell rang"),
        ("The king rules", "The king ruled"),
        ("The queen smiles", "The queen smiled"),
        ("The doctor heals", "The doctor healed"),
        ("The soldier marches", "The soldier marched"),
        ("The farmer plants", "The farmer planted"),
        ("The artist paints", "The artist painted"),
        ("The river drifts", "The river drifted"),
        ("The mountain towers", "The mountain towered"),
        ("The forest whispers", "The forest whispered"),
        ("The ocean crashes", "The ocean crashed"),
        ("The star twinkles", "The star twinkled"),
        ("The moon glows", "The moon glowed"),
        ("The cloud drifts", "The cloud drifted"),
        ("The thunder roars", "The thunder roared"),
        ("The lightning flashes", "The lightning flashed"),
        ("The snake crawls", "The snake crawled"),
        ("The rabbit hops", "The rabbit hopped"),
        ("The eagle soars", "The eagle soared"),
        ("The whale dives", "The whale dived"),
        ("The tiger hunts", "The tiger hunted"),
    ]
    sentences["tense"] = [{"A": p, "B": q, "type": "tense"}
                           for p, q in tense_pairs[:n_per_type]]

    # === Voice ===
    voice_pairs = [
        ("The cat chases the dog", "The dog is chased by the cat"),
        ("The dog bites the man", "The man is bitten by the dog"),
        ("The girl reads the book", "The book is read by the girl"),
        ("The boy throws the ball", "The ball is thrown by the boy"),
        ("The teacher praised the student", "The student was praised by the teacher"),
        ("The wind blows the leaves", "The leaves are blown by the wind"),
        ("The chef cooks the meal", "The meal is cooked by the chef"),
        ("The artist paints the wall", "The wall is painted by the artist"),
        ("The writer finished the novel", "The novel was finished by the writer"),
        ("The company builds the house", "The house is built by the company"),
        ("The driver stops the car", "The car is stopped by the driver"),
        ("The police caught the thief", "The thief was caught by the police"),
        ("The mother loves the child", "The child is loved by the mother"),
        ("The scientist discovered the element", "The element was discovered by the scientist"),
        ("The river carries the boat", "The boat is carried by the river"),
        ("The fire destroyed the forest", "The forest was destroyed by the fire"),
        ("The sun warms the earth", "The earth is warmed by the sun"),
        ("The musician plays the piano", "The piano is played by the musician"),
        ("The farmer grows the wheat", "The wheat is grown by the farmer"),
        ("The king ruled the kingdom", "The kingdom was ruled by the king"),
        ("The cat watches the bird", "The bird is watched by the cat"),
        ("The dog guards the house", "The house is guarded by the dog"),
        ("The wind pushes the cloud", "The cloud is pushed by the wind"),
        ("The rain waters the garden", "The garden is watered by the rain"),
        ("The sun heats the water", "The water is heated by the sun"),
        ("The teacher guides the class", "The class is guided by the teacher"),
        ("The doctor treats the patient", "The patient is treated by the doctor"),
        ("The builder made the wall", "The wall was made by the builder"),
        ("The coach trains the team", "The team is trained by the coach"),
        ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
        ("The editor revised the article", "The article was revised by the editor"),
        ("The pilot flies the plane", "The plane is flown by the pilot"),
        ("The chef baked the cake", "The cake was baked by the chef"),
        ("The author wrote the poem", "The poem was written by the author"),
        ("The director filmed the scene", "The scene was filmed by the director"),
        ("The nurse cared for the patient", "The patient was cared for by the nurse"),
        ("The mechanic fixed the engine", "The engine was fixed by the mechanic"),
        ("The baker made the bread", "The bread was made by the baker"),
        ("The cleaner washed the floor", "The floor was washed by the cleaner"),
        ("The singer performed the song", "The song was performed by the singer"),
    ]
    sentences["voice"] = [{"A": p, "B": q, "type": "voice"}
                           for p, q in voice_pairs[:n_per_type]]

    # === Negation ===
    neg_pairs = [
        ("The cat can sleep", "The cat cannot sleep"),
        ("The dog will run", "The dog will not run"),
        ("The bird does sing", "The bird does not sing"),
        ("The girl is reading", "The girl is not reading"),
        ("The boy has eaten", "The boy has not eaten"),
        ("The car was moving", "The car was not moving"),
        ("The child should play", "The child should not play"),
        ("The man could work", "The man could not work"),
        ("The woman would dance", "The woman would not dance"),
        ("The fish can swim", "The fish cannot swim"),
        ("The student must study", "The student must not study"),
        ("The teacher will speak", "The teacher will not speak"),
        ("The river is flowing", "The river is not flowing"),
        ("The wind might blow", "The wind might not blow"),
        ("The sun is shining", "The sun is not shining"),
        ("The rain was falling", "The rain was not falling"),
        ("The fire has burned", "The fire has not burned"),
        ("The snow will melt", "The snow will not melt"),
        ("The bell was ringing", "The bell was not ringing"),
        ("The dog can bark", "The dog cannot bark"),
        ("The king will rule", "The king will not rule"),
        ("The queen is smiling", "The queen is not smiling"),
        ("The doctor should help", "The doctor should not help"),
        ("The soldier can fight", "The soldier cannot fight"),
        ("The farmer will plant", "The farmer will not plant"),
        ("The artist could paint", "The artist could not paint"),
        ("The river was flowing", "The river was not flowing"),
        ("The wind is blowing", "The wind is not blowing"),
        ("The star was twinkling", "The star was not twinkling"),
        ("The moon is glowing", "The moon is not glowing"),
        ("The cloud was drifting", "The cloud was not drifting"),
        ("The snake can crawl", "The snake cannot crawl"),
        ("The rabbit will hop", "The rabbit will not hop"),
        ("The eagle is soaring", "The eagle is not soaring"),
        ("The whale was diving", "The whale was not diving"),
        ("The tiger can hunt", "The tiger cannot hunt"),
        ("The lion is roaring", "The lion is not roaring"),
        ("The wolf will howl", "The wolf will not howl"),
        ("The bear was sleeping", "The bear was not sleeping"),
        ("The deer is running", "The deer is not running"),
    ]
    sentences["negation"] = [{"A": p, "B": q, "type": "negation"}
                              for p, q in neg_pairs[:n_per_type]]

    return sentences


def generate_general_sentences(n=60):
    """生成多样化的一般句子(无特定约束对)"""
    general = [
        "The weather is beautiful today",
        "Science advances through careful observation",
        "Music brings joy to many people",
        "The city never sleeps at night",
        "Knowledge is power and wisdom",
        "Time flows like a gentle river",
        "The mountain stands tall and proud",
        "Dreams can inspire great achievements",
        "The ocean holds countless mysteries",
        "Art reflects the soul of humanity",
        "The forest provides shelter for animals",
        "Technology changes the world rapidly",
        "The garden blooms in spring",
        "History teaches us valuable lessons",
        "The sky darkens before the storm",
        "Patience is a virtue worth cultivating",
        "The bridge connects two distant shores",
        "Literature opens doors to imagination",
        "The desert stretches endlessly ahead",
        "Courage helps us face our fears",
        "The clock ticks steadily onward",
        "Nature finds a way to adapt",
        "The lake reflects the autumn colors",
        "Friendship enriches our daily lives",
        "The train arrives at the station",
        "Curiosity drives scientific discovery",
        "The wind carries seeds across fields",
        "Hope sustains us through difficult times",
        "The bird builds its nest carefully",
        "Education empowers future generations",
        "The river cuts through the valley",
        "Innovation requires creative thinking",
        "The moon illuminates the dark night",
        "Compassion makes the world better",
        "The volcano erupts with tremendous force",
        "Wisdom comes from diverse experiences",
        "The butterfly transforms inside the cocoon",
        "Perseverance leads to eventual success",
        "The glacier moves imperceptibly slowly",
        "Harmony emerges from balanced elements",
        "The eagle nests on the cliff",
        "Understanding grows through open dialogue",
        "The tide rises and falls predictably",
        "Resilience helps communities recover",
        "The comet streaks across the heavens",
        "Creativity flourishes in freedom",
        "The seedling reaches toward the light",
        "Justice requires fairness and equity",
        "The glacier carved the deep valley",
        "Imagination transcends physical boundaries",
        "The owl hunts in the darkness",
        "Cooperation achieves more than competition",
        "The rain nourishes the thirsty soil",
        "Excellence demands consistent effort",
        "The spider weaves an intricate web",
        "Empathy bridges divides between people",
        "The lighthouse guides ships safely",
        "Progress requires both vision and action",
        "The salmon swims upstream relentlessly",
    ]
    return general[:n]


# ============================================================
# 隐藏状态收集
# ============================================================

def collect_hidden_states(model, tokenizer, device, sentences, n_layers,
                          desc="collecting"):
    """
    收集多个句子在各层的隐藏状态(last token position)

    Args:
        sentences: list of str
        n_layers: int

    Returns:
        all_h: dict {layer_idx: np.array [n_sentences, d_model]}
    """
    layers = get_layers(model)
    all_h = {l: [] for l in range(n_layers)}

    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    for si, text in enumerate(sentences):
        captured.clear()
        try:
            input_ids = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=64).input_ids.to(device)
            with torch.no_grad():
                _ = model(input_ids)

            for li in range(n_layers):
                key = f"L{li}"
                if key in captured:
                    # last token position
                    h = captured[key][0, -1, :].numpy()
                    all_h[li].append(h)
                else:
                    all_h[li].append(np.zeros(1))  # placeholder

        except Exception as e:
            print(f"    [!] Sentence {si} failed: {e}")
            for li in range(n_layers):
                all_h[li].append(np.zeros(1))

        if (si + 1) % 20 == 0:
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] {desc}: {si+1}/{len(sentences)} done")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # Convert to numpy arrays
    for li in range(n_layers):
        if len(all_h[li]) > 0 and all_h[li][0].shape != (1,):
            all_h[li] = np.array(all_h[li])  # [n_sentences, d_model]
        else:
            all_h[li] = None

    return all_h


# ============================================================
# Exp1: 谱间隙分析
# ============================================================

def spectral_gap_analysis(all_h, sample_layers, d_model):
    """
    分析各层PCA解释方差比, 判断是否存在谱间隙

    Returns:
        results: dict with per-layer variance ratios and gap metrics
    """
    results = {}

    for li in sample_layers:
        h = all_h[li]
        if h is None:
            continue

        # Center
        mean_l = h.mean(axis=0)
        h_centered = h - mean_l

        # SVD
        n_samples = h_centered.shape[0]
        k_svd = min(100, n_samples - 1, d_model - 1)
        k_svd = max(k_svd, 2)

        try:
            U, S, Vt = np.linalg.svd(h_centered, full_matrices=False)
        except Exception:
            continue

        # Explained variance ratio
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            continue
        var_ratio = (S**2) / total_var
        cumulative_var = np.cumsum(var_ratio)

        # Effective rank (95% variance)
        erank_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
        # Effective rank (99% variance)
        erank_99 = int(np.searchsorted(cumulative_var, 0.99)) + 1

        # Gap analysis: ratio σ_k / σ_{k+1} for k=1..20
        gap_ratios = {}
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
            if k < len(S) and S[k] > 1e-10:
                gap_ratios[k] = float(S[k-1] / S[k])

        results[f"L{li}"] = {
            "top10_singular_values": [float(s) for s in S[:10]],
            "top10_var_ratio": [float(r) for r in var_ratio[:10]],
            "top10_cumulative_var": [float(c) for c in cumulative_var[:10]],
            "erank_95": erank_95,
            "erank_99": erank_99,
            "gap_ratios": gap_ratios,
            "var_ratio_top50": [float(r) for r in var_ratio[:50]],
        }

    return results


# ============================================================
# Exp2: 线性动力学闭包
# ============================================================

def linear_dynamical_closure(train_h, test_h, sample_layers, d_model,
                              k_values=None, random_seed=42):
    """
    线性动力学闭包测试

    对每个k:
      1. 从训练集计算全局PCA投影 P_k
      2. z_l = P_k (h_l - mean_l) ∈ R^k
      3. 拟合 z_{l+1} = A_l z_l + b_l
      4. 在测试集上评估 R²

    Returns:
        closure_results: dict
    """
    if k_values is None:
        k_values = K_VALUES

    rng = np.random.RandomState(random_seed)

    # === 步骤1: 计算全局PCA ===
    # 收集所有层的约束差分向量
    all_deltas = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is None:
            continue
        # 约束对: 偶数索引=A, 奇数索引=B
        n = h.shape[0]
        if n >= 2:
            for i in range(0, n - 1, 2):
                delta = h[i+1] - h[i]
                all_deltas.append(delta)

    if len(all_deltas) < 10:
        print("    [!] Not enough delta vectors for PCA")
        return {}

    all_deltas = np.array(all_deltas)  # [n_deltas, d_model]
    print(f"    PCA on {all_deltas.shape[0]} delta vectors, d={all_deltas.shape[1]}")

    # SVD of delta matrix
    mean_delta = all_deltas.mean(axis=0)
    delta_centered = all_deltas - mean_delta
    U_delta, S_delta, Vt_delta = np.linalg.svd(delta_centered, full_matrices=False)

    # PCA基: Vt_delta的行 = 主方向
    # P_k = Vt_delta[:k, :]  → z = P_k @ (h - mean)
    max_k = max(k_values)

    # 也计算纯隐藏状态的PCA (用于对照)
    all_h_concat = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            all_h_concat.append(h)
    if len(all_h_concat) > 0:
        all_h_concat = np.vstack(all_h_concat)
        mean_h_global = all_h_concat.mean(axis=0)
        h_centered = all_h_concat - mean_h_global
        _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)
    else:
        Vt_h = Vt_delta
        mean_h_global = mean_delta

    # === 步骤2: 计算每层的mean ===
    layer_means = {}
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            layer_means[li] = h.mean(axis=0)

    # === 步骤3: 对每个k拟合和测试 ===
    closure_results = {}

    for k in k_values:
        if k > min(Vt_delta.shape[0], Vt_delta.shape[1]):
            continue

        # 约束差分PCA投影
        P_delta = Vt_delta[:k, :]  # [k, d_model]

        # 隐藏状态PCA投影 (对照)
        if k <= min(Vt_h.shape[0], Vt_h.shape[1]):
            P_h = Vt_h[:k, :]  # [k, d_model]
        else:
            P_h = P_delta

        # 随机投影 (对照)
        R_random = rng.randn(k, d_model)
        Q, _ = np.linalg.qr(R_random.T)
        P_random = Q.T[:k, :]  # [k, d_model]

        for proj_name, P in [("delta_pca", P_delta), ("h_pca", P_h), ("random", P_random)]:
            r2_train_list = []
            r2_test_list = []

            for li_idx in range(len(sample_layers) - 1):
                li = sample_layers[li_idx]
                li_next = sample_layers[li_idx + 1]

                h_train = train_h.get(li)
                h_train_next = train_h.get(li_next)
                h_test = test_h.get(li)
                h_test_next = test_h.get(li_next)

                if any(x is None for x in [h_train, h_train_next, h_test, h_test_next]):
                    continue

                mean_l = layer_means.get(li, np.zeros(d_model))
                mean_l_next = layer_means.get(li_next, np.zeros(d_model))

                # 投影
                z_train = (P @ (h_train - mean_l).T).T          # [n_train, k]
                z_train_next = (P @ (h_train_next - mean_l_next).T).T
                z_test = (P @ (h_test - mean_l).T).T
                z_test_next = (P @ (h_test_next - mean_l_next).T).T

                # 拟合 z_{l+1} = A z_l + b (Ridge回归)
                n_train = z_train.shape[0]
                if n_train < k + 5:
                    continue

                # 添加偏置项
                ones = np.ones((n_train, 1))
                X = np.hstack([z_train, ones])  # [n_train, k+1]
                Y = z_train_next                  # [n_train, k]

                # Ridge: (X^T X + λI)^{-1} X^T Y
                lam = 0.1  # 正则化
                XtX = X.T @ X + lam * np.eye(X.shape[1])
                XtY = X.T @ Y
                try:
                    W = np.linalg.solve(XtX, XtY)  # [k+1, k]
                except np.linalg.LinAlgError:
                    continue

                A = W[:k, :]  # [k, k]
                b = W[k, :]   # [k]

                # 训练集R²
                Y_pred_train = X @ W
                ss_res_train = np.sum((Y - Y_pred_train)**2)
                ss_tot_train = np.sum((Y - Y.mean(axis=0))**2)
                r2_train = 1 - ss_res_train / max(ss_tot_train, 1e-10)

                # 测试集R²
                ones_test = np.ones((z_test.shape[0], 1))
                X_test = np.hstack([z_test, ones_test])
                Y_pred_test = X_test @ W
                ss_res_test = np.sum((z_test_next - Y_pred_test)**2)
                ss_tot_test = np.sum((z_test_next - z_test_next.mean(axis=0))**2)
                r2_test = 1 - ss_res_test / max(ss_tot_test, 1e-10)

                r2_train_list.append(r2_train)
                r2_test_list.append(r2_test)

            if len(r2_train_list) > 0:
                key = f"k={k}_{proj_name}"
                closure_results[key] = {
                    "k": k,
                    "projection": proj_name,
                    "r2_train_mean": float(np.mean(r2_train_list)),
                    "r2_train_std": float(np.std(r2_train_list)),
                    "r2_test_mean": float(np.mean(r2_test_list)),
                    "r2_test_std": float(np.std(r2_test_list)),
                    "n_layer_pairs": len(r2_train_list),
                    "r2_per_layer_train": [float(x) for x in r2_train_list],
                    "r2_per_layer_test": [float(x) for x in r2_test_list],
                }

    return closure_results


# ============================================================
# Exp3: 跨约束稳定性
# ============================================================

def cross_constraint_closure(train_h_by_type, test_h_by_type, sample_layers,
                              d_model, k=5):
    """
    跨约束类型动力学闭包测试

    在一种约束类型上训练F, 在另一种上测试
    """
    # 用SVA训练PCA和F
    train_h_sva = train_h_by_type.get("number_sva")
    if train_h_sva is None:
        return {}

    # 计算PCA基 (SVA约束差分)
    all_deltas = []
    for li in sample_layers:
        h = train_h_sva.get(li)
        if h is None:
            continue
        n = h.shape[0]
        for i in range(0, n - 1, 2):
            delta = h[i+1] - h[i]
            all_deltas.append(delta)

    if len(all_deltas) < 5:
        return {}

    all_deltas = np.array(all_deltas)
    _, _, Vt = np.linalg.svd(all_deltas - all_deltas.mean(axis=0), full_matrices=False)
    P = Vt[:k, :]  # [k, d_model]

    # 计算层均值 (SVA训练集)
    layer_means = {}
    for li in sample_layers:
        h = train_h_sva.get(li)
        if h is not None:
            layer_means[li] = h.mean(axis=0)

    # 在SVA上训练F
    r2_cross = {}

    for train_type in ["number_sva"]:
        train_h = train_h_by_type.get(train_type)
        if train_h is None:
            continue

        # 拟合每层的A, b
        models = {}
        for li_idx in range(len(sample_layers) - 1):
            li = sample_layers[li_idx]
            li_next = sample_layers[li_idx + 1]

            h = train_h.get(li)
            h_next = train_h.get(li_next)
            if h is None or h_next is None:
                continue

            mean_l = layer_means.get(li, np.zeros(d_model))
            mean_l_next = layer_means.get(li_next, np.zeros(d_model))

            z = (P @ (h - mean_l).T).T
            z_next = (P @ (h_next - mean_l_next).T).T

            ones = np.ones((z.shape[0], 1))
            X = np.hstack([z, ones])
            Y = z_next

            lam = 0.1
            XtX = X.T @ X + lam * np.eye(X.shape[1])
            XtY = X.T @ Y
            try:
                W = np.linalg.solve(XtX, XtY)
                models[(li, li_next)] = W
            except:
                continue

        # 在各种约束类型上测试
        for test_type, test_h in test_h_by_type.items():
            r2_list = []
            for li_idx in range(len(sample_layers) - 1):
                li = sample_layers[li_idx]
                li_next = sample_layers[li_idx + 1]

                if (li, li_next) not in models:
                    continue

                h = test_h.get(li)
                h_next = test_h.get(li_next)
                if h is None or h_next is None:
                    continue

                mean_l = layer_means.get(li, np.zeros(d_model))
                mean_l_next = layer_means.get(li_next, np.zeros(d_model))

                z = (P @ (h - mean_l).T).T
                z_next = (P @ (h_next - mean_l_next).T).T

                ones = np.ones((z.shape[0], 1))
                X_test = np.hstack([z, ones])
                Y_pred = X_test @ models[(li, li_next)]

                ss_res = np.sum((z_next - Y_pred)**2)
                ss_tot = np.sum((z_next - z_next.mean(axis=0))**2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                r2_list.append(r2)

            if len(r2_list) > 0:
                r2_cross[f"train={train_type}_test={test_type}"] = {
                    "train_type": train_type,
                    "test_type": test_type,
                    "r2_mean": float(np.mean(r2_list)),
                    "r2_std": float(np.std(r2_list)),
                    "r2_per_layer": [float(x) for x in r2_list],
                    "k": k,
                }

    return r2_cross


# ============================================================
# Exp4: Markov性测试
# ============================================================

def markov_order_test(train_h, test_h, sample_layers, d_model, k=5):
    """
    测试动力学是否是一阶Markov:
      z_{l+1} = F_1(z_l) vs z_{l+1} = F_2(z_l, z_{l-1})
    """
    # 计算PCA基
    all_deltas = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is None:
            continue
        n = h.shape[0]
        for i in range(0, n - 1, 2):
            all_deltas.append(h[i+1] - h[i])

    if len(all_deltas) < 5:
        return {}

    all_deltas = np.array(all_deltas)
    _, _, Vt = np.linalg.svd(all_deltas - all_deltas.mean(axis=0), full_matrices=False)
    P = Vt[:k, :]

    layer_means = {}
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            layer_means[li] = h.mean(axis=0)

    r2_order1 = []
    r2_order2 = []

    for li_idx in range(1, len(sample_layers) - 1):
        li_prev = sample_layers[li_idx - 1]
        li = sample_layers[li_idx]
        li_next = sample_layers[li_idx + 1]

        h_train = train_h.get(li)
        h_train_prev = train_h.get(li_prev)
        h_train_next = train_h.get(li_next)
        h_test = test_h.get(li)
        h_test_prev = test_h.get(li_prev)
        h_test_next = test_h.get(li_next)

        if any(x is None for x in [h_train, h_train_prev, h_train_next,
                                     h_test, h_test_prev, h_test_next]):
            continue

        mean_l = layer_means.get(li, np.zeros(d_model))
        mean_l_prev = layer_means.get(li_prev, np.zeros(d_model))
        mean_l_next = layer_means.get(li_next, np.zeros(d_model))

        # 投影
        z_train = (P @ (h_train - mean_l).T).T
        z_train_prev = (P @ (h_train_prev - mean_l_prev).T).T
        z_train_next = (P @ (h_train_next - mean_l_next).T).T
        z_test = (P @ (h_test - mean_l).T).T
        z_test_prev = (P @ (h_test_prev - mean_l_prev).T).T
        z_test_next = (P @ (h_test_next - mean_l_next).T).T

        n_train = z_train.shape[0]
        if n_train < 2 * k + 5:
            continue

        # 一阶: z_{l+1} = A z_l + b
        ones_train = np.ones((n_train, 1))
        X1 = np.hstack([z_train, ones_train])
        Y = z_train_next
        lam = 0.1
        W1 = np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ Y)

        ones_test = np.ones((z_test.shape[0], 1))
        X1_test = np.hstack([z_test, ones_test])
        Y_pred1 = X1_test @ W1
        ss_res1 = np.sum((z_test_next - Y_pred1)**2)
        ss_tot = np.sum((z_test_next - z_test_next.mean(axis=0))**2)
        r2_1 = 1 - ss_res1 / max(ss_tot, 1e-10)

        # 二阶: z_{l+1} = A z_l + B z_{l-1} + c
        X2 = np.hstack([z_train, z_train_prev, ones_train])
        W2 = np.linalg.solve(X2.T @ X2 + lam * np.eye(X2.shape[1]), X2.T @ Y)

        X2_test = np.hstack([z_test, z_test_prev, ones_test])
        Y_pred2 = X2_test @ W2
        ss_res2 = np.sum((z_test_next - Y_pred2)**2)
        r2_2 = 1 - ss_res2 / max(ss_tot, 1e-10)

        r2_order1.append(r2_1)
        r2_order2.append(r2_2)

    if len(r2_order1) == 0:
        return {}

    return {
        "k": k,
        "r2_order1_mean": float(np.mean(r2_order1)),
        "r2_order1_std": float(np.std(r2_order1)),
        "r2_order2_mean": float(np.mean(r2_order2)),
        "r2_order2_std": float(np.std(r2_order2)),
        "delta_r2_mean": float(np.mean(np.array(r2_order2) - np.array(r2_order1))),
        "r2_order1_per_layer": [float(x) for x in r2_order1],
        "r2_order2_per_layer": [float(x) for x in r2_order2],
        "conclusion": "First-order Markov" if np.mean(np.array(r2_order2) - np.array(r2_order1)) < 0.05
                      else "Higher-order memory needed",
    }


# ============================================================
# 主函数
# ============================================================

def run_phase226(model_name: str):
    """Phase 226 完整流程"""

    print(f"\n{'='*60}")
    print(f"=== Phase 226: Dynamical Closure ({model_name}) ===")
    print(f"{'='*60}")

    t_start = time.time()

    # === 0. 加载模型 ===
    model, tokenizer, device = load_model_bf16_sdpa(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    sample_layers = get_sample_layers(n_layers)
    print(f"  n_layers={n_layers}, d_model={d_model}")
    print(f"  Sample layers: {sample_layers}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === 1. 生成句子 ===
    constraint_sentences = generate_sentences(n_per_type=40)
    general_sentences = generate_general_sentences(n=60)

    # 训练/测试分割
    # 每种约束类型: 前30对训练, 后10对测试
    # A/B交替: A=偶数, B=奇数
    train_sentences_all = []
    test_sentences_all = []
    train_h_by_type = {}
    test_h_by_type = {}

    for ctype, pairs in constraint_sentences.items():
        n_train = min(30, len(pairs))
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:n_train+10]

        train_sents = []
        for p in train_pairs:
            train_sents.append(p["A"])
            train_sents.append(p["B"])

        test_sents = []
        for p in test_pairs:
            test_sents.append(p["A"])
            test_sents.append(p["B"])

        # 收集训练集隐藏状态
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Collecting train h for {ctype} ({len(train_sents)} sents)...")
        train_h = collect_hidden_states(model, tokenizer, device, train_sents,
                                         n_layers, desc=f"train_{ctype}")
        train_h_by_type[ctype] = train_h

        # 收集测试集隐藏状态
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Collecting test h for {ctype} ({len(test_sents)} sents)...")
        test_h = collect_hidden_states(model, tokenizer, device, test_sents,
                                        n_layers, desc=f"test_{ctype}")
        test_h_by_type[ctype] = test_h

        train_sentences_all.extend(train_sents)
        test_sentences_all.extend(test_sents)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 也收集一般句子的隐藏状态 (用于额外测试)
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Collecting general h ({len(general_sentences)} sents)...")
    general_h = collect_hidden_states(model, tokenizer, device, general_sentences,
                                       n_layers, desc="general")

    # === 合并所有训练数据 ===
    # 将所有约束类型的训练数据合并
    train_h_merged = {}
    test_h_merged = {}
    for li in sample_layers:
        train_parts = []
        test_parts = []
        for ctype in constraint_sentences:
            if train_h_by_type.get(ctype, {}).get(li) is not None:
                train_parts.append(train_h_by_type[ctype][li])
            if test_h_by_type.get(ctype, {}).get(li) is not None:
                test_parts.append(test_h_by_type[ctype][li])
        if train_parts:
            train_h_merged[li] = np.vstack(train_parts)
        if test_parts:
            test_h_merged[li] = np.vstack(test_parts)

    print(f"\n  Train set size: {train_h_merged[sample_layers[0]].shape[0] if sample_layers[0] in train_h_merged else 'N/A'}")
    print(f"  Test set size: {test_h_merged[sample_layers[0]].shape[0] if sample_layers[0] in test_h_merged else 'N/A'}")

    # === 2. Exp1: 谱间隙分析 ===
    print(f"\n{'='*40}")
    print(f"  Exp1: Spectral Gap Analysis")
    print(f"{'='*40}")

    exp1_results = spectral_gap_analysis(train_h_merged, sample_layers, d_model)

    # 打印关键结果
    for layer_key in sorted(exp1_results.keys()):
        r = exp1_results[layer_key]
        top5_var = sum(r["top10_var_ratio"][:5])
        top10_var = sum(r["top10_var_ratio"][:10])
        print(f"  {layer_key}: erank95={r['erank_95']}, erank99={r['erank_99']}, "
              f"top5_var={top5_var:.3f}, top10_var={top10_var:.3f}")
        # 谱间隙
        gaps = r["gap_ratios"]
        for k in [3, 4, 5, 6, 7, 10]:
            if k in gaps:
                print(f"    gap({k}/{k+1}) = {gaps[k]:.2f}")

    # === 3. Exp2: 线性动力学闭包 ===
    print(f"\n{'='*40}")
    print(f"  Exp2: Linear Dynamical Closure")
    print(f"{'='*40}")

    exp2_results = linear_dynamical_closure(
        train_h_merged, test_h_merged, sample_layers, d_model,
        k_values=K_VALUES
    )

    # 打印关键结果
    print(f"\n  --- R² vs k (test set) ---")
    print(f"  {'k':>4} | {'delta_pca':>12} | {'h_pca':>12} | {'random':>12} | {'Δpca-rand':>12}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for k in K_VALUES:
        delta_key = f"k={k}_delta_pca"
        h_key = f"k={k}_h_pca"
        rand_key = f"k={k}_random"

        r2_delta = exp2_results.get(delta_key, {}).get("r2_test_mean", float('nan'))
        r2_h = exp2_results.get(h_key, {}).get("r2_test_mean", float('nan'))
        r2_rand = exp2_results.get(rand_key, {}).get("r2_test_mean", float('nan'))
        delta_over_rand = r2_delta - r2_rand if not (np.isnan(r2_delta) or np.isnan(r2_rand)) else float('nan')

        print(f"  {k:4d} | {r2_delta:12.4f} | {r2_h:12.4f} | {r2_rand:12.4f} | {delta_over_rand:12.4f}")

    # === 4. Exp3: 跨约束稳定性 ===
    print(f"\n{'='*40}")
    print(f"  Exp3: Cross-Constraint Stability (k=5)")
    print(f"{'='*40}")

    exp3_results = cross_constraint_closure(
        train_h_by_type, test_h_by_type, sample_layers, d_model, k=5
    )

    for key, val in exp3_results.items():
        print(f"  {key}: R²={val['r2_mean']:.4f} ± {val['r2_std']:.4f}")

    # === 5. Exp4: Markov性测试 ===
    print(f"\n{'='*40}")
    print(f"  Exp4: Markov Order Test (k=5)")
    print(f"{'='*40}")

    exp4_results = markov_order_test(
        train_h_merged, test_h_merged, sample_layers, d_model, k=5
    )

    if exp4_results:
        print(f"  Order-1 R²: {exp4_results['r2_order1_mean']:.4f} ± {exp4_results['r2_order1_std']:.4f}")
        print(f"  Order-2 R²: {exp4_results['r2_order2_mean']:.4f} ± {exp4_results['r2_order2_std']:.4f}")
        print(f"  ΔR²: {exp4_results['delta_r2_mean']:.4f}")
        print(f"  Conclusion: {exp4_results['conclusion']}")

    # === 6. 汇总结果 ===
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "n_train": int(train_h_merged[sample_layers[0]].shape[0]) if sample_layers[0] in train_h_merged else 0,
        "n_test": int(test_h_merged[sample_layers[0]].shape[0]) if sample_layers[0] in test_h_merged else 0,
        "exp1_spectral_gap": exp1_results,
        "exp2_dynamical_closure": exp2_results,
        "exp3_cross_constraint": exp3_results,
        "exp4_markov": exp4_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 保存
    out_path = OUTPUT_DIR / f"phase226_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {out_path}")

    # === 7. 释放模型 ===
    print(f"\n  Releasing {model_name}...")
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    t_total = time.time() - t_start
    print(f"  Total time: {t_total/60:.1f} min")

    return all_results


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_phase226(name)
            except Exception as e:
                print(f"\n!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
    else:
        run_phase226(model_name)

    print("\nPhase 226 complete!")
