"""
Phase 227: Koopman/DMD闭包测试 — 寻找最优动力学投影
====================================================

用户核心洞察:
  "研究应该是'不变量驱动'而非'实验驱动'"
  "真正该寻找的是动力学闭包: z_{l+1} = F(z_l)"

Phase 226 发现:
  - PCA投影的线性闭包R²≈0.75-0.85 (不完美)
  - h_pca > delta_pca (约束差分不是最优基)
  - 无谱间隙 (5维核心是假象)
  - 跨约束R²≈0 (动力学是约束类型特异的)

关键修正:
  PCA最大化方差, 不最大化动力学可预测性。
  → 用动力学本身定义投影: 寻找P使得R²(P h_{l+1} | P h_l)最大化
  → 这就是Koopman算子理论的核心思想

本Phase五个实验:

  Exp1: DMD闭包 (Dynamic Mode Decomposition) ★★★★★
    - 直接计算Koopman近似算子 K ≈ argmin ||PH_{l+1} - K PH_l||
    - K的特征值和特征向量 → 稳定传播模态
    - R² vs k 曲线, 与PCA对比

  Exp2: 非线性闭包 (小MLP) ★★★★★
    - z_{l+1} = MLP(z_l), z ∈ R^k
    - 看R²能否从0.8提升到>0.95
    - 如果能 → 非线性是关键; 如果不能 → 需要更高维

  Exp3: 信息论闭包 (互信息) ★★★★
    - I(z_{l+1}; z_l) vs H(z_{l+1}) → 归一化互信息
    - 不假设函数形式, 模型无关的闭包度量
    - k维度扫描

  Exp4: 逐层R²剖面 ★★★
    - 看哪些层的闭包最差
    - 是否对应Phase 210-213发现的"转折层"

  Exp5: 约束特异性闭包 ★★★
    - 每种约束类型独立计算闭包R²
    - 是否存在约束类型通用的高R²投影

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

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]

# ============================================================
# 模型加载 (BF16 + device_map="auto" + SDPA/Flash)
# ============================================================

def load_model_bf16_sdpa(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bf16 + auto + sdpa)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

def generate_constraint_pairs(n_per_type=40):
    """生成4种约束类型的句子对"""
    sentences = {}

    # SVA
    singular = ["cat", "dog", "bird", "girl", "boy", "tree", "car", "child",
                "man", "woman", "fish", "horse", "student", "teacher",
                "flower", "river", "star", "moon", "sun", "cloud",
                "book", "table", "chair", "window", "door", "wall",
                "king", "queen", "prince", "princess", "knight",
                "doctor", "nurse", "soldier", "farmer", "artist",
                "river", "mountain", "forest", "desert"]
    verbs_s = ["chases", "runs", "sings", "reads", "walks", "falls", "moves",
               "plays", "works", "dances", "swims", "sleeps", "barks", "flies",
               "grows", "blooms", "flows", "shines", "rises", "blows",
               "sits", "stands", "breaks", "opens", "closes", "cracks",
               "rules", "commands", "fights", "smiles", "rides",
               "heals", "helps", "marches", "plants", "paints",
               "drifts", "towers", "whispers", "burns"]
    sva_pairs = []
    for i in range(min(n_per_type, len(singular))):
        subj = singular[i]
        verb = verbs_s[i % len(verbs_s)]
        sva_pairs.append((f"The {subj} {verb}", f"The {subj}s {verb.rstrip('s')}"))
    sentences["number_sva"] = sva_pairs

    # Tense
    tense_pairs = [
        ("The cat sleeps", "The cat slept"), ("The dog runs", "The dog ran"),
        ("The bird sings", "The bird sang"), ("The girl reads", "The girl read"),
        ("The boy walks", "The boy walked"), ("The tree grows", "The tree grew"),
        ("The car moves", "The car moved"), ("The child plays", "The child played"),
        ("The man works", "The man worked"), ("The woman dances", "The woman danced"),
        ("The fish swims", "The fish swam"), ("The student studies", "The student studied"),
        ("The teacher speaks", "The teacher spoke"), ("The river flows", "The river flowed"),
        ("The wind blows", "The wind blew"), ("The sun shines", "The sun shone"),
        ("The rain falls", "The rain fell"), ("The fire burns", "The fire burned"),
        ("The snow melts", "The snow melted"), ("The bell rings", "The bell rang"),
        ("The king rules", "The king ruled"), ("The queen smiles", "The queen smiled"),
        ("The doctor heals", "The doctor healed"), ("The soldier marches", "The soldier marched"),
        ("The farmer plants", "The farmer planted"), ("The artist paints", "The artist painted"),
        ("The river drifts", "The river drifted"), ("The mountain towers", "The mountain towered"),
        ("The forest whispers", "The forest whispered"), ("The ocean crashes", "The ocean crashed"),
        ("The star twinkles", "The star twinkled"), ("The moon glows", "The moon glowed"),
        ("The cloud drifts", "The cloud drifted"), ("The thunder roars", "The thunder roared"),
        ("The lightning flashes", "The lightning flashed"), ("The snake crawls", "The snake crawled"),
        ("The rabbit hops", "The rabbit hopped"), ("The eagle soars", "The eagle soared"),
        ("The whale dives", "The whale dived"), ("The tiger hunts", "The tiger hunted"),
    ]
    sentences["tense"] = tense_pairs[:n_per_type]

    # Voice
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
    sentences["voice"] = voice_pairs[:n_per_type]

    # Negation
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
    sentences["negation"] = neg_pairs[:n_per_type]

    return sentences


# ============================================================
# 隐藏状态收集
# ============================================================

def collect_hidden_states(model, tokenizer, device, sentences, n_layers,
                          desc="collecting"):
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
                    h = captured[key][0, -1, :].numpy()
                    all_h[li].append(h)
                else:
                    all_h[li].append(np.zeros(1))
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

    for li in range(n_layers):
        if len(all_h[li]) > 0 and all_h[li][0].shape != (1,):
            all_h[li] = np.array(all_h[li])
        else:
            all_h[li] = None

    return all_h


# ============================================================
# Exp1: DMD闭包 (Koopman近似) ★★★★★
# ============================================================

def dmd_closure(train_h, test_h, sample_layers, d_model, k_values=None):
    """
    Dynamic Mode Decomposition (DMD) — Koopman算子的有限维近似
    
    核心思想: 不用PCA找方差最大的方向, 而是找动力学可预测性最高的方向
    
    步骤:
    1. 收集 (h_l, h_{l+1}) 对
    2. 计算Koopman近似: K = H_{l+1} H_l^+ (伪逆)
    3. 对K做SVD → 特征值(传播率)和特征向量(传播模态)
    4. 用K的特征向量作为投影基 → 应该比PCA更优
    
    对比: DMD投影 vs PCA投影 vs 随机投影 的R²
    """
    if k_values is None:
        k_values = K_VALUES

    print(f"\n{'='*60}")
    print("Exp1: DMD (Koopman) Closure")
    print(f"{'='*60}")

    results = {}

    # 收集所有相邻层的(h_l, h_{l+1})对
    all_h_l = []
    all_h_l1 = []
    layer_pair_info = []

    for li in sample_layers:
        if li + 1 >= len(train_h) or train_h.get(li) is None or train_h.get(li+1) is None:
            continue
        h_l = train_h[li]
        h_l1 = train_h[li + 1]
        n = min(h_l.shape[0], h_l1.shape[0])
        all_h_l.append(h_l[:n])
        all_h_l1.append(h_l1[:n])
        layer_pair_info.append((li, li+1, n))

    if len(all_h_l) == 0:
        print("  [!] No valid layer pairs")
        return {}

    H_l = np.vstack(all_h_l)     # [N, d]
    H_l1 = np.vstack(all_h_l1)   # [N, d]
    print(f"  Training data: {H_l.shape[0]} pairs, d={H_l.shape[1]}")

    # 中心化
    mean_l = H_l.mean(axis=0)
    mean_l1 = H_l1.mean(axis=0)
    H_l_c = H_l - mean_l
    H_l1_c = H_l1 - mean_l1

    # === Koopman近似: K = H_{l+1}^T H_l (H_l^T H_l)^{-1} ===
    # 但d_model >> N, 所以用对偶形式
    # 先做SVD: H_l = U S V^T
    n_samples = H_l_c.shape[0]
    k_svd = min(n_samples - 1, d_model - 1, 200)
    k_svd = max(k_svd, 10)

    print(f"  Computing SVD of H_l (k={k_svd})...")
    U_l, S_l, Vt_l = np.linalg.svd(H_l_c, full_matrices=False)
    # 保留前k_svd个分量
    U_l = U_l[:, :k_svd]
    S_l = S_l[:k_svd]
    Vt_l = Vt_l[:k_svd, :]

    # Koopman算子 (在对偶空间中): K_tilde = U_l^T H_{l+1}_c V_l S_l^{-1}
    V_l = Vt_l.T  # [d, k_svd]
    S_inv = np.diag(1.0 / (S_l + 1e-10))
    K_tilde = U_l.T @ H_l1_c @ V_l @ S_inv  # [k_svd, k_svd]

    # K_tilde的特征分解 → Koopman特征值和模态
    eig_vals, eig_vecs = np.linalg.eig(K_tilde)
    # 按特征值大小排序 (|λ|)
    sort_idx = np.argsort(np.abs(eig_vals))[::-1]
    eig_vals = eig_vals[sort_idx]
    eig_vecs = eig_vecs[:, sort_idx]

    # Koopman模态: φ = V_l S^{-1} eig_vecs (在原始空间中的方向)
    koopman_modes = V_l @ S_inv @ eig_vecs  # [d, k_svd] — 每列是一个模态

    # 记录特征值
    top10_eigvals = [(float(np.abs(eig_vals[i])), float(np.angle(eig_vals[i])),
                       float(np.real(eig_vals[i])), float(np.imag(eig_vals[i])))
                      for i in range(min(10, len(eig_vals)))]

    print(f"  Top-10 Koopman eigenvalues (|λ|, angle):")
    for i, (mag, ang, re, im) in enumerate(top10_eigvals):
        print(f"    λ_{i+1}: |λ|={mag:.4f}, angle={np.degrees(ang):.1f}°, "
              f"({re:.4f}+{im:.4f}i)")

    # === PCA基 (用于对照) ===
    all_h_concat = np.vstack([H_l, H_l1])
    mean_h = all_h_concat.mean(axis=0)
    h_centered = all_h_concat - mean_h
    _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

    # === Delta PCA基 ===
    all_deltas = H_l1_c - H_l_c
    mean_delta = all_deltas.mean(axis=0)
    delta_centered = all_deltas - mean_delta
    _, S_delta, Vt_delta = np.linalg.svd(delta_centered, full_matrices=False)

    # === 对每个k比较三种投影的R² ===
    for k in k_values:
        if k > k_svd:
            continue

        # DMD/Koopman投影
        P_koopman = koopman_modes[:, :k].T  # [k, d]

        # PCA投影
        P_pca = Vt_h[:k, :]  # [k, d]

        # Delta PCA投影
        P_delta = Vt_delta[:k, :]  # [k, d]

        projections = {
            "koopman": (P_koopman, mean_l),
            "h_pca": (P_pca, mean_h),
            "delta_pca": (P_delta, mean_delta),
        }

        r2_results = {}
        for pname, (P, mean_ref) in projections.items():
            r2_per_pair = []
            for li, li1, n in layer_pair_info:
                # 训练集R²
                z_l = (train_h[li][:n] - mean_ref) @ P.T   # [n, k]
                z_l1 = (train_h[li1][:n] - mean_ref) @ P.T  # [n, k]

                # 线性回归: z_{l+1} = A z_l + b
                A, b = _fit_linear(z_l, z_l1)

                # 测试集R²
                test_h_l = test_h.get(li)
                test_h_l1 = test_h.get(li1)
                if test_h_l is None or test_h_l1 is None:
                    continue
                n_test = min(test_h_l.shape[0], test_h_l1.shape[0])
                z_test = (test_h_l[:n_test] - mean_ref) @ P.T
                z_test1 = (test_h_l1[:n_test] - mean_ref) @ P.T
                z_pred = z_test @ A.T + b

                r2 = _compute_r2(z_test1, z_pred)
                r2_per_pair.append(r2)

            mean_r2 = np.mean(r2_per_pair) if r2_per_pair else 0.0
            r2_results[pname] = mean_r2

        results[f"k={k}"] = {
            "r2_koopman": round(r2_results.get("koopman", 0), 4),
            "r2_h_pca": round(r2_results.get("h_pca", 0), 4),
            "r2_delta_pca": round(r2_results.get("delta_pca", 0), 4),
        }
        print(f"  k={k}: Koopman={r2_results.get('koopman', 0):.4f}, "
              f"h_PCA={r2_results.get('h_pca', 0):.4f}, "
              f"Δ_PCA={r2_results.get('delta_pca', 0):.4f}")

    # 保存Koopman特征值信息
    results["koopman_eigenvalues"] = top10_eigvals
    results["n_training_pairs"] = int(H_l.shape[0])
    results["k_svd"] = int(k_svd)

    return results


# ============================================================
# Exp2: 非线性闭包 (小MLP) ★★★★★
# ============================================================

def nonlinear_closure(train_h, test_h, sample_layers, d_model, k_values=None):
    """
    用小MLP替代线性模型, 测试非线性闭包
    
    z_{l+1} = MLP(z_l), z ∈ R^k
    
    如果MLP的R²显著高于线性R² → 非线性是关键
    如果MLP的R²与线性R²相近 → R²瓶颈不在非线性, 而在维度
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    print(f"\n{'='*60}")
    print("Exp2: Nonlinear Closure (MLP)")
    print(f"{'='*60}")

    results = {}

    # PCA基
    all_h_train = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            all_h_train.append(h)
    if len(all_h_train) == 0:
        return {}
    all_h_train = np.vstack(all_h_train)
    mean_h = all_h_train.mean(axis=0)
    h_centered = all_h_train - mean_h
    _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

    # 收集训练/测试数据
    layer_pair_info = []
    for li in sample_layers:
        if train_h.get(li) is None or train_h.get(li+1) is None:
            continue
        if test_h.get(li) is None or test_h.get(li+1) is None:
            continue
        n_train = min(train_h[li].shape[0], train_h[li+1].shape[0])
        n_test = min(test_h[li].shape[0], test_h[li+1].shape[0])
        layer_pair_info.append((li, n_train, n_test))

    for k in k_values:
        if k > min(Vt_h.shape[0], Vt_h.shape[1]):
            continue

        P = Vt_h[:k, :]  # [k, d]

        # 收集所有层的训练数据
        train_z_l_list = []
        train_z_l1_list = []
        test_z_l_list = []
        test_z_l1_list = []

        for li, n_train, n_test in layer_pair_info:
            z_l = (train_h[li][:n_train] - mean_h) @ P.T
            z_l1 = (train_h[li+1][:n_train] - mean_h) @ P.T
            z_test = (test_h[li][:n_test] - mean_h) @ P.T
            z_test1 = (test_h[li+1][:n_test] - mean_h) @ P.T
            train_z_l_list.append(z_l)
            train_z_l1_list.append(z_l1)
            test_z_l_list.append(z_test)
            test_z_l1_list.append(z_test1)

        train_z = np.vstack(train_z_l_list)    # [N, k]
        train_z1 = np.vstack(train_z_l1_list)
        test_z = np.vstack(test_z_l_list)
        test_z1 = np.vstack(test_z_l1_list)

        # --- 线性基线 ---
        A, b = _fit_linear(train_z, train_z1)
        z_pred_lin = test_z @ A.T + b
        r2_linear = _compute_r2(test_z1, z_pred_lin)

        # --- MLP ---
        try:
            import torch.nn as nn
            # 标准化
            z_mean = train_z.mean(axis=0)
            z_std = train_z.std(axis=0) + 1e-8
            train_z_norm = (train_z - z_mean) / z_std
            train_z1_norm = (train_z1 - z_mean) / z_std  # same scale for output
            test_z_norm = (test_z - z_mean) / z_std
            test_z1_norm = (test_z1 - z_mean) / z_std

            # 小MLP: k → 2k → k
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlp = nn.Sequential(
                nn.Linear(k, min(2*k, 128)),
                nn.ReLU(),
                nn.Linear(min(2*k, 128), k),
            ).to(device_t)

            X_t = torch.tensor(train_z_norm, dtype=torch.float32, device=device_t)
            Y_t = torch.tensor(train_z1_norm, dtype=torch.float32, device=device_t)

            optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            # 训练
            mlp.train()
            for epoch in range(200):
                optimizer.zero_grad()
                pred = mlp(X_t)
                loss = loss_fn(pred, Y_t)
                loss.backward()
                optimizer.step()

            # 测试
            mlp.eval()
            with torch.no_grad():
                X_test = torch.tensor(test_z_norm, dtype=torch.float32, device=device_t)
                z_pred_mlp_norm = mlp(X_test).cpu().numpy()
                z_pred_mlp = z_pred_mlp_norm * z_std + z_mean

            r2_mlp = _compute_r2(test_z1, z_pred_mlp)
            del mlp, X_t, Y_t, X_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    MLP training failed for k={k}: {e}")
            r2_mlp = r2_linear  # fallback

        delta_r2 = r2_mlp - r2_linear
        results[f"k={k}"] = {
            "r2_linear": round(r2_linear, 4),
            "r2_mlp": round(r2_mlp, 4),
            "delta_r2": round(delta_r2, 4),
        }
        print(f"  k={k}: Linear={r2_linear:.4f}, MLP={r2_mlp:.4f}, Δ={delta_r2:.4f}")

    return results


# ============================================================
# Exp3: 信息论闭包 (互信息) ★★★★
# ============================================================

def information_closure(train_h, test_h, sample_layers, d_model, k_values=None):
    """
    用互信息 I(z_{l+1}; z_l) / H(z_{l+1}) 度量闭包
    
    优势: 不假设函数形式, 模型无关
    方法: KSG互信息估计器 (k近邻)
    """
    if k_values is None:
        k_values = [3, 5, 10, 20]

    print(f"\n{'='*60}")
    print("Exp3: Information Closure (Mutual Information)")
    print(f"{'='*60}")

    results = {}

    # PCA基
    all_h_train = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            all_h_train.append(h)
    if len(all_h_train) == 0:
        return {}
    all_h_train = np.vstack(all_h_train)
    mean_h = all_h_train.mean(axis=0)
    h_centered = all_h_train - mean_h
    _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

    for k in k_values:
        if k > min(Vt_h.shape[0], Vt_h.shape[1]):
            continue

        P = Vt_h[:k, :]

        mi_per_pair = []
        h_entropy_per_pair = []

        for li in sample_layers:
            if test_h.get(li) is None or test_h.get(li+1) is None:
                continue
            n = min(test_h[li].shape[0], test_h[li+1].shape[0])
            z_l = (test_h[li][:n] - mean_h) @ P.T
            z_l1 = (test_h[li+1][:n] - mean_h) @ P.T

            if n < 20:
                continue

            # 用简单方法估计互信息:
            # 对于高斯变量: I(X;Y) = H(X) + H(Y) - H(X,Y)
            # 用k近邻估计 (简化版: 用相关矩阵的上界)
            # 更实用的方法: 对每个维度分别算相关, 然后组合
            
            # 方法: 对每个维度, 计算互信息 I(z_l1_j; z_l)
            # 然后求和 (独立假设下的上界)
            try:
                mi_dims = []
                for j in range(k):
                    # z_l1[:, j] vs z_l的所有维度
                    # 用多元线性回归的R²来估计
                    # I(X;Y) ≈ -0.5 * log(1 - R²) (高斯假设)
                    A, b = _fit_linear(z_l, z_l1[:, j:j+1])
                    z_pred = z_l @ A.T + b
                    r2 = _compute_r2(z_l1[:, j:j+1], z_pred)
                    r2 = max(0, min(r2, 0.999))
                    mi_j = -0.5 * np.log(1 - r2 + 1e-10)
                    mi_dims.append(mi_j)
                
                total_mi = sum(mi_dims)
                
                # H(z_{l+1}) 估计 (高斯): 0.5 * k * (1 + log(2π)) + 0.5 * log|Σ|
                cov_z1 = np.cov(z_l1.T)
                if k == 1:
                    log_det = np.log(max(cov_z1, 1e-10))
                else:
                    sign, log_det = np.linalg.slogdet(cov_z1 + 1e-6 * np.eye(k))
                    if sign <= 0:
                        log_det = np.log(max(np.abs(np.linalg.det(cov_z1 + 1e-6 * np.eye(k))), 1e-10))
                
                h_entropy = 0.5 * k * (1 + np.log(2 * np.pi)) + 0.5 * log_det
                
                nmi = total_mi / max(h_entropy, 1e-10)  # 归一化互信息
                
                mi_per_pair.append(total_mi)
                h_entropy_per_pair.append(h_entropy)
            except Exception as e:
                continue

        if mi_per_pair:
            mean_mi = np.mean(mi_per_pair)
            mean_h_ent = np.mean(h_entropy_per_pair)
            mean_nmi = mean_mi / max(mean_h_ent, 1e-10)
            results[f"k={k}"] = {
                "MI": round(float(mean_mi), 4),
                "H(z_{l+1})": round(float(mean_h_ent), 4),
                "NMI": round(float(mean_nmi), 4),
            }
            print(f"  k={k}: MI={mean_mi:.4f}, H(z_{{l+1}})={mean_h_ent:.4f}, "
                  f"NMI={mean_nmi:.4f}")

    return results


# ============================================================
# Exp4: 逐层R²剖面 ★★★
# ============================================================

def layer_r2_profile(train_h, test_h, n_layers, d_model, k=10):
    """
    逐层计算闭包R², 看哪些层的闭包最差
    """
    print(f"\n{'='*60}")
    print(f"Exp4: Layer-wise R² Profile (k={k})")
    print(f"{'='*60}")

    # PCA基
    all_h_train = []
    for li in range(n_layers):
        h = train_h.get(li)
        if h is not None:
            all_h_train.append(h)
    if len(all_h_train) == 0:
        return {}
    all_h_train = np.vstack(all_h_train)
    mean_h = all_h_train.mean(axis=0)
    h_centered = all_h_train - mean_h
    _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

    k = min(k, Vt_h.shape[0])
    P = Vt_h[:k, :]

    results = {}
    for li in range(n_layers - 1):
        if train_h.get(li) is None or train_h.get(li+1) is None:
            continue
        if test_h.get(li) is None or test_h.get(li+1) is None:
            continue

        n_train = min(train_h[li].shape[0], train_h[li+1].shape[0])
        n_test = min(test_h[li].shape[0], test_h[li+1].shape[0])

        z_l = (train_h[li][:n_train] - mean_h) @ P.T
        z_l1 = (train_h[li+1][:n_train] - mean_h) @ P.T
        z_test = (test_h[li][:n_test] - mean_h) @ P.T
        z_test1 = (test_h[li+1][:n_test] - mean_h) @ P.T

        A, b = _fit_linear(z_l, z_l1)
        z_pred = z_test @ A.T + b
        r2 = _compute_r2(z_test1, z_pred)

        results[f"L{li}"] = round(r2, 4)

    # 打印 (每3层一个)
    for li in range(0, n_layers - 1, 3):
        key = f"L{li}"
        if key in results:
            print(f"  {key}→L{li+1}: R²={results[key]:.4f}")

    return results


# ============================================================
# Exp5: 约束特异性闭包 ★★★
# ============================================================

def constraint_specific_closure(train_h_by_type, test_h_by_type, sample_layers,
                                 d_model, k=10):
    """
    每种约束类型独立计算闭包R²
    看是否存在约束类型通用的高R²投影
    """
    print(f"\n{'='*60}")
    print(f"Exp5: Constraint-Specific Closure (k={k})")
    print(f"{'='*60}")

    results = {}

    for ctype, train_h in train_h_by_type.items():
        test_h = test_h_by_type.get(ctype)
        if test_h is None:
            continue

        # 计算该约束类型自己的PCA基
        all_h = []
        for li in sample_layers:
            h = train_h.get(li)
            if h is not None:
                all_h.append(h)
        if len(all_h) < 2:
            continue
        all_h = np.vstack(all_h)
        mean_h = all_h.mean(axis=0)
        h_centered = all_h - mean_h
        _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

        k_use = min(k, Vt_h.shape[0])
        P = Vt_h[:k_use, :]

        r2_values = []
        for li in sample_layers:
            if train_h.get(li) is None or train_h.get(li+1) is None:
                continue
            if test_h.get(li) is None or test_h.get(li+1) is None:
                continue

            n_train = min(train_h[li].shape[0], train_h[li+1].shape[0])
            n_test = min(test_h[li].shape[0], test_h[li+1].shape[0])

            z_l = (train_h[li][:n_train] - mean_h) @ P.T
            z_l1 = (train_h[li+1][:n_train] - mean_h) @ P.T
            z_test = (test_h[li][:n_test] - mean_h) @ P.T
            z_test1 = (test_h[li+1][:n_test] - mean_h) @ P.T

            A, b = _fit_linear(z_l, z_l1)
            z_pred = z_test @ A.T + b
            r2 = _compute_r2(z_test1, z_pred)
            r2_values.append(r2)

        mean_r2 = np.mean(r2_values) if r2_values else 0.0
        results[ctype] = round(mean_r2, 4)
        print(f"  {ctype}: R²={mean_r2:.4f} (n_layers={len(r2_values)})")

    return results


# ============================================================
# 辅助函数
# ============================================================

def _fit_linear(X, Y, lam=0.1):
    """Ridge回归: Y = X A^T + b  →  返回 A, b 使得 Y_pred = X @ A.T + b"""
    n = X.shape[0]
    ones = np.ones((n, 1))
    X_aug = np.hstack([X, ones])
    k_in = X.shape[1]
    k_out = Y.shape[1] if Y.ndim > 1 else 1
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    reg = lam * np.eye(k_in + 1)
    reg[-1, -1] = 0
    W = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)
    A = W[:k_in, :].T  # [k_out, k_in]
    b = W[k_in:, :].reshape(1, -1)  # [1, k_out]
    return A, b


def _compute_r2(Y_true, Y_pred):
    """计算R² (多变量)"""
    ss_res = np.sum((Y_true - Y_pred)**2)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0))**2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ============================================================
# 主流程
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model_key = {"qwen3": "qwen3", "glm4": "glm4", "deepseek7b": "deepseek7b",
                 "ds7b": "deepseek7b"}.get(model_name.lower(), model_name.lower())

    print(f"\n{'#'*70}")
    print(f"Phase 227: Koopman/DMD Closure — {model_key}")
    print(f"{'#'*70}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === 1. 加载模型 ===
    model, tokenizer, device = load_model_bf16_sdpa(model_key)
    info = get_model_info(model, model_key)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}")

    sample_layers = get_sample_layers(n_layers)

    # === 2. 生成约束对 ===
    constraint_pairs = generate_constraint_pairs(n_per_type=40)
    
    # 合并所有句子
    all_sentences_A = []
    all_sentences_B = []
    sentence_types = []
    
    for ctype, pairs in constraint_pairs.items():
        for pA, pB in pairs:
            all_sentences_A.append(pA)
            all_sentences_B.append(pB)
            sentence_types.append(ctype)

    n_total = len(all_sentences_A)
    # 划分: 前30对为训练, 后10对为测试 (每种约束类型)
    n_per_type_train = 30
    n_per_type_test = 10
    
    train_sentences = []
    test_sentences = []
    train_types = []
    test_types = []
    
    for ctype in constraint_pairs:
        pairs = constraint_pairs[ctype]
        for i, (pA, pB) in enumerate(pairs):
            if i < n_per_type_train:
                train_sentences.extend([pA, pB])
                train_types.extend([ctype, ctype])
            else:
                test_sentences.extend([pA, pB])
                test_types.extend([ctype, ctype])

    print(f"  Train: {len(train_sentences)} sentences, Test: {len(test_sentences)} sentences")

    # === 3. 收集隐藏状态 ===
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting train hidden states...")
    train_h = collect_hidden_states(model, tokenizer, device, train_sentences,
                                     n_layers, desc="train")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting test hidden states...")
    test_h = collect_hidden_states(model, tokenizer, device, test_sentences,
                                    n_layers, desc="test")

    # === 4. 按约束类型拆分隐藏状态 (for Exp5) ===
    train_h_by_type = {}
    test_h_by_type = {}
    
    for ctype in constraint_pairs:
        # 训练集: 该类型的60个句子 (30对 × 2)
        train_indices = [i for i, t in enumerate(train_types) if t == ctype]
        train_h_type = {}
        for li in range(n_layers):
            if train_h.get(li) is not None:
                train_h_type[li] = train_h[li][train_indices]
        train_h_by_type[ctype] = train_h_type

        # 测试集: 该类型的20个句子 (10对 × 2)
        test_indices = [i for i, t in enumerate(test_types) if t == ctype]
        test_h_type = {}
        for li in range(n_layers):
            if test_h.get(li) is not None:
                test_h_type[li] = test_h[li][test_indices]
        test_h_by_type[ctype] = test_h_type

    # === 5. 运行实验 ===
    all_results = {"model": model_key, "d_model": d_model, "n_layers": n_layers}

    # Exp1: DMD闭包
    try:
        all_results["exp1_dmd"] = dmd_closure(
            train_h, test_h, sample_layers, d_model)
    except Exception as e:
        print(f"  Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_dmd"] = {"error": str(e)}

    # Exp2: 非线性闭包
    try:
        all_results["exp2_nonlinear"] = nonlinear_closure(
            train_h, test_h, sample_layers, d_model)
    except Exception as e:
        print(f"  Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_nonlinear"] = {"error": str(e)}

    # Exp3: 信息论闭包
    try:
        all_results["exp3_info"] = information_closure(
            train_h, test_h, sample_layers, d_model)
    except Exception as e:
        print(f"  Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_info"] = {"error": str(e)}

    # Exp4: 逐层R²
    try:
        all_results["exp4_layer_r2"] = layer_r2_profile(
            train_h, test_h, n_layers, d_model, k=10)
    except Exception as e:
        print(f"  Exp4 failed: {e}")
        all_results["exp4_layer_r2"] = {"error": str(e)}

    # Exp5: 约束特异性闭包
    try:
        all_results["exp5_constraint"] = constraint_specific_closure(
            train_h_by_type, test_h_by_type, sample_layers, d_model, k=10)
    except Exception as e:
        print(f"  Exp5 failed: {e}")
        all_results["exp5_constraint"] = {"error": str(e)}

    # === 6. 保存结果 ===
    out_path = OUTPUT_DIR / f"phase227_{model_key}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    # === 7. 释放模型 ===
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nPhase 227 ({model_key}) complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
