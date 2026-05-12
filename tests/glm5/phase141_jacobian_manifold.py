"""
Phase 141: Jacobian Geometry & Language Manifold
=================================================
核心转折：从"统计现象学"进入"局部生成机制"

理论框架：
  - 语言 = 流形上的受限动力学 (constrained dynamics on manifold)
  - hidden state = 流形坐标
  - layers = 运输映射 J_l = ∂h_{l+1}/∂h_l
  - semantic operators = 局部向量场 V_op(h)
  - LM head = 低秩离散读取器

四大实验：
  Exp A: 层Jacobian谱分析 (Priority 1)
    - 在每层注入随机扰动和语义扰动，测量传播
    - 核心数学对象: J_l = ∂h_{l+1}/∂h_l
    - 关键改进: 在同一层注入(而非embedding层)，公平比较

  Exp B: 语言流形内在维数 (Priority 2)
    - 大量句子的hidden states做local PCA
    - 验证 dim(M_language) << d_model

  Exp C: 语义向量场一致性 (Priority 3)
    - 在不同h点计算 V_not(h)
    - 测试: V_not是常向量(平移)还是h依赖(非线性算子)?

  Exp D: 真正的算子交换子 (Priority 4)
    - 计算 A(B(x)) vs B(A(x))
    - 测量 [A,B] = AB - BA (真正的非交换性)

关键方法论改进(针对Phase 140的批评)：
  1. 不再比较embedding层的"自然语义差"与"人工随机扰动"
  2. 在同一中间层注入同范数的语义方向和随机方向
  3. 研究Jacobian而非hidden difference
  4. 使用scope歧义句构造真正的交换子实验

时间：2026-05-12 16:30
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import gc
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

# ============================================================
# 句子集
# ============================================================

# Jacobian测试句(用于Exp A和C)
JACOBIAN_TEST_SENTENCES = [
    "The cat can swim across the river",
    "She has finished her homework",
    "They should leave the building",
    "The bird will fly south this winter",
    "He could answer the question",
    "The dog can bark very loudly",
    "She must study for the exam",
    "They would agree with the plan",
]

# NOT算子对(用于Exp A和C)
NOT_OPERATOR_PAIRS = [
    ("The cat can swim across the river", "The cat can not swim across the river"),
    ("She has finished her homework", "She has not finished her homework"),
    ("They should leave the building", "They should not leave the building"),
    ("The bird will fly south this winter", "The bird will not fly south this winter"),
    ("He could answer the question", "He could not answer the question"),
    ("The dog can bark very loudly", "The dog can not bark very loudly"),
    ("She must study for the exam", "She must not study for the exam"),
    ("They would agree with the plan", "They would not agree with the plan"),
    ("The fish can breathe underwater", "The fish can not breathe underwater"),
    ("He has eaten the apple", "He has not eaten the apple"),
    ("The child will sleep early", "The child will not sleep early"),
    ("She can speak three languages", "She can not speak three languages"),
    ("They may enter the room", "They may not enter the room"),
    ("The plant will grow quickly", "The plant will not grow quickly"),
    ("He should rest after lunch", "He should not rest after lunch"),
]

# PAST算子对(用于Exp A)
PAST_OPERATOR_PAIRS = [
    ("The cat walks home slowly", "The cat walked home slowly"),
    ("She plays the piano well", "She played the piano well"),
    ("He works hard every day", "He worked hard every day"),
    ("They talk loudly in class", "They talked loudly in class"),
    ("The dog jumps over the fence", "The dog jumped over the fence"),
    ("She cooks dinner every night", "She cooked dinner every night"),
    ("He reads books on weekends", "He read books on weekends"),
    ("They walk slowly to school", "They walked slowly to school"),
    ("The bird sings in the morning", "The bird sang in the morning"),
    ("She paints beautiful pictures", "She painted beautiful pictures"),
    ("He drives fast on the highway", "He drove fast on the highway"),
    ("They build houses in the village", "They built houses in the village"),
    ("The cat chases mice around", "The cat chased mice around"),
    ("She writes letters to friends", "She wrote letters to friends"),
    ("He runs fast every morning", "He ran fast every morning"),
]

# ============================================================
# Exp B: 大量句子用于流形维数估计
# ============================================================
MANIFOLD_SENTENCES = [
    # === 声明句 ===
    "The cat sat on the mat", "Dogs love to play in the park",
    "She reads books every evening", "He works at the hospital",
    "The sun rises in the east", "Water flows downhill",
    "Birds fly south in winter", "The train arrives at noon",
    "Children play in the garden", "The teacher explains the lesson",
    "Rivers flow into the sea", "The moon shines at night",
    "Flowers bloom in spring", "The wind blows from the north",
    "Fish swim in the ocean", "The clock ticks on the wall",
    "Students study in the library", "The rain falls softly",
    "Stars twinkle in the sky", "The fire burns brightly",
    "Leaves fall from the trees", "The river runs deep",
    "Snow covers the mountain", "The bell rings at eight",
    "The door opens slowly", "A car drives down the street",
    "The baby cries loudly", "The music plays softly",
    "The book lies on the table", "The horse runs across the field",
    # === 否定句 ===
    "The cat can not swim", "She does not like coffee",
    "He has not finished yet", "They will not come tomorrow",
    "The bird can not fly", "She must not enter",
    "He could not answer", "They should not worry",
    "The dog does not bark", "She will not forget",
    "He can not drive", "They must not stop",
    "The fish can not walk", "She has not decided",
    "He would not agree", "They do not understand",
    "The cat does not sleep", "She can not cook",
    "He should not wait", "They might not come",
    # === 将来时 ===
    "She will travel to Japan", "The project will finish tomorrow",
    "They will build a house", "He will join the team",
    "The sun will rise at six", "We will eat dinner soon",
    "The game will start late", "She will write a letter",
    "He will fix the car", "They will clean the room",
    "The dog will bark", "She will paint the wall",
    "He will teach the class", "They will watch the show",
    "The cat will sleep", "She will cook the meal",
    "He will run the race", "They will sing songs",
    "The bird will fly high", "She will read a book",
    # === 过去时 ===
    "He walked to the store", "They built a sandcastle",
    "She danced all night", "The dog fetched the ball",
    "He drove to the city", "They taught the children",
    "She wrote a poem", "The cat chased the mouse",
    "He fixed the machine", "They watched the movie",
    "She sang a song", "The bird built a nest",
    "He ran the marathon", "They painted the house",
    "She cooked the dinner", "The dog found the bone",
    "He read the book", "They grew the crops",
    "She drew the picture", "The cat caught the fish",
    # === 疑问句 ===
    "What time does the train arrive", "How many students passed the exam",
    "Where is the nearest hospital", "When will the meeting start",
    "Why did she leave early", "Who wrote this book",
    "Which color do you prefer", "How long will it take",
    "Can the dog swim", "Will it rain tomorrow",
    "Has she finished the work", "Should they leave now",
    "Could he answer the question", "Would you like some tea",
    "Is the cat sleeping", "Are they coming home",
    # === 复数 ===
    "The cats sit on the mat", "Dogs run in the park",
    "Sheep graze on the hill", "Children learn quickly",
    "Trees grow tall in the forest", "Rivers flow to the sea",
    "Birds build nests in trees", "Students read many books",
    "Flowers bloom in the garden", "Stars shine in the night",
    "Cars move along the road", "Doors open automatically",
    "Leaves fall in autumn", "Clouds drift across the sky",
    "Horses run in the field", "Fish swim in the lake",
    "Books lie on the shelf", "Ships sail on the ocean",
    # === 长句 ===
    "The scientist who discovered the new element received a prize",
    "She quickly finished her homework before going out to play",
    "The old man walked slowly through the park every morning",
    "They carefully built the house on top of the hill",
    "The little bird sang beautifully in the tall tree",
    "He always reads the newspaper before eating breakfast",
    "The students who studied hard passed the difficult exam",
    "She gently placed the flowers in the glass vase",
    "The large dog chased the small cat around the yard",
    "They never expected to find the treasure buried there",
    # === 情态 ===
    "The cat can catch mice easily", "She must finish the work today",
    "He should study harder for the test", "They might go to the beach",
    "The dog would follow its owner", "She could solve the puzzle",
    "He may leave early tomorrow", "They can speak three languages",
    "The bird will return in spring", "She must be very careful",
    "He should eat more vegetables", "They would enjoy the concert",
    "The cat could climb the tree", "She may join the club",
    # === 被动 ===
    "The book was written by a famous author",
    "The house was built last year",
    "The song was sung beautifully",
    "The meal was cooked by the chef",
    "The car was repaired quickly",
    "The letter was sent yesterday",
    "The picture was painted by a child",
    "The bridge was designed by engineers",
    "The game was won by the home team",
    "The cake was baked this morning",
    # === 条件 ===
    "If it rains we will stay home", "Unless she calls we will go",
    "Although he was tired he continued", "Because she studied she passed",
    "While they waited they talked", "Since he arrived we started",
    "Before she left she called", "After they ate they rested",
    "Whenever it snows they ski", "Whether or not he agrees we proceed",
    # === 更多变体 ===
    "The red car drives fast", "A tall building stands nearby",
    "The small puppy barks loudly", "An old woman walks slowly",
    "The young boy runs quickly", "A large tree provides shade",
    "The cold wind blows hard", "A warm fire burns bright",
    "The dark night falls early", "A bright star shines high",
    "The sweet cake tastes good", "A sour lemon smells fresh",
    "The hard rock feels smooth", "A soft pillow looks comfortable",
    "The heavy door closes slowly", "A light feather floats gently",
    "The clear water runs deep", "A thick fog covers the valley",
    "The sharp knife cuts well", "A round ball rolls far",
]

# ============================================================
# Exp D: 真正的算子交换子
# ============================================================
# 使用scope歧义句: A(B(x)) ≠ B(A(x))
# 经典案例: "not all" vs "all not", "not always" vs "always not"

COMMUTATOR_DATA = {
    "ALL_NOT": {
        "description": "量词-否定交换: 'not all'(部分否定) vs 'all not'(全部否定)",
        "pairs": [
            # (base, NOT(ALL(x)), ALL(NOT(x)))
            # NOT(ALL(x)) = "not all X V" (部分否定: 有些X没V)
            # ALL(NOT(x)) = "all X do not V" (全部否定: 所有X都不V)
            ("students passed the exam",
             "not all students passed the exam",      # NOT∘ALL
             "all students did not pass the exam"),   # ALL∘NOT
            ("birds can fly",
             "not all birds can fly",
             "all birds can not fly"),
            ("cats like water",
             "not all cats like water",
             "all cats do not like water"),
            ("doctors work hard",
             "not all doctors work hard",
             "all doctors do not work hard"),
            ("rich people are happy",
             "not all rich people are happy",
             "all rich people are not happy"),
            ("students like math",
             "not all students like math",
             "all students do not like math"),
            ("animals can swim",
             "not all animals can swim",
             "all animals can not swim"),
            ("flowers bloom in winter",
             "not all flowers bloom in winter",
             "all flowers do not bloom in winter"),
            ("players are tall",
             "not all players are tall",
             "all players are not tall"),
            ("buildings are safe",
             "not all buildings are safe",
             "all buildings are not safe"),
        ],
    },
    "ALWAYS_NOT": {
        "description": "频率-否定交换: 'not always'(有时不) vs 'always not'(总是不)",
        "pairs": [
            # (base, NOT(ALWAYS(x)), ALWAYS(NOT(x)))
            ("the train arrives on time",
             "the train does not always arrive on time",  # NOT∘ALWAYS: 有时不准时
             "the train always does not arrive on time"), # ALWAYS∘NOT: 总是不准时
            ("she eats breakfast",
             "she does not always eat breakfast",
             "she always does not eat breakfast"),
            ("he comes home early",
             "he does not always come home early",
             "he always does not come home early"),
            ("they win the game",
             "they do not always win the game",
             "they always do not win the game"),
            ("the dog barks at night",
             "the dog does not always bark at night",
             "the dog always does not bark at night"),
            ("she reads the newspaper",
             "she does not always read the newspaper",
             "she always does not read the newspaper"),
            ("he drives carefully",
             "he does not always drive carefully",
             "he always does not drive carefully"),
            ("they attend the meeting",
             "they do not always attend the meeting",
             "they always do not attend the meeting"),
        ],
    },
    "SOME_NOT": {
        "description": "存在量词-否定交换: 'not some'(没有) vs 'some not'(有些不)",
        "pairs": [
            # (base, NOT(SOME(x)), SOME(NOT(x)))
            ("some students passed the exam",
             "not some students passed the exam",     # NOT∘SOME: 没有学生通过(=none)
             "some students did not pass the exam"),  # SOME∘NOT: 有些学生没通过
            ("some birds can sing",
             "not some birds can sing",
             "some birds can not sing"),
            ("some flowers are red",
             "not some flowers are red",
             "some flowers are not red"),
            ("some people like coffee",
             "not some people like coffee",
             "some people do not like coffee"),
            ("some books are interesting",
             "not some books are interesting",
             "some books are not interesting"),
        ],
    },
}

# ============================================================
# 工具函数
# ============================================================

def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_forward(model, input_ids, attention_mask, output_hidden_states=True):
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=output_hidden_states)
    except Exception as e:
        print(f"    [WARN] Forward failed: {e}")
        return None


def get_hidden_states_for_sentence(model, tokenizer, device, sentence):
    """获取句子在各层的hidden states, 返回 list of [1, seq, d_model]"""
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attn_mask = torch.ones(1, len(ids), device=device, dtype=torch.long)
    out = safe_forward(model, input_ids, attn_mask)
    if out is None:
        return None
    return out.hidden_states


def get_last_token_hidden(hs_all_layers, layer_idx):
    """从hidden states中提取指定层last token的向量"""
    if hs_all_layers is None or layer_idx >= len(hs_all_layers):
        return None
    return hs_all_layers[layer_idx][0, -1, :].float().cpu().numpy()


# ============================================================
# Exp A: 层Jacobian谱分析
# ============================================================

def expA_jacobian_spectral(model, tokenizer, device, model_info, model_name):
    """
    Exp A: 层Jacobian谱分析
    
    核心改进：在同一中间层注入同范数的语义方向和随机方向
    公平比较: 两者都是R^d_model中的向量，在同一层注入
    
    方法：
    1. 对base和NOT(x)做forward，得到各层hidden states
    2. 在层l，计算语义方向 v_sem = h_l(NOT(x)) - h_l(x), 归一化
    3. 在层l注入 v_sem * eps，测量层l+1的变化 → J_l @ v_sem (近似)
    4. 在层l注入 v_rand * eps (同范数随机方向)，测量层l+1的变化 → J_l @ v_rand
    5. 用k个随机方向估计J_l的奇异值谱
    6. 比较: ||J_l @ v_sem|| vs ||J_l @ v_rand||
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    layers = get_layers(model)
    
    # 采样层 (均匀+首尾)
    n_sample = min(8, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    n_probes_random = 30  # 随机探测方向数
    eps = 1e-4  # 有限差分步长
    
    results = {
        "model": model_name,
        "sample_layers": sample_layers,
        "n_probes_random": n_probes_random,
        "eps": eps,
        "jacobian_spectral": {},  # {layer: {sv_estimates, ...}}
        "semantic_vs_random": {},  # {layer: {ratio, ...}}
    }
    
    # === Part 1: 对每个测试句计算Jacobian JVP ===
    n_sentences = min(5, len(JACOBIAN_TEST_SENTENCES))
    
    for sent_idx in range(n_sentences):
        base_sent = JACOBIAN_TEST_SENTENCES[sent_idx]
        not_sent = NOT_OPERATOR_PAIRS[sent_idx][1]
        past_sent = PAST_OPERATOR_PAIRS[sent_idx][1]
        
        print(f"\n  句子 {sent_idx+1}/{n_sentences}: '{base_sent[:40]}...'")
        
        # 获取base, NOT(x), PAST(x)的hidden states
        hs_base = get_hidden_states_for_sentence(model, tokenizer, input_device, base_sent)
        hs_not = get_hidden_states_for_sentence(model, tokenizer, input_device, not_sent)
        hs_past = get_hidden_states_for_sentence(model, tokenizer, input_device, past_sent)
        
        if hs_base is None or hs_not is None or hs_past is None:
            print(f"    [WARN] Forward failed for sentence {sent_idx}")
            continue
        
        for layer_idx in sample_layers:
            layer_key = f"L{layer_idx}"
            if layer_idx + 1 >= len(hs_base):
                continue
            
            # === 获取语义方向 ===
            h_base = get_last_token_hidden(hs_base, layer_idx)
            h_not = get_last_token_hidden(hs_not, layer_idx)
            h_past = get_last_token_hidden(hs_past, layer_idx)
            h_base_next = get_last_token_hidden(hs_base, layer_idx + 1)
            
            if h_base is None or h_not is None or h_past is None or h_base_next is None:
                continue
            
            # 语义方向 (归一化)
            v_sem_not = h_not - h_base
            v_sem_not_norm = np.linalg.norm(v_sem_not)
            if v_sem_not_norm < 1e-10:
                continue
            v_sem_not_unit = v_sem_not / v_sem_not_norm
            
            v_sem_past = h_past - h_base
            v_sem_past_norm = np.linalg.norm(v_sem_past)
            if v_sem_past_norm < 1e-10:
                continue
            v_sem_past_unit = v_sem_past / v_sem_past_norm
            
            # === 计算随机JVPs ===
            torch.manual_seed(42 + sent_idx * 100 + layer_idx)
            jvp_random = []  # 存储 J_l @ v_i 的结果
            jvp_random_norms = []
            
            for probe_idx in range(n_probes_random):
                # 随机方向
                v_rand = np.random.randn(d_model)
                v_rand = v_rand / np.linalg.norm(v_rand)  # 单位向量
                
                # 注入扰动并测量 (传入baseline h_next)
                delta_next = _inject_and_measure(
                    model, tokenizer, input_device, layers, 
                    base_sent, layer_idx, v_rand, eps, h_base_next
                )
                
                if delta_next is not None:
                    jvp_random.append(delta_next / eps)  # J_l @ v_rand
                    jvp_random_norms.append(np.linalg.norm(delta_next / eps))
            
            # === 计算语义JVPs ===
            # NOT方向
            delta_not = _inject_and_measure(
                model, tokenizer, input_device, layers,
                base_sent, layer_idx, v_sem_not_unit, eps, h_base_next
            )
            jvp_not_norm = float(np.linalg.norm(delta_not / eps)) if delta_not is not None else None
            
            # PAST方向
            delta_past = _inject_and_measure(
                model, tokenizer, input_device, layers,
                base_sent, layer_idx, v_sem_past_unit, eps, h_base_next
            )
            jvp_past_norm = float(np.linalg.norm(delta_past / eps)) if delta_past is not None else None
            
            # === 汇总 ===
            if layer_key not in results["jacobian_spectral"]:
                results["jacobian_spectral"][layer_key] = {
                    "jvp_random_norms_all": [],
                    "jvp_semantic_not_norms_all": [],
                    "jvp_semantic_past_norms_all": [],
                }
            
            if jvp_random_norms:
                results["jacobian_spectral"][layer_key]["jvp_random_norms_all"].extend(jvp_random_norms)
                
                # 从JVP矩阵估计奇异值谱
                if len(jvp_random) >= 5:
                    jvp_matrix = np.array(jvp_random)  # [n_probes, d_model]
                    # SVD: 近似J_l的奇异值
                    try:
                        U_jvp, S_jvp, Vt_jvp = np.linalg.svd(jvp_matrix, full_matrices=False)
                        results["jacobian_spectral"][layer_key][f"sv_estimate_sent{sent_idx}"] = S_jvp[:20].tolist()
                    except Exception:
                        pass
            
            if delta_not is not None:
                jvp_not_n = float(np.linalg.norm(delta_not / eps))
                results["jacobian_spectral"][layer_key]["jvp_semantic_not_norms_all"].append(jvp_not_n)
            if delta_past is not None:
                jvp_past_n = float(np.linalg.norm(delta_past / eps))
                results["jacobian_spectral"][layer_key]["jvp_semantic_past_norms_all"].append(jvp_past_n)
    
    # === Part 2: 聚合语义vs随机比较 ===
    print("\n  === Jacobian谱分析汇总 ===")
    for layer_key in sorted(results["jacobian_spectral"].keys(), key=lambda x: int(x[1:])):
        data = results["jacobian_spectral"][layer_key]
        
        rand_norms = data.get("jvp_random_norms_all", [])
        not_norms = data.get("jvp_semantic_not_norms_all", [])
        past_norms = data.get("jvp_semantic_past_norms_all", [])
        
        if not rand_norms:
            continue
        
        rand_mean = float(np.mean(rand_norms))
        rand_std = float(np.std(rand_norms))
        
        result_layer = {
            "random_jvp_mean": rand_mean,
            "random_jvp_std": rand_std,
            "n_random_probes": len(rand_norms),
        }
        
        if not_norms:
            not_mean = float(np.mean(not_norms))
            result_layer["not_jvp_mean"] = not_mean
            result_layer["not_vs_random_ratio"] = not_mean / rand_mean if rand_mean > 1e-10 else None
        
        if past_norms:
            past_mean = float(np.mean(past_norms))
            result_layer["past_jvp_mean"] = past_mean
            result_layer["past_vs_random_ratio"] = past_mean / rand_mean if rand_mean > 1e-10 else None
        
        # 合并奇异值估计
        sv_keys = [k for k in data if k.startswith("sv_estimate_")]
        if sv_keys:
            all_sv = []
            for k in sv_keys:
                all_sv.extend(data[k])
            # 取中位数作为稳健估计
            sv_array = np.array(all_sv).reshape(len(sv_keys), -1)
            result_layer["sv_median_top20"] = np.median(sv_array, axis=0).tolist()
        
        results["semantic_vs_random"][layer_key] = result_layer
        
        # 打印
        not_ratio = result_layer.get("not_vs_random_ratio", "N/A")
        not_ratio_str = f"{not_ratio:.2f}x" if isinstance(not_ratio, float) else not_ratio
        past_ratio = result_layer.get("past_vs_random_ratio", "N/A")
        past_ratio_str = f"{past_ratio:.2f}x" if isinstance(past_ratio, float) else past_ratio
        print(f"    {layer_key}: random_JVP={rand_mean:.2f}±{rand_std:.2f}, "
              f"NOT/random={not_ratio_str}, PAST/random={past_ratio_str}")
    
    return results


def _inject_and_measure(model, tokenizer, device, layers, base_sentence, 
                         layer_idx, direction_np, eps, h_next_baseline_np):
    """
    在指定层注入扰动方向，测量下一层的变化
    
    使用 register_forward_pre_hook 在层l的输入注入扰动
    使用 register_forward_hook 在层l的输出捕获结果
    
    关键: baseline必须来自无扰动的forward pass，不能来自同一次forward!
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        layers: 层列表
        base_sentence: 基础句子
        layer_idx: 注入层索引
        direction_np: 注入方向 (numpy [d_model], 已归一化)
        eps: 扰动强度
        h_next_baseline_np: 无扰动时层l+1的输出 (numpy [d_model])
    
    Returns:
        delta_next: h_{l+1}' - h_{l+1} (numpy [d_model])
    """
    ids = tokenizer.encode(base_sentence, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attn_mask = torch.ones(1, len(ids), device=device, dtype=torch.long)
    
    # 创建扰动张量 (只扰动last token位置)
    direction_t = torch.tensor(direction_np, dtype=torch.float32)
    
    # 前向传播，注入扰动
    captured_output = {}
    
    def pre_hook_fn(module, args):
        """在层输入注入扰动"""
        hidden_states = args[0]
        perturbed = hidden_states.clone()
        # 只在last token位置注入
        perturbed[:, -1, :] += eps * direction_t.to(perturbed.device, perturbed.dtype)
        return (perturbed,) + args[1:]
    
    def post_hook_fn(module, input, output):
        """捕获层输出"""
        if isinstance(output, tuple):
            captured_output['h_next'] = output[0].detach().float().cpu()
        else:
            captured_output['h_next'] = output.detach().float().cpu()
    
    # 注册hooks
    pre_hook = layers[layer_idx].register_forward_pre_hook(pre_hook_fn)
    post_hook = layers[layer_idx].register_forward_hook(post_hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=False)
    except Exception as e:
        print(f"      [WARN] Inject forward failed: {e}")
        pre_hook.remove()
        post_hook.remove()
        return None
    finally:
        pre_hook.remove()
        post_hook.remove()
    
    if 'h_next' not in captured_output:
        return None
    
    h_next_perturbed = captured_output['h_next'][0, -1, :].numpy()  # [d_model]
    
    # 用外部传入的baseline (来自无扰动forward)
    delta_next = h_next_perturbed - h_next_baseline_np
    
    return delta_next


# ============================================================
# Exp B: 语言流形内在维数
# ============================================================

def expB_manifold_dimension(model, tokenizer, device, model_info, model_name):
    """
    Exp B: 语言流形内在维数
    
    验证: dim(M_language) << d_model
    
    方法:
    1. 对200+句子做forward，收集各层hidden states
    2. 对每层的hidden states做PCA
    3. 计算参与率 (participation ratio): PR = (Σλ_i)^2 / Σ(λ_i^2)
    4. PR就是内在维数的估计
    
    PR的直觉：如果只有k个方向有显著方差，则PR ≈ k
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    
    n_sentences = min(200, len(MANIFOLD_SENTENCES))
    print(f"  使用 {n_sentences} 个句子估计流形维数")
    
    # 采样层
    n_sample = min(10, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    # 收集hidden states
    print(f"  收集hidden states...")
    layer_hidden_states = {l: [] for l in sample_layers}  # {layer: [n_sentences, d_model]}
    
    for sent_idx, sentence in enumerate(MANIFOLD_SENTENCES[:n_sentences]):
        if sent_idx % 50 == 0:
            print(f"    进度: {sent_idx}/{n_sentences}")
        
        hs = get_hidden_states_for_sentence(model, tokenizer, input_device, sentence)
        if hs is None:
            continue
        
        for l in sample_layers:
            if l < len(hs):
                h = hs[l][0, -1, :].float().cpu().numpy()
                layer_hidden_states[l].append(h)
    
    # PCA和参与率计算
    print(f"\n  === 流形维数估计 ===")
    results = {
        "model": model_name,
        "n_sentences": n_sentences,
        "sample_layers": sample_layers,
        "layer_analysis": {},
    }
    
    for l in sample_layers:
        H = np.array(layer_hidden_states[l])  # [n, d_model]
        n_actual = H.shape[0]
        
        if n_actual < 10:
            print(f"    L{l}: 数据不足 ({n_actual} sentences)")
            continue
        
        # 中心化
        H_centered = H - H.mean(axis=0, keepdims=True)
        
        # 协方差矩阵 (用SVD避免显式计算)
        # H_centered: [n, d], 协方差 = H^T H / (n-1)
        # SVD of H: H = U S Vt → 协方差的特征值 = S^2 / (n-1)
        
        t0 = time.time()
        try:
            # 使用经济SVD
            U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        except Exception as e:
            print(f"    L{l}: SVD失败: {e}")
            continue
        
        svd_time = time.time() - t0
        
        # 特征值 (方差)
        eigenvalues = S**2 / (n_actual - 1)
        total_variance = np.sum(eigenvalues)
        
        if total_variance < 1e-10:
            continue
        
        # 参与率: PR = (Σλ)^2 / Σ(λ^2)
        participation_ratio = total_variance**2 / np.sum(eigenvalues**2)
        
        # 累积方差比
        cumulative_var = np.cumsum(eigenvalues) / total_variance
        
        # 找到90%/95%/99%方差所需的维度
        dim_90 = int(np.searchsorted(cumulative_var, 0.90)) + 1
        dim_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
        dim_99 = int(np.searchsorted(cumulative_var, 0.99)) + 1
        
        # 有效维度比
        pr_pct = participation_ratio / d_model * 100
        
        layer_result = {
            "n_sentences": n_actual,
            "participation_ratio": float(participation_ratio),
            "pr_pct_of_d": float(pr_pct),
            "dim_90pct_variance": dim_90,
            "dim_95pct_variance": dim_95,
            "dim_99pct_variance": dim_99,
            "dim_90_pct": float(dim_90 / d_model * 100),
            "dim_95_pct": float(dim_95 / d_model * 100),
            "dim_99_pct": float(dim_99 / d_model * 100),
            "total_variance": float(total_variance),
            "top_eigenvalues": eigenvalues[:20].tolist(),
            "svd_time": round(svd_time, 2),
        }
        
        results["layer_analysis"][f"L{l}"] = layer_result
        
        print(f"    L{l}: PR={participation_ratio:.0f}/{d_model} ({pr_pct:.1f}%), "
              f"90%var={dim_90} ({dim_90/d_model*100:.1f}%), "
              f"95%var={dim_95} ({dim_95/d_model*100:.1f}%), "
              f"SVD={svd_time:.1f}s")
    
    # 全局汇总
    pr_values = [v["participation_ratio"] for v in results["layer_analysis"].values()]
    if pr_values:
        results["summary"] = {
            "mean_pr": float(np.mean(pr_values)),
            "min_pr": float(np.min(pr_values)),
            "max_pr": float(np.max(pr_values)),
            "mean_pr_pct": float(np.mean(pr_values) / d_model * 100),
        }
        print(f"\n  汇总: 平均PR = {np.mean(pr_values):.0f}/{d_model} "
              f"({np.mean(pr_values)/d_model*100:.1f}%)")
    
    return results


# ============================================================
# Exp C: 语义向量场一致性
# ============================================================

def expC_semantic_vector_field(model, tokenizer, device, model_info, model_name):
    """
    Exp C: 语义向量场一致性
    
    核心问题: V_not(h) 是常数向量(平移) 还是 h-依赖(非线性算子)?
    
    方法:
    1. 在N个不同的base句子上计算 V_not(h_i) = h(not(x_i)) - h(x_i)
    2. 测量 V_not(h_i) 之间的cosine similarity
    3. 如果高: NOT ≈ 平移 (translation)
    4. 如果低: NOT ≈ 非线性算子 (context-dependent)
    
    进一步:
    - 测量 ||V_not(h)|| 的方差: 如果恒定→平移，如果变化→非线性
    - 测量V_not(h)与h的相关性: 如果相关→曲率存在
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    
    n_pairs = min(15, len(NOT_OPERATOR_PAIRS))
    
    # 采样层
    n_sample = min(8, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    print(f"  使用 {n_pairs} 个NOT句对, 采样 {len(sample_layers)} 层")
    
    # 收集各层的V_not
    layer_v_not = {l: [] for l in sample_layers}  # {layer: list of V_not vectors}
    layer_h_base = {l: [] for l in sample_layers}  # {layer: list of h_base vectors}
    
    for pair_idx, (base_sent, not_sent) in enumerate(NOT_OPERATOR_PAIRS[:n_pairs]):
        if pair_idx % 5 == 0:
            print(f"    进度: {pair_idx}/{n_pairs}")
        
        hs_base = get_hidden_states_for_sentence(model, tokenizer, input_device, base_sent)
        hs_not = get_hidden_states_for_sentence(model, tokenizer, input_device, not_sent)
        
        if hs_base is None or hs_not is None:
            continue
        
        for l in sample_layers:
            if l >= len(hs_base) or l >= len(hs_not):
                continue
            
            h_b = hs_base[l][0, -1, :].float().cpu().numpy()
            h_n = hs_not[l][0, -1, :].float().cpu().numpy()
            v_not = h_n - h_b
            
            layer_v_not[l].append(v_not)
            layer_h_base[l].append(h_b)
    
    # 分析
    print(f"\n  === 语义向量场一致性 ===")
    results = {
        "model": model_name,
        "n_pairs": n_pairs,
        "sample_layers": sample_layers,
        "vector_field_analysis": {},
    }
    
    for l in sample_layers:
        v_nots = layer_v_not[l]
        h_bases = layer_h_base[l]
        
        if len(v_nots) < 3:
            continue
        
        V = np.array(v_nots)  # [n, d_model]
        H = np.array(h_bases)  # [n, d_model]
        
        # 1. V_not之间的cosine similarity
        mean_v = np.mean(V, axis=0)
        mean_v_norm = np.linalg.norm(mean_v)
        
        if mean_v_norm < 1e-10:
            continue
        
        cosines = []
        for v in V:
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                cosines.append(float(np.dot(v, mean_v) / (v_norm * mean_v_norm)))
        
        direction_consistency = float(np.mean(cosines)) if cosines else 0.0
        
        # 2. V_not的范数变化
        v_norms = [np.linalg.norm(v) for v in V]
        norm_mean = float(np.mean(v_norms))
        norm_std = float(np.std(v_norms))
        norm_cv = norm_std / norm_mean if norm_mean > 1e-10 else float('inf')  # 变异系数
        
        # 3. V_not与h_base的相关性 (曲率指标)
        # 如果 V_not 与 h_base 相关 → 流形有曲率
        # 测量: V_not的各分量与h_base各分量的相关系数的平均
        correlation_with_h = 0.0
        n_dims_test = min(100, d_model)  # 测试100个维度
        dim_correlations = []
        for d in range(n_dims_test):
            v_d = V[:, d]
            h_d = H[:, d]
            if np.std(v_d) > 1e-10 and np.std(h_d) > 1e-10:
                corr = np.corrcoef(v_d, h_d)[0, 1]
                if not np.isnan(corr):
                    dim_correlations.append(abs(corr))
        
        if dim_correlations:
            correlation_with_h = float(np.mean(dim_correlations))
        
        # 4. 两两cosine similarity的分布
        pairwise_cosines = []
        for i in range(len(V)):
            for j in range(i+1, len(V)):
                vi_norm = np.linalg.norm(V[i])
                vj_norm = np.linalg.norm(V[j])
                if vi_norm > 1e-10 and vj_norm > 1e-10:
                    pairwise_cosines.append(float(np.dot(V[i], V[j]) / (vi_norm * vj_norm)))
        
        pairwise_cos_mean = float(np.mean(pairwise_cosines)) if pairwise_cosines else 0.0
        pairwise_cos_std = float(np.std(pairwise_cosines)) if pairwise_cosines else 0.0
        
        # 5. 判断: 平移 vs 非线性
        # 高一致性(>0.7) + 低norm变异(<0.3) → 平移
        # 低一致性(<0.4) 或 高norm变异(>0.5) → 非线性
        is_translation = direction_consistency > 0.7 and norm_cv < 0.3
        operator_type = "translation" if is_translation else "nonlinear_operator"
        
        layer_result = {
            "direction_consistency": direction_consistency,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "norm_cv": norm_cv,
            "correlation_with_h": correlation_with_h,
            "pairwise_cosine_mean": pairwise_cos_mean,
            "pairwise_cosine_std": pairwise_cos_std,
            "operator_type": operator_type,
            "n_pairs": len(V),
        }
        
        results["vector_field_analysis"][f"L{l}"] = layer_result
        
        print(f"    L{l}: consistency={direction_consistency:.3f}, "
              f"norm={norm_mean:.2f}±{norm_std:.2f} (CV={norm_cv:.2f}), "
              f"h_corr={correlation_with_h:.3f}, "
              f"pairwise_cos={pairwise_cos_mean:.3f}±{pairwise_cos_std:.3f}, "
              f"type={operator_type}")
    
    # 全局汇总
    consistencies = [v["direction_consistency"] for v in results["vector_field_analysis"].values()]
    h_corrs = [v["correlation_with_h"] for v in results["vector_field_analysis"].values()]
    
    if consistencies:
        results["summary"] = {
            "mean_consistency": float(np.mean(consistencies)),
            "mean_h_correlation": float(np.mean(h_corrs)),
            "global_operator_type": "mostly_translation" if np.mean(consistencies) > 0.7 else "mostly_nonlinear",
        }
        print(f"\n  汇总: 平均一致性={np.mean(consistencies):.3f}, "
              f"平均h相关性={np.mean(h_corrs):.3f}")
    
    return results


# ============================================================
# Exp D: 真正的算子交换子
# ============================================================

def expD_operator_commutator(model, tokenizer, device, model_info, model_name):
    """
    Exp D: 真正的算子交换子
    
    核心改进: 不再测干涉项 I = Δh(AB) - Δh(A) - Δh(B)
    而是测真正的交换子 [A,B] = AB - BA = h(A(B(x))) - h(B(A(x)))
    
    使用scope歧义句:
    - NOT∘ALL: "not all students passed" (部分否定)
    - ALL∘NOT: "all students did not pass" (全部否定)
    
    这两个句子语义完全不同! 这是真正的非交换性。
    """
    n_layers = model_info.n_layers
    input_device = get_input_device(model)
    
    # 采样层
    n_sample = min(8, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    results = {
        "model": model_name,
        "sample_layers": sample_layers,
        "commutator_analysis": {},
    }
    
    for comp_name, comp_data in COMMUTATOR_DATA.items():
        print(f"\n  交换子类型: {comp_name} — {comp_data['description']}")
        
        layer_commutator = {l: {"ab_ba_norms": [], "ab_norms": [], "ba_norms": [],
                                 "ab_ba_cosines": []} for l in sample_layers}
        
        for pair_idx, triple in enumerate(comp_data["pairs"]):
            if len(triple) == 3:
                base, ab_sent, ba_sent = triple  # AB(x), BA(x)
            else:
                continue
            
            print(f"    对 {pair_idx+1}: AB='{ab_sent[:40]}...' BA='{ba_sent[:40]}...'")
            
            # 获取hidden states
            hs_base = get_hidden_states_for_sentence(model, tokenizer, input_device, base)
            hs_ab = get_hidden_states_for_sentence(model, tokenizer, input_device, ab_sent)
            hs_ba = get_hidden_states_for_sentence(model, tokenizer, input_device, ba_sent)
            
            if hs_base is None or hs_ab is None or hs_ba is None:
                continue
            
            for l in sample_layers:
                if l >= len(hs_base) or l >= len(hs_ab) or l >= len(hs_ba):
                    continue
                
                h_base = hs_base[l][0, -1, :].float().cpu().numpy()
                h_ab = hs_ab[l][0, -1, :].float().cpu().numpy()
                h_ba = hs_ba[l][0, -1, :].float().cpu().numpy()
                
                # 交换子 [A,B] = AB(x) - BA(x)
                commutator = h_ab - h_ba
                comm_norm = np.linalg.norm(commutator)
                
                # 单独算子响应
                ab_response = h_ab - h_base
                ba_response = h_ba - h_base
                ab_norm = np.linalg.norm(ab_response)
                ba_norm = np.linalg.norm(ba_response)
                
                # 交换子范数与算子响应的比较
                avg_response_norm = (ab_norm + ba_norm) / 2
                
                # AB和BA的cosine (相似度)
                if ab_norm > 1e-10 and ba_norm > 1e-10:
                    ab_ba_cos = float(np.dot(ab_response, ba_response) / (ab_norm * ba_norm))
                else:
                    ab_ba_cos = 0.0
                
                layer_commutator[l]["ab_ba_norms"].append(comm_norm)
                layer_commutator[l]["ab_norms"].append(ab_norm)
                layer_commutator[l]["ba_norms"].append(ba_norm)
                layer_commutator[l]["ab_ba_cosines"].append(ab_ba_cos)
        
        # 聚合
        comm_results = {}
        for l in sample_layers:
            data = layer_commutator[l]
            if not data["ab_ba_norms"]:
                continue
            
            layer_key = f"L{l}"
            comm_mean = float(np.mean(data["ab_ba_norms"]))
            ab_mean = float(np.mean(data["ab_norms"]))
            ba_mean = float(np.mean(data["ba_norms"]))
            avg_resp = (ab_mean + ba_mean) / 2
            relative_comm = comm_mean / avg_resp if avg_resp > 1e-10 else None
            cos_mean = float(np.mean(data["ab_ba_cosines"]))
            
            comm_results[layer_key] = {
                "commutator_norm_mean": comm_mean,
                "AB_response_norm_mean": ab_mean,
                "BA_response_norm_mean": ba_mean,
                "relative_commutator": relative_comm,
                "AB_BA_cosine_mean": cos_mean,
                "n_pairs": len(data["ab_ba_norms"]),
                "is_noncommutative": relative_comm is not None and relative_comm > 0.1,
            }
            
            rel_str = f"{relative_comm:.3f}" if relative_comm else "N/A"
            noncomm = "YES" if relative_comm and relative_comm > 0.1 else "no"
            print(f"      {layer_key}: [A,B]={comm_mean:.2f}, "
                  f"relative={rel_str}, "
                  f"cos(AB,BA)={cos_mean:.3f}, "
                  f"noncommutative={noncomm}")
        
        results["commutator_analysis"][comp_name] = comm_results
    
    # 全局汇总
    all_relative_comms = []
    for comp_name, comp_results in results["commutator_analysis"].items():
        for layer_key, data in comp_results.items():
            if data.get("relative_commutator") is not None:
                all_relative_comms.append(data["relative_commutator"])
    
    if all_relative_comms:
        results["summary"] = {
            "mean_relative_commutator": float(np.mean(all_relative_comms)),
            "max_relative_commutator": float(np.max(all_relative_comms)),
            "n_noncommutative": sum(1 for rc in all_relative_comms if rc > 0.1),
            "n_total": len(all_relative_comms),
        }
        print(f"\n  汇总: 平均相对交换子={np.mean(all_relative_comms):.3f}, "
              f"非交换比例={sum(1 for rc in all_relative_comms if rc > 0.1)}/{len(all_relative_comms)}")
    
    return results


# ============================================================
# 主函数
# ============================================================

def run_phase141(model_name: str):
    """运行Phase 141所有实验"""
    print(f"\n{'='*60}")
    print(f"Phase 141: Jacobian Geometry & Language Manifold — {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"\n模型信息: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    all_results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # === Exp A: Jacobian谱分析 (Priority 1) ===
    print(f"\n{'='*40}")
    print("Exp A: 层Jacobian谱分析")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expA"] = expA_jacobian_spectral(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp A failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expA"] = {"error": str(e)}
    print(f"Exp A 完成: {time.time()-t0:.1f}s")
    
    # 释放GPU内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Exp B: 流形维数 (Priority 2) ===
    print(f"\n{'='*40}")
    print("Exp B: 语言流形内在维数")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expB"] = expB_manifold_dimension(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp B failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expB"] = {"error": str(e)}
    print(f"Exp B 完成: {time.time()-t0:.1f}s")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Exp C: 语义向量场 (Priority 3) ===
    print(f"\n{'='*40}")
    print("Exp C: 语义向量场一致性")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expC"] = expC_semantic_vector_field(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp C failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expC"] = {"error": str(e)}
    print(f"Exp C 完成: {time.time()-t0:.1f}s")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Exp D: 算子交换子 (Priority 4) ===
    print(f"\n{'='*40}")
    print("Exp D: 真正的算子交换子")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expD"] = expD_operator_commutator(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp D failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expD"] = {"error": str(e)}
    print(f"Exp D 完成: {time.time()-t0:.1f}s")
    
    # 释放模型
    release_model(model)
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase141_jacobian_manifold.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    results = run_phase141(model_name)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"tests/glm5_temp/phase141_{model_name}_jacobian_manifold_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {filename}")
