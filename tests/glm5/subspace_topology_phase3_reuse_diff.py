"""
Phase 57: 概念复用与差异化的神经元编码机制
============================================
核心问题: 
  - "苹果"如何复用"水果"的特征? 哪些维度共享, 哪些维度独特?
  - "翻译英文"vs"翻译法文"如何复用翻译机制? 哪些维度共享?
  - 逻辑功能(AND/OR)如何复用逻辑基础设施?

实验设计:
  Part 1: 概念层级对 (apple/fruit, dog/animal, red/color, Paris/city)
    - 30+句子/概念, 提取目标词位置的激活
    - PCA提取各概念的子空间, 比较共享/独特子空间
    - 逐层分析: 哪层开始出现复用? 哪层保持差异化?

  Part 2: 任务复用对 (translate_en/translate_fr, sum/subtract)
    - 相同输入不同任务指令, 提取任务位置激活
    - 找共享"任务基础设施"vs独特"任务目标"子空间

  Part 3: 逻辑功能对 (and/or, if/therefore, not/yes)
    - 提取逻辑词位置的激活
    - 找共享"逻辑基础设施"vs独特"逻辑操作"子空间

  Part 4: 跨概念复用骨干
    - 所有概念对的共享维度是否是同一组神经元?
    - 复用骨干的维度和稳定性

跨模型: Qwen3, GLM4, DS7B (依次运行)
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 项目路径
PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import (
    load_model, get_layers, get_model_info, get_layer_weights,
    get_W_U, release_model, safe_decode, MODEL_CONFIGS
)

# ===== 时间日志 =====
def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ===== 概念句子模板 =====
# 每个概念30+句子, 多样化上下文
CONCEPT_TEMPLATES = {
    # Part 1: 概念层级对
    "apple": [
        "I ate a fresh {w} this morning",
        "The {w} was sweet and juicy",
        "She picked a red {w} from the tree",
        "My favorite {w} is the green one",
        "The {w} fell on the ground",
        "He bought three {w}s at the market",
        "That {w} looks delicious",
        "The {w} tree is blooming",
        "I prefer a crisp {w} over a soft one",
        "The {w} has seeds inside",
        "She sliced the {w} for the salad",
        "A rotten {w} was in the basket",
        "The {w} pie smells amazing",
        "He took a bite of the {w}",
        "This {w} is organic",
        "The {w} juice was refreshing",
        "She painted a still life of an {w}",
        "The {w} orchard is beautiful in autumn",
        "I need to buy some {w}s",
        "The {w} skin was shiny",
        "She offered me an {w}",
        "The {w} tasted sour",
        "A worm was in the {w}",
        "The {w} cider is warm",
        "He grew an {w} from seed",
        "The {w} harvest was plentiful",
        "She made {w} sauce",
        "The {w} is ripe and ready",
        "I found a wild {w} in the forest",
        "The {w} season has begun",
    ],
    "fruit": [
        "I ate a fresh {w} this morning",
        "The {w} was sweet and juicy",
        "She picked some {w} from the tree",
        "My favorite {w} is the tropical kind",
        "The {w} fell on the ground",
        "He bought fresh {w} at the market",
        "That {w} looks delicious",
        "The {w} trees are blooming",
        "I prefer seasonal {w} over imported",
        "The {w} has seeds inside",
        "She sliced the {w} for the salad",
        "A rotten {w} was in the basket",
        "The {w} salad smells amazing",
        "He took a piece of {w}",
        "This {w} is organic",
        "The {w} juice was refreshing",
        "She painted a still life of {w}",
        "The {w} orchard is beautiful in autumn",
        "I need to buy some {w}",
        "The {w} skin was shiny",
        "She offered me some {w}",
        "The {w} tasted sour",
        "A worm was in the {w}",
        "The {w} punch is warm",
        "He grew {w} from seed",
        "The {w} harvest was plentiful",
        "She made {w} preserves",
        "The {w} is ripe and ready",
        "I found wild {w} in the forest",
        "The {w} season has begun",
    ],
    "dog": [
        "The {w} barked loudly at night",
        "She adopted a rescue {w} yesterday",
        "My {w} loves to play fetch",
        "The {w} chased the ball across the yard",
        "A friendly {w} approached us",
        "He trained his {w} to sit",
        "The {w} wagged its tail happily",
        "I saw a {w} running in the park",
        "The {w} needs a walk",
        "She fed the {w} some treats",
        "The {w} slept on the couch",
        "A stray {w} wandered the streets",
        "The {w} howled at the moon",
        "He pet the {w} gently",
        "The {w} is very loyal",
        "I heard the {w} barking outside",
        "The {w} dug a hole in the garden",
        "She bought a new {w} collar",
        "The {w} jumped over the fence",
        "A small {w} sat by the door",
        "The {w} licked my hand",
        "He took the {w} to the vet",
        "The {w} growled at strangers",
        "I love my {w} very much",
        "The {w} fetched the stick",
        "She named her {w} Max",
        "The {w} has brown fur",
        "A big {w} guarded the house",
        "The {w} panted in the heat",
        "He walked the {w} every morning",
    ],
    "animal": [
        "The {w} barked loudly at night",
        "She adopted a rescue {w} yesterday",
        "My {w} loves to play in the yard",
        "The {w} chased its prey across the field",
        "A friendly {w} approached us",
        "He studied the {w} in its habitat",
        "The {w} moved cautiously through the forest",
        "I saw an {w} running in the wild",
        "The {w} needs food and shelter",
        "She observed the {w} carefully",
        "The {w} slept in the cave",
        "A wild {w} wandered the plains",
        "The {w} howled at the moon",
        "He tracked the {w} through the snow",
        "The {w} is very adaptable",
        "I heard the {w} calling outside",
        "The {w} dug a burrow in the ground",
        "She researched the {w} behavior",
        "The {w} jumped over the stream",
        "A small {w} hid by the rocks",
        "The {w} sensed danger nearby",
        "He studied the {w} population",
        "The {w} growled at intruders",
        "I love all {w} species",
        "The {w} hunted for food",
        "She classified the {w} correctly",
        "The {w} has thick fur",
        "A big {w} roamed the territory",
        "The {w} panted in the heat",
        "He observed the {w} every morning",
    ],
    "red": [
        "The {w} color filled the canvas",
        "She wore a {w} dress to the party",
        "The {w} light means stop",
        "He painted the door {w}",
        "The sunset was {w} and gold",
        "A {w} rose bloomed in the garden",
        "The {w} ink stained the paper",
        "I prefer the {w} one over the blue",
        "The {w} flag waved in the wind",
        "She chose the {w} wine",
        "The {w} carpet was luxurious",
        "A {w} bird sat on the branch",
        "The {w} leaves signaled autumn",
        "He drove the {w} car",
        "The {w} planet glowed in the sky",
        "I saw a {w} flash of lightning",
        "The {w} berries were ripe",
        "She mixed {w} and blue to make purple",
        "The {w} ribbon decorated the gift",
        "A {w} scar marked his arm",
        "The {w} team scored a goal",
        "He wore a {w} tie to work",
        "The {w} apple caught my eye",
        "I like the {w} version better",
        "The {w} light blinked urgently",
        "She used {w} thread for the embroidery",
        "The {w} pepper was very spicy",
        "A {w} glow filled the room",
        "The {w} indicator showed danger",
        "He painted the fence {w}",
    ],
    "color": [
        "The {w} filled the canvas beautifully",
        "She wore a vibrant {w} to the party",
        "The {w} of the light matters",
        "He chose the right {w} for the door",
        "The sunset had many {w}s",
        "A beautiful {w} appeared in the garden",
        "The {w} of the ink was striking",
        "I prefer warm {w}s over cool ones",
        "The national {w}s waved in the wind",
        "She studied the {w} of the wine",
        "The {w} of the carpet was elegant",
        "A vivid {w} caught my attention",
        "The {w} of the leaves changed with seasons",
        "He discussed the {w} of the car",
        "The {w} of the planet was visible",
        "I noticed the {w} shifting",
        "The {w} of the berries indicated ripeness",
        "She mixed {w}s to create new shades",
        "The {w} of the ribbon was perfect",
        "A striking {w} marked the design",
        "The team {w} was distinctive",
        "He selected a {w} for the project",
        "The {w} of the apple was appealing",
        "I like the {w} palette better",
        "The {w} indicated a warning",
        "She worked with {w} in her art",
        "The {w} of the pepper was unusual",
        "A warm {w} filled the room",
        "The {w} conveyed an important signal",
        "He analyzed the {w} scientifically",
    ],
}

# Part 2: 任务复用对
TASK_TEMPLATES = {
    "translate_en": [
        "Please translate this to English: {text}",
        "Can you translate the following into English: {text}",
        "Translate into English: {text}",
        "I need this translated to English: {text}",
        "English translation of: {text}",
    ],
    "translate_fr": [
        "Please translate this to French: {text}",
        "Can you translate the following into French: {text}",
        "Translate into French: {text}",
        "I need this translated to French: {text}",
        "French translation of: {text}",
    ],
}

TASK_INPUTS = [
    "Hello world",
    "The cat sat on the mat",
    "I love music",
    "She reads books every day",
    "The weather is nice today",
    "We went to the park",
    "He plays guitar well",
    "They are good friends",
    "The food was delicious",
    "She speaks three languages",
    "I enjoy cooking",
    "The movie was exciting",
    "He runs fast",
    "We had a great time",
    "The book is interesting",
    "She sings beautifully",
    "I feel happy today",
    "The garden looks lovely",
    "He works hard",
    "They traveled abroad",
    "I like chocolate",
    "The river flows south",
    "She smiled at me",
    "We learned something new",
    "The sky is clear tonight",
    "He fixed the problem",
    "I need more time",
    "The coffee is hot",
    "She won the prize",
    "We celebrated together",
]

# Part 3: 逻辑功能对
LOGIC_TEMPLATES = {
    "and": [
        "apples {w} oranges are both fruits",
        "she {w} he went to the store",
        "cats {w} dogs are common pets",
        "bread {w} butter make a sandwich",
        "salt {w} pepper are on the table",
        "sun {w} rain make a rainbow",
        "fire {w} ice are opposites",
        "love {w} trust build relationships",
        "patience {w} practice lead to mastery",
        "reading {w} writing go together",
        "science {w} art both require creativity",
        "work hard {w} play hard",
        "tea {w} coffee are popular drinks",
        "music {w} dance are related arts",
        "left {w} right are directions",
    ],
    "or": [
        "apples {w} oranges, which do you prefer",
        "she {w} he will go to the store",
        "cats {w} dogs, which are better pets",
        "bread {w} butter, choose one",
        "salt {w} pepper, which is more important",
        "sun {w} rain, what will it be",
        "fire {w} ice, pick one",
        "love {w} trust, which matters more",
        "patience {w} talent, which wins",
        "reading {w} writing, which is harder",
        "science {w} art, which interests you",
        "work hard {w} give up, the choice is yours",
        "tea {w} coffee, what would you like",
        "music {w} dance, which do you enjoy",
        "left {w} right, which way",
    ],
    "not": [
        "that is {w} what I expected",
        "she did {w} come to the party",
        "he is {w} happy about this",
        "the result was {w} surprising",
        "this is {w} the right answer",
        "they are {w} going home yet",
        "I am {w} sure about this",
        "the door is {w} open",
        "she was {w} impressed",
        "he will {w} forget this",
        "the weather is {w} good today",
        "they have {w} finished yet",
        "I can {w} believe this",
        "the car is {w} working",
        "she does {w} know the truth",
    ],
    "but": [
        "it was late {w} she kept working",
        "he tried hard {w} failed anyway",
        "the food was expensive {w} delicious",
        "she was tired {w} continued walking",
        "it rained {w} we still had fun",
        "he is young {w} very talented",
        "the movie was long {w} entertaining",
        "she was scared {w} faced her fears",
        "the task was hard {w} rewarding",
        "he was busy {w} made time for us",
        "the test was difficult {w} she passed",
        "it was cold {w} sunny outside",
        "the book was thick {w} easy to read",
        "she was shy {w} spoke confidently",
        "the road was rough {w} scenic",
    ],
}

# 概念对定义
CONCEPT_PAIRS = [
    ("apple", "fruit", "specific_instance", "苹果/水果"),
    ("dog", "animal", "specific_instance", "狗/动物"),
    ("red", "color", "specific_instance", "红色/颜色"),
]

LOGIC_PAIRS = [
    ("and", "or", "disjunction_pair", "AND/OR"),
    ("not", "but", "contrast_pair", "NOT/BUT"),
]


def find_target_token_pos_in_full(tokenizer, input_ids, target_word):
    """在完整的token序列(含BOS)中找到目标词位置
    
    返回: (position_in_full_sequence, target_token_length) 或 (None, None)
    """
    # 策略: 逐token解码并匹配
    tokens_list = input_ids[0].tolist()
    
    for i in range(len(tokens_list)):
        # 逐个token检查
        decoded = tokenizer.decode(tokens_list[i])
        if target_word.lower() in decoded.lower():
            return i, 1
        # 检查多token组合
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            # 精确匹配: 解码后剥离空格等于目标词
            if target_word.lower() == decoded.strip().lower():
                return i, j - i
    
    # 策略2: 更宽松的匹配
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 2:
                return i, j - i
    
    return None, None


def collect_activations_at_target(model, tokenizer, device, sentences, target_word, 
                                   n_layers, target_layers):
    """收集目标词位置的激活向量"""
    activations = {li: [] for li in target_layers}
    found_count = 0
    
    for sent_template in sentences:
        sentence = sent_template.replace("{w}", target_word)
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(device)
        seq_len = input_ids.shape[1]
        
        # 找目标词位置 (在完整序列中)
        pos, target_len = find_target_token_pos_in_full(tokenizer, input_ids, target_word)
        
        if pos is None or pos >= seq_len:
            continue
        
        # 用目标词的中间token位置
        actual_pos = pos + (target_len // 2)
        actual_pos = min(actual_pos, seq_len - 1)  # 确保在范围内
        
        # 用hook收集各层输出
        layers = get_layers(model)
        captured = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in target_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                for h in hooks:
                    h.remove()
                continue
        
        for h in hooks:
            h.remove()
        
        # 提取目标位置的激活
        for li in target_layers:
            key = f"L{li}"
            if key in captured:
                act = captured[key][0, actual_pos, :].numpy()  # [d_model]
                activations[li].append(act)
    
    if found_count > 0:
        log_time(f"    找到 '{target_word}' 在 {found_count}/{len(sentences)} 句中的位置")
    else:
        log_time(f"    警告: 未找到 '{target_word}' 在任何句中的位置!")
    
    return activations


def compute_subspace_analysis(activations_a, activations_b, n_dims=15):
    """分析两个概念激活的共享/独特子空间"""
    if len(activations_a) < 5 or len(activations_b) < 5:
        return None
    
    A = np.array(activations_a)  # [n_samples, d_model]
    B = np.array(activations_b)  # [n_samples, d_model]
    
    # 中心化
    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    A_centered = A - mean_a
    B_centered = B - mean_b
    
    # 1. 余弦相似度 (均值方向)
    cos_mean = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10))
    
    # 2. PCA提取各概念的子空间
    n_comp = min(n_dims, min(A_centered.shape) - 1, min(B_centered.shape) - 1)
    n_comp = max(n_comp, 2)
    
    from sklearn.decomposition import PCA
    pca_a = PCA(n_components=n_comp)
    pca_a.fit(A_centered)
    
    pca_b = PCA(n_components=n_comp)
    pca_b.fit(B_centered)
    
    # 3. 子空间重叠度
    V_a = pca_a.components_  # [n_comp, d_model]
    V_b = pca_b.components_  # [n_comp, d_model]
    
    # 子空间重叠 = ||V_a @ V_b^T||_F^2 / n_comp
    overlap_matrix = V_a @ V_b.T  # [n_comp, n_comp]
    subspace_overlap = float(np.sum(overlap_matrix ** 2) / n_comp)
    
    # 4. 逐维度重叠 (哪个维度共享最多)
    dim_overlaps = np.sum(overlap_matrix ** 2, axis=1)  # [n_comp] - A的每个维度与B的重叠
    
    # 5. 共享子空间和独特子空间
    # 使用SVD分解overlap_matrix来找到共享方向
    U_ov, S_ov, Vt_ov = np.linalg.svd(overlap_matrix, full_matrices=False)
    
    # 共享方向: 最大的S_ov对应的V_a和V_b方向
    n_shared = min(5, len(S_ov))
    shared_strength = float(S_ov[0] ** 2)  # 最强共享方向的强度
    
    # 6. 解释方差比
    var_ratio_a = pca_a.explained_variance_ratio_
    var_ratio_b = pca_b.explained_variance_ratio_
    
    # 7. 共享能量分解
    # 将A的方差分解为: 在B子空间中的投影(共享) + 正交部分(独特)
    A_proj_B = V_b.T @ (V_b @ A_centered.T)  # B子空间中的投影
    A_shared_energy = float(np.sum(A_proj_B ** 2))
    A_total_energy = float(np.sum(A_centered ** 2))
    shared_ratio_A = A_shared_energy / max(A_total_energy, 1e-10)
    
    B_proj_A = V_a.T @ (V_a @ B_centered.T)
    B_shared_energy = float(np.sum(B_proj_A ** 2))
    B_total_energy = float(np.sum(B_centered ** 2))
    shared_ratio_B = B_shared_energy / max(B_total_energy, 1e-10)
    
    # 8. 独特维度分析
    # A的独特维度 = A中与B子空间最正交的PCA维度
    ortho_scores = 1 - dim_overlaps  # A每个维度与B的正交度
    
    # 9. 均值差异的方向
    delta_mean = mean_a - mean_b
    delta_norm = np.linalg.norm(delta_mean)
    
    # delta_mean在共享子空间和独特子空间中的投影
    delta_proj_shared = V_b.T @ (V_b @ delta_mean) if delta_norm > 1e-10 else np.zeros_like(delta_mean)
    delta_proj_unique = delta_mean - delta_proj_shared
    
    shared_delta_ratio = float(np.sum(delta_proj_shared**2) / max(np.sum(delta_mean**2), 1e-10))
    unique_delta_ratio = float(np.sum(delta_proj_unique**2) / max(np.sum(delta_mean**2), 1e-10))
    
    return {
        "cos_mean": cos_mean,
        "subspace_overlap": subspace_overlap,
        "dim_overlaps": dim_overlaps.tolist(),
        "shared_strength": shared_strength,
        "svd_shared": S_ov[:min(10, len(S_ov))].tolist(),
        "var_ratio_a": var_ratio_a[:min(10, len(var_ratio_a))].tolist(),
        "var_ratio_b": var_ratio_b[:min(10, len(var_ratio_b))].tolist(),
        "shared_ratio_A": shared_ratio_A,
        "shared_ratio_B": shared_ratio_B,
        "avg_shared_ratio": (shared_ratio_A + shared_ratio_B) / 2,
        "ortho_scores": ortho_scores.tolist(),
        "n_samples_a": len(activations_a),
        "n_samples_b": len(activations_b),
        "delta_mean_norm": float(delta_norm),
        "shared_delta_ratio": shared_delta_ratio,
        "unique_delta_ratio": unique_delta_ratio,
        "pca_a_components": V_a.tolist(),  # 保存用于跨概念分析
        "pca_b_components": V_b.tolist(),
    }


def collect_logic_activations(model, tokenizer, device, templates, target_word, 
                              n_layers, target_layers):
    """收集逻辑词位置的激活"""
    activations = {li: [] for li in target_layers}
    found_count = 0
    
    for sent_template in templates:
        sentence = sent_template.replace("{w}", target_word)
        
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(device)
        seq_len = input_ids.shape[1]
        
        pos, target_len = find_target_token_pos_in_full(tokenizer, input_ids, target_word)
        
        if pos is None or pos >= seq_len:
            continue
        
        found_count += 1
        actual_pos = pos + (target_len // 2)
        actual_pos = min(actual_pos, seq_len - 1)
        
        layers = get_layers(model)
        captured = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in target_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                for h in hooks:
                    h.remove()
                continue
        
        for h in hooks:
            h.remove()
        
        for li in target_layers:
            key = f"L{li}"
            if key in captured and actual_pos < captured[key].shape[1]:
                act = captured[key][0, actual_pos, :].numpy()
                activations[li].append(act)
    
    return activations


def collect_task_activations(model, tokenizer, device, task_key, n_layers, target_layers):
    """收集任务指令位置的激活"""
    templates = TASK_TEMPLATES[task_key]
    activations = {li: [] for li in target_layers}
    found_count = 0
    
    for template in templates:
        for text in TASK_INPUTS:
            sentence = template.replace("{text}", text)
            
            inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            # 找目标语言词的位置
            target_lang = "English" if "en" in task_key else "French"
            
            pos, target_len = find_target_token_pos_in_full(tokenizer, input_ids, target_lang)
            
            if pos is None or pos >= seq_len:
                continue
            
            found_count += 1
            actual_pos = pos + (target_len // 2)
            actual_pos = min(actual_pos, seq_len - 1)
            
            layers = get_layers(model)
            captured = {}
            
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            hooks = []
            for li in target_layers:
                hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids)
                except Exception as e:
                    for h in hooks:
                        h.remove()
                    continue
            
            for h in hooks:
                h.remove()
            
            for li in target_layers:
                key = f"L{li}"
                if key in captured and actual_pos < captured[key].shape[1]:
                    act = captured[key][0, actual_pos, :].numpy()
                    activations[li].append(act)
    
    return activations


def run_experiment(model_name):
    """运行完整实验"""
    import torch
    global torch
    
    log_time(f"=" * 60)
    log_time(f"Phase 57: 概念复用与差异化 - {model_name}")
    log_time(f"=" * 60)
    
    # 加载模型
    attn_impl = "eager" if model_name == "deepseek7b" else "sdpa"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"加载模型 {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
            attn_implementation=attn_impl, use_cache=False,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl, use_cache=False,
        )
    
    model.eval()
    device = next(model.parameters()).device
    model_info = get_model_info(model, model_name)
    
    log_time(f"模型: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 采样层
    target_layers = sorted(set(
        [0, 1] + list(range(0, n_layers, max(1, n_layers//8))) + [n_layers//2, n_layers-2, n_layers-1]
    ))
    target_layers = [l for l in target_layers if l < n_layers]
    log_time(f"采样层: {target_layers}")
    
    results = {
        "model": model_name,
        "model_class": model_info.model_class,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": target_layers,
        "concept_pairs": {},
        "logic_pairs": {},
        "task_pair": {},
        "cross_concept_backbone": {},
    }
    
    # ===== Part 1: 概念层级对 =====
    log_time(f"\n{'='*40}")
    log_time(f"Part 1: 概念层级对分析")
    log_time(f"{'='*40}")
    
    for word_a, word_b, pair_type, pair_name in CONCEPT_PAIRS:
        log_time(f"\n--- 概念对: {pair_name} ({word_a}/{word_b}) ---")
        
        templates = CONCEPT_TEMPLATES.get(word_a, CONCEPT_TEMPLATES.get(word_b, []))
        if not templates:
            log_time(f"  跳过: 无模板")
            continue
        
        # 收集两个概念的激活
        log_time(f"  收集 '{word_a}' 激活 ({len(templates)} 句)...")
        act_a = collect_activations_at_target(
            model, tokenizer, device, templates, word_a, n_layers, target_layers
        )
        
        log_time(f"  收集 '{word_b}' 激活 ({len(templates)} 句)...")
        act_b = collect_activations_at_target(
            model, tokenizer, device, templates, word_b, n_layers, target_layers
        )
        
        # 逐层分析
        pair_results = {}
        for li in target_layers:
            n_a = len(act_a.get(li, []))
            n_b = len(act_b.get(li, []))
            
            if n_a < 5 or n_b < 5:
                log_time(f"  L{li}: 样本不足 (a={n_a}, b={n_b}), 跳过")
                continue
            
            analysis = compute_subspace_analysis(act_a[li], act_b[li], n_dims=15)
            if analysis is None:
                continue
            
            pair_results[str(li)] = {
                "cos_mean": analysis["cos_mean"],
                "subspace_overlap": analysis["subspace_overlap"],
                "shared_ratio_A": analysis["shared_ratio_A"],
                "shared_ratio_B": analysis["shared_ratio_B"],
                "avg_shared_ratio": analysis["avg_shared_ratio"],
                "shared_delta_ratio": analysis["shared_delta_ratio"],
                "unique_delta_ratio": analysis["unique_delta_ratio"],
                "delta_mean_norm": analysis["delta_mean_norm"],
                "var_ratio_a_top5": analysis["var_ratio_a"][:5],
                "var_ratio_b_top5": analysis["var_ratio_b"][:5],
                "dim_overlaps_top5": analysis["dim_overlaps"][:5],
                "ortho_scores_top5": analysis["ortho_scores"][:5],
                "svd_shared_top5": analysis["svd_shared"][:5],
                "n_samples_a": analysis["n_samples_a"],
                "n_samples_b": analysis["n_samples_b"],
            }
            
            log_time(f"  L{li}: cos={analysis['cos_mean']:.4f}, "
                     f"overlap={analysis['subspace_overlap']:.4f}, "
                     f"shared_A={analysis['shared_ratio_A']:.4f}, "
                     f"shared_B={analysis['shared_ratio_B']:.4f}, "
                     f"delta_shared={analysis['shared_delta_ratio']:.4f}, "
                     f"delta_unique={analysis['unique_delta_ratio']:.4f}")
        
        pair_key = f"{word_a}_{word_b}"
        results["concept_pairs"][pair_key] = {
            "pair_name": pair_name,
            "pair_type": pair_type,
            "layers": pair_results,
        }
        
        # 释放内存
        del act_a, act_b
        torch.cuda.empty_cache()
    
    # ===== Part 2: 任务复用对 =====
    log_time(f"\n{'='*40}")
    log_time(f"Part 2: 任务复用对 (translate_en/translate_fr)")
    log_time(f"{'='*40}")
    
    act_en = collect_task_activations(model, tokenizer, device, "translate_en", n_layers, target_layers)
    act_fr = collect_task_activations(model, tokenizer, device, "translate_fr", n_layers, target_layers)
    
    task_results = {}
    for li in target_layers:
        n_en = len(act_en.get(li, []))
        n_fr = len(act_fr.get(li, []))
        
        if n_en < 5 or n_fr < 5:
            log_time(f"  L{li}: 样本不足 (en={n_en}, fr={n_fr}), 跳过")
            continue
        
        analysis = compute_subspace_analysis(act_en[li], act_fr[li], n_dims=15)
        if analysis is None:
            continue
        
        task_results[str(li)] = {
            "cos_mean": analysis["cos_mean"],
            "subspace_overlap": analysis["subspace_overlap"],
            "shared_ratio_A": analysis["shared_ratio_A"],
            "shared_ratio_B": analysis["shared_ratio_B"],
            "avg_shared_ratio": analysis["avg_shared_ratio"],
            "shared_delta_ratio": analysis["shared_delta_ratio"],
            "unique_delta_ratio": analysis["unique_delta_ratio"],
            "delta_mean_norm": analysis["delta_mean_norm"],
            "var_ratio_a_top5": analysis["var_ratio_a"][:5],
            "var_ratio_b_top5": analysis["var_ratio_b"][:5],
            "n_samples_a": analysis["n_samples_a"],
            "n_samples_b": analysis["n_samples_b"],
        }
        
        log_time(f"  L{li}: cos={analysis['cos_mean']:.4f}, "
                 f"overlap={analysis['subspace_overlap']:.4f}, "
                 f"shared={analysis['avg_shared_ratio']:.4f}, "
                 f"delta_unique={analysis['unique_delta_ratio']:.4f}")
    
    results["task_pair"]["translate_en_fr"] = {
        "pair_name": "翻译英文/法文",
        "pair_type": "task_reuse",
        "layers": task_results,
    }
    
    del act_en, act_fr
    torch.cuda.empty_cache()
    
    # ===== Part 3: 逻辑功能对 =====
    log_time(f"\n{'='*40}")
    log_time(f"Part 3: 逻辑功能对")
    log_time(f"{'='*40}")
    
    for word_a, word_b, pair_type, pair_name in LOGIC_PAIRS:
        log_time(f"\n--- 逻辑对: {pair_name} ({word_a}/{word_b}) ---")
        
        templates_a = LOGIC_TEMPLATES.get(word_a, [])
        templates_b = LOGIC_TEMPLATES.get(word_b, [])
        
        if not templates_a or not templates_b:
            log_time(f"  跳过: 无模板")
            continue
        
        act_a = collect_logic_activations(model, tokenizer, device, templates_a, word_a, n_layers, target_layers)
        act_b = collect_logic_activations(model, tokenizer, device, templates_b, word_b, n_layers, target_layers)
        
        logic_results = {}
        for li in target_layers:
            n_a = len(act_a.get(li, []))
            n_b = len(act_b.get(li, []))
            
            if n_a < 5 or n_b < 5:
                log_time(f"  L{li}: 样本不足 (a={n_a}, b={n_b}), 跳过")
                continue
            
            analysis = compute_subspace_analysis(act_a[li], act_b[li], n_dims=15)
            if analysis is None:
                continue
            
            logic_results[str(li)] = {
                "cos_mean": analysis["cos_mean"],
                "subspace_overlap": analysis["subspace_overlap"],
                "shared_ratio_A": analysis["shared_ratio_A"],
                "shared_ratio_B": analysis["shared_ratio_B"],
                "avg_shared_ratio": analysis["avg_shared_ratio"],
                "shared_delta_ratio": analysis["shared_delta_ratio"],
                "unique_delta_ratio": analysis["unique_delta_ratio"],
                "delta_mean_norm": analysis["delta_mean_norm"],
                "n_samples_a": analysis["n_samples_a"],
                "n_samples_b": analysis["n_samples_b"],
            }
            
            log_time(f"  L{li}: cos={analysis['cos_mean']:.4f}, "
                     f"overlap={analysis['subspace_overlap']:.4f}, "
                     f"shared={analysis['avg_shared_ratio']:.4f}, "
                     f"delta_unique={analysis['unique_delta_ratio']:.4f}")
        
        pair_key = f"{word_a}_{word_b}"
        results["logic_pairs"][pair_key] = {
            "pair_name": pair_name,
            "pair_type": pair_type,
            "layers": logic_results,
        }
        
        del act_a, act_b
        torch.cuda.empty_cache()
    
    # ===== Part 4: 跨概念复用骨干分析 =====
    log_time(f"\n{'='*40}")
    log_time(f"Part 4: 跨概念复用骨干")
    log_time(f"{'='*40}")
    
    # 对每个层, 检查所有概念对的共享维度是否一致
    # 方法: 如果apple-fruit的共享方向与dog-animal的共享方向对齐, 
    #       那么存在跨概念的"复用骨干"
    
    # 重新收集所有概念的激活 (只在中间层)
    mid_layer = n_layers // 2
    
    concept_activations = {}
    for word_a, word_b, _, _ in CONCEPT_PAIRS:
        templates = CONCEPT_TEMPLATES.get(word_a, CONCEPT_TEMPLATES.get(word_b, []))
        if not templates:
            continue
        
        for word in [word_a, word_b]:
            if word in concept_activations:
                continue
            log_time(f"  收集 '{word}' 在L{mid_layer}的激活...")
            act = collect_activations_at_target(
                model, tokenizer, device, templates, word, n_layers, [mid_layer]
            )
            if mid_layer in act and len(act[mid_layer]) >= 5:
                concept_activations[word] = np.array(act[mid_layer])
    
    # 计算跨概念对齐
    cross_results = {}
    all_words = list(concept_activations.keys())
    
    if len(all_words) >= 2:
        # 对每对概念计算共享子空间方向
        from sklearn.decomposition import PCA
        
        concept_pcas = {}
        for word in all_words:
            A = concept_activations[word] - concept_activations[word].mean(axis=0)
            pca = PCA(n_components=min(15, min(A.shape) - 1))
            pca.fit(A)
            concept_pcas[word] = pca.components_  # [n_comp, d_model]
        
        # 跨概念对: 比较不同概念对的共享方向
        pair_shared_dirs = {}
        for word_a, word_b, _, pair_name in CONCEPT_PAIRS:
            if word_a not in concept_pcas or word_b not in concept_pcas:
                continue
            V_a = concept_pcas[word_a]
            V_b = concept_pcas[word_b]
            
            # 找最强共享方向
            overlap = V_a @ V_b.T  # [n_comp_a, n_comp_b]
            # SVD找共享方向
            U, S, Vt = np.linalg.svd(overlap, full_matrices=False)
            # 共享方向在原始空间中
            shared_dir_in_a = U[:, 0] @ V_a  # A的共享方向在d_model空间
            shared_dir_in_b = Vt[0, :] @ V_b  # B的共享方向在d_model空间
            
            pair_shared_dirs[pair_name] = {
                "dir_a": shared_dir_in_a.tolist(),
                "dir_b": shared_dir_in_b.tolist(),
                "strength": float(S[0] ** 2),
            }
        
        # 不同概念对的共享方向是否一致?
        pair_names = list(pair_shared_dirs.keys())
        backbone_overlaps = {}
        for i, pn1 in enumerate(pair_names):
            for j, pn2 in enumerate(pair_names):
                if j <= i:
                    continue
                d1 = np.array(pair_shared_dirs[pn1]["dir_a"])
                d2 = np.array(pair_shared_dirs[pn2]["dir_a"])
                cos_backbone = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))
                backbone_overlaps[f"{pn1}_vs_{pn2}"] = cos_backbone
                log_time(f"  复用骨干对齐 {pn1} vs {pn2}: cos={cos_backbone:.4f}")
        
        cross_results = {
            "layer": mid_layer,
            "backbone_overlaps": backbone_overlaps,
            "pair_strengths": {k: v["strength"] for k, v in pair_shared_dirs.items()},
        }
    
    results["cross_concept_backbone"] = cross_results
    
    # ===== 保存结果 =====
    output_dir = PROJECT / "results" / "subspace_topology"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"exp3_reuse_diff_{model_name}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    
    log_time(f"结果已保存: {output_file}")
    
    # 释放模型
    release_model(model)
    
    return results


def print_summary(results):
    """打印关键发现摘要"""
    print("\n" + "=" * 60)
    print("Phase 57 关键发现摘要")
    print("=" * 60)
    
    # Part 1: 概念层级
    print("\n--- Part 1: 概念层级复用/差异化 ---")
    for pair_key, pair_data in results.get("concept_pairs", {}).items():
        print(f"\n  {pair_data['pair_name']}:")
        for li_str, lr in sorted(pair_data["layers"].items(), key=lambda x: int(x[0])):
            print(f"    L{li_str}: cos={lr['cos_mean']:.4f}, "
                  f"overlap={lr['subspace_overlap']:.4f}, "
                  f"shared_A={lr['shared_ratio_A']:.4f}, "
                  f"shared_B={lr['shared_ratio_B']:.4f}, "
                  f"delta_unique={lr['unique_delta_ratio']:.4f}")
    
    # Part 2: 任务复用
    print("\n--- Part 2: 任务复用 ---")
    for pair_key, pair_data in results.get("task_pair", {}).items():
        print(f"\n  {pair_data['pair_name']}:")
        for li_str, lr in sorted(pair_data["layers"].items(), key=lambda x: int(x[0])):
            print(f"    L{li_str}: cos={lr['cos_mean']:.4f}, "
                  f"overlap={lr['subspace_overlap']:.4f}, "
                  f"shared={lr['avg_shared_ratio']:.4f}, "
                  f"delta_unique={lr['unique_delta_ratio']:.4f}")
    
    # Part 3: 逻辑功能
    print("\n--- Part 3: 逻辑功能复用 ---")
    for pair_key, pair_data in results.get("logic_pairs", {}).items():
        print(f"\n  {pair_data['pair_name']}:")
        for li_str, lr in sorted(pair_data["layers"].items(), key=lambda x: int(x[0])):
            print(f"    L{li_str}: cos={lr['cos_mean']:.4f}, "
                  f"overlap={lr['subspace_overlap']:.4f}, "
                  f"shared={lr['avg_shared_ratio']:.4f}, "
                  f"delta_unique={lr['unique_delta_ratio']:.4f}")
    
    # Part 4: 跨概念骨干
    print("\n--- Part 4: 跨概念复用骨干 ---")
    backbone = results.get("cross_concept_backbone", {})
    for k, v in backbone.get("backbone_overlaps", {}).items():
        print(f"  {k}: cos={v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    results = run_experiment(args.model)
    print_summary(results)
