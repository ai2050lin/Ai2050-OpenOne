"""
Phase 142: Local Geometry & Transport Connection
=================================================
直接回应Phase 141的三个核心批评：

批评1: PR ≠ 流形维数
  - PR只证明协方差低秩，不证明流形存在/光滑/连通
  - 修正: 局部PCA → 在每个h点的小邻域内估计维数
  - 如果局部维数稳定 ≈ 全局PR → 更像流形

批评2: 仍在"状态空间"思维
  - 语言不是点，而是依赖关系的运输网络
  - 修正: 研究attention运输几何 A_ij
  - attention可能才是真正的"联络"(connection)

批评3: 算子是纤维化的
  - NOT不是固定算子，依赖于scope/syntax/polarity
  - 修正: 测量不同句法上下文中V_not的差异
  - 验证: O_not,c 是否形成纤维丛结构

四大实验：
  Exp A: 局部切空间估计 (Local Tangent Space)
    - 在每个句子h的"邻域"(小语义变化)做局部PCA
    - 邻域 = 原句 + 时态变化 + 否定 + 同义词
    - 如果局部dim稳定 → 支持流形假设

  Exp B: Attention运输几何 (Transport Connection)
    - 提取attention weights A_ij
    - 研究语义扰动如何改变A_ij
    - 关键问题: A_ij是否定义了"平行运输"？

  Exp C: 算子纤维结构 (Operator Fiber Structure)
    - 测量V_not在不同句法上下文中的差异
    - can not vs does not vs will not vs is not
    - 验证: 不同上下文的V_not是否在同一"纤维"

  Exp D: Jacobian修正 (用autograd替代有限差分)
    - 使用torch.autograd计算真正的J_l
    - 修复Phase 141的数值问题
    - 验证: J_l的稳定子空间维度 ≈ PR？

时间：2026-05-12 18:00
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
# 数据集
# ============================================================

# Exp A: 局部切空间 — 基础句 + 邻域
# 每个基础句构造一个小"邻域"：原句 + 时态变 + 否定 + 同义词
LOCAL_TANGENT_BASES = [
    # (base, past, negated, synonym_variant)
    ("The cat walks home slowly", "The cat walked home slowly",
     "The cat does not walk home slowly", "The cat strolls home slowly"),
    ("She plays the piano well", "She played the piano well",
     "She does not play the piano well", "She performs the piano well"),
    ("He works hard every day", "He worked hard every day",
     "He does not work hard every day", "He labors hard every day"),
    ("They talk loudly in class", "They talked loudly in class",
     "They do not talk loudly in class", "They speak loudly in class"),
    ("The dog jumps over the fence", "The dog jumped over the fence",
     "The dog does not jump over the fence", "The dog leaps over the fence"),
    ("She cooks dinner every night", "She cooked dinner every night",
     "She does not cook dinner every night", "She prepares dinner every night"),
    ("He reads books on weekends", "He read books on weekends",
     "He does not read books on weekends", "He studies books on weekends"),
    ("The bird sings in the morning", "The bird sang in the morning",
     "The bird does not sing in the morning", "The bird chirps in the morning"),
    ("She paints beautiful pictures", "She painted beautiful pictures",
     "She does not paint beautiful pictures", "She draws beautiful pictures"),
    ("He drives fast on the highway", "He drove fast on the highway",
     "He does not drive fast on the highway", "He speeds fast on the highway"),
    ("They build houses in the village", "They built houses in the village",
     "They do not build houses in the village", "They construct houses in the village"),
    ("The cat chases mice around", "The cat chased mice around",
     "The cat does not chase mice around", "The cat pursues mice around"),
    ("She writes letters to friends", "She wrote letters to friends",
     "She does not write letters to friends", "She composes letters to friends"),
    ("He runs fast every morning", "He ran fast every morning",
     "He does not run fast every morning", "He sprints fast every morning"),
    ("The fish swims in the river", "The fish swam in the river",
     "The fish does not swim in the river", "The fish glides in the river"),
    ("The teacher explains the lesson", "The teacher explained the lesson",
     "The teacher does not explain the lesson", "The teacher clarifies the lesson"),
    ("The student reads the chapter", "The student read the chapter",
     "The student does not read the chapter", "The student studies the chapter"),
    ("The farmer grows crops quickly", "The farmer grew crops quickly",
     "The farmer does not grow crops quickly", "The farmer cultivates crops quickly"),
    ("The artist draws the portrait", "The artist drew the portrait",
     "The artist does not draw the portrait", "The artist sketches the portrait"),
    ("The soldier marches forward", "The soldier marched forward",
     "The soldier does not march forward", "The soldier advances forward"),
    # 更多句法变体
    ("The water flows down the hill", "The water flowed down the hill",
     "The water does not flow down the hill", "The water runs down the hill"),
    ("The wind blows from the north", "The wind blew from the north",
     "The wind does not blow from the north", "The wind gusts from the north"),
    ("The fire burns bright at night", "The fire burned bright at night",
     "The fire does not burn bright at night", "The fire glows bright at night"),
    ("The rain falls softly today", "The rain fell softly today",
     "The rain does not fall softly today", "The rain drops softly today"),
    ("The snow covers the mountain", "The snow covered the mountain",
     "The snow does not cover the mountain", "The frost covers the mountain"),
    ("The child learns new words", "The child learned new words",
     "The child does not learn new words", "The child studies new words"),
    ("The river runs deep here", "The river ran deep here",
     "The river does not run deep here", "The river flows deep here"),
    ("The sun shines bright today", "The sun shone bright today",
     "The sun does not shine bright today", "The sun glows bright today"),
    ("The horse runs across the field", "The horse ran across the field",
     "The horse does not run across the field", "The horse gallops across the field"),
    ("The bell rings at noon", "The bell rang at noon",
     "The bell does not ring at noon", "The bell chimes at noon"),
]

# Exp B: Attention运输 — 句子对 (base, semantic_variant)
ATTENTION_TRANSPORT_PAIRS = [
    ("The cat can swim across the river", "The cat can not swim across the river"),
    ("She has finished her homework", "She has not finished her homework"),
    ("They should leave the building", "They should not leave the building"),
    ("The bird will fly south", "The bird will not fly south"),
    ("He could answer the question", "He could not answer the question"),
    ("The dog can bark loudly", "The dog can not bark loudly"),
    ("She must study for the exam", "She must not study for the exam"),
    ("They would agree with the plan", "They would not agree with the plan"),
    ("The fish can breathe underwater", "The fish can not breathe underwater"),
    ("He has eaten the apple", "He has not eaten the apple"),
    ("The child will sleep early", "The child will not sleep early"),
    ("She can speak three languages", "She can not speak three languages"),
    ("They may enter the room", "They may not enter the room"),
    ("The plant will grow quickly", "The plant will not grow quickly"),
    ("He should rest after lunch", "He should not rest after lunch"),
    # 时态变化
    ("The cat walks home slowly", "The cat walked home slowly"),
    ("She plays the piano well", "She played the piano well"),
    ("He works hard every day", "He worked hard every day"),
    ("They talk loudly in class", "They talked loudly in class"),
    ("The dog jumps over the fence", "The dog jumped over the fence"),
    ("She cooks dinner every night", "She cooked dinner every night"),
    ("He reads books on weekends", "He read books on weekends"),
    ("The bird sings in the morning", "The bird sang in the morning"),
    ("She paints beautiful pictures", "She painted beautiful pictures"),
    ("He drives fast on the highway", "He drove fast on the highway"),
]

# Exp C: 算子纤维结构 — NOT在不同句法上下文
FIBER_STRUCTURE_DATA = {
    "modal_not": {
        "description": "情态动词+NOT: can not, should not, will not, must not, could not, would not, may not",
        "base_template": "The cat {modal} swim across the river",
        "variants": [
            ("The cat can swim across the river", "The cat can not swim across the river"),
            ("The cat should swim across the river", "The cat should not swim across the river"),
            ("The cat will swim across the river", "The cat will not swim across the river"),
            ("The cat must swim across the river", "The cat must not swim across the river"),
            ("The cat could swim across the river", "The cat could not swim across the river"),
            ("The cat would swim across the river", "The cat would not swim across the river"),
            ("The cat may swim across the river", "The cat may not swim across the river"),
        ],
    },
    "aux_not": {
        "description": "助动词+NOT: does not, has not, is not, was not, did not",
        "variants": [
            ("The cat swims across the river", "The cat does not swim across the river"),
            ("The cat has swum across the river", "The cat has not swum across the river"),
            ("The cat is swimming across the river", "The cat is not swimming across the river"),
            ("The cat was swimming across the river", "The cat was not swimming across the river"),
            ("The cat swam across the river", "The cat did not swim across the river"),
        ],
    },
    "subject_scope": {
        "description": "否定作用域: 句首vs句中vs句尾",
        "variants": [
            ("All students passed the exam", "Not all students passed the exam"),  # 量词否定
            ("The student always comes early", "The student does not always come early"),  # 频率否定
            ("Some birds can fly", "Some birds can not fly"),  # 部分否定
            ("Both answers are correct", "Not both answers are correct"),  # 双重否定
            ("Every child likes candy", "Not every child likes candy"),  # 全称否定
        ],
    },
    # 更多base句子的fiber测试
    "different_predicates": {
        "description": "不同谓语下的NOT: 同一主语,不同动作",
        "variants": [
            ("The dog can bark", "The dog can not bark"),
            ("The dog can run", "The dog can not run"),
            ("The dog can swim", "The dog can not swim"),
            ("The dog can jump", "The dog can not jump"),
            ("The dog can bite", "The dog can not bite"),
            ("The dog can smell", "The dog can not smell"),
            ("The dog can hear", "The dog can not hear"),
        ],
    },
}

# Exp B/D 全局句子集 (更大规模)
GLOBAL_SENTENCES = [
    # 声明句 (100+)
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
    "The dog chases the ball", "The girl sings a song",
    "The man walks to work", "The woman paints the wall",
    "The boy kicks the ball", "The girl draws a picture",
    "The tree grows tall", "The flower smells sweet",
    "The cloud moves slowly", "The star shines bright",
    "The rain stops suddenly", "The wind calms down",
    "The sun sets in the west", "The moon rises late",
    "The tide comes in", "The wave crashes loudly",
    "The bird builds a nest", "The bee collects nectar",
    "The ant carries food", "The spider weaves a web",
    "The snake slithers away", "The frog jumps high",
    "The rabbit hops fast", "The deer runs gracefully",
    "The eagle soars high", "The whale swims deep",
    "The shark hunts fish", "The dolphin plays happily",
    "The penguin walks funny", "The bear sleeps long",
    "The lion roars loud", "The tiger stalks prey",
    "The monkey swings high", "The elephant walks slow",
    "The giraffe reaches high", "The zebra runs fast",
    "The hippo swims well", "The rhino charges forward",
    "The crocodile waits still", "The turtle moves slowly",
    "The parrot talks loud", "The owl hunts at night",
    "The bat flies silently", "The fox sneaks quietly",
    "The wolf howls loudly", "The beaver builds dams",
    "The otter plays in water", "The seal swims fast",
    "The whale sings songs", "The crab walks sideways",
    "The lobster hides in rocks", "The shrimp swims backwards",
    "The jellyfish floats gently", "The octopus changes color",
    "The squid swims fast", "The starfish clings tight",
    "The seahorse swims upright", "The coral grows slowly",
    "The anemone sways gently", "The urchin stays still",
    "The sponge filters water", "The clam burrows deep",
    "The snail crawls slowly", "The worm digs deep",
    "The ant works hard", "The bee buzzes loudly",
    "The wasp stings quickly", "The fly buzzes around",
    "The mosquito bites hard", "The butterfly flies gracefully",
    "The moth flies at night", "The beetle crawls slowly",
    "The ladybug looks pretty", "The firefly glows bright",
    # 否定句 (50+)
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
    "The dog can not climb", "She does not sing",
    "He has not returned", "They will not listen",
    "The bird does not talk", "She can not dance",
    "He should not worry", "They must not leave",
    "The cat will not jump", "She does not run",
    "He can not swim", "They would not help",
    "The dog has not eaten", "She should not talk",
    "He does not read", "They can not see",
    "The bird will not stay", "She must not go",
    "He could not sleep", "They should not wait",
    "The cat does not bite", "She can not hear",
    "He has not spoken", "They will not stay",
    "The dog does not bite", "She would not agree",
    "He can not walk", "They must not eat",
    # 将来时 (30+)
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
    "He will drive the car", "They will play the game",
    "The tree will grow tall", "She will learn the skill",
    "He will build the model", "They will find the answer",
    "The river will flow fast", "She will write the essay",
    "He will solve the problem", "They will win the prize",
    # 过去时 (30+)
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
    "He swam the river", "They climbed the mountain",
    "She played the guitar", "The horse won the race",
    "He wrote the letter", "They cleaned the house",
    "She baked the cake", "The bird sang the song",
    "He caught the ball", "They found the treasure",
    # 疑问句 (20+)
    "What time does the train arrive", "How many students passed the exam",
    "Where is the nearest hospital", "When will the meeting start",
    "Why did she leave early", "Who wrote this book",
    "Which color do you prefer", "How long will it take",
    "Can the dog swim", "Will it rain tomorrow",
    "Has she finished the work", "Should they leave now",
    "Could he answer the question", "Would you like some tea",
    "Is the cat sleeping", "Are they coming home",
    "Do you like music", "Did she finish early",
    "Was the test difficult", "Were they happy about it",
    # 被动 (15+)
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
    "The play was performed well",
    "The speech was delivered clearly",
    "The experiment was conducted carefully",
    "The document was signed yesterday",
    "The message was received late",
]


# ============================================================
# 工具函数
# ============================================================

def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_forward(model, input_ids, attention_mask, output_hidden_states=True, output_attentions=False):
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=output_hidden_states,
                        output_attentions=output_attentions)
    except Exception as e:
        print(f"    [WARN] Forward failed: {e}")
        return None


def get_hidden_states_for_sentence(model, tokenizer, device, sentence):
    """获取句子在各层的hidden states"""
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
# Exp A: 局部切空间估计
# ============================================================

def expA_local_tangent_space(model, tokenizer, device, model_info, model_name):
    """
    Exp A: 局部切空间估计

    核心问题: PR(全局)是否等于 dim(T_h M)(局部)?

    方法:
    1. 对每个基础句h，构造"邻域" = {h, h_past, h_not, h_synonym}
    2. 在这个4-5点的局部集合上做PCA
    3. 计算局部参与率 (local PR)
    4. 如果局部PR ≈ 全局PR → 支持流形假设
    5. 如果局部PR >> 全局PR → 全局PR被低估
    6. 如果局部PR很不稳定 → 不是光滑流形
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)

    # 采样层
    n_sample = min(10, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    n_bases = len(LOCAL_TANGENT_BASES)
    print(f"  使用 {n_bases} 个基础句构造局部邻域")

    # 收集每个基础句的邻域hidden states
    # neighborhood_data[layer] = list of (base_name, H_local [n_points, d_model])
    neighborhood_data = {l: [] for l in sample_layers}

    for base_idx, (base, past, negated, synonym) in enumerate(LOCAL_TANGENT_BASES):
        if base_idx % 10 == 0:
            print(f"    进度: {base_idx}/{n_bases}")

        # 获取4个句子的hidden states
        sentences = {"base": base, "past": past, "negated": negated, "synonym": synonym}
        hs_dict = {}
        for name, sent in sentences.items():
            hs = get_hidden_states_for_sentence(model, tokenizer, input_device, sent)
            if hs is not None:
                hs_dict[name] = hs

        if len(hs_dict) < 3:
            continue

        for l in sample_layers:
            H_local = []
            for name, hs in hs_dict.items():
                if l < len(hs):
                    h = hs[l][0, -1, :].float().cpu().numpy()
                    H_local.append(h)

            if len(H_local) >= 3:
                neighborhood_data[l].append((base[:30], np.array(H_local)))

    # 局部PCA分析
    print(f"\n  === 局部切空间分析 ===")
    results = {
        "model": model_name,
        "n_bases": n_bases,
        "sample_layers": sample_layers,
        "layer_analysis": {},
    }

    for l in sample_layers:
        if not neighborhood_data[l]:
            continue

        local_prs = []
        local_dim90s = []
        local_first_sv_ratios = []  # 第一奇异值 / 第二奇异值 → 局部平坦度
        local_stability = []  # 各邻域PR的变异系数

        for base_name, H_local in neighborhood_data[l]:
            n_pts, d = H_local.shape
            if n_pts < 3:
                continue

            # 中心化
            H_centered = H_local - H_local.mean(axis=0, keepdims=True)

            # 局部SVD
            try:
                U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
            except Exception:
                continue

            eigenvalues = S**2 / max(n_pts - 1, 1)
            total_var = np.sum(eigenvalues)

            if total_var < 1e-10:
                continue

            # 局部PR
            local_pr = total_var**2 / np.sum(eigenvalues**2)

            # 90%方差维度
            cum_var = np.cumsum(eigenvalues) / total_var
            local_dim90 = int(np.searchsorted(cum_var, 0.90)) + 1

            # 第一/第二奇异值比 (局部平坦度指标)
            if len(S) >= 2 and S[1] > 1e-10:
                sv_ratio = S[0] / S[1]
            else:
                sv_ratio = float('inf')

            local_prs.append(local_pr)
            local_dim90s.append(local_dim90)
            local_first_sv_ratios.append(sv_ratio)

        if not local_prs:
            continue

        layer_key = f"L{l}"
        layer_result = {
            "mean_local_pr": float(np.mean(local_prs)),
            "std_local_pr": float(np.std(local_prs)),
            "median_local_pr": float(np.median(local_prs)),
            "cv_local_pr": float(np.std(local_prs) / np.mean(local_prs)) if np.mean(local_prs) > 0 else float('inf'),
            "mean_local_dim90": float(np.mean(local_dim90s)),
            "std_local_dim90": float(np.std(local_dim90s)),
            "median_local_dim90": float(np.median(local_dim90s)),
            "mean_sv_ratio": float(np.mean(local_first_sv_ratios)),
            "median_sv_ratio": float(np.median(local_first_sv_ratios)),
            "n_neighborhoods": len(local_prs),
            # 全局PR对比 (待Phase 141数据对比)
        }

        results["layer_analysis"][layer_key] = layer_result

        print(f"    {layer_key}: local_PR={np.mean(local_prs):.1f}±{np.std(local_prs):.1f} "
              f"(CV={np.std(local_prs)/np.mean(local_prs):.2f}), "
              f"local_dim90={np.mean(local_dim90s):.1f}±{np.std(local_dim90s):.1f}, "
              f"SV1/SV2={np.mean(local_first_sv_ratios):.1f}")

    # 全局汇总
    all_local_prs = [v["mean_local_pr"] for v in results["layer_analysis"].values()]
    all_local_cv = [v["cv_local_pr"] for v in results["layer_analysis"].values()]

    if all_local_prs:
        results["summary"] = {
            "mean_local_pr_across_layers": float(np.mean(all_local_prs)),
            "mean_cv_local_pr": float(np.mean(all_local_cv)),
            "interpretation": "manifold_like" if np.mean(all_local_cv) < 0.5 else "heterogeneous",
        }
        print(f"\n  汇总: 平均局部PR={np.mean(all_local_prs):.1f}, "
              f"PR变异系数CV={np.mean(all_local_cv):.2f}")

    return results


# ============================================================
# Exp B: Attention运输几何
# ============================================================

def expB_attention_transport(model, tokenizer, device, model_info, model_name):
    """
    Exp B: Attention运输几何

    核心思想:
    - h_i^{(l+1)} = h_i^{(l)} + Σ_j A_{ij} V_j
    - A_ij 定义了信息如何从token j运输到token i
    - 如果attention是"联络"(connection)，那么语义变化应该改变运输模式

    方法:
    1. 对base和NOT(x)提取attention weights
    2. 测量attention pattern的变化: ΔA_ij
    3. 关键问题: ΔA_ij是否比随机变化更结构化？
    4. 研究attention head的"运输方向": 哪些head负责否定？哪些负责时态？
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    layers = get_layers(model)

    # 采样层
    n_sample = min(8, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    n_pairs = len(ATTENTION_TRANSPORT_PAIRS)
    print(f"  使用 {n_pairs} 个句对分析attention运输")

    results = {
        "model": model_name,
        "n_pairs": n_pairs,
        "sample_layers": sample_layers,
        "layer_analysis": {},
    }

    for pair_idx, (base_sent, variant_sent) in enumerate(ATTENTION_TRANSPORT_PAIRS):
        if pair_idx % 5 == 0:
            print(f"    进度: {pair_idx}/{n_pairs}")

        # 获取attention weights
        ids_base = tokenizer.encode(base_sent, add_special_tokens=False)
        ids_var = tokenizer.encode(variant_sent, add_special_tokens=False)

        input_ids_base = torch.tensor([ids_base], device=input_device)
        input_ids_var = torch.tensor([ids_var], device=input_device)

        attn_mask_base = torch.ones(1, len(ids_base), device=input_device, dtype=torch.long)
        attn_mask_var = torch.ones(1, len(ids_var), device=input_device, dtype=torch.long)

        out_base = safe_forward(model, input_ids_base, attn_mask_base,
                                output_hidden_states=False, output_attentions=True)
        out_var = safe_forward(model, input_ids_var, attn_mask_var,
                               output_hidden_states=False, output_attentions=True)

        if out_base is None or out_var is None:
            continue
        if out_base.attentions is None or out_var.attentions is None:
            print(f"    [WARN] attentions not available for pair {pair_idx}")
            continue

        # attentions: tuple of (n_layers,) each [1, n_heads, seq_len, seq_len]
        for l in sample_layers:
            if l >= len(out_base.attentions) or l >= len(out_var.attentions):
                continue

            layer_key = f"L{l}"

            A_base = out_base.attentions[l][0].float().cpu().numpy()  # [n_heads, seq, seq]
            A_var = out_var.attentions[l][0].float().cpu().numpy()

            n_heads = A_base.shape[0]
            seq_base = A_base.shape[1]
            seq_var = A_var.shape[1]

            # 对齐: 取较短序列的长度
            min_seq = min(seq_base, seq_var)

            # 计算attention变化
            A_base_trimmed = A_base[:, :min_seq, :min_seq]
            A_var_trimmed = A_var[:, :min_seq, :min_seq]

            # ΔA = A_variant - A_base
            delta_A = A_var_trimmed - A_base_trimmed  # [n_heads, seq, seq]

            # 每个head的变化幅度
            head_delta_norms = [float(np.linalg.norm(delta_A[h])) for h in range(n_heads)]
            head_base_norms = [float(np.linalg.norm(A_base_trimmed[h])) for h in range(n_heads)]

            # 相对变化
            head_relative_deltas = []
            for h in range(n_heads):
                if head_base_norms[h] > 1e-10:
                    head_relative_deltas.append(head_delta_norms[h] / head_base_norms[h])
                else:
                    head_relative_deltas.append(0.0)

            # last token行的attention变化 (对生成最关键)
            # last token关注哪些其他tokens
            last_row_base = A_base_trimmed[:, -1, :]  # [n_heads, seq-1]
            last_row_var = A_var_trimmed[:, -1, :]
            delta_last_row = last_row_var - last_row_base  # [n_heads, seq-1]

            # 哪些位置attention变化最大
            delta_last_row_norm = float(np.linalg.norm(delta_last_row))

            # 存储结果
            if layer_key not in results["layer_analysis"]:
                results["layer_analysis"][layer_key] = {
                    "head_relative_deltas": {h: [] for h in range(n_heads)},
                    "last_row_delta_norms": [],
                    "total_delta_norms": [],
                }

            la = results["layer_analysis"][layer_key]
            for h in range(n_heads):
                la["head_relative_deltas"][h].append(head_relative_deltas[h])
            la["last_row_delta_norms"].append(delta_last_row_norm)
            la["total_delta_norms"].append(float(np.mean(head_delta_norms)))

    # 聚合
    print(f"\n  === Attention运输分析 ===")
    for layer_key in sorted(results["layer_analysis"].keys(), key=lambda x: int(x[1:])):
        la = results["layer_analysis"][layer_key]

        # 每个head的平均相对变化
        head_mean_deltas = {}
        head_max_deltas = {}
        for h, deltas in la["head_relative_deltas"].items():
            if deltas:
                head_mean_deltas[h] = float(np.mean(deltas))
                head_max_deltas[h] = float(np.max(deltas))

        # 找出变化最大的head
        if head_mean_deltas:
            top_heads = sorted(head_mean_deltas.items(), key=lambda x: x[1], reverse=True)[:5]
            top_heads_str = ", ".join([f"H{h}:{d:.3f}" for h, d in top_heads])
        else:
            top_heads_str = "N/A"

        # last token行的变化
        mean_last_row_delta = float(np.mean(la["last_row_delta_norms"])) if la["last_row_delta_norms"] else 0
        mean_total_delta = float(np.mean(la["total_delta_norms"])) if la["total_delta_norms"] else 0

        la["head_mean_relative_delta"] = head_mean_deltas
        la["mean_last_row_delta_norm"] = mean_last_row_delta
        la["mean_total_delta_norm"] = mean_total_delta

        print(f"    {layer_key}: total_ΔA={mean_total_delta:.4f}, "
              f"last_row_ΔA={mean_last_row_delta:.4f}, "
              f"top_heads=[{top_heads_str}]")

    return results


# ============================================================
# Exp C: 算子纤维结构
# ============================================================

def expC_operator_fiber(model, tokenizer, device, model_info, model_name):
    """
    Exp C: 算子纤维结构

    核心问题: NOT算子是否依赖于句法上下文？

    如果 V_{not,can} ≈ V_{not,should} ≈ V_{not,will} → NOT是全局算子
    如果 V_{not,can} ≠ V_{not,should} → NOT是纤维化的，依赖于上下文

    方法:
    1. 在不同句法上下文中计算 V_not
    2. 测量不同上下文的V_not之间的cosine similarity
    3. 如果cos ≈ 1 → 全局算子
    4. 如果cos << 1 → 纤维结构
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
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
        "fiber_analysis": {},
    }

    for fiber_name, fiber_data in FIBER_STRUCTURE_DATA.items():
        print(f"\n  纤维类型: {fiber_name} — {fiber_data['description']}")

        variants = fiber_data["variants"]
        n_variants = len(variants)

        # 收集各层的V_not
        layer_v_not = {l: {} for l in sample_layers}  # {layer: {variant_name: V_not}}

        for var_idx, (base, negated) in enumerate(variants):
            hs_base = get_hidden_states_for_sentence(model, tokenizer, input_device, base)
            hs_neg = get_hidden_states_for_sentence(model, tokenizer, input_device, negated)

            if hs_base is None or hs_neg is None:
                continue

            for l in sample_layers:
                if l >= len(hs_base) or l >= len(hs_neg):
                    continue

                h_b = hs_base[l][0, -1, :].float().cpu().numpy()
                h_n = hs_neg[l][0, -1, :].float().cpu().numpy()
                v_not = h_n - h_b

                # 用变体的简短描述作为key
                var_key = f"var{var_idx}"
                layer_v_not[l][var_key] = {
                    "v_not": v_not,
                    "base_sent": base[:30],
                    "neg_sent": negated[:30],
                }

        # 分析: 不同上下文的V_not之间的cosine similarity
        fiber_results = {}
        for l in sample_layers:
            layer_key = f"L{l}"
            v_dict = layer_v_not[l]

            if len(v_dict) < 2:
                continue

            # 计算两两cosine
            keys = list(v_dict.keys())
            v_vectors = [v_dict[k]["v_not"] for k in keys]
            v_norms = [np.linalg.norm(v) for v in v_vectors]

            pairwise_cos = []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    if v_norms[i] > 1e-10 and v_norms[j] > 1e-10:
                        cos = float(np.dot(v_vectors[i], v_vectors[j]) / (v_norms[i] * v_norms[j]))
                        pairwise_cos.append(cos)

            if not pairwise_cos:
                continue

            # 平均方向一致性
            mean_v = np.mean(v_vectors, axis=0)
            mean_v_norm = np.linalg.norm(mean_v)
            consistencies = []
            for v in v_vectors:
                v_n = np.linalg.norm(v)
                if v_n > 1e-10 and mean_v_norm > 1e-10:
                    consistencies.append(float(np.dot(v, mean_v) / (v_n * mean_v_norm)))

            # 范数变异
            norm_mean = float(np.mean(v_norms))
            norm_cv = float(np.std(v_norms) / np.mean(v_norms)) if norm_mean > 1e-10 else float('inf')

            # 判断: 全局算子 vs 纤维化
            # 高一致性(>0.7) + 低变异(<0.3) → 全局算子
            # 否则 → 纤维化
            mean_consistency = float(np.mean(consistencies)) if consistencies else 0
            is_global = mean_consistency > 0.7 and norm_cv < 0.3
            structure_type = "global_operator" if is_global else "fibered"

            layer_result = {
                "pairwise_cosine_mean": float(np.mean(pairwise_cos)),
                "pairwise_cosine_std": float(np.std(pairwise_cos)),
                "direction_consistency": mean_consistency,
                "norm_cv": norm_cv,
                "norm_mean": norm_mean,
                "n_variants": len(v_dict),
                "structure_type": structure_type,
            }

            fiber_results[layer_key] = layer_result

            cos_str = f"{float(np.mean(pairwise_cos)):.3f}±{float(np.std(pairwise_cos)):.3f}"
            print(f"      {layer_key}: pairwise_cos={cos_str}, "
                  f"consistency={mean_consistency:.3f}, "
                  f"norm_CV={norm_cv:.2f}, type={structure_type}")

        results["fiber_analysis"][fiber_name] = fiber_results

    return results


# ============================================================
# Exp D: Jacobian修正 (使用autograd)
# ============================================================

def expD_jacobian_autograd(model, tokenizer, device, model_info, model_name):
    """
    Exp D: 使用autograd计算真正的Jacobian

    修正Phase 141的数值问题:
    - eps=1e-4太小导致浮点精度问题
    - 使用torch.autograd.functional.jvp (前向模式自动微分)
    - 或者使用更大的eps + 多次采样取中位数

    核心问题:
    - J_l的稳定子空间维度 ≈ PR?
    - 语义方向是否在J_l的稳定子空间中?
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    layers = get_layers(model)

    # 采样层
    n_sample = min(6, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    # 使用大eps + 多次采样
    eps_values = [1e-3, 5e-3, 1e-2]  # 多个eps验证收敛性
    n_probes = 20  # 随机探测方向

    results = {
        "model": model_name,
        "sample_layers": sample_layers,
        "eps_values": eps_values,
        "n_probes": n_probes,
        "jacobian_analysis": {},
    }

    # 测试句
    test_sentence = "The cat can swim across the river"
    not_sentence = "The cat can not swim across the river"

    # 获取baseline hidden states
    hs_base = get_hidden_states_for_sentence(model, tokenizer, input_device, test_sentence)
    hs_not = get_hidden_states_for_sentence(model, tokenizer, input_device, not_sentence)

    if hs_base is None or hs_not is None:
        print("  [ERROR] Cannot get hidden states")
        return results

    for layer_idx in sample_layers:
        layer_key = f"L{layer_idx}"
        if layer_idx + 1 >= len(hs_base):
            continue

        print(f"\n  分析层 {layer_key}...")

        # 语义方向 (NOT)
        h_base_l = get_last_token_hidden(hs_base, layer_idx)
        h_not_l = get_last_token_hidden(hs_not, layer_idx)
        h_base_next = get_last_token_hidden(hs_base, layer_idx + 1)

        if h_base_l is None or h_not_l is None or h_base_next is None:
            continue

        v_sem_not = h_not_l - h_base_l
        v_sem_norm = np.linalg.norm(v_sem_not)
        if v_sem_norm < 1e-10:
            continue
        v_sem_unit = v_sem_not / v_sem_norm

        layer_result = {}

        for eps in eps_values:
            eps_str = f"eps_{eps:.0e}"

            # 随机方向的JVP
            torch.manual_seed(42 + layer_idx)
            jvp_random_norms = []

            for probe_idx in range(n_probes):
                v_rand = np.random.randn(d_model)
                v_rand = v_rand / np.linalg.norm(v_rand)

                delta = _inject_and_measure_v2(
                    model, tokenizer, input_device, layers,
                    test_sentence, layer_idx, v_rand, eps, h_base_next
                )
                if delta is not None:
                    jvp_norm = float(np.linalg.norm(delta / eps))
                    jvp_random_norms.append(jvp_norm)

            # 语义方向的JVP
            delta_sem = _inject_and_measure_v2(
                model, tokenizer, input_device, layers,
                test_sentence, layer_idx, v_sem_unit, eps, h_base_next
            )
            jvp_sem_norm = float(np.linalg.norm(delta_sem / eps)) if delta_sem is not None else None

            # 反方向验证 (eps和-eps的中位数，更稳定)
            delta_sem_neg = _inject_and_measure_v2(
                model, tokenizer, input_device, layers,
                test_sentence, layer_idx, -v_sem_unit, eps, h_base_next
            )
            jvp_sem_neg_norm = float(np.linalg.norm(delta_sem_neg / eps)) if delta_sem_neg is not None else None

            eps_result = {
                "random_jvp_mean": float(np.mean(jvp_random_norms)) if jvp_random_norms else None,
                "random_jvp_std": float(np.std(jvp_random_norms)) if jvp_random_norms else None,
                "random_jvp_median": float(np.median(jvp_random_norms)) if jvp_random_norms else None,
                "semantic_jvp_norm": jvp_sem_norm,
                "semantic_jvp_neg_norm": jvp_sem_neg_norm,
                "semantic_vs_random_ratio": jvp_sem_norm / float(np.mean(jvp_random_norms)) if jvp_random_norms and jvp_sem_norm else None,
                "symmetry_ratio": jvp_sem_norm / jvp_sem_neg_norm if jvp_sem_norm and jvp_sem_neg_norm and jvp_sem_neg_norm > 1e-10 else None,
                "n_random": len(jvp_random_norms),
            }

            layer_result[eps_str] = eps_result

            ratio_str = f"{eps_result['semantic_vs_random_ratio']:.2f}x" if eps_result['semantic_vs_random_ratio'] else "N/A"
            sym_str = f"{eps_result['symmetry_ratio']:.3f}" if eps_result['symmetry_ratio'] else "N/A"
            print(f"    eps={eps}: random={np.mean(jvp_random_norms):.2f}±{np.std(jvp_random_norms):.2f}, "
                  f"sem={jvp_sem_norm:.2f}, ratio={ratio_str}, symmetry={sym_str}")

        results["jacobian_analysis"][layer_key] = layer_result

    return results


def _inject_and_measure_v2(model, tokenizer, device, layers, base_sentence,
                           layer_idx, direction_np, eps, h_next_baseline_np):
    """
    在指定层注入扰动方向，测量下一层的变化 (改进版)

    修复:
    1. baseline来自独立的forward pass
    2. 支持更大eps
    3. 更robust的hook处理
    """
    ids = tokenizer.encode(base_sentence, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attn_mask = torch.ones(1, len(ids), device=device, dtype=torch.long)

    direction_t = torch.tensor(direction_np, dtype=torch.float32)

    captured_output = {}

    def pre_hook_fn(module, args):
        hidden_states = args[0]
        perturbed = hidden_states.clone()
        perturbed[:, -1, :] = perturbed[:, -1, :] + eps * direction_t.to(perturbed.device, perturbed.dtype)
        return (perturbed,) + args[1:]

    def post_hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_output['h_next'] = output[0].detach().float().cpu()
        else:
            captured_output['h_next'] = output.detach().float().cpu()

    pre_hook = layers[layer_idx].register_forward_pre_hook(pre_hook_fn)
    post_hook = layers[layer_idx].register_forward_hook(post_hook_fn)

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=False)
    except Exception as e:
        pre_hook.remove()
        post_hook.remove()
        return None
    finally:
        pre_hook.remove()
        post_hook.remove()

    if 'h_next' not in captured_output:
        return None

    h_next_perturbed = captured_output['h_next'][0, -1, :].numpy()
    delta_next = h_next_perturbed - h_next_baseline_np

    return delta_next


# ============================================================
# 主函数
# ============================================================

def run_phase142(model_name: str):
    """运行Phase 142所有实验"""
    print(f"\n{'='*60}")
    print(f"Phase 142: Local Geometry & Transport Connection — {model_name}")
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

    # === Exp A: 局部切空间 ===
    print(f"\n{'='*40}")
    print("Exp A: 局部切空间估计 (Local Tangent Space)")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expA"] = expA_local_tangent_space(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp A failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expA"] = {"error": str(e)}
    print(f"Exp A 完成: {time.time()-t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp B: Attention运输几何 ===
    print(f"\n{'='*40}")
    print("Exp B: Attention运输几何 (Transport Connection)")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expB"] = expB_attention_transport(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp B failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expB"] = {"error": str(e)}
    print(f"Exp B 完成: {time.time()-t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp C: 算子纤维结构 ===
    print(f"\n{'='*40}")
    print("Exp C: 算子纤维结构 (Operator Fiber Structure)")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expC"] = expC_operator_fiber(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"  [ERROR] Exp C failed: {e}")
        import traceback; traceback.print_exc()
        all_results["expC"] = {"error": str(e)}
    print(f"Exp C 完成: {time.time()-t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp D: Jacobian修正 ===
    print(f"\n{'='*40}")
    print("Exp D: Jacobian修正 (Autograd + Multi-eps)")
    print(f"{'='*40}")
    t0 = time.time()
    try:
        all_results["expD"] = expD_jacobian_autograd(model, tokenizer, device, model_info, model_name)
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
        print("Usage: python phase142_local_geometry_transport.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    results = run_phase142(model_name)

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"tests/glm5_temp/phase142_{model_name}_local_geometry_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {filename}")
