"""
Phase 60: 因果验证 — 解决Phase 59核心漏洞
============================================

四个方案，按优先级：
  方案1: 共现频率对照实验 — 区分"编码机制"vs"共现统计"
  方案2: 强度维度直接提取 — 验证"方向×强度"分解假设
  方案3: 概念区分能力验证 — 验证top-k子空间的有效性
  方案4: 情感多维度分析 — 区分"多维编码"vs"编码不稳定"

用法:
  python tests/glm5/phase60_causal_validation.py --model qwen3 --part 1
  python tests/glm5/phase60_causal_validation.py --model qwen3 --part 2
  python tests/glm5/phase60_causal_validation.py --model qwen3 --part 3
  python tests/glm5/phase60_causal_validation.py --model qwen3 --part 4
  python tests/glm5/phase60_causal_validation.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

RESULT_DIR = PROJECT / "results" / "subspace_topology"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)


# =====================================================================
# 模型加载 (BF16 + device_map="auto" + Flash Attention)
# =====================================================================

def load_model_bf16(model_name: str):
    """BF16加载 + device_map=auto + flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash attention, 失败回退eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        log_time(f"{model_name} loaded with flash_attention_2")
    except Exception as e:
        log_time(f"Flash attention failed ({e}), falling back to eager")
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
    log_time(f"{model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


import torch


# =====================================================================
# 通用: 收集hidden states
# =====================================================================

def collect_hidden_states(model, tokenizer, device, sentences, target_layers, batch_size=4):
    """
    收集多个句子在指定层的hidden states (残差流)

    Returns:
        dict: {layer_idx: np.array [n_sentences, d_model]}
    """
    from model_utils import get_model_info
    info = get_model_info(model, model_name_global)

    all_hidden = {li: [] for li in target_layers}

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch_sents, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)

        # 获取输入设备
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        for li in target_layers:
            # 取每个句子的最后一个非padding token
            hs = outputs.hidden_states[li].float().cpu().numpy()  # [batch, seq, d]
            for i in range(len(batch_sents)):
                # 找到最后一个非padding位置
                mask = inputs["attention_mask"][i].numpy()
                last_pos = np.where(mask > 0)[0][-1]
                all_hidden[li].append(hs[i, last_pos])

        if batch_start % (batch_size * 10) == 0:
            log_time(f"  Collected {batch_start + len(batch_sents)}/{len(sentences)}")

    for li in target_layers:
        all_hidden[li] = np.array(all_hidden[li])

    return all_hidden


def compute_overlap(S1, S2):
    """
    计算两个子空间的重叠度 (principal angle based)

    S1, S2: orthonormal basis [d_model, k]
    Returns: float in [0, 1]
    """
    M = S1.T @ S2  # [k, k]
    svals = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(svals ** 2))


def extract_subspace(activations, n_dims=10):
    """从激活矩阵提取top-k子空间 (PCA)"""
    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_dims].T  # [d_model, n_dims] orthonormal basis


# =====================================================================
# 方案1: 共现频率对照实验
# =====================================================================

# 30个词对, 覆盖: 反义词(高overlap预期), 近义词(中overlap), 高共现非语义(控制组),
# 低共现近义词(关键测试), 跨类别(低overlap预期)
COOCCURRENCE_PAIRS = [
    # 反义词对 (预期: 高共现 + 高overlap)
    ("hot", "cold"), ("big", "small"), ("love", "hate"),
    ("light", "dark"), ("good", "bad"), ("fast", "slow"),
    ("up", "down"), ("rich", "poor"), ("strong", "weak"),

    # 近义词对 (预期: 低共现 + 中overlap) — 关键测试!
    ("hot", "warm"), ("big", "large"), ("love", "like"),
    ("fast", "quick"), ("strong", "powerful"), ("good", "nice"),

    # 高共现但无语义关系 (控制组) — 如果共现决定overlap, 这些应有高overlap
    ("hot", "day"), ("cold", "night"), ("big", "city"),
    ("fast", "car"), ("strong", "wind"), ("good", "morning"),

    # 低共现但语义相关 (反向测试) — 如果编码机制独立, 应有中高overlap
    ("hot", "scorching"), ("cold", "freezing"), ("big", "enormous"),
    ("fast", "rapid"), ("strong", "fierce"), ("good", "excellent"),

    # 跨类别 (预期: 低共现 + 低overlap)
    ("hot", "beautiful"), ("cold", "intelligent"), ("big", "honest"),
]

# 每个词的模板 (30个/词, 确保足够样本)
COOCCURRENCE_TEMPLATES = {
    "hot": [
        "The hot coffee burned my tongue", "She turned on the hot water",
        "He enjoys hot weather in summer", "The hot stove was dangerous",
        "This hot sauce is very spicy", "The hot air balloon rose quickly",
        "She took a hot shower this morning", "The hot sun beat down on us",
        "He prefers hot tea over iced", "The hot pavement burned bare feet",
        "A hot breeze blew through the window", "The hot springs were relaxing",
        "She ordered a hot meal at the restaurant", "The hot chocolate was perfect",
        "He touched the hot iron accidentally", "The hot debate continued for hours",
        "This hot topic is trending online", "She felt hot and bothered",
        "The hot rod raced down the street", "The hot day made everyone tired",
        "He sold the hot item quickly", "The hot pan sizzled with oil",
        "She wiped her hot forehead", "The hot fire crackled loudly",
        "The hot bath was soothing", "He opened the hot oven carefully",
        "The hot sand burned their feet", "She drank the hot milk slowly",
        "The hot climate was hard to adapt to", "He touched the hot surface",
    ],
    "cold": [
        "The cold wind chilled my bones", "She turned on the cold water",
        "He hates cold weather in winter", "The cold ice was slippery",
        "This cold drink is very refreshing", "The cold front moved in quickly",
        "She took a cold shower this morning", "The cold moon shone on us",
        "He prefers cold juice over hot", "The cold ground froze bare feet",
        "A cold breeze blew through the window", "The cold winter was harsh",
        "She ordered a cold meal at the deli", "The cold lemonade was perfect",
        "He touched the cold metal carefully", "The cold war lasted for decades",
        "This cold case was finally solved", "She felt cold and shivering",
        "The cold storage preserved the food", "The cold day made everyone bundle up",
        "He caught a cold last week", "The cold facts were undeniable",
        "She gave a cold stare", "The cold rain soaked through",
        "The cold bath was invigorating", "He opened the cold refrigerator",
        "The cold snow covered their feet", "She rubbed her cold hands",
        "The cold climate was difficult to endure", "He touched the cold surface",
    ],
    "warm": [
        "The warm blanket felt cozy", "She enjoyed the warm summer breeze",
        "He gave me a warm welcome", "The warm water was soothing",
        "A warm smile appeared on her face", "The warm sunlight streamed through",
        "She made a warm cup of tea", "The warm weather was pleasant",
        "He has a warm personality", "The warm colors looked beautiful",
        "She wrapped the baby in warm clothes", "A warm feeling spread through her",
        "The warm oven filled the kitchen", "He sent a warm greeting",
        "The warm jacket kept him comfortable", "She prepared a warm meal",
        "The warm sand felt nice underfoot", "He offered a warm handshake",
        "A warm glow filled the room", "The warm bread smelled delicious",
        "She preferred warm tones in decoration", "The warm breeze carried flowers",
        "He gave a warm hug to his friend", "The warm milk helped her sleep",
        "A warm atmosphere filled the house", "The warm soup was comforting",
        "She found a warm spot by the fire", "He spoke in a warm voice",
        "The warm evening was perfect for walking", "She appreciated his warm support",
    ],
    "scorching": [
        "The scorching heat was unbearable", "She walked on the scorching sand",
        "A scorching sun beat down all day", "The scorching temperature broke records",
        "He faced the scorching desert wind", "The scorching pavement was dangerous",
        "She complained about the scorching weather", "The scorching flames consumed everything",
        "A scorching review was published", "The scorching summer lasted months",
        "He endured the scorching conditions bravely", "The scorching asphalt softened",
        "She applied sunscreen against the scorching rays", "The scorching commentary went viral",
        "A scorching wind blew from the south", "The scorching oven was too hot",
        "He gave a scorching speech", "The scorching surface burned to touch",
        "She survived the scorching marathon", "The scorching criticism was harsh",
        "A scorching day at the beach", "The scorching attack was unexpected",
        "He received a scorching rebuke", "The scorching iron glowed red",
        "She described the scorching ordeal", "The scorching spotlight was intense",
        "A scorching breeze offered no relief", "The scorching pan seared the steak",
        "He made a scorching comeback", "The scorching landscape was desolate",
    ],
    "big": [
        "The big house loomed over the street", "She made a big decision yesterday",
        "He has a big family with six kids", "The big dog scared the children",
        "This is a really big problem", "The big screen showed the movie",
        "She won a big prize at the fair", "The big storm damaged the roof",
        "He took a big step forward", "The big city never sleeps",
        "A big crowd gathered downtown", "The big news spread quickly",
        "She has big plans for the future", "The big boat sailed away",
        "That was a big mistake", "His big dream finally came true",
        "The big mountain was covered in snow", "She wore a big hat to the beach",
        "A big change happened last week", "The big question remains unanswered",
        "He made a big impression on the team", "The big tree provided shade",
        "She ordered a big portion of pasta", "A big gap appeared in the data",
        "The big moment finally arrived", "He scored a big victory in court",
        "The big picture reveals hidden patterns", "She made a big contribution",
        "A big difference was noticed immediately", "The big challenge is yet to come",
    ],
    "small": [
        "The small cottage was charming", "She made a small adjustment",
        "He has a small garden behind the house", "The small kitten was adorable",
        "This is a small issue we can fix", "The small screen was hard to read",
        "She won a small award at the show", "The small rain barely wet the ground",
        "He took a small step backward", "The small town was peaceful",
        "A small group gathered quietly", "The small detail was overlooked",
        "She has small hopes for the outcome", "The small boat rocked gently",
        "That was a small error", "His small effort was appreciated",
        "The small hill was easy to climb", "She wore a small bracelet",
        "A small change made a difference", "The small print was hard to see",
        "He made a small suggestion", "The small flower was delicate",
        "She ordered a small coffee", "A small crack appeared in the wall",
        "The small moment passed quickly", "He scored a small victory",
        "The small detail mattered most", "She made a small donation",
        "A small improvement was visible", "The small task was simple",
    ],
    "large": [
        "The large house loomed over the street", "She made a large purchase yesterday",
        "He has a large family with six kids", "The large dog scared the children",
        "This is a really large problem", "The large screen showed the movie",
        "She won a large prize at the fair", "The large storm damaged the roof",
        "He took a large step forward", "The large city never sleeps",
        "A large crowd gathered downtown", "The large order was shipped",
        "She has large plans for the future", "The large boat sailed away",
        "That was a large error", "His large dream finally came true",
        "The large mountain was covered in snow", "She wore a large hat to the beach",
        "A large change happened last week", "The large question remains unanswered",
        "He made a large impression on the team", "The large tree provided shade",
        "She ordered a large portion of pasta", "A large gap appeared in the data",
        "The large moment finally arrived", "He scored a large victory in court",
        "The large picture reveals hidden patterns", "She made a large contribution",
        "A large difference was noticed immediately", "The large challenge is yet to come",
    ],
    "enormous": [
        "The enormous building dominated the skyline", "She made an enormous effort",
        "He has an enormous appetite", "The enormous elephant charged forward",
        "This enormous task took months", "The enormous screen was impressive",
        "She won an enormous fortune", "The enormous storm was devastating",
        "He took an enormous risk", "The enormous city sprawled endlessly",
        "An enormous crowd filled the stadium", "The enormous project was ambitious",
        "She has enormous potential", "The enormous ship sailed the ocean",
        "That was an enormous failure", "His enormous ego was obvious",
        "The enormous mountain was intimidating", "She wore an enormous ring",
        "An enormous change transformed the company", "The enormous debt was overwhelming",
        "He made an enormous contribution", "The enormous tree was centuries old",
        "She ordered an enormous cake", "An enormous gap divided the groups",
        "The enormous pressure was unbearable", "He achieved enormous success",
        "The enormous scale was unprecedented", "She made an enormous discovery",
        "An enormous responsibility fell on him", "The enormous challenge inspired them",
    ],
    "love": [
        "She loves her family deeply", "He fell in love at first sight",
        "The love story touched everyone", "She expressed her love through art",
        "He wrote a love letter to her", "The love between them was obvious",
        "She found love in unexpected places", "A mother love is unconditional",
        "He discovered his love for music", "The love of learning drives him",
        "She felt love radiating from the crowd", "He showed love through actions",
        "The love song played on the radio", "She received love from her community",
        "His love for adventure never faded", "The love poem was beautiful",
        "She cherished the love of her grandmother", "He pursued love relentlessly",
        "Love conquers all obstacles", "The love festival was celebrated widely",
        "She shared her love of nature", "He returned her love gratefully",
        "The love of his life appeared", "She dedicated her work to love",
        "His love burned brightly for years", "The love scene was moving",
        "She whispered words of love", "He lost his love too soon",
        "The love match was perfect", "She embraced the love around her",
    ],
    "hate": [
        "She hates unfairness with passion", "He could not hide his hate",
        "The hate speech was condemned", "She expressed hate toward injustice",
        "He wrote about hate in society", "The hate between them was obvious",
        "She found hate in unexpected places", "A heart full of hate is heavy",
        "He discovered his hate for cruelty", "The hate of ignorance drives him",
        "She felt hate radiating from the crowd", "He showed hate through anger",
        "The hate rally was disturbing", "She received hate from opponents",
        "His hate for dishonesty never faded", "The hate crime was prosecuted",
        "She witnessed the hate in his eyes", "He pursued hate relentlessly",
        "Hate destroys all reasoning", "The hate group was monitored",
        "She shared her hate of corruption", "He returned hate with forgiveness",
        "The hate of his enemies grew", "She dedicated her work against hate",
        "His hate burned darkly for years", "The hate incident was reported",
        "She whispered words of hate", "He lost his hate over time",
        "The hate conflict escalated", "She fought against hate in society",
    ],
    "like": [
        "She likes reading mystery novels", "He seems like a nice person",
        "The cake tastes like vanilla", "She felt like dancing",
        "He would like to travel more", "They look like twins",
        "She acted like nothing happened", "He sounds like his father",
        "This feels like home", "She walks like a model",
        "He behaves like a gentleman", "The weather is like spring",
        "She smells like roses", "He plays like a professional",
        "This works like magic", "She cooks like a chef",
        "He thinks like a scientist", "The result was like expected",
        "She sings like an angel", "He drives like a racer",
        "This looks like a disaster", "She writes like a poet",
        "He runs like the wind", "The fabric feels like silk",
        "She dresses like a celebrity", "He eats like a horse",
        "This sounds like a plan", "She reacts like her mother",
        "He fights like a warrior", "The plan worked like a charm",
    ],
    "light": [
        "The light was too bright", "She turned on the light",
        "He carried a light suitcase", "The light rain was gentle",
        "This fabric is very light", "A light breeze blew through",
        "She painted the room light blue", "The light meal was satisfying",
        "He made a light joke", "The light was fading quickly",
        "She read by the light of a candle", "A light fog covered the valley",
        "The traffic light turned green", "He has a light complexion",
        "She slept a light sleep", "The light switch was broken",
        "A light snow fell all morning", "He spoke in a light tone",
        "The light reflected off the water", "She felt light headed suddenly",
        "The light at the end of the tunnel", "He wore a light jacket",
        "A light touch was all it took", "The light show was spectacular",
        "She made a light comment", "The light house guided ships",
        "He detected a light vibration", "The light moment broke the tension",
        "She preferred light exercise", "The light fragrance was pleasant",
    ],
    "dark": [
        "The dark room was scary", "She turned off the dark light",
        "He wore a dark suit to the event", "The dark clouds gathered quickly",
        "This chocolate is very dark", "A dark shadow crossed the wall",
        "She painted the room dark green", "The dark secret haunted him",
        "He told a dark joke", "The dark was closing in",
        "She navigated in the dark carefully", "A dark forest surrounded the castle",
        "The dark matter is mysterious", "He has a dark complexion",
        "She had dark thoughts at night", "The dark ages were difficult",
        "A dark night fell over the city", "He spoke in a dark tone",
        "The dark water looked ominous", "She felt a dark presence",
        "The dark side of human nature", "He entered the dark alley cautiously",
        "A dark stain appeared on the carpet", "The dark humor made some laugh",
        "She preferred dark colors", "The dark moon was barely visible",
        "He detected a dark pattern in the data", "The dark moment passed eventually",
        "She avoided dark places", "The dark aroma was intense",
    ],
    "good": [
        "She is a good person", "He had a good idea",
        "The food was really good", "She did a good job",
        "This is good news", "He made a good decision",
        "She gave good advice", "The weather is good today",
        "He has good taste in music", "She found a good solution",
        "The book received good reviews", "He made a good impression",
        "She has good health", "The results were good",
        "He showed good sportsmanship", "She made a good point",
        "This is a good opportunity", "He had a good reason",
        "She set a good example", "The movie was really good",
        "He took a good look around", "She had a good time at the party",
        "The team played a good game", "He gave a good performance",
        "She made a good investment", "This is a good place to start",
        "He has a good memory", "The product is of good quality",
        "She has good intentions", "He chose a good time to visit",
    ],
    "bad": [
        "She had a bad experience", "He made a bad mistake",
        "The weather turned bad quickly", "She felt bad about the outcome",
        "This is bad news for everyone", "He gave a bad excuse",
        "She received bad advice", "The food tasted bad",
        "He has bad habits he needs to break", "She found a bad solution",
        "The review was really bad", "He left a bad impression",
        "She is in bad health lately", "The results were bad",
        "He showed bad behavior in class", "She made a bad argument",
        "This is a bad situation", "He had a bad reason for leaving",
        "She set a bad example", "The movie was really bad",
        "He had a bad day at work", "She had a bad time at the event",
        "The team played a bad game", "He gave a bad performance",
        "She made a bad investment", "This is a bad place to be",
        "He has a bad memory for names", "The product is of bad quality",
        "She has bad intentions", "He chose a bad time to call",
    ],
    "nice": [
        "She is a nice person", "He had a nice idea",
        "The weather was really nice", "She did a nice job",
        "This is a nice surprise", "He made a nice gesture",
        "She gave a nice compliment", "The view is nice from here",
        "He has a nice smile", "She found a nice restaurant",
        "The dress looks nice on you", "He made a nice impression",
        "She has a nice personality", "The results were nice",
        "He was nice to everyone", "She made a nice point",
        "This is a nice opportunity", "He had a nice reason for visiting",
        "She had a nice time at the party", "The movie was really nice",
        "He took a nice walk in the park", "She looked nice in the photo",
        "The team had a nice win", "He gave a nice performance",
        "She made a nice choice", "This is a nice place to visit",
        "He has a nice way of explaining", "The product has nice features",
        "She has nice handwriting", "He chose a nice gift",
    ],
    "fast": [
        "The fast car sped down the highway", "She made a fast decision",
        "He is a fast runner on the track", "The fast train arrived early",
        "This is a fast growing company", "The fast pace was exhausting",
        "She completed the work fast", "The fast food was convenient",
        "He took the fast route to work", "The fast response was appreciated",
        "A fast current carried the boat", "The fast internet connection helped",
        "She has fast reflexes", "The fast clock was wrong",
        "He moved fast through the crowd", "The fast service impressed everyone",
        "She learned fast from her mistakes", "The fast dance was energetic",
        "He typed fast on the keyboard", "The fast lane was busy",
        "A fast heartbeat indicated stress", "The fast break in basketball scored",
        "She spoke fast with excitement", "The fast turnaround was remarkable",
        "He drove fast on the empty road", "The fast development surprised experts",
        "She finished the race fast", "The fast forward button was stuck",
        "He read the book fast", "The fast motion was blurred",
    ],
    "slow": [
        "The slow traffic frustrated everyone", "She made a slow decision",
        "He is a slow walker on the trail", "The slow train was delayed",
        "This is a slow growing plant", "The slow pace was relaxing",
        "She completed the work slowly", "The slow food movement is growing",
        "He took the slow route to enjoy the scenery", "The slow response was frustrating",
        "A slow river meandered through the valley", "The slow internet was annoying",
        "She has slow reflexes sometimes", "The slow clock was behind",
        "He moved slow through the museum", "The slow service disappointed everyone",
        "She learned slow but thoroughly", "The slow dance was romantic",
        "He typed slow but accurately", "The slow lane was nearly empty",
        "A slow heartbeat indicated calm", "The slow waltz was elegant",
        "She spoke slow with emphasis", "The slow recovery took months",
        "He drove slow in the school zone", "The slow progress was discouraging",
        "She finished the race slowly", "The slow motion revealed details",
        "He read the book slowly", "The slow season was quiet",
    ],
    "quick": [
        "She made a quick decision", "He gave a quick response",
        "The quick fix worked temporarily", "She took a quick look around",
        "He had a quick breakfast", "The quick thinking saved the day",
        "She made quick progress on the project", "He gave a quick nod of approval",
        "The quick action prevented disaster", "She took a quick shower",
        "He made a quick exit from the room", "The quick glance revealed the truth",
        "She gave a quick summary of events", "He took a quick nap during lunch",
        "The quick reflexes helped in the game", "She made a quick recovery",
        "He wrote a quick note to himself", "The quick calculation was correct",
        "She gave a quick smile and left", "He made a quick phone call",
        "The quick version was sufficient", "She made a quick assessment",
        "He gave a quick wave goodbye", "The quick handshake was firm",
        "She took a quick breath before speaking", "He made a quick adjustment",
        "The quick review identified the errors", "She gave a quick answer",
        "He made a quick stop at the store", "The quick pace was energizing",
    ],
    "rapid": [
        "The rapid growth surprised analysts", "She made a rapid recovery",
        "He noticed a rapid change in temperature", "The rapid response team arrived",
        "The river had a rapid current", "She experienced rapid progress",
        "The rapid spread of information was remarkable", "He faced rapid decline in health",
        "The rapid movement caught everyone off guard", "She observed rapid evolution in the field",
        "The rapid expansion created challenges", "He made a rapid ascent up the mountain",
        "The rapid pulse indicated excitement", "She witnessed rapid development downtown",
      "The rapid rotation created a blur", "He studied the rapid changes carefully",
        "The rapid acceleration pushed him back", "She described the rapid transformation",
        "The rapid turnover was costly", "He managed the rapid growth well",
        "The rapid pace was unsustainable", "She tracked the rapid shift in opinion",
        "The rapid deployment was successful", "He predicted the rapid collapse",
        "The rapid cooling solidified the metal", "She noted the rapid improvement",
        "The rapid advance of technology", "He documented the rapid erosion",
        "The rapid absorption surprised the chemist", "She measured the rapid heartbeat",
    ],
    "strong": [
        "The strong wind blew all day", "She has a strong personality",
        "He made a strong argument", "The strong coffee kept him awake",
        "This is a strong possibility", "The strong foundation held firm",
        "She gave a strong performance", "The strong smell filled the room",
        "He showed strong leadership skills", "The strong current pulled swimmers",
        "A strong earthquake shook the city", "She felt a strong connection",
        "The strong bond between them was clear", "He had a strong desire to succeed",
        "She made a strong impression", "The strong economy boosted confidence",
        "He took a strong stance on the issue", "The strong flavor was distinctive",
        "She had strong opinions on the matter", "The strong evidence was convincing",
        "A strong light illuminated the stage", "He developed strong muscles",
        "She had a strong sense of justice", "The strong fabric lasted years",
        "He felt strong emotions inside", "The strong candidate won easily",
        "She built a strong team", "The strong signal was clear",
        "He gave a strong warning", "The strong relationship endured",
    ],
    "weak": [
        "The weak signal was frustrating", "She has a weak immune system",
        "He made a weak argument", "The weak coffee was disappointing",
        "This is a weak possibility", "The weak foundation was cracking",
        "She gave a weak performance", "The weak smell was barely noticeable",
        "He showed weak leadership skills", "The weak current was gentle",
        "A weak tremor was felt briefly", "She felt a weak connection",
        "The weak bond between them was obvious", "He had a weak desire to continue",
        "She made a weak impression", "The weak economy worried everyone",
        "He took a weak stance on the issue", "The weak flavor was bland",
        "She had weak opinions on the matter", "The weak evidence was dismissed",
        "A weak light barely reached the corner", "He had weak muscles from illness",
        "She had a weak sense of direction", "The weak fabric tore easily",
        "He felt weak after the illness", "The weak candidate lost badly",
        "She built a weak team", "The weak signal kept dropping",
        "He gave a weak warning", "The weak relationship ended quickly",
    ],
    "powerful": [
        "The powerful engine roared to life", "She gave a powerful speech",
        "He is a powerful leader in the industry", "The powerful storm destroyed homes",
        "This is a powerful tool for analysis", "The powerful computer processed quickly",
        "She made a powerful impression", "The powerful fragrance filled the room",
        "He showed powerful determination", "The powerful current was dangerous",
        "A powerful earthquake struck the region", "She felt a powerful connection",
        "The powerful bond was unbreakable", "He had a powerful desire to win",
        "She delivered a powerful message", "The powerful economy dominated globally",
        "He took a powerful stance against injustice", "The powerful flavor was intense",
        "She held powerful positions in government", "The powerful evidence was undeniable",
        "A powerful light illuminated the area", "He developed powerful muscles through training",
        "She had a powerful sense of purpose", "The powerful medication worked fast",
        "He felt powerful emotions during the ceremony", "The powerful nation influenced others",
        "She built a powerful coalition", "The powerful signal reached far",
        "He gave a powerful warning", "The powerful alliance held strong",
    ],
    "fierce": [
        "The fierce wind howled all night", "She gave a fierce performance",
        "He is a fierce competitor in sports", "The fierce storm raged for hours",
        "This is a fierce debate in congress", "The fierce battle lasted days",
        "She made a fierce argument", "The fierce animal attacked fiercely",
        "He showed fierce determination", "The fierce current swept boats away",
        "A fierce fight broke out suddenly", "She felt a fierce loyalty to her team",
        "The fierce rivalry was legendary", "He had a fierce temper sometimes",
        "She delivered a fierce critique", "The fierce opposition was expected",
        "He took a fierce stance on the issue", "The fierce flavor was overpowering",
        "She held fierce convictions about justice", "The fierce resistance continued",
        "A fierce glare silenced the room", "He developed a fierce reputation",
        "She had a fierce work ethic", "The fierce competition drove innovation",
        "He felt fierce pride in his work", "The fierce dog guarded the house",
        "She built a fierce following online", "The fierce wind chill was dangerous",
        "He gave a fierce warning to opponents", "The fierce loyalty was admirable",
    ],
    "up": [
        "She looked up at the sky", "He climbed up the hill",
        "The prices went up again", "She stood up from her chair",
        "He woke up early this morning", "The elevator went up to the top floor",
        "She picked up the phone", "He sped up on the highway",
        "The temperature is going up", "She tied up her hair",
        "He signed up for the course", "The balloon floated up into the air",
        "She opened up about her feelings", "He built up his savings",
        "The wind picked up suddenly", "She dressed up for the occasion",
        "He showed up at the party", "The fire flared up again",
        "She spoke up during the meeting", "He set up the equipment",
        "The sun came up over the horizon", "She grew up in a small town",
        "He filled up the gas tank", "The volume was turned up",
        "She heated up the leftovers", "He brought up an important point",
        "The stock went up three points", "She lined up the candidates",
        "He added up the total", "The investigation turned up new evidence",
    ],
    "down": [
        "She looked down at the ground", "He walked down the stairs",
        "The prices went down this month", "She sat down on the bench",
        "He calmed down after the argument", "The elevator went down to the lobby",
        "She put down the book", "He slowed down on the curve",
        "The temperature is going down", "She let down her hair",
        "He wrote down the instructions", "The balloon drifted down slowly",
        "She broke down in tears", "He tore down the old building",
        "The wind died down gradually", "She cut down on sugar intake",
        "He tracked down the source", "The fire burned down to embers",
        "She turned down the offer", "He shut down the computer",
        "The sun went down behind the hills", "She settled down in the countryside",
        "He boiled down the sauce", "The volume was turned down",
        "She cooled down the soup", "He brought down the ceiling",
        "The stock went down five points", "She narrowed down the options",
        "He stripped down the engine", "The search turned down no results",
    ],
    "rich": [
        "The rich businessman donated generously", "She comes from a rich family",
        "He has rich experience in finance", "The rich flavor was delightful",
        "This region has rich natural resources", "The rich soil produced abundant crops",
        "She wore a rich velvet dress", "The rich history fascinated scholars",
        "He prepared a rich chocolate cake", "The rich culture attracted tourists",
        "She has rich knowledge of the subject", "The rich color was striking",
        "He enjoyed a rich lifestyle", "The rich aroma filled the kitchen",
        "This is a rich source of information", "The rich heritage was preserved",
        "She created a rich tapestry of stories", "The rich dessert was indulgent",
        "He developed a rich vocabulary over years", "The rich ecosystem was diverse",
        "She painted with rich warm colors", "The rich sound of the orchestra",
        "He made a rich investment portfolio", "The rich texture was appealing",
        "She offered a rich variety of options", "The rich symbolism was profound",
        "He had a rich imagination", "The rich gravy complemented the meat",
        "She had rich brown hair", "The rich tradition continued for generations",
    ],
    "poor": [
        "The poor family struggled to survive", "She comes from a poor background",
        "He had poor results on the test", "The poor quality was obvious",
        "This region has poor infrastructure", "The poor soil produced little",
        "She made a poor decision", "The poor performance was disappointing",
        "He is in poor health lately", "The poor lighting made it hard to see",
        "She had poor attendance this semester", "The poor planning caused delays",
        "This is a poor excuse for being late", "The poor condition was concerning",
        "She showed poor judgment in the matter", "The poor communication led to misunderstandings",
        "He gave a poor effort on the project", "The poor response was frustrating",
        "She had poor timing for her announcement", "The poor design was criticized",
        "He made poor choices in his youth", "The poor visibility caused accidents",
        "She had poor coordination on the field", "The poor reviews hurt sales",
        "He offered a poor solution to the problem", "The poor organization was chaotic",
        "She had poor eyesight without glasses", "The poor drainage flooded the yard",
        "He had poor luck with investments", "The poor signal dropped frequently",
    ],
    "day": [
        "The day was bright and sunny", "She enjoyed a day at the beach",
        "He worked all day without rest", "The day started early",
        "This day will be remembered forever", "The day shift begins at dawn",
        "She planned the day carefully", "He had a productive day",
        "The day ended with a beautiful sunset", "She celebrated her special day",
        "He looked forward to the day ahead", "The day was long and tiring",
        "This day marks a new beginning", "The day care was full of children",
        "She spent the day reading", "He saved the best for last that day",
        "The day room was warm and cozy", "She cherished every day they had",
        "He made the most of the day", "The day trip was enjoyable",
        "She counted the days until vacation", "He missed the good old days",
        "The day before was even busier", "She had a day off from work",
        "He described the day in detail", "The day after was more relaxed",
        "She worked day and night on the project", "He took it one day at a time",
        "The longest day of the year arrived", "She recorded the day in her journal",
    ],
    "night": [
        "The night was dark and quiet", "She enjoyed a night at the theater",
        "He worked all night without sleep", "The night started late",
        "This night will be remembered forever", "The night shift begins at midnight",
        "She planned the night carefully", "He had a restless night",
        "The night ended with a full moon", "She celebrated a special night",
        "He looked forward to the night ahead", "The night was long and cold",
        "This night marks a new tradition", "The night club was full of people",
        "She spent the night reading", "He saved the best for the night",
        "The night air was cool and fresh", "She cherished every night they had",
        "He made the most of the night", "The night out was enjoyable",
        "She counted the nights until the trip", "He missed the quiet nights",
        "The night before was even colder", "She had a night out with friends",
        "He described the night in vivid detail", "The night after was more peaceful",
        "She worked night and day on the project", "He took it one night at a time",
        "The longest night of the year arrived", "She recorded the night in her journal",
    ],
    "city": [
        "The city skyline was impressive", "She moved to the city last year",
        "He loves the energy of the city", "The city traffic was terrible",
        "This city has great restaurants", "The city council met yesterday",
        "She explored the city on foot", "He grew up in a small city",
        "The city park was beautiful", "She preferred city life to the country",
        "He navigated the city efficiently", "The city lights twinkled at night",
        "This city never sleeps", "The city center was bustling",
        "She studied the city map carefully", "He represented the city in the competition",
        "The city budget was approved", "She described the city perfectly",
        "He enjoyed city tours", "The city population grew rapidly",
        "She loved the city architecture", "He worked in the city downtown",
        "The city was founded centuries ago", "She painted the city landscape",
        "He admired the city from above", "The city festival attracted thousands",
        "She knew the city like the back of her hand", "The city limits expanded",
        "He documented city life in photos", "The city attracted young professionals",
    ],
    "car": [
        "The car sped down the highway", "She bought a new car last week",
        "He repaired the car himself", "The car park was nearly full",
        "This car gets great mileage", "The car alarm went off suddenly",
        "She drove the car carefully", "He rented a car for the trip",
        "The car was parked outside", "She preferred driving her own car",
        "He washed the car on Saturday", "The car accident was minor",
        "This car has a powerful engine", "The car door was stuck",
        "She left her keys in the car", "He traded in his old car",
        "The car needed new tires", "She followed the car ahead closely",
        "He customized his car extensively", "The car broke down on the highway",
        "She filled the car with gas", "He insured the car comprehensively",
        "The car interior was spotless", "She registered the car online",
        "He admired the classic car", "The car manufacturer recalled the model",
        "She booked a car for the weekend", "He detailed the car meticulously",
        "The car showroom was impressive", "She tested the car before buying",
    ],
    "wind": [
        "The wind was howling outside", "She felt the wind on her face",
        "He watched the wind move the trees", "The wind chill was extreme",
        "This wind direction is unusual", "The wind turbine generated power",
        "She ran against the wind", "He measured the wind speed",
        "The wind carried leaves across the yard", "She sailed with the wind",
        "He opened the window to feel the wind", "The wind died down by evening",
        "This wind pattern is predictable", "The wind instrument played a note",
        "She leaned into the wind", "He predicted the wind would change",
        "The wind swept across the plains", "She closed the door against the wind",
        "He described the wind as fierce", "The wind advisory was issued",
        "She hiked despite the strong wind", "He built a wind shelter",
        "The wind rustled the papers", "She loved the ocean wind",
        "He tracked the wind direction", "The wind knocked down the sign",
        "She faced the wind bravely", "He recorded the wind data",
        "The wind scattered the seeds", "She enjoyed the evening wind",
    ],
    "morning": [
        "The morning was fresh and cool", "She woke up early this morning",
        "He exercised every morning", "The morning light was beautiful",
        "This morning felt different", "The morning routine was comforting",
        "She had coffee every morning", "He read the newspaper in the morning",
        "The morning meeting was productive", "She enjoyed the morning silence",
        "He walked in the morning dew", "The morning flight departed on time",
        "This morning brought good news", "The morning shift started early",
        "She watched the morning sunrise", "He cooked breakfast every morning",
        "The morning fog lifted slowly", "She preferred the morning hours",
        "He arrived in the morning", "The morning air was crisp",
        "She started her morning jog", "He checked his email every morning",
        "The morning was peaceful and calm", "She loved the morning light",
        "He had a morning appointment", "The morning traffic was heavy",
        "She took a morning class", "He meditated each morning",
        "The morning birds sang beautifully", "She greeted the morning with a smile",
    ],
    "beautiful": [
        "The beautiful sunset took her breath away", "She wore a beautiful dress",
        "He painted a beautiful landscape", "The beautiful music moved everyone",
        "This is a beautiful place to visit", "The beautiful garden was in full bloom",
        "She has beautiful handwriting", "He admired the beautiful architecture",
        "The beautiful weather was perfect for hiking", "She created a beautiful design",
        "He described the beautiful scenery vividly", "The beautiful melody lingered",
        "She found a beautiful shell on the beach", "He gave a beautiful performance",
        "The beautiful colors of autumn", "She captured a beautiful moment on camera",
        "He built a beautiful model ship", "The beautiful flowers lined the path",
        "She wrote a beautiful poem", "He discovered a beautiful solution",
        "The beautiful day was unforgettable", "She chose beautiful decorations",
        "He shared a beautiful story", "The beautiful view was worth the climb",
        "She arranged beautiful flowers in a vase", "He composed beautiful music",
        "The beautiful pattern was intricate", "She imagined a beautiful future",
        "He carved beautiful designs in wood", "The beautiful fabric shimmered",
    ],
    "intelligent": [
        "She is an intelligent student", "He made an intelligent observation",
        "The intelligent design was efficient", "She gave an intelligent response",
        "This is an intelligent decision", "The intelligent system learned quickly",
        "He wrote an intelligent analysis", "She has intelligent ideas",
        "The intelligent conversation was stimulating", "He used intelligent strategies",
        "She made an intelligent choice", "The intelligent robot performed well",
        "He asked intelligent questions", "She provided an intelligent solution",
        "The intelligent approach saved time", "He developed intelligent software",
        "She showed intelligent behavior", "The intelligent planning paid off",
        "He gave an intelligent critique", "She presented an intelligent argument",
        "The intelligent layout was functional", "He created an intelligent system",
        "She made intelligent investments", "The intelligent design won awards",
        "He proposed an intelligent alternative", "She demonstrated intelligent thinking",
        "The intelligent algorithm was efficient", "He made an intelligent prediction",
        "She offered an intelligent perspective", "The intelligent use of resources",
    ],
    "honest": [
        "She is an honest person", "He gave an honest answer",
        "The honest assessment was helpful", "She appreciated his honest feedback",
        "This is an honest mistake", "The honest truth was hard to hear",
        "He wrote an honest review", "She made an honest effort",
        "The honest conversation was refreshing", "He had honest intentions",
        "She made an honest living", "The honest approach was respected",
        "He asked for an honest opinion", "She provided an honest evaluation",
        "The honest reporting was praised", "He valued honest communication",
        "She kept her honest word", "The honest politician was rare",
        "He gave an honest assessment", "She offered an honest apology",
        "The honest exchange built trust", "He made an honest confession",
        "She told the honest version of events", "The honest work was appreciated",
        "He maintained honest relationships", "She delivered an honest performance",
        "The honest admission was brave", "He praised her honest character",
        "She wrote an honest memoir", "The honest dialogue resolved the conflict",
    ],
    "freezing": [
        "The freezing temperature broke records", "She stepped into the freezing water",
        "He endured the freezing cold all night", "The freezing wind cut through his coat",
        "This freezing rain was dangerous", "The freezing point was reached quickly",
        "She was freezing despite the blanket", "He touched the freezing metal",
        "The freezing weather closed the schools", "She shivered in the freezing night",
        "He found a freezing stray cat outside", "The freezing fog reduced visibility",
        "She made freezing drinks for the party", "He described the freezing conditions",
        "The freezing lake was solid ice", "She wrapped up against the freezing air",
        "He felt freezing fingers and toes", "The freezing chamber preserved samples",
        "She noticed the freezing rain on the window", "He worked in the freezing warehouse",
        "The freezing draft came through the crack", "She served freezing ice cream",
        "He warned about the freezing conditions", "The freezing spray hit his face",
        "She felt freezing cold inside the tent", "He survived the freezing temperatures",
        "The freezing moisture turned to frost", "She rubbed her freezing hands together",
        "He built a shelter from the freezing sleet", "The freezing night was unbearable",
    ],
    "excellent": [
        "She received an excellent grade", "He gave an excellent presentation",
        "The excellent service impressed everyone", "She made an excellent choice",
        "This is an excellent opportunity", "The excellent weather was perfect",
        "He wrote an excellent report", "She has excellent communication skills",
        "The excellent team won the championship", "He provided excellent customer service",
        "She demonstrated excellent leadership", "The excellent design won an award",
        "He achieved excellent results", "She showed excellent judgment",
        "This excellent product exceeded expectations", "The excellent performance earned praise",
        "He delivered an excellent speech", "She prepared an excellent meal",
        "The excellent strategy worked perfectly", "He offered an excellent solution",
        "She developed an excellent reputation", "The excellent training was effective",
        "He made an excellent investment", "She had excellent attendance",
        "The excellent reviews boosted sales", "He gave an excellent recommendation",
        "She found an excellent restaurant", "The excellent condition was remarkable",
        "He completed an excellent project", "She made an excellent impression",
    ],
}


def run_part1_cooccurrence(model_name):
    """
    方案1: 共现频率对照实验

    核心逻辑:
    - 计算每对词的子空间overlap
    - 统计每对词的共现频率(用PMI)
    - 如果overlap完全由共现解释 → overlap是统计伪影
    - 如果存在低共现高overlap的词对 → 有独立编码机制
    """
    log_time(f"=== Part 1: Co-occurrence vs Overlap ({model_name}) ===")

    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, release_model
    info = get_model_info(model, model_name)

    # 选择关键层
    n_layers = info.n_layers
    target_layers = [
        n_layers // 4,      # 早期
        n_layers // 2,      # 中期
        3 * n_layers // 4,  # 中后期
    ]

    # 收集所有词的hidden states
    all_words = sorted(set(w for pair in COOCCURRENCE_PAIRS for w in pair))
    log_time(f"Collecting hidden states for {len(all_words)} words, {len(COOCCURRENCE_PAIRS)} pairs")

    word_hidden = {}
    for wi, word in enumerate(all_words):
        sents = COOCCURRENCE_TEMPLATES[word][:30]  # 30模板/词
        hidden = collect_hidden_states(model, tokenizer, device, sents, target_layers)
        word_hidden[word] = hidden
        log_time(f"  Word {wi+1}/{len(all_words)}: {word}")

    # 计算overlap
    n_dims = 10
    overlap_results = {}

    for li in target_layers:
        layer_overlaps = {}
        for w1, w2 in COOCCURRENCE_PAIRS:
            S1 = extract_subspace(word_hidden[w1][li], n_dims)
            S2 = extract_subspace(word_hidden[w2][li], n_dims)
            ov = compute_overlap(S1, S2)
            layer_overlaps[(w1, w2)] = ov
        overlap_results[li] = layer_overlaps

    # 计算PMI (Pointwise Mutual Information) 作为共现度量
    # 用词本身的模板共现: 如果w1和w2的模板中有相同的上下文词, PMI高
    # 简化版: 用ngram共现统计
    log_time("Computing co-occurrence statistics...")

    # 方法: 用训练语料的统计替代 — 这里用模板内n-gram overlap作为proxy
    # 更好的方法: 用word2vec/glove的余弦相似度作为共现proxy
    # 最直接: 统计Wikipedia中两词在同一句子的频率

    # 简化但有效的方法: 用模型tokenizer的共编码率
    # 如果两个词经常一起出现在训练中, 它们的token embedding会更接近
    # 用token embedding余弦相似度作为共现proxy

    token_embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()

    cooccurrence_results = {}
    for w1, w2 in COOCCURRENCE_PAIRS:
        # 获取token ids
        id1 = tokenizer.encode(w1, add_special_tokens=False)
        id2 = tokenizer.encode(w2, add_special_tokens=False)

        if len(id1) == 1 and len(id2) == 1:
            # 单token词: 用embedding余弦相似度
            e1 = token_embed[id1[0]]
            e2 = token_embed[id2[0]]
            cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
        else:
            # 多token词: 用平均embedding
            e1 = token_embed[id1].mean(axis=0)
            e2 = token_embed[id2].mean(axis=0)
            cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

        cooccurrence_results[(w1, w2)] = cos_sim

    # 汇总结果
    results = {"model": model_name, "n_dims": n_dims, "target_layers": target_layers}

    for li in target_layers:
        layer_data = []
        for pair_type, pairs in [
            ("antonym", COOCCURRENCE_PAIRS[:9]),
            ("synonym", COOCCURRENCE_PAIRS[9:15]),
            ("high_cooc_control", COOCCURRENCE_PAIRS[15:21]),
            ("low_cooc_semantic", COOCCURRENCE_PAIRS[21:27]),
            ("cross_category", COOCCURRENCE_PAIRS[27:30]),
        ]:
            for w1, w2 in pairs:
                ov = overlap_results[li][(w1, w2)]
                cos = cooccurrence_results[(w1, w2)]
                layer_data.append({
                    "pair": f"{w1}-{w2}", "type": pair_type,
                    "overlap": round(ov, 4), "embedding_cos": round(cos, 4),
                })

        results[f"layer_{li}"] = layer_data

    # 关键分析: overlap排序 vs embedding_cos排序
    log_time("\n=== 关键分析: Overlap vs Embedding Cosine ===")

    for li in target_layers:
        log_time(f"\nLayer {li}:")
        data = results[f"layer_{li}"]

        # 各类型平均
        type_stats = defaultdict(lambda: {"overlap": [], "cos": []})
        for d in data:
            type_stats[d["type"]]["overlap"].append(d["overlap"])
            type_stats[d["type"]]["cos"].append(d["embedding_cos"])

        log_time(f"  {'Type':<22} {'Avg Overlap':>12} {'Avg EmbedCos':>12} {'N':>4}")
        log_time(f"  {'-'*50}")
        for t in ["antonym", "synonym", "high_cooc_control", "low_cooc_semantic", "cross_category"]:
            ovs = type_stats[t]["overlap"]
            css = type_stats[t]["cos"]
            log_time(f"  {t:<22} {np.mean(ovs):>12.4f} {np.mean(css):>12.4f} {len(ovs):>4}")

        # 关键检验: 是否存在低cos高overlap的词对
        log_time(f"\n  === 关键检验: 低EmbedCos + 高Overlap ===")
        for d in sorted(data, key=lambda x: x["overlap"], reverse=True)[:10]:
            log_time(f"    {d['pair']:<25} overlap={d['overlap']:.4f}  emb_cos={d['embedding_cos']:.4f}  type={d['type']}")

        # Spearman相关性
        from scipy.stats import spearmanr
        overlaps = [d["overlap"] for d in data]
        cosines = [d["embedding_cos"] for d in data]
        rho, pval = spearmanr(overlaps, cosines)
        log_time(f"\n  Spearman rho={rho:.4f}, p={pval:.4f}")

        # 如果rho<0.8, 说明embedding共现不能完全解释overlap
        if rho < 0.8:
            log_time(f"  *** Embedding cos CANNOT fully explain overlap! Independent encoding mechanism exists! ***")
        else:
            log_time(f"  WARNING: Embedding cos can explain overlap. May be co-occurrence artifact.")

    # 保存
    save_path = RESULT_DIR / f"phase60_part1_{model_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Saved to {save_path}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =====================================================================
# 方案2: 强度维度直接提取
# =====================================================================

INTENSITY_AXIS_WORDS = {
    "temperature": ["freezing", "cold", "cool", "warm", "hot", "scorching"],
    "size": ["microscopic", "tiny", "small", "medium", "big", "large", "huge", "enormous"],
    "emotion_valence": ["hate", "dislike", "neutral", "like", "love"],
    "emotion_intensity": ["annoyed", "angry", "furious", "upset", "sad", "devastated"],
    "speed": ["crawling", "slow", "moderate", "fast", "rapid", "lightning"],
}

INTENSITY_TEMPLATES = COOCCURRENCE_TEMPLATES  # 复用已有的模板

# 补充缺失词的模板
_EXTRA_TEMPLATES = {
    "microscopic": [
        "The microscopic organism was invisible to the naked eye", "She used a microscopic lens to examine the sample",
        "He studied microscopic structures in the lab", "The microscopic particles were analyzed",
        "This microscopic detail was crucial", "The microscopic change had huge effects",
        "She observed microscopic bacteria under the microscope", "He found microscopic cracks in the metal",
        "The microscopic world is fascinating", "She detected microscopic amounts of the chemical",
        "The microscopic image revealed surprising patterns", "He specialized in microscopic photography",
        "The microscopic fibers were strong", "She measured the microscopic distance precisely",
        "The microscopic flaw caused the failure", "He described the microscopic features accurately",
        "The microscopic analysis took hours", "She published a microscopic study of the tissue",
        "The microscopic evidence was conclusive", "He developed a microscopic technique",
        "The microscopic structure was complex", "She identified the microscopic culprit",
        "The microscopic observation was recorded", "He magnified the microscopic specimen",
        "The microscopic scale was hard to imagine", "She compared microscopic samples",
        "The microscopic layer was thin", "He noticed a microscopic improvement",
        "The microscopic world beneath our feet", "She catalogued microscopic species",
    ],
    "medium": [
        "The medium pizza was enough for two", "She chose the medium size",
        "He prefers a medium roast coffee", "The medium difficulty level was appropriate",
        "This is a medium priority task", "The medium heat setting works best",
        "She ordered a medium drink", "He played at a medium tempo",
        "The medium weight was manageable", "She has a medium build",
        "He set the volume to medium", "The medium range was adequate",
        "This is of medium quality", "She took a medium approach",
        "He chose the medium option", "The medium gray color was neutral",
        "She cooked on medium heat", "He wore a medium shirt",
        "The medium texture was just right", "She applied medium pressure",
        "He found a medium ground", "The medium intensity was comfortable",
        "She worked at a medium pace", "He made a medium investment",
        "The medium well steak was perfect", "She preferred medium rare",
        "He used medium grit sandpaper", "The medium was ideal for the experiment",
        "She selected the medium setting", "He maintained a medium distance",
    ],
    "crawling": [
        "The crawling baby explored the room", "She noticed a crawling insect on the wall",
        "He moved at a crawling pace in traffic", "The crawling traffic was frustrating",
        "This crawling progress was painful", "The crawling sensation on her skin was creepy",
        "He watched the crawling caterpillar", "She set the crawling speed to minimum",
        "The crawling vine covered the wall", "He described the crawling motion precisely",
        "The crawling data transfer took hours", "She felt crawling dread inside",
        "He heard a crawling sound in the attic", "The crawling fog crept over the hill",
        "She observed the crawling tide", "He studied the crawling robot design",
        "The crawling pace was unbearable", "She witnessed the crawling chaos unfold",
        "He avoided the crawling mass of insects", "The crawling exploration was methodical",
        "She documented the crawling behavior", "He tracked the crawling progress",
        "The crawling creature left a trail", "She painted the crawling figure",
        "He built a crawling mechanism", "The crawling baby was adorable",
        "She heard the crawling thing at night", "He designed a crawling vehicle",
        "The crawling lava moved slowly", "She monitored the crawling invasion",
    ],
    "moderate": [
        "The moderate temperature was pleasant", "She took a moderate approach",
        "He has moderate political views", "The moderate exercise was beneficial",
        "This is a moderate amount of risk", "The moderate pace was sustainable",
        "She made moderate progress", "He expressed moderate concern",
        "The moderate inflation rate was expected", "She chose a moderate difficulty",
        "He maintained moderate speed", "The moderate weather continued",
        "She had moderate expectations", "He showed moderate improvement",
        "The moderate rainfall was helpful", "She consumed a moderate portion",
        "He recommended moderate exercise daily", "The moderate winds were manageable",
        "She set moderate goals", "He achieved moderate success",
        "The moderate humidity was comfortable", "She used moderate force",
        "He has a moderate income", "The moderate climate was ideal",
        "She preferred moderate flavors", "He took a moderate stance",
        "The moderate distance was walkable", "She made a moderate adjustment",
        "He applied moderate pressure", "The moderate volume was appropriate",
    ],
    "rapid": COOCCURRENCE_TEMPLATES.get("rapid", []),
    "lightning": [
        "The lightning struck the tree", "She moved with lightning speed",
        "He saw lightning flash across the sky", "The lightning storm was intense",
        "This lightning fast response saved lives", "The lightning illuminated the darkness",
        "She was struck by lightning once", "He made a lightning decision",
        "The lightning reflexes helped in the game", "Lightning never strikes the same place twice",
        "She captured lightning in a bottle", "He gave a lightning quick answer",
        "The lightning rod protected the building", "She watched the lightning display",
        "He dodged the lightning strike narrowly", "The lightning crackled overhead",
        "This happened in a lightning flash", "The lightning was followed by thunder",
        "She described the lightning vividly", "He studied lightning patterns",
        "The lightning bug glowed in the dark", "She survived the lightning strike",
        "He invented a lightning detector", "The lightning went sideways",
        "She felt the lightning energy", "He predicted the lightning would return",
        "The lightning lit up the whole sky", "She painted the lightning bolt",
        "He charged like lightning into battle", "The lightning frequency increased",
    ],
    "annoyed": [
        "She was annoyed by the noise", "He looked annoyed at the delay",
        "The annoyed customer demanded a refund", "She felt annoyed by his behavior",
        "He became annoyed with the slow service", "An annoyed expression crossed her face",
        "She was mildly annoyed by the comment", "He sounded annoyed on the phone",
        "The annoyed neighbor complained loudly", "She grew annoyed at the interruption",
        "He was clearly annoyed by the question", "She wrote an annoyed email",
        "The annoyed teacher gave extra homework", "He left the room annoyed",
        "She seemed annoyed by the suggestion", "He gave an annoyed sigh",
        "The annoyed crowd grew restless", "She was visibly annoyed by the mistake",
        "He got annoyed when they were late", "She showed her annoyed feelings clearly",
        "The annoyed response was unexpected", "He was slightly annoyed by the change",
        "She made an annoyed gesture", "The annoyed player argued the call",
        "He expressed his annoyed opinion", "She was annoyed but tried to hide it",
        "The annoyed employee quit his job", "He had an annoyed tone of voice",
        "She sent an annoyed message", "The annoyed audience booed",
    ],
    "angry": [
        "She was angry at the unfairness", "He gave an angry speech",
        "The angry mob gathered outside", "She felt angry about the decision",
        "He became angry when he heard the news", "An angry look crossed her face",
        "She was very angry at the betrayal", "He sounded angry on the phone",
        "The angry customer demanded justice", "She grew angry at the insult",
        "He was clearly angry about the failure", "She wrote an angry letter",
        "The angry protester waved a sign", "He left the room angry",
        "She seemed angry about the delay", "He gave an angry shout",
        "The angry crowd chanted slogans", "She was visibly angry at the mistake",
        "He got angry when they lied", "She showed her angry feelings openly",
        "The angry response was forceful", "He was extremely angry about the theft",
        "She made an angry gesture", "The angry player threw the racket",
        "He expressed his angry opinion loudly", "She was angry but controlled herself",
        "The angry employee filed a complaint", "He had an angry outburst",
        "She sent an angry text", "The angry debate heated up",
    ],
    "furious": [
        "She was furious at the betrayal", "He gave a furious speech",
        "The furious storm raged for days", "She felt furious about the injustice",
        "He became furious when he was accused", "A furious look crossed her face",
        "She was absolutely furious at the lie", "He sounded furious on the phone",
        "The furious customer demanded the manager", "She grew furious at the corruption",
        "He was clearly furious about the betrayal", "She wrote a furious email",
        "The furious wind destroyed the roof", "He left the room furious",
        "She seemed furious about the accusation", "He gave a furious shout",
        "The furious debate lasted all night", "She was visibly furious at the insult",
        "He got furious when they stole his work", "She showed her furious feelings openly",
        "The furious response was overwhelming", "He was beyond furious about the fraud",
        "She made a furious gesture", "The furious player attacked the referee",
        "He expressed his furious opinion explosively", "She was furious and could not hide it",
        "The furious employee quit on the spot", "He had a furious temper",
        "She sent a furious message", "The furious reaction was unexpected",
    ],
    "upset": [
        "She was upset about the news", "He looked upset after the call",
        "The upset child cried quietly", "She felt upset by the criticism",
        "He became upset when he heard the truth", "An upset expression appeared on her face",
        "She was really upset by the rejection", "He sounded upset on the phone",
        "The upset stomach bothered her all day", "She grew upset at the misunderstanding",
        "He was clearly upset about the loss", "She wrote an upset message",
        "The upset parent complained to the school", "He left the room upset",
        "She seemed upset about the change", "He gave an upset sigh",
        "The upset crowd demanded answers", "She was visibly upset by the comment",
        "He got upset when they canceled the event", "She showed her upset feelings quietly",
        "The upset response was understandable", "He was deeply upset by the death",
        "She made an upset gesture", "The upset student asked for help",
        "He expressed his upset feelings carefully", "She was upset but trying to cope",
        "The upset customer asked for a refund", "He had an upset stomach after the meal",
        "She sent an upset reply", "The upset family mourned together",
    ],
    "sad": [
        "She felt sad about the loss", "He looked sad and dejected",
        "The sad movie made everyone cry", "She was sad when her friend left",
        "He became sad thinking about the past", "A sad expression covered her face",
        "She was deeply sad about the tragedy", "He sounded sad on the phone",
        "The sad news spread quickly", "She grew sad at the memory",
        "He was clearly sad about the failure", "She wrote a sad poem",
        "The sad song played on the radio", "He left the room sad",
        "She seemed sad about the outcome", "He gave a sad smile",
        "The sad story touched everyone", "She was visibly sad at the funeral",
        "He got sad when he saw the destruction", "She showed her sad feelings openly",
        "The sad reality was hard to accept", "He was terribly sad about the diagnosis",
        "She made a sad face", "The sad player sat on the bench",
        "He expressed his sad feelings quietly", "She was sad but hopeful",
        "The sad ending surprised no one", "He had a sad look in his eyes",
        "She sent a sad message", "The sad occasion brought them together",
    ],
    "devastated": [
        "She was devastated by the loss", "He looked completely devastated",
        "The devastated town was unrecognizable", "She felt devastated after the accident",
        "He became devastated when he heard", "A devastated expression froze on her face",
        "She was absolutely devastated by the news", "He sounded devastated on the phone",
        "The devastated community came together", "She grew devastated at the destruction",
        "He was clearly devastated by the failure", "She wrote a devastated email",
        "The devastated landscape showed the damage", "He left the room devastated",
        "She seemed devastated by the diagnosis", "He gave a devastated look",
        "The devastated family mourned together", "She was visibly devastated at the scene",
        "He was devastated when the project failed", "She showed her devastated feelings openly",
        "The devastated response was heartbreak", "He was utterly devastated by the betrayal",
        "She made a devastated gesture", "The devastated player could not continue",
        "He expressed his devastated feelings openly", "She was devastated beyond words",
        "The devastated employee could not work", "He had a devastated spirit",
        "She sent a devastated message", "The devastated region needed aid",
    ],
}

# 合并模板
for k, v in _EXTRA_TEMPLATES.items():
    if k not in COOCCURRENCE_TEMPLATES:
        INTENSITY_TEMPLATES[k] = v
    else:
        INTENSITY_TEMPLATES[k] = COOCCURRENCE_TEMPLATES[k]


def run_part2_intensity(model_name):
    """
    方案2: 强度维度直接提取

    核心假设: 语义轴编码为 direction × intensity
    - 对轴上所有词的mean embedding做SVD
    - PC1应编码"intensity", PC2应编码"direction"
    - 验证: 投影到PC1-PC2平面, 应形成U形分布(两端高、中间低)
    """
    log_time(f"=== Part 2: Intensity Dimension Extraction ({model_name}) ===")

    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, release_model, get_W_U
    info = get_model_info(model, model_name)

    n_layers = info.n_layers
    target_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
    ]

    # 获取W_U用于方向解码
    W_U = get_W_U(model, model_name)

    results = {"model": model_name, "target_layers": target_layers}

    for axis_name, words in INTENSITY_AXIS_WORDS.items():
        log_time(f"\n--- Axis: {axis_name} ---")
        log_time(f"Words: {words}")

        # 收集每个词的hidden states
        word_means = {}  # {word: {layer: mean_vector}}
        for word in words:
            sents = INTENSITY_TEMPLATES.get(word, [])[:30]
            if len(sents) < 5:
                log_time(f"  WARNING: only {len(sents)} templates for {word}, skipping")
                continue

            hidden = collect_hidden_states(model, tokenizer, device, sents, target_layers)
            word_means[word] = {}
            for li in target_layers:
                word_means[word][li] = hidden[li].mean(axis=0)

        if len(word_means) < 3:
            log_time(f"  Skipping axis {axis_name}: not enough words with templates")
            continue

        for li in target_layers:
            # 构建mean matrix: [n_words, d_model]
            sorted_words = [w for w in words if w in word_means]
            mean_matrix = np.array([word_means[w][li] for w in sorted_words])

            # SVD on mean embeddings
            mean_center = mean_matrix.mean(axis=0, keepdims=True)
            centered = mean_matrix - mean_center
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # 投影到PC1-PC2平面
            projections = centered @ Vt[:2].T  # [n_words, 2]

            log_time(f"  Layer {li}: Singular values = {S[:5].round(2)}")
            log_time(f"  Projections on PC1-PC2:")
            for i, w in enumerate(sorted_words):
                log_time(f"    {w:<15} PC1={projections[i,0]:>8.3f}  PC2={projections[i,1]:>8.3f}")

            # 解码PC1和PC2: 投影到logit空间
            pc1_logit = Vt[0] @ W_U.T  # [vocab]
            pc2_logit = Vt[1] @ W_U.T

            # Top tokens boosted by each PC
            top_pc1 = np.argsort(pc1_logit)[-20:][::-1]
            top_pc2 = np.argsort(pc2_logit)[-20:][::-1]

            pc1_tokens = [tokenizer.decode([t]).strip() for t in top_pc1]
            pc2_tokens = [tokenizer.decode([t]).strip() for t in top_pc2]

            log_time(f"  PC1 top tokens: {pc1_tokens[:15]}")
            log_time(f"  PC2 top tokens: {pc2_tokens[:15]}")

            # 检验U形分布: PC1(预期=intensity)在轴两端应该大
            # 即: PC1投影值的绝对值在两端大、中间小
            pc1_vals = projections[:, 0]
            n = len(pc1_vals)

            # 计算U形度: 检查|PC1|是否在两端大中间小
            abs_pc1 = np.abs(pc1_vals)
            edge_mean = (abs_pc1[0] + abs_pc1[-1]) / 2  # 两端平均
            if n > 2:
                mid_idx = n // 2
                mid_mean = np.mean(abs_pc1[max(0, mid_idx-1):mid_idx+2])  # 中间平均
                u_shape_score = edge_mean / (mid_mean + 1e-10)
            else:
                u_shape_score = 1.0

            # 也检查线性度: PC2(预期=direction)是否单调排列
            pc2_vals = projections[:, 1]
            # 检查是否单调递增或递减
            diffs = np.diff(pc2_vals)
            monotonic_score = float(np.sum(diffs > 0)) / max(len(diffs), 1)
            monotonic_score = max(monotonic_score, 1 - monotonic_score)  # 正反方向都算

            log_time(f"  U-shape score (PC1): {u_shape_score:.3f} (>1 = U-shaped)")
            log_time(f"  Monotonic score (PC2): {monotonic_score:.3f} (>0.8 = monotonic)")

            results[f"{axis_name}_L{li}"] = {
                "words": sorted_words,
                "projections": projections.tolist(),
                "singular_values": S[:5].tolist(),
                "pc1_top_tokens": pc1_tokens[:15],
                "pc2_top_tokens": pc2_tokens[:15],
                "u_shape_score": round(u_shape_score, 4),
                "monotonic_score": round(monotonic_score, 4),
            }

    save_path = RESULT_DIR / f"phase60_part2_{model_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Saved to {save_path}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =====================================================================
# 方案3: 概念区分能力验证
# =====================================================================

DISCRIMINATION_CONCEPTS = {
    "temperature": ["hot", "cold", "warm", "cool", "freezing"],
    "size": ["big", "small", "large", "tiny", "huge"],
    "emotion": ["love", "hate", "like", "dislike", "neutral"],
    "animal": ["cat", "dog", "bird", "fish", "horse"],
    "fruit": ["apple", "banana", "orange", "grape", "mango"],
}

# 补充模板
_EXTRA_TEMPLATES_3 = {
    "cat": [f"The cat {v}" for v in [
        "sat on the windowsill", "purred softly on the couch", "chased the mouse across the floor",
        "slept in the sun all afternoon", "scratched the furniture again", "meowed at the door",
        "jumped onto the counter", "played with the yarn ball", "licked its paw carefully",
        "hid under the bed during the storm", "stalked a bird in the garden", "curled up on the blanket",
        "knocked the vase off the shelf", "watched the fish swim", "climbed the tree quickly",
        "ate the canned food eagerly", "groomed itself for hours", "followed me around the house",
        "caught a moth in midair", "sat in the cardboard box", "rubbed against my leg",
        "slept at the foot of the bed", "stared out the window", "played with the laser pointer",
        "drank milk from the saucer", "climbed into my lap", "hissed at the stranger",
        "brought a mouse as a gift", "rolled over for a belly rub", "purred when I petted it",
    ]],
    "dog": [f"The dog {v}" for v in [
        "barked at the mailman", "chased the ball across the yard", "wagged its tail happily",
        "slept on the rug by the fire", "dug a hole in the garden", "howled at the moon",
        "jumped up to greet me", "played fetch with the stick", "licked my face excitedly",
        "ran through the park freely", "guarded the house faithfully", "sniffed the ground carefully",
        "rolled in the grass", "begged for food at the table", "swam in the lake",
        "pulled on the leash eagerly", "slept in the doghouse", "learned a new trick quickly",
        "growled at the intruder", "brought back the newspaper", "followed the scent trail",
        "played with the other dogs", "drooled on the carpet", "shook off the water",
        "cuddled on the couch", "ran to the door when called", "chewed the bone contentedly",
        "performed the trick perfectly", "howled along with the music", "protected the child",
    ]],
    "bird": [f"The bird {v}" for v in [
        "flew across the sky", "sang a beautiful melody", "built a nest in the tree",
        "perched on the windowsill", "soared above the mountains", "migrated south for winter",
        "pecked at the seeds", "spread its wings wide", "dived into the water",
        "chirped at dawn", "landed on the branch", "flew in a V formation",
        "guarded the eggs carefully", "hopped along the ground", "preened its feathers",
        "sang from the treetop", "navigated by the stars", "flew through the forest",
        "nested on the cliff face", "caught a fish in its beak", "soared on the thermal",
        "called to its mate", "fluffed its feathers in the cold", "darted through the air",
        "rested on the wire", "sang at the break of day", "built an intricate nest",
        "flew away at the noise", "gathered twigs for the nest", "puffed up its chest",
    ]],
    "fish": [f"The fish {v}" for v in [
        "swam through the coral reef", "glided through the clear water", "jumped out of the lake",
        "hid among the seaweed", "chased the smaller fish", "nibbled at the bait",
        "swam in the aquarium", "spawned in the shallow stream", "darted away from the predator",
        "rested near the bottom", "breathed through its gills", "schooled with others",
        "swam upstream to spawn", "was caught on the line", "flopped on the deck",
        "circled the tank slowly", "fed on the plankton", "sank into the deep water",
        "surfaced for insects", "adapted to the cold water", "lived in the reef",
        "changed color for camouflage", "swam against the current", "found shelter in the rocks",
        "wiggled through the net", "grew to enormous size", "migrated across the ocean",
        "survived the winter freeze", "thrived in the warm pond", "fought the fishing line",
    ]],
    "horse": [f"The horse {v}" for v in [
        "galloped across the field", "neighed at the other horses", "stood in the stable",
        "pulled the cart steadily", "jumped over the fence", "grazed in the meadow",
        "trotted along the trail", "bucked the rider off", "cantered around the arena",
        "drank from the stream", "shook its mane proudly", "wore a leather saddle",
        "raced down the track", "nuzzled my hand gently", "stamped its hoof impatiently",
        "ran free on the plains", "carried the supplies up the hill", "reared up on its hind legs",
        "whinnied in the distance", "walked slowly back to the barn", "competed in the show",
        "breathed heavily after the run", "wore a blanket in winter", "followed the herd",
        "trained for the competition", "pulled ahead in the race", "rested under the shade tree",
        "swam across the river", "explored the new pasture", "responded to the riders cue",
    ]],
    "banana": [f"The banana {v}" for v in [
        "was ripe and yellow", "peeled easily in my hand", "tasted sweet and creamy",
        "fell from the bunch", "was mashed for the bread", "turned brown on the counter",
        "grew in the tropical climate", "provided quick energy", "was sliced for the cereal",
        "smelled fragrant and sweet", "was imported from overseas", "cost a dollar per pound",
        "was blended into the smoothie", "hung in the kitchen", "was slightly green still",
        "got squished in my bag", "was frozen for later use", "had a thick yellow peel",
        "was the perfect snack", "was organic and fair trade", "was shared between friends",
        "lay forgotten in the lunchbox", "was the only fruit left", "made the milkshake thick",
        "was overripe and soft", "grew in a hanging cluster", "was dipped in chocolate",
        "was the monkeys favorite", "was served at breakfast", "was perfectly fresh today",
    ]],
    "orange": [f"The orange {v}" for v in [
        "was juicy and sweet", "peeled easily with my fingers", "had a bright citrus scent",
        "grew on the tree outside", "was squeezed for fresh juice", "rolled off the counter",
        "was ripe and firm", "had thick dimpled skin", "was segmented perfectly",
        "provided vitamin C", "was imported from Florida", "cost two dollars per pound",
        "was cut into wedges", "lay in the fruit bowl", "was slightly tart today",
        "was painted in the still life", "was dried for decoration", "had seeds inside",
        "was the perfect thirst quencher", "was organic and fresh", "was shared at lunch",
        "lay forgotten at the bottom", "was the last one in the bag", "made the salad colorful",
        "was overripe and moldy", "grew in the sunny orchard", "was zested for the cake",
        "was the color of sunset", "was served at the brunch", "was perfectly ripe today",
    ]],
    "grape": [f"The grape {v}" for v in [
        "was sweet and crisp", "hung in a cluster on the vine", "burst with juice in my mouth",
        "was picked fresh from the vineyard", "was dried into a raisin", "turned into fine wine",
        "was seedless and convenient", "grew in the Mediterranean climate", "was pressed for juice",
        "smelled faintly sweet", "was imported from Chile", "cost three dollars per pound",
        "was washed before eating", "lay in the fruit basket", "was green and tart",
        "was crushed underfoot", "was frozen for a snack", "had thin delicate skin",
        "was the perfect appetizer", "was organic and local", "was shared at the picnic",
        "lay forgotten in the fridge", "was the best variety available", "made the cheese plate complete",
        "was overripe and wrinkled", "grew on the trellis", "was fermented carefully",
        "was the chefs favorite", "was served at the reception", "was perfectly plump today",
    ]],
    "mango": [f"The mango {v}" for v in [
        "was incredibly sweet and fragrant", "had juicy golden flesh", "grew in the tropical orchard",
        "was imported from India", "was sliced for the fruit salad", "had a large flat pit inside",
        "was perfectly ripe and soft", "peeled with a knife carefully", "made the smoothie delicious",
        "was blended into a lassi", "had a distinctive aroma", "was sticky and messy to eat",
        "was dried for a healthy snack", "grew on the tall tree", "was the king of fruits",
        "was green and unripe still", "cost four dollars each", "was served with sticky rice",
        "was pickled in brine", "was the essence of summer", "was shared with neighbors",
        "lay forgotten on the shelf", "was the most popular flavor", "made the salsa tropical",
        "was overripe and fermented", "grew in the backyard garden", "was pureed for the dessert",
        "was the color of sunset", "was served at the luau", "was perfectly delicious today",
    ]],
}

# 合并
for k, v in _EXTRA_TEMPLATES_3.items():
    if k not in COOCCURRENCE_TEMPLATES:
        INTENSITY_TEMPLATES[k] = v
    if k not in COOCCURRENCE_TEMPLATES:
        COOCCURRENCE_TEMPLATES[k] = v


def run_part3_discrimination(model_name):
    """
    方案3: 概念区分能力验证

    核心检验: top-k子空间的激活值能否可靠地区分不同概念?
    如果能 → overlap定义有效
    如果不能 → overlap定义需要修正
    """
    log_time(f"=== Part 3: Discrimination Capability ({model_name}) ===")

    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, release_model
    info = get_model_info(model, model_name)

    n_layers = info.n_layers
    target_layer = n_layers // 2  # 只测中间层

    results = {"model": model_name, "target_layer": target_layer}

    for concept_group, words in DISCRIMINATION_CONCEPTS.items():
        log_time(f"\n--- Concept Group: {concept_group} ({words}) ---")

        # 收集hidden states
        all_hidden = {}
        all_labels = []
        for word in words:
            sents = INTENSITY_TEMPLATES.get(word, COOCCURRENCE_TEMPLATES.get(word, []))[:30]
            if len(sents) < 5:
                log_time(f"  Skipping {word}: not enough templates")
                continue

            hidden = collect_hidden_states(model, tokenizer, device, sents, [target_layer])
            all_hidden[word] = hidden[target_layer]
            all_labels.extend([word] * len(hidden[target_layer]))

        if len(all_hidden) < 2:
            continue

        # 准备数据
        X = np.vstack([all_hidden[w] for w in all_hidden])  # [n_samples, d_model]
        y = np.array(all_labels)

        # 方法1: 用top-k子空间维度做分类
        # 对每个概念提取top-k子空间, 合并所有概念的top-k维度
        n_dims = 10
        all_top_dims = set()
        word_subspaces = {}
        for word in all_hidden:
            S = extract_subspace(all_hidden[word], n_dims)
            word_subspaces[word] = S
            # 找到贡献最大的维度
            variances = np.var(all_hidden[word], axis=0)
            top_dims = np.argsort(variances)[-n_dims*2:]  # top 2n_dims
            all_top_dims.update(top_dims.tolist())

        # 只用top-k维度做分类
        top_dim_list = sorted(all_top_dims)
        X_top = X[:, top_dim_list]

        # 用全部维度做分类 (对照)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Top-k维度分类
        clf = LogisticRegression(max_iter=1000)
        scores_top = cross_val_score(clf, X_top, y, cv=5, scoring='accuracy')

        # 随机维度对照
        n_random = len(top_dim_list)
        random_dims = np.random.choice(X.shape[1], n_random, replace=False)
        X_random = X[:, random_dims]
        scores_random = cross_val_score(clf, X_random, y, cv=5, scoring='accuracy')

        # 全维度分类
        scores_full = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        log_time(f"  Classification accuracy (5-fold CV):")
        log_time(f"    Random {n_random} dims: {scores_random.mean():.3f} ± {scores_random.std():.3f}")
        log_time(f"    Top-k {len(top_dim_list)} dims: {scores_top.mean():.3f} ± {scores_top.std():.3f}")
        log_time(f"    All {X.shape[1]} dims:  {scores_full.mean():.3f} ± {scores_full.std():.3f}")
        log_time(f"    Random baseline: {1/len(all_hidden):.3f}")

        # Overlap方法评估
        overlaps = {}
        for w1 in all_hidden:
            for w2 in all_hidden:
                if w1 < w2:
                    ov = compute_overlap(word_subspaces[w1], word_subspaces[w2])
                    overlaps[(w1, w2)] = ov

        log_time(f"  Overlap matrix:")
        sorted_words = sorted(all_hidden.keys())
        header = "          " + "  ".join(f"{w[:6]:>6}" for w in sorted_words)
        log_time(f"  {header}")
        for w1 in sorted_words:
            row = f"  {w1:<8}"
            for w2 in sorted_words:
                if w1 == w2:
                    row += f"  {'1.000':>6}"
                elif (w1, w2) in overlaps:
                    row += f"  {overlaps[(w1,w2)]:>6.3f}"
                else:
                    row += f"  {overlaps[(w2,w1)]:>6.3f}"
            log_time(row)

        results[concept_group] = {
            "words": sorted_words,
            "accuracy_random": round(scores_random.mean(), 4),
            "accuracy_topk": round(scores_top.mean(), 4),
            "accuracy_full": round(scores_full.mean(), 4),
            "n_dims_topk": len(top_dim_list),
            "random_baseline": round(1/len(all_hidden), 4),
            "overlaps": {f"{w1}-{w2}": round(v, 4) for (w1, w2), v in overlaps.items()},
        }

    save_path = RESULT_DIR / f"phase60_part3_{model_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Saved to {save_path}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =====================================================================
# 方案4: 情感多维度分析
# =====================================================================

EMOTION_WORDS = {
    "positive_strong": ["love", "joy", "pride", "gratitude"],
    "positive_mild": ["like", "content", "calm", "pleased"],
    "neutral": ["neutral", "indifferent", "bored", "apathetic"],
    "negative_mild": ["annoyed", "disappointed", "worried", "uneasy"],
    "negative_strong": ["hate", "anger", "fear", "disgust"],
    "negative_deep": ["sadness", "grief", "despair", "agony"],
}

EMOTION_TEMPLATES = {**COOCCURRENCE_TEMPLATES, **_EXTRA_TEMPLATES, **_EXTRA_TEMPLATES_3}

# 补充情感词模板
_EXTRA_EMOTION_TEMPLATES = {
    "joy": [f"She felt joy {v}" for v in [
        "when she saw the sunrise", "at the birth of her child", "in the simple moments",
        "that radiated from within", "as she danced in the rain", "when the music played",
        "in every small victory", "that filled her heart completely", "when laughter erupted",
        "at the unexpected surprise", "that was infectious to all", "when spring finally came",
        "in the company of friends", "that made her eyes sparkle", "when the goal was achieved",
        "that could not be contained", "at the beautiful sight", "when the wait was over",
        "that lifted her spirits high", "in the warm embrace", "when justice was served",
        "that was pure and simple", "at the kind gesture", "when the garden bloomed",
        "that transcended all worries", "in the quiet morning", "when hope returned",
        "that made everything worthwhile", "at the reunion", "when peace was found",
    ]],
    "pride": [f"He felt pride {v}" for v in [
        "in his accomplishment", "at his daughters graduation", "for the team effort",
        "that swelled in his chest", "when the award was announced", "in the finished work",
        "at the recognition received", "that was well deserved", "in his heritage",
        "when the crowd cheered", "for the community achievement", "that was evident to all",
        "in the quality of his work", "at the successful launch", "that motivated him further",
        "in his cultural identity", "when the results came in", "that he carried gracefully",
        "in the family tradition", "at the milestone reached", "that inspired others",
        "in the progress made", "when acknowledged by peers", "that was humble yet firm",
        "in his country", "at the completion of the project", "that never became arrogance",
        "in the collective effort", "when praised sincerely", "in standing up for justice",
    ]],
    "gratitude": [f"She expressed gratitude {v}" for v in [
        "for the kind help", "to her mentor", "for every opportunity",
        "that was heartfelt and sincere", "to the volunteers", "for the support received",
        "in her daily prayers", "that overflowed in her words", "to the medical team",
        "for the second chance", "that was deeply felt", "to all who contributed",
        "for the warm welcome", "that she showed openly", "to her teachers",
        "for the simple things", "that filled her with peace", "to the rescue workers",
        "for the generous gift", "that was evident in her actions", "to her parents",
        "for the beautiful day", "that she practiced daily", "to the community",
        "for the unexpected kindness", "that transformed her outlook", "to the organizers",
        "for the safe return", "that she never took for granted", "to everyone involved",
    ]],
    "content": [f"He felt content {v}" for v in [
        "with his simple life", "as he sat by the fire", "with the days accomplishments",
        "in the quiet evening", "with what he had", "as the sun set slowly",
        "with his modest home", "after a good meal", "with the peaceful surroundings",
        "in his own company", "with the progress made", "as he read his book",
        "with the quiet morning", "after the long walk", "with his chosen path",
        "in the garden", "with the small victories", "as the rain fell gently",
        "with his routine", "in the familiar place", "with the steady pace",
        "after meditation", "with the present moment", "as evening approached",
        "with his decisions", "in the calm space", "with the days work",
        "after the exercise", "with his friendships", "in the still morning",
    ]],
    "calm": [f"The calm {v}" for v in [
        "sea stretched to the horizon", "voice reassured everyone", "morning was peaceful",
        "expression never changed", "breathing helped her focus", "water reflected the sky",
        "atmosphere was soothing", "demeanor put others at ease", "evening was serene",
        "response surprised everyone", "music played softly", "night was still",
        "weather was perfect", "presence was comforting", "lake was mirror smooth",
        "attitude was refreshing", "pace was leisurely", "mood was relaxed",
        "breeze was gentle", "confidence was reassuring", "moment was precious",
        "energy was healing", "silence was welcome", "face showed no worry",
        "handshake was steady", "approach was methodical", "reaction was measured",
        "spirit was unbroken", "resolve was quiet but firm", "wisdom came with stillness",
    ]],
    "pleased": [f"She was pleased {v}" for v in [
        "with the results", "to meet you", "by the unexpected compliment",
        "with the progress made", "at the kind gesture", "with her performance",
        "to announce the winner", "by the positive feedback", "with the outcome",
        "at the improvement shown", "with the arrangement", "to help out",
        "by the thoughtful gift", "with the quality of work", "at the recognition",
        "with his effort", "by the warm reception", "with the final version",
        "at the success rate", "with the choice made", "by the smooth process",
        "with the solution found", "at the attendance", "with the teams effort",
        "by the professional approach", "with the quick response", "at the beautiful result",
        "with the new design", "by the generous offer", "with the overall experience",
    ]],
    "neutral": [f"The neutral {v}" for v in [
        "tone conveyed no emotion", "position was balanced", "color was neither warm nor cool",
        "stance was impartial", "expression gave nothing away", "observer watched quietly",
        "ground was level and even", "response was noncommittal", "party mediated the dispute",
        "perspective was objective", "zone separated the armies", "statement was factual",
        "evaluation was fair", "attitude was professional", "report was unbiased",
        "choice was arbitrary", "voice was steady and even", "reaction was minimal",
        "assessment was clinical", "territory was unclaimed", "finding was inconclusive",
        "opinion was reserved", "view was dispassionate", "approach was methodical",
        "feedback was balanced", "outcome was expected", "review was moderate",
        "analysis was detached", "position was diplomatic", "judgment was impartial",
    ]],
    "indifferent": [f"He was indifferent {v}" for v in [
        "to the outcome", "about the news", "to criticism and praise alike",
        "toward the proposal", "about the controversy", "to the weather",
        "regarding the decision", "about the changes", "to the applause",
        "toward the argument", "about the offer", "to the attention",
        "regarding the results", "about the schedule", "to the complaint",
        "toward the suggestion", "about the fashion", "to the praise",
        "regarding the policy", "about the event", "to the insult",
        "toward the debate", "about the price", "to the invitation",
        "regarding the issue", "about the rumor", "to the opportunity",
        "toward the initiative", "about the performance", "to the remark",
    ]],
    "bored": [f"She was bored {v}" for v in [
        "by the long lecture", "with the routine", "during the meeting",
        "by the repetitive task", "with the same old story", "on the rainy afternoon",
        "by the predictable plot", "with the lack of challenge", "at the waiting room",
        "by the slow pace", "with the empty conversation", "during the flight",
        "by the monotonous voice", "with the uneventful day", "at the back of the class",
        "by the tedious process", "with the familiar scenery", "during the long drive",
        "by the uninteresting topic", "with the idle time", "at the quiet office",
        "by the lengthy explanation", "with nothing to do", "during the delay",
        "by the uninspiring speech", "with the stale routine", "at the empty station",
        "by the dull presentation", "with the lack of stimulation", "during the slow afternoon",
    ]],
    "apathetic": [f"He seemed apathetic {v}" for v in [
        "about the election", "toward the cause", "about the results",
        "to the suffering of others", "regarding the proposal", "about world events",
        "toward the movement", "about the future", "to the call for action",
        "regarding the crisis", "about the opportunity", "toward the appeal",
        "about the consequences", "to the warning", "regarding the problem",
        "about the debate", "toward the initiative", "about the changes",
        "to the encouragement", "regarding the situation", "about the challenge",
        "toward the request", "about the news", "to the suggestion",
        "regarding the issue", "about the event", "toward the effort",
        "about the outcome", "to the criticism", "regarding the development",
    ]],
    "disappointed": [f"She was disappointed {v}" for v in [
        "by the results", "in his performance", "with the outcome",
        "at the news", "by the lack of effort", "in the product quality",
        "with the service", "at the cancellation", "by the broken promise",
        "in herself", "with the experience", "at the rejection",
        "by the mediocre response", "in the leadership", "with the delay",
        "at the failure", "by the poor quality", "in the team",
        "with the progress", "at the turn of events", "by the insufficient support",
        "in the result", "with the explanation", "at the missed opportunity",
        "by the negative feedback", "in the system", "with the communication",
        "at the poor showing", "by the weak performance", "in the final product",
    ]],
    "worried": [f"He was worried {v}" for v in [
        "about the exam", "by the strange noise", "about her health",
        "by the delay", "about the future", "by the warning signs",
        "about the children", "by the rumors", "about the deadline",
        "by the financial situation", "about the weather", "by the test results",
        "about the journey", "by the uncertainty", "about the outcome",
        "by the developments", "about the risk", "by the missing person",
        "about the economy", "by the negative trend", "about the commitment",
        "by the strange behavior", "about the meeting", "by the lack of response",
        "about the project", "by the ominous signs", "about the consequences",
        "by the deteriorating condition", "about the decision", "by the potential danger",
    ]],
    "uneasy": [f"She felt uneasy {v}" for v in [
        "about the situation", "in the dark alley", "about the decision",
        "during the interview", "about the silence", "in the empty house",
        "about the agreement", "around strangers", "about the plan",
        "without explanation", "about the proposal", "in the crowd",
        "about the arrangement", "at the strange sound", "about the direction",
        "when left alone", "about the commitment", "in the unfamiliar place",
        "about the new policy", "after the warning", "about the confrontation",
        "near the edge", "about the outcome", "without clear information",
        "about the implications", "under the scrutiny", "about the change",
        "in the tense atmosphere", "about the responsibility", "with the uncertainty",
    ]],
    "fear": [f"The fear {v}" for v in [
        "was palpable in the room", "gripped her suddenly", "of failure drove him",
        "was etched on his face", "kept her awake at night", "of the unknown was overwhelming",
        "made his hands tremble", "was irrational but real", "of heights was debilitating",
        "paralyzed her completely", "was evident in his voice", "of losing was intense",
        "made her heart race", "was conquered through courage", "of the dark was childish",
        "showed in his wide eyes", "of spiders was common", "motivated the precaution",
        "was a powerful emotion", "of public speaking was real", "made them hesitate",
        "was replaced by determination", "of consequences loomed large", "made her cautious",
        "was ultimately unfounded", "of change held them back", "was a natural response",
        "of rejection hurt deeply", "was written on every face", "of failure was his greatest",
    ]],
    "disgust": [f"The disgust {v}" for v in [
        "was evident on her face", "at the behavior was clear", "made him recoil",
        "was visceral and immediate", "at the corruption was shared", "showed in his expression",
        "was written across her features", "at the waste was obvious", "was overwhelming",
        "at the injustice fueled protest", "was barely concealed", "at the smell was instant",
        "made her stomach turn", "at the betrayal was deep", "was difficult to hide",
        "at the cruelty was profound", "was a natural reaction", "at the display was genuine",
        "made him turn away", "at the suggestion was clear", "was mixed with anger",
        "at the condition was understandable", "showed in her curled lip", "at the behavior was mutual",
        "was beyond words", "at the mess was apparent", "was a gut reaction",
        "at the dishonesty was plain", "made him feel sick", "at the scene was universal",
    ]],
    "sadness": [f"The sadness {v}" for v in [
        "was overwhelming", "in her eyes was unmistakable", "enveloped the room",
        "was deeper than words", "of the occasion was profound", "hung heavy in the air",
        "was tempered by hope", "in his voice was clear", "was a quiet ache",
        "of the memory lingered", "was apparent to everyone", "filled the silence",
        "was a heavy burden", "of loss was shared", "welled up inside her",
        "was written on his face", "of the situation was real", "was palpable",
        "was mixed with gratitude", "of the ending was bittersweet", "was unexpected",
        "came in waves", "of the news was devastating", "was hard to express",
        "was a familiar companion", "of the parting was mutual", "was consuming",
        "was etched in every line", "of the truth was hard to bear", "was undeniable",
    ]],
    "grief": [f"The grief {v}" for v in [
        "was inconsolable", "of the family was profound", "overwhelmed her completely",
        "was a long process", "of the community was shared", "was fresh and raw",
        "took time to process", "of the nation was collective", "was expressed openly",
        "was compounded by guilt", "of the survivors was deep", "was felt by all present",
        "was unbearable at times", "of the mother was beyond words", "was a natural response",
        "was met with support", "of the children was heartbreaking", "was a private matter",
        "was visible in every face", "of the widow was enduring", "was transformed over time",
        "was acknowledged respectfully", "of the colleagues was sincere", "was overwhelming at first",
        "was shared in silence", "of the friends was genuine", "was processed slowly",
        "was expressed through tears", "of the town was palpable", "was eventually accepted",
    ]],
    "despair": [f"The despair {v}" for v in [
        "was absolute", "in her eyes was haunting", "settled over the city",
        "was replaced by determination", "of the situation was clear", "drove him to action",
        "was a dark place", "in his voice was unmistakable", "was followed by hope",
        "of the refugees was documented", "was almost too much to bear", "was temporary",
        "was a powerful motivator", "of the patients was evident", "was overcome through support",
        "was a turning point", "in the community was real", "was met with compassion",
        "was a familiar feeling", "of the victims was acknowledged", "was eventually overcome",
        "was palpable in the room", "of the family was profound", "was not the end",
        "was a call for help", "in the aftermath was expected", "was met with resistance",
        "was written on every face", "of the situation demanded action", "was transformed into resolve",
    ]],
    "agony": [f"The agony {v}" for v in [
        "was excruciating", "of the decision was clear", "was etched on his face",
        "of waiting was unbearable", "was visible in every movement", "of defeat was crushing",
        "was prolonged and intense", "of the patient was addressed", "was beyond description",
        "of the loss was profound", "was reflected in her expression", "of the moment was intense",
        "was both physical and emotional", "of the wound was severe", "was apparent to all",
        "of the choice weighed heavily", "was a private suffering", "of the experience changed her",
        "was evident in his cries", "of the realization was sudden", "was unbearable to witness",
        "of the struggle was real", "was met with empathy", "of the journey was transformative",
        "was a deep and lasting pain", "of the process was necessary", "was overwhelming at times",
        "of the memory persisted", "was a crucible that forged strength", "of the truth was devastating",
    ]],
}

EMOTION_TEMPLATES.update(_EXTRA_EMOTION_TEMPLATES)


def run_part4_emotion_multidim(model_name):
    """
    方案4: 情感多维度分析

    核心问题: 情感低overlap是因为"多维编码"还是"编码不稳定"?
    验证方法:
    - 24个情感词的完整overlap矩阵
    - MDS降维 → 如果形成有序几何结构 → 多维编码
    - 如果随机分布 → 编码不稳定
    """
    log_time(f"=== Part 4: Emotion Multidimensional Analysis ({model_name}) ===")

    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, release_model
    info = get_model_info(model, model_name)

    n_layers = info.n_layers
    target_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
    ]

    # 收集所有情感词
    all_emotion_words = []
    word_categories = {}
    for category, words in EMOTION_WORDS.items():
        for w in words:
            all_emotion_words.append(w)
            word_categories[w] = category

    log_time(f"Collecting {len(all_emotion_words)} emotion words")

    word_hidden = {}
    for wi, word in enumerate(all_emotion_words):
        sents = EMOTION_TEMPLATES.get(word, COOCCURRENCE_TEMPLATES.get(word, []))[:30]
        if len(sents) < 5:
            log_time(f"  WARNING: only {len(sents)} templates for {word}")
            continue

        hidden = collect_hidden_states(model, tokenizer, device, sents, target_layers)
        word_hidden[word] = hidden
        log_time(f"  Word {wi+1}/{len(all_emotion_words)}: {word}")

    results = {"model": model_name, "target_layers": target_layers}

    for li in target_layers:
        n_dims = 10
        available_words = sorted([w for w in all_emotion_words if w in word_hidden])

        # 计算完整overlap矩阵
        overlap_matrix = np.zeros((len(available_words), len(available_words)))
        for i, w1 in enumerate(available_words):
            S1 = extract_subspace(word_hidden[w1][li], n_dims)
            for j, w2 in enumerate(available_words):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                elif j > i:
                    S2 = extract_subspace(word_hidden[w2][li], n_dims)
                    ov = compute_overlap(S1, S2)
                    overlap_matrix[i, j] = ov
                    overlap_matrix[j, i] = ov

        log_time(f"\nLayer {li} Overlap Matrix ({len(available_words)} words):")

        # 打印overlap矩阵 (简化)
        log_time(f"  {'Word':<12} " + " ".join(f"{w[:6]:>6}" for w in available_words[:12]))
        for i, w in enumerate(available_words[:12]):
            row = f"  {w:<12} " + " ".join(f"{overlap_matrix[i,j]:>6.3f}" for j in range(min(12, len(available_words))))
            log_time(row)

        # MDS降维
        from sklearn.manifold import MDS
        distance_matrix = 1 - overlap_matrix
        np.clip(distance_matrix, 0, None, out=distance_matrix)

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
        positions = mds.fit_transform(distance_matrix)

        # 分析MDS结构的有序性
        # 1. 按类别着色, 检查是否形成聚类
        category_positions = defaultdict(list)
        for i, w in enumerate(available_words):
            category_positions[word_categories[w]].append(positions[i])

        # 计算类内距离 vs 类间距离
        intra_dists = []
        inter_dists = []
        cats = list(category_positions.keys())
        for ci, cat1 in enumerate(cats):
            for cj, cat2 in enumerate(cats):
                for p1 in category_positions[cat1]:
                    for p2 in category_positions[cat2]:
                        d = np.linalg.norm(p1 - p2)
                        if ci == cj:
                            intra_dists.append(d)
                        else:
                            inter_dists.append(d)

        intra_mean = np.mean(intra_dists) if intra_dists else 0
        inter_mean = np.mean(inter_dists) if inter_dists else 0
        cluster_score = inter_mean / (intra_mean + 1e-10)

        log_time(f"  MDS cluster score: {cluster_score:.3f} (>1 = categories cluster together)")
        log_time(f"  Intra-category distance: {intra_mean:.3f}")
        log_time(f"  Inter-category distance: {inter_mean:.3f}")

        # 2. 检查是否有环状/轴状结构
        # 按valence排序 (positive → neutral → negative), 检查是否在2D上有序
        valence_order = ["positive_strong", "positive_mild", "neutral", "negative_mild", "negative_strong", "negative_deep"]
        valence_positions = []
        for cat in valence_order:
            if cat in category_positions:
                valence_positions.append(np.mean(category_positions[cat], axis=0))

        # 检查valence_positions是否形成有序曲线 (计算路径长度 vs 直线距离)
        if len(valence_positions) >= 3:
            path_length = sum(np.linalg.norm(np.array(valence_positions[i+1]) - np.array(valence_positions[i]))
                              for i in range(len(valence_positions)-1))
            direct_length = np.linalg.norm(np.array(valence_positions[-1]) - np.array(valence_positions[0]))
            curvature = path_length / (direct_length + 1e-10)
            log_time(f"  Valence curvature: {curvature:.3f} (>1 = curved/axis structure)")

        # 3. 检查intensity维度: strong vs mild是否可分离
        strong_positions = []
        mild_positions = []
        for cat in ["positive_strong", "negative_strong", "negative_deep"]:
            if cat in category_positions:
                strong_positions.extend(category_positions[cat])
        for cat in ["positive_mild", "negative_mild"]:
            if cat in category_positions:
                mild_positions.extend(category_positions[cat])

        if strong_positions and mild_positions:
            strong_center = np.mean(strong_positions, axis=0)
            mild_center = np.mean(mild_positions, axis=0)
            intensity_separation = np.linalg.norm(strong_center - mild_center)
            log_time(f"  Strong vs Mild separation: {intensity_separation:.3f}")

        # 类内overlap vs 类间overlap
        intra_overlaps = []
        inter_overlaps = []
        for i, w1 in enumerate(available_words):
            for j, w2 in enumerate(available_words):
                if i < j:
                    ov = overlap_matrix[i, j]
                    if word_categories[w1] == word_categories[w2]:
                        intra_overlaps.append(ov)
                    else:
                        inter_overlaps.append(ov)

        log_time(f"  Intra-category avg overlap: {np.mean(intra_overlaps):.4f}")
        log_time(f"  Inter-category avg overlap: {np.mean(inter_overlaps):.4f}")
        log_time(f"  Ratio: {np.mean(intra_overlaps)/(np.mean(inter_overlaps)+1e-10):.3f}")

        results[f"layer_{li}"] = {
            "words": available_words,
            "overlap_matrix": overlap_matrix.tolist(),
            "mds_positions": positions.tolist(),
            "word_categories": {w: word_categories[w] for w in available_words},
            "cluster_score": round(cluster_score, 4),
            "intra_overlap": round(np.mean(intra_overlaps), 4) if intra_overlaps else 0,
            "inter_overlap": round(np.mean(inter_overlaps), 4) if inter_overlaps else 0,
        }

    save_path = RESULT_DIR / f"phase60_part4_{model_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Saved to {save_path}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =====================================================================
# Main
# =====================================================================

model_name_global = ""  # 全局变量, collect_hidden_states需要

def main():
    global model_name_global

    parser = argparse.ArgumentParser(description="Phase 60: Causal Validation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    model_name_global = args.model

    if args.part == "1" or args.part == "all":
        run_part1_cooccurrence(args.model)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    if args.part == "2" or args.part == "all":
        run_part2_intensity(args.model)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    if args.part == "3" or args.part == "all":
        run_part3_discrimination(args.model)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    if args.part == "4" or args.part == "all":
        run_part4_emotion_multidim(args.model)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    log_time("Phase 60 complete!")


if __name__ == "__main__":
    main()
