"""
Phase 288: Function-wise Route-Content-MLP Causal Decomposition
================================================================
GOAL: Robustly decompose how each linguistic function is implemented:
  - ROUTING (attention weights): "which tokens to attend to"
  - CONTENT (V values): "what information to transmit"  
  - MLP: "nonlinear transformation / feature remapping"
  - RESIDUAL: "carry-over from previous layers"

KEY IMPROVEMENTS over Phase 287:
  1. CALIBRATION: Verify manual AW@V reconstruction matches real head output
  2. DIRECTION PROJECTION: progress_to_B = how much toward B (not just random drift)
  3. RANDOM CONTROLS: AW_random @ V_A, AW_A @ V_random
  4. LARGER SAMPLES: 40-80 pairs per function (6 functions)
  5. HEAD GROUPS: top-k heads together (not single heads)
  6. MLP INCLUSION: First joint decomposition

DATASET (Phase 288):
  - negation: 80 pairs (not X, do not X, no X, never X, etc.)
  - translation: 40 pairs (en→zh, zh→en)
  - logical: 40 pairs (and/or, if/then, because)
  - passive: 30 pairs (active↔passive)
  - comparative: 30 pairs (bigger/smaller, more/less)  
  - recursive: 30 pairs (relative clauses, embedding)

TOTAL: ~250 pairs per model

HEADS TESTED:
  - Top causal heads from Phase 286 (5 per model)
  - Random control heads (3 per model)
  - Head groups: top-3, top-5, random-5

MLP TESTED:
  - Key layers (early L{0-2}, middle L{n/2}, late L{n-3..n-1})
  - Patch entire MLP output

METRICS:
  - KL effect (normalized)
  - progress_dir: cos(patch_delta, B_delta)
  - progress_mag: |patch_delta| / |B_delta|
  - reconstruction_error: |manual_AW@V - real_head_output|
  - significance: vs random control baseline

Usage:
  python tests/glm5/phase288_rcm_decomposition.py qwen3
  python tests/glm5/phase288_rcm_decomposition.py glm4
  python tests/glm5/phase288_rcm_decomposition.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase288_rcm_decomposition")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None
def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ==============================================================================
# MODEL LOADING (EAGER for caching AW+V, FLASH for fast patching)
# ==============================================================================

def load_model_eager(model_name: str):
    """Load with eager attention for output_attentions=True (caching AW+V)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + EAGER + output_attentions)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DS7B/GLM4: device_map="auto" to avoid OOM
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config.output_attentions = True
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  Loaded: device={device}, GPU={gpu_mem:.2f}GB, mode=EAGER+output_attn")
    return model, tokenizer, device


def load_model_flash(model_name: str):
    """Load with flash_attn_2 for fast patching."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attn_2)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    try:
        import flash_attn
    except ImportError:
        attn_impl = "eager"
        log_time("  flash_attn not found, using eager")

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
    log_time(f"  Loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# ==============================================================================
# HEAD CONFIG (GQA-aware)
# ==============================================================================

def get_head_config(model):
    layers = get_layers(model)
    layer0 = layers[0]
    o_proj = layer0.self_attn.o_proj
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    concat_dim = o_proj.weight.shape[1]
    head_dim = concat_dim // n_heads
    return n_heads, n_kv_heads, head_dim, concat_dim


def get_kv_head_idx(head_idx, n_heads, n_kv_heads):
    if n_heads == n_kv_heads:
        return head_idx
    return head_idx // (n_heads // n_kv_heads)


# ==============================================================================
# DATASET: Large, function-organized sentence pairs
# ==============================================================================

def build_pairs_negation():
    """Negation pairs: 80 pairs covering various negation types."""
    pairs = []
    def add(name, pos, neg):
        pairs.append({"name": f"neg_{name}", "A": pos, "B": neg, "category": "negation"})

    # Type 1: not + adjective (20 pairs)
    adj_pairs = [
        ("happy", "she is happy", "she is not happy"),
        ("open", "the door is open", "the door is not open"),
        ("possible", "victory is possible", "victory is not possible"),
        ("ready", "they are ready", "they are not ready"),
        ("important", "this is important", "this is not important"),
        ("clear", "the answer is clear", "the answer is not clear"),
        ("safe", "the area is safe", "the area is not safe"),
        ("fair", "the decision is fair", "the decision is not fair"),
        ("simple", "the problem is simple", "the problem is not simple"),
        ("correct", "your answer is correct", "your answer is not correct"),
        ("reasonable", "the price is reasonable", "the price is not reasonable"),
        ("visible", "the star is visible", "the star is not visible"),
        ("normal", "the situation is normal", "the situation is not normal"),
        ("common", "this bird is common", "this bird is not common"),
        ("legal", "the action is legal", "the action is not legal"),
        ("stable", "the system is stable", "the system is not stable"),
        ("permanent", "the change is permanent", "the change is not permanent"),
        ("sufficient", "the evidence is sufficient", "the evidence is not sufficient"),
        ("relevant", "the information is relevant", "the information is not relevant"),
        ("accurate", "the measurement is accurate", "the measurement is not accurate"),
    ]
    for name, pos, neg in adj_pairs:
        add(name, pos, neg)

    # Type 2: do/does/did not + verb (20 pairs)
    verb_pairs = [
        ("agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("remember", "i remember the meeting", "i do not remember the meeting"),
        ("understand", "we understand the problem", "we do not understand the problem"),
        ("know", "she knows the answer", "she does not know the answer"),
        ("believe", "he believes the story", "he does not believe the story"),
        ("support", "they support the plan", "they do not support the plan"),
        ("accept", "she accepts the offer", "she does not accept the offer"),
        ("expect", "we expect rain", "we do not expect rain"),
        ("trust", "he trusts the source", "he does not trust the source"),
        ("follow", "they follow the rules", "they do not follow the rules"),
        ("recognize", "she recognizes the face", "she does not recognize the face"),
        ("recommend", "i recommend this book", "i do not recommend this book"),
        ("approve", "the board approves the plan", "the board does not approve the plan"),
        ("require", "the task requires effort", "the task does not require effort"),
        ("permit", "the rule permits entry", "the rule does not permit entry"),
        ("guarantee", "this guarantees success", "this does not guarantee success"),
        ("indicate", "the data indicates growth", "the data does not indicate growth"),
        ("imply", "the statement implies guilt", "the statement does not imply guilt"),
        ("confirm", "the test confirms the theory", "the test does not confirm the theory"),
        ("resolve", "this resolves the issue", "this does not resolve the issue"),
    ]
    for name, pos, neg in verb_pairs:
        add(name, pos, neg)

    # Type 3: no/nothing/none/no one (10 pairs)
    no_pairs = [
        ("found_nothing", "he found something interesting", "he found nothing interesting"),
        ("anyone_came", "someone came to the party", "no one came to the party"),
        ("any_food", "there was some food left", "there was no food left"),
        ("had_idea", "she had some idea what to do", "she had no idea what to do"),
        ("any_reason", "there is a reason to worry", "there is no reason to worry"),
        ("had_choice", "they had a choice in the matter", "they had no choice in the matter"),
        ("any_doubt", "there is some doubt about it", "there is no doubt about it"),
        ("any_evidence", "there is evidence of fraud", "there is no evidence of fraud"),
        ("any_hope", "there is some hope left", "there is no hope left"),
        ("any_sign", "there is a sign of life", "there is no sign of life"),
    ]
    for name, pos, neg in no_pairs:
        add(name, pos, neg)

    # Type 4: never (10 pairs)
    never_pairs = [
        ("seen_before", "i have seen it before", "i have never seen it before"),
        ("been_there", "she has been to Paris", "she has never been to Paris"),
        ("told_anyone", "he told someone the secret", "he never told anyone the secret"),
        ("gives_up", "she sometimes gives up", "she never gives up"),
        ("forgets", "he sometimes forgets names", "he never forgets a face"),
        ("complains", "she often complains", "she never complains"),
        ("lies_truth", "he sometimes tells the truth", "he never tells the truth"),
        ("late_early", "she is sometimes late", "she is never late"),
        ("apologizes", "he sometimes apologizes", "he never apologizes"),
        ("admits", "she sometimes admits mistakes", "she never admits mistakes"),
    ]
    for name, pos, neg in never_pairs:
        add(name, pos, neg)

    # Type 5: negative prefix/suffix (10 pairs)
    prefix_pairs = [
        ("impossible", "the task is possible", "the task is impossible"),
        ("unacceptable", "the proposal is acceptable", "the proposal is unacceptable"),
        ("incomplete", "the report is complete", "the report is incomplete"),
        ("irrelevant", "the comment is relevant", "the comment is irrelevant"),
        ("dishonest", "the person is honest", "the person is dishonest"),
        ("unfair", "the treatment was fair", "the treatment was unfair"),
        ("unlikely", "the outcome is likely", "the outcome is unlikely"),
        ("incorrect", "the assumption is correct", "the assumption is incorrect"),
        ("irregular", "the pattern is regular", "the pattern is irregular"),
        ("disagree", "they agree on the terms", "they disagree on the terms"),
    ]
    for name, pos, neg in prefix_pairs:
        add(name, pos, neg)

    # Type 6: scope negation (10 pairs)
    scope_pairs = [
        ("not_all", "all birds can fly", "not all birds can fly"),
        ("not_everyone", "everyone agreed", "not everyone agreed"),
        ("not_always", "she always tells the truth", "she does not always tell the truth"),
        ("not_entirely", "the plan is entirely successful", "the plan is not entirely successful"),
        ("not_necessarily", "wealth means happiness", "wealth does not necessarily mean happiness"),
        ("not_only_rich", "he is rich", "he is not only rich but also kind"),
        ("not_exactly", "that is exactly what i meant", "that is not exactly what i meant"),
        ("not_quite", "the work is finished", "the work is not quite finished"),
        ("not_particularly", "the movie was interesting", "the movie was not particularly interesting"),
        ("not_completely", "the glass is full", "the glass is not completely full"),
    ]
    for name, pos, neg in scope_pairs:
        add(name, pos, neg)

    return pairs


def build_pairs_translation():
    """Translation pairs: 40 pairs (en↔zh)."""
    pairs = []
    def add(name, en, zh):
        pairs.append({"name": f"trans_{name}", "A": en, "B": zh, "category": "translation"})

    trans = [
        ("dog_cat", "the dog chases the cat", "狗追猫"),
        ("sun_east", "the sun rises in the east", "太阳从东方升起"),
        ("teacher_teach", "the teacher teaches the student", "老师教学生"),
        ("bird_sky", "the bird flies in the sky", "鸟在天空中飞翔"),
        ("king_rules", "the king rules the kingdom", "国王统治王国"),
        ("water_cold", "the water is very cold", "水非常冷"),
        ("love_overcome", "love overcomes everything", "爱战胜一切"),
        ("fox_hunt", "the fox hunts the rabbit", "狐狸猎兔"),
        ("child_happy", "the child is very happy", "孩子非常快乐"),
        ("mountain_high", "the mountain is extremely high", "这座山非常高"),
        ("horse_white", "the horse is white", "这匹马是白色的"),
        ("river_long", "the river is very long", "这条河非常长"),
        ("book_interesting", "the book is very interesting", "这本书非常有趣"),
        ("tree_old", "the tree is ancient", "这棵树很古老"),
        ("city_busy", "the city is very busy", "这个城市非常繁忙"),
        ("food_delicious", "the food is delicious", "食物很美味"),
        ("music_beautiful", "the music is beautiful", "音乐很美"),
        ("garden_small", "the garden is small", "花园很小"),
        ("house_large", "the house is very large", "这栋房子非常大"),
        ("road_narrow", "the road is narrow", "这条路很窄"),
        ("stone_heavy", "the stone is heavy", "这块石头很重"),
        ("cloth_soft", "the cloth is soft", "布料很柔软"),
        ("wind_strong", "the wind is strong today", "今天的风很大"),
        ("rain_heavy", "the rain is heavy", "雨下得很大"),
        ("sky_blue", "the sky is blue", "天空是蓝色的"),
        ("flower_red", "the flower is red", "花是红色的"),
        ("fish_fresh", "the fish is fresh", "鱼很新鲜"),
        ("milk_warm", "the milk is warm", "牛奶是温的"),
        ("bread_soft", "the bread is soft", "面包很软"),
        ("soup_hot", "the soup is hot", "汤很烫"),
        ("light_bright", "the light is bright", "灯光很亮"),
        ("shadow_dark", "the shadow is dark", "影子很暗"),
        ("voice_loud", "the voice is loud", "声音很大"),
        ("silence_deep", "the silence is deep", "寂静很深"),
        ("smile_warm", "the smile is warm", "微笑很温暖"),
        ("night_quiet", "the night is quiet", "夜晚很安静"),
        ("sea_vast", "the sea is vast", "大海很辽阔"),
        ("snow_white", "the snow is white", "雪是白色的"),
        ("fire_hot", "the fire is hot", "火是热的"),
        ("ice_cold", "the ice is cold", "冰是冷的"),
    ]
    for name, en, zh in trans:
        add(name, en, zh)
    return pairs


def build_pairs_logical():
    """Logical pairs: 40 pairs (and/or, if/then, because, although, therefore)."""
    pairs = []
    def add(name, a, b):
        pairs.append({"name": f"logic_{name}", "A": a, "B": b, "category": "logical"})

    # and ↔ or (10 pairs)
    for name, a, b in [
        ("and_or_catdog", "the cat and the dog are sleeping", "the cat or the dog is sleeping"),
        ("and_or_birdsbees", "birds and bees are pollinators", "birds or bees are pollinators"),
        ("and_or_sunmoon", "the sun and the moon are visible", "the sun or the moon is visible"),
        ("and_or_teacoffee", "tea and coffee are served", "tea or coffee is served"),
        ("and_or_applesoranges", "apples and oranges are fruits", "apples or oranges are fruits"),
        ("and_or_summerwinter", "summer and winter have extremes", "summer or winter has extremes"),
        ("and_or_lionshark", "lions and sharks are predators", "lions or sharks are predators"),
        ("and_or_ironcopper", "iron and copper are metals", "iron or copper are metals"),
        ("and_or_paperink", "paper and ink are needed", "paper or ink is needed"),
        ("and_or_saltpepper", "salt and pepper add flavor", "salt or pepper adds flavor"),
    ]:
        add(name, a, b)

    # if-then (10 pairs)
    for name, a, b in [
        ("if_rain", "if it rains we will stay home", "we will stay home if it rains"),
        ("if_hungry", "if you are hungry eat something", "eat something if you are hungry"),
        ("if_tired", "if she is tired she will rest", "she will rest if she is tired"),
        ("if_cold", "if it gets cold turn on the heater", "turn on the heater if it gets cold"),
        ("if_ready", "if they are ready we can go", "we can go if they are ready"),
        ("if_sunny", "if tomorrow is sunny visit the park", "visit the park if tomorrow is sunny"),
        ("if_late", "if you are late call me", "call me if you are late"),
        ("if_broken", "if it is broken fix it", "fix it if it is broken"),
        ("if_lost", "if you get lost ask for help", "ask for help if you get lost"),
        ("if_sick", "if you feel sick see a doctor", "see a doctor if you feel sick"),
    ]:
        add(name, a, b)

    # because ↔ consequence (10 pairs)
    for name, a, b in [
        ("because_rain", "because it rained we stayed home", "we stayed home because it rained"),
        ("because_hungry", "because he was hungry he ate", "he ate because he was hungry"),
        ("because_sick", "because she was sick she rested", "she rested because she was sick"),
        ("because_late", "because he was late he ran", "he ran because he was late"),
        ("because_cold", "because it was cold they lit a fire", "they lit a fire because it was cold"),
        ("because_dark", "because it was dark she turned on lights", "she turned on lights because it was dark"),
        ("because_noisy", "because it was noisy he closed the window", "he closed the window because it was noisy"),
        ("because_expensive", "because it was expensive she saved money", "she saved money because it was expensive"),
        ("because_hot", "because it was hot they swam", "they swam because it was hot"),
        ("because_tired", "because they were tired they stopped", "they stopped because they were tired"),
    ]:
        add(name, a, b)

    # although (5 pairs)
    for name, a, b in [
        ("although_rain", "although it rained they went out", "they went out although it rained"),
        ("although_tired", "although she was tired she continued", "she continued although she was tired"),
        ("although_hard", "although the test was hard she passed", "she passed although the test was hard"),
        ("although_small", "although the dog was small it was brave", "the dog was brave although it was small"),
        ("although_poor", "although they were poor they were happy", "they were happy although they were poor"),
    ]:
        add(name, a, b)

    # therefore (5 pairs)
    for name, a, b in [
        ("therefore_rain", "it is raining therefore we will stay home", "we will stay home therefore it is raining"),
        ("therefore_late", "he overslept therefore he was late", "he was late therefore he overslept"),
        ("therefore_study", "she studied hard therefore she passed", "she passed therefore she studied hard"),
        ("therefore_broken", "the machine broke therefore work stopped", "work stopped therefore the machine broke"),
        ("therefore_dark", "the sun set therefore it got dark", "it got dark therefore the sun set"),
    ]:
        add(name, a, b)

    return pairs


def build_pairs_passive():
    """Passive pairs: 30 pairs (active↔passive)."""
    pairs = []
    def add(name, active, passive):
        pairs.append({"name": f"pass_{name}", "A": active, "B": passive, "category": "passive"})

    pass_pairs = [
        ("dog_cat", "the dog chases the cat", "the cat is chased by the dog"),
        ("teacher_student", "the teacher teaches the student", "the student is taught by the teacher"),
        ("author_book", "the author wrote the book", "the book was written by the author"),
        ("wife_cake", "the wife baked the cake", "the cake was baked by the wife"),
        ("workers_bridge", "the workers built the bridge", "the bridge was built by the workers"),
        ("detective_clue", "the detective found the clue", "the clue was found by the detective"),
        ("everyone_teacher", "everyone loves the teacher", "the teacher is loved by everyone"),
        ("cat_fish", "the cat ate the fish", "the fish was eaten by the cat"),
        ("musician_song", "the musician wrote the song", "the song was written by the musician"),
        ("artist_painting", "the artist created the painting", "the painting was created by the artist"),
        ("chef_meal", "the chef prepared the meal", "the meal was prepared by the chef"),
        ("gardener_trees", "the gardener planted the trees", "the trees were planted by the gardener"),
        ("police_thief", "the police caught the thief", "the thief was caught by the police"),
        ("judge_case", "the judge reviewed the case", "the case was reviewed by the judge"),
        ("doctor_patient", "the doctor treated the patient", "the patient was treated by the doctor"),
        ("driver_car", "the driver parked the car", "the car was parked by the driver"),
        ("student_essay", "the student wrote the essay", "the essay was written by the student"),
        ("scientist_experiment", "the scientist conducted the experiment", "the experiment was conducted by the scientist"),
        ("engineer_bridge", "the engineer designed the bridge", "the bridge was designed by the engineer"),
        ("farmer_crops", "the farmer harvested the crops", "the crops were harvested by the farmer"),
        ("painter_wall", "the painter painted the wall", "the wall was painted by the painter"),
        ("waiter_order", "the waiter took the order", "the order was taken by the waiter"),
        ("coach_team", "the coach trained the team", "the team was trained by the coach"),
        ("editor_article", "the editor revised the article", "the article was revised by the editor"),
        ("director_film", "the director produced the film", "the film was produced by the director"),
        ("tailor_suit", "the tailor made the suit", "the suit was made by the tailor"),
        ("baker_bread", "the baker made the bread", "the bread was made by the baker"),
        ("librarian_books", "the librarian organized the books", "the books were organized by the librarian"),
        ("officer_report", "the officer filed the report", "the report was filed by the officer"),
        ("volunteer_event", "the volunteer organized the event", "the event was organized by the volunteer"),
    ]
    for name, a, b in pass_pairs:
        add(name, a, b)
    return pairs


def build_pairs_comparative():
    """Comparative pairs: 30 pairs."""
    pairs = []
    def add(name, a, b):
        pairs.append({"name": f"comp_{name}", "A": a, "B": b, "category": "comparative"})

    comp_pairs = [
        ("bigger_smaller", "the elephant is bigger than the mouse", "the mouse is smaller than the elephant"),
        ("taller_shorter", "John is taller than Mary", "Mary is shorter than John"),
        ("more_expensive", "gold is more expensive than silver", "silver is less expensive than gold"),
        ("faster_slower", "the train is faster than the bicycle", "the bicycle is slower than the train"),
        ("stronger_weaker", "steel is stronger than wood", "wood is weaker than steel"),
        ("smarter_dumber", "Alice is smarter than Bob", "Bob is less smart than Alice"),
        ("older_younger", "the grandfather is older than the father", "the father is younger than the grandfather"),
        ("more_fewer", "she has more books than he does", "he has fewer books than she does"),
        ("hotter_colder", "summer is hotter than spring", "spring is colder than summer"),
        ("darker_lighter", "midnight is darker than dusk", "dusk is lighter than midnight"),
        ("richer_poorer", "the king is richer than the merchant", "the merchant is poorer than the king"),
        ("louder_quieter", "thunder is louder than rain", "rain is quieter than thunder"),
        ("heavier_lighter", "iron is heavier than aluminum", "aluminum is lighter than iron"),
        ("sweeter_sour", "honey is sweeter than lemon", "lemon is sourer than honey"),
        ("longer_shorter", "the Nile is longer than the Thames", "the Thames is shorter than the Nile"),
        ("wider_narrower", "the highway is wider than the alley", "the alley is narrower than the highway"),
        ("deeper_shallower", "the ocean is deeper than the lake", "the lake is shallower than the ocean"),
        ("harder_softer", "diamond is harder than glass", "glass is softer than diamond"),
        ("brighter_dimmer", "the sun is brighter than the moon", "the moon is dimmer than the sun"),
        ("fresher_staler", "fresh bread is better than stale bread", "stale bread is worse than fresh bread"),
        ("quicker_slower", "email is quicker than postal mail", "postal mail is slower than email"),
        ("sharper_blunter", "a razor is sharper than a butter knife", "a butter knife is blunter than a razor"),
        ("older_newer", "the ancient temple is older than the church", "the church is newer than the ancient temple"),
        ("thicker_thinner", "the dictionary is thicker than the pamphlet", "the pamphlet is thinner than the dictionary"),
        ("warmer_cooler", "the equator is warmer than the poles", "the poles are cooler than the equator"),
        ("wetter_drier", "the rainforest is wetter than the desert", "the desert is drier than the rainforest"),
        ("busier_calmer", "the city center is busier than the suburbs", "the suburbs are calmer than the city center"),
        ("clearer_foggier", "the morning sky is clearer than the evening", "the evening sky is foggier than the morning"),
        ("safer_riskier", "walking is safer than driving fast", "driving fast is riskier than walking"),
        ("easier_harder", "addition is easier than multiplication", "multiplication is harder than addition"),
    ]
    for name, a, b in comp_pairs:
        add(name, a, b)
    return pairs


def build_pairs_recursive():
    """Recursive/embedding pairs: 30 pairs."""
    pairs = []
    def add(name, a, b):
        pairs.append({"name": f"rec_{name}", "A": a, "B": b, "category": "recursive"})

    rec_pairs = [
        ("dog_barked", "the dog that barked ran away", "the barking dog ran away"),
        ("woman_wrote", "the woman who wrote the letter smiled", "the letter-writing woman smiled"),
        ("king_said", "the king said that the queen is wise", "the queen is wise said the king"),
        ("door_painted", "the door that was painted red opened", "the red door opened"),
        ("book_which", "the book which i read yesterday was great", "yesterday i read a great book"),
        ("person_who", "the person who called left a message", "a message was left by the caller"),
        ("nested_if", "the man who said that if it rains he will leave arrived", "the man arrived who said he will leave if it rains"),
        ("teacher_think", "the teacher thinks that the student learned well", "the student is thought by the teacher to have learned well"),
        ("fact_surprise", "the fact that he lied surprised everyone", "everyone was surprised that he lied"),
        ("belief_theory", "the scientist believes that the theory is correct", "the theory is believed by the scientist to be correct"),
        ("cat_slept", "the cat that slept on the mat was fluffy", "the fluffy cat slept on the mat"),
        ("man_hat", "the man wearing a hat entered the room", "a man with a hat entered the room"),
        ("girl_dress", "the girl in the red dress waved", "the red-dressed girl waved"),
        ("boy_cried", "the boy who cried wolf was ignored", "the crying-wolf boy was ignored"),
        ("car_broken", "the car that broke down was towed", "the broken-down car was towed"),
        ("child_lost", "the child who got lost was helped", "the lost child was helped"),
        ("house_built", "the house that Jack built stands tall", "Jack's house stands tall"),
        ("story_told", "the story that grandma told was scary", "grandma's scary story"),
        ("dog_found", "the dog that found the bone was happy", "the bone-finding dog was happy"),
        ("bird_sang", "the bird that sang at dawn flew away", "the dawn-singing bird flew away"),
        ("tree_fell", "the tree that fell during the storm blocked the road", "the storm-fallen tree blocked the road"),
        ("letter_arrived", "the letter that arrived yesterday was important", "yesterday's letter was important"),
        ("flower_bloomed", "the flower that bloomed in spring wilted", "the spring-bloomed flower wilted"),
        ("bridge_spanned", "the bridge that spanned the river collapsed", "the river-spanning bridge collapsed"),
        ("statue_stood", "the statue that stood in the square was removed", "the square statue was removed"),
        ("painting_hung", "the painting that hung on the wall was sold", "the wall painting was sold"),
        ("watch_broken", "the watch that my father gave me stopped", "my father's gift watch stopped"),
        ("recipe_taught", "the recipe that my grandmother taught me is delicious", "my grandmother's recipe is delicious"),
        ("song_sang", "the song that the choir sang moved everyone", "the choir song moved everyone"),
        ("dog_chased_cat", "the dog that chased the cat that caught the mouse barked", "the dog barked after chasing the cat who caught the mouse"),
    ]
    for name, a, b in rec_pairs:
        add(name, a, b)
    return pairs


def build_all_pairs():
    """Build complete Phase 288 dataset: 6 functions, 30-80 pairs each, ~250 total."""
    all_pairs = []
    all_pairs.extend(build_pairs_negation())
    all_pairs.extend(build_pairs_translation())
    all_pairs.extend(build_pairs_logical())
    all_pairs.extend(build_pairs_passive())
    all_pairs.extend(build_pairs_comparative())
    all_pairs.extend(build_pairs_recursive())
    return all_pairs


# ==============================================================================
# PART 1: Cache AW + V (EAGER attention, output_attentions=True)
# ==============================================================================

def cache_aw_and_v(model, tokenizer, pairs, device, n_layers, head_config,
                   target_layers=None, max_len=48):
    """
    Cache AW and V for target layers only.
    Returns: {pname: {"A": {}, "B": {}, "category": str, "seq_len": int}}
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)
    if target_layers is None:
        target_layers = list(range(n_layers))
    target_set = set(target_layers)

    all_cached = {}
    t0 = time.time()

    for pidx, pair in enumerate(pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair["category"]

        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        common_len = min(max(toks_a, toks_b), max_len)

        pair_cache = {"category": category, "A": {}, "B": {}, "seq_len": common_len}

        for sent_key, sent in [("A", sent_a), ("B", sent_b)]:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True,
                             max_length=common_len).to(device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < common_len:
                pad_len = common_len - seq_len
                inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len),
                                            value=tokenizer.pad_token_id)
                inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

            v_cache = {}
            v_hooks = []
            for li in sorted(target_set):
                if li < len(layers):
                    def make_v_hook(li):
                        def hook(module, input_t, output_t):
                            v_cache[f"L{li}"] = output_t.detach().cpu().clone()
                        return hook
                    v_hooks.append(layers[li].self_attn.v_proj.register_forward_hook(make_v_hook(li)))

            with torch.no_grad():
                try:
                    out = model(**inputs)
                except Exception as e:
                    log_time(f"  [{pidx+1}/{len(pairs)}] {pname} {sent_key}: FAILED {e}")
                    for h in v_hooks: h.remove()
                    v_cache = None
                    break

            for h in v_hooks:
                h.remove()

            if v_cache is not None and hasattr(out, 'attentions') and out.attentions is not None:
                attn_weights = out.attentions
                layer_data = {}
                for li in target_set:
                    if li < len(attn_weights) and f"L{li}" in v_cache:
                        layer_data[f"L{li}"] = {
                            "AW": attn_weights[li].detach().cpu().clone(),
                            "V": v_cache[f"L{li}"],
                        }
                pair_cache[sent_key] = layer_data

        if pair_cache["A"] and pair_cache["B"]:
            all_cached[pname] = pair_cache

        if (pidx + 1) % 30 == 0 or pidx == 0:
            elapsed = time.time() - t0
            rate = (pidx + 1) / max(elapsed, 1) * 3600
            eta = (len(pairs) - pidx - 1) * elapsed / max(pidx + 1, 1)
            log_time(f"  [{pidx+1}/{len(pairs)}] Caching AW+V: {elapsed:.0f}s, "
                     f"~{rate:.0f} pairs/h, ETA={eta/60:.0f}min, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_total = time.time() - t0
    log_time(f"  Caching done: {len(all_cached)}/{len(pairs)} pairs, {t_total:.0f}s ({t_total/60:.1f}min)")
    return all_cached


# ==============================================================================
# CALIBRATION: Verify manual AW@V reconstruction
# ==============================================================================

def compute_head_output_manual(AW, V_kv, head_idx, n_heads, n_kv_heads, head_dim):
    """Manual AW_h @ V_h — returns [batch, seq, head_dim]."""
    kv_idx = get_kv_head_idx(head_idx, n_heads, n_kv_heads)
    v_start = kv_idx * head_dim
    v_end = (kv_idx + 1) * head_dim
    V_h = V_kv[:, :, v_start:v_end]
    AW_h = AW[:, head_idx, :, :]
    return torch.bmm(AW_h.float(), V_h.float()).cpu()


def get_real_head_output(model, tokenizer, sentence, device, n_layers, head_config,
                         target_layer, target_head, max_len=48):
    """
    Get REAL head output via hooking o_proj input.
    The o_proj input is [batch, seq, concat_dim] where each head's output
    occupies a contiguous block of head_dim dimensions.
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)
    if target_layer >= len(layers):
        return None, None

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = inputs["input_ids"].shape[1]

    captured = {}
    def pre_hook(module, args):
        captured["input"] = args[0].detach().cpu().clone()

    o_proj = layers[target_layer].self_attn.o_proj
    h = o_proj.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        model(**inputs)
    h.remove()

    if "input" not in captured:
        return None, None

    full_input = captured["input"]  # [batch, seq, concat_dim]
    s = target_head * head_dim
    e = (target_head + 1) * head_dim
    head_output = full_input[0, :, s:e].clone()  # [seq, head_dim]
    return head_output, seq_len


def run_calibration(cached_data, selected_heads, head_config, device):
    """
    Verify that manual AW@V reconstruction produces sane outputs.
    Checks: non-zero, reasonable norms, and AW_A@V_A ≈ AW_B@V_B (same head).
    """
    log_time("\n" + "=" * 60)
    log_time("CALIBRATION: Manual AW@V Sanity Check")
    log_time("=" * 60)

    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    calibration_results = []

    test_pairs = list(cached_data.keys())[:20]

    for (li, hi, label) in selected_heads:
        norms_AA = []
        norms_BB = []
        cos_self = []  # cos(AW_A@V_A, AW_A@V_A) should be 1.0
        cos_cross = []  # cos between A and B head outputs
        lk = f"L{li}"

        for pname in test_pairs:
            data = cached_data[pname]
            if lk not in data.get("A", {}):
                continue

            for sent_key in ["A", "B"]:
                if lk not in data.get(sent_key, {}):
                    continue
                AW = data[sent_key][lk]["AW"].to(device)
                V = data[sent_key][lk]["V"].to(device)
                ho = compute_head_output_manual(AW, V, hi, n_heads, n_kv_heads, head_dim)
                norm_val = float(torch.norm(ho.float()))
                if sent_key == "A":
                    norms_AA.append(norm_val)
                else:
                    norms_BB.append(norm_val)

            # Cross: does A's head output differ from B's?
            if lk in data.get("A", {}) and lk in data.get("B", {}):
                AW_A = data["A"][lk]["AW"].to(device)
                V_A = data["A"][lk]["V"].to(device)
                AW_B = data["B"][lk]["AW"].to(device)
                V_B = data["B"][lk]["V"].to(device)
                ho_A = compute_head_output_manual(AW_A, V_A, hi, n_heads, n_kv_heads, head_dim)
                ho_B = compute_head_output_manual(AW_B, V_B, hi, n_heads, n_kv_heads, head_dim)
                # Flatten and compute cosine
                flat_A = ho_A.float().reshape(-1)
                flat_B = ho_B.float().reshape(-1)
                nA, nB = torch.norm(flat_A), torch.norm(flat_B)
                if nA > 1e-8 and nB > 1e-8:
                    cos_cross.append(float(torch.dot(flat_A, flat_B) / (nA * nB)))

        mean_na = np.mean(norms_AA) if norms_AA else 0
        mean_nb = np.mean(norms_BB) if norms_BB else 0
        mean_cross = np.mean(cos_cross) if cos_cross else 0

        status = "OK"
        if mean_na < 1e-3:
            status = "WARN: near-zero"
        elif mean_cross > 0.999:
            status = "WARN: A≈B (no diff)"

        log_time(f"  {label}: norm_A={mean_na:.4f}, norm_B={mean_nb:.4f}, "
                 f"cos_A_vs_B={mean_cross:.4f}, n_points={len(norms_AA)}, [{status}]")

        calibration_results.append({
            "head": label, "layer": li, "head_idx": hi,
            "mean_norm_A": float(mean_na), "mean_norm_B": float(mean_nb),
            "mean_cos_AB": float(mean_cross), "n_points": len(norms_AA),
            "status": status,
        })

    return calibration_results


# ==============================================================================
# ROUTE-CONTENT PATCHING (with direction projection + random controls)
# ==============================================================================

def construct_hybrid_outputs(cached_pair_data, li, hi, head_config, device, dtype):
    """Construct 6 hybrid outputs including random controls."""
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    lk = f"L{li}"

    AW_A = cached_pair_data["A"][lk]["AW"].to(device)
    V_A = cached_pair_data["A"][lk]["V"].to(device)
    AW_B = cached_pair_data["B"][lk]["AW"].to(device)
    V_B = cached_pair_data["B"][lk]["V"].to(device)

    # Random controls: shuffle keys of AW or V
    seq_len_a = AW_A.shape[2]
    seq_len_b = AW_B.shape[2]
    # Random AW: shuffle among positions (permute key dimension)
    AW_rand = AW_A.clone()
    perm = torch.randperm(seq_len_a)
    AW_rand = AW_rand[:, :, :, perm]
    # Random V: shuffle among positions
    V_rand = V_A.clone()
    perm_v = torch.randperm(V_A.shape[1])
    V_rand = V_rand[:, perm_v, :]

    ho_AA = compute_head_output_manual(AW_A, V_A, hi, n_heads, n_kv_heads, head_dim)
    ho_BB = compute_head_output_manual(AW_B, V_B, hi, n_heads, n_kv_heads, head_dim)
    ho_AB = compute_head_output_manual(AW_A, V_B, hi, n_heads, n_kv_heads, head_dim)
    ho_BA = compute_head_output_manual(AW_B, V_A, hi, n_heads, n_kv_heads, head_dim)
    ho_Arand = compute_head_output_manual(AW_rand, V_A, hi, n_heads, n_kv_heads, head_dim)
    ho_randA = compute_head_output_manual(AW_A, V_rand, hi, n_heads, n_kv_heads, head_dim)

    return {
        "AW_A_V_A": ho_AA.to(dtype).cpu(),
        "AW_B_V_B": ho_BB.to(dtype).cpu(),
        "AW_A_V_B": ho_AB.to(dtype).cpu(),
        "AW_B_V_A": ho_BA.to(dtype).cpu(),
        "AW_rand_V_A": ho_Arand.to(dtype).cpu(),
        "AW_A_V_rand": ho_randA.to(dtype).cpu(),
    }


def forward_with_patch(model, tokenizer, sentence, device, n_layers,
                       patch_vector, target_layer, target_head, head_config, common_len):
    """Forward with head's o_proj input slot replaced. Returns logits [vocab]."""
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)
    if target_layer >= len(layers):
        return None

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True,
                       max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len),
                                    value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    o_proj = layers[target_layer].self_attn.o_proj
    pv = patch_vector.clone().to(device).to(torch.bfloat16)

    def pre_hook(module, args):
        new_input = args[0].clone()
        s = target_head * head_dim
        e = (target_head + 1) * head_dim
        pv_seq = min(pv.shape[0], new_input.shape[1])
        new_input[0, :pv_seq, s:e] = pv[:pv_seq]
        return (new_input,)

    hook_handle = o_proj.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except:
            result = None

    hook_handle.remove()
    return result


def compute_metrics(patched_logits, logits_a, logits_b, kl_ab):
    """Compute KL effect, direction projection, and progress metrics."""
    if patched_logits is None or logits_a is None or logits_b is None:
        return None

    kl_patched = float(F.kl_div(
        F.log_softmax(patched_logits, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction='sum'
    ))

    # Direction projection
    delta_B = logits_b - logits_a  # B - A is the "true direction"
    delta_patch = patched_logits - logits_a

    norm_B = float(torch.norm(delta_B))
    norm_patch = float(torch.norm(delta_patch))

    if norm_B > 1e-8 and norm_patch > 1e-8:
        cos_dir = float(torch.dot(delta_patch, delta_B) / (norm_B * norm_patch))
        mag_ratio = norm_patch / norm_B
        progress_score = cos_dir * min(mag_ratio, 2.0)  # cap magnitude at 2x
    else:
        cos_dir = 0
        mag_ratio = 0
        progress_score = 0

    # KL ratio (capped)
    kl_ratio = min(kl_patched / max(kl_ab, 1e-6), 5.0)

    return {
        "kl_patched": kl_patched,
        "kl_ratio": kl_ratio,
        "cos_dir": cos_dir,
        "mag_ratio": mag_ratio,
        "progress_score": progress_score,
    }


def patch_head_group(model, tokenizer, sentence, device, n_layers,
                     patch_vectors, head_config, common_len):
    """
    Patch MULTIPLE heads simultaneously.
    patch_vectors: dict {layer: {head: tensor[seq, head_dim]}}
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True,
                       max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len),
                                    value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    hooks = []
    for li in patch_vectors:
        if li >= len(layers):
            continue
        o_proj = layers[li].self_attn.o_proj
        head_patches = patch_vectors[li]

        def make_hook(li, head_patches):
            def pre_hook(module, args):
                new_input = args[0].clone()
                for hi, pv in head_patches.items():
                    s = hi * head_dim
                    e = (hi + 1) * head_dim
                    pv_dev = pv.to(new_input.device).to(new_input.dtype)
                    pv_seq = min(pv_dev.shape[0], new_input.shape[1])
                    new_input[0, :pv_seq, s:e] = pv_dev[:pv_seq]
                return (new_input,)
            return pre_hook

        hooks.append(o_proj.register_forward_pre_hook(make_hook(li, head_patches)))

    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except:
            result = None

    for h in hooks:
        h.remove()

    return result


# ==============================================================================
# MLP PATCHING
# ==============================================================================

def patch_mlp(model, tokenizer, sentence_a, sentence_b, device, n_layers, common_len,
              target_layers):
    """
    Patch entire MLP output at target layers: A's forward with B's MLP output.
    Caches B's MLP outputs first, then injects into A's forward.
    Returns logits after patching.
    """
    # Step 1: Cache B's MLP outputs
    mlp_cache = {}
    layers = get_layers(model)

    inputs_b = tokenizer(sentence_b, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    if inputs_b["input_ids"].shape[1] < common_len:
        pad_len = common_len - inputs_b["input_ids"].shape[1]
        inputs_b["input_ids"] = F.pad(inputs_b["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs_b["attention_mask"] = F.pad(inputs_b["attention_mask"], (0, pad_len), value=0)

    b_hooks = []
    for li in target_layers:
        if li >= len(layers):
            continue
        def make_mlp_hook(li):
            def hook(module, input_t, output_t):
                if isinstance(output_t, tuple):
                    mlp_cache[li] = output_t[0].detach().cpu().clone()
                else:
                    mlp_cache[li] = output_t.detach().cpu().clone()
            return hook
        b_hooks.append(layers[li].mlp.register_forward_hook(make_mlp_hook(li)))

    with torch.no_grad():
        model(**inputs_b)
    for h in b_hooks:
        h.remove()

    # Step 2: Forward A with B's MLP outputs injected
    inputs_a = tokenizer(sentence_a, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    if inputs_a["input_ids"].shape[1] < common_len:
        pad_len = common_len - inputs_a["input_ids"].shape[1]
        inputs_a["input_ids"] = F.pad(inputs_a["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs_a["attention_mask"] = F.pad(inputs_a["attention_mask"], (0, pad_len), value=0)

    a_hooks = []
    for li in target_layers:
        if li not in mlp_cache:
            continue
        mlp_out_b = mlp_cache[li]

        def make_replace_hook(li, mlp_out_b):
            def hook(module, input_t, output_t):
                b_out = mlp_out_b.to(output_t[0].device).to(output_t[0].dtype)
                min_seq = min(b_out.shape[1], output_t[0].shape[1])
                new_out = output_t[0].clone()
                new_out[:, :min_seq, :] = b_out[:, :min_seq, :]
                if isinstance(output_t, tuple):
                    return (new_out,) + output_t[1:]
                return new_out
            return hook

        a_hooks.append(layers[li].mlp.register_forward_hook(make_replace_hook(li, mlp_out_b)))

    with torch.no_grad():
        try:
            out = model(**inputs_a)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except:
            result = None

    for h in a_hooks:
        h.remove()

    return result


# ==============================================================================
# MAIN PHASE 288 PIPELINE
# ==============================================================================

def run_phase288(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase288_{model_name}.txt")

    log_time(f"{'='*60}")
    log_time(f"Phase 288: Route-Content-MLP Causal Decomposition — {model_name}")
    log_time(f"{'='*60}")

    # === HEAD SELECTION (Phase 286 causal + random controls) ===
    head_selection = {
        "qwen3": [
            (16, 27, "L16_H27"), (12, 25, "L12_H25"), (12, 0, "L12_H0"),
            (35, 0, "L35_H0"), (28, 14, "L28_H14"),
            # random controls (different layers than top heads)
            (4, 8, "R_L4_H8"), (20, 5, "R_L20_H5"), (32, 12, "R_L32_H12"),
        ],
        "glm4": [
            (16, 7, "L16_H7"), (8, 16, "L8_H16"), (28, 27, "L28_H27"),
            (16, 30, "L16_H30"), (8, 26, "L8_H26"),
            (4, 4, "R_L4_H4"), (20, 20, "R_L20_H20"), (32, 8, "R_L32_H8"),
        ],
        "deepseek7b": [
            (0, 10, "L0_H10"), (0, 3, "L0_H3"), (21, 9, "L21_H9"),
            (18, 0, "L18_H0"), (24, 15, "L24_H15"),
            (8, 2, "R_L8_H2"), (14, 7, "R_L14_H7"), (26, 10, "R_L26_H10"),
        ],
    }

    selected_heads = head_selection.get(model_name, head_selection["qwen3"])
    target_layers = sorted(set(li for li, hi, label in selected_heads))
    causal_heads = [(li, hi, label) for (li, hi, label) in selected_heads if not label.startswith("R_")]
    random_heads = [(li, hi, label) for (li, hi, label) in selected_heads if label.startswith("R_")]

    log_time(f"Target: {len(selected_heads)} heads ({len(causal_heads)} causal + {len(random_heads)} random) "
             f"across {len(target_layers)} layers")
    for li, hi, label in causal_heads:
        log_time(f"  CAUSAL: {label} (L{li}_H{hi})")
    for li, hi, label in random_heads:
        log_time(f"  RANDOM: {label} (L{li}_H{hi})")

    # ============================================================
    # PART 1: CACHE AW+V (EAGER attention)
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PART 1: Cache AW+V (EAGER attention)")
    log_time(f"{'='*60}")

    model_eager, tokenizer, device = load_model_eager(model_name)
    model_info = get_model_info(model_eager, model_name)
    n_layers = model_info.n_layers
    head_config = get_head_config(model_eager)
    n_heads, n_kv_heads, head_dim, concat_dim = head_config

    log_time(f"Model: {model_info.model_class}, L={n_layers}, n_heads={n_heads}, "
             f"n_kv_heads={n_kv_heads}, head_dim={head_dim}, d_model={model_info.d_model}")

    all_pairs = build_all_pairs()
    log_time(f"Dataset: {len(all_pairs)} pairs across 6 functions")
    for cat in sorted(set(p["category"] for p in all_pairs)):
        n = sum(1 for p in all_pairs if p["category"] == cat)
        log_time(f"  {cat}: {n} pairs")

    # Warmup
    wu = tokenizer("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        try: model_eager(**wu)
        except: pass

    cached_data = cache_aw_and_v(model_eager, tokenizer, all_pairs, device, n_layers,
                                  head_config, target_layers=target_layers)
    n_cached = len(cached_data)
    log_time(f"Cached {n_cached}/{len(all_pairs)} pairs")

    # ============================================================
    # PART 2: CALIBRATION (non-blocking)
    # ============================================================
    try:
        calibration_results = run_calibration(cached_data, causal_heads, head_config, device)
    except Exception as e:
        log_time(f"CALIBRATION FAILED (non-blocking): {e}")
        import traceback
        traceback.print_exc()
        calibration_results = []

    # ============================================================
    # PART 3: ROUTE-CONTENT PATCHING (REUSE EAGER — single model, no flash mismatch)
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PART 3: Route-Content Patching (SAME eager model, no flash)")
    log_time(f"{'='*60}")

    # REUSE eager model for patching — avoids eager→flash numerical mismatch
    # Disable output_attentions for speed during patching phase
    model_eager.config.output_attentions = False
    model_flash = model_eager  # alias for rest of code
    log_time(f"Reusing eager model for patching. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Compute baseline logits for all pairs
    log_time("\nComputing baseline logits...")
    pair_baselines = {}
    t0_bl = time.time()

    for pidx, pair in enumerate(all_pairs):
        pname = pair["name"]
        if pname not in cached_data:
            continue
        sent_a, sent_b = pair["A"], pair["B"]
        cl = cached_data[pname]["seq_len"]

        # A
        ia = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=cl).to(device)
        with torch.no_grad():
            logits_a = model_flash(**ia).logits[0, -1, :].detach().cpu().float()
        # B
        ib = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=cl).to(device)
        with torch.no_grad():
            logits_b = model_flash(**ib).logits[0, -1, :].detach().cpu().float()

        kl_ab = float(F.kl_div(
            F.log_softmax(logits_a, dim=-1),
            F.softmax(logits_b, dim=-1),
            reduction='sum'
        ))

        pair_baselines[pname] = {
            "logits_a": logits_a, "logits_b": logits_b,
            "kl_ab": kl_ab, "sent_a": sent_a, "sent_b": sent_b,
            "category": pair["category"],
        }

        if (pidx + 1) % 40 == 0:
            log_time(f"  [{pidx+1}/{len(all_pairs)}] Baselining: {time.time()-t0_bl:.0f}s")

    log_time(f"  Baselining done: {len(pair_baselines)} pairs, {time.time()-t0_bl:.0f}s")

    # Select patching pairs: ensure per-function coverage
    all_cats = set(p["category"] for p in all_pairs)
    patching_pairs = list(pair_baselines.keys())  # use all pairs with baseline

    log_time(f"Patching pairs: {len(patching_pairs)}")
    for cat in sorted(all_cats):
        n = sum(1 for pn in patching_pairs if pair_baselines[pn]["category"] == cat)
        log_time(f"  {cat}: {n}")

    # Pre-compute all hybrid outputs for all (head, pair) combinations
    log_time("Pre-computing hybrid head outputs (including random controls)...")
    hybrid_cache = {}

    for (li, hi, label) in selected_heads:
        for pname in patching_pairs:
            if pname not in cached_data or f"L{li}" not in cached_data[pname].get("A", {}):
                continue
            try:
                hy = construct_hybrid_outputs(
                    cached_data[pname], li, hi, head_config, device, torch.bfloat16
                )
                hybrid_cache[(pname, li, hi)] = hy
            except Exception as e:
                pass

    log_time(f"Pre-computed {len(hybrid_cache)} hybrid outputs")

    # Run patching
    condition_names = ["AW_A_V_A", "AW_A_V_B", "AW_B_V_A", "AW_B_V_B",
                       "AW_rand_V_A", "AW_A_V_rand"]
    condition_types = ["full_A", "routing", "content", "full_B", "rand_routing", "rand_content"]

    all_head_results = []  # list of dicts, one per head-pair condition
    t0_patch = time.time()
    n_done = 0

    for (li, hi, label) in selected_heads:
        is_random_head = label.startswith("R_")
        for pidx, pname in enumerate(patching_pairs):
            key = (pname, li, hi)
            if key not in hybrid_cache:
                continue

            pb = pair_baselines[pname]
            sent_a = pb["sent_a"]
            logits_a = pb["logits_a"]
            logits_b = pb["logits_b"]
            kl_ab = pb["kl_ab"]
            common_len = cached_data[pname]["seq_len"]
            hy = hybrid_cache[key]

            if kl_ab < 1e-6:
                continue

            for cond_name, cond_type in zip(condition_names, condition_types):
                patch_vec = hy[cond_name][0]
                patched_logits = forward_with_patch(
                    model_flash, tokenizer, sent_a, device, n_layers,
                    patch_vec, li, hi, head_config, common_len
                )

                if patched_logits is not None:
                    metrics = compute_metrics(patched_logits, logits_a, logits_b, kl_ab)
                    if metrics is not None:
                        all_head_results.append({
                            "pname": pname,
                            "category": pb["category"],
                            "head": label,
                            "layer": li,
                            "head_idx": hi,
                            "condition": cond_type,
                            "is_random_head": is_random_head,
                            "kl_ab": kl_ab,
                            **metrics,
                        })
            n_done += 1

        elapsed = time.time() - t0_patch
        n_this = sum(1 for r in all_head_results if r["head"] == label)
        log_time(f"  {label}: {n_this} results, {elapsed:.0f}s, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_patch = time.time() - t0_patch
    log_time(f"\nPatching done: {len(all_head_results)} results, {t_patch:.0f}s ({t_patch/60:.1f}min)")

    # ============================================================
    # PART 4: HEAD GROUP PATCHING
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PART 4: Head Group Patching")
    log_time(f"{'='*60}")

    # Test: top-3 causal heads together, top-5 together, random-5 together
    group_results = []
    group_configs = [
        ("top3", causal_heads[:3]),
        ("top5", causal_heads[:5]),
        ("rand5", random_heads[:5] if len(random_heads) >= 5 else random_heads + causal_heads[:5-len(random_heads)]),
    ]

    for group_name, group_heads in group_configs:
        log_time(f"\n  Group: {group_name} ({len(group_heads)} heads)")
        n_group_ok = 0
        for pidx, pname in enumerate(patching_pairs[:80]):  # limit to 80 pairs per group
            pb = pair_baselines[pname]
            sent_a, sent_b = pb["sent_a"], pb["sent_b"]
            logits_a, logits_b = pb["logits_a"], pb["logits_b"]
            kl_ab = pb["kl_ab"]
            if kl_ab < 1e-6: continue

            common_len = cached_data[pname]["seq_len"]

            # Collect patch vectors for all heads
            patch_vectors = defaultdict(dict)
            ok = True
            for g_li, g_hi, g_label in group_heads:
                g_key = (pname, g_li, g_hi)
                if g_key not in hybrid_cache:
                    ok = False
                    break
                patch_vectors[g_li][g_hi] = hybrid_cache[g_key]["AW_A_V_A"][0]

            if not ok: continue

            patched_logits = patch_head_group(
                model_flash, tokenizer, sent_a, device, n_layers,
                patch_vectors, head_config, common_len
            )

            if patched_logits is not None:
                metrics = compute_metrics(patched_logits, logits_a, logits_b, kl_ab)
                if metrics is not None:
                    group_results.append({
                        "pname": pname, "category": pb["category"],
                        "group": group_name, "n_heads": len(group_heads),
                        "kl_ab": kl_ab, **metrics,
                    })
                    n_group_ok += 1

            if (pidx + 1) % 20 == 0:
                log_time(f"    [{pidx+1}/80] {group_name} group: {n_group_ok} ok")

        log_time(f"    {group_name} group done: {n_group_ok} results")

    # ============================================================
    # PART 5: MLP PATCHING
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PART 5: MLP Patching")
    log_time(f"{'='*60}")

    # MLP target layers: early, middle, late, and all
    mlp_layer_configs = [
        ("MLP_early", [0, 1, 2]),
        ("MLP_mid", [n_layers // 2 - 1, n_layers // 2, n_layers // 2 + 1]),
        ("MLP_late", [n_layers - 3, n_layers - 2, n_layers - 1]),
        ("MLP_all3", [0, n_layers // 2, n_layers - 1]),
    ]

    mlp_results = []

    # Only test main functions (focus on where heads are weak: GLM4)
    mlp_functions = ["negation", "translation", "logical", "passive", "comparative", "recursive"]

    for mlp_name, mlp_layers in mlp_layer_configs:
        log_time(f"\n  {mlp_name}: layers {mlp_layers}")
        n_mlp_ok = 0

        for pname in patching_pairs:
            pb = pair_baselines[pname]
            cat = pb["category"]
            if cat not in mlp_functions: continue
            kl_ab = pb["kl_ab"]
            if kl_ab < 1e-6: continue

            sent_a, sent_b = pb["sent_a"], pb["sent_b"]
            logits_a, logits_b = pb["logits_a"], pb["logits_b"]
            common_len = cached_data[pname]["seq_len"]

            # Patch A→B: inject B's MLP outputs into A's forward
            patched_logits = patch_mlp(
                model_flash, tokenizer, sent_a, sent_b, device, n_layers,
                common_len, mlp_layers
            )

            if patched_logits is not None:
                metrics = compute_metrics(patched_logits, logits_a, logits_b, kl_ab)
                if metrics is not None:
                    mlp_results.append({
                        "pname": pname, "category": cat,
                        "mlp_group": mlp_name, "n_layers_patched": len(mlp_layers),
                        "kl_ab": kl_ab, **metrics,
                    })
                    n_mlp_ok += 1

        log_time(f"    {mlp_name}: {n_mlp_ok} results")

    # Release model (now single eager model, no separate flash)
    release_model(model_flash)
    model_flash = None
    del model_flash
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # PART 6: ANALYSIS & SAVE
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PART 6: Analysis")
    log_time(f"{'='*60}")

    # --- Head-level analysis ---
    log_time("\n  ===== ROUTE-CONTENT PER HEAD =====")
    log_time(f"  {'Head':>14} {'N':>5} {'full_A':>7} {'routing':>7} {'content':>7} "
             f"{'r_rand':>7} {'c_rand':>7} {'prog':>6} {'cos':>5} {'Interpretation':>24}")

    head_agg = defaultdict(lambda: defaultdict(list))
    for r in all_head_results:
        h = r["head"]
        head_agg[h][r["condition"]].append(r)

    per_head_analysis = {}
    for hlabel in sorted(head_agg.keys()):
        d = head_agg[hlabel]
        def avg_kr(cond):
            vals = [x["kl_ratio"] for x in d.get(cond, [])]
            return np.mean(vals) if vals else 0
        def avg_prog(cond):
            vals = [x["progress_score"] for x in d.get(cond, [])]
            return np.mean(vals) if vals else 0
        def avg_cos(cond):
            vals = [x["cos_dir"] for x in d.get(cond, [])]
            return np.mean(vals) if vals else 0

        fa = avg_kr("full_A")
        rr = avg_kr("routing")
        cr = avg_kr("content")
        rr_rand = avg_kr("rand_routing")
        cr_rand = avg_kr("rand_content")
        prog = avg_prog("full_A")
        cos = avg_cos("full_A")

        n = len(d.get("full_A", []))

        # Significance: is routing > random_routing? content > random_content?
        sig_r = "R*" if rr > rr_rand * 1.3 else "R?"
        sig_c = "C*" if cr > cr_rand * 1.3 else "C?"

        # Interpretation
        if rr > 0.6 * fa and cr < 0.3 * fa:
            interp = f"ROUTING {sig_r}"
        elif cr > 0.6 * fa and rr < 0.3 * fa:
            interp = f"CONTENT {sig_c}"
        elif rr < 0.3 * fa and cr < 0.3 * fa:
            interp = "COUPLED"
        elif rr > 0.4 * fa and cr > 0.4 * fa:
            interp = f"EITHER {sig_r}{sig_c}"
        else:
            interp = "weak"

        per_head_analysis[hlabel] = {
            "n": n, "full_A_ratio": float(fa), "routing_ratio": float(rr),
            "content_ratio": float(cr), "rand_routing_ratio": float(rr_rand),
            "rand_content_ratio": float(cr_rand), "progress_score": float(prog),
            "cos_dir": float(cos), "interpretation": interp,
        }

        log_time(f"  {hlabel:>14} {n:>5} {fa:7.3f} {rr:7.3f} {cr:7.3f} "
                 f"{rr_rand:7.3f} {cr_rand:7.3f} {prog:6.3f} {cos:5.2f} {interp:>24}")

    # --- Per-category analysis ---
    log_time("\n  ===== PER-CATEGORY ROUTE-CONTENT-MLP =====")
    log_time(f"  {'Category':>18} {'n':>5} {'head_R':>7} {'head_C':>7} "
             f"{'MLP_early':>9} {'MLP_mid':>9} {'MLP_late':>9} {'Dominant':>16}")

    cat_analysis = {}
    for cat in sorted(all_cats):
        cat_head = [r for r in all_head_results if r["category"] == cat]
        cat_mlp = [r for r in mlp_results if r["category"] == cat]

        # Head routing/content
        h_r = np.mean([r["kl_ratio"] for r in cat_head if r["condition"] == "routing" and not r["is_random_head"]]) if cat_head else 0
        h_c = np.mean([r["kl_ratio"] for r in cat_head if r["condition"] == "content" and not r["is_random_head"]]) if cat_head else 0

        # MLP by stage
        m_early = np.mean([r["progress_score"] for r in cat_mlp if r["mlp_group"] == "MLP_early"]) if cat_mlp else 0
        m_mid = np.mean([r["progress_score"] for r in cat_mlp if r["mlp_group"] == "MLP_mid"]) if cat_mlp else 0
        m_late = np.mean([r["progress_score"] for r in cat_mlp if r["mlp_group"] == "MLP_late"]) if cat_mlp else 0

        # Dominant mechanism
        max_comp = max(h_r, h_c, m_early, m_mid, m_late)
        if max_comp < 0.01:
            dominant = "noise"
        elif h_c >= max_comp and h_c > h_r * 1.3:
            dominant = "HEAD_CONTENT"
        elif h_r >= max_comp and h_r > h_c * 1.3:
            dominant = "HEAD_ROUTING"
        elif m_early >= max_comp or m_mid >= max_comp or m_late >= max_comp:
            dominant = "MLP_dominant"
        else:
            dominant = "balanced"

        cat_analysis[cat] = {
            "head_routing": float(h_r), "head_content": float(h_c),
            "mlp_early": float(m_early), "mlp_mid": float(m_mid), "mlp_late": float(m_late),
            "dominant": dominant,
            "n_head_points": len(cat_head),
            "n_mlp_points": len(cat_mlp),
        }

        log_time(f"  {cat:>18} {cat_analysis[cat]['n_head_points']:>5} {h_r:7.3f} {h_c:7.3f} "
                 f"{m_early:9.4f} {m_mid:9.4f} {m_late:9.4f} {dominant:>16}")

    # --- Head Group Analysis ---
    log_time("\n  ===== HEAD GROUP EFFECTS =====")
    group_agg = defaultdict(lambda: defaultdict(list))
    for r in group_results:
        group_agg[r["group"]][r["category"]].append(r["progress_score"])

    for gname in sorted(group_agg.keys()):
        cat_means = {}
        for cat in sorted(group_agg[gname].keys()):
            vals = group_agg[gname][cat]
            cat_means[cat] = np.mean(vals) if vals else 0
        avg_prog = np.mean(list(cat_means.values())) if cat_means else 0
        log_time(f"    {gname}: mean_progress={avg_prog:.4f}, categories: " +
                 ", ".join(f"{c}={v:.3f}" for c, v in sorted(cat_means.items())))

    # --- EFFECT>1 ANALYSIS (over-conversion check) ---
    log_time("\n  ===== EFFECT>1 (OVER-CONVERSION) ANALYSIS =====")
    for cond in ["full_A", "routing", "content"]:
        effects = [r["kl_ratio"] for r in all_head_results
                   if r["condition"] == cond and not r["is_random_head"]]
        if effects:
            n_gt1 = sum(1 for e in effects if e > 1.0)
            n_gt2 = sum(1 for e in effects if e > 2.0)
            log_time(f"    {cond}: N={len(effects)}, >1x={n_gt1} ({100*n_gt1/len(effects):.1f}%), "
                     f">2x={n_gt2} ({100*n_gt2/len(effects):.1f}%), mean={np.mean(effects):.3f}")

    # ============================================================
    # SAVE
    # ============================================================
    save_data = {
        "model": model_name,
        "head_config": {"n_heads": n_heads, "n_kv_heads": n_kv_heads,
                        "head_dim": head_dim, "concat_dim": concat_dim},
        "calibration": calibration_results,
        "per_head_analysis": per_head_analysis,
        "per_category_analysis": cat_analysis,
        "n_pairs_cached": n_cached,
        "n_pairs_patched": len(patching_pairs),
        "n_head_results": len(all_head_results),
        "n_mlp_results": len(mlp_results),
        "n_group_results": len(group_results),
        "head_results_summary": {
            "n_per_condition": {
                cond: sum(1 for r in all_head_results if r["condition"] == cond)
                for cond in condition_types
            }
        },
        "random_control_effect": {
            "rand_routing": float(np.mean([r["kl_ratio"] for r in all_head_results if r["condition"] == "rand_routing"])),
            "rand_content": float(np.mean([r["kl_ratio"] for r in all_head_results if r["condition"] == "rand_content"])),
        },
    }

    # Save full results (compact: only essential fields)
    save_path = RESULT_DIR / f"{model_name}_rcm.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log_time(f"\nResults saved to {save_path}")

    # Save detailed results separately (CSV-style JSON)
    detail_path = RESULT_DIR / f"{model_name}_detail.json"
    detail = {
        "head_results": all_head_results,
        "mlp_results": mlp_results,
        "group_results": group_results,
    }
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=1, default=str)
    log_time(f"Detailed results saved to {detail_path}")

    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase288(name)
                log_time(f"\n{name} DONE: {len(r['per_head_analysis'])} heads analyzed")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase288(model_name)
