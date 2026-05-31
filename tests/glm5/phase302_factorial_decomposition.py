"""
Phase 302: Full Factorial I/R/F Decomposition + Causal Test
============================================================
Goal: Fix the biggest flaw in Phase 301 — F was defined as residual (F=Δh-R),
not independently extracted. This makes cos(R,F) estimates unreliable and
orthogonalization results ambiguous.

Key improvement: Use balanced factorial design to independently estimate I, R, F.
  I(token) = grand_mean_subtracted token average across all (role, frame)
  R(role)  = grand_mean_subtracted role average across all (token, frame)
  F(frame) = grand_mean_subtracted frame average across all (token, role)
  RF       = residual after subtracting I, R, F from cell means

This gives us clean, independent estimates of each component.

Expanded stimulus set: 20+ dual-role tokens with 5-8 frame pairs each.

Causal conditions:
  1. R_only          — pure role direction (independently estimated)
  2. F_only          — pure frame direction (independently estimated)
  3. R+F             — additive combination (no binding term)
  4. R+F+RF          — full combination including binding term
  5. full_delta      — actual observed difference (ground truth)
  6. RF_only         — binding/interaction term only
  7. R_loo           — leave-one-out role direction
  8. R_loo+F         — LOO additive
  9. random control  — random direction with same norm

Key predictions:
  - If R+F+RF >> R+F: binding term RF is a real computational unit
  - If R+F ≈ R+F+RF: additive model suffices (no binding needed)
  - Compare F(factorial) vs F(residual) from Phase 301

Layer coverage: 8 layers [nl//8, nl//4, 3nl//8, nl//2, 5nl//8, 3nl//4, 7nl//8, nl-2]

Usage:
  python tests/glm5/phase302_factorial_decomposition.py qwen3
  python tests/glm5/phase302_factorial_decomposition.py glm4
  python tests/glm5/phase302_factorial_decomposition.py deepseek7b
"""
import sys, os, gc, time, json, math, itertools
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase302_factorial_decomposition")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass

# =====================================================================
# EXPANDED STIMULUS SET — 20+ dual-role tokens, 5-8 frame pairs each
# =====================================================================
def build_stimuli():
    """
    Build balanced factorial stimulus set.
    
    Structure: each token appears in both roles, in multiple frames.
    Frames are designed to be diverse (copula, transitive, intransitive, modal, etc.)
    
    For each (token, role, frame), we have 2 sentences (different objects/subjects).
    """
    stimuli = []
    
    # ===== adj_verb tokens (10 tokens) =====
    adj_verb = {
        "open": {
            "adj": {
                "F1_copula": ["the door is open", "the gate is open"],
                "F2_remain": ["the door remains open", "the gate remains open"],
                "F3_attrib": ["the open door", "the open gate"],
                "F4_seem": ["the shop seemed open", "the road seemed open"],
                "F5_become": ["the door became open", "the gate became open"],
            },
            "verb": {
                "F1_transitive": ["they open the door", "they open the gate"],
                "F2_intransitive": ["the door will open", "the gate will open"],
                "F3_begin": ["they began to open the shop", "they began to open the road"],
                "F4_modal": ["they can open the door", "they can open the gate"],
                "F5_causative": ["they made the door open", "they made the gate open"],
            },
        },
        "clear": {
            "adj": {
                "F1_copula": ["the path is clear", "the road is clear"],
                "F2_remain": ["the path remains clear", "the road remains clear"],
                "F3_attrib": ["the clear path", "the clear road"],
                "F4_seem": ["the desk seemed clear", "the table seemed clear"],
                "F5_become": ["the sky became clear", "the water became clear"],
            },
            "verb": {
                "F1_transitive": ["they clear the path", "they clear the road"],
                "F2_intransitive": ["the path will clear", "the road will clear"],
                "F3_begin": ["they began to clear the desk", "they began to clear the table"],
                "F4_modal": ["they can clear the path", "they can clear the road"],
                "F5_causative": ["they made the path clear", "they made the road clear"],
            },
        },
        "warm": {
            "adj": {
                "F1_copula": ["the room is warm", "the house is warm"],
                "F2_remain": ["the room remains warm", "the house remains warm"],
                "F3_attrib": ["the warm room", "the warm house"],
                "F4_seem": ["the water seemed warm", "the food seemed warm"],
                "F5_become": ["the room became warm", "the house became warm"],
            },
            "verb": {
                "F1_transitive": ["they warm the room", "they warm the house"],
                "F2_intransitive": ["the room will warm", "the house will warm"],
                "F3_begin": ["they began to warm the water", "they began to warm the food"],
                "F4_modal": ["they can warm the room", "they can warm the house"],
                "F5_causative": ["they made the room warm", "they made the house warm"],
            },
        },
        "clean": {
            "adj": {
                "F1_copula": ["the floor is clean", "the table is clean"],
                "F2_remain": ["the floor remains clean", "the table remains clean"],
                "F3_attrib": ["the clean floor", "the clean table"],
                "F4_seem": ["the room seemed clean", "the house seemed clean"],
                "F5_become": ["the floor became clean", "the table became clean"],
            },
            "verb": {
                "F1_transitive": ["they clean the floor", "they clean the table"],
                "F2_intransitive": ["the floor will clean", "the table will clean"],
                "F3_begin": ["they began to clean the room", "they began to clean the house"],
                "F4_modal": ["they can clean the floor", "they can clean the table"],
                "F5_causative": ["they made the floor clean", "they made the table clean"],
            },
        },
        "dry": {
            "adj": {
                "F1_copula": ["the ground is dry", "the cloth is dry"],
                "F2_remain": ["the ground remains dry", "the cloth remains dry"],
                "F3_attrib": ["the dry ground", "the dry cloth"],
                "F4_seem": ["the air seemed dry", "the wind seemed dry"],
                "F5_become": ["the ground became dry", "the cloth became dry"],
            },
            "verb": {
                "F1_transitive": ["they dry the cloth", "they dry the clothes"],
                "F2_intransitive": ["the ground will dry", "the cloth will dry"],
                "F3_begin": ["they began to dry the clothes", "they began to dry the dishes"],
                "F4_modal": ["they can dry the cloth", "they can dry the clothes"],
                "F5_causative": ["they made the cloth dry", "they made the clothes dry"],
            },
        },
        "close": {
            "adj": {
                "F1_copula": ["the store is close", "the school is close"],
                "F2_remain": ["the store remains close", "the school remains close"],
                "F3_attrib": ["the close store", "the close school"],
                "F4_seem": ["the park seemed close", "the beach seemed close"],
                "F5_become": ["the store became close", "the school became close"],
            },
            "verb": {
                "F1_transitive": ["they close the store", "they close the school"],
                "F2_intransitive": ["the store will close", "the school will close"],
                "F3_begin": ["they began to close the door", "they began to close the gate"],
                "F4_modal": ["they can close the store", "they can close the school"],
                "F5_causative": ["they made the door close", "they made the gate close"],
            },
        },
        "free": {
            "adj": {
                "F1_copula": ["the bird is free", "the person is free"],
                "F2_remain": ["the bird remains free", "the person remains free"],
                "F3_attrib": ["the free bird", "the free person"],
                "F4_seem": ["the animal seemed free", "the child seemed free"],
                "F5_become": ["the bird became free", "the person became free"],
            },
            "verb": {
                "F1_transitive": ["they free the bird", "they free the person"],
                "F2_intransitive": ["the bird will free", "the person will free"],
                "F3_begin": ["they began to free the animals", "they began to free the people"],
                "F4_modal": ["they can free the bird", "they can free the person"],
                "F5_causative": ["they made the bird free", "they made the person free"],
            },
        },
        "quiet": {
            "adj": {
                "F1_copula": ["the room is quiet", "the house is quiet"],
                "F2_remain": ["the room remains quiet", "the house remains quiet"],
                "F3_attrib": ["the quiet room", "the quiet house"],
                "F4_seem": ["the street seemed quiet", "the town seemed quiet"],
                "F5_become": ["the room became quiet", "the house became quiet"],
            },
            "verb": {
                "F1_transitive": ["they quiet the room", "they quiet the crowd"],
                "F2_intransitive": ["the room will quiet", "the crowd will quiet"],
                "F3_begin": ["they began to quiet the children", "they began to quiet the audience"],
                "F4_modal": ["they can quiet the room", "they can quiet the crowd"],
                "F5_causative": ["they made the room quiet", "they made the crowd quiet"],
            },
        },
        "cool": {
            "adj": {
                "F1_copula": ["the water is cool", "the air is cool"],
                "F2_remain": ["the water remains cool", "the air remains cool"],
                "F3_attrib": ["the cool water", "the cool air"],
                "F4_seem": ["the room seemed cool", "the house seemed cool"],
                "F5_become": ["the water became cool", "the air became cool"],
            },
            "verb": {
                "F1_transitive": ["they cool the water", "they cool the room"],
                "F2_intransitive": ["the water will cool", "the room will cool"],
                "F3_begin": ["they began to cool the food", "they began to cool the drink"],
                "F4_modal": ["they can cool the water", "they can cool the room"],
                "F5_causative": ["they made the water cool", "they made the room cool"],
            },
        },
        "smooth": {
            "adj": {
                "F1_copula": ["the surface is smooth", "the road is smooth"],
                "F2_remain": ["the surface remains smooth", "the road remains smooth"],
                "F3_attrib": ["the smooth surface", "the smooth road"],
                "F4_seem": ["the skin seemed smooth", "the fabric seemed smooth"],
                "F5_become": ["the surface became smooth", "the road became smooth"],
            },
            "verb": {
                "F1_transitive": ["they smooth the surface", "they smooth the fabric"],
                "F2_intransitive": ["the surface will smooth", "the fabric will smooth"],
                "F3_begin": ["they began to smooth the wood", "they began to smooth the metal"],
                "F4_modal": ["they can smooth the surface", "they can smooth the fabric"],
                "F5_causative": ["they made the surface smooth", "they made the fabric smooth"],
            },
        },
    }
    
    # ===== adj_noun tokens (6 tokens) =====
    adj_noun = {
        "light": {
            "adj": {
                "F1_copula": ["the bag is light", "the box is light"],
                "F2_remain": ["the bag remains light", "the box remains light"],
                "F3_attrib": ["the light bag", "the light box"],
                "F4_seem": ["the load seemed light", "the dress seemed light"],
                "F5_become": ["the bag became light", "the box became light"],
            },
            "noun": {
                "F1_copula": ["the light is bright", "the light is warm"],
                "F2_exist": ["that light is bright", "that light is warm"],
                "F3_locative": ["near the light", "by the light"],
                "F4_action": ["they saw the light", "they found the light"],
                "F5_possessive": ["her light is bright", "his light is warm"],
            },
        },
        "cold": {
            "adj": {
                "F1_copula": ["the water is cold", "the wind is cold"],
                "F2_remain": ["the water remains cold", "the wind remains cold"],
                "F3_attrib": ["the cold water", "the cold wind"],
                "F4_seem": ["the room seemed cold", "the air seemed cold"],
                "F5_become": ["the water became cold", "the wind became cold"],
            },
            "noun": {
                "F1_copula": ["the cold is severe", "the cold is bitter"],
                "F2_exist": ["that cold is severe", "that cold is bitter"],
                "F3_locative": ["in the cold", "despite the cold"],
                "F4_action": ["they felt the cold", "they noticed the cold"],
                "F5_possessive": ["her cold is severe", "his cold is bitter"],
            },
        },
        "right": {
            "adj": {
                "F1_copula": ["the answer is right", "the choice is right"],
                "F2_remain": ["the answer remains right", "the choice remains right"],
                "F3_attrib": ["the right answer", "the right choice"],
                "F4_seem": ["the decision seemed right", "the path seemed right"],
                "F5_become": ["the answer became right", "the choice became right"],
            },
            "noun": {
                "F1_copula": ["the right is clear", "the right is important"],
                "F2_exist": ["that right is clear", "that right is important"],
                "F3_locative": ["on the right", "to the right"],
                "F4_action": ["they claimed the right", "they defended the right"],
                "F5_possessive": ["her right is clear", "his right is important"],
            },
        },
        "fair": {
            "adj": {
                "F1_copula": ["the price is fair", "the game is fair"],
                "F2_remain": ["the price remains fair", "the game remains fair"],
                "F3_attrib": ["the fair price", "the fair game"],
                "F4_seem": ["the deal seemed fair", "the trial seemed fair"],
                "F5_become": ["the price became fair", "the game became fair"],
            },
            "noun": {
                "F1_copula": ["the fair is large", "the fair is popular"],
                "F2_exist": ["that fair is large", "that fair is popular"],
                "F3_locative": ["at the fair", "near the fair"],
                "F4_action": ["they visited the fair", "they enjoyed the fair"],
                "F5_possessive": ["her fair is large", "his fair is popular"],
            },
        },
        "round": {
            "adj": {
                "F1_copula": ["the table is round", "the ball is round"],
                "F2_remain": ["the table remains round", "the ball remains round"],
                "F3_attrib": ["the round table", "the round ball"],
                "F4_seem": ["the shape seemed round", "the face seemed round"],
                "F5_become": ["the shape became round", "the face became round"],
            },
            "noun": {
                "F1_copula": ["the round is over", "the round is final"],
                "F2_exist": ["that round is over", "that round is final"],
                "F3_locative": ["in the round", "during the round"],
                "F4_action": ["they won the round", "they finished the round"],
                "F5_possessive": ["her round is over", "his round is final"],
            },
        },
        "solid": {
            "adj": {
                "F1_copula": ["the ground is solid", "the wall is solid"],
                "F2_remain": ["the ground remains solid", "the wall remains solid"],
                "F3_attrib": ["the solid ground", "the solid wall"],
                "F4_seem": ["the base seemed solid", "the structure seemed solid"],
                "F5_become": ["the ground became solid", "the wall became solid"],
            },
            "noun": {
                "F1_copula": ["the solid is hard", "the solid is dense"],
                "F2_exist": ["that solid is hard", "that solid is dense"],
                "F3_locative": ["in the solid", "through the solid"],
                "F4_action": ["they examined the solid", "they measured the solid"],
                "F5_possessive": ["her solid is hard", "his solid is dense"],
            },
        },
    }
    
    # ===== noun_verb tokens (6 tokens) =====
    noun_verb = {
        "fire": {
            "noun": {
                "F1_copula": ["the fire is hot", "the fire is big"],
                "F2_exist": ["that fire is hot", "that fire is big"],
                "F3_locative": ["near the fire", "by the fire"],
                "F4_action": ["they saw the fire", "they started the fire"],
                "F5_possessive": ["her fire is hot", "his fire is big"],
            },
            "verb": {
                "F1_transitive": ["they fire the gun", "they fire the worker"],
                "F2_intransitive": ["the gun will fire", "the engine will fire"],
                "F3_begin": ["they began to fire the gun", "they began to fire the worker"],
                "F4_modal": ["they can fire the gun", "they can fire the worker"],
                "F5_causative": ["they made the gun fire", "they made the engine fire"],
            },
        },
        "record": {
            "noun": {
                "F1_copula": ["the record is old", "the record is broken"],
                "F2_exist": ["that record is old", "that record is broken"],
                "F3_locative": ["on the record", "for the record"],
                "F4_action": ["they broke the record", "they set the record"],
                "F5_possessive": ["her record is old", "his record is broken"],
            },
            "verb": {
                "F1_transitive": ["they record music", "they record data"],
                "F2_intransitive": ["the device will record", "the system will record"],
                "F3_begin": ["they began to record music", "they began to record data"],
                "F4_modal": ["they can record music", "they can record data"],
                "F5_causative": ["they made the device record", "they made the system record"],
            },
        },
        "run": {
            "noun": {
                "F1_copula": ["the run is long", "the run is hard"],
                "F2_exist": ["that run is long", "that run is hard"],
                "F3_locative": ["during the run", "after the run"],
                "F4_action": ["they enjoyed the run", "they finished the run"],
                "F5_possessive": ["her run is long", "his run is hard"],
            },
            "verb": {
                "F1_transitive": ["they run the program", "they run the company"],
                "F2_intransitive": ["they will run", "they can run"],
                "F3_begin": ["they began to run", "they started to run"],
                "F4_modal": ["they can run fast", "they can run far"],
                "F5_causative": ["they made the horse run", "they made the engine run"],
            },
        },
        "play": {
            "noun": {
                "F1_copula": ["the play is good", "the play is long"],
                "F2_exist": ["that play is good", "that play is long"],
                "F3_locative": ["at the play", "during the play"],
                "F4_action": ["they watched the play", "they enjoyed the play"],
                "F5_possessive": ["her play is good", "his play is long"],
            },
            "verb": {
                "F1_transitive": ["they play music", "they play tennis"],
                "F2_intransitive": ["they will play", "they can play"],
                "F3_begin": ["they began to play", "they started to play"],
                "F4_modal": ["they can play well", "they can play together"],
                "F5_causative": ["they made the children play", "they made the team play"],
            },
        },
        "sign": {
            "noun": {
                "F1_copula": ["the sign is clear", "the sign is large"],
                "F2_exist": ["that sign is clear", "that sign is large"],
                "F3_locative": ["near the sign", "by the sign"],
                "F4_action": ["they saw the sign", "they followed the sign"],
                "F5_possessive": ["her sign is clear", "his sign is large"],
            },
            "verb": {
                "F1_transitive": ["they sign the paper", "they sign the contract"],
                "F2_intransitive": ["they will sign", "they can sign"],
                "F3_begin": ["they began to sign", "they started to sign"],
                "F4_modal": ["they can sign today", "they can sign later"],
                "F5_causative": ["they made them sign", "they made the player sign"],
            },
        },
        "state": {
            "noun": {
                "F1_copula": ["the state is large", "the state is rich"],
                "F2_exist": ["that state is large", "that state is rich"],
                "F3_locative": ["in the state", "across the state"],
                "F4_action": ["they visited the state", "they left the state"],
                "F5_possessive": ["her state is large", "his state is rich"],
            },
            "verb": {
                "F1_transitive": ["they state the facts", "they state the rules"],
                "F2_intransitive": ["they will state", "they can state"],
                "F3_begin": ["they began to state", "they started to state"],
                "F4_modal": ["they can state clearly", "they can state again"],
                "F5_causative": ["they made them state", "they made the witness state"],
            },
        },
    }
    
    all_tokens = {}
    all_tokens.update(adj_verb)
    all_tokens.update(adj_noun)
    all_tokens.update(noun_verb)
    
    for token, roles in all_tokens.items():
        rp = "adj_verb" if token in adj_verb else ("adj_noun" if token in adj_noun else "noun_verb")
        for role, frames in roles.items():
            for frame_label, sentences in frames.items():
                for sent in sentences:
                    stimuli.append({
                        "sentence": sent,
                        "target_word": token,
                        "token_label": token,
                        "role_label": role,
                        "frame_label": frame_label,
                        "role_pair": rp,
                    })
    
    return stimuli


def build_causal_stimuli():
    """
    Causal test pairs: for each dual-role token, create matched adj/noun/verb pairs.
    Uses sentence pairs that differ ONLY in the frame (syntactic context) of the target word.
    """
    test_pairs = [
        # adj_verb
        ("the door is open", "open", "adj", "adj_verb"),
        ("they open the door", "open", "verb", "adj_verb"),
        ("the field is clear", "clear", "adj", "adj_verb"),
        ("they clear the field", "clear", "verb", "adj_verb"),
        ("the meal is warm", "warm", "adj", "adj_verb"),
        ("they warm the meal", "warm", "verb", "adj_verb"),
        ("the shirt is clean", "clean", "adj", "adj_verb"),
        ("they clean the shirt", "clean", "verb", "adj_verb"),
        ("the cloth is dry", "dry", "adj", "adj_verb"),
        ("they dry the cloth", "dry", "verb", "adj_verb"),
        ("the store is close", "close", "adj", "adj_verb"),
        ("they close the store", "close", "verb", "adj_verb"),
        ("the bird is free", "free", "adj", "adj_verb"),
        ("they free the bird", "free", "verb", "adj_verb"),
        ("the room is quiet", "quiet", "adj", "adj_verb"),
        ("they quiet the room", "quiet", "verb", "adj_verb"),
        ("the water is cool", "cool", "adj", "adj_verb"),
        ("they cool the water", "cool", "verb", "adj_verb"),
        ("the surface is smooth", "smooth", "adj", "adj_verb"),
        ("they smooth the surface", "smooth", "verb", "adj_verb"),
        # adj_noun
        ("the bag is light", "light", "adj", "adj_noun"),
        ("the light is bright", "light", "noun", "adj_noun"),
        ("the water is cold", "cold", "adj", "adj_noun"),
        ("the cold is severe", "cold", "noun", "adj_noun"),
        ("the answer is right", "right", "adj", "adj_noun"),
        ("the right is clear", "right", "noun", "adj_noun"),
        ("the price is fair", "fair", "adj", "adj_noun"),
        ("the fair is large", "fair", "noun", "adj_noun"),
        ("the table is round", "round", "adj", "adj_noun"),
        ("the round is over", "round", "noun", "adj_noun"),
        ("the ground is solid", "solid", "adj", "adj_noun"),
        ("the solid is hard", "solid", "noun", "adj_noun"),
        # noun_verb
        ("the fire is hot", "fire", "noun", "noun_verb"),
        ("they fire the gun", "fire", "verb", "noun_verb"),
        ("the record is old", "record", "noun", "noun_verb"),
        ("they record music", "record", "verb", "noun_verb"),
        ("the run is long", "run", "noun", "noun_verb"),
        ("they run the program", "run", "verb", "noun_verb"),
        ("the play is good", "play", "noun", "noun_verb"),
        ("they play music", "play", "verb", "noun_verb"),
        ("the sign is clear", "sign", "noun", "noun_verb"),
        ("they sign the paper", "sign", "verb", "noun_verb"),
        ("the state is large", "state", "noun", "noun_verb"),
        ("they state the facts", "state", "verb", "noun_verb"),
    ]
    stimuli = []
    for sent, target, role, rp in test_pairs:
        stimuli.append({
            "sentence": sent,
            "target_word": target,
            "token_label": target,
            "role_label": role,
            "frame_label": "causal_test",
            "role_pair": rp,
            "group": "causal_test",
        })
    return stimuli


# =====================================================================
# MODEL LOADING — BF16 + device_map="auto" + flash_attn priority
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto + flash_attn)...")
    
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            log(f"  attn_implementation={attn_impl} succeeded")
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:100]}")
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
        sorted_lids = sorted(layer_devices.keys(), key=int)
        for lid in sorted_lids[:3]:
            log(f"    L{lid}: {layer_devices[lid]}")
        if len(sorted_lids) > 6:
            log(f"    ...")
        for lid in sorted_lids[-3:]:
            log(f"    L{lid}: {layer_devices[lid]}")
    
    return model, tok


# =====================================================================
# CAPTURE & POSITION UTILITIES
# =====================================================================
def _capture_single(model, tokenizer, sent, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    logits = out.logits.detach().cpu().float()
    return {"hidden": hs, "logits": logits}

def _find_token_pos(decoded_tokens, target):
    target_lower = target.lower()
    for i, t in enumerate(decoded_tokens):
        if t == target_lower: return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower: return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower: return i
    return None

def resolve_positions(stimuli, tokenizer):
    resolved = []
    for stim in stimuli:
        toks = tokenizer.encode(stim["sentence"], add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, stim["target_word"])
        if pos is not None:
            new_stim = dict(stim); new_stim["target_pos"] = pos; resolved.append(new_stim)
    return resolved

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


# =====================================================================
# ACTIVATION PATCHING
# =====================================================================
def run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, patch_vec, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    layers = get_layers(model)
    patched_logits = None
    injection_done = [False]
    patch_tensor = torch.tensor(patch_vec, dtype=torch.bfloat16, device=input_device)
    
    def inject_hook(module, input, output):
        if not injection_done[0]:
            out_tuple = list(output)
            out_tuple[0] = out_tuple[0].clone()
            out_tuple[0][0, pos, :] += patch_tensor.to(out_tuple[0].dtype)
            injection_done[0] = True
            return tuple(out_tuple)
        return output
    
    handle = layers[layer_idx].register_forward_hook(inject_hook)
    
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, output_hidden_states=False)
            patched_logits = out.logits.detach().cpu().float()
        except Exception as e:
            log(f"  Patched forward failed at L{layer_idx}: {str(e)[:80]}")
    
    handle.remove()
    return patched_logits


# =====================================================================
# FACTORIAL DECOMPOSITION
# =====================================================================
def factorial_decomposition(cell_means, dual_tokens, token_roles, token_frames, d_model):
    """
    Full factorial decomposition using balanced design:
    
    For each (token, role, frame) cell, we have cell_means[(token, role, frame)].
    
    Decomposition (two-way ANOVA style):
      grand_mean = average of all cell means
      I(token) = token_mean - grand_mean
      R(role)  = role_mean - grand_mean
      F(frame) = frame_mean - grand_mean
      
    But we need to be careful: I, R, F are not independent in this formulation.
    We use the sequential (Type I) approach:
      1. Compute grand mean μ
      2. I(token) = mean over (role, frame) for this token - μ
      3. R(role) = mean over (token, frame) for this role - μ
      4. F(frame) = mean over (token, role) for this frame - μ
      5. RF(token, role) = cell_mean(token, role, ·) - I(token) - R(role) - μ
         Actually, we want RF as the role effect specific to each token.
    
    Since our data is inherently token-specific (each token has specific role pairs),
    we decompose per-token:
    
    For each token t with roles r1, r2:
      μ_t = grand mean across all (role, frame) cells for token t
      R_t(r1) = mean across frames for (t, r1) - μ_t  [role main effect for r1]
      R_t(r2) = mean across frames for (t, r2) - μ_t  [role main effect for r2]
      F_t(frame) = mean across roles for (t, frame) - μ_t  [frame main effect]
      RF_t(r, frame) = cell(t, r, frame) - μ_t - R_t(r) - F_t(frame)  [interaction]
      
    Then:
      R_direction = R_t(r2) - R_t(r1)  [role contrast for this token]
      F_direction(average) = average F_t(frame) weighted by frame frequency
      
    Returns dict with per-token decomposition results.
    """
    decomp = {}
    
    for token in dual_tokens:
        roles_list = sorted(token_roles[token])
        if len(roles_list) != 2: continue
        r1, r2 = roles_list
        frames_list = sorted(token_frames[token])
        
        # Get cell means for this token
        cells = {}
        for role in roles_list:
            for frame in frames_list:
                key = (token, role, frame)
                if key in cell_means:
                    cells[(role, frame)] = cell_means[key]
        
        if len(cells) < 4:  # need at least 2 roles × 2 frames
            continue
        
        # Grand mean for this token
        all_vecs = list(cells.values())
        grand_mean = np.mean(all_vecs, axis=0)
        
        # Role means (marginal)
        role_means = {}
        for role in roles_list:
            r_vecs = [cells[(role, f)] for f in frames_list if (role, f) in cells]
            if r_vecs:
                role_means[role] = np.mean(r_vecs, axis=0)
        
        # Frame means (marginal)
        frame_means = {}
        for frame in frames_list:
            f_vecs = [cells[(r, frame)] for r in roles_list if (r, frame) in cells]
            if f_vecs:
                frame_means[frame] = np.mean(f_vecs, axis=0)
        
        # Main effects
        R_effect = {}
        for role in roles_list:
            if role in role_means:
                R_effect[role] = role_means[role] - grand_mean
        
        F_effect = {}
        for frame in frames_list:
            if frame in frame_means:
                F_effect[frame] = frame_means[frame] - grand_mean
        
        # Role direction: contrast between r2 and r1
        if r1 in R_effect and r2 in R_effect:
            R_direction = R_effect[r2] - R_effect[r1]
        else:
            continue
        
        # Frame direction: average frame effect (across frames, weighted equally)
        if F_effect:
            F_direction_avg = np.mean(list(F_effect.values()), axis=0)
        else:
            F_direction_avg = np.zeros(d_model)
        
        # Per-frame frame directions
        F_directions = {f: F_effect[f] for f in frames_list if f in F_effect}
        
        # Interaction terms: RF(r, frame) = cell - μ - R(r) - F(frame)
        RF_interaction = {}
        for role in roles_list:
            for frame in frames_list:
                if (role, frame) in cells and role in R_effect and frame in F_effect:
                    residual = cells[(role, frame)] - grand_mean - R_effect[role] - F_effect[frame]
                    RF_interaction[(role, frame)] = residual
        
        # Average RF interaction for the role contrast
        RF_r2_avg = np.mean([RF_interaction[(r2, f)] for f in frames_list 
                            if (r2, f) in RF_interaction], axis=0) if any((r2, f) in RF_interaction for f in frames_list) else np.zeros(d_model)
        RF_r1_avg = np.mean([RF_interaction[(r1, f)] for f in frames_list 
                            if (r1, f) in RF_interaction], axis=0) if any((r1, f) in RF_interaction for f in frames_list) else np.zeros(d_model)
        
        # RF direction = difference in interaction terms
        RF_direction = RF_r2_avg - RF_r1_avg
        
        decomp[token] = {
            "grand_mean": grand_mean,
            "R_effect_r1": R_effect.get(r1, np.zeros(d_model)),
            "R_effect_r2": R_effect.get(r2, np.zeros(d_model)),
            "R_direction": R_direction,
            "F_direction_avg": F_direction_avg,
            "F_directions": F_directions,
            "RF_interaction": RF_interaction,
            "RF_direction": RF_direction,
            "role_means": role_means,
            "frame_means": frame_means,
            "r1": r1, "r2": r2,
            "frames": frames_list,
            "n_cells": len(cells),
        }
    
    return decomp


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase302_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 302: Full Factorial I/R/F Decomposition + Causal Test -- {model_name}")

    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers; d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")
    
    # ---- Build stimuli ----
    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)
    log(f"  Observation stimuli: {len(sub_stimuli)}, Causal test stimuli: {len(causal_stimuli)}")
    
    # Count tokens and roles
    token_roles = defaultdict(set)
    token_frames = defaultdict(set)
    token_rp = {}
    for stim in sub_stimuli:
        token_roles[stim["token_label"]].add(stim["role_label"])
        token_frames[stim["token_label"]].add(stim["frame_label"])
        token_rp[stim["token_label"]] = stim.get("role_pair", "")
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])
    log(f"  Dual-role tokens: {len(dual_tokens)} = {dual_tokens}")
    for t in dual_tokens:
        log(f"    {t}: roles={sorted(token_roles[t])}, frames={len(token_frames[t])}, pair={token_rp.get(t,'')}")
    
    # Deduplicate sentences
    all_sentences = []; sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences); all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} unique sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 50 == 0:
            el = time.time() - t0; rate = (i + 1) / max(el, 1)
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={(len(all_sentences)-i-1)/rate:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # Organize observation data
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]; role = stim["role_label"]; frame = stim.get("frame_label", "")
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, frame)].append((idx, pos))
    
    # Organize causal test pairs
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]; role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim
    dual_test = [(t, sorted(rs.keys())) for t, rs in test_pairs.items() if len(rs) >= 2]
    log(f"  Causal test pairs: {len(dual_test)} tokens with both roles")
    
    # ---- Layer selection: 8 layers for full coverage ----
    sample_layers = sorted(set([
        max(1, nl // 8), max(1, nl // 4), max(1, 3 * nl // 8),
        nl // 2, 5 * nl // 8, 3 * nl // 4, 7 * nl // 8, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers (8-point coverage): {sample_layers}")
    
    # =====================================================================
    # FACTORIAL DECOMPOSITION + CAUSAL TEST
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"FACTORIAL I/R/F DECOMPOSITION + CAUSAL TEST")
    log(f"{'='*60}")
    
    results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        
        # ---- Compute cell means ----
        cell_means = {}
        for (token, role, frame), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, frame)] = np.mean(vecs, axis=0)
        
        # ---- Factorial decomposition ----
        decomp = factorial_decomposition(cell_means, dual_tokens, token_roles, token_frames, d_model)
        log(f"  Factorial decomposition: {len(decomp)} tokens")
        
        # ---- Compute LOO R direction ----
        token_R_loo = {}
        for token in dual_tokens:
            if token not in decomp: continue
            other_R = {t: decomp[t]["R_direction"] for t in decomp if t != token}
            if other_R:
                token_R_loo[token] = np.mean(list(other_R.values()), axis=0)
        
        # ---- Causal test ----
        layer_results = {}
        
        for token, roles_list in dual_test:
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            s1 = test_pairs[token][r1]; s2 = test_pairs[token][r2]
            
            idx1 = s1.get("_idx"); pos1 = s1.get("target_pos")
            idx2 = s2.get("_idx"); pos2 = s2.get("target_pos")
            if idx1 is None or idx2 is None: continue
            
            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None: continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]: continue
            
            logits1 = captures[idx1]["logits"][0, -1, :].numpy().copy()
            logits2 = captures[idx2]["logits"][0, -1, :].numpy().copy()
            target_shift = logits2 - logits1
            
            # Full delta
            v1 = captures[idx1]["hidden"][li][0, pos1, :].numpy().copy()
            v2 = captures[idx2]["hidden"][li][0, pos2, :].numpy().copy()
            full_delta = v2 - v1
            
            # Get decomposition for this token
            d = decomp.get(token)
            if d is None:
                log(f"  {token}: no decomposition available, skipping")
                continue
            
            R_dir = d["R_direction"]
            F_dir = d["F_direction_avg"]
            RF_dir = d["RF_direction"]
            
            # LOO R
            R_loo = token_R_loo.get(token, R_dir)
            
            # Compare with residual F from Phase 301 approach
            F_residual = full_delta - R_dir
            
            # Norms
            R_norm = float(np.linalg.norm(R_dir))
            F_norm = float(np.linalg.norm(F_dir))
            RF_norm = float(np.linalg.norm(RF_dir))
            full_norm = float(np.linalg.norm(full_delta))
            
            # Cosine between factorial R and F
            cos_RF_factorial = cosine_sim(R_dir, F_dir)
            # Cosine between factorial R and residual F
            cos_RF_residual = cosine_sim(R_dir, F_residual)
            # Cosine between factorial F and residual F
            cos_F_fact_resid = cosine_sim(F_dir, F_residual)
            
            key = f"{token}_{r1}->{r2}"
            layer_results[key] = {
                "token": token, "r1": r1, "r2": r2, "role_pair": token_rp.get(token, ""),
                "R_norm": R_norm, "F_norm": F_norm, "RF_norm": RF_norm, "full_norm": full_norm,
                "cos_RF_factorial": cos_RF_factorial,
                "cos_RF_residual": cos_RF_residual,
                "cos_F_fact_resid": cos_F_fact_resid,
                "n_cells": d["n_cells"],
            }
            
            # ---- Define all patch conditions ----
            conditions = {
                # Factorial decomposition components
                "R_only": R_dir,
                "F_only": F_dir,
                "R+F": R_dir + F_dir,
                "R+F+RF": R_dir + F_dir + RF_dir,
                "RF_only": RF_dir,
                "full_delta": full_delta,
                
                # LOO versions
                "R_loo": R_loo,
                "R_loo+F": R_loo + F_dir,
                
                # Phase 301 residual F approach (for comparison)
                "F_residual": F_residual,
                "R+F_residual": R_dir + F_residual,
            }
            
            # ---- Run causal tests ----
            for cond_name, patch_vec in conditions.items():
                pnorm = np.linalg.norm(patch_vec)
                if pnorm < 1e-10:
                    layer_results[key][f"{cond_name}_cos_shift"] = 0.0
                    layer_results[key][f"{cond_name}_norm"] = 0.0
                    continue
                
                patched_logits = run_with_patched_hidden(model, tok, s1["sentence"],
                                                          li, pos1, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - logits1, target_shift)
                    layer_results[key][f"{cond_name}_cos_shift"] = float(cos_shift)
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
                else:
                    layer_results[key][f"{cond_name}_cos_shift"] = None
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
            
            # ---- Random controls (5 directions) ----
            rand_shifts = []
            for ri in range(5):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model); rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * full_norm
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            layer_results[key]["avg_random_shift"] = float(np.mean(rand_shifts)) if rand_shifts else 0.0
            
            n_done = len(layer_results)
            if n_done % 5 == 0 or n_done == len(dual_test):
                log(f"  {n_done}/{len(dual_test)} test pairs done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect(); torch.cuda.empty_cache()
        
        results[str(li)] = layer_results
        log(f"  Layer {li}: {len(layer_results)} test pairs completed")
        
        # Print layer summary
        if layer_results:
            R_only_cs = [v.get("R_only_cos_shift") for v in layer_results.values() if v.get("R_only_cos_shift") is not None]
            F_only_cs = [v.get("F_only_cos_shift") for v in layer_results.values() if v.get("F_only_cos_shift") is not None]
            RF_only_cs = [v.get("RF_only_cos_shift") for v in layer_results.values() if v.get("RF_only_cos_shift") is not None]
            RpF_cs = [v.get("R+F_cos_shift") for v in layer_results.values() if v.get("R+F_cos_shift") is not None]
            RpFpRF_cs = [v.get("R+F+RF_cos_shift") for v in layer_results.values() if v.get("R+F+RF_cos_shift") is not None]
            full_cs = [v.get("full_delta_cos_shift") for v in layer_results.values() if v.get("full_delta_cos_shift") is not None]
            rand_cs = [v.get("avg_random_shift", 0) for v in layer_results.values()]
            
            log(f"    R_only={np.mean(R_only_cs):+.4f}" if R_only_cs else "    R_only=N/A")
            log(f"    F_only={np.mean(F_only_cs):+.4f}" if F_only_cs else "    F_only=N/A")
            log(f"    RF_only={np.mean(RF_only_cs):+.4f}" if RF_only_cs else "    RF_only=N/A")
            log(f"    R+F={np.mean(RpF_cs):+.4f}" if RpF_cs else "    R+F=N/A")
            log(f"    R+F+RF={np.mean(RpFpRF_cs):+.4f}" if RpFpRF_cs else "    R+F+RF=N/A")
            log(f"    full_delta={np.mean(full_cs):+.4f}" if full_cs else "    full_delta=N/A")
            log(f"    random={np.mean(rand_cs):+.4f}" if rand_cs else "    random=N/A")
            
            # KEY DIAGNOSTIC: Is RF a real binding term?
            if RpF_cs and RpFpRF_cs:
                rf_boost = np.mean(RpFpRF_cs) - np.mean(RpF_cs)
                if rf_boost > 0.05:
                    log(f"    *** RF BINDING TERM BOOSTS: +{rf_boost:.4f} ***")
                elif rf_boost < -0.05:
                    log(f"    *** RF BINDING TERM HURTS: {rf_boost:.4f} ***")
                else:
                    log(f"    RF binding term negligible: {rf_boost:+.4f}")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "dual_tokens": dual_tokens,
        "n_dual_tokens": len(dual_tokens),
        "factorial_causal": make_serializable(results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_factorial_decomposition.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY for {model_name}")
    log(f"{'='*60}")
    
    # 1. Core comparison: R+F vs R+F+RF across layers
    log(f"\n--- Core: R+F vs R+F+RF (is RF binding real?) ---")
    for li_str, layer_res in results.items():
        if not layer_res: continue
        
        RpF_cs = [v.get("R+F_cos_shift") for v in layer_res.values() if v.get("R+F_cos_shift") is not None]
        RpFpRF_cs = [v.get("R+F+RF_cos_shift") for v in layer_res.values() if v.get("R+F+RF_cos_shift") is not None]
        R_only_cs = [v.get("R_only_cos_shift") for v in layer_res.values() if v.get("R_only_cos_shift") is not None]
        F_only_cs = [v.get("F_only_cos_shift") for v in layer_res.values() if v.get("F_only_cos_shift") is not None]
        RF_only_cs = [v.get("RF_only_cos_shift") for v in layer_res.values() if v.get("RF_only_cos_shift") is not None]
        full_cs = [v.get("full_delta_cos_shift") for v in layer_res.values() if v.get("full_delta_cos_shift") is not None]
        RpF_resid_cs = [v.get("R+F_residual_cos_shift") for v in layer_res.values() if v.get("R+F_residual_cos_shift") is not None]
        rand_cs = [v.get("avg_random_shift", 0) for v in layer_res.values()]
        
        log(f"\n  Layer {li_str}:")
        if R_only_cs: log(f"    R_only:       avg={np.mean(R_only_cs):+.4f} pos={sum(1 for s in R_only_cs if s>0)}/{len(R_only_cs)}")
        if F_only_cs: log(f"    F_only:       avg={np.mean(F_only_cs):+.4f} pos={sum(1 for s in F_only_cs if s>0)}/{len(F_only_cs)}")
        if RF_only_cs: log(f"    RF_only:      avg={np.mean(RF_only_cs):+.4f} pos={sum(1 for s in RF_only_cs if s>0)}/{len(RF_only_cs)}")
        if RpF_cs: log(f"    R+F:          avg={np.mean(RpF_cs):+.4f} pos={sum(1 for s in RpF_cs if s>0)}/{len(RpF_cs)}")
        if RpFpRF_cs: log(f"    R+F+RF:       avg={np.mean(RpFpRF_cs):+.4f} pos={sum(1 for s in RpFpRF_cs if s>0)}/{len(RpFpRF_cs)}")
        if full_cs: log(f"    full_delta:   avg={np.mean(full_cs):+.4f}")
        if RpF_resid_cs: log(f"    R+F_residual: avg={np.mean(RpF_resid_cs):+.4f} (Phase 301 style)")
        if rand_cs: log(f"    random:       avg={np.mean(rand_cs):+.4f}")
        
        if RpF_cs and RpFpRF_cs:
            rf_boost = np.mean(RpFpRF_cs) - np.mean(RpF_cs)
            log(f"    RF boost: {rf_boost:+.4f}")
    
    # 2. Factorial vs Residual F comparison
    log(f"\n--- Factorial F vs Residual F (Phase 301 comparison) ---")
    for li_str, layer_res in results.items():
        if not layer_res: continue
        
        cos_F_vals = [v.get("cos_F_fact_resid", 0) for v in layer_res.values() 
                     if v.get("cos_F_fact_resid") is not None]
        cos_RF_fact = [v.get("cos_RF_factorial", 0) for v in layer_res.values()
                      if v.get("cos_RF_factorial") is not None]
        cos_RF_resid = [v.get("cos_RF_residual", 0) for v in layer_res.values()
                       if v.get("cos_RF_residual") is not None]
        
        if cos_RF_fact:
            log(f"  L{li_str}: cos(R,F)_factorial avg={np.mean(cos_RF_fact):+.4f} std={np.std(cos_RF_fact):.4f} "
                f"range=[{min(cos_RF_fact):+.4f}, {max(cos_RF_fact):+.4f}]")
        if cos_RF_resid:
            log(f"  L{li_str}: cos(R,F)_residual  avg={np.mean(cos_RF_resid):+.4f} std={np.std(cos_RF_resid):.4f} "
                f"range=[{min(cos_RF_resid):+.4f}, {max(cos_RF_resid):+.4f}]")
        if cos_F_vals:
            log(f"  L{li_str}: cos(F_fact, F_resid) avg={np.mean(cos_F_vals):+.4f}")
    
    # 3. Per-role-pair breakdown at mid layer
    mid_li = str(nl // 2)
    log(f"\n--- Per-Role-Pair at Layer {mid_li} ---")
    if mid_li in results:
        rp_groups = defaultdict(list)
        for key, v in results[mid_li].items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp, items in sorted(rp_groups.items()):
            R_cs = [v.get("R_only_cos_shift", 0) for v in items]
            F_cs = [v.get("F_only_cos_shift", 0) for v in items]
            RF_cs = [v.get("RF_only_cos_shift", 0) for v in items]
            RpF_cs = [v.get("R+F_cos_shift", 0) for v in items]
            RpFpRF_cs = [v.get("R+F+RF_cos_shift", 0) for v in items]
            
            log(f"  {rp} ({len(items)} tokens):")
            log(f"    R={np.mean(R_cs):+.4f} F={np.mean(F_cs):+.4f} RF={np.mean(RF_cs):+.4f} "
                f"R+F={np.mean(RpF_cs):+.4f} R+F+RF={np.mean(RpFpRF_cs):+.4f}")
    
    # 4. Cos(R,F) distribution comparison
    log(f"\n--- Cos(R,F) Distribution: Factorial vs Residual ---")
    all_cos_fact = []; all_cos_resid = []
    for li_str, layer_res in results.items():
        for key, v in layer_res.items():
            cv = v.get("cos_RF_factorial")
            if cv is not None: all_cos_fact.append((li_str, v["token"], v.get("role_pair",""), cv))
            cv2 = v.get("cos_RF_residual")
            if cv2 is not None: all_cos_resid.append((li_str, v["token"], v.get("role_pair",""), cv2))
    
    if all_cos_fact:
        vals = [c[3] for c in all_cos_fact]
        extreme_count = sum(1 for v in vals if abs(v) > 0.9)
        log(f"  Factorial cos(R,F): mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
            f"range=[{min(vals):+.4f}, {max(vals):+.4f}], |cos|>0.9: {extreme_count}/{len(vals)}")
    if all_cos_resid:
        vals = [c[3] for c in all_cos_resid]
        extreme_count = sum(1 for v in vals if abs(v) > 0.9)
        log(f"  Residual cos(R,F):  mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
            f"range=[{min(vals):+.4f}, {max(vals):+.4f}], |cos|>0.9: {extreme_count}/{len(vals)}")
    
    release_model(model)
    log(f"Phase 302 complete for {model_name}")

if __name__ == "__main__":
    main()
