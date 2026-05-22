"""
Phase 58b: 骨干子空间语义解码 — 修正版
========================================
问题诊断: Phase 58用统一模板导致模板效应压倒词义差异 (shared=0.993)
修正: 使用词汇特定自然语境模板, 每个词有独立的多样化上下文

核心实验:
  Part 1: 25对概念对 — 词汇特定模板
  Part 2: 骨干子空间提取与W_U解码
  Part 3: 神经元级归属分析
  Part 4: shared_ratio = f(semantic_similarity) 函数

关键区别: 每个词有15个独立的自然语境模板, 不使用统一模板
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import (
    load_model, get_layers, get_model_info, get_layer_weights,
    get_W_U, release_model, safe_decode, MODEL_CONFIGS
)

import torch

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)

# ===== 词汇特定模板 =====
# 每个词有15个自然的、多样化的上下文模板
WORD_TEMPLATES = {
    # --- 上下位 ---
    "apple": [
        "I ate a fresh apple this morning", "The apple was sweet and juicy",
        "She picked a red apple from the tree", "He bought three apples at the market",
        "The apple pie smells amazing", "She sliced the apple for the salad",
        "A rotten apple was in the basket", "The apple tree is blooming",
        "This apple is organic", "The apple juice was refreshing",
        "She offered me an apple", "The apple tasted sour",
        "I found a wild apple in the forest", "The apple season has begun",
        "The apple cider is warm",
    ],
    "fruit": [
        "I ate fresh fruit this morning", "The fruit was sweet and juicy",
        "She picked some fruit from the tree", "He bought fresh fruit at the market",
        "The fruit salad smells amazing", "She sliced the fruit for the salad",
        "A rotten fruit was in the basket", "The fruit trees are blooming",
        "This fruit is organic", "The fruit juice was refreshing",
        "She offered me some fruit", "The fruit tasted sour",
        "I found wild fruit in the forest", "The fruit season has begun",
        "The fruit punch is warm",
    ],
    "dog": [
        "The dog barked loudly at the mailman", "She adopted a rescue dog last week",
        "He walks his dog every morning", "The dog chased the ball across the yard",
        "My dog loves to play fetch", "The dog slept on the couch all day",
        "A stray dog wandered into the yard", "The dog park was crowded today",
        "This dog is very friendly", "The dog trainer taught new tricks",
        "She fed the dog before leaving", "The dog howled at the moon",
        "He trained his dog to sit", "The dog breed is very popular",
        "The dog wagged its tail happily",
    ],
    "animal": [
        "The animal barked loudly at the intruder", "She adopted a rescue animal last week",
        "He studies animal behavior every morning", "The animal chased its prey across the field",
        "My favorite animal is the elephant", "The animal slept in the den all day",
        "A wild animal wandered into the yard", "The animal shelter was crowded today",
        "This animal is very dangerous", "The animal trainer taught new tricks",
        "She fed the animal before leaving", "The animal howled at the moon",
        "He trained the animal to perform", "The animal species is very rare",
        "The animal moved gracefully through the forest",
    ],
    "red": [
        "She wore a red dress to the party", "The red paint covered the entire wall",
        "He prefers red wine over white", "The red light means stop",
        "The red rose looked beautiful", "She has red hair and freckles",
        "The red car sped past us", "A red flag was raised",
        "This red is very vibrant", "The red ink stained the paper",
        "He turned red with embarrassment", "The red carpet was rolled out",
        "She chose the red one", "The red sunset was breathtaking",
        "The red bus arrived late",
    ],
    "color": [
        "She wore a bright color to the party", "The paint color covered the entire wall",
        "He prefers warm color tones", "The color scheme means a lot",
        "The color palette looked beautiful", "She changed her hair color recently",
        "The color combination worked well", "A new color was introduced",
        "This color is very vibrant", "The color palette inspired the artist",
        "He mixed the color carefully", "The color trend was popular",
        "She chose a neutral color", "The color theory is fascinating",
        "The color choices reflected the brand",
    ],
    "Paris": [
        "Paris is known for the Eiffel Tower", "She visited Paris last summer",
        "The streets of Paris are charming", "He fell in love with Paris",
        "Paris has amazing museums", "The food in Paris is incredible",
        "Paris Fashion Week is world-famous", "She moved to Paris for work",
        "Paris in springtime is magical", "The architecture of Paris is stunning",
        "He studied art in Paris", "Paris has an efficient metro system",
        "She dreams of living in Paris", "The cafes in Paris are cozy",
        "Paris hosted the Olympics before",
    ],
    "city": [
        "The city is known for its skyline", "She visited the city last summer",
        "The streets of the city are busy", "He fell in love with the city",
        "The city has amazing museums", "The food in the city is diverse",
        "The city festival is world-famous", "She moved to the city for work",
        "The city in springtime is beautiful", "The architecture of the city is modern",
        "He studied art in the city", "The city has efficient public transport",
        "She dreams of living in the city", "The cafes in the city are popular",
        "The city hosted the conference before",
    ],
    "piano": [
        "She played the piano beautifully", "The piano stood in the corner of the room",
        "He learned piano at age five", "The piano concert was sold out",
        "A grand piano dominated the stage", "She practices piano every day",
        "The piano keys were ivory", "He tuned the piano himself",
        "This piano is an antique", "The piano sonata was moving",
        "She composed music for piano", "The piano needed repair",
        "He bought a digital piano", "The piano teacher was patient",
        "The piano accompaniment was perfect",
    ],
    "instrument": [
        "She played the instrument beautifully", "The instrument stood in the corner of the room",
        "He learned the instrument at age five", "The instrument concert was sold out",
        "A brass instrument dominated the stage", "She practices the instrument every day",
        "The instrument keys were polished", "He tuned the instrument himself",
        "This instrument is an antique", "The instrument solo was moving",
        "She composed music for the instrument", "The instrument needed repair",
        "He bought a new instrument", "The instrument teacher was patient",
        "The instrument accompaniment was perfect",
    ],
    # --- 同义 ---
    "big": [
        "The big house loomed over the street", "She made a big decision yesterday",
        "He has a big family with six kids", "The big dog scared the children",
        "This is a really big problem", "The big screen showed the movie",
        "She won a big prize at the fair", "The big storm damaged the roof",
        "He took a big step forward", "The big city never sleeps",
        "A big crowd gathered downtown", "The big news spread quickly",
        "She has big plans for the future", "The big boat sailed away",
        "That was a big mistake",
    ],
    "large": [
        "The large house loomed over the street", "She made a large purchase yesterday",
        "He has a large family with six kids", "The large dog scared the children",
        "This is a really large problem", "The large screen showed the movie",
        "She won a large prize at the fair", "The large storm damaged the roof",
        "He took a large step forward", "The large city never sleeps",
        "A large crowd gathered downtown", "The large order was shipped",
        "She has large plans for the future", "The large boat sailed away",
        "That was a large error",
    ],
    "happy": [
        "She felt happy about the news", "The happy children played outside",
        "He looked happy after the exam", "A happy smile crossed her face",
        "They shared a happy moment together", "The happy couple danced all night",
        "She sent a happy birthday wish", "The happy ending surprised everyone",
        "He lived a happy life in the country", "The happy song played on the radio",
        "A happy memory came to mind", "The happy occasion called for celebration",
        "She seemed happy with the results", "The happy dog wagged its tail",
        "He was happy to help out",
    ],
    "glad": [
        "She felt glad about the news", "The glad children played outside",
        "He looked glad after the exam", "A glad smile crossed her face",
        "They shared a glad moment together", "The glad couple danced all night",
        "She sent a glad birthday wish", "The glad ending surprised everyone",
        "He lived a glad life in the country", "The glad song played on the radio",
        "A glad memory came to mind", "The glad occasion called for celebration",
        "She seemed glad with the results", "The glad dog wagged its tail",
        "He was glad to help out",
    ],
    "fast": [
        "The fast car won the race", "She runs very fast in competitions",
        "He made a fast decision", "The fast train arrived early",
        "This is a fast computer", "The fast food restaurant was nearby",
        "She gave a fast response", "The fast pace exhausted him",
        "He learned fast from his mistakes", "The fast river flowed downhill",
        "A fast heartbeat worried her", "The fast track program was intense",
        "She typed fast on the keyboard", "The fast wind blew the leaves",
        "That was a fast game",
    ],
    "quick": [
        "The quick car won the race", "She runs very quick in sprints",
        "He made a quick decision", "The quick train arrived early",
        "This is a quick computer", "The quick lunch break was brief",
        "She gave a quick response", "The quick pace exhausted him",
        "He learned quick from his mistakes", "The quick river flowed downhill",
        "A quick heartbeat worried her", "The quick review was helpful",
        "She typed quick on the keyboard", "The quick wind blew the leaves",
        "That was a quick game",
    ],
    "begin": [
        "The class will begin at noon", "She began her journey last week",
        "He began to understand the problem", "The concert began with a solo",
        "Let us begin the meeting", "The story begins in a small town",
        "She began working early today", "The game began after the anthem",
        "He began his speech confidently", "The project begins next month",
        "They began construction yesterday", "The semester begins in September",
        "She began to feel better", "The race began at dawn",
        "We began the experiment carefully",
    ],
    "start": [
        "The class will start at noon", "She started her journey last week",
        "He started to understand the problem", "The concert started with a solo",
        "Let us start the meeting", "The story starts in a small town",
        "She started working early today", "The game started after the anthem",
        "He started his speech confidently", "The project starts next month",
        "They started construction yesterday", "The semester starts in September",
        "She started to feel better", "The race started at dawn",
        "We started the experiment carefully",
    ],
    "beautiful": [
        "The beautiful garden attracted visitors", "She wore a beautiful dress",
        "He painted a beautiful landscape", "The beautiful music moved everyone",
        "This is a beautiful day", "The beautiful sunset was unforgettable",
        "She has beautiful handwriting", "The beautiful building stood downtown",
        "He wrote a beautiful poem", "The beautiful beach was pristine",
        "A beautiful melody played softly", "The beautiful woman smiled",
        "She found a beautiful shell", "The beautiful design won awards",
        "The beautiful weather continued",
    ],
    "pretty": [
        "The pretty garden attracted visitors", "She wore a pretty dress",
        "He painted a pretty landscape", "The pretty music charmed everyone",
        "This is a pretty day", "The pretty sunset was delightful",
        "She has pretty handwriting", "The pretty building stood downtown",
        "He wrote a pretty poem", "The pretty beach was charming",
        "A pretty melody played softly", "The pretty woman smiled",
        "She found a pretty shell", "The pretty design won awards",
        "The pretty weather continued",
    ],
    # --- 反义 ---
    "hot": [
        "The hot coffee burned my tongue", "It was a hot summer day",
        "She likes hot sauce on everything", "The hot air balloon rose slowly",
        "The stove was hot to touch", "He took a hot shower",
        "The hot topic was controversial", "She served hot soup for dinner",
        "The hot spring was relaxing", "The hot metal glowed red",
        "He felt hot and sweaty", "The hot weather continued",
        "She drank hot chocolate", "The hot rod raced down the street",
        "The hot desert stretched endlessly",
    ],
    "cold": [
        "The cold water shocked my system", "It was a cold winter day",
        "She likes cold drinks in summer", "The cold wind blew fiercely",
        "The ice was cold to touch", "He took a cold shower",
        "The cold war was tense", "She served cold soup for appetizer",
        "The cold stream was refreshing", "The cold metal felt smooth",
        "He felt cold and shivery", "The cold weather persisted",
        "She drank cold lemonade", "The cold front approached",
        "The cold tundra stretched endlessly",
    ],
    "up": [
        "She looked up at the ceiling", "He climbed up the stairs",
        "The prices went up again", "She stood up from her chair",
        "He grew up in a small town", "The sun came up early",
        "She turned up the volume", "He picked up the phone",
        "The temperature went up", "She signed up for the class",
        "He woke up at dawn", "The road led up the mountain",
        "She folded up the letter", "He sped up the car",
        "The balloon went up in the air",
    ],
    "down": [
        "She looked down at the floor", "He walked down the stairs",
        "The prices went down again", "She sat down on her chair",
        "He calmed down after the argument", "The sun went down early",
        "She turned down the volume", "He put down the phone",
        "The temperature went down", "She wrote down the notes",
        "He lay down on the bed", "The road led down the mountain",
        "She tore down the poster", "He slowed down the car",
        "The balloon floated down slowly",
    ],
    "love": [
        "She loves her family deeply", "He fell in love at first sight",
        "They love spending time together", "She wrote a love letter",
        "The love story touched many hearts", "He expressed his love openly",
        "She has love for all animals", "The love of music inspired him",
        "They shared a love of nature", "Her love for cooking grew",
        "He found love unexpectedly", "The love between them was obvious",
        "She loves reading novels", "The love song was beautiful",
        "He sent love to his friends",
    ],
    "hate": [
        "She hates her job deeply", "He developed hate at first sight",
        "They hate being apart", "She wrote a hate letter",
        "The hate crime shocked many people", "He expressed his hate openly",
        "She has hate for certain foods", "The hate of injustice motivated him",
        "They shared a hate of corruption", "Her hate for waste grew",
        "He found hate unexpectedly", "The hate between them was obvious",
        "She hates reading instructions", "The hate speech was condemned",
        "He sent hate to his enemies",
    ],
    "light": [
        "The light filled the room", "She turned on the light",
        "He carried a light suitcase", "The light rain fell gently",
        "She prefers light colors", "The light breeze was pleasant",
        "He read by the light of the lamp", "The traffic light turned green",
        "She wore a light jacket", "The light show was spectacular",
        "He provided light for the group", "The light was too bright",
        "She lit a light candle", "The light reflected off the water",
        "The morning light was beautiful",
    ],
    "dark": [
        "The dark filled the room", "She turned off the dark",
        "He carried a dark suitcase", "The dark clouds gathered",
        "She prefers dark colors", "The dark night was cold",
        "He read in the dark corner", "The dark mood persisted",
        "She wore a dark jacket", "The dark matter was mysterious",
        "He feared the dark as a child", "The dark was too overwhelming",
        "She entered the dark room", "The dark reflected no light",
        "The evening dark was approaching",
    ],
    "young": [
        "The young student was eager to learn", "She met him when he was young",
        "The young tree bent in the wind", "He has a young family",
        "The young generation thinks differently", "She looked young for her age",
        "The young athlete broke the record", "He advised the young entrepreneur",
        "The young bird left the nest", "She hired a young assistant",
        "The young child played happily", "He was young and ambitious",
        "The young artist showed promise", "She felt young at heart",
        "The young horse ran fast",
    ],
    "old": [
        "The old professor was wise", "She met him when he was old",
        "The old tree fell in the storm", "He has an old family tradition",
        "The old generation thinks differently", "She looked old for her age",
        "The old athlete retired gracefully", "He advised the old businessman",
        "The old bird could not fly", "She hired an old advisor",
        "The old man sat quietly", "He was old and experienced",
        "The old artist was famous", "She felt old before her time",
        "The old horse walked slowly",
    ],
    # --- 相关 ---
    "doctor": [
        "The doctor prescribed medication", "She visited the doctor yesterday",
        "He became a doctor at thirty", "The doctor examined the patient",
        "She consulted a doctor about the pain", "The doctor performed surgery",
        "He trusted his doctor completely", "The doctor works long hours",
        "She thanked the doctor for the help", "The doctor gave clear instructions",
        "He called the doctor urgently", "The doctor specialized in children",
        "She admired the doctor greatly", "The doctor diagnosed the condition",
        "He recommended a good doctor",
    ],
    "hospital": [
        "The hospital admitted new patients", "She visited the hospital yesterday",
        "He worked at the hospital for years", "The hospital treated the emergency",
        "She went to the hospital about the pain", "The hospital performed many surgeries",
        "He trusted the hospital completely", "The hospital operates twenty four hours",
        "She thanked the hospital for the care", "The hospital gave clear directions",
        "He called the hospital urgently", "The hospital specialized in children",
        "She admired the hospital staff", "The hospital managed the crisis",
        "He recommended a good hospital",
    ],
    "chef": [
        "The chef prepared an amazing meal", "She studied under a famous chef",
        "He became a chef after culinary school", "The chef demonstrated the recipe",
        "She asked the chef for cooking tips", "The chef won a prestigious award",
        "He watched the chef work carefully", "The chef created a new dish",
        "She praised the chef for the dinner", "The chef used fresh ingredients",
        "He assisted the chef in the kitchen", "The chef specialized in French cuisine",
        "She admired the chef greatly", "The chef trained many apprentices",
        "He recognized the chef immediately",
    ],
    "kitchen": [
        "The kitchen was spotlessly clean", "She studied the kitchen layout carefully",
        "He worked in the kitchen for years", "The kitchen had modern equipment",
        "She organized the kitchen efficiently", "The kitchen won a design award",
        "He cleaned the kitchen thoroughly", "The kitchen was the heart of the home",
        "She renovated the kitchen recently", "The kitchen stored fresh ingredients",
        "He built the kitchen cabinets himself", "The kitchen featured modern appliances",
        "She admired the kitchen design", "The kitchen served many meals",
        "He recognized the kitchen immediately",
    ],
    "teacher": [
        "The teacher explained the concept clearly", "She respected her teacher greatly",
        "He became a teacher after college", "The teacher graded the exams",
        "She asked the teacher for help", "The teacher inspired the students",
        "He thanked the teacher for guidance", "The teacher prepared lesson plans",
        "She praised the teacher for patience", "The teacher used creative methods",
        "He assisted the teacher in class", "The teacher specialized in science",
        "She admired the teacher greatly", "The teacher mentored young people",
        "He recommended a good teacher",
    ],
    "school": [
        "The school explained the policy clearly", "She respected her school greatly",
        "He attended the school after moving", "The school evaluated the students",
        "She asked the school for assistance", "The school inspired the community",
        "He thanked the school for support", "The school planned the curriculum",
        "She praised the school for excellence", "The school used innovative methods",
        "He assisted the school as a volunteer", "The school specialized in science",
        "She admired the school greatly", "The school mentored young people",
        "He recommended a good school",
    ],
    "bird": [
        "The bird sang beautifully at dawn", "She spotted a rare bird",
        "He watched the bird build a nest", "The bird flew across the sky",
        "She fed the bird breadcrumbs", "The bird migrated south for winter",
        "He photographed the bird in flight", "The bird perched on the branch",
        "She studied bird behavior", "The bird had colorful feathers",
        "He rescued an injured bird", "The bird species was endangered",
        "She admired the bird greatly", "The bird chirped happily",
        "He sketched the bird carefully",
    ],
    "nest": [
        "The nest was carefully constructed", "She spotted a bird nest",
        "He watched the nest being built", "The nest sat high in the tree",
        "She found the nest in the garden", "The nest was abandoned in winter",
        "He photographed the nest carefully", "The nest held several eggs",
        "She studied the nest structure", "The nest had soft lining",
        "He discovered an old nest", "The nest was well hidden",
        "She admired the nest greatly", "The nest was perfectly shaped",
        "He sketched the nest carefully",
    ],
    "fish": [
        "The fish swam in the clear pond", "She caught a big fish",
        "He watched the fish glide through water", "The fish jumped out of the lake",
        "She fed the fish every morning", "The fish migrated upstream",
        "He cooked the fish for dinner", "The fish had shiny scales",
        "She studied fish behavior", "The fish was brightly colored",
        "He kept fish in an aquarium", "The fish species was rare",
        "She admired the fish greatly", "The fish breathed through gills",
        "He sketched the fish carefully",
    ],
    "water": [
        "The water flowed in the clear stream", "She drank a big glass of water",
        "He watched the water move slowly", "The water splashed out of the pool",
        "She added water every morning", "The water rose after the rain",
        "He boiled the water for tea", "The water had a slight reflection",
        "She studied water chemistry", "The water was crystal clear",
        "He kept water in a bottle", "The water source was pristine",
        "She admired the water view", "The water evaporated in the sun",
        "He sketched the water surface carefully",
    ],
    # --- 无关 ---
    "planet": [
        "The planet orbits a distant star", "She studied the planet carefully",
        "He discovered a new planet", "The planet has a thick atmosphere",
        "She observed the planet through a telescope", "The planet was named after a god",
        "He wrote about the planet in his journal", "The planet rotated slowly",
        "She mapped the planet surface", "The planet had no moons",
        "He explored the planet in simulation", "The planet was gas giant",
        "She calculated the planet mass", "The planet was billions of years old",
        "He compared the planet to Earth",
    ],
    "math": [
        "The math problem was challenging", "She studied math carefully",
        "He learned advanced math in college", "The math test was difficult",
        "She solved the math equation quickly", "The math theorem was famous",
        "He wrote about math in his paper", "The math class ran slowly",
        "She taught math to children", "The math had no easy solution",
        "He explored math in depth", "The math concept was abstract",
        "She calculated the math result", "The math was developed centuries ago",
        "He compared math to art",
    ],
    "democracy": [
        "Democracy requires active participation", "She studied democracy in college",
        "He defended democracy passionately", "Democracy evolved over centuries",
        "She believes in democratic principles", "Democracy allows free elections",
        "He wrote about democracy extensively", "Democracy faces many challenges",
        "She taught democracy to students", "Democracy protects minority rights",
        "He explored democracy in practice", "Democracy is a complex system",
        "She analyzed democracy critically", "Democracy was practiced in ancient times",
        "He compared democracy to other systems",
    ],
    "bacteria": [
        "The bacteria multiplied rapidly", "She studied bacteria carefully",
        "He identified the bacteria species", "The bacteria evolved resistance",
        "She observed bacteria under a microscope", "The bacteria was named after a scientist",
        "He wrote about bacteria in his research", "The bacteria reproduced slowly",
        "She cultured the bacteria in the lab", "The bacteria had no nucleus",
        "He explored bacteria in the environment", "The bacteria was beneficial",
        "She analyzed bacteria genetically", "The bacteria was millions of years old",
        "He compared bacteria to viruses",
    ],
    "poem": [
        "The poem moved her to tears", "She studied the poem carefully",
        "He wrote a beautiful poem", "The poem had deep meaning",
        "She recited the poem from memory", "The poem was famous worldwide",
        "He published the poem last year", "The poem flowed rhythmically",
        "She analyzed the poem in class", "The poem had no title",
        "He translated the poem into English", "The poem was written centuries ago",
        "She compared the poem to a song", "The poem inspired many readers",
        "He memorized the poem as a child",
    ],
}

SEMANTIC_PAIRS = {
    "hyponym_1": {"w_a": "apple", "w_b": "fruit", "relation": "hyponym", "distance": 1},
    "hyponym_2": {"w_a": "dog", "w_b": "animal", "relation": "hyponym", "distance": 1},
    "hyponym_3": {"w_a": "red", "w_b": "color", "relation": "hyponym", "distance": 1},
    "hyponym_4": {"w_a": "Paris", "w_b": "city", "relation": "hyponym", "distance": 1},
    "hyponym_5": {"w_a": "piano", "w_b": "instrument", "relation": "hyponym", "distance": 1},
    "synonym_1": {"w_a": "big", "w_b": "large", "relation": "synonym", "distance": 2},
    "synonym_2": {"w_a": "happy", "w_b": "glad", "relation": "synonym", "distance": 2},
    "synonym_3": {"w_a": "fast", "w_b": "quick", "relation": "synonym", "distance": 2},
    "synonym_4": {"w_a": "begin", "w_b": "start", "relation": "synonym", "distance": 2},
    "synonym_5": {"w_a": "beautiful", "w_b": "pretty", "relation": "synonym", "distance": 2},
    "antonym_1": {"w_a": "hot", "w_b": "cold", "relation": "antonym", "distance": 3},
    "antonym_2": {"w_a": "up", "w_b": "down", "relation": "antonym", "distance": 3},
    "antonym_3": {"w_a": "love", "w_b": "hate", "relation": "antonym", "distance": 3},
    "antonym_4": {"w_a": "light", "w_b": "dark", "relation": "antonym", "distance": 3},
    "antonym_5": {"w_a": "young", "w_b": "old", "relation": "antonym", "distance": 3},
    "associated_1": {"w_a": "doctor", "w_b": "hospital", "relation": "associated", "distance": 4},
    "associated_2": {"w_a": "chef", "w_b": "kitchen", "relation": "associated", "distance": 4},
    "associated_3": {"w_a": "teacher", "w_b": "school", "relation": "associated", "distance": 4},
    "associated_4": {"w_a": "bird", "w_b": "nest", "relation": "associated", "distance": 4},
    "associated_5": {"w_a": "fish", "w_b": "water", "relation": "associated", "distance": 4},
    "unrelated_1": {"w_a": "apple", "w_b": "planet", "relation": "unrelated", "distance": 5},
    "unrelated_2": {"w_a": "dog", "w_b": "math", "relation": "unrelated", "distance": 5},
    "unrelated_3": {"w_a": "red", "w_b": "democracy", "relation": "unrelated", "distance": 5},
    "unrelated_4": {"w_a": "piano", "w_b": "bacteria", "relation": "unrelated", "distance": 5},
    "unrelated_5": {"w_a": "city", "w_b": "poem", "relation": "unrelated", "distance": 5},
}


def find_target_pos_in_full(tokenizer, input_ids, target_word):
    """在完整token序列中找目标词位置"""
    tokens_list = input_ids[0].tolist()
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped == target_word.lower():
                return i, j - i
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 3:
                return i, j - i
    return None, None


def collect_word_activations(model, tokenizer, device, word, templates,
                              target_layers, n_layers):
    """收集一个词在所有模板中的激活"""
    activations = {li: [] for li in target_layers}
    found = 0
    with torch.no_grad():
        for tmpl in templates:
            sentence = tmpl  # 模板已经是完整句子
            inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            pos, tlen = find_target_pos_in_full(tokenizer, input_ids, word)
            if pos is None or pos >= seq_len:
                continue
            actual_pos = pos + (tlen // 2)
            actual_pos = min(actual_pos, seq_len - 1)
            found += 1
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states
            for li in target_layers:
                layer_act = hidden[li + 1][0, actual_pos].detach().cpu().float().numpy()
                activations[li].append(layer_act)
    return activations, found


def extract_subspace(vectors, n_dims=15):
    """PCA提取子空间"""
    if len(vectors) < 2:
        return None, None, None
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / len(X_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx][:, :min(n_dims, len(eigenvalues))], eigenvalues[idx][:min(n_dims, len(eigenvalues))], mean


def compute_subspace_overlap(basis_a, basis_b):
    if basis_a is None or basis_b is None:
        return 0.0
    proj = basis_b.T @ basis_a @ basis_a.T @ basis_b
    return float(np.trace(proj) / min(basis_a.shape[1], basis_b.shape[1]))


def compute_shared_specific(activations_a, activations_b, n_dims=20):
    """提取共享/独特子空间 + 方差分解"""
    all_acts = activations_a + activations_b
    X_all = np.array(all_acts)
    mean_all = X_all.mean(axis=0)
    
    X_centered = X_all - mean_all
    cov = X_centered.T @ X_centered / len(X_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    n = min(n_dims, len(eigenvalues))
    shared_basis = eigenvectors[:, :n]
    
    X_a = np.array(activations_a) - mean_all
    X_b = np.array(activations_b) - mean_all
    proj_a = X_a @ shared_basis @ shared_basis.T
    proj_b = X_b @ shared_basis @ shared_basis.T
    res_a = X_a - proj_a
    res_b = X_b - proj_b
    
    var_total_a = np.sum(X_a ** 2)
    var_shared_a = np.sum(proj_a ** 2)
    var_total_b = np.sum(X_b ** 2)
    var_shared_b = np.sum(proj_b ** 2)
    
    shared_ratio_a = var_shared_a / max(var_total_a, 1e-10)
    shared_ratio_b = var_shared_b / max(var_total_b, 1e-10)
    
    mean_a = np.array(activations_a).mean(axis=0) - mean_all
    mean_b = np.array(activations_b).mean(axis=0) - mean_all
    delta = mean_a - mean_b
    
    delta_shared = shared_basis.T @ delta
    delta_shared_energy = np.sum(delta_shared ** 2)
    delta_total_energy = np.sum(delta ** 2)
    delta_unique_ratio = 1.0 - delta_shared_energy / max(delta_total_energy, 1e-10)
    
    cos = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10)
    
    # 子空间重叠度
    basis_a, _, _ = extract_subspace(activations_a, n_dims=10)
    basis_b, _, _ = extract_subspace(activations_b, n_dims=10)
    overlap = compute_subspace_overlap(basis_a, basis_b)
    
    return {
        "shared_ratio_A": float(shared_ratio_a),
        "shared_ratio_B": float(shared_ratio_b),
        "avg_shared_ratio": float((shared_ratio_a + shared_ratio_b) / 2),
        "delta_unique_ratio": float(delta_unique_ratio),
        "cos_mean": float(cos),
        "subspace_overlap": float(overlap),
        "n_samples_a": len(activations_a),
        "n_samples_b": len(activations_b),
    }, shared_basis, mean_all, eigenvalues[:n]


def decode_direction(direction, W_U, tokenizer, top_k=30):
    """投影方向到W_U解码"""
    logits = W_U @ direction
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probs)[::-1][:top_k]
    decoded = []
    for idx in top_indices:
        token_str = safe_decode(tokenizer, idx)
        decoded.append({"token": token_str, "index": int(idx),
                       "logit": float(logits[idx]), "prob": float(probs[idx])})
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--n_dims", type=int, default=20)
    args = parser.parse_args()
    model_name = args.model
    n_dims = args.n_dims
    
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    target_layers = sorted(set([0, 1] + list(range(0, n_layers, max(1, n_layers // 10)))))
    log_time(f"{model_name}: n_layers={n_layers}, d_model={d_model}, layers={target_layers}")
    
    # ===== Part 1: 25对概念对 =====
    log_time(f"Part 1: {len(SEMANTIC_PAIRS)} pairs with word-specific templates")
    
    all_pair_results = {}
    all_activations = {}
    
    for pair_key, pair_info in SEMANTIC_PAIRS.items():
        w_a = pair_info["w_a"]
        w_b = pair_info["w_b"]
        
        if w_a not in WORD_TEMPLATES or w_b not in WORD_TEMPLATES:
            log_time(f"  SKIP {pair_key}: templates missing")
            continue
        
        log_time(f"  {pair_key}: {w_a}/{w_b} ({pair_info['relation']})")
        
        acts_a, found_a = collect_word_activations(
            model, tokenizer, device, w_a, WORD_TEMPLATES[w_a], target_layers, n_layers)
        acts_b, found_b = collect_word_activations(
            model, tokenizer, device, w_b, WORD_TEMPLATES[w_b], target_layers, n_layers)
        
        log_time(f"    Found: {w_a}={found_a}/15, {w_b}={found_b}/15")
        
        all_activations[pair_key] = {"a": acts_a, "b": acts_b}
        
        layer_results = {}
        for li in target_layers:
            if len(acts_a[li]) >= 2 and len(acts_b[li]) >= 2:
                metrics, _, _, _ = compute_shared_specific(acts_a[li], acts_b[li], n_dims)
                layer_results[str(li)] = metrics
            else:
                layer_results[str(li)] = {"error": "insufficient"}
        
        all_pair_results[pair_key] = {"pair_info": pair_info, "layers": layer_results}
    
    # ===== Part 2: 骨干子空间语义解码 =====
    log_time("Part 2: Backbone decode...")
    backbone_results = {}
    
    for li in target_layers:
        all_acts = []
        for pair_key, pair_data in all_activations.items():
            for wk in ["a", "b"]:
                acts = pair_data[wk].get(li, [])
                all_acts.extend(acts)
        
        if len(all_acts) < 10:
            continue
        
        X = np.array(all_acts)
        mean_all = X.mean(axis=0)
        X_c = X - mean_all
        cov = X_c.T @ X_c / len(X_c)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        n_backbone = min(20, len(eigvals))
        backbone_basis = eigvecs[:, :n_backbone]
        total_var = eigvals.sum()
        backbone_var_ratio = float(eigvals[:n_backbone].sum() / total_var)
        
        # 解码骨干方向
        backbone_decoded = []
        for d in range(min(5, n_backbone)):
            top_words = decode_direction(backbone_basis[:, d], W_U, tokenizer, top_k=30)
            backbone_decoded.append({
                "direction": d, "eigenvalue": float(eigvals[d]),
                "var_explained": float(eigvals[d] / total_var),
                "top_words": top_words[:10],
            })
        
        # 解码特异方向
        specific_decoded = []
        for d in range(n_backbone, min(n_backbone + 5, len(eigvals))):
            top_words = decode_direction(eigvecs[:, d], W_U, tokenizer, top_k=30)
            specific_decoded.append({
                "direction": d, "eigenvalue": float(eigvals[d]),
                "var_explained": float(eigvals[d] / total_var),
                "top_words": top_words[:10],
            })
        
        # 神经元归属
        proj = X_c @ backbone_basis
        recon = proj @ backbone_basis.T
        res = X_c - recon
        total_var_neuron = np.var(X_c, axis=0)
        shared_var_neuron = np.var(recon, axis=0)
        backbone_score = shared_var_neuron / (total_var_neuron + 1e-10)
        
        top_bn = np.argsort(backbone_score)[::-1][:20].tolist()
        top_sn = np.argsort(backbone_score)[:20].tolist()
        
        # 解码骨干和特异神经元
        bn_decoded = []
        for ni in top_bn[:5]:
            direction = np.zeros(d_model)
            direction[ni] = 1.0
            tw = decode_direction(direction, W_U, tokenizer, top_k=5)
            bn_decoded.append({"neuron": ni, "score": float(backbone_score[ni]),
                              "top_words": tw[:3]})
        
        sn_decoded = []
        for ni in top_sn[:5]:
            direction = np.zeros(d_model)
            direction[ni] = 1.0
            tw = decode_direction(direction, W_U, tokenizer, top_k=5)
            sn_decoded.append({"neuron": ni, "score": float(backbone_score[ni]),
                              "top_words": tw[:3]})
        
        # 骨干 vs 特异方向解码质量
        # 骨干方向是否解码为更通用/抽象的词?
        # 特异方向是否解码为更具体/概念的词?
        
        backbone_results[str(li)] = {
            "backbone_var_ratio": backbone_var_ratio,
            "n_samples": len(all_acts),
            "backbone_decoded": backbone_decoded,
            "specific_decoded": specific_decoded,
            "neuron_attribution": {
                "mean_backbone_score": float(backbone_score.mean()),
                "median_backbone_score": float(np.median(backbone_score)),
                "top_backbone_neurons": top_bn,
                "top_specific_neurons": top_sn,
                "backbone_neuron_decoded": bn_decoded,
                "specific_neuron_decoded": sn_decoded,
            },
            "eigenvalue_spectrum": [float(e) for e in eigvals[:30]],
        }
        
        log_time(f"  L{li}: backbone_var={backbone_var_ratio:.3f} n={len(all_acts)} "
                 f"mean_score={backbone_score.mean():.3f}")
    
    # ===== Part 3: shared_ratio = f(distance) =====
    mid_layer = target_layers[len(target_layers) // 2]
    log_time(f"Part 3: Similarity function (mid_layer={mid_layer})")
    
    sim_data = []
    for pk, pr in all_pair_results.items():
        ld = pr["layers"].get(str(mid_layer), {})
        if "error" not in ld:
            sim_data.append({
                "pair_key": pk, "relation": pr["pair_info"]["relation"],
                "distance": pr["pair_info"]["distance"],
                "shared_ratio": ld.get("avg_shared_ratio", 0),
                "cos_mean": ld.get("cos_mean", 0),
                "delta_unique": ld.get("delta_unique_ratio", 0),
                "overlap": ld.get("subspace_overlap", 0),
            })
    
    relation_stats = defaultdict(lambda: {"shared": [], "cos": [], "delta": [], "overlap": []})
    for sd in sim_data:
        rel = sd["relation"]
        relation_stats[rel]["shared"].append(sd["shared_ratio"])
        relation_stats[rel]["cos"].append(sd["cos_mean"])
        relation_stats[rel]["delta"].append(sd["delta_unique"])
        relation_stats[rel]["overlap"].append(sd["overlap"])
    
    relation_summary = {}
    for rel, stats in relation_stats.items():
        relation_summary[rel] = {
            "mean_shared": float(np.mean(stats["shared"])),
            "std_shared": float(np.std(stats["shared"])),
            "mean_cos": float(np.mean(stats["cos"])),
            "std_cos": float(np.std(stats["cos"])),
            "mean_delta": float(np.mean(stats["delta"])),
            "mean_overlap": float(np.mean(stats["overlap"])),
            "n": len(stats["shared"]),
        }
    
    # ===== 保存 =====
    output = {
        "model": model_name, "n_dims": n_dims, "d_model": d_model,
        "n_layers": n_layers, "target_layers": target_layers,
        "pair_results": all_pair_results,
        "backbone_decode": backbone_results,
        "similarity_function": {
            "mid_layer": mid_layer,
            "per_pair": sim_data,
            "relation_summary": relation_summary,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    out_dir = PROJECT / "results" / "subspace_topology"
    out_file = out_dir / f"exp4b_backbone_decode_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Saved to {out_file}")
    
    # ===== 摘要 =====
    log_time("")
    log_time("=" * 60)
    log_time(f"PHASE 58b SUMMARY - {model_name}")
    log_time("=" * 60)
    
    log_time("")
    log_time(f"--- Shared Ratio by Relation (L{mid_layer}) ---")
    for rel in ["hyponym", "synonym", "antonym", "associated", "unrelated"]:
        if rel in relation_summary:
            rs = relation_summary[rel]
            log_time(f"  {rel:12s}: shared={rs['mean_shared']:.3f}+-{rs['std_shared']:.3f} "
                     f"cos={rs['mean_cos']:.3f} delta={rs['mean_delta']:.3f} "
                     f"overlap={rs['mean_overlap']:.3f} n={rs['n']}")
    
    log_time("")
    log_time("--- Per-Pair Detail ---")
    for sd in sorted(sim_data, key=lambda x: (x['distance'], x['pair_key'])):
        log_time(f"  {sd['pair_key']:20s} dist={sd['distance']} "
                 f"shared={sd['shared_ratio']:.3f} cos={sd['cos_mean']:.3f} "
                 f"delta={sd['delta_unique']:.3f}")
    
    log_time("")
    log_time("--- Backbone Decode (key layers) ---")
    for lk in sorted(backbone_results.keys(), key=int):
        bd = backbone_results[lk]
        log_time(f"  L{lk}: var={bd['backbone_var_ratio']:.3f}")
        for d_info in bd["backbone_decoded"][:3]:
            tw = [t['token'].strip()[:15] for t in d_info['top_words'][:5]]
            log_time(f"    PC{d_info['direction']}: var={d_info['var_explained']:.4f} top={tw}")
        for d_info in bd.get("specific_decoded", [])[:2]:
            tw = [t['token'].strip()[:15] for t in d_info['top_words'][:5]]
            log_time(f"    Spec{d_info['direction']}: var={d_info['var_explained']:.4f} top={tw}")
    
    log_time("")
    log_time("--- Neuron Attribution ---")
    for lk in sorted(backbone_results.keys(), key=int):
        na = backbone_results[lk]["neuron_attribution"]
        log_time(f"  L{lk}: mean_score={na['mean_backbone_score']:.3f} "
                 f"top_bn={na['top_backbone_neurons'][:5]}")
    
    release_model(model)
    log_time("Done!")


if __name__ == "__main__":
    main()
