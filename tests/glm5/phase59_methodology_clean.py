"""
Phase 59: 方法论清洁化 + 单轴多概念定位
==========================================
核心目标:
  1. 去除模板主成分, 获取纯语义子空间
  2. 单语义轴多概念定位 (温度轴/大小轴/情感轴)
  3. n_dims稳定性分析

Phase 58c发现: overlap(synonym) > overlap(antonym) > overlap(hyponym) > overlap(associated) >> overlap(unrelated)
但模板PC0方差占>98%, 需要验证去模板后结论是否仍然成立

方法:
  Part A: 去模板主成分
    - 用100+模板/词收集大量激活
    - 提取所有激活的PC0(模板结构方向)
    - 减去PC0后在残差中提取语义子空间
    - 重新计算overlap, 验证排序不变

  Part B: 单轴多概念定位
    - 温度轴: hot/warm/cool/cold/freezing
    - 大小轴: tiny/small/big/large/huge
    - 情感轴: love/like/neutral/dislike/hate
    - 5概念×3轴=15词, 两两overlap矩阵
    - 验证: 概念在轴上是否形成线性/环形拓扑

  Part C: n_dims稳定性
    - n_dims=5,10,15,20的overlap稳定性
"""

import sys, os, json, argparse, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import load_model, get_model_info, get_W_U, release_model, safe_decode
import torch

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)

# =====================================================================
# Part A: 去模板主成分
# =====================================================================

# 大量模板: 每个词50+模板, 语法结构多样化
# 关键: 同一概念的不同模板之间, PC0=模板结构, PC1+=语义内容
PART_A_WORDS = {
    "apple": [
        "I ate a fresh apple this morning", "The apple was sweet and juicy",
        "She picked a red apple from the tree", "He bought three apples at the market",
        "The apple pie smells amazing", "She sliced the apple for the salad",
        "A rotten apple was in the basket", "The apple tree is blooming",
        "This apple is organic", "The apple juice was refreshing",
        "She offered me an apple", "The apple tasted sour",
        "I found a wild apple in the forest", "The apple season has begun",
        "The apple cider is warm", "An apple a day keeps the doctor away",
        "The green apple fell from the branch", "He peeled the apple carefully",
        "The apple orchard was beautiful in fall", "She baked an apple cake",
        "That apple is too ripe", "The apple blossom is white",
        "He caught the apple I threw", "The apple sauce was homemade",
        "She planted an apple seed last spring", "The apple harvest was plentiful",
        "I prefer a crisp apple", "The apple turnover was delicious",
        "The apple barrel was nearly empty", "She pressed the apples into cider",
        "The golden apple shone in the sunlight", "He bit into the apple eagerly",
        "The apple basket overflowed", "She dried apple slices for snacks",
        "The apple variety was new to me", "He pruned the apple trees in winter",
        "The apple butter tasted like cinnamon", "She carved an apple for decoration",
        "The apple jam was sweet and tangy", "He picked the largest apple",
        "The apple grove was peaceful", "She wrapped the apple in paper",
        "The frozen apple was still good", "He sorted the apples by size",
        "The apple crisp was baking in the oven", "She stored apples in the cellar",
        "The apple branch swayed in the wind", "He delivered apples to the market",
        "The apple muffins smelled wonderful", "She garnished the plate with apple slices",
    ],
    "fruit": [
        "I ate fresh fruit this morning", "The fruit was sweet and juicy",
        "She picked some fruit from the tree", "He bought fresh fruit at the market",
        "The fruit salad smells amazing", "She sliced the fruit for the salad",
        "A rotten fruit was in the basket", "The fruit trees are blooming",
        "This fruit is organic", "The fruit juice was refreshing",
        "She offered me some fruit", "The fruit tasted sour",
        "I found wild fruit in the forest", "The fruit season has begun",
        "The fruit punch is warm", "Eating fruit daily is healthy",
        "The green fruit fell from the branch", "He peeled the fruit carefully",
        "The fruit orchard was beautiful in fall", "She baked a fruit cake",
        "That fruit is too ripe", "The fruit blossom is white",
        "He caught the fruit I tossed", "The fruit sauce was homemade",
        "She planted a fruit tree last spring", "The fruit harvest was plentiful",
        "I prefer crisp fruit", "The fruit turnover was delicious",
        "The fruit basket was nearly empty", "She pressed the fruit into juice",
        "The golden fruit shone in the sunlight", "He bit into the fruit eagerly",
        "The fruit basket overflowed", "She dried fruit slices for snacks",
        "The fruit variety was new to me", "He pruned the fruit trees in winter",
        "The fruit preserve tasted like honey", "She carved fruit for decoration",
        "The fruit jam was sweet and tangy", "He picked the largest fruit",
        "The fruit grove was peaceful", "She wrapped the fruit in paper",
        "The frozen fruit was still good", "He sorted the fruit by size",
        "The fruit crisp was baking in the oven", "She stored fruit in the cellar",
        "The fruit branch swayed in the wind", "He delivered fruit to the market",
        "The fruit muffins smelled wonderful", "She garnished the plate with fruit slices",
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
        "His big idea changed everything", "The big reveal surprised everyone",
        "She faced a big challenge bravely", "The big story broke at midnight",
        "A big project was announced today", "He showed big improvement this semester",
        "The big opportunity cannot be missed", "She made a big discovery",
        "The big event attracted thousands", "A big smile appeared on her face",
        "He invested a big amount of money", "The big door opened slowly",
        "She has a big heart", "The big fish got away",
        "A big building dominated the skyline", "He made a big promise",
        "The big window overlooked the valley", "She told a big lie",
        "A big crowd cheered loudly", "The big dog barked fiercely",
        "The big exam is tomorrow", "He ate a big breakfast",
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
        "His large idea changed everything", "The large reveal surprised everyone",
        "She faced a large challenge bravely", "The large story broke at midnight",
        "A large project was announced today", "He showed large improvement this semester",
        "The large opportunity cannot be missed", "She made a large discovery",
        "The large event attracted thousands", "A large smile appeared on her face",
        "He invested a large amount of money", "The large door opened slowly",
        "She has a large heart", "The large fish got away",
        "A large building dominated the skyline", "He made a large promise",
        "The large window overlooked the valley", "She told a large lie",
        "A large crowd cheered loudly", "The large dog barked fiercely",
        "The large exam is tomorrow", "He ate a large breakfast",
    ],
    "hot": [
        "The hot coffee burned my tongue", "She turned on the hot water",
        "He enjoys hot weather in summer", "The hot stove was dangerous",
        "This hot sauce is very spicy", "The hot air balloon rose quickly",
        "She took a hot shower this morning", "The hot sun beat down on us",
        "He prefers hot tea over iced", "The hot pavement burned bare feet",
        "A hot breeze blew through the window", "The hot springs were relaxing",
        "She ordered a hot meal at the restaurant", "The hot chocolate was perfect for winter",
        "He touched the hot iron accidentally", "The hot debate continued for hours",
        "This hot topic is trending online", "She felt hot and bothered",
        "The hot rod raced down the street", "A hot flash overwhelmed her suddenly",
        "The hot day made everyone tired", "He sold the hot item quickly",
        "The hot ticket sold out in minutes", "She followed the hot lead",
        "The hot pan sizzled with oil", "A hot spot was discovered on the map",
        "He dropped the hot potato immediately", "The hot line rang constantly",
        "She avoided the hot zone", "The hot trend faded quickly",
        "He served the hot dish promptly", "The hot tip paid off",
        "She wiped her hot forehead", "The hot fire crackled loudly",
        "A hot wind howled through the canyon", "The hot bath was soothing",
        "He opened the hot oven carefully", "The hot sand burned their feet",
        "She wore hot pink to the party", "The hot pepper made him cry",
        "A hot summer night is memorable", "The hot exhaust filled the garage",
        "He tasted the hot soup cautiously", "The hot iron left a mark",
        "She fanned her hot face", "The hot engine overheated",
        "The hot coal glowed in the dark", "He avoided the hot issue",
        "The hot vapor rose from the pot", "She drank the hot milk slowly",
        "The hot climate was hard to adapt to", "He touched the hot surface",
    ],
    "cold": [
        "The cold wind chilled my bones", "She turned on the cold water",
        "He hates cold weather in winter", "The cold ice was slippery",
        "This cold drink is very refreshing", "The cold front moved in quickly",
        "She took a cold shower this morning", "The cold moon shone on us",
        "He prefers cold juice over hot", "The cold ground froze bare feet",
        "A cold breeze blew through the window", "The cold winter was harsh",
        "She ordered a cold meal at the deli", "The cold lemonade was perfect for summer",
        "He touched the cold metal carefully", "The cold war lasted for decades",
        "This cold case was finally solved", "She felt cold and shivering",
        "The cold storage preserved the food", "A cold sweat broke out suddenly",
        "The cold day made everyone bundle up", "He caught a cold last week",
        "The cold facts were undeniable", "She gave a cold stare",
        "The cold rain soaked through", "A cold spot was found in the data",
        "He left her out in the cold", "The cold reality set in",
        "She avoided the cold shoulder", "The cold spell lasted weeks",
        "He served the cold dish promptly", "The cold shoulder hurt her feelings",
        "She wrapped up in the cold night", "The cold fire provided no warmth",
        "A cold wind howled through the canyon", "The cold bath was invigorating",
        "He opened the cold refrigerator", "The cold snow covered their feet",
        "She wore a cold expression", "The cold pepper was actually mild",
        "A cold winter night is brutal", "The cold air filled the room",
        "He tasted the cold soup reluctantly", "The cold steel felt heavy",
        "She rubbed her cold hands", "The cold engine would not start",
        "The cold marble was smooth", "He avoided the cold truth",
        "The cold vapor condensed on the glass", "She drank the cold water quickly",
        "The cold climate was difficult to endure", "He touched the cold surface",
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
        "The love triangle created drama", "She believed in true love",
        "He earned the love of his students", "The love letter was decades old",
        "She proclaimed her love publicly", "The love nest was cozy and warm",
        "He returned her love with kindness", "A love child was born that year",
        "She lost the love of her life", "The lovebirds sat together on the bench",
        "He embraced love wholeheartedly", "The love feast brought everyone together",
        "She nurtured love like a garden", "He declared his love passionately",
        "The love match surprised their friends", "She doubted love could last forever",
        "He valued love above all else", "The love offering was generous",
        "She feared love would hurt her", "The love test was difficult to pass",
        "He gave love freely to everyone", "The love god smiled upon them",
        "She rejected his love politely", "The love promise was kept for years",
        "He yearned for love and companionship", "The love of nature was evident",
        "She sacrificed love for her career", "The love bond was unbreakable",
        "He cherished every love moment", "The love scene in the movie was moving",
        "She whispered love words softly", "The love relationship grew stronger",
        "He found love again after loss", "The love gift was handmade",
    ],
    "hate": [
        "She hates injustice with a passion", "He developed hate over the years",
        "The hate speech was condemned", "She expressed hate through protests",
        "He wrote about hate in his diary", "The hate between them was palpable",
        "She encountered hate in online forums", "Hate can consume a person entirely",
        "He discovered his hate for corruption", "The hate of violence drives activists",
        "She felt hate radiating from the crowd", "He showed hate through aggression",
        "The hate group was monitored", "She received hate from strangers",
        "His hate for lying never faded", "The hate crime was prosecuted",
        "She witnessed hate firsthand", "He pursued hate relentlessly online",
        "The hate spiral created conflict", "She believed hate could be overcome",
        "He recognized the hate in their words", "The hate mail was disturbing",
        "She protested against hate publicly", "The hate rally was controversial",
        "He responded to hate with kindness", "A hate incident was reported that day",
        "She struggled with hate in her heart", "The haters gathered on the platform",
        "He rejected hate wholeheartedly", "The hate epidemic spread online",
        "She fought hate like a battle", "He denounced hate passionately",
        "The hate crime shocked the community", "She doubted hate could ever end",
        "He studied hate academically", "The hate symbol was removed",
        "She feared hate would destroy society", "The hate propaganda was banned",
        "He gave examples of hate throughout history", "The hate preacher was arrested",
        "She rejected his hate politely", "The hate crime law was strengthened",
        "He recognized hate as a destructive force", "The hate of oppression was universal",
        "She channeled hate into activism", "The hate cycle seemed unbreakable",
        "He avoided hate at all costs", "The hate incident made the news",
        "She spoke against hate boldly", "The hate relationship was toxic",
        "He found hate in unexpected places", "The hate message was deleted",
    ],
    "doctor": [
        "The doctor examined the patient carefully", "She visited the doctor last week",
        "He wants to become a doctor someday", "The doctor prescribed medication",
        "She called the doctor about her symptoms", "The doctor performed the surgery",
        "He trusted his doctor completely", "The doctor gave helpful advice",
        "She thanked the doctor for the diagnosis", "The doctor ordered more tests",
        "He followed the doctor recommendations", "The doctor specializes in cardiology",
        "She waited for the doctor in the lobby", "The doctor answered all her questions",
        "His doctor said he is recovering well", "The doctor clinic was very busy",
        "She respected the doctor opinion", "The doctor reassured the worried family",
        "He chose a new doctor this year", "The doctor reputation was excellent",
        "She liked her doctor bedside manner", "The doctor explained the procedure clearly",
        "He paid the doctor bill reluctantly", "The doctor worked long hours",
        "She admired the doctor dedication", "The doctor training took many years",
        "He received a doctor referral", "The doctor practiced at the hospital",
        "She graduated as a doctor last month", "The doctor noted the improvement",
        "He challenged the doctor diagnosis", "The doctor schedule was fully booked",
        "She emailed the doctor with concerns", "The doctor team collaborated on the case",
        "He respected the doctor expertise", "The doctor was on call that night",
        "She appreciated the doctor honesty", "The doctor bedside manner was gentle",
        "He sought a second doctor opinion", "The doctor followed up after the visit",
        "She described her pain to the doctor", "The doctor ordered an x-ray",
        "He trusted the doctor judgment", "The doctor suggested a lifestyle change",
        "She rated her doctor highly online", "The doctor graduated from a top school",
        "He introduced the doctor to the family", "The doctor published a research paper",
        "She preferred a female doctor", "The doctor was known for his patience",
        "He credited the doctor for his recovery", "The doctor clinic opened early",
    ],
    "hospital": [
        "The hospital treated many patients today", "She visited the hospital last week",
        "He works at the hospital downtown", "The hospital provided excellent care",
        "She called the hospital about her test results", "The hospital performed the surgery",
        "He was admitted to the hospital", "The hospital staff was very helpful",
        "She thanked the hospital for the treatment", "The hospital ordered more equipment",
        "He followed the hospital discharge instructions", "The hospital specializes in cardiology",
        "She waited in the hospital lobby", "The hospital answered all her questions",
        "His hospital stay was short", "The hospital cafeteria was busy",
        "She respected the hospital reputation", "The hospital reassured the worried family",
        "He chose a new hospital this year", "The hospital reputation was excellent",
        "She liked the hospital cleanliness", "The hospital explained the billing clearly",
        "He paid the hospital bill reluctantly", "The hospital operated around the clock",
        "She admired the hospital dedication", "The hospital construction took many years",
        "He received a hospital referral", "The hospital served the community well",
        "She was born in that hospital", "The hospital noted the improvement",
        "He challenged the hospital charges", "The hospital parking lot was full",
        "She emailed the hospital with concerns", "The hospital team collaborated on the case",
        "He respected the hospital policies", "The hospital was on lockdown that night",
        "She appreciated the hospital efficiency", "The hospital wing was newly renovated",
        "He sought a second hospital opinion", "The hospital followed up after the discharge",
        "She described her experience at the hospital", "The hospital installed a new MRI",
        "He trusted the hospital staff", "The hospital suggested a follow-up visit",
        "She rated the hospital highly online", "The hospital partnered with the university",
        "He drove past the hospital daily", "The hospital published an annual report",
        "She preferred a private hospital room", "The hospital was known for its research",
        "He credited the hospital for his recovery", "The hospital emergency room was packed",
    ],
    "planet": [
        "The planet orbits around its star", "She studied the planet in astronomy class",
        "He discovered a new planet last year", "The planet atmosphere is toxic",
        "She observed the planet through a telescope", "The planet has two moons",
        "He named the planet after his daughter", "The planet surface is covered in ice",
        "She believes life exists on another planet", "The planet rotation takes 24 hours",
        "He wrote about the planet in his journal", "The planet gravity is much stronger",
        "She explored the planet virtually", "The planet temperature is extreme",
        "His favorite planet is Saturn", "The planet system was recently mapped",
        "She photographed the planet from space", "The planet rings are spectacular",
        "He calculated the planet mass accurately", "The planet distance from the sun varies",
        "She modeled the planet orbit on computer", "The planet climate is changing",
        "He landed on the planet in the simulation", "The planet diameter is enormous",
        "She compared the planet to Earth", "The planet surface pressure is high",
        "He tracked the planet movement nightly", "The planet magnetic field is unusual",
        "She described the planet in her thesis", "The planet has a thick atmosphere",
        "He observed the planet transit across the star", "The planet interior is molten",
        "She classified the planet as a gas giant", "The planet year is very long",
        "He found evidence of water on the planet", "The planet tilt causes seasons",
        "She mapped the planet terrain carefully", "The planet core generates heat",
        "He predicted the planet existence mathematically", "The planet orbit is elliptical",
        "She visited the planet in a dream", "The planet ecosystem is unique",
        "He researched the planet formation process", "The planet wind speeds are incredible",
        "She sketched the planet landscape", "The planet soil contains minerals",
        "He sent a probe to the planet", "The planet sky appears purple",
        "She imagined living on the planet", "The planet has active volcanoes",
        "He measured the planet density precisely", "The planet was formed billions of years ago",
    ],
}

# Part B: 单轴多概念
AXIS_CONCEPTS = {
    "temperature": ["hot", "warm", "cool", "cold", "freezing"],
    "size": ["tiny", "small", "big", "large", "huge"],
    "sentiment": ["love", "like", "neutral", "dislike", "hate"],
}

# 为Part B新增的词补充模板
EXTRA_WORD_TEMPLATES = {
    "warm": [
        "The warm sun felt good on my skin", "She wore a warm sweater today",
        "He gave a warm welcome to the guests", "The warm water was comfortable",
        "A warm breeze blew through the garden", "The warm milk helped her sleep",
        "She sent warm greetings from abroad", "The warm blanket kept him cozy",
        "He enjoyed the warm weather this weekend", "A warm smile crossed her face",
        "The warm colors painted the sunset", "She made a warm fire in the hearth",
        "He felt a warm glow of satisfaction", "The warm bread smelled delicious",
        "She appreciated his warm encouragement", "The warm bath relaxed her muscles",
        "He gave a warm hug to his mother", "The warm soup was comforting",
        "She remembered the warm summer days", "A warm feeling spread through her chest",
        "He offered a warm hand to the stranger", "The warm air rose from the pavement",
        "She described the warm reception positively", "The warm light filled the room",
        "He preferred warm climates over cold ones", "The warm toast was buttered perfectly",
        "She enjoyed the warm hospitality of the hosts", "A warm current flowed through the ocean",
        "He created a warm atmosphere at the party", "The warm muffins were fresh from the oven",
        "She valued the warm friendship they shared", "The warm evening was perfect for a walk",
        "He cooked a warm meal for the family", "The warm sand felt soft between her toes",
        "She wrapped up in a warm shawl", "The warm tea soothed her throat",
        "He appreciated the warm support from colleagues", "The warm sunshine brightened the day",
        "She baked warm cookies for the children", "The warm embrace felt like home",
        "He spoke with warm enthusiasm about the project", "The warm glow of the fireplace was inviting",
        "She enjoyed the warm spring afternoon", "A warm response surprised him",
        "The warm towel was refreshing after the swim", "He maintained a warm relationship with neighbors",
        "She prepared a warm drink for the guest", "The warm coat protected against the chill",
        "He gave a warm recommendation for the book", "The warm rain was gentle and soothing",
    ],
    "cool": [
        "The cool breeze refreshed the runners", "She wore a cool outfit to the party",
        "He remained cool under pressure", "The cool water was refreshing after hiking",
        "A cool evening followed the hot day", "The cool drink hit the spot",
        "She gave a cool response to the proposal", "The cool air conditioning felt wonderful",
        "He played it cool during the interview", "A cool demeanor impressed the judges",
        "The cool shade provided relief", "She kept her cool in the argument",
        "He bought cool new sneakers", "The cool night air was crisp",
        "A cool attitude can be off-putting", "The cool fabric was comfortable in summer",
        "She admired his cool confidence", "The cool metal felt smooth",
        "He gave a cool nod of approval", "The cool mist covered the valley",
        "She found a cool spot under the tree", "The cool tempo of the music was relaxing",
        "He maintained a cool distance from the drama", "The cool color palette was calming",
        "She preferred cool tones in her artwork", "The cool river flowed gently downstream",
        "He appreciated the cool efficiency of the system", "A cool wind blew from the north",
        "The cool reception surprised the visitors", "She enjoyed the cool morning jog",
        "He gave a cool handshake to the client", "The cool ice pack reduced the swelling",
        "She noticed his cool detachment from the group", "The cool autumn air was invigorating",
        "He offered a cool analysis of the situation", "The cool surface of the lake reflected the sky",
        "She kept a cool head during the crisis", "The cool whistle of the train echoed",
        "He delivered a cool performance on stage", "The cool fog rolled in from the sea",
        "She savored the cool mint flavor", "A cool rain began to fall",
        "He found a cool solution to the problem", "The cool stone floor felt nice",
        "She appreciated the calm cool voice of the instructor", "The cool evening called for a light jacket",
        "He walked with a cool steady stride", "The cool design attracted many customers",
        "She felt cool relief from the pain medication", "The cool underground cave was damp",
        "He made a cool calculation about the risk", "The cool jazz played softly in the background",
    ],
    "freezing": [
        "The freezing wind cut through my coat", "She stepped into the freezing rain",
        "He was freezing despite the thick jacket", "The freezing temperature broke records",
        "A freezing fog covered the landscape", "The freezing water turned to ice instantly",
        "She was freezing after falling in the lake", "The freezing point of water is zero degrees",
        "He felt freezing cold in the unheated room", "A freezing draft came through the window",
        "The freezing conditions made travel impossible", "She shivered in the freezing night air",
        "He described the freezing Arctic expedition", "The freezing rain created dangerous roads",
        "A freezing gust nearly knocked her over", "The freezing pipes burst in the winter",
        "She was freezing and could not stop shaking", "The freezing spray stung their faces",
        "He built a shelter from the freezing wind", "A freezing chill ran down her spine",
        "The freezing weather lasted for weeks", "She survived the freezing night outdoors",
        "He warned about the freezing conditions ahead", "The freezing fog reduced visibility to zero",
        "A freezing cold front moved in from the north", "The freezing lake was solid enough to skate on",
        "She touched the freezing metal railing", "The freezing rain turned to sleet",
        "He returned home freezing and exhausted", "A freezing breeze carried the scent of snow",
        "The freezing temperatures damaged the crops", "She pulled her scarf tight against the freezing air",
        "He could see his breath in the freezing cold", "The freezing snowfall continued throughout the day",
        "She wore multiple layers to avoid freezing", "A freezing night awaited the stranded travelers",
        "The freezing wind chill made it feel even colder", "He rubbed his freezing hands together",
        "The freezing water pipes needed insulation", "She complained about the freezing office temperature",
        "A freezing storm was forecast for the weekend", "The freezing cold snap surprised everyone",
        "He remembered the freezing winters of his childhood", "The freezing ground was too hard to dig",
        "She could not feel her freezing fingers", "A freezing mist hung over the river",
        "The freezing cold penetrated every layer", "He stomped his freezing feet to warm them",
        "The freezing rain coated everything in ice", "She prepared for the freezing journey ahead",
    ],
    "tiny": [
        "The tiny insect crawled across the leaf", "She noticed a tiny crack in the wall",
        "He lives in a tiny apartment downtown", "The tiny kitten fit in her palm",
        "A tiny detail made all the difference", "The tiny seeds grew into tall trees",
        "She wore tiny earrings to the event", "The tiny village was hidden in the mountains",
        "He found a tiny fossil on the beach", "A tiny fraction of the budget was used",
        "The tiny button was hard to fasten", "She made tiny adjustments to the recipe",
        "He spotted a tiny bird on the branch", "The tiny room felt claustrophobic",
        "A tiny mistake cost the company millions", "The tiny flowers dotted the meadow",
        "She gave a tiny nod of agreement", "The tiny crack widened over time",
        "He wrote in tiny handwriting", "A tiny light appeared in the distance",
        "The tiny footprint was barely visible", "She packed a tiny suitcase for the trip",
        "He offered a tiny suggestion for improvement", "The tiny droplet fell from the faucet",
        "A tiny minority opposed the plan", "The tiny boat rocked in the waves",
        "She added a tiny pinch of salt", "The tiny bell chimed softly",
        "He noticed a tiny change in her behavior", "A tiny speck of dust floated in the air",
        "The tiny shop was easy to miss", "She heard a tiny sound from the closet",
        "He repaired the tiny circuit carefully", "A tiny window let in minimal light",
        "The tiny population of the island surprised him", "She folded the paper into a tiny square",
        "He used a tiny brush for the details", "A tiny ripple disturbed the still water",
        "The tiny gap between the boards was sealed", "She made a tiny cut in the fabric",
        "He selected the tiny option from the menu", "A tiny flame flickered in the darkness",
        "The tiny mark on the painting was valuable", "She took a tiny step forward",
        "He built a tiny model of the city", "A tiny stream wound through the forest",
        "The tiny creature scurried under the rock", "She noticed a tiny improvement in her health",
        "He found a tiny gem in the rough stone", "A tiny smile appeared on her face",
    ],
    "small": [
        "The small dog barked at the mailman", "She ordered a small coffee this morning",
        "He lives in a small town in the Midwest", "The small package arrived yesterday",
        "A small mistake was easily corrected", "The small business thrived despite competition",
        "She chose the small size for her order", "The small crowd gathered at the park",
        "He made a small contribution to the cause", "A small change led to big results",
        "The small child reached for the cookie", "She noticed a small dent in the car",
        "He found a small coin on the ground", "The small window faced the garden",
        "A small amount of rain fell overnight", "The small room was cozy and warm",
        "She gave a small wave from across the street", "The small boat sailed across the lake",
        "He saved a small portion for later", "A small bird landed on the fence",
        "The small crack in the glass was repaired", "She paid a small fee for the service",
        "He took a small bite of the cake", "A small percentage passed the exam",
        "The small group discussed the issue quietly", "She made a small adjustment to the settings",
        "He found a small flower growing in the cracks", "A small stream flowed behind the house",
        "The small kitten curled up in her lap", "She wore a small ring on her finger",
        "He earned a small profit from the sale", "A small cloud drifted across the sky",
        "The small clock ticked quietly on the shelf", "She added a small amount of pepper",
        "He noticed a small error in the report", "A small animal scurried across the path",
        "The small plate held just enough food", "She spoke in a small voice",
        "He built a small fire in the fireplace", "A small piece was missing from the puzzle",
        "The small town had one main street", "She made a small gesture of thanks",
        "He caught a small fish in the pond", "A small gap remained in the fence",
        "The small garden produced fresh vegetables", "She kept a small notebook in her bag",
        "He carried a small backpack on the hike", "A small star twinkled in the night sky",
        "The small cake was decorated beautifully", "She showed a small sign of improvement",
    ],
    "huge": [
        "The huge mountain dominated the landscape", "She made a huge discovery last year",
        "He faced a huge challenge at work", "The huge elephant walked slowly",
        "A huge crowd attended the concert", "The huge building blocked the sunlight",
        "She received a huge bonus this quarter", "The huge storm caused widespread damage",
        "He noticed a huge difference immediately", "A huge gap existed between the two groups",
        "The huge tree was hundreds of years old", "She invested a huge sum of money",
        "He scored a huge victory in the election", "A huge rock blocked the road",
        "The huge ship sailed into the harbor", "She took a huge risk with her career",
        "He made a huge mistake that cost dearly", "A huge amount of data was collected",
        "The huge cake fed fifty people", "She noticed a huge improvement in his health",
        "He ordered a huge portion of food", "A huge effort was required to finish the project",
        "The huge balloon floated above the parade", "She found a huge fossil at the dig site",
        "He carried the huge box upstairs", "A huge wave crashed against the shore",
        "The huge project took three years to complete", "She made a huge impact on the community",
        "He built a huge sandcastle on the beach", "A huge explosion was heard miles away",
        "The huge stadium held eighty thousand fans", "She signed a huge contract with the company",
        "He enjoyed huge success in his career", "A huge shadow fell across the field",
        "The huge door opened with a creak", "She saw a huge bear in the forest",
        "He lifted the huge weight with effort", "A huge crowd cheered the winning goal",
        "The huge dinosaur skeleton filled the museum hall", "She overcame a huge obstacle in her path",
        "He discovered a huge cave in the mountainside", "A huge collection of books lined the walls",
        "The huge snowdrift blocked the driveway", "She completed a huge marathon last weekend",
        "He heard a huge crash from the kitchen", "A huge spider hung from the ceiling",
        "The huge whale surfaced near the boat", "She planned a huge celebration for the anniversary",
        "He repaired the huge machine in the factory", "A huge opportunity awaited the brave",
    ],
    "like": [
        "I really like this song on the radio", "She likes to read before bed",
        "He would like a cup of tea please", "They like going to the beach in summer",
        "Do you like the new design", "She acted like nothing had happened",
        "He looks like his father", "It sounds like a good plan to me",
        "She feels like taking a walk", "Like most people I enjoy weekends",
        "He ran like the wind", "She smiled like she knew a secret",
        "Things like this happen all the time", "He eats like a horse",
        "She danced like nobody was watching", "Like I said the deadline is Friday",
        "He slept like a log last night", "She sings like an angel",
        "There is nothing like homemade bread", "He worked like a dog all week",
        "She felt like she belonged here", "Like always he arrived late",
        "He drives like a maniac sometimes", "She laughed like it was the funniest joke",
        "The weather felt like spring today", "He fought like a champion",
        "She cooked like a professional chef", "Like everyone else she was nervous",
        "He studied like his life depended on it", "She explained like a patient teacher",
        "It tastes like cinnamon and sugar", "He painted like a true artist",
        "She dressed like a movie star", "Like many students he struggled with math",
        "He swam like a fish in the water", "The fabric feels like silk",
        "She organized like a natural leader", "Like before the results were mixed",
        "He climbed like a mountain goat", "She wrote like a seasoned author",
        "It smells like fresh rain outside", "He performed like a veteran actor",
        "She decorated like an interior designer", "Like the others he was curious",
        "He solved it like a puzzle master", "She taught like she truly cared",
        "It looks like it might rain soon", "He spoke like an expert on the topic",
        "She handled it like a professional", "Like I mentioned the project is on track",
    ],
    "neutral": [
        "She maintained a neutral expression during the meeting", "The referee stayed neutral throughout the game",
        "He took a neutral stance on the issue", "The neutral color worked well in the room",
        "A neutral tone is best for formal writing", "She preferred neutral flavors in her food",
        "The country remained neutral during the conflict", "He gave a neutral response to the question",
        "The neutral territory was safe for negotiations", "A neutral observer would see both sides",
        "She dressed in neutral tones for the interview", "The neutral position attracted moderate voters",
        "He kept his opinion neutral and objective", "The neutral zone separated the two armies",
        "A neutral third party mediated the dispute", "She adopted a neutral posture during the debate",
        "The neutral soil was ideal for the experiment", "He remained neutral despite the pressure to choose",
        "The neutral flag flew over the embassy", "A neutral background made the painting stand out",
        "She spoke in a neutral voice to calm everyone", "The neutral wire completed the circuit safely",
        "He liked the neutral design of the product", "The neutral review pointed out pros and cons",
        "A neutral atmosphere was maintained at the summit", "She found the book interesting but neutral",
        "He maintained a neutral balance between work and life", "The neutral statement avoided controversy",
        "The neutral space was designed for everyone", "She played a neutral character in the film",
        "He kept a neutral face while listening", "The neutral palette was soothing to the eye",
        "A neutral witness described the event accurately", "She held a neutral view on the matter",
        "The neutral report presented facts without bias", "He took a neutral approach to the problem",
        "The neutral position was respected by both sides", "A neutral environment encouraged open discussion",
        "She wore neutral makeup for the natural look", "The neutral solution had no effect on pH",
        "He preferred a neutral setting for the meeting", "The neutral comment did not take sides",
        "A neutral party oversaw the election process", "She kept her social media neutral",
        "The neutral ground was agreed upon by both", "He offered a neutral perspective on the debate",
        "The neutral shade matched everything perfectly", "A neutral attitude helped resolve the conflict",
        "She kept her facial expression neutral", "The neutral territory was respected by all",
    ],
    "dislike": [
        "She strongly dislikes waking up early", "He expressed dislike for the new policy",
        "The dislike between them was obvious", "She showed dislike for spicy food",
        "He could not hide his dislike of the movie", "A strong dislike formed over time",
        "She voiced her dislike at the meeting", "The dislike was mutual between the rivals",
        "He developed a dislike for loud noises", "She felt dislike toward the unfair system",
        "His dislike of dishonesty was well known", "The dislike stemmed from a past incident",
        "She made her dislike clear from the start", "He could sense the dislike in her voice",
        "A growing dislike for the job troubled him", "She never expressed dislike openly",
        "He acknowledged his dislike of confrontation", "The dislike for the plan was widespread",
        "She harbored a secret dislike for the idea", "He struggled with dislike for his neighbor",
        "A sudden dislike for the taste surprised her", "The dislike of change is common",
        "He could not overcome his dislike of flying", "She stated her dislike politely but firmly",
        "The dislike was evident in his body language", "He overcame his dislike eventually",
        "She realized her dislike was irrational", "The dislike grew stronger over the years",
        "He admitted his dislike of public speaking", "She could not conceal her dislike of the smell",
        "A deep dislike for injustice motivated her", "He suppressed his dislike for the sake of peace",
        "She displayed dislike through crossed arms", "The dislike for the restaurant was shared",
        "He tolerated his dislike of the cold weather", "A mild dislike was all she felt",
        "She expressed dislike without being rude", "The dislike for the song was unanimous",
        "He noticed her subtle dislike of the suggestion", "She had a strong dislike for waiting",
        "The dislike between the teams was legendary", "He felt dislike but remained professional",
        "She rarely showed dislike in public", "A personal dislike colored his judgment",
        "He understood her dislike of the proposal", "The dislike was hard to miss",
        "She overcame her dislike through exposure", "He never acted on his dislike",
        "The dislike for the texture was visceral", "She accepted his dislike without question",
    ],
}


def find_target_pos_in_full(tokenizer, input_ids, target_word):
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


def collect_word_activations(model, tokenizer, device, word, templates, target_layers, n_layers):
    activations = {li: [] for li in target_layers}
    found = 0
    with torch.no_grad():
        for tmpl in templates:
            inputs = tokenizer(tmpl, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            pos, tlen = find_target_pos_in_full(tokenizer, input_ids, word)
            if pos is None or pos >= seq_len:
                continue
            actual_pos = min(pos + (tlen // 2), seq_len - 1)
            found += 1
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states
            for li in target_layers:
                activations[li].append(hidden[li + 1][0, actual_pos].detach().cpu().float().numpy())
    return activations, found


def pca_subspace(vectors, n_dims=10):
    """PCA提取子空间, 返回 (basis, eigenvalues, mean)"""
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_c = X - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n = min(n_dims, Vt.shape[0])
    eigenvalues = (S ** 2) / len(X_c)
    return Vt[:n].T, eigenvalues[:n], mean


def subspace_overlap(basis_a, basis_b):
    """子空间重叠度: tr(B^T A A^T B) / k"""
    if basis_a is None or basis_b is None:
        return 0.0
    proj = basis_b.T @ basis_a @ basis_a.T @ basis_b
    k = min(basis_a.shape[1], basis_b.shape[1])
    return float(np.trace(proj) / k)


def projection_energy_ratio(vectors, basis):
    """计算向量集在子空间中的投影能量比"""
    X = np.array(vectors)
    if len(X) == 0 or basis is None:
        return 0.0
    mean = X.mean(axis=0)
    X_c = X - mean
    proj = X_c @ basis @ basis.T
    return float(np.sum(proj ** 2) / (np.sum(X_c ** 2) + 1e-10))


def decode_direction(direction, W_U, tokenizer, top_k=20):
    logits = W_U @ direction
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [{"token": safe_decode(tokenizer, idx), "prob": float(probs[idx])}
            for idx in top_indices]


# =====================================================================
# Part A: 去模板主成分验证
# =====================================================================

def part_a_template_removal(model, tokenizer, device, info, W_U, model_name):
    """Part A: 去模板主成分, 验证overlap排序不变"""
    log_time("=" * 60)
    log_time("PART A: Template Principal Component Removal")
    log_time("=" * 60)
    
    d_model = info.d_model
    n_layers = info.n_layers
    # 选3个代表性层
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    n_dims = 10
    
    # Part A使用5对概念对
    test_pairs = [
        ("apple", "fruit", "hyponym"),
        ("big", "large", "synonym"),
        ("hot", "cold", "antonym"),
        ("doctor", "hospital", "associated"),
        ("apple", "planet", "unrelated"),
    ]
    
    # Step 1: 收集所有激活
    log_time("Step 1: Collecting activations...")
    all_word_acts = {}
    for w_a, w_b, rel in test_pairs:
        for w in [w_a, w_b]:
            if w not in all_word_acts:
                templates = PART_A_WORDS.get(w, EXTRA_WORD_TEMPLATES.get(w, []))
                if not templates:
                    log_time(f"  WARNING: No templates for {w}, skipping")
                    continue
                acts, found = collect_word_activations(model, tokenizer, device, w, templates, target_layers, n_layers)
                all_word_acts[w] = acts
                log_time(f"  {w}: {found} samples")
    
    results = {}
    
    for li in target_layers:
        layer_key = str(li)
        log_time(f"\n--- Layer {li} ---")
        
        # Step 2: 对每个词提取PCA, 计算原始overlap
        word_bases_orig = {}
        for w in all_word_acts:
            acts = all_word_acts[w].get(li, [])
            if len(acts) < 2:
                continue
            basis, eigs, mean = pca_subspace(acts, n_dims)
            word_bases_orig[w] = {"basis": basis, "eigs": eigs, "mean": mean, "acts": acts}
        
        # 原始overlap
        orig_overlaps = {}
        for w_a, w_b, rel in test_pairs:
            if w_a in word_bases_orig and w_b in word_bases_orig:
                ov = subspace_overlap(word_bases_orig[w_a]["basis"], word_bases_orig[w_b]["basis"])
                orig_overlaps[rel] = ov
                log_time(f"  Original {rel}({w_a}/{w_b}): overlap={ov:.3f}")
        
        # Step 3: 提取模板PC0 — 合并所有词的激活
        all_acts_combined = []
        for w in word_bases_orig:
            all_acts_combined.extend(word_bases_orig[w]["acts"])
        all_acts_arr = np.array(all_acts_combined)
        all_mean = all_acts_arr.mean(axis=0)
        all_centered = all_acts_arr - all_mean
        U, S, Vt = np.linalg.svd(all_centered, full_matrices=False)
        
        # 方差解释比
        var_explained = (S ** 2) / (S ** 2).sum()
        log_time(f"  PC0 var explained: {var_explained[0]:.4f} ({var_explained[0]*100:.1f}%)")
        log_time(f"  PC1 var explained: {var_explained[1]:.4f} ({var_explained[1]*100:.1f}%)")
        log_time(f"  Top5 PC cumsum: {var_explained[:5].cumsum()}")
        
        # Step 4: 去除PC0和PC0+PC1
        for n_remove in [1, 2, 3]:
            # 投影矩阵: 去除前n_remove个主成分
            remove_basis = Vt[:n_remove].T  # [d_model, n_remove]
            keep_basis = Vt[n_remove:].T    # [d_model, d_model-n_remove]
            projector = keep_basis @ keep_basis.T  # 去除前n_remove个PC
            
            word_bases_clean = {}
            for w in word_bases_orig:
                acts = np.array(word_bases_orig[w]["acts"])
                # 减去全局均值, 去除PC0, 加回全局均值
                acts_centered = acts - all_mean
                acts_cleaned = (acts_centered @ projector) + all_mean
                # 在清洗后的数据上做PCA
                basis_c, eigs_c, mean_c = pca_subspace(acts_cleaned, n_dims)
                word_bases_clean[w] = {"basis": basis_c, "eigs": eigs_c, "mean": mean_c}
            
            clean_overlaps = {}
            for w_a, w_b, rel in test_pairs:
                if w_a in word_bases_clean and w_b in word_bases_clean:
                    ov = subspace_overlap(word_bases_clean[w_a]["basis"], word_bases_clean[w_b]["basis"])
                    clean_overlaps[rel] = ov
            
            log_time(f"  After removing PC0-{n_remove-1}:")
            for rel in orig_overlaps:
                orig = orig_overlaps.get(rel, 0)
                clean = clean_overlaps.get(rel, 0)
                delta = clean - orig
                log_time(f"    {rel:12s}: orig={orig:.3f} clean={clean:.3f} delta={delta:+.3f}")
            
            results[f"L{li}_remove{n_remove}"] = {
                "orig_overlaps": {k: float(v) for k, v in orig_overlaps.items()},
                "clean_overlaps": {k: float(v) for k, v in clean_overlaps.items()},
            }
    
    return results


# =====================================================================
# Part B: 单轴多概念定位
# =====================================================================

def part_b_axis_mapping(model, tokenizer, device, info, W_U, model_name):
    """Part B: 单语义轴多概念定位"""
    log_time("=" * 60)
    log_time("PART B: Single Axis Multi-Concept Mapping")
    log_time("=" * 60)
    
    d_model = info.d_model
    n_layers = info.n_layers
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    n_dims = 10
    
    # 收集所有轴概念的激活
    all_axis_acts = {}
    for axis_name, words in AXIS_CONCEPTS.items():
        log_time(f"Axis: {axis_name}")
        for w in words:
            templates = PART_A_WORDS.get(w, EXTRA_WORD_TEMPLATES.get(w, []))
            if not templates:
                log_time(f"  WARNING: No templates for {w}")
                continue
            acts, found = collect_word_activations(model, tokenizer, device, w, templates, target_layers, n_layers)
            all_axis_acts[w] = acts
            log_time(f"  {w}: {found} samples")
    
    results = {}
    
    for li in target_layers:
        layer_key = str(li)
        log_time(f"\n--- Layer {li} ---")
        
        # 提取每个词的子空间
        word_bases = {}
        word_means = {}
        for w, acts_dict in all_axis_acts.items():
            acts = acts_dict.get(li, [])
            if len(acts) < 2:
                continue
            basis, eigs, mean = pca_subspace(acts, n_dims)
            word_bases[w] = basis
            word_means[w] = mean
        
        # 对每条轴, 计算两两overlap矩阵
        for axis_name, words in AXIS_CONCEPTS.items():
            available = [w for w in words if w in word_bases]
            if len(available) < 2:
                continue
            
            # overlap矩阵
            n = len(available)
            overlap_matrix = np.zeros((n, n))
            mean_dist_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                    else:
                        overlap_matrix[i, j] = subspace_overlap(word_bases[available[i]], word_bases[available[j]])
                    # 均值距离
                    mean_dist_matrix[i, j] = float(np.linalg.norm(word_means[available[i]] - word_means[available[j]]))
            
            log_time(f"  Axis {axis_name}: overlap matrix")
            log_time(f"    Words: {available}")
            for i in range(n):
                row = " ".join([f"{overlap_matrix[i,j]:.3f}" for j in range(n)])
                log_time(f"    {available[i]:10s}: [{row}]")
            
            # 均值距离矩阵
            log_time(f"    Mean distances:")
            for i in range(n):
                row = " ".join([f"{mean_dist_matrix[i,j]:.2f}" for j in range(n)])
                log_time(f"    {available[i]:10s}: [{row}]")
            
            # 相邻概念vs非相邻概念的overlap对比
            adj_overlaps = []
            nonadj_overlaps = []
            for i in range(n):
                for j in range(i+1, n):
                    if j == i + 1:  # 相邻
                        adj_overlaps.append(overlap_matrix[i, j])
                    else:  # 非相邻
                        nonadj_overlaps.append(overlap_matrix[i, j])
            
            if adj_overlaps and nonadj_overlaps:
                log_time(f"    Adjacent overlap: {np.mean(adj_overlaps):.3f} +- {np.std(adj_overlaps):.3f}")
                log_time(f"    Non-adjacent overlap: {np.mean(nonadj_overlaps):.3f} +- {np.std(nonadj_overlaps):.3f}")
            
            # delta方向解码 (相邻概念的差异方向)
            log_time(f"    Delta direction decoding:")
            for i in range(len(available) - 1):
                w_a = available[i]
                w_b = available[i + 1]
                delta = word_means[w_b] - word_means[w_a]
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 0:
                    delta_dir = delta / delta_norm
                    decoded = decode_direction(delta_dir, W_U, tokenizer, top_k=10)
                    tokens_str = ", ".join([f"{d['token']}({d['prob']:.3f})" for d in decoded[:5]])
                    log_time(f"      {w_a}→{w_b}: [{tokens_str}]")
            
            results[f"L{li}_{axis_name}"] = {
                "words": available,
                "overlap_matrix": overlap_matrix.tolist(),
                "mean_dist_matrix": mean_dist_matrix.tolist(),
                "adj_overlaps": [float(x) for x in adj_overlaps],
                "nonadj_overlaps": [float(x) for x in nonadj_overlaps],
            }
    
    return results


# =====================================================================
# Part C: n_dims稳定性
# =====================================================================

def part_c_ndims_stability(model, tokenizer, device, info, model_name):
    """Part C: n_dims对overlap的影响"""
    log_time("=" * 60)
    log_time("PART C: n_dims Stability Analysis")
    log_time("=" * 60)
    
    n_layers = info.n_layers
    target_layer = n_layers // 2
    dims_list = [5, 10, 15, 20]
    
    # 使用3对概念
    test_pairs = [
        ("big", "large", "synonym"),
        ("hot", "cold", "antonym"),
        ("apple", "fruit", "hyponym"),
        ("doctor", "hospital", "associated"),
        ("apple", "planet", "unrelated"),
    ]
    
    # 收集激活
    all_acts = {}
    for w_a, w_b, rel in test_pairs:
        for w in [w_a, w_b]:
            if w not in all_acts:
                templates = PART_A_WORDS.get(w, EXTRA_WORD_TEMPLATES.get(w, []))
                if templates:
                    acts, found = collect_word_activations(model, tokenizer, device, w, templates, [target_layer], n_layers)
                    all_acts[w] = acts[target_layer]
                    log_time(f"  {w}: {found} samples")
    
    results = {}
    for nd in dims_list:
        log_time(f"\n--- n_dims = {nd} ---")
        word_bases = {}
        for w, acts in all_acts.items():
            if len(acts) >= 2:
                basis, _, _ = pca_subspace(acts, nd)
                word_bases[w] = basis
        
        for w_a, w_b, rel in test_pairs:
            if w_a in word_bases and w_b in word_bases:
                ov = subspace_overlap(word_bases[w_a], word_bases[w_b])
                log_time(f"  {rel:12s}({w_a}/{w_b}): overlap={ov:.3f}")
                results[f"nd{nd}_{rel}"] = float(ov)
    
    # 随机baseline: 两个随机子空间在d_model维空间中的期望overlap
    d_model = info.d_model
    for nd in dims_list:
        random_overlaps = []
        for _ in range(100):
            A = np.random.randn(d_model, nd)
            B = np.random.randn(d_model, nd)
            # QR分解正交化
            A, _ = np.linalg.qr(A)
            B, _ = np.linalg.qr(B)
            random_overlaps.append(subspace_overlap(A, B))
        expected = np.mean(random_overlaps)
        log_time(f"  Random baseline (n_dims={nd}): overlap={expected:.4f}")
        results[f"nd{nd}_random"] = float(expected)
    
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--part", choices=["a", "b", "c", "all"], default="all",
                        help="Run specific part or all")
    args = parser.parse_args()
    
    model_name = args.model
    
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    n_layers = info.n_layers
    log_time(f"{model_name}: n_layers={n_layers}, d_model={info.d_model}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": info.d_model}
    
    try:
        if args.part in ["a", "all"]:
            r_a = part_a_template_removal(model, tokenizer, device, info, W_U, model_name)
            all_results["part_a"] = r_a
        
        if args.part in ["b", "all"]:
            r_b = part_b_axis_mapping(model, tokenizer, device, info, W_U, model_name)
            all_results["part_b"] = r_b
        
        if args.part in ["c", "all"]:
            r_c = part_c_ndims_stability(model, tokenizer, device, info, model_name)
            all_results["part_c"] = r_c
    finally:
        release_model(model)
    
    # 保存结果
    out_dir = PROJECT / "results" / "subspace_topology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"phase59_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    log_time(f"Results saved to {out_file}")
    
    # 打印摘要
    log_time("\n" + "=" * 70)
    log_time(f"PHASE 59 SUMMARY - {model_name}")
    log_time("=" * 70)
    
    if "part_a" in all_results:
        log_time("\nPart A: Template PC Removal")
        for key, data in all_results["part_a"].items():
            log_time(f"  {key}:")
            for k, v in data.items():
                log_time(f"    {k}: {v}")
    
    if "part_b" in all_results:
        log_time("\nPart B: Axis Mapping")
        for key, data in all_results["part_b"].items():
            log_time(f"  {key}:")
            log_time(f"    adj_overlaps: {data.get('adj_overlaps', [])}")
            log_time(f"    nonadj_overlaps: {data.get('nonadj_overlaps', [])}")
    
    if "part_c" in all_results:
        log_time("\nPart C: n_dims Stability")
        for nd in [5, 10, 15, 20]:
            line = f"  n_dims={nd}:"
            for rel in ["synonym", "antonym", "hyponym", "associated", "unrelated"]:
                k = f"nd{nd}_{rel}"
                if k in all_results["part_c"]:
                    line += f" {rel}={all_results['part_c'][k]:.3f}"
            rand_k = f"nd{nd}_random"
            if rand_k in all_results["part_c"]:
                line += f" random={all_results['part_c'][rand_k]:.4f}"
            log_time(line)
    
    log_time("Done!")


if __name__ == "__main__":
    main()
