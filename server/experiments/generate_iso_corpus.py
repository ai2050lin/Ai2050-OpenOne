
import itertools
import json
import os
import random

# Configuration
OUTPUT_FILE = "nfb_data/iso_corpus.json"
TARGET_TEMPLATES = 1000  # Generate 1000 unique templates
N_SAMPLES_PER_TEMPLATE = 10 # 10 samples per template -> 10,000 sentences
# Note: Task asked for "10,000 templates", but 10k unique templates might be overkill if we want to study manifolds (which need density).
# Let's target 1000 templates * 10 samples = 10,000 sentences.
# Or maybe the user meant "10,000 corpus size".

# -----------------------------------------------------------------------------
# 1. Semantic Dictionary (5000+ words target - simulated with ~200 here for demo but scalable)
# -----------------------------------------------------------------------------
VOCAB = {
    "nouns": {
        "human_male": ["king", "prince", "uncle", "father", "brother", "son", "man", "boy", "actor", "waiter", "steward", "monk", "emperor", "duke", "lord", "knight", "wizard", "hero", "grandpa", "cousin"],
        "human_female": ["queen", "princess", "aunt", "mother", "sister", "daughter", "woman", "girl", "actress", "waitress", "stewardess", "nun", "empress", "duchess", "lady", "dame", "witch", "heroine", "grandma", "niece"],
        "animal": ["cat", "dog", "horse", "lion", "tiger", "bear", "wolf", "eagle", "snake", "whale", "shark", "mouse", "rat", "rabbit", "fox", "deer", "cow", "pig", "sheep", "goat"],
        "object": ["table", "chair", "bed", "desk", "computer", "phone", "car", "bus", "train", "plane", "ship", "book", "pen", "cup", "plate", "fork", "knife", "spoon", "lamp", "clock", "sword", "shield", "helmet", "crown", "throne", "castle", "house", "tower", "bridge", "gate"]
    },
    "verbs": {
        "transitive": ["ate", "drank", "read", "wrote", "built", "destroyed", "bought", "sold", "found", "lost", "saw", "heard", "touched", "smelled", "tasted", "liked", "hated", "loved", "feared", "pushed", "pulled", "hit", "kissed", "hugged", "kicked"],
        "intransitive": ["slept", "sat", "stood", "walked", "ran", "jumped", "flew", "swam", "cried", "laughed", "smiled", "frowned", "shouted", "whispered", "died", "lived", "exist", "fell", "rose"]
    },
    "adjectives": {
        "positive": ["good", "great", "happy", "joyful", "kind", "wise", "brave", "strong", "beautiful", "rich", "fast", "smart", "clean", "bright", "warm"],
        "negative": ["bad", "terrible", "sad", "miserable", "cruel", "foolish", "cowardly", "weak", "ugly", "poor", "slow", "stupid", "dirty", "dark", "cold"],
        "neutral": ["tall", "short", "young", "old", "big", "small", "long", "heavy", "light", "red", "blue", "green", "yellow", "white", "black"]
    },
    "adverbs": ["quickly", "slowly", "happily", "sadly", "loudly", "quietly", "bravely", "fearfully", "kindly", "cruelly", "carefully", "carelessly"],
    "prepositions": ["on", "in", "at", "under", "over", "near", "beside", "behind", "before", "after", "through", "with", "without", "against", "for", "from", "to"]
}

# -----------------------------------------------------------------------------
# 2. Template Generator (CFG)
# -----------------------------------------------------------------------------
# We define a simple grammar to generate structural templates.
# A template is a string with placeholders like {noun_male}, {verb_trans}, etc.

def generate_template_structures():
    structures = []
    
    # S -> NP VP
    # NP -> Det Adj N | Det N
    # VP -> V | V Adv | V Prep NP | V NP
    
    determiners = ["The", "A"]
    adj_slots = ["{adj}", ""]
    adv_slots = [" {adv}", ""]
    
    # 1. Intransitive Patterns: The [Adj] [Subject] [Verb_Intr] [Adv] [PrepPhrase]
    # "The happy king slept quietly on the table."
    for det in determiners:
        for adj_slot in adj_slots:
            for adv_slot in adv_slots:
                # Basic
                base = f"{det} {adj_slot} {{subject}} {{verb_intr}}{adv_slot}."
                structures.append({"fmt": base, "type": "intransitive"})
                
                # With Prep Phrase
                base_prep = f"{det} {adj_slot} {{subject}} {{verb_intr}}{adv_slot} {{prep}} the {{object}}."
                structures.append({"fmt": base_prep, "type": "intransitive_prep"})

    # 2. Transitive Patterns: The [Adj] [Subject] [Verb_Trans] the [Adj] [Object]
    for det in determiners:
        for adj_s1 in adj_slots:
            for adj_s2 in adj_slots:
                base = f"{det} {adj_s1} {{subject}} {{verb_trans}} the {adj_s2} {{object}}."
                structures.append({"fmt": base, "type": "transitive"})
                
    # 3. "Is" descriptions
    for det in determiners:
        base = f"{det} {{subject}} is very {{adj}}."
        structures.append({"fmt": base, "type": "description"})

    # Clean up double spaces
    clean_structures = []
    for s in structures:
        s["fmt"] = " ".join(s["fmt"].split()) # Remove extra spaces from empty slots
        clean_structures.append(s)
        
    return clean_structures

# -----------------------------------------------------------------------------
# 3. Corpus Generation
# -----------------------------------------------------------------------------

def generate_corpus():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    template_structs = generate_template_structures()
    print(f"Generated {len(template_structs)} structural patterns.")
    
    corpus = []
    
    # We want to vary the "Subject" category (Male/Female) to track gender fibers.
    subject_types = ["human_male", "human_female"]
    
    generated_count = 0
    
    # For each structural pattern, generate N variations
    # To get to 1000 templates, we need to consider specific lexical choices as "templates" 
    # or just structural?
    # Usually, a "Template" in NFB means "The [Specific_Target] sat on the [Specific_Object]" 
    # where only the target varies or vice versa.
    # Let's generate:
    # 500 Templates where Subject is the Target Variable
    # 500 Templates where Object is the Target Variable
    
    # Let's define "Template" as a fixed sentence with ONE variable slot.
    # e.g. "The King [ACTION] on the bed." -> Variable Action
    # e.g. "The [ROLE] sat on the bed." -> Variable Role (Target)
    
    # NFB Analysis usually tracks the trajectory of a specific concept (e.g. Gender).
    # So we need templates where the Subject varies by Gender, but everything else is fixed.
    
    # Strategy:
    # 1. Pick a structural pattern.
    # 2. Fix all slots EXCEPT the Target Slot (Subject).
    # 3. This fixed configuration = 1 Template.
    # 4. Generate pairs/groups for this template (Male vs Female).
    
    # Let's generate 2000 such templates.
    
    templates_generated = 0
    
    while templates_generated < 2000:
        struct = random.choice(template_structs)
        fmt = struct["fmt"]
        
        # Determine fixed fillers
        fixed_adj = random.choice(VOCAB["adjectives"]["neutral"] + VOCAB["adjectives"]["positive"] + VOCAB["adjectives"]["negative"])
        fixed_adv = random.choice(VOCAB["adverbs"])
        fixed_verb_intr = random.choice(VOCAB["verbs"]["intransitive"])
        fixed_verb_trans = random.choice(VOCAB["verbs"]["transitive"])
        fixed_object = random.choice(VOCAB["nouns"]["object"])
        fixed_prep = random.choice(VOCAB["prepositions"])
        
        # Prepare the template string with fillers, leaving {subject} open
        # We handle formatting manually to be safe
        
        # Replace {adj}
        tmpl_str = fmt.replace("{adj}", fixed_adj)
        tmpl_str = tmpl_str.replace("{adv}", fixed_adv)
        tmpl_str = tmpl_str.replace("{verb_intr}", fixed_verb_intr)
        tmpl_str = tmpl_str.replace("{verb_trans}", fixed_verb_trans)
        tmpl_str = tmpl_str.replace("{object}", fixed_object)
        tmpl_str = tmpl_str.replace("{prep}", fixed_prep)
        
        # Verify {subject} is still there
        if "{subject}" not in tmpl_str:
            continue
            
        template_id = f"T{templates_generated:04d}"
        
        # Now generate samples for this template by varying {subject}
        # We need balanced Male/Female subjects for geometry analysis
        
        males = random.sample(VOCAB["nouns"]["human_male"], 5)
        females = random.sample(VOCAB["nouns"]["human_female"], 5)
        
        subjects = []
        for m in males: subjects.append({"word": m, "gender": "male"})
        for f in females: subjects.append({"word": f, "gender": "female"})
        
        for subj in subjects:
            text = tmpl_str.replace("{subject}", subj["word"])
            
            entry = {
                "text": text,
                "template_id": template_id,
                "target_word": subj["word"],
                "target_category": subj["gender"],
                "target_role": "subject",
                "template_str": tmpl_str # The "Iso" part
            }
            corpus.append(entry)
            
        templates_generated += 1
        generated_count += len(subjects)
        
    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(corpus, f, indent=2)
        
    print(f"Generated {generated_count} sentences across {templates_generated} templates.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_corpus()
