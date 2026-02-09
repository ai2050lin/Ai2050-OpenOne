
import os

import nltk
import requests
from transformers import GPT2Tokenizer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

def download_tinyshakespeare(filepath):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists(filepath):
        print(f"Downloading TinyShakespeare to {filepath}...")
        response = requests.get(url)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("TinyShakespeare already exists.")

def create_vocab_split(input_file, output_file):
    print("Creating Vocabulary Split (Structure vs Content)...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # 1. Tokenize a sample to find frequent "Structure" words
    # For efficiency, just take the first 100k characters
    sample_text = text[:100000]
    tokens = nltk.word_tokenize(sample_text)
    pos_tags = nltk.pos_tag(tokens)
    
    structure_words = set()
    content_words = set()
    
    # Heuristic: 
    # Structure: DT (The), IN (On), CC (And), PRP (He), MD (Can), TO (To), RB (Not)
    # Content: NN (Noun), VB (Verb), JJ (Adjective)
    
    structure_tags = {'DT', 'IN', 'CC', 'PRP', 'PRP$', 'MD', 'TO', 'RB', 'RBR', 'RBS', 'WDT', 'WP', 'WP$', 'WRB', 'CD', 'EX', 'PDT'}
    
    for word, tag in pos_tags:
        if tag in structure_tags or not word.isalpha(): # Punctuation is structure
            structure_words.add(word.lower())
        else:
            content_words.add(word.lower())
            
    # Also add top 100 most frequent words to structure just in case
    fd = nltk.FreqDist(word.lower() for word in tokens if word.isalpha())
    for word, count in fd.most_common(50):
        structure_words.add(word)
        if word in content_words:
            content_words.remove(word)
            
    print(f"Structure Vocabulary Size (approx): {len(structure_words)}")
    print(f"Content Vocabulary Size (approx): {len(content_words)}")
    
    # Save the split to a file for the model to load
    # We will map GPT2 Token IDs -> Stream ID (0=Structure, 1=Content)
    
    vocab = tokenizer.get_vocab()
    token_map = {} # ID -> 0 or 1
    
    count_structure = 0
    count_content = 0
    
    for word, idx in vocab.items():
        # GPT2 tokens have 'Ġ' for space
        clean_word = word.replace('Ġ', '').lower()
        
        if clean_word in structure_words or not clean_word.isalpha():
            token_map[idx] = 0 # Structure
            count_structure += 1
        else:
            token_map[idx] = 1 # Content
            count_content += 1
            
    print(f"GPT2 Vocab Split - Structure: {count_structure}, Content: {count_content}")
    
    import json
    with open(output_file, 'w') as f:
        json.dump(token_map, f)
        
    print(f"Saved vocabulary split to {output_file}")

if __name__ == "__main__":
    data_dir = "experiments/data"
    input_path = os.path.join(data_dir, "input.txt")
    vocab_path = os.path.join(data_dir, "vocab_split.json")
    
    download_tinyshakespeare(input_path)
    create_vocab_split(input_path, vocab_path)
