
import itertools
import json
import os
import random
from typing import Dict, List


class IsoCorpusGenerator:
    """
    Generates a controlled corpus where Syntax (Manifold) and Semantics (Fiber) are decoupled.
    Phase 1 DataFabrication Step.
    """
    def __init__(self, output_file="iso_corpus.jsonl"):
        self.output_file = output_file
        
        # 1. Define Syntax Templates (The Manifold Points)
        # These structure the logical relationship between entities.
        self.templates = [
            {
                "id": "T001",
                "structure": "{Subj} {Verb} {Obj} {Prep} {Tool}",
                "description": "Instrumental Action"
            },
            {
                "id": "T002",
                "structure": "{Subj} {Verb} {Obj} {Prep} {Location}",
                "description": "Locative Action"
            },
            {
                "id": "T003",
                "structure": "The {Adj} {Subj} {Verb} {Obj}",
                "description": "Adjective Modification"
            },
             {
                "id": "T004",
                "structure": "If {Subj} {Verb} {Obj}, then {Subj} will be {Adj}",
                "description": "Causal Implication"
            }
        ]
        
        # 2. Define Semantic Dictionary (The Fiber Basis)
        # These are the content atoms. Ideally, vectors of these words form the fiber space.
        self.semantics = {
            "Subj": ["scientist", "engineer", "artist", "doctor", "teacher", "student", "dog", "cat", "robot"],
            "Verb": ["analyzed", "built", "painted", "cured", "taught", "studied", "chased", "observed", "designed"],
            "Obj": ["the structure", "the bridge", "the canvas", "the patient", "the lesson", "the theory", "the ball", "the stars", "the mechanism"],
            "Tool": ["a microscope", "a hammer", "a brush", "medicine", "a book", "a telescope", "a computer", "a laser"],
            "Location": ["in the lab", "on the site", "in the studio", "in the hospital", "in the classroom", "in the library", "in the park", "in space"],
            "Prep": ["with", "using", "via", "through"], # For T001
            # Note: For T002, Prep might be "in", "at", "on" which are baked into Location or separate. 
            # Let's handle T002 Prep specifically in generation logic or use generic Preps if applicable.
            # Simplified for T002: "in", "at" are distinct from "with". 
            "Adj": ["happy", "sad", "smart", "fast", "slow", "red", "blue", "complex", "simple"]
        }
        
    def generate_controlled_samples(self, n_samples_per_template=1000):
        """
        Generates samples while tracking their topological coordinates (Template ID) 
        and fiber coordinates (Word Choices).
        """
        data = []
        
        print(f"Generating {n_samples_per_template} samples per template...")
        
        for template in self.templates:
            
            # Simple Monte Carlo sampling for now. 
            # In a rigorous extraction, we might need grid sampling for smaller vocabularies.
            
            for _ in range(n_samples_per_template):
                sample_meta = {
                    "template_id": template["id"],
                    "template_desc": template["description"],
                    "fiber_coords": {}
                }
                
                # Fill the template
                text = template["structure"]
                
                # We identify placeholders like {Subj}
                # A robust implementation would use regex, but simple string replace works for fixed templates.
                
                # Specific logic for T002 to ensure preposition correctness if needed
                current_preps = self.semantics["Prep"]
                if template["id"] == "T002":
                     # For locative, usually 'in' or 'at' are used, 'with' makes less sense.
                     # Let's override Prep for T002 for grammatical correctness
                     current_preps = ["in", "at", "near"]
                
                # Helper to replace and record
                for key in ["Subj", "Verb", "Obj", "Tool", "Location", "Adj"]:
                     if f"{{{key}}}" in text:
                         choice = random.choice(self.semantics[key])
                         text = text.replace(f"{{{key}}}", choice)
                         sample_meta["fiber_coords"][key] = choice
                
                # Handle Prep separately as it logic-dependent
                if "{Prep}" in text:
                    choice = random.choice(current_preps)
                    text = text.replace("{Prep}", choice)
                    sample_meta["fiber_coords"]["Prep"] = choice
                
                sample_meta["text"] = text
                data.append(sample_meta)
                
        return data

    def save_corpus(self, data):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {self.output_file}")


if __name__ == "__main__":
    # Create output directory if not exists
    os.makedirs("data", exist_ok=True)
    
    generator = IsoCorpusGenerator("data/iso_corpus.jsonl")
    corpus = generator.generate_controlled_samples(n_samples_per_template=2000)
    generator.save_corpus(corpus)
