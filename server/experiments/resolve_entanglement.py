
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file
sys.stdout = open("entanglement_results.txt", "w", encoding="utf-8")

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
LAYER_ID = 6 # Middle layer where we expect semantic mixing

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_resid_at_last_token(model, text, layer_id):
    cache_name = f"blocks.{layer_id}.hook_resid_post"
    _, cache = model.run_with_cache(text)
    # [batch, pos, d_model] -> [d_model]
    return cache[cache_name][0, -1, :]

def analyze_entanglement_at_layer(model, layer_id):
    print(f"\n--- Analyzing Layer {layer_id} ---")
    
    # Template: "The [SUB] [VERB] the [OBJ]."
    # We want to extract Subject Axis and Object Axis.
    
    subjects = ["cat", "dog", "boy", "girl"]
    objects = ["mouse", "rabbit", "ball", "car"]
    verb = "chased"
    
    # 1. Extract Subject Vectors (Fix Object, Vary Subject)
    # v_sub = Avg( Act(Si, O_fixed) - Act(Sj, O_fixed) )
    sub_diffs = []
    
    # We'll compare each subject to the first one ("cat") as a baseline, 
    # capturing the "Subject-ness" variation.
    # Actually, let's take pairwise differences to get a robust "Subject Direction".
    
    # Fix object to "mouse"
    fixed_obj = "mouse"
    base_text = f"The {subjects[0]} {verb} the {fixed_obj}."
    base_act = get_resid_at_last_token(model, base_text, layer_id)
    
    for s in subjects[1:]:
        text = f"The {s} {verb} the {fixed_obj}."
        act = get_resid_at_last_token(model, text, layer_id)
        sub_diffs.append(act - base_act)
        
    # Valid Subject Axis is the principal component of these variations
    sub_matrix = torch.stack(sub_diffs).float() # [N-1, d_model]
    # PCA to get main direction
    try:
        U, S, V = torch.pca_lowrank(sub_matrix, q=1)
        subject_axis = V[:, 0] # [d_model]
    except:
        subject_axis = sub_diffs[0] # Fallback
        
    print(f"Subject Axis Norm: {subject_axis.norm().item():.4f}")

    # 2. Extract Object Vectors (Fix Subject, Vary Object)
    obj_diffs = []
    fixed_sub = "cat"
    base_text_obj = f"The {fixed_sub} {verb} the {objects[0]}."
    base_act_obj = get_resid_at_last_token(model, base_text_obj, layer_id)
    
    for o in objects[1:]:
        text = f"The {fixed_sub} {verb} the {o}."
        act = get_resid_at_last_token(model, text, layer_id)
        obj_diffs.append(act - base_act_obj)
        
    obj_matrix = torch.stack(obj_diffs).float()
    try:
        U, S, V = torch.pca_lowrank(obj_matrix, q=1)
        object_axis = V[:, 0]
    except:
        object_axis = obj_diffs[0]
        
    print(f"Object Axis Norm: {object_axis.norm().item():.4f}")
    
    # 3. Measure Entanglement (Cosine Similarity)
    cos_sim = F.cosine_similarity(subject_axis.unsqueeze(0), object_axis.unsqueeze(0)).item()
    print(f"Cosine Similarity (Entanglement): {cos_sim:.4f}")
    
    # 4. Orthogonalization (Gram-Schmidt)
    # Project Object axis to be orthogonal to Subject axis
    # v_obj_orth = v_obj - proj_{v_sub}(v_obj)
    # proj = (v_obj . v_sub) / (v_sub . v_sub) * v_sub
    
    dot_prod = torch.dot(object_axis, subject_axis)
    proj = (dot_prod / (torch.dot(subject_axis, subject_axis))) * subject_axis
    object_axis_orth = object_axis - proj
    
    new_sim = F.cosine_similarity(subject_axis.unsqueeze(0), object_axis_orth.unsqueeze(0)).item()
    print(f"Post-Orthogonalization Similarity: {new_sim:.4f}")
    # 5. Verification: Project new data onto orthogonal axes
    # Does changing Subject affect the projection onto Object Axis Orth?
    
    print("  [Verification]")
    # Test case: "The girl chasing the mouse"
    # Compare with: "The boy chasing the mouse" (Subject Change)
    # Projection on Object Axis Orth should be CONSTANT.
    
    t1 = f"The girl {verb} the {fixed_obj}."
    t2 = f"The boy {verb} the {fixed_obj}."
    
    h1 = get_resid_at_last_token(model, t1, layer_id)
    h2 = get_resid_at_last_token(model, t2, layer_id)
    
    # Original Projection (Object Axis)
    proj_obj_1 = torch.dot(h1, object_axis) / object_axis.norm()
    proj_obj_2 = torch.dot(h2, object_axis) / object_axis.norm()
    diff_orig = abs(proj_obj_1 - proj_obj_2).item()
    
    # New Projection (Object Axis Orth)
    proj_orth_1 = torch.dot(h1, object_axis_orth) / object_axis_orth.norm()
    proj_orth_2 = torch.dot(h2, object_axis_orth) / object_axis_orth.norm()
    diff_new = abs(proj_orth_1 - proj_orth_2).item()
    

    print(f"  Subject Change Impact on Object Axis (Original): {diff_orig:.4f}")
    print(f"  Subject Change Impact on Object Axis (Orthogonal): {diff_new:.4f}")
    
    # Check Parallelism of Subject Vectors
    # Axis was built from [cat, dog, boy, girl] vs mouse.
    # Verification used girl vs boy.
    
    # Let's verify if (girl - boy) aligns with the computed subject_axis
    v_verification_sub = h1 - h2 # girl - boy
    sim_verify = F.cosine_similarity(v_verification_sub.unsqueeze(0), subject_axis.unsqueeze(0)).item()
    print(f"  Similarity between Verification Step Vector (Girl-Boy) and Subject Axis: {sim_verify:.4f}")
    
    return cos_sim

def main():
    print("Starting Entanglement Analysis...")
    model = load_model()
    if model is None:
        return
        
    # Analyze multiple layers
    layers = [6] # Focus on the interesting layer
    
    results = {}
    for l in layers:
        sim = analyze_entanglement_at_layer(model, l)
        results[l] = sim
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
