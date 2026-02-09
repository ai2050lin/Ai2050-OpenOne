import os
import sys

os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.fiber_net import FiberNet
from transformer_lens import HookedTransformer


def main():
    print("Loading Base Model (GPT-2 Small)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    print("Initializing FiberNet...")
    fiber_net = FiberNet(base_model)
    
    # 1. Test before learning
    prompt = "The current emperor of Mars is"
    print(f"\n[Test 1] Before Memory Injection:")
    print(f"Prompt: '{prompt}'")
    
    # Generate
    output = fiber_net.base_model.generate(prompt, max_new_tokens=5, temperature=0, verbose=False)
    print(f"Output: {output}")
    
    # 2. Teach new fact
    print(f"\n[Action] Instant Learning...")
    # Key concept: The context leading up to the answer
    context = "The current emperor of Mars is"
    # Target answer: " Elon" (Note the space for GPT tokenizer)
    target = " Elon" 
    
    fiber_net.instant_learn(context, target)
    
    # 3. Test after learning
    print(f"\n[Test 2] After Memory Injection:")
    print(f"Prompt: '{prompt}'")
    
    # We must use fiber_net.forward to enable the hook, but generate uses model.generate.
    # We need to wrap the generate call to ensure hooks are applied?
    # HookedTransformer.generate() usually doesn't take external hooks easily unless added to context.
    # Ah, FiberNet.forward uses run_with_hooks.
    # But generate() calls forward() internally. 
    # BUT, we defined hooks inside FiberNet.forward.
    # To make generate work, we should add the hook permanently or use a context manager around generate.
    
    # Let's manually register the hook for the generation context
    hook_name = f"blocks.{fiber_net.memory_layer_idx}.hook_resid_post"
    
    def memory_hook(resid_pre, hook):
        # We need to reshape/broadcast memory logic if batching, but for generate (batch=1) it's fine.
        return resid_pre + fiber_net.fiber_memory.read(resid_pre)
        
    with base_model.hooks(fwd_hooks=[(hook_name, memory_hook)]):
        output_after = base_model.generate(prompt, max_new_tokens=5, temperature=0, verbose=False)
        
    print(f"Output: {output_after}")
    
    if "Elon" in output_after:
        print("\n[SUCCESS] FiberNet successfully injected the new fact!")
        with open("success.txt", "w") as f: f.write("SUCCESS")
    else:
        print("\n[FAILURE] FiberNet failed to inject the fact.")
        with open("success.txt", "w") as f: f.write("FAILURE")

    # 4. Generalization? (Parallel Transport Check)
    # Does "The ruler of the Red Planet is" also work?
    # Probably not yet, because the "Key" is exact embedding of the specific context.
    # Real FiberNet needs a generalized Key (the "Fiber Coordinate").
    print("\n[Analysis] To support Generalization (e.g. 'Ruler of Red Planet' -> 'Elon'),")
    print("the 'Key' for memory must be the abstract Fiber Coordinate, not the literal text embedding.")
    print("This requires the Base Manifold to project both prompts to the same location.")

if __name__ == "__main__":
    main()
