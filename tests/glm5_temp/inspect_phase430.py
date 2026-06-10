"""Deep inspect Phase 430 data structure"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results_dir = "d:/Ai2050/TransformerLens-Project/results/phase430_natural_transport"

for model_name in ["qwen3"]:
    for rnd in [1]:
        fname = f"{model_name}_phase430_r{rnd}.json"
        fpath = os.path.join(results_dir, fname)
        d = json.load(open(fpath, 'r', encoding='utf-8'))
        per_obj = d.get('per_object', {})
        
        for obj_name in ["car"]:
            obj_data = per_obj[obj_name]
            print(f"=== {model_name} R{rnd} - {obj_name} ===")
            print(f"Top-level keys: {list(obj_data.keys())}")
            
            # Print each key's type and size
            for k, v in obj_data.items():
                if isinstance(v, dict):
                    print(f"  {k}: dict with {len(v)} keys -> {list(v.keys())[:5]}")
                    # Go one level deeper for the first key
                    if v:
                        first_key = list(v.keys())[0]
                        first_val = v[first_key]
                        if isinstance(first_val, dict):
                            print(f"    [{first_key}]: dict -> {list(first_val.keys())[:8]}")
                            # Go one more level
                            if first_val:
                                fk2 = list(first_val.keys())[0]
                                fv2 = first_val[fk2]
                                if isinstance(fv2, dict):
                                    print(f"      [{fk2}]: {list(fv2.keys())[:8]}")
                                    # One more
                                    if fv2:
                                        fk3 = list(fv2.keys())[0]
                                        fv3 = fv2[fk3]
                                        if isinstance(fv3, dict):
                                            print(f"        [{fk3}]: {list(fv3.keys())[:8]}")
                                            # Print actual values
                                            for k4 in list(fv3.keys())[:6]:
                                                v4 = fv3[k4]
                                                if isinstance(v4, (int, float)):
                                                    print(f"          {k4}: {v4:.4f}")
                                                elif isinstance(v4, str):
                                                    print(f"          {k4}: {v4[:50]}")
                                                elif isinstance(v4, dict):
                                                    print(f"          {k4}: dict({len(v4)}) {list(v4.keys())[:4]}")
                                        else:
                                            print(f"        [{fk3}]: {type(fv3).__name__} = {str(fv3)[:100]}")
                                else:
                                    print(f"      [{fk2}]: {type(fv2).__name__} = {str(fv2)[:100]}")
                        else:
                            print(f"    [{first_key}]: {type(first_val).__name__} = {str(first_val)[:100]}")
                elif isinstance(v, list):
                    print(f"  {k}: list[{len(v)}]")
                else:
                    print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
