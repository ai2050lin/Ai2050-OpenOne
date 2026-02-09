import json
import time

import requests

API_BASE = "http://localhost:8888"

def test_snn_api():
    print("1. Initializing SNN...")
    init_data = {
        "layers": {
            "Retina_Shape": 20,
            "Retina_Color": 20,
            "Object_Fiber": 20
        },
        "connections": [
            {"src": "Retina_Shape", "tgt": "Object_Fiber", "type": "one_to_one", "weight": 0.8},
            {"src": "Retina_Color", "tgt": "Object_Fiber", "type": "one_to_one", "weight": 0.8}
        ]
    }
    
    try:
        res = requests.post(f"{API_BASE}/snn/initialize", json=init_data)
        if res.status_code == 200:
            print(f"✓ SNN Initialized: {res.json()}")
        else:
            print(f"✗ Init failed: {res.text}")
            return
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return

    print("\n2. Injecting Stimulus (Apple + Red)...")
    try:
        # Stimulate Shape #5
        requests.post(f"{API_BASE}/snn/stimulate", json={
            "layer_name": "Retina_Shape", "pattern_idx": 5, "intensity": 2.0
        })
        # Stimulate Color #5
        requests.post(f"{API_BASE}/snn/stimulate", json={
            "layer_name": "Retina_Color", "pattern_idx": 5, "intensity": 2.0
        })
        print("✓ Stimuli injected")
    except Exception as e:
        print(f"✗ Stimulus error: {e}")

    print("\n3. Running Simulation Steps...")
    try:
        step_res = requests.post(f"{API_BASE}/snn/step", json={"steps": 50})
        data = step_res.json()
        print(f"✓ Simulation ran 50 steps. Current time: {data['time']}")
        
        # Check for spikes
        spikes = data['spikes']
        total_spikes = sum(len(s) for s in spikes.values())
        print(f"✓ Total spikes detected: {total_spikes}")
        if total_spikes > 0:
            print(f"  Spike sample: {json.dumps(spikes, indent=2)}")
        else:
            print("  Warning: No spikes detected (might be normal if neuron accumulation time > 50 steps)")
            
    except Exception as e:
        print(f"✗ Step error: {e}")

if __name__ == "__main__":
    test_snn_api()
