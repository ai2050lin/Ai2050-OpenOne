
import json

import requests

BASE_URL = "http://localhost:5000"

def test_endpoint(name, method, path, data=None):
    print(f"Testing {name} ({method} {path})...")
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{path}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{path}", json=data)
        
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
        except:
            print(f"Response (text): {response.text[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 30)

if __name__ == "__main__":
    test_endpoint("Health Check", "GET", "/health")
    test_endpoint("V1 Health Check", "GET", "/api/v1/health")
    test_endpoint("Generate Next", "POST", "/generate_next", {
        "prompt": "Test",
        "num_tokens": 1
    })
    test_endpoint("FiberNet V2 Demo", "GET", "/fibernet_v2/demo")
