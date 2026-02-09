import json

import requests

url = "http://localhost:8888/head_details"
payload = {
    "prompt": "Test prompt",
    "layer_idx": 0,
    "head_idx": 0
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print("Error details:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
    else:
        print("Success! Response keys:", response.json().keys())
except Exception as e:
    print(f"Request failed: {e}")
