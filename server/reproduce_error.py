import json

import requests

url = "http://localhost:8888/analyze"
payload = {
    "prompt": "The quick brown fox",
    "top_k": 5
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
        print("Success!")
except Exception as e:
    print(f"Request failed: {e}")
