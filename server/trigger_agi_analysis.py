import json
import time

import requests


def trigger():
    url = "http://localhost:8888/verify_agi"
    print(f"Triggering analysis at {url}...")
    print("This might take a minute as it runs RSA on 4B model...")
    start = time.time()
    try:
        resp = requests.post(url, timeout=600) # 10 min timeout
        print(f"Completed in {time.time() - start:.2f}s")
        
        if resp.status_code == 200:
            print("Analysis Success!")
            data = resp.json()
            # Save to file just in case
            with open("qwen_analysis_results.json", "w") as f:
                json.dump(data, f, indent=2)
            print("Results saved to qwen_analysis_results.json")
            print(json.dumps(data, indent=2))
        else:
            print(f"Failed: {resp.status_code}")
            print(resp.text)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Ensure server.py is running on port 8888.")

if __name__ == "__main__":
    trigger()
