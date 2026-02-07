
import json

import requests


def test_endpoint():
    try:
        response = requests.get("http://localhost:8888/fibernet_v2/demo")
        if response.status_code == 200:
            data = response.json()
            print("Successfully fetched FiberNet V2 demo data.")
            print(f"Manifold Nodes: {len(data['manifold_nodes'])}")
            print(f"Fibers: {len(data['fibers'])}")
            print(f"Connections: {len(data['connections'])}")
            return True
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False

if __name__ == "__main__":
    test_endpoint()
