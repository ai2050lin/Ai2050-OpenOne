
import os
import sys

snapshot_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

def live_verify():
    print(f"PATH: {snapshot_path}")
    print(f"ABS PATH: {os.path.abspath(snapshot_path)}")
    print(f"EXISTS: {os.path.exists(snapshot_path)}")
    
    if os.path.exists(snapshot_path):
        files = os.listdir(snapshot_path)
        print(f"FILES: {files}")
        
        config_file = os.path.join(snapshot_path, "config.json")
        print(f"CONFIG FILE: {config_file}")
        print(f"CONFIG EXISTS: {os.path.exists(config_file)}")
        
        try:
            with open(config_file, "r") as f:
                print(f"CONFIG READABLE, Size: {len(f.read())}")
        except Exception as e:
            print(f"CONFIG READ FAIL: {e}")

    try:
        from transformers import AutoConfig
        print("Attempting AutoConfig.from_pretrained...")
        cfg = AutoConfig.from_pretrained(snapshot_path, local_files_only=True)
        print("SUCCESS AutoConfig")
    except Exception as e:
        print(f"FAIL AutoConfig: {e}")

if __name__ == "__main__":
    live_verify()
