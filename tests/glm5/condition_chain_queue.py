"""Run the remaining frozen CUDA jobs serially, retrying only lock contention."""
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE / "result"


def run(script, args, folder, key):
    start = time.monotonic()
    for attempt in range(1, 61):
        path = BASE / folder / f"{key}-queued-{attempt:02d}.log"
        if path.exists():
            continue
        with path.open("xb") as log:
            result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-u",
                                     str(HERE/script), *args],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return 0
        if b"RuntimeError: GPU is leased by another research/live process" not in path.read_bytes()[-3000:]:
            return result.returncode
        print(f"{key} lock wait elapsed={time.monotonic()-start:.1f}s attempt={attempt}/60 retry_in=10s", flush=True)
        time.sleep(10)
    return 75


if __name__ == "__main__":
    jobs = [
        ("condition_chain_candidates.py", ["run"], "condition_chain_candidates_20261115_v3", "q4"),
        ("condition_chain_chat.py", ["run","q14"], "condition_chain_chat_20261115_v2", "q14"),
        ("condition_chain_chat.py", ["run","glm"], "condition_chain_chat_20261115_v2", "glm"),
        ("condition_chain_isolated.py", ["run"], "condition_chain_isolated_20261115_v4", "isolated"),
        ("condition_chain_reference_stream.py", ["q14"], "condition_chain_chat_20261115_v2", "q14-stream"),
        ("condition_chain_diverse_candidates.py", ["run"], "condition_chain_diverse_candidates_20261115_v5", "diverse"),
    ]
    for script,args,folder,key in jobs:
        if len(sys.argv)>1 and key not in sys.argv[1:]:
            continue
        print(key, run(script,args,folder,key), flush=True)
