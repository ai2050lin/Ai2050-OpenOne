# -*- coding: utf-8 -*-
"""Download gpt2-small weights into the path server.py expects.

Target dir: C:\\Users\\Admin\\.cache\\huggingface\\gpt2-local\\
Server check: model.safetensors exists and size > 100_000_000 bytes,
tokenizer loads with local_files_only=True from the same dir.
Uses hf-mirror.com (domestic mirror) for reachability.
"""
import io
import os
import sys
import time

REPORT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_gpt2_dl.txt"
TARGET = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "gpt2-local")

def log(msg):
    with io.open(REPORT, "a", encoding="utf-8", newline="\n") as f:
        f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))

def main():
    log("target=%s" % TARGET)
    os.makedirs(TARGET, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # hf-mirror cannot proxy Xet CAS -> 401
    log("HF_ENDPOINT=%s XET_DISABLED=%s" % (os.environ.get("HF_ENDPOINT"), os.environ.get("HF_HUB_DISABLE_XET")))
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(
        repo_id="openai-community/gpt2",
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ],
        local_dir=TARGET,
        max_workers=4,
    )
    log("snapshot_download done -> %s (%.1fs)" % (path, time.time() - t0))
    names = sorted(os.listdir(TARGET))
    total = 0
    for n in names:
        p = os.path.join(TARGET, n)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            total += sz
            log("  %-28s %10.1f MB" % (n, sz / 1048576.0))
    log("total=%d files=%d" % (total, len(names)))
    w = os.path.join(TARGET, "model.safetensors")
    ok_w = os.path.isfile(w) and os.path.getsize(w) > 100_000_000
    log("WEIGHT_CHECK size>100MB: %s" % ok_w)
    # tokenizer load check (cheap, local only)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    log("TOKENIZER_CHECK: %s vocab=%d" % (tok.__class__.__name__, tok.vocab_size))
    log("ALL_OK" if (ok_w and tok is not None) else "INCOMPLETE")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAILED: %r" % (exc,))
        sys.exit(1)
