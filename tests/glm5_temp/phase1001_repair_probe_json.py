#!/usr/bin/env python3
"""Remove captured console lines from the Phase 1001 probe JSON."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "source_path_probe.json"
)


text = PATH.read_text(encoding="utf-8-sig")
start = text.index("\n[\n") + 1
end_marker = "\n[model_utils] GPU memory released"
end = text.index(end_marker, start)
payload = json.loads(text[start:end])
PATH.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"repaired={PATH} rows={len(payload)}")
