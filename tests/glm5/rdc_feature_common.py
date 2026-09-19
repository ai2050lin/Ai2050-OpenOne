"""Versioned, restartable RDC artifacts. No implicit phase completion or model load."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'tests/glm5/result'
CAMPAIGN = RESULT / 'rdc_feature_campaign_20260909'

def stamp():
    return datetime.now().astimezone().isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    # Windows can deny rename while a read-only HTTP request briefly holds the old file.
    # Preserve atomic writes and bounded retry; never truncate a live result in place.
    for attempt in range(80):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt==79:raise
            time.sleep(.025)

def immutable(path, value):
    if Path(path).exists():
        assert read(path) == value, f'Immutable artifact changed: {path}'
    else:
        save(path, value)

def bits(tensor):
    import torch
    assert tensor.dtype == torch.bfloat16
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().copy()

def unbits(a):
    return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)

def event(run, kind, **payload):
    folder = CAMPAIGN / run
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'events.jsonl'
    # Single research writer. Cursor is a monotonically increasing event count.
    cursor = sum(1 for _ in path.open(encoding='utf-8')) if path.exists() else 0
    data = dict(cursor=cursor+1, run_id=run, timestamp=stamp(), kind=kind, **payload)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
    return data

def status(run, **payload):
    save(CAMPAIGN / run / 'status.json', dict(run_id=run, updated_at=stamp(), **payload))

def npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)
