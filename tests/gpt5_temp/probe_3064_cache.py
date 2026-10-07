# -*- coding: utf-8 -*-
"""Probe DynamicCache access path for transformers version in venv.
No model load. Writes findings to probe_3064_cache_out.txt
"""
import io
import sys
import inspect
import traceback

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_3064_cache_out.txt"

lines = []


def w(s):
    lines.append(str(s))


try:
    import transformers
    w("transformers version: %s" % transformers.__version__)
    import torch
    w("torch version: %s" % torch.__version__)

    from transformers.cache_utils import DynamicCache
    w("")
    w("=== DynamicCache class attributes ===")
    w("has attr key_cache on class: %s" % hasattr(DynamicCache, "key_cache"))
    w("has attr value_cache on class: %s" % hasattr(DynamicCache, "value_cache"))
    w("has attr layers on class: %s" % hasattr(DynamicCache, "layers"))
    w("has __getitem__: %s" % hasattr(DynamicCache, "__getitem__"))
    w("has to_legacy_cache: %s" % hasattr(DynamicCache, "to_legacy_cache"))
    w("has update: %s" % hasattr(DynamicCache, "update"))

    w("")
    w("=== try: build a tiny DynamicCache via update() ===")
    import torch as T
    cache = DynamicCache()
    k = T.zeros(1, 4, 5, 128)  # (batch, n_kv_heads, seq, head_dim)
    v = T.zeros(1, 4, 5, 128)
    for li in range(2):
        cache.update(k, v, li)
    w("after 2 updates: type=%s" % type(cache).__name__)
    w("dir filtered: %s" % [a for a in dir(cache)
                             if not a.startswith("_")])
    # candidate access paths
    w("")
    w("=== candidate access paths on populated cache ===")
    try:
        kc = cache.key_cache
        w("path A cache.key_cache -> OK, type=%s, len=%s"
          % (type(kc).__name__, len(kc)))
        w("  key_cache[1] shape=%s" % (tuple(kc[1].shape),))
    except Exception as e:
        w("path A cache.key_cache -> FAIL: %r" % (e,))
    try:
        ly = cache.layers
        w("path B cache.layers -> OK, type=%s, len=%s"
          % (type(ly).__name__, len(ly)))
        w("  layers[1] type=%s" % type(ly[1]).__name__)
        try:
            w("  layers[1].keys shape=%s"
              % (tuple(ly[1].keys.shape),))
        except Exception as e2:
            w("  layers[1].keys FAIL: %r" % (e2,))
        try:
            w("  layers[1].keys_tensor shape=%s"
              % (tuple(ly[1].keys_tensor.shape),))
        except Exception as e3:
            w("  layers[1].keys_tensor FAIL: %r" % (e3,))
    except Exception as e:
        w("path B cache.layers -> FAIL: %r" % (e,))
    try:
        item = cache[1]
        w("path C cache[1] -> OK, type=%s, len=%s"
          % (type(item).__name__, len(item)))
        w("  cache[1][0] shape=%s" % (tuple(item[0].shape),))
    except Exception as e:
        w("path C cache[1] -> FAIL: %r" % (e,))
    try:
        leg = cache.to_legacy_cache()
        w("path D to_legacy_cache -> OK, type=%s, len=%s"
          % (type(leg).__name__, len(leg)))
        w("  leg[1][0] shape=%s" % (tuple(leg[1][0].shape),))
    except Exception as e:
        w("path D to_legacy_cache -> FAIL: %r" % (e,))
except Exception:
    w("FATAL in probe:")
    w(traceback.format_exc())

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("PROBE_DONE")
