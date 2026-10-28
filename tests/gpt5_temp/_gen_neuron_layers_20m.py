# -*- coding: utf-8 -*-
"""一次性维护任务：为 20M token census 运行生成 neuron_layers 逐层摘要（36 块，零 GPU）。

背景：/api/neurons/layer 与 /tower 依赖 <census 运行>/neuron_layers/<code_hash>/L<b>.json
（由 build_layer_bases 生成）；本机两套 census（1M/20M）均无该目录，导致 v8「神经元」
阵列一直返回 pending，单元浏览器处于待激活状态。本脚本调用
server.neuron_census_service.build_layer_bases 补齐 20M 套（pick_census 选中的是最大 n_tokens 运行）。

用法（仓库根目录）：
    .venv/Scripts/python.exe tests/gpt5_temp/_gen_neuron_layers_20m.py
产物：D:/ai2050_native/_census/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-20000000/neuron_layers/<code_hash>/L0..L35.json
幂等：已存在的块自动跳过，可重复执行。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server import neuron_census_service as S  # noqa: E402

MODEL_KEY = "6be1034f0b55a6df-bf16"
RUN = "2ca5f0a756d36c8a-20000000"


def main() -> int:
    c = S.pick_census(MODEL_KEY, RUN)
    print(f"[pick] model_key={c['model_key']} run={c['run']} n_tokens={c['n_tokens']}", flush=True)
    t0 = time.time()
    out = S.build_layer_bases(MODEL_KEY, c["run"])
    out_dir = S.layer_base_dir(MODEL_KEY, c["run"])
    n = len(list(out_dir.glob("L*.json"))) if out_dir.is_dir() else 0
    print(f"[done] computed={out.get('computed', 0)} total_files={n} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0 if n >= 36 else 1


if __name__ == "__main__":
    raise SystemExit(main())
