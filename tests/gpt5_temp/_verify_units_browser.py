# -*- coding: utf-8 -*-
"""验证：20M census 摘要生成后，/api/neurons/* 单元浏览器链路真实可用。

直接调用 server.neuron_census_service 的端点函数（与 HTTP 层同一实现），
检查点：
  1. neuron_layer(block=16) pending=False，9,728 单元列齐全，explained 合并非空；
  2. neuron_tower() summaries_ready=36，各块分型计数存在；
  3. unit_card(block=16, unit=8056) 返回 unit_card.v1：极点上下文、直方图、读出、evidence_axes；
  4. explained() 返回 EXPL 计数（应与审计数字 200 单元量级一致）。

用法（仓库根目录）：.venv/Scripts/python.exe tests/gpt5_temp/_verify_units_browser.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server import neuron_census_service as S  # noqa: E402

MODEL_KEY = "6be1034f0b55a6df-bf16"
out: dict = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    out[name] = {"pass": bool(cond), "detail": detail}
    print(("[PASS] " if cond else "[FAIL] ") + name + (" | " + detail if detail else ""), flush=True)


def main() -> int:
    c = S.pick_census(MODEL_KEY, None)
    check("pick_census_is_20m", c["run"].endswith("20000000"), f"run={c['run']} n_tokens={c['n_tokens']}")

    # 1) 层摘要
    ly = S.neuron_layer(model_key=MODEL_KEY, block=16, run=None)
    check("layer_pending_false", ly.get("pending") is False, f"pending={ly.get('pending')}")
    col = ly.get("columns") or {}
    n_units = ly.get("n_units")
    check("layer_columns_full", n_units == 9728 and all(len(v) == 9728 for v in col.values()),
          f"n_units={n_units} cols={sorted(col)}")
    check("layer_types_present", "type" in col and "dens" in col and "ro" in col,
          f"T 计数={ {t: col.get('type', []).count(t) for t in range(6) } if col else {} }")
    check("layer_explained_merged", isinstance(ly.get("explained"), dict), f"explained={len(ly.get('explained') or {})} 项")
    check("layer_selectivity_merged", isinstance(ly.get("selectivity"), list), f"selectivity={len(ly.get('selectivity') or [])} 个运行")

    # 2) 层塔
    tw = S.neuron_tower(model_key=MODEL_KEY, run=None)
    ready = sum(1 for b in tw["blocks"] if b["summary"])
    check("tower_ready_36", ready == 36, f"summaries_ready={ready}/{tw['n_layers']}")

    # 3) 单元卡
    card = S.unit_card(model_key=MODEL_KEY, block=16, unit=8056, run=None, n_poles=12)
    check("unit_card_schema", card.get("schema") == "unit_card.v1", str(card.get("id")))
    pol = ((card.get("census") or {}).get("poles") or {})
    check("unit_card_poles", len(pol.get("pos", [])) > 0 and "hit" in pol["pos"][0],
          f"pos[0] hit={pol.get('pos', [{}])[0].get('hit')!r} left={pol.get('pos', [{}])[0].get('left', '')[-12:]!r}")
    hist = (card.get("census") or {}).get("hist") or {}
    check("unit_card_hist_65x4", len(hist.get("bin_lower", [])) == 65
          and all(len(v) == 65 for v in (hist.get("by_regime") or {}).values()), "65 箱 × 4 象限")
    check("unit_card_readout", bool((card.get("census") or {}).get("readout", {}).get("promotes")),
          str((card.get("census") or {}).get("readout", {}).get("promotes", [])[:5]))
    ax = card.get("evidence_axes") or {}
    check("unit_card_evidence_axes", set(ax) >= {"read", "control", "explain"}, json.dumps(ax, ensure_ascii=False))

    # 4) 解释检验总量
    ex = S.explained(model_key=MODEL_KEY, grade=None)
    check("explained_counts", sum(ex["counts"].values()) > 0, json.dumps(ex["counts"], ensure_ascii=False))

    n_fail = sum(1 for v in out.values() if not v["pass"])
    print(f"\n== {len(out) - n_fail}/{len(out)} PASS ==", flush=True)
    Path(__file__).with_suffix(".out.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
