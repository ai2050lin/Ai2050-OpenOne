# -*- coding: utf-8 -*-
import io, os

p = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-07.md"
entry = u"""
- [修复] fusion-theme.css 的 `.program-image` background 简写覆盖了 `.horizon/.brain/.forum` 的 url() 栏目图（同特异性后加载胜出）→ 主题层末尾按原 url 重声明三变体+ lighthouse，保留融合白色渐变叠层；playwright 3/3 PASS（computed backgroundImage 含 url）、截图 site_fusion_index_fixed.png。
"""
with io.open(p, "a", encoding="utf-8") as f:
    f.write(entry)
print("LOG_OK", os.path.getsize(p))
