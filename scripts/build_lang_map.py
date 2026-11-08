# -*- coding: utf-8 -*-
"""重建语言图谱（lang_map.v1）：读 shared/dist_protocol/lang_map/structure.json、语言目录 v3、语言树模板映射与项目词表，
写 lang_map.json、words.json 与 manifest.json（含 sha256）。校验不通过时不写文件。
用法  python scripts/build_lang_map.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import lang_map  # noqa: E402

if __name__ == '__main__':
    manifest = lang_map.build()
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
