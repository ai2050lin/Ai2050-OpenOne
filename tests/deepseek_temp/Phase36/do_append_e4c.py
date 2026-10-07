# -*- coding: utf-8 -*-
"""把 E4c 附录条目 append 进 deepseek MEMO（保持 BOM+CRLF、append-only）。"""
import time

P = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'

before = open(P, 'rb').read()
n_before = len(before)

entry = (
    "\r\n"
    "## 附录 P36-V1: E4c 词嵌入参数可视化表（非 Phase 编号；纯呈现，无新实验、无新判决）[" + time.strftime('%Y-%m-%d %H:%M') + "]\r\n"
    "\r\n"
    "- 触发：用户要求把 E4/E4b 的词嵌入参数做成可视化表格（按参数值高低着色深浅）。\r\n"
    "- 产物：`tests/deepseek/Phase36/e4c_heatmap_table.py` sha8 `4398ca73`；`tests/deepseek_temp/Phase36/e4c_heatmap_data.json` sha8 `7765dff1`；`e4_heatmap.html` sha8 `0c685591`（33.8 KB，三表：逐 token 参数 / 组级 K=20 / 原始前 96 维热力；零 GPU，全量重算 4.0 s）。\r\n"
    "- 数据保真：同协议同词池同 seed 现场重算，C1（HIGH 901 vs RARE 815、+10.5%、p=7.94e-07、rbc -0.87）、高斯带（854±27）、C2 比（1.57）与 Phase 36 报告逐字一致；个别整数为舍入边界差（907.5 → 907/908），已在页内注明。\r\n"
    "- 表 B 按 E4b 判决呈现：RARE(n=14) PR_spec=11.5 标注 ⚠「K 失配伪影，不可与 K=20 直接比较」；页面结论链 = 双向否证 + 语义域相干性（A1 0.120 / A2 0.111 / A3 0.141 vs HIGH 0.065 / GAUSS 0.003）。\r\n"
    "- 无新结论、无新挂账；仅呈现层。登记册追加于 `tests/deepseek_temp/Phase36/_sha8_register.txt`。\r\n"
)

with open(P, 'ab') as f:
    f.write(entry.encode('utf-8'))

after = open(P, 'rb').read()
print('before=%d after=%d delta=%d prefix_ok=%s' % (
    n_before, len(after), len(after) - n_before, after[:n_before] == before))
