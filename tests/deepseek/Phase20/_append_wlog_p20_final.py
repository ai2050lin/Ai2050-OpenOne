# -*- coding: utf-8 -*-
"""向当日工作区日志 2026-10-02.md 追加 Phase 20 收尾补记（append-only、CRLF）。"""
import os
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'

BLOCK = (
    '\r\n'
    '## Phase 20 收尾链补记（post-hoc 域分解 / 勘误 / 记忆精简 / 技能同步）（11:35）\r\n'
    '\r\n'
    '- **P9 事后域分解（不改判）**：新增 `tests\\deepseek\\Phase20\\posthoc_p9_xhalf_domain.py` → '
    '`posthoc_p9_xhalf_domain.{json,txt}`（sha8 `6f02eee7`）。as-coded `max|Δxhalf|` = **0.2645**（A0，超差 **100% 由 ℓ=1**）/ **0.0338**（A1）vs 容差 0.05；'
    'P16 的 `XH_FAITHFUL_TOL` 只在 **ℓ≥6** 标定过 ⇒ 冻结 REACH 域上两对 **0.006171（A0）/ 0.003713（A1）皆 PASS**'
    '（A0 该值与 P16 标定值 ~**1e-16** 一致，因本就是同一比较）。⇒ **P9 的 FAIL 属「判据文字未写明域」的歧义，非物理不稳定**。\r\n'
    '- **同轮勘误补三条**：`E-comv`（冻结谱重算按构造与臂无关 ⇒ 配对 Δ≡0，属**空检查**，跨精度证据须看本臂自谱版）、'
    '`E-xhdom`（判据容差须写明适用域）、`E-rho`（`rho_b_all` 是 **dict**，渲染器误用标量取值器 ⇒ **交付前**逐字节回滚基线重跑链，'
    'MEMO **469,271 B / `ec7be6b6` → 488,772 B / `68eddd46`**）。\r\n'
    '- **Memory 精简 + 回填**：`MEMORY.md` 由 **6,340 → 4,254 字符**（10,676 → **7,018 B**，**−33%**，LF-only 无 BOM）；'
    '合入 §5 的 P20 三条教训、§7 死线改写为「**Phase 21 = 把跨精度推进到组件级向量预算与权重实现级**」、§8 技能计数。\r\n'
    '- **技能同步**：`rdc-phase-closeout` 新增**第 33 条教训**（「交付前自查 ⇒ 生成件缺陷可逐字节回滚基线后重跑链」+「`E-rho` 取值器按对象类型分层」），'
    '计数 **32 → 33**（57,005 → 59,346 B，LF 保持）；`rdc-main-axis-probe` 已含 **61 坑**（P20 覆盖完成，无需改）。\r\n'
    '- **独立复核复跑**：`disk_verify_phase20.txt` = **PASS 215 / FAIL 0 / WARN 0 → `ALL_PASS`**。\r\n'
    '- **交付件**：`tests\\deepseek_temp\\Phase20\\present_phase20.html`（**38,227 B / 362 行**，含 P9 域分解表 6dp + `[E-xhdom]`）。\r\n'
)


def main():
    old = open(P, 'rb').read()
    assert b'\xef\xbb\xbf' not in old[:3], 'unexpected BOM'
    t = old.decode('utf-8-sig')
    assert t.count('\r\n') > 0 and (t.count('\n') - t.count('\r\n')) == 0, 'not CRLF-only'
    body = BLOCK.encode('utf-8')
    open(P, 'ab').write(body)
    chk = open(P, 'rb').read()
    ct = chk.decode('utf-8-sig')
    print('bytes %d -> %d (+%d)' % (len(old), len(chk), len(chk) - len(old)))
    print('crlf', ct.count('\r\n'), 'bare_lf', ct.count('\n') - ct.count('\r\n'))
    print('prefix_ok', chk.startswith(old))
    print('sha8', hashlib.sha256(chk).hexdigest()[:8])
    print('has_block', '收尾链补记' in ct)
    print('DONE')


if __name__ == '__main__':
    main()
