# -*- coding: utf-8 -*-
"""向当日工作区日志追加 Phase 21 收尾链补记（append-only、CRLF、幂等）。"""
import os
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'

BLOCK = (
    '\r\n'
    '## Phase 21 收尾链补记（disk_verify / MEMORY 回填 / 技能同步 / 交付）（21:45）\r\n'
    '\r\n'
    '- **独立磁盘复核**：`disk_verify_phase21.txt` = **PASS 71 / FAIL 0 → ALL_PASS**'
    '（seal/exec/result/judgement 指纹、A0_bf16 对 P8 盘上文件的独立复现、每臂 `share_v` 求和==1 与 `max` 自洽、'
    '配对 Δ 重算、spearman 平均秩独立实现、verdict 由 predictions 导出、Ledger 条目、MEMO 前缀锚、present 存在）。\r\n'
    '- **同轮勘误（生成件缺陷，不动产物）**：`E-dvsha` —— `disk_verify` 里 '
    "`chk('seal sha == exec.seal_sha256', h8(SEAL) == EX['seal_sha256'], ...)` "
    '把 **8 字符 `sha8`** 与 **64 字符 `sha256`** 相比 ⇒ **永假式**（打印出的 got/exp 是两个相同的 8 位前缀，'
    '极易误判为数据不一致）。独立核对确认 `seal full sha256 = bea6d41548ca...f6fc3a8` **逐位等于** `exec.seal_sha256`；'
    '修比较式后复跑得 **71/0**。\r\n'
    '- **MEMORY 回填**：`MEMORY.md` 新增 P21（§2 条目 / 限界⑬ / §5 P21 四条教训 / §7 死线改 **Phase 22** / '
    '§8 计数 **62 坑·34 教训**），最终 **4,944 字符 / 8,153 B / `b5b716a4`**（LF-only 无 BOM）。'
    '**同时实测到 `Read` 工具对该文件返回陈旧缓存**（判定一律改用 Grep/Python）。\r\n'
    '- **技能同步**：`rdc-main-axis-probe` 坑计数 **61 → 63**（新增：单位阵探针取权重；argmax 并列内翻转 + 跨模型地板不可比），'
    '81,059 → **83,071 B**（`3f598b76`）；`rdc-phase-closeout` 新增**第 34 条教训**'
    '（锚点预检对源 / `sha8` 与 64-hex 长度配对 / `Read` 陈旧缓存），N 线节标题改 **Phase 8 → Phase 21**，'
    '59,346 → **61,589 B**（`a864ea5c`）。\r\n'
    '- **交付件**：`tests\\deepseek_temp\\Phase21\\present_phase21.html`（六卡：A 校准 / B 四臂 / C 跨精度配对 / '
    'D 预测 / E 效应侧 / F 限界；全部数字由 result 现场渲染）。\r\n'
)


def main():
    old = open(P, 'rb').read()
    assert old[:3] != b'\xef\xbb\xbf', 'unexpected BOM'
    t = old.decode('utf-8-sig')
    assert t.count('\r\n') > 0 and (t.count('\n') - t.count('\r\n')) == 0, 'not CRLF-only'
    if '## Phase 21 收尾链补记' in t:
        print('ALREADY present, skip')
        print('bytes', len(old), 'sha8', hashlib.sha256(old).hexdigest()[:8])
        return
    open(P, 'ab').write(BLOCK.encode('utf-8'))
    chk = open(P, 'rb').read()
    ct = chk.decode('utf-8-sig')
    print('bytes %d -> %d (+%d)' % (len(old), len(chk), len(chk) - len(old)))
    print('crlf', ct.count('\r\n'), 'bare_lf', ct.count('\n') - ct.count('\r\n'))
    print('prefix_ok', chk.startswith(old))
    print('sha8', hashlib.sha256(chk).hexdigest()[:8])
    print('has_block', '## Phase 21 收尾链补记' in ct)
    print('DONE')


if __name__ == '__main__':
    main()
