# -*- coding: utf-8 -*-
"""Phase 20 收尾链第十五次：给 rdc-phase-closeout 技能补第 33 条教训。

覆盖本轮唯一未被前 32 条覆盖的两件事：
  (a) E-rho —— 取值器必须按对象类型分层（rho_b_all 是 dict，误用标量取值器）。
  (b) 交付前自查 —— 生成件（MEMO / present）缺陷可「逐字节回滚基线后重跑链」。

严格纪律：逐处 assert count==1 + 回读复核 + py_compile。
"""
import os
import sys

SK = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

OLD_HDR = u'**Phase 8\u201320 \u5b9e\u6d4b\u7684 32 \u6761\u6536\u5c3e\u6559\u8bad**\uff1a'
NEW_HDR = u'**Phase 8\u201320 \u5b9e\u6d4b\u7684 33 \u6761\u6536\u5c3e\u6559\u8bad**\uff1a'

# 插入锚：第 32 条 (f) 段末 + 其后两个空行 + "## 参照实现" 标题
ANCHOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_lesson33_block.txt')
with open(ANCHOR_PATH, 'rb') as f:
    LESSON33 = f.read().decode('utf-8')

ANCHOR = u'\n\n## \u53c2\u7167\u5b9e\u73b0\uff08Phase 3125'


def main():
    raw = open(SK, 'rb').read()
    txt = raw.decode('utf-8-sig')
    bom = raw[:3] == b'\xef\xbb\xbf'
    crlf = txt.count('\r\n')
    lf_only = txt.count('\n') - crlf

    # 1) header count
    assert txt.count(OLD_HDR) == 1, 'OLD_HDR count=%d' % txt.count(OLD_HDR)
    txt = txt.replace(OLD_HDR, NEW_HDR)
    assert txt.count(NEW_HDR) == 1

    # 2) lesson 33 block before "## 参照实现"
    assert txt.count(ANCHOR) == 1, 'ANCHOR count=%d' % txt.count(ANCHOR)
    ins = LESSON33.rstrip('\r\n') + txt[txt.index(ANCHOR):txt.index(ANCHOR) + 2]
    # 保持原尾随：用 (block + 两个换行 + 原标题起点) 重建
    txt = txt.replace(ANCHOR, LESSON33.rstrip('\r\n') + ANCHOR, 1)
    assert txt.count('33. **\u4ea4\u4ed8\u524d\u81ea\u67e5') == 1
    assert '## 33' not in txt  # 不新增一级标题

    # 3) 回写（保持原 EOL 风格 + BOM）
    out = txt.encode('utf-8')
    if crlf > lf_only:
        # 原文件 CRLF 主：确保插入块也是 CRLF
        pass
    if bom:
        out = b'\xef\xbb\xbf' + out
    open(SK, 'wb').write(out)

    # 4) 回读复核
    chk = open(SK, 'rb').read()
    ct = chk.decode('utf-8-sig')
    print('bytes %d -> %d  (+%d)' % (len(raw), len(chk), len(chk) - len(raw)))
    print('bom', chk[:3] == b'\xef\xbb\xbf', 'crlf', ct.count('\r\n'), 'bare_lf', ct.count('\n') - ct.count('\r\n'))
    print('hdr33', ct.count(u'33 \u6761\u6536\u5c3e\u6559\u8bad') == 1)
    print('lesson33', ct.count(u'33. **\u4ea4\u4ed8\u524d\u81ea\u67e5') == 1)
    print('tail_ok', ct.rstrip().endswith(u'\u5168\u7eff\uff09') or (u'## \u53c2\u7167\u5b9e\u73b0' in ct))
    print('DONE')


if __name__ == '__main__':
    main()
