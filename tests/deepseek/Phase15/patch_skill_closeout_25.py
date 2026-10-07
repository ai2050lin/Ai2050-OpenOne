# -*- coding: utf-8 -*-
"""给 rdc-phase-closeout 追加教训 25（wlog/MEMO 追加段被 bash 反引号命令替换吞字 + 回滚点验证）。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
s = io.open(P, encoding='utf-8').read()
b0 = hashlib.sha256(s.encode('utf-8')).hexdigest()
BT = chr(96)


def c(x):
    return BT + x + BT


ANCHOR = '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'
assert s.count(ANCHOR) == 1, 'anchor not unique: %d' % s.count(ANCHOR)

NEW = ('25. **wlog / MEMO 的「追加段」绝不能用 bash 内联 ' + c('python -c') + ' 写 —— 段内反引号会被 bash 做**命令替换**并静默吞掉；'
       '且必须为「写坏」预置**可验证的回滚点**（Phase 15 实证：一次看似成功的写坏）**：\n'
       '    - **症状**：把一整段中文 Markdown（含大量 ' + c('`code`') + '）塞进 ' + c('python -c "…"') + ' 的双引号里执行。'
       'bash 先做命令替换 ⇒ 每个反引号片段被替换为其「命令输出」（通常为空）⇒ 写入的段落里**所有代码标识符凭空消失**，'
       '而文件确实变大（38,233 → 41,305 B）、' + c('bare_lf == 0') + '、标题齐全 ⇒ **任何「结构类」自检都看不出来**。\n'
       '    - **检测（必做）**：对**段内**统计反引号个数（污染版 = 0，正常版 = 80）—— 建议直接做成磁盘复核项（本 Phase 加第 ' + c('I6d') + ' 项），'
       '凡是模板含反引号的追加段，都用它做长期不变量。\n'
       '    - **回滚（必做，且要先证明回滚点）**：按「段头字符串」切出前缀，**断言该前缀的 ' + c('sha256/bytes') + ' 等于本次上一步自检报出的值**'
       '（本 Phase = ' + c('8f69229b…c842c3 / 38,233 B') + '）—— 相等才说明「回滚点 = 已复核状态」，随后才能重写。'
       '若上一步没留 sha，就只能靠行数与目视，风险大。\n'
       '    - **正确做法**：长文本一律**先用编辑器 Write 成 ' + c('.py') + ' 文件**（或在字符串里用 ' + c('chr(96)') + ' 拼接反引号），'
       '再交给解释器执行；追加脚本自身把「段头」当锚点，从而天然支持幂等回滚 + 重写。\n'
       '    - **配套**：把「本次变更后的 ' + c('TOTAL checks') + ' 与 FAIL 数」写进 wlog 时，'
       '要用**现场跑出来的数**；若复核项本身被扩充（如本轮 122 → 126），须在 wlog 里写明「首跑 → 扩项后」两次数，避免数字前后矛盾。\n\n')

s = s.replace(ANCHOR, NEW + ANCHOR)
io.open(P, 'w', encoding='utf-8').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert s2 == s, 'disk readback mismatch'
print('OK sha256 before %s after %s ; bytes %d'
      % (b0[:16], hashlib.sha256(s.encode('utf-8')).hexdigest()[:16], len(s.encode('utf-8'))))
print('lesson25 present:', '25. **wlog / MEMO 的「追加段」绝不能用 bash 内联' in s2)
