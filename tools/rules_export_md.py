"""把 data/rules/ledger.json 导出成人类可读的 LEDGER.md。

为什么单独有这个脚本：md 副本是"离线也能查"的那一份，之前只在后端 /api/rules/export
被调用时才刷新，于是很容易跟 json 脱节（历史上停更了 46 条）。这里让它不依赖服务。

标签表与 server/rule_ledger_service.py 的 STATUS_LABEL / RETEST_LABEL / DOMAIN_LABEL 保持一致。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
OUT = ROOT / 'data' / 'rules' / 'LEDGER.md'

STATUS = ('confirmed', 'refuted', 'inconclusive', 'superseded', 'open')
STATUS_LABEL = {
    'confirmed': '已证实',
    'refuted': '已否证',
    'inconclusive': '未决',
    'superseded': '已被修正',
    'open': '待测',
}
RETEST_LABEL = {
    'block': '不要重测',
    'need_n': '重测需加样本',
    'allow': '可以重测',
}
DOMAIN_LABEL = {
    'mlp': 'MLP 单元',
    'attn': '注意力头',
    'cross': '跨层 / 协同',
    'metric': '指标与对照',
    'method': '实验方法',
}


def main():
    d = json.loads(LEDGER.read_text(encoding='utf-8'))
    rules = sorted(d.get('rules', []), key=lambda r: (r.get('domain', ''), r.get('id', '')))
    lines = ['# 规则账本（自动导出）', '',
             '> 每条规则 = 一条可判定命题 + 结论 + 证据。口径不变时不要重复测。', '',
             '> 由 `python tools/rules_export_md.py` 从 `data/rules/ledger.json` 生成，不要手改。', '']
    for st in STATUS:
        grp = [r for r in rules if r.get('status') == st]
        if not grp:
            continue
        lines += ['## %s（%d）' % (STATUS_LABEL.get(st, st), len(grp)), '']
        for r in grp:
            rt = r.get('retest') or {}
            lines.append('### %s %s' % (r.get('id'), r.get('claim')))
            lines.append('')
            lines.append('- 领域：%s ｜ 置信：%s ｜ 已测 %s 次 ｜ %s'
                         % (DOMAIN_LABEL.get(r.get('domain', ''), r.get('domain')),
                            r.get('confidence', ''), rt.get('count', 0),
                            RETEST_LABEL.get(rt.get('status', ''), '')))
            if r.get('criterion'):
                lines.append('- 判据：%s' % r['criterion'])
            m = r.get('measure') or {}
            if m:
                lines.append('- 关键数字：%s' % '，'.join('%s=%s' % (k, v) for k, v in m.items()))
            if r.get('why'):
                lines.append('- 为什么：%s' % r['why'])
            if rt.get('note'):
                lines.append('- 复查：%s' % rt['note'])
            for ev in r.get('evidence', []) or []:
                lines.append('- 证据：%s ｜ `%s` ｜ `%s`'
                             % (ev.get('phase', ''), ev.get('script', ''), ev.get('data', '')))
            lines.append('')
    text = '\n'.join(lines)
    OUT.write_text(text, encoding='utf-8', newline='')
    print('写入 %s：%d 条' % (OUT, len(rules)))
    from collections import Counter
    c = Counter(r.get('status') for r in rules)
    print('  ' + '  '.join('%s=%d' % (k, c.get(k, 0)) for k in STATUS))


if __name__ == '__main__':
    main()
