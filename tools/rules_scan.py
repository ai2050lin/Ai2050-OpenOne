"""扫描记忆日志与设计文档，抽取"测试类"记录并按主题去重计数。

目的：为规则账本（rule ledger）提供导入素材，同时量化"同一类测试被重复做了多少遍"。
输出：tempdata/rule_scan.json + 终端摘要。
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEM = sorted(ROOT.glob('.workbuddy/memory/2026-*.md'))
DESIGN = [p for p in (ROOT / 'design').glob('*.md')]

# 主题词库：(主题 id, 显示名, 匹配正则)
TOPICS = [
    ('ablate_topk', '消融 top-k 单元', r'消融|ablat|置零|删\s*top|top-?\s*1%|top97|删随机'),
    ('ablate_layer', '整层消融', r'整层|full\s*ablat|全置零|删\s*100%'),
    ('cluster', '单元共现聚类', r'聚类|cluster|KMeans|共现'),
    ('head_type', '注意力头分型', r'头类型|induction|prev-token|induction头|头分型'),
    ('dla', 'DLA 直接对数归因', r'DLA|direct logit|logit attribution'),
    ('pos_readout', '位置读出（q_end/ans）', r'q_end|opt_end|答案位|读出位'),
    ('heldout', '跨数据集 held-out 复现', r'held-?out|跨数据集|CMMLU|复现'),
    ('gain', '读取增益 / 写方向几何', r'读取增益|gain|写方向|ov_pr|qk_pr|W_K|W_V'),
    ('norm', '范数 / 常数平移', r'范数|norm|常数平移|中心化'),
    ('steering', 'steering / 方向干预', r'steering|方向干预|加方向'),
    ('sae', 'SAE / 稀疏自编码器', r'\bSAE\b|稀疏自编码|sparse autoencoder'),
    ('probe', '线性探针', r'探针|probe|linear probe'),
    ('cot', '生成任务 / CoT 行为', r'生成任务|CoT|chain.of.thought|推理过程'),
    ('attn_pattern', '注意力模式统计', r'注意力模式|attention pattern|注意力质量|注意力熵'),
    ('circuit', '跨层电路 / 归因图', r'跨层电路|circuit|归因图|attribution graph'),
    ('template', '模板对照实验', r'模板对照|模板选择'),
    ('bench', '基准正确率实测', r'C-Eval|CMMLU|bench|正确率实测'),
]

# 结论信号
SIG = [
    ('refuted', r'FAIL|失败|否证|不成立|被推翻|未通过|作废|撤回|否定'),
    ('confirmed', r'PASS|通过|成立|证实|命中|确认'),
    ('inconclusive', r'待定|未决|不足以|样本不足|不显著|BLOCKED|挂账'),
]

LINE = re.compile(r'^[ \t]*(?:[-*·]|\d+\.)\s+')


def scan(paths, label):
    hits = defaultdict(list)
    total_lines = 0
    for p in paths:
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        for ln in text.splitlines():
            if len(ln.strip()) < 12:
                continue
            total_lines += 1
            for tid, tname, pat in TOPICS:
                if not re.search(pat, ln, re.I):
                    continue
                sig = None
                for s, spat in SIG:
                    if re.search(spat, ln):
                        sig = s
                        break
                hits[tid].append({
                    'src': p.name,
                    'date': p.stem if p.suffix == '.md' and p.stem[0].isdigit() else '',
                    'signal': sig,
                    'line': ln.strip()[:400],
                })
    return hits, total_lines


def main():
    mem_hits, mem_lines = scan(MEM, 'memory')
    des_hits, des_lines = scan(DESIGN, 'design')

    report = {'topics': {}, 'meta': {
        'memory_files': len(MEM), 'memory_lines': mem_lines,
        'design_files': len(DESIGN), 'design_lines': des_lines,
    }}
    print('扫描：memory %d 文件 / %d 行；design %d 文件 / %d 行'
          % (len(MEM), mem_lines, len(DESIGN), des_lines))
    print()
    print('%-22s %8s %8s %8s %8s %8s' % ('主题', '记录数', '否证', '证实', '未决', '涉及天数'))
    rows = []
    for tid, tname, _ in TOPICS:
        m = mem_hits.get(tid, [])
        d = des_hits.get(tid, [])
        allh = m + d
        if not allh:
            continue
        sig = Counter(h['signal'] for h in allh)
        days = len({h['date'] for h in allh if h['date']})
        rows.append((tid, tname, len(allh), sig['refuted'], sig['confirmed'], sig['inconclusive'], days))
        report['topics'][tid] = {
            'name': tname, 'n_records': len(allh), 'n_memory': len(m), 'n_design': len(d),
            'refuted': sig['refuted'], 'confirmed': sig['confirmed'], 'inconclusive': sig['inconclusive'],
            'days': days,
            'samples': allh[:6],
        }
    rows.sort(key=lambda r: -r[2])
    for r in rows:
        print('%-22s %8d %8d %8d %8d %8d' % (r[1], r[2], r[3], r[4], r[5], r[6]))
    print()
    print('合计记录 %d 条，其中带明确结论信号 %d 条'
          % (sum(r[2] for r in rows),
             sum(r[3] + r[4] + r[5] for r in rows)))

    out = ROOT / 'tempdata' / 'rule_scan.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
