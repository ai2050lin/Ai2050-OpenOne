"""规则账本（rule ledger）：把"这一类测试"的结论固化下来，避免重复测。

问题：同一类测试（消融 top-k、聚类、探针、held-out 复现……）在日志里被反复做过很多遍，
每次都要重新读旧记录才能判断"这个是不是测过了"。

做法：一条规则 = 一条可判定的命题 + 结论状态 + 关键数字 + 证据链 + 复查守卫。
  · status          confirmed / refuted / inconclusive / superseded / open
  · retest.status   block（别再测）/ need_n（要测就得上到 n）/ allow（口径变了可以测）
  · retest.count    已被独立测过的次数（含失败与重跑）

落盘：data/rules/ledger.json（人可读、可手改、可 git diff）。
只做记账，不跑模型；写入幂等，按 id upsert。
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
SCAN = ROOT / 'tempdata' / 'rule_scan.json'

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

router = APIRouter(prefix='/api/rules', tags=['rules'])


def _blank() -> dict[str, Any]:
    return {
        'schema': 'rule_ledger.v1',
        'version': 1,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'model_scope': 'Qwen3-4B',
        'policy': ('一条规则只在给定模型 + 给定口径下成立；口径不变时不重复测。'
                   'retest.status=block 表示已有足够证据，再测属于重复劳动；'
                   'need_n 表示结论方向已知但样本不足，重测必须上到该题量才可能改变判定。'),
        'rules': [],
    }


def _load() -> dict[str, Any]:
    if not LEDGER.exists():
        return _blank()
    try:
        d = json.loads(LEDGER.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return _blank()
    if not isinstance(d, dict) or 'rules' not in d:
        return _blank()
    d.setdefault('schema', 'rule_ledger.v1')
    d.setdefault('version', 1)
    d.setdefault('updated_at', '')
    d.setdefault('policy', _blank()['policy'])
    return d


def _save(d: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    d['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    tmp = LEDGER.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
    tmp.replace(LEDGER)


# ---------- 查重 ----------

_CN = re.compile(r'[\u4e00-\u9fff]')
_EN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_\-]{1,}')
_NUM = re.compile(r'\d+(?:\.\d+)?')
_STOP = set('的 了 是 在 和 与 或 也 就 都 而 及 对 把 被 从 到 上 下 这 那 个 我 你 他 它 我们'
            ' a an the of to in on for and or is are was be it this that with as at by'.split())


def _tokens(text: str) -> set[str]:
    """中文按二元字组、英文按词、数字单独成 token，去掉停用词。够用的近似度量，不引模型。"""
    t = (text or '').lower()
    out: set[str] = set()
    for w in _EN.findall(t):
        if w not in _STOP and len(w) > 1:
            out.add(w)
    for w in _NUM.findall(t):
        out.add('#' + w)
    cn = ''.join(_CN.findall(t))
    for i in range(len(cn) - 1):
        out.add(cn[i:i + 2])
    return out


def _sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    # 余弦式：偏向"被查询覆盖了多少"，避免长条文被短查询压到很低
    return inter / math.sqrt(len(a) * len(b))


def _hay(rule: dict[str, Any]) -> tuple[set[str], set[str]]:
    """返回 (正文 token, 补充关键词 token)。分开算相似度再取大值——
    否则补关键词会把分母撑大，反而把本该命中的条目压下去。"""
    parts = [rule.get('claim', ''), rule.get('topic', ''), rule.get('why', '')]
    parts += rule.get('tags', []) or []
    for ev in rule.get('evidence', []) or []:
        parts.append(ev.get('note', ''))
    body = _tokens(' '.join(str(p) for p in parts if p))
    kw = _tokens(' '.join(str(k) for k in (rule.get('keywords') or [])))
    return body, kw


def _score(qt: set[str], rule: dict[str, Any]) -> float:
    body, kw = _hay(rule)
    # 关键词命中视为强信号：查重时人写的是"意图词"（KL / 删掉 / steering），
    # 这些词出现在补充关键词里比出现在长句 claim 里更说明问题。
    return max(_sim(qt, body), _sim(qt, kw) * 1.15)


# ---------- API ----------

class RuleIn(BaseModel):
    id: str = Field(..., min_length=3)
    claim: str = Field(..., min_length=4)
    domain: str = 'mlp'
    topic: str = ''
    status: str = 'open'
    confidence: str = 'medium'
    criterion: str = ''
    measure: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    retest: dict[str, Any] = Field(default_factory=dict)
    why: str = ''
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)


@router.get('/overview')
def overview() -> dict:
    d = _load()
    rules = d.get('rules', [])
    by_status = Counter(r.get('status', 'open') for r in rules)
    by_domain = Counter(r.get('domain', 'mlp') for r in rules)
    blocked = [r for r in rules if (r.get('retest') or {}).get('status') == 'block']
    need_n = [r for r in rules if (r.get('retest') or {}).get('status') == 'need_n']
    total_retests = sum(int((r.get('retest') or {}).get('count') or 0) for r in rules)

    scan = {}
    if SCAN.exists():
        try:
            scan = json.loads(SCAN.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            scan = {}

    return {
        'schema': 'rule_ledger_overview.v1',
        'available': bool(rules),
        'updated_at': d.get('updated_at', ''),
        'model_scope': d.get('model_scope', ''),
        'policy': d.get('policy', ''),
        'counts': {
            'total': len(rules),
            'by_status': {k: by_status.get(k, 0) for k in STATUS},
            'by_domain': dict(by_domain),
            'blocked': len(blocked),
            'need_n': len(need_n),
            'recorded_retests': total_retests,
        },
        'labels': {'status': STATUS_LABEL, 'retest': RETEST_LABEL, 'domain': DOMAIN_LABEL},
        'scan': (scan.get('meta') or {}),
        'scan_total_records': sum((v or {}).get('n_records', 0)
                                  for v in (scan.get('topics') or {}).values()),
        'scan_topics': sorted(
            [{'id': k, **{kk: vv for kk, vv in v.items() if kk != 'samples'}}
             for k, v in (scan.get('topics') or {}).items()],
            key=lambda x: -x.get('n_records', 0),
        )[:12],
        'rules': rules,
    }


@router.get('/check')
def check(q: str = '', top: int = 5) -> dict:
    """准备做新测试前先查：这件事是不是已经测过。"""
    d = _load()
    rules = d.get('rules', [])
    qt = _tokens(q)
    scored = []
    for r in rules:
        s = _score(qt, r)
        if s > 0.06:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    hits = []
    for s, r in scored[:max(1, min(top, 20))]:
        rt = r.get('retest') or {}
        hits.append({
            'id': r.get('id'),
            'claim': r.get('claim'),
            'status': r.get('status'),
            'status_label': STATUS_LABEL.get(r.get('status', ''), r.get('status')),
            'domain': r.get('domain'),
            'similarity': round(min(1.0, s), 3),
            'retest': rt,
            'retest_label': RETEST_LABEL.get(rt.get('status', ''), ''),
            'measure': r.get('measure') or {},
            'why': r.get('why', ''),
        })

    # 建议按"最相似的条目"给，而不是按"第一个 block 条目"——
    # 否则一个边缘命中会把真正该看的那条挤掉。
    kind = 'new'
    if not hits:
        advice = '账本里没有近似条目：这可能是新方向，值得测；测完请登记。'
    else:
        top1 = hits[0]
        rt1 = top1.get('retest') or {}
        if rt1.get('status') == 'block':
            kind = 'block'
            advice = ('已测过，不要重复：%s（%s，已测 %s 次）。%s'
                      % (top1['claim'], top1['status_label'], rt1.get('count', '?'), top1['why']))
        elif rt1.get('status') == 'need_n':
            kind = 'need_n'
            advice = ('方向已知但样本不足：%s（%s，已测 %s 次，重测至少要 %s 题量级）。%s'
                      % (top1['claim'], top1['status_label'], rt1.get('count', '?'),
                         rt1.get('need_n', '?'), top1['why']))
        else:
            blocked = [h for h in hits[1:] if (h.get('retest') or {}).get('status') == 'block']
            if blocked:
                kind = 'related'
                advice = ('最相近的是 %s（%s，可以重测），但它旁边 %s（%s）已经测过并给出结论：%s'
                          % (top1['claim'], top1['status_label'], blocked[0]['id'],
                             blocked[0]['status_label'], blocked[0]['why']))
            else:
                advice = ('最相近的是 %s（%s），不是禁止重测的状态：先读它，确认口径是否真的不同。'
                          % (top1['claim'], top1['status_label']))
    return {'schema': 'rule_check.v1', 'query': q, 'n_hits': len(hits),
            'advice_kind': kind, 'advice': advice, 'hits': hits}


@router.post('/upsert')
def upsert(item: RuleIn) -> dict:
    if item.status not in STATUS:
        return {'ok': False, 'error': 'status 必须是 %s' % (STATUS,)}
    d = _load()
    rules = d.setdefault('rules', [])
    body = item.model_dump()
    body.setdefault('retest', {})
    body['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    for i, r in enumerate(rules):
        if r.get('id') == body['id']:
            body['created_at'] = r.get('created_at', body['updated_at'])
            rules[i] = {**r, **body}
            _save(d)
            return {'ok': True, 'mode': 'updated', 'id': body['id'], 'total': len(rules)}
    body['created_at'] = body['updated_at']
    rules.append(body)
    _save(d)
    return {'ok': True, 'mode': 'created', 'id': body['id'], 'total': len(rules)}


@router.get('/export')
def export_md() -> dict:
    d = _load()
    rules = sorted(d.get('rules', []), key=lambda r: (r.get('domain', ''), r.get('id', '')))
    lines = ['# 规则账本（自动导出）', '',
             '> 每条规则 = 一条可判定命题 + 结论 + 证据。口径不变时不要重复测。', '']
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
            for ev in r.get('evidence', []) or []:
                lines.append('- 证据：%s ｜ `%s` ｜ `%s`'
                             % (ev.get('phase', ''), ev.get('script', ''), ev.get('data', '')))
            lines.append('')
    text = '\n'.join(lines)
    out = ROOT / 'data' / 'rules' / 'LEDGER.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8', newline='')
    return {'schema': 'rule_export.v1', 'ok': True, 'path': out.relative_to(ROOT).as_posix(),
            'n': len(rules), 'markdown': text}
