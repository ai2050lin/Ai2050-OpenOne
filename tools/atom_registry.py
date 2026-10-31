# -*- coding: utf-8 -*-
"""P0.3 原子注册表（G5-DP01 P0.3）：四类原子同表同卡，schema v1 冻结。

四类 kind：
  subspace     rank≤10 残差流/MLP 子空间（basis: 正交基坐标列表，每项 [层, 维度] 或向量哈希引用）
  unit_group   census 单元组（units: [(block, unit), ...]）
  head_dir     注意力头读写方向（head: [layer, head]，basis: 方向基坐标）
  template_dir 模板对照方向（basis: 方向坐标 + template_ref）

特征卡字段对齐 AGENTS.md 5.3：atom_id / kind / layer / 读出边界 / method_version /
provenance / 激活统计 / 模板选择性 / 因果检验 / 跨层上下游（evidence 边列表）/ 反例 / 证据等级。
evidence 边类型沿用 AGENTS.md 5.4：observation|holdout_repl|cross_family|holdout_pred|
in_range_interv|param_path。
文件 append-only：registry vN immutable，升级开新版本文件，legacy 字节保全。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'research' / 'gpt5' / 'atlas' / 'atom_registry.v1.json'

SCHEMA = 'atom_registry.v1'
KINDS = ('subspace', 'unit_group', 'head_dir', 'template_dir')
EDGE_KINDS = ('observation', 'holdout_repl', 'cross_family', 'holdout_pred',
              'in_range_interv', 'param_path')
EVIDENCE_GRADES = ('candidate', 'confirmed', 'refuted')      # 证据等级（低→高，refuted 也入账）
CARD_REQUIRED = ('atom_id', 'kind', 'layer', 'method_version', 'provenance',
                 'activation_stats', 'template_selectivity', 'causal_tests',
                 'evidence', 'counterexamples', 'grade')


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]


def validate_atom(atom: dict) -> list:
    """返回问题清单；空 = 合法。缺字段/越界 kind 都拦下。"""
    errs = []
    if not isinstance(atom, dict):
        return ['atom 不是 dict']
    for k in CARD_REQUIRED:
        if k not in atom:
            errs.append(f'缺必填字段 {k}')
    if atom.get('kind') not in KINDS:
        errs.append(f"kind 非法：{atom.get('kind')!r}（须∈{KINDS}）")
    if atom.get('grade') not in EVIDENCE_GRADES:
        errs.append(f"grade 非法：{atom.get('grade')!r}（须∈{EVIDENCE_GRADES}）")
    for e in atom.get('evidence') or []:
        if not isinstance(e, dict) or e.get('kind') not in EDGE_KINDS:
            errs.append(f'evidence 边缺 kind 或非法：{e!r}')
            break
    prov = atom.get('provenance') or {}
    for k in ('git_sha', 'model_revision'):
        if not prov.get(k) or prov.get(k) == 'unknown':
            errs.append(f'provenance.{k} 缺失或 unknown')
    # kind 特有字段
    if atom.get('kind') in ('subspace', 'head_dir', 'template_dir') and not atom.get('basis'):
        errs.append(f"{atom.get('kind')} 需 basis（方向基坐标）")
    if atom.get('kind') == 'unit_group' and not atom.get('units'):
        errs.append('unit_group 需 units（[(block, unit), ...]）')
    return errs


def load(path: Path = None) -> dict:
    p = Path(path or REGISTRY)
    if not p.exists():
        return {'schema': SCHEMA, 'atoms': []}
    return json.loads(p.read_text(encoding='utf-8'))


def query(kind: str = None, layer=None, path: Path = None) -> list:
    reg = load(path)
    out = reg['atoms']
    if kind is not None:
        out = [a for a in out if a.get('kind') == kind]
    if layer is not None:
        out = [a for a in out if a.get('layer') == layer]
    return out


def append(atom: dict, path: Path = None) -> dict:
    """校验→查重→追加→整册 sha8 重算。返回 {'ok', 'errors'|'sha8', 'n_atoms'}。"""
    p = Path(path or REGISTRY)
    errs = validate_atom(atom)
    if errs:
        return {'ok': False, 'errors': errs}
    reg = load(p)
    reg.setdefault('atoms', [])
    if any(a.get('atom_id') == atom['atom_id'] for a in reg['atoms']):
        return {'ok': False, 'errors': [f"atom_id 已存在：{atom['atom_id']}（append-only，换 ID 或升级版本）"]}
    reg['atoms'].append(atom)
    reg['n_atoms'] = len(reg['atoms'])
    reg['sha8'] = _sha8(json.dumps(reg, ensure_ascii=False, sort_keys=True))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, ensure_ascii=False, indent=1) + '\n',
                 encoding='utf-8', newline='')
    return {'ok': True, 'sha8': reg['sha8'], 'n_atoms': reg['n_atoms']}
