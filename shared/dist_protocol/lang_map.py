# -*- coding: utf-8 -*-
"""语言图谱结构（lang_map.v1）：领域 → 子领域 → 语言项 → 词 四层，作为整个研究的基础对象。

文件（shared/dist_protocol/lang_map/）
  structure.json   人工结构：三个领域、子领域、语言项归属（引用语言目录 v3 的条目或新增条目）、横切维度、交叉归属、虚词草稿
  lang_map.json    由 build() 生成：上三层 + 每项的例句、状态、模板、词数
  words.json       由 build() 生成：词层（按语言项分组），每个词记来源与审定状态
  manifest.json    版本、生成时间与以上文件的 sha256

约束：只描述语言本身，不含任何模型信息；分类与 3D 布局是组织方式，不是模型内部结构。
词、义项与 token 分开：同形不同义的词作为不同义项挂在不同位置。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DIR = HERE / 'lang_map'
SCHEMA = 'lang_map.v1'
STATUS = ['已定义', '有例句', '有词表', '有测试模板']
LEXICONS = {
    'ENT-XF v2': ('lexicon/ENT-XF_nouns.v2.tsv', ['id', 'cat', 'cat_zh', 'super', 'zh', 'en', 'color', 'color_alt', 'size']),
    'EA-v2': ('lexicon/EA-v2_new_entities.tsv', ['id', 'cat', 'super', 'zh', 'en']),
}
COLOR_ZH = {'red': '红', 'yellow': '黄', 'green': '绿', 'blue': '蓝', 'white': '白', 'black': '黑', 'brown': '棕', 'orange': '橙',
            'purple': '紫', 'pink': '粉', 'gray': '灰', 'grey': '灰', 'transparent': '透明'}


def _digest(path: Path) -> str:
    """sha256 按 LF 换行计算：Windows 检出（CRLF）不误报。"""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _lex(name: str) -> List[Dict[str, str]]:
    rel, cols = LEXICONS[name]
    rows = []
    for line in (HERE / rel).read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        f = line.split('\t')
        if f[0] == 'id':
            continue
        rows.append(dict(zip(cols, f + [''] * (len(cols) - len(f)))))
    return rows


def _templates(structure: dict) -> Dict[str, List[str]]:
    """语言树 family_map（模板族 → 因子 → 旧语言项）换算到新 ID，再加主线材料。"""
    tree = _json(HERE / 'lang_tree.json')
    compat = structure['tree_compat']
    out: Dict[str, List[str]] = {}
    for family, factors in sorted(tree.get('family_map', {}).items()):
        for old in factors.values():
            new = compat.get(old)
            if new and family not in out.setdefault(new, []):
                out[new].append(family)
    for item, fams in structure.get('extra_templates', {}).items():
        out.setdefault(item, []).extend(f for f in fams if f not in out.get(item, []))
    return out


def _words(structure: dict) -> Dict[str, List[dict]]:
    words: Dict[str, List[dict]] = {}

    def add(item, zh, en='', note='', src='', fn=None, review='reviewed'):
        words.setdefault(item, []).append({'zh': zh, 'en': en, 'note': note, 'src': src, 'fn': fn or [], 'review': review})

    ent, eav = _lex('ENT-XF v2'), _lex('EA-v2')
    cat_zh = {r['cat']: r['cat_zh'] for r in ent}
    for r in ent:
        add('SEM.CATEGORY', r['zh'], r['en'], r['cat_zh'], 'ENT-XF v2')
    for r in eav:
        add('SEM.CATEGORY', r['zh'], r['en'], cat_zh.get(r['cat'], r['cat']), 'EA-v2')
    colors: Dict[str, List[str]] = {}
    for r in ent:
        if r['color']:
            colors.setdefault(r['color'], []).append(r['zh'])
    for c, ws in sorted(colors.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        add('SEM.ATTRIBUTE', COLOR_ZH.get(c, c), c, f"颜色 · {len(ws)} 个实体：{'、'.join(ws[:6])}{'…' if len(ws) > 6 else ''}", 'ENT-XF v2 颜色标签')
    for size, zh, en in (('S', '小', 'small'), ('L', '大', 'large')):
        n = sum(1 for r in ent if r['size'] == size)
        add('SEM.ATTRIBUTE', zh, en, f'大小 · {n} 个实体', 'ENT-XF v2 大小标签')
    fw = structure['function_words']
    for item, ws in fw['items'].items():
        for w in ws:
            add(item, w['zh'], w.get('en', ''), '', '虚词草稿', w.get('fn', []), fw.get('status', 'draft'))
    return words


def build(structure_path: Path = DIR / 'structure.json', out_dir: Path = DIR, created: str | None = None) -> dict:
    structure = _json(structure_path)
    catalog = _json(HERE / 'lang_catalog.json')
    items = {i['id']: i for d in catalog['domains'] for g in d['groups'] for i in g['items']}
    templates = _templates(structure)
    words = _words(structure)
    also = structure.get('also', {})
    domains = []
    for d in structure['domains']:
        subs = []
        for s in d['subs']:
            nodes = []
            for spec in s['items']:
                if 'catalog' in spec:
                    src = items[spec['catalog']]
                    node = {'id': spec['catalog'], 'name': src['name'], 'example': src.get('example', ''), 'origin': 'catalog'}
                else:
                    node = {'id': spec['id'], 'name': spec['name'], 'example': spec.get('example', ''), 'origin': 'new'}
                ws = words.get(node['id'], [])
                node.update({'also': also.get(node['id'], []), 'templates': templates.get(node['id'], []), 'n_words': len(ws),
                             'flags': [True, bool(node['example']), bool(ws), bool(templates.get(node['id']))]})
                nodes.append(node)
            subs.append({'id': s['id'], 'name': s['name'], 'items': nodes})
        domains.append({'id': d['id'], 'name': d['name'], 'desc': d.get('desc', ''), 'subs': subs})
    # 反向索引：哪些词表达这个功能（词的 fn 边），供详情直接显示，不必再取各词类的词层
    fn_words: Dict[str, List[dict]] = {}
    for host, ws in words.items():
        for w in ws:
            for t in w['fn']:
                fn_words.setdefault(t, []).append({'zh': w['zh'], 'item': host, 'review': w['review']})
    for d in domains:
        for s in d['subs']:
            for it in s['items']:
                it['fn_words'] = fn_words.get(it['id'], [])
    dims = [{'id': x['catalog'], 'name': items[x['catalog']]['name'], 'example': items[x['catalog']].get('example', ''),
             'templates': templates.get(x['catalog'], [])} for x in structure['dims']]
    n_items = sum(len(s['items']) for d in domains for s in d['subs'])
    doc = {'schema': SCHEMA, 'version': structure['version'], 'status': STATUS, 'domains': domains, 'dims': dims,
           'tree_compat': structure['tree_compat'],
           'stats': {'domains': len(domains), 'subs': sum(len(d['subs']) for d in domains), 'items': n_items,
                     'catalog_items': len(items), 'new_items': sum(1 for d in domains for s in d['subs'] for i in s['items'] if i['origin'] == 'new'),
                     'dims': len(dims), 'words': sum(len(v) for v in words.values()),
                     'draft_words': sum(1 for v in words.values() for w in v if w['review'] != 'reviewed')},
           'sources': {'catalog': f"lang_catalog.json v{catalog['version']}", 'lexicons': sorted(LEXICONS), 'function_words': structure['function_words']['source']},
           'note': '只描述语言本身，不含模型信息；分类与布局是组织方式，不是模型内部结构。'}
    words_doc = {'schema': SCHEMA + '.words', 'version': structure['version'], 'items': {k: words[k] for k in sorted(words)}}
    errors = validate(doc, words_doc, catalog)
    if errors:
        raise ValueError('语言图谱校验未通过：' + '；'.join(errors))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, obj in (('lang_map.json', doc), ('words.json', words_doc)):
        text = json.dumps(obj, ensure_ascii=False, indent=1) + '\n'
        (out_dir / name).write_text(text, encoding='utf-8', newline='\n')
        files[name] = hashlib.sha256(text.encode()).hexdigest()
    files['structure.json'] = _digest(structure_path)
    manifest = {'schema': SCHEMA + '.manifest', 'version': structure['version'],
                'created': created or datetime.now().isoformat(timespec='seconds'), 'files': files, 'stats': doc['stats']}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    return manifest


def validate(doc: dict, words_doc: dict, catalog: dict | None = None) -> List[str]:
    """结构校验：ID 唯一、主归属唯一、语言目录条目全部归位、交叉归属与词的功能边都指向存在的节点、词有来源。"""
    errs: List[str] = []
    ids: Dict[str, str] = {}

    def reg(i, level):
        if i in ids:
            errs.append(f'ID 重复：{i}')
        ids[i] = level

    item_ids = []
    for d in doc['domains']:
        reg(d['id'], 'domain')
        for s in d['subs']:
            reg(s['id'], 'sub')
            for it in s['items']:
                reg(it['id'], 'item')
                item_ids.append(it['id'])
    for x in doc['dims']:
        reg(x['id'], 'dim')
    for d in doc['domains']:
        for s in d['subs']:
            for it in s['items']:
                for t in it.get('also', []):
                    if ids.get(t) != 'sub':
                        errs.append(f'{it["id"]} 的交叉归属指向不存在的子领域 {t}')
                    if t == s['id']:
                        errs.append(f'{it["id"]} 的交叉归属与主归属相同')
    if catalog is not None:
        cat_ids = {i['id'] for dd in catalog['domains'] for g in dd['groups'] for i in g['items']}
        missing = sorted(cat_ids - set(ids))
        if missing:
            errs.append('语言目录条目未归位：' + '、'.join(missing))
    for old, new in doc.get('tree_compat', {}).items():
        if new not in ids:
            errs.append(f'旧语言项 {old} 映射到不存在的节点 {new}')
    for item, ws in words_doc['items'].items():
        if ids.get(item) != 'item':
            errs.append(f'词层挂在不存在的语言项 {item}')
        for w in ws:
            if not w.get('zh') or not w.get('src'):
                errs.append(f'{item} 有缺名称或来源的词')
            for t in w.get('fn', []):
                if ids.get(t) not in ('item', 'dim'):
                    errs.append(f'词「{w.get("zh")}」的功能边指向不存在的语言项 {t}')
    return errs


def load(path: Path = DIR) -> dict:
    """读已生成的语言图谱；核对 manifest 里的 sha256，不一致时报错（文件被手改但没重建）。"""
    manifest = _json(path / 'manifest.json')
    for name in ('lang_map.json', 'words.json'):
        digest = _digest(path / name)   # 换行按 LF 计，Windows 检出不误报
        if digest != manifest['files'][name]:
            raise ValueError(f'{name} 与 manifest 不一致：先运行 scripts/build_lang_map.py')
    return {'map': _json(path / 'lang_map.json'), 'words': _json(path / 'words.json'), 'manifest': manifest}
