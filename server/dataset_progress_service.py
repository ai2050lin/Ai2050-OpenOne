# -*- coding: utf-8 -*-
"""数据集进度（dataset_progress.v1）：围绕每个数据集，汇总两条研究线的阶段进度与可贡献的计算。

两条线（设计见 design/attention_neuron_plan_v1.md）：
  神经元逆向（Goodfire 式）  N1 采集 → N2 响应画像 → N3 四轴检验 → N4 因果使用 → N5 入图谱
  注意力语言图谱              A1 行为门 → A2 可读出 → A3 注意力路由 → A4 优先级与层级 → A5 保真度

每个阶段的状态只从已有产物读出，本模块不计算新的统计量、不改写任何数字：
  done 完成 / partial 部分 / negative 已检验未支持 / todo 未开始 / na 不适用 / unavailable 数据源不可读。
「完成」只表示该阶段的产物存在且判据有结论，不是证据等级；未支持的检验与通过的检验同样记录。

数据源（全部是现有端点的实现或研究文件）：
  模式族：distributed_service.api_catalog（行为门、分片）、platform_service 语言谱图（可读出等级与逐层曲线）、
          research/neurons/*.json（neurons.v1 模板选择性）、research/interventions/INT-N-*（神经元干预）、
          research/transcoder/GQ-ATOM-*（L30 原子必要性）；
  自然语料：普查运行、逐层摘要（neuron_tower）、EXPL-v1/v2 解释检验、research/neurons/goodfire/GF-*、
          L30-v1/v2/v3 归因与消融、research/transcoder/TC-*（字典原子对照）、普查作业（可贡献分片）；
  冲突材料：research/mechanism/<案例>/report.json（mechanism_service）；
  考试题库：lang_bench_service（只在本机中心可读）。

端点：GET /api/progress/datasets（进主数据包，见 deploy/datapack.py）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter

from server.neuron_census_service import research_root

router = APIRouter(prefix='/api/progress', tags=['progress'])

SCHEMA = 'dataset_progress.v1'
STAGES = {
    'neuron': [
        {'id': 'N1', 'label': '采集', 'desc': '该数据集上的 MLP 中间单元激活已采集（普查或模板逐位置采集）'},
        {'id': 'N2', 'label': '响应画像', 'desc': '单元对语料或因子的响应：分型、模板选择性，留出复现'},
        {'id': 'N3', 'label': '四轴检验', 'desc': 'Goodfire 式逐单元检验（可读 / 读出 / 控制 / 连接）或解释检验'},
        {'id': 'N4', 'label': '因果使用', 'desc': '消融、归因或干预显示单元被模型使用'},
        {'id': 'N5', 'label': '入图谱', 'desc': '单元卡经模板确认后连到语言项（路线 N 与两路线会合）'},
    ],
    'attention': [
        {'id': 'A1', 'label': '行为门', 'desc': '模型能完成任务的条件格；未过门的格不进内部分析'},
        {'id': 'A2', 'label': '可读出', 'desc': '逐层读出：语言变量在哪些层可从残差读出（固定判据的等级）'},
        {'id': 'A3', 'label': '注意力路由', 'desc': '逐头直接作用与总效应并报，路径修补区分搬运与上游'},
        {'id': 'A4', 'label': '优先级与层级', 'desc': '冲突时谁赢、在哪层决出、哪些线索翻转；上下游干预测层级'},
        {'id': 'A5', 'label': '保真度', 'desc': '只保留图谱成分时，留出材料上答案差值 / 损失的保留比例'},
    ],
}
STATUS = {'done': '✓ 完成', 'partial': '◐ 部分', 'negative': '✗ 未支持', 'todo': '○ 未开始', 'na': '— 不适用',
          'unavailable': '? 数据源不可读'}
MAIN_MODEL = 'qwen3-4b'


def _cell(status: str, value: str = '', detail: str = '', link: Optional[Dict[str, str]] = None, sources: Optional[List[str]] = None):
    c = {'status': status, 'value': value}
    if detail:
        c['detail'] = detail
    if link:
        c['link'] = link
    if sources:
        c['sources'] = sources
    return c


def _safe(fn: Callable[[], Any], default=None):
    try:
        return fn()
    except Exception:                                              # 任何数据源不可读都只影响对应格
        return default


def _json(rel: str) -> Optional[Dict[str, Any]]:
    p = research_root() / rel
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _short(bucket: str) -> str:
    return (bucket or '').split('|')[0]


# ---------------- 最小对模板（模式族）----------------

def _lang_pieces() -> Dict[str, List[Dict[str, Any]]]:
    """tm_id → 语言谱图里该模板的路线 T piece（含逐层可读出曲线）。"""
    import platform_service as PS
    tree = PS.lang_tree_ep()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for it in tree.get('items', []):
        if not (it.get('status') or {}).get('pieces_T'):
            continue
        doc = _safe(lambda i=it['id']: PS.lang_item_ep(i), {}) or {}
        for p in doc.get('pieces', []):
            if p.get('route') == 'T' and not p.get('synthetic'):
                out.setdefault(p.get('tm_id'), []).append({**p, 'item_name': it.get('name'), 'item_id': it.get('id')})
    return out


def _selectivity() -> Dict[str, List[Dict[str, Any]]]:
    """tm_id → neurons.v1 选择性结果（只取摘要：逐块选中单元数）。"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted((research_root() / 'neurons').glob('*.json')):
        doc = _safe(lambda q=p: json.loads(q.read_text(encoding='utf-8')))
        if not doc or doc.get('schema') != 'neurons.v1':
            continue
        summ = doc.get('summary') or {}
        by = {ro: [int(v.get('n_selected_by_block', {}).get(str(b), 0)) for b in range(len(v.get('n_selected_by_block', {})))]
              for ro, v in summ.items() if isinstance(v, dict)}
        out.setdefault(doc.get('tm_id'), []).append({'file': p.name, 'model_key': doc.get('model_key'),
                                                     'version': doc.get('version'), 'factor': doc.get('factor'), 'by_readout': by})
    return out


def _template_neuron_causal(tm_id: str) -> List[Dict[str, Any]]:
    """模板单元的因果检验（研究文件；按模板 id 关联）。"""
    rows = []
    if tm_id == 'TG-GR-Q':
        r = _json('interventions/INT-N-v1-qwen3-4b.json')
        if r and r.get('evaluation'):
            segs = r['evaluation'].get('segments', {})
            passed = [s for s, d in segs.items() if (d.get('necessity_primary') or {}).get('verdict') == '通过']
            rows.append({'id': 'INT-N-v1', 'ok': bool(passed),
                         'text': f"INT-N-v1（qwen3-4b，句末替换语气单元）：必要性主检验通过的层段 {'、'.join(passed) or '无'}"})
        g = _json('transcoder/GQ-ATOM-v1/report.json')
        if g and g.get('verdict'):
            v = g['verdict']
            rows.append({'id': 'GQ-ATOM-v1', 'ok': bool(v.get('feature_used') or v.get('neuron_used')),
                         'text': f"GQ-ATOM-v1（第 {g.get('layer')} 层前 10 单元消融）：特征{'被' if v.get('feature_used') else '未被'}使用，"
                                 f"神经元{'被' if v.get('neuron_used') else '未被'}使用"})
    return rows


def _template_rows(catalog: Dict[str, Any], pieces, sel) -> List[Dict[str, Any]]:
    rows = []
    for it in catalog.get('items', []):
        if it.get('module') == 'legacy' or not str(it.get('tm_id', '')).startswith('TG-'):
            continue
        tm = it['tm_id']
        real = [m for m in it.get('models', []) if not m.get('synthetic')]
        ds = {'id': f"tm:{it['id']}", 'kind': 'template', 'name': it.get('name') or it['id'], 'family_id': it['id'], 'tm_id': tm,
              'group': it.get('category'), 'models': [_short(m['bucket']) for m in real], 'neuron': {}, 'attention': {}, 'layers': {}}
        # A1 行为门
        if real:
            parts = [f"{_short(m['bucket'])} {m.get('behavior_pass', 0)}/{m.get('behavior_cells', 0)} 格" for m in real if m.get('behavior_cells')]
            ok = any(m.get('behavior_pass') for m in real)
            ds['attention']['A1'] = _cell('done' if ok else 'negative', '；'.join(parts) or '已采集',
                                          '过门格数 / 条件格数（逐模型）', {'lens': 'datasets', 'sub': 'catalog'})
        else:
            ds['attention']['A1'] = _cell('todo', '还没有真实运行结果')
        # A2 可读出（语言谱图的路线 T piece）
        ps = pieces.get(tm, [])
        if ps:
            best = max(ps, key=lambda p: p.get('level', 0))
            lv = {}
            for p in ps:
                lv[_short(p['model'])] = max(lv.get(_short(p['model']), 0), p.get('level', 0))
            st = 'done' if best.get('level', 0) >= 3 else 'partial' if best.get('level', 0) >= 2 else 'negative'
            ds['attention']['A2'] = _cell(st, '；'.join(f'{m} 等级 {v}' for m, v in sorted(lv.items())),
                                          f"语言项：{'、'.join(sorted({p['item_name'] for p in ps if p.get('item_name')}))}。等级只描述可读出证据",
                                          {'lens': 'datasets', 'sub': 'lang', 'item': best.get('item_id')})
            main = next((p for p in ps if _short(p['model']) == MAIN_MODEL), best)
            ro = (main.get('readouts') or {}).get(main.get('readout')) or {}
            curve = ro.get('curve') or []
            if curve:
                ds['layers']['attention'] = {'label': f"逐层可读出间隔（{_short(main['model'])}，读出位置 {main.get('readout')}，因子 {main.get('factor')}）",
                                             'kind': 'sequential', 'values': [None if c.get('sep') is None else round(float(c['sep']), 3) for c in curve],
                                             'marks': [int(c['layer']) for c in curve if c.get('readable')],
                                             'note': '残差读出（hs[l]），不是注意力头的贡献；标记 = 达到可读出判据的层'}
        else:
            ds['attention']['A2'] = _cell('todo' if real else 'na', '没有规律分析缓存' if real else '')
        ds['attention']['A3'] = _cell('todo', '模板材料上还没有逐头路由分析', '注意力路由目前只在冲突材料上做（CF-v1）')
        ds['attention']['A4'] = _cell('todo', '还没有设计冲突或优先级对照')
        ds['attention']['A5'] = _cell('todo', '图谱保真度未接入')
        # 神经元线
        ss = sel.get(tm, [])
        if ss:
            ds['neuron']['N1'] = _cell('done', f'{len(ss)} 份逐位置 MLP 单元采集（{len({s["model_key"] for s in ss})} 个模型）',
                                       '采集只在节点本地保存原生全场，摘要进研究文件')
            main = next((s for s in ss if MAIN_MODEL in s['file']), ss[0])
            ro_name, by = next(iter(main['by_readout'].items()), (None, []))
            tot = sum(by)
            ds['neuron']['N2'] = _cell('done' if tot else 'negative', f'{tot:,} 个单元通过选择性判据（{main["file"]}，读出 {ro_name}）',
                                       'discovery 拟合、留出同号复现、按取值单位置换的族错误阈值', {'lens': 'neurons', 'sub': 'atlas'})
            if by:
                ds['layers']['neuron'] = {'label': f'逐层选择性单元数（{main["file"]}，读出 {ro_name}）', 'kind': 'sequential',
                                          'values': by, 'note': '每层 9,728 个 MLP 中间单元中通过判据的个数'}
        else:
            ds['neuron']['N1'] = _cell('todo', '文法未声明 mlp_units，或还没有逐位置采集')
            ds['neuron']['N2'] = _cell('todo', '')
        ds['neuron']['N3'] = _cell('todo', '还没有对该模板的单元跑四轴检验')
        causal = _template_neuron_causal(tm)
        if causal:
            ok = [c for c in causal if c['ok']]
            ds['neuron']['N4'] = _cell('partial' if ok else 'negative', f"{len(ok)} / {len(causal)} 项检验显示被使用",
                                       '；'.join(c['text'] for c in causal), sources=[c['id'] for c in causal])
        else:
            ds['neuron']['N4'] = _cell('todo', '')
        ds['neuron']['N5'] = _cell('todo', '路线 N（神经元经模板确认后入树）尚未实现')
        # 贡献
        status = it.get('status')
        ds['contribute'] = {'kind': 'template', 'status': status, 'status_label': it.get('status_label'),
                            'tasks_active': it.get('tasks_active', 0), 'claim_command': it.get('claim_command'),
                            'open': status in ('open', 'running'),
                            'note': '冻结模板的分片按「该模型尚无结果的分片」优先派发；节点领取后自动执行'}
        rows.append(ds)
    return rows


# ---------------- 自然语料（神经元普查）----------------

def _corpus_rows() -> List[Dict[str, Any]]:
    from server import neuron_census_service as NS
    runs = _safe(NS.census_runs, []) or []
    if not runs:
        return []
    import census_service as CS
    jobs = (_safe(CS.jobs_list, {}) or {}).get('jobs', [])
    rows = []
    for fp in sorted({r['corpus_fp'] for r in runs}):
        rr = [r for r in runs if r['corpus_fp'] == fp]
        big = max(rr, key=lambda r: r.get('n_tokens', 0))
        mk = big['model_key']
        srcs = '、'.join(s['source'] for s in big.get('sources', []))
        ds = {'id': f'corpus:{fp}', 'kind': 'corpus', 'name': f'普查自然语料 {fp}', 'desc': srcs, 'models': sorted({r['model_key'] for r in rr}),
              'neuron': {}, 'attention': {}, 'layers': {}}
        ds['neuron']['N1'] = _cell('done', f"{big.get('n_tokens', 0):,} 个位置 × {big.get('n_layers')} 层 × {big.get('d_ff'):,} 单元",
                                   f'普查运行 {big["run"]}（{len(rr)} 次）', {'lens': 'neurons', 'sub': 'atlas'})
        tower = _safe(lambda: NS.neuron_tower(model_key=mk, run=big['run']))
        if tower and tower.get('summaries_ready'):
            n = tower['summaries_ready']
            ds['neuron']['N2'] = _cell('done' if n == tower.get('n_layers') else 'partial',
                                       f"{n} / {tower.get('n_layers')} 层逐单元分型与读出摘要",
                                       '分型沿用 L30-v1 预注册规则；其他块是同一规则的描述性应用', {'lens': 'neurons', 'sub': 'atlas'})
            tested = [sum((b.get('explained') or {}).values()) for b in tower.get('blocks', [])]
            ok = [sum(v for k, v in (b.get('explained') or {}).items() if k != '未解释') for b in tower.get('blocks', [])]
            if any(tested):
                ds['layers']['neuron'] = {'label': '逐层做过解释检验的单元数（深色 = 已解释）', 'kind': 'progress',
                                          'values': tested, 'values_ok': ok, 'note': 'EXPL-v1 / v2；每层 9,728 个单元中只抽检了少数'}
        else:
            ds['neuron']['N2'] = _cell('todo', '逐层摘要还没有生成（后台子进程）')
        # N3：解释检验 + Goodfire 四轴
        e1, e2 = _json('interventions/EXPL-v1-results.json'), _json('interventions/EXPL-v2-results.json')
        gf = []
        for p in sorted((research_root() / 'neurons' / 'goodfire').glob('GF-*.json')):
            d = _safe(lambda q=p: json.loads(q.read_text(encoding='utf-8')))
            if d and d.get('verdict'):
                gf.append((p.stem, d['verdict']))
        parts = []
        if e1 and e1.get('summary'):
            parts.append(f"EXPL-v1：{e1['summary'].get('n_units')} 个单元，解释可输 {e1['summary'].get('win_share', 0):.0%}")
        if e2 and e2.get('summary'):
            parts.append(f"EXPL-v2：中层可解释 {e2['summary'].get('B_explained_share', 0):.0%}")
        if gf:
            parts.append(f"Goodfire 四轴：{len(gf)} 个单元，全部通过 {sum(v.get('status') == 'passed' for _, v in gf)} 个")
        tc = _json('transcoder/TC-v1-qwen3-4b-L30/report.json')
        if tc and tc.get('atom_compare'):
            fw = any(v.get('feature_wins') for v in tc['atom_compare'].values())
            parts.append(f"TC-v1 原子对照（第 {tc.get('layer')} 层）：字典特征{'胜出' if fw else '未胜出'}，同 k 下神经元损失恢复更高" if not fw else
                         f"TC-v1 原子对照（第 {tc.get('layer')} 层）：字典特征胜出")
        ds['neuron']['N3'] = _cell('partial' if parts else 'todo', '；'.join(parts) or '还没有四轴或解释检验',
                                   'Goodfire 式检验在桌面客户端「神经元图谱 → 本机测试」运行，结果只存本机',
                                   {'lens': 'neurons', 'sub': 'probe'}, [s for s, _ in gf])
        # N4：归因与消融
        c4 = []
        v2 = _json('neurons/L30-v2-conditional.json')
        if v2:
            c4.append(('L30-v2 条件化消融', v2.get('verdict') == '支持'))
        v3 = _json('neurons/L30-v3-trigger.json')
        if v3:
            c4.append((f"L30-v3 触发点消融主检验（{(v3.get('primary') or {}).get('n_evaluable')} 个单元）", (v3.get('primary') or {}).get('verdict') == '支持'))
            c4.append(('L30-v3 全部触发点', (v3.get('secondary') or {}).get('verdict') == '支持'))
        if c4:
            ok = sum(1 for _, v in c4 if v)
            ds['neuron']['N4'] = _cell('partial' if ok else 'negative', f'{ok} / {len(c4)} 项支持',
                                       '；'.join(f"{n}：{'支持' if v else '未支持'}" for n, v in c4))
        else:
            ds['neuron']['N4'] = _cell('todo', '')
        ds['neuron']['N5'] = _cell('todo', '路线 N 与两路线会合尚未实现')
        for s in STAGES['attention']:
            ds['attention'][s['id']] = _cell('na', '', '自然语料不设行为门与冲突；注意力图谱在模板与冲突材料上做')
        js = [j for j in jobs if j.get('corpus_fp') == fp]
        cnt: Dict[str, int] = {}
        for j in js:
            for k, v in (j.get('counts') or {}).items():
                cnt[k] = cnt.get(k, 0) + int(v)
        ds['contribute'] = {'kind': 'census', 'jobs': [j['job_id'] for j in js], 'counts': cnt,
                            'open': bool(cnt.get('queued') or cnt.get('replica_pending')),
                            'note': '普查分片 = token 切片 × 层组；中心按切片重算校验，抽样派给另一个账号的节点复算'}
        rows.append(ds)
    return rows


# ---------------- 冲突材料（机制案例）----------------

def _conflict_rows() -> List[Dict[str, Any]]:
    from server import mechanism_service as MS
    rows = []
    for c in (MS.index().get('cases') or []):
        if c.get('kind') != 'mechanism' or c.get('error'):
            continue
        doc = (_safe(lambda i=c['id']: MS.case(i)) or {}).get('report') or {}
        if not doc.get('priority_table'):
            continue
        link = {'lens': 'modes'}
        ds = {'id': f"case:{c['id']}", 'kind': 'conflict', 'name': f"冲突材料 {doc.get('set', c.get('set'))}", 'case': c['id'],
              'desc': '上下文与记忆冲突：先给反事实陈述，再问同一事实', 'models': [Path(str(doc.get('model', ''))).name],
              'neuron': {}, 'attention': {}, 'layers': {}}
        ds['attention']['A1'] = _cell('done', f"{doc.get('n_gated')} / {doc.get('n_facts')} 个事实过门",
                                      f"发现 {doc.get('n_discovery')} / 留出 {doc.get('n_heldout')}", {'lens': 'datasets', 'sub': 'overview'})
        pt = doc['priority_table']
        plain = pt.get('plain') or next(iter(pt.values()))
        ds['attention']['A2'] = _cell('done', f"决出层中位数 {plain.get('decisive_layer_median')}（直接陈述）",
                                      '逐层 logit lens 赛跑曲线：上下文答案 − 记忆答案', link)
        heads = doc.get('heads') or {}
        ds['attention']['A3'] = _cell('done', f"1,152 个头逐头直接作用 + 总效应；前 20 重叠 {heads.get('overlap_top_dla_vs_total')} 个",
                                      '两种排序几乎不重叠：直接写入的头在后层，总效应大的头在中层上游', {'lens': 'sentence', 'sub': 'case'})
        v2 = doc.get('v2') or {}
        hier = v2.get('hierarchy') or {}
        checks = c.get('checks') or {}
        n_ok = sum(1 for v in checks.values() if v)
        wins = '、'.join(f"{k} {v.get('context_win_rate', 0):.0%}" for k, v in pt.items())
        ds['attention']['A4'] = _cell('done' if checks and n_ok == len(checks) else 'partial' if n_ok else 'negative',
                                      f"{n_ok} / {len(checks)} 条预注册判据通过；上游 → 写入头 直接作用下降 {hier.get('drop', 0):.0%}" if hier else f'{n_ok} / {len(checks)} 条判据通过',
                                      f'上下文胜率：{wins}', link)
        ds['attention']['A5'] = _cell('todo', '头电路保真度 v1 判据失效；图谱保真度未接入', c.get('fidelity_void') or '')
        neu = doc.get('neurons') or {}
        rep = neu.get('sign_replication_heldout')
        ds['neuron']['N1'] = _cell('done', '最后位置全部 MLP 单元的直接作用', '只在读出位置采集；跨位置的单元作用未采集')
        ds['neuron']['N2'] = _cell('negative' if rep is not None and rep < 0.6 else 'done',
                                   f"直接作用前 {len(neu.get('top', []))} 个单元留出同号 {rep:.0%}" if rep is not None else '',
                                   '记忆答案因事实而异，全局排序的单元不跨事实复现；需要按事实做单元卡', {'lens': 'neurons', 'sub': 'priority'})
        ds['neuron']['N3'] = _cell('todo', '还没有对这些单元跑四轴检验')
        ds['neuron']['N4'] = _cell('todo', '')
        ds['neuron']['N5'] = _cell('todo', '')
        share = doc.get('component_share_plain') or {}
        if neu.get('attn_total_by_layer') and neu.get('mlp_total_by_layer'):
            ds['layers']['attention'] = {'label': '逐层注意力头直接作用合计（正 = 推向上下文答案）', 'kind': 'diverging',
                                         'values': [round(v, 3) for v in neu['attn_total_by_layer']],
                                         'note': f"直接陈述；注意力合计 {share.get('attention', 0):+.2f}"}
            ds['layers']['neuron'] = {'label': '逐层 MLP 直接作用合计（负 = 推向记忆答案）', 'kind': 'diverging',
                                      'values': [round(v, 3) for v in neu['mlp_total_by_layer']],
                                      'note': f"直接陈述；MLP 合计 {share.get('mlp', 0):+.2f}"}
        ds['contribute'] = {'kind': 'none', 'open': False,
                            'note': '冲突案例目前由研究脚本在本机运行（scripts/conflict_case.py），还没有拆成分布式任务'}
        rows.append(ds)
    return rows


# ---------------- 考试题库（行为广度）----------------

def _exam_rows() -> List[Dict[str, Any]]:
    try:
        from server import lang_bench_service as LB
    except Exception:
        return []
    ov = _safe(LB.overview)
    if not ov or not ov.get('available'):
        return []
    meta = ov.get('meta') or {}
    last = ov.get('last_result') or {}
    ds = {'id': 'exam:lang_bench', 'kind': 'exam', 'name': '考试题库（行为广度）',
          'desc': ' + '.join((meta.get('by_source') or {}).keys()) or 'C-Eval', 'models': [last['model_name']] if last.get('model_name') else [],
          'neuron': {}, 'attention': {}, 'layers': {}}
    ds['attention']['A1'] = _cell('done' if last else 'todo',
                                  f"{last.get('n_questions')} 题，正确率 {last.get('acc', 0):.1%}" if last else f"{meta.get('n_questions', 0):,} 题，还没有实测",
                                  '只做行为广度与消融后的分科退化画像；选项字母会污染读出，不作机制材料', {'lens': 'datasets', 'sub': 'overview'})
    for s in STAGES['attention'][1:]:
        ds['attention'][s['id']] = _cell('na', '', '考试题不作机制材料')
    for s in STAGES['neuron']:
        ds['neuron'][s['id']] = _cell('na', '', '考试题不作机制材料')
    ds['contribute'] = {'kind': 'none', 'open': False, 'note': '题库实测在本机中心运行'}
    return [ds]


def build() -> Dict[str, Any]:
    import distributed_service as D
    catalog = _safe(D.api_catalog, {}) or {}
    pieces = _safe(_lang_pieces, {}) or {}
    sel = _safe(_selectivity, {}) or {}
    datasets = (_safe(_conflict_rows, []) or []) + (_safe(lambda: _template_rows(catalog, pieces, sel), []) or []) \
        + (_safe(_corpus_rows, []) or []) + (_safe(_exam_rows, []) or [])
    totals = {}
    for track in STAGES:
        totals[track] = {s['id']: {k: sum(1 for d in datasets if (d[track].get(s['id']) or {}).get('status') == k) for k in STATUS}
                         for s in STAGES[track]}
    return {'schema': SCHEMA, 'stages': STAGES, 'status_labels': STATUS, 'datasets': datasets, 'totals': totals,
            'note': '状态只描述产物是否存在、判据有何结论，不是证据等级；未支持的检验与通过的同样列出。'}


@router.get('/datasets')
def datasets_ep():
    return build()
