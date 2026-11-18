"""Build one public status release from actual results; never read sealed samples.

Run before building the client/center. A subsequent publication uses the previous
state's digest as its parent, and can be sent to POST /api/research/mainline.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol.mainline_state import RELEASE, canonical, digest, validate

SNAPSHOT = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'


def build():
    import numpy as np
    previous_bundle = json.loads(RELEASE.read_text(encoding='utf-8'))
    previous = previous_bundle['state']
    v = copy.deepcopy(previous['view'])
    artifacts = {}

    def add(doc):
        sha = digest(doc)
        artifacts[sha] = doc
        return sha

    def proof(path, run_id):
        raw = (ROOT / path).read_bytes()
        doc = json.loads(raw)
        ref = add({'schema': 'mainline_run_report.v1', 'kind': 'real', 'run_id': run_id,
                   'model_snapshot': SNAPSHOT, 'source_path': path,
                   'source_sha256': hashlib.sha256(raw).hexdigest(), 'report': doc})
        return doc, ref

    h3, h3_ref = proof('research/mainline/H3/confirm.json', 'MA-H3-v4')
    frozen, frozen_ref = proof('research/mainline/H3/frozen.json', 'MA-H3-v4')
    hb, hb_ref = proof('research/mainline/H3b/evalfit_h3b.json', 'MA-H3b-v1')
    hb_freeze, hb_freeze_ref = proof('research/mainline/H3b/frozen_h3b.json', 'MA-H3b-v1')
    if hashlib.sha256((ROOT / 'research/mainline/H3/frozen.json').read_bytes()).hexdigest() != h3['frozen_sha256']:
        raise ValueError('H3 confirmation and frozen carrier differ')
    splits = json.loads((ROOT / 'research/mainline/H1/splits.json').read_text(encoding='utf-8'))
    events = [{k: e[k] for k in ('split', 'when', 'purpose', 'frozen_before')} for e in splits['custody']['opened']]
    opening = next(e for e in events if e['split'] == 'confirm')
    if opening['frozen_before'] != h3['frozen_sha256']:
        raise ValueError('opening event and frozen carrier differ')
    custody_ref = add({'schema': 'mainline_custody.v1', 'kind': 'real', 'run_id': 'MA-H1',
                       'model_snapshot': SNAPSHOT, 'events': events, 'phase': 343,
                       'source_sha256': hashlib.sha256((ROOT / 'research/mainline/H1/splits.json').read_bytes()).hexdigest()})
    f = h3['P1_sufficiency']['f_C']
    best = max(h3['P1_sufficiency']['controls'].values())
    keep = max(hb['curves']['exact']['f_keep'])
    remove = hb['curves']['exact']['f_remove'][3]
    v['STAGES'][1]['note'] = 'MA-H1：252 个实体按家族分为 84 / 84 / 84。冻结时已封存；confirm 已于 ' + opening['when'] + ' 一次性开封（Phase 343），不得再用于调参后的确认。lockbox 与留出句式仍封存。'
    v['STAGES'][1]['def'] = '历史冻结门：材料与统计口径在冻结时确定；当前开封状态以下方记录为准。'
    v['STAGES'][1]['gates'][2] = '冻结当时确认与锁箱已封存（历史验收）'
    v['STAGES'][3]['note'] = f'MA-H3-v4：84 个新实体、{h3["n_items"]} 对；594/1188 组件（50%），留出干预保真 f(C)={f:.4f}，最好匹配对照 {best:.4f}。P1–P3 为报告内判定通过，不是图谱独立预测。C2 校准已限定旧充分性口径；H3b-v1 已测排序前≤20%头未达充分性门，未冻结载体（Phase 347）。'
    claim = v['CLAIMS'][0]
    claim['stage'] = 'H3'
    claim['run'] = f'H3 留出确认已执行：f(C)={f:.4f}，594 个组件。H3b-v1（Phase 347）已完成：exact 保留最高 {keep:.4f}，未达 0.6；移除前 5%（58 个头）后差距保留 {remove:.4f}。未冻结头载体；入口头∪接力头只是新假设。下一步需新预注册与新 MA-H1b，旧 confirm 不再作独立确认。'
    claim['refute'] = '工作主张仍待区分读取门控与末层选择。H3 的 P1–P3 仅检验原口径下的组件保真；H3b 未找到已测排序前≤20%头的充分载体。联合路径需新冻结、新材料和匹配对照，不能据现有结果宣布完整机制。'
    for box in v['LOCKBOXES']:
        if box['id'] == '主线 confirm（MA-H1）':
            box.update(state='opened', phase=343, note=opening['when'] + ' 一次性开封，H3 冻结后评估；已暴露，后续方法必须用新确认材料。')
    v['RUNS'] = [r for r in v['RUNS'] if r['id'] not in ('MA-H3-v4', 'MA-H3b-v1')]
    for run, phase, title, text, ref in [
        ('MA-H3-v4', 343, '组件载体留出确认', f'84 实体、1176 对；f(C)={f:.4f}，594 组件', h3_ref),
        ('MA-H3b-v1', 347, '校准后的头级充分性', f'320 对；保留最高 {keep:.4f}，未充分、未冻结载体', hb_ref),
    ]:
        v['RUNS'].append({'id': run, 'phase': phase, 'title': title, 'executed': True, 'exec': text,
                          'checked': True, 'check': '实际结果与模型快照关联；不代替科学审查',
                          'evidence': None, 'evidenceText': '范围内观测；未自动晋级科学证据',
                          'replicated': False, 'replicaText': '第二物理节点未做', 'report_sha256': ref})
    v['FINDINGS'] = [x for x in v['FINDINGS'] if x['phase'] not in (343, 345, 347)] + [
        {'phase': 343, 'kind': 'heldout', 'title': '组件载体在新实体上恢复范围效应，但占一半组件',
         'text': f'f(C)={f:.4f}；这是反事实背景下的留出干预保真，不能作为图谱独立预测。', 'artifact_sha256': h3_ref},
        {'phase': 345, 'kind': 'observation', 'title': '充分性检验需要校准，旧口径不能证明实现必然分布式',
         'text': 'C2 在已知 IOI 电路上揭示了干预背景与预算口径的影响。旧结果保留，并收紧解释范围。'},
        {'phase': 347, 'kind': 'negative', 'title': '少量头的移除影响集中，已测保留集合尚不充分',
         'text': f'exact 前 5% 头移除后保留 {remove:.4f}；保留最高 {keep:.4f}，未达门槛。入口∪接力仍是未检验新假设。', 'artifact_sha256': hb_ref},
    ]
    v['TESTS'][0]['rows'][0] = ['未见材料内部响应与角色绑定预测', None]
    v['METHOD_RESULTS'] = [
        {'run_id': 'MA-H3-v4', 'label': '留出干预保真', 'value': f, 'unit': 'f(C)',
         'scope': '594/1188 组件、84 新实体；不是独立预测或破解比例', 'artifact_sha256': h3_ref},
        {'run_id': 'MA-H3b-v1', 'label': '已测 exact 排序头保留最高值', 'value': keep, 'unit': 'f_keep',
         'scope': 'discovery eval、0.5%–20%头；未达0.6，不冻结载体', 'artifact_sha256': hb_ref},
    ]
    custody = [{'id': b['id'], 'state': b['state'], 'proof_sha256': custody_ref,
                'split': 'lockbox_frame' if 'lockbox_frame' in b['id'] else 'lockbox' if 'lockbox' in b['id'] else 'confirm',
                **({'opened_at': opening['when']} if b['state'] == 'opened' and 'MA-H1' in b['id'] else {})}
               for b in v['LOCKBOXES'] if 'MA-H1' in b['id']]
    ranks_path = ROOT / 'research/mainline/H3/rankings.npz'
    if hashlib.sha256(ranks_path.read_bytes()).hexdigest() != hb_freeze['rankings_sha256']:
        raise ValueError('H3b rankings differ from recorded source')
    ranks = np.load(ranks_path, allow_pickle=False)
    groups = {'exact_top_58': [int(i) for i in np.argsort(-ranks['score_exact'][:1152])[:58]],
              'ig_top_12': [int(i) for i in np.argsort(-ranks['score_atp_ig'][:1152])[:12]]}
    graphs, packages = [], []
    for run, report_ref, members in [('MA-H3-v4', h3_ref, [i for i in frozen['carriers'] if i < 1152]),
                                      ('MA-H3b-v1', hb_ref, sorted(set(sum(groups.values(), []))))]:
        cards = []
        for i in members:
            cards.append({'id': run + ':head:' + str(i), 'label': f'L{i//32}H{i%32} · 组成员候选',
                          'layer': i // 32, 'boundary': 'attention_output',
                          'index': {'space': 'attention', 'head': i % 32, 'block': i // 32},
                          'method': 'atp_ig_top_half' if run == 'MA-H3-v4' else 'ranked_group_membership',
                          'run_id': run, 'evidence': {'level': 'observation', 'artifact_sha256': report_ref},
                          'scope': {'group_membership_only': True, 'individual_causality_established': False},
                          'counterexamples': ['组效应不能归于每个成员；小集合充分性未成立；两个机制解释仍未区分'],
                          'causal_tests': 'group_effect_only_not_individual',
                          'summary': {'limitations': ['原生位置可回查；显示为观测候选，不声明该头单独必要或充分。']}})
        if run == 'MA-H3-v4':
            for component in frozen['carriers']:
                if component < 1152:
                    continue
                layer = component - 1152
                cards.append({'id': f'{run}:mlp_block:{layer}', 'label': f'B{layer} MLP 整块组成员候选（非单神经元）',
                              'layer': layer, 'boundary': 'mlp_output', 'index': {'space': 'mlp', 'field': 'residual_write', 'block': layer},
                              'method': 'atp_ig_top_half', 'run_id': run,
                              'evidence': {'level': 'observation', 'artifact_sha256': report_ref},
                              'scope': {'group_membership_only': True, 'individual_causality_established': False},
                              'counterexamples': ['确认的是 594 组件集合，不是该 MLP 块或单神经元的独立效应。'],
                              'causal_tests': 'group_effect_only_not_individual',
                              'summary': {'limitations': ['MLP 整块写出位置；不推断块内单元语义或因果作用。']}})
        graph_ref = add({'schema': 'research_graph.v1', 'kind': 'real', 'run_id': run, 'model_snapshot': SNAPSHOT,
                         'feature_cards': cards, 'graph_edges': [], 'groups': groups if run.endswith('v1') else {},
                         'limitations': ['只登记真实排序/冻结集合中的 Head；不编造 Head→Head 因果边；MLP块不冒充单神经元。']})
        graphs.append(graph_ref)
        packages.append(add({'schema': 'research_workpackage.v1', 'kind': 'real', 'run_id': run,
                             'title': '范围效应组件载体确认' if run == 'MA-H3-v4' else '头级充分性与反例',
                             'research_status': 'scoped_result' if run == 'MA-H3-v4' else 'not_supported',
                             'finding': f'范围内留出干预保真 f(C)={f:.4f}，不能替代独立预测。' if run == 'MA-H3-v4' else claim['run'],
                             'next_action': '新预注册、新 MA-H1b；联合路径待验证' if run == 'MA-H3b-v1' else '校准口径与独立预测另行验证',
                             'model_snapshot': SNAPSHOT, 'stages': {
                                 'registration_validated': {'state': 'done', 'report_sha256': report_ref},
                                 'validation': {'state': 'done' if run == 'MA-H3-v4' else 'not_supported'},
                                 'independent_prediction': {'state': 'not_tested'},
                                 'replication': {'state': 'not_tested'}}}))
        add({'schema': 'research_space_binding.v1', 'kind': 'real', 'run_id': run, 'model_snapshot': SNAPSHOT,
             'report_sha256': report_ref, 'graph_sha256': graph_ref, 'traces': [],
             'context': {'model_dimensions': {'n_layers': 36, 'n_heads': 32, 'd_model': 2560, 'd_ff': 9728},
                         'scope': '原生 Head 组成员观察；无本运行逐样本轨迹绑定，非单头因果证明。'}})
    state = {'schema': 'research_mainline.v1', 'kind': 'real',
             'sequence': previous.get('sequence', 0) + 1,
             'parent_revision': digest(previous) if previous.get('schema') else None,
             'updated_at': datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'),
             'model_snapshot': SNAPSHOT, 'view': v, 'custody': custody,
             'active_branch': {'id': 'H3b', 'phase': 347, 'status': 'not_supported',
                               'summary': '校准后小头集合充分性未成立；联合入口与接力路径需新设计、新确认。'},
             'records': [{'run_id': 'MA-H3-v4', 'report_sha256': h3_ref, 'model_snapshot': SNAPSHOT},
                         {'run_id': 'MA-H3b-v1', 'report_sha256': hb_ref, 'model_snapshot': SNAPSHOT}],
             'graph_sha256': graphs, 'workpackage_sha256': packages,
             'scientific_note': '发布与哈希校验不晋级科学证据；留出修补保真与独立图谱预测分别记账。'}
    bundle = {'schema': 'mainline_release.v1', 'state': state, 'artifacts': artifacts}
    validate(bundle)
    comparable = lambda doc: {k: v for k, v in doc.items() if k not in ('sequence', 'parent_revision', 'updated_at')}
    if previous.get('schema') and comparable(state) == comparable(previous):
        print('mainline release unchanged', previous['sequence'], digest(previous))
        return
    RELEASE.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('mainline release', state['sequence'], digest(state), 'proofs', len(artifacts))


if __name__ == '__main__':
    build()
