# -*- coding: utf-8 -*-
"""Probe registry v1.23 anchor roles / counter_evidence for 3201 audit criteria."""
import json

P = r'tests/glm5/result/rdc_query_construction_20260913/phase3200/g5e2_ftr23_heldout_upgrade/atlas_registry_v1_23.json'
r = json.load(open(P, encoding='utf-8'))
tax = r['evidence_level_taxonomy']
print('taxonomy keys:', list(tax.keys()))
for f in r['features']:
    roles = [a['role'] for a in f['anchors']]
    print('%s | %-14s | anchors=%d roles=%s | n_ce=%d | scope=%s' % (
        f['id'], f['evidence_level'], len(f['anchors']),
        ','.join(sorted(set(roles))) if roles else '-', len(f['counter_evidence']),
        'Y' if f.get('scope_limits') else 'N'))
print('upgrade_log nodes:', [u['node'] for u in r['upgrade_log']])
gl = json.load(open(r'tests/glm5/result/rdc_query_construction_20260913/phase3175/g5a12_residual_arms/gap_ledger_v1_5.json', encoding='utf-8'))
print('appendix:')
for a in gl['appendix']:
    print(' ', a['id'], a['status'], '|', a['title'][:50])
