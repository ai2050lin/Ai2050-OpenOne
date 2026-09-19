# -*- coding: utf-8 -*-
"""2878+2879 ledger update: register syntax axes, translation real
negative, syntax mlp growth point, channel-dissociation linkage."""
import io
import json
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_ledger_update_2879.txt'
log_lines = []


def log(m):
    log_lines.append(m)


led = AtlasLedger.load(verify_sha=True)
rep = []
stale = led.verify(rep)
log('pre-update stale: %s' % (stale if stale else 'none'))

# ---------- SHA report ----------
import hashlib, os
RB = led.__class__ and r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
     r'\rdc_query_construction_20260913'


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


P2878 = os.path.join(RB, 'phase2878', 'syntax_trans_vocab')
P2879 = os.path.join(RB, 'phase2879', 'syntax_mlp_spectrum')
for d in (P2878, P2879):
    for fn in sorted(os.listdir(d)):
        log('SHA %s/%s = %s' % (os.path.basename(d), fn,
                                sha8(os.path.join(d, fn))))

# ---------- axes ----------
ids = [a['axis_id'] for a in led.doc['axes']]
if 'syntax' not in ids:
    led.doc['axes'].append({
        'axis_id': 'syntax', 'kind': 'syntactic',
        'words_n': 48, 'labels_n': 3,
        'axes_names': ['number', 'gerund', 'comparative'],
        'vocab_source': {'path': 'phase2878/syntax_trans_vocab/'
                                  'syntax_trans_vocab.npz',
                         'sha256_8': sha8(os.path.join(
                             P2878, 'syntax_trans_vocab.npz')),
                         'phase': 2878},
        'status': 'active',
        'notes': 'tense quarantined (VG2b 0.1902 < 0.2); '
                 'translation family settled separately'})
if 'translation' not in ids:
    led.doc['axes'].append({
        'axis_id': 'translation', 'kind': 'cross_lingual',
        'words_n': 35, 'labels_n': 3,
        'axes_names': ['lang_fr', 'lang_de', 'lang_es'],
        'vocab_source': {'path': 'phase2878/syntax_trans_vocab/'
                                  'syntax_trans_vocab.npz',
                         'sha256_8': sha8(os.path.join(
                             P2878, 'syntax_trans_vocab.npz')),
                         'phase': 2878},
        'status': 'quarantined',
        'notes': 'REAL NEGATIVE (2878 Gen2): after homograph '
                 'decontamination VG2b pair-pair cos = -0.0007/-0.0029/'
                 '0.0011 - no unified cross-lingual axis direction '
                 'exists in the tied unembed; axes not measured'})

# ---------- blocks ----------
bid = [b['block_id'] for b in led.doc['blocks']]
if 'B3_syntax_mlp' not in bid:
    led.add_block({
        'block_id': 'B3_syntax_mlp', 'axis_id': 'syntax',
        'kind': 'mlp_response', 'shape': [48, 10],
        'src': {'path': 'phase2879/syntax_mlp_spectrum/'
                        'syntax_mlp_spectrum.npz',
                'sha256_8': sha8(os.path.join(
                    P2879, 'syntax_mlp_spectrum.npz')),
                'key': 'B3_spec'}})

# ---------- measurements ----------
mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2878_syntax_trans_vocab' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2878_syntax_trans_vocab', 'type': 'vocab_legal_dual',
        'verdict': 'Gen1 vocab_legal=false (homograph contamination: '
                   '" chat"->English; fr pairs 4/8) -> Gen2 expansion + '
                   'frozen homograph exclusion: VG1 syntax 4/4 trans '
                   '3/3; syn_legal=true (number/gerund/comparative, 48 '
                   'words 22 pairs; tense 0.1902 quarantined); '
                   'trans_legal=false REAL NEGATIVE (VG2b ~0)',
        'source': {'path': 'phase2878/syntax_trans_vocab/result.json',
                   'sha256_8': sha8(os.path.join(P2878, 'result.json')),
                   'phase': 2878}})
if 'M2879_syntax_mlp' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2879_syntax_mlp', 'type': 'channel_dissociation',
        'verdict': 'v1=true(det 0.0)/E1=mlp_carries_syntax_axis(acc '
                   '0.7083 vs null p95 0.5208, 1.82x)/E2=margin_absent'
                   '(0.0744 < 0.0889; weak trend only); layer profile '
                   'peaks L35 (0.2425)',
        'source': {'path': 'phase2879/syntax_mlp_spectrum/result.json',
                   'sha256_8': sha8(os.path.join(P2879, 'result.json')),
                   'phase': 2879}})

# ---------- growth curve ----------
gids = [g['point_id'] for g in led.doc['growth_curve']]
if 'G_axis3_syntax_mlp' not in gids:
    led.doc['growth_curve'].append({
        'point_id': 'G_axis3_syntax_mlp', 'axis_id': 'syntax',
        'block': 'B3_syntax_mlp', 'components': 10, 'acc': 0.7083,
        'shared_with_prev': 0, 'new': 10, 'phase': 2879,
        'notes': 'mlp channel curve: class 0.875 / attr 0.500 / syntax '
                 '0.7083; E2 margin absent (retrieval succeeds without '
                 'global same-axis clustering - unbalanced 3-axis null '
                 'p95 0.5208)'})

# ---------- linkage ----------
lids = [l['link_id'] for l in led.doc['linkage']]
if 'L5_channel_law_syntax' not in lids:
    led.doc['linkage'].append({
        'link_id': 'L5_channel_law_syntax', 'from': {'axis': 'syntax'},
        'to': {'kind': 'mlp_response', 'block': 'B3_syntax_mlp'},
        'evidence': 'E1 retrieval 0.7083 (1.82x null p95) with zero new '
                    'head components; E2 margin absent - mlp carrier '
                    'law holds for a third axis family (class 0.875 / '
                    'attr 0.5 / syntax 0.7083), while translation axes '
                    'have NO unified direction in the tied unembed '
                    '(VG2b ~0, 2878)',
        'phase': 2879, 'status': 'confirmed'})

# also extend L4 evidence string with syntax point
for l in led.doc['linkage']:
    if l['link_id'] == 'L4_axis_independence':
        ev = l['evidence']
        if 'syntax 0.7083' not in ev:
            l['evidence'] = ev + ' / syntax E1 0.7083 (2879)'

led.save()
log('ledger saved')

rep2 = []
led2 = AtlasLedger.load(verify_sha=False)
stale2 = led2.verify(rep2)
log('post-update stale: %s' % (stale2 if stale2 else 'none'))
log('axes: %s' % [a['axis_id'] for a in led2.doc['axes']])
log('blocks: %s' % [b['block_id'] for b in led2.doc['blocks']])
log('measurements n=%d' % len(led2.doc['measurements']))
log('growth points n=%d' % len(led2.doc['growth_curve']))
log('linkage n=%d' % len(led2.doc['linkage']))
for r in led2.growth_table():
    log('GROWTH ' + r)

with io.open(OUT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(log_lines) + '\n')
print('written')
