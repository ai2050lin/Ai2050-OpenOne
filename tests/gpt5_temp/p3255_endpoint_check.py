# -*- coding: utf-8 -*-
"""3255 engineering-arm check: call atlas_gaps coroutine directly (read-only)
and assert the landed v1.12 doc shape + APPX-6 (3254) + version chain."""
import asyncio, json, sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne')
from server.ai_rnd_service import atlas_gaps

doc = asyncio.get_event_loop().run_until_complete(atlas_gaps())
assert doc['schema'] == 'atlas_gaps_v1'
assert doc['version'] == '1.12', doc['version']
assert doc['file'] == 'gap_ledger_v1_12.json'
assert doc['updated'] == '2026-10-20'
assert '5934631b' in (doc['supersedes'] or '')
# gaps unchanged (5, statuses from sealed v1.11)
gaps = {g['id']: g['status'] for g in doc['gaps']}
assert set(gaps) == {'GAP-1', 'GAP-2', 'GAP-3', 'GAP-4', 'GAP-5'}, set(gaps)
# appendix = 6 entries, APPX-6 last with sealed anchor
ap = doc['appendix']
assert [a['id'] for a in ap] == ['APPX-%d' % i for i in range(1, 7)]
a6 = ap[-1]
assert a6['anchor'] == 'p3250' and a6['anchor_sha8'] == '9d0f1825'
assert a6['status'] == 'appendix_recorded'
assert 'XC_p5_b8' in a6['note']
assert '3239' in a6['title'] or '3239' in a6['note']
# version chain: auto-discovered, newest first, contains 1.12 + 1.11
vs = [v['version'] for v in doc['versions']]
assert vs[0] == '1.12' and '1.11' in vs, vs
print('ENDPOINT CHECK OK: gaps v1.12, 5 gaps unchanged, appendix APPX-1..6, '
      'APPX-6 anchor p3250/9d0f1825, chain %s' % vs)
