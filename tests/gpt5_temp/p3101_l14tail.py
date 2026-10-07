# -*- coding: utf-8 -*-
import io
import json

led = json.load(io.open(
    r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model'][0]
cs = l14['connects']
out = []
for c in cs[-5:]:
    if isinstance(c, dict):
        out.append('phase=%s axis=%s verdict=%s'
                   % (c.get('phase'), c.get('axis'), c.get('verdict')))
    else:
        out.append('STR: %s' % str(c)[:150])
io.open(r'tests/gpt5_temp/p3101_l14tail.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('OK')
