import json
import io

R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3078\omega_p75_routing_timing')
res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
out = []
out.append('verdict=%s' % res['verdict'])
out.append('forwards=%s elapsed=%s'
           % (res['forwards'], res['elapsed_s']))
out.append('seal %s'
           % json.dumps(seal, sort_keys=True))
st = res['stats']
for key in ('sp_intra', 'pp_intra',
            'sp_cross', 'sp_amp'):
    out.append(key + '=')
    for row in st[key]:
        out.append('  %r' % (row,))
out.append('icc=%r' % (st['icc'],))
out.append('icc_sig=%r' % (st['icc_sig'],))
out.append('sp_d34b=%r' % (st['sp_d34b'],))
out.append('sp_d35b=%r' % (st['sp_d35b'],))
out.append('gates=%s'
           % json.dumps(res['gates'],
                        sort_keys=True))
an = res['anchors']
out.append('a1m_diff=%r' % (an['a1m_diff'],))
p = (r'D:\AI2050\Ai2050-OpenOne'
     r'\tests\gpt5_temp\p3078_probe2.txt')
io.open(p, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('OK')
