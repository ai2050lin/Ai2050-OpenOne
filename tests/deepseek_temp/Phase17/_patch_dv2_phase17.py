# -*- coding: utf-8 -*-
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\disk_verify_phase17.py'
s = io.open(P, encoding='utf-8').read()

OLD = ("ck('P2 anchor detail 全 ok（三臂）',\n"
       "   all(all(REC[a]['E9_anchor']['ok'] if isinstance(REC[a]['E9_anchor']['ok'], bool)\n"
       "           else all(v.get('ok') for v in REC[a]['E9_anchor']['detail'].values() if isinstance(v, dict)))\n"
       "       for a in AO))\n")
NEW = ("_anc_bits = {}\n"
       "for a in AO:\n"
       "    _d = REC[a]['E9_anchor']['detail']\n"
       "    _bits = [bool(REC[a]['E9_anchor'].get('ok'))]\n"
       "    for _k, _v in _d.items():\n"
       "        if isinstance(_v, dict) and 'ok' in _v:\n"
       "            _bits.append(bool(_v['ok']))\n"
       "            if isinstance(_v.get('got'), (int, float)) and isinstance(_v.get('expected'), (int, float)):\n"
       "                _bits.append(abs(float(_v['got']) - float(_v['expected'])) <= 1e-6)\n"
       "    _anc_bits[a] = all(_bits)\n"
       "ck('P2 anchor detail 全 ok + 逐位<=1e-6（三臂）', all(_anc_bits.values()), str(_anc_bits))\n")
assert s.count(OLD) == 1, s.count(OLD)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s.replace(OLD, NEW))
t = io.open(P, encoding='utf-8').read()
assert '_anc_bits' in t and 'is not iterable' not in t
print('PATCH OK')
