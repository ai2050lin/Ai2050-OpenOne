# -*- coding: utf-8 -*-
"""Phase 12 closeout 脚本补丁：修正一个键名 + 追加事后描述性块（明确标注 post-hoc）。"""
import io, hashlib, py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase12\closeout_phase12.py'
t = io.open(P, encoding='utf-8').read()


def rep(old, new, tag):
    global t
    n = t.count(old)
    assert n == 1, 'PATCH %s: 期望 1 处，实际 %d' % (tag, n)
    t = t.replace(old, new, 1)
    print('[patch12c] %s OK' % tag)


rep("   'J_inject_reference_from_phase11': {tag(k): v for k, v in R['G_family']['G3']['pairs']},",
    "   'G3_pairs_site_Jswap_Jinject': R['G_family']['G3']['pairs'],",
    'rename-G3-pairs-key')

OLD = "jp = os.path.join(P12T, 'judgement_phase12.json')"
NEW = '''# ---- 事后描述性（post-hoc，无判决角色；明确标注）----
import math
_js = [(s, R['profile_swap'][str(s)]['jump_ratio']) for s in SV]
_js = [(s, v) for (s, v) in _js if v is not None and math.isfinite(v) and v > 0]
_half_log = None
if len(_js) >= 4 and _js[-1][1] > 0 and _js[0][1] > 0:
    _target = math.sqrt(_js[0][1] * _js[-1][1])
    for i in range(len(_js) - 1):
        a, b = _js[i][1], _js[i + 1][1]
        if b <= _target <= a:
            tt = (math.log(a) - math.log(_target)) / (math.log(a) - math.log(b))
            _half_log = _js[i][0] + tt * (_js[i + 1][0] - _js[i][0])
            break
_shallow = [v for (s, v) in _js if s <= 12]
_deep = [v for (s, v) in _js if s > 12]
J['posthoc_descriptive'] = {
 'note': 'POST-HOC，无判决角色；仅为报告提供尺度感，未写入任何预注册判据',
 'J_swap_endpoints': [_js[0][1], _js[-1][1]] if _js else None,
 'J_swap_log2_decline': (math.log2(_js[0][1] / _js[-1][1]) if _js else None),
 'J_swap_half_log_depth': _half_log,
 'J_swap_mean_L6_L12': (sum(_shallow) / len(_shallow)) if _shallow else None,
 'J_swap_mean_L14_L34': (sum(_deep) / len(_deep)) if _deep else None,
 'G1_is_an_instrument_artifact': (
   'G1 的归一化 XN=(xhalf-min)/(max-min) 隐含假设剖面随深度【上升】；实测 rho(xhalf,depth)=-0.7833 为负，'
   '故 XN[0]=1.0，first_reach(0.1/0.5/0.9) 全部退化返回首站点 => G1a_crystallized 是仪器伪影，本 Phase 不引用。'
   '正确的方向读法是：xhalf 随深度【下降】（承诺点前移 / 越深越容易被推动）。'),
 'G4_note': ('确认集只有 4 个位点 (7,11,20,34)，其 xhalf = 0.437/0.441/0.441/0.473 单调上升 => rho=+0.80，'
             '与发现集 18 位点的 -0.7833 符号相反。这不是物理冲突，而是采样密度不足：发现集剖面非单调'
             '（L30 触底 0.390 后 L34 反弹到 0.465），4 个位点落在不同支上。'),
}
jp = os.path.join(P12T, 'judgement_phase12.json')'''

rep(OLD, NEW, 'add-posthoc-block')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
assert t2.count("J['posthoc_descriptive'] = {") == 1
assert t2.count("'G3_pairs_site_Jswap_Jinject': R['G_family']['G3']['pairs'],") == 1
py_compile.compile(P, doraise=True)
print('[patch12c] sha8=%s ; py_compile OK' % hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8])
