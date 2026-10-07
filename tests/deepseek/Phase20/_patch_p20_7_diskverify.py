# -*- coding: utf-8 -*-
"""补丁 7：disk_verify_phase20.py
 (v1) 主脚本 `spearman()` 用**序数秩**（`argsort(argsort)`）+ std 守卫；复核脚本却用**平均秩**并断言 1e-9 相等
      ⇒ 一旦存在并列（如两层 b 恰为同一值）必 FAIL。改为：**断言用主脚本文义（序数秩）**，
      另算平均秩作信息量，两者差异（= 存在并列）记 WARN 而非 FAIL。
 (v2) 标签 Q9/Q10 的「重算」此前用「四臂皆 ≥2.0 / 四臂皆 >0」（= seal 严格文字），
      而主脚本实现是**逐配对同侧**（gap_sign_same / coupled_same）⇒ 两者可分离。
      改为按**主脚本文义**导出（这才是铁律 (ad) 要求的「重算 → 按主脚本文义导出标签 → 比对」），
      并把 seal 严格形作为 WARN 单列。
 (v3) `S['b_mlp'] and {...}` 取巧写法 ⇒ 直取 dict。
"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\disk_verify_phase20.py'
s = io.open(P, encoding='utf-8').read()
n = 0


def rep(old, new):
    global s, n
    c = s.count(old)
    assert c == 1, 'count=%d :: %r' % (c, old[:80])
    s = s.replace(old, new); n += 1


# (v1-a) 新增主脚本文义的序数秩 spearman
rep("def cross_alpha(xs, ys, frac):\n",
    "def spearman_ord(a, b):\n"
    "    \"\"\"与主脚本 `spearman()` 逐字同义：序数秩（argsort(argsort)）+ std 守卫。\"\"\"\n"
    "    a = np.asarray(a, float); b = np.asarray(b, float)\n"
    "    ok = np.isfinite(a) & np.isfinite(b)\n"
    "    a, b = a[ok], b[ok]\n"
    "    if len(a) < 3:\n"
    "        return None\n"
    "    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:\n"
    "        return None\n"
    "    ra = np.argsort(np.argsort(a)).astype(float)\n"
    "    rb = np.argsort(np.argsort(b)).astype(float)\n"
    "    ra -= ra.mean(); rb -= rb.mean()\n"
    "    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))\n"
    "    return float((ra * rb).sum() / den) if den > 1e-12 else None\n"
    "\n"
    "\n"
    "def cross_alpha(xs, ys, frac):\n")

# (v1-b) G2sp
rep("    sp = spearman_avg([wa[l] for l in reach], [abs(ba[l]) for l in reach])\n"
    "    chk('G2sp', '%s spearman(w_own,b_all) 平均秩' % a, abs(sp - S['spearman_wall_ball_own']) <= 1e-9,\n"
    "        round(sp, 9), round(S['spearman_wall_ball_own'], 9), 1e-9)\n",
    "    _xo = [wa[l] for l in reach]; _yo = [abs(ba[l]) for l in reach]\n"
    "    sp = spearman_ord(_xo, _yo); spa = spearman_avg(_xo, _yo)\n"
    "    chk('G2sp', '%s spearman(w_own,b_all)（主脚本文义：序数秩）' % a,\n"
    "        sp is not None and abs(sp - S['spearman_wall_ball_own']) <= 1e-9,\n"
    "        None if sp is None else round(sp, 9), round(S['spearman_wall_ball_own'], 9), 1e-9)\n"
    "    if sp is not None and spa is not None and abs(sp - spa) > 1e-9:\n"
    "        w('  [WARN] %s b 谱存在并列 ⇒ 序数秩 %s vs 平均秩 %s' % (a, round(sp, 9), round(spa, 9)))\n"
    "        N['WARN'] += 1\n")

# (v1-c) G4r / G4r2
rep("    rho = spearman_avg([Sa['b_all'][ia[l]] for l in com], [Sb['b_all'][ib[l]] for l in com])\n"
    "    chk('G4r', '%s ρ(b_all) 平均秩重算' % pk, abs(rho - p['rho_b_all']['rho']) <= 1e-9,\n"
    "        round(rho, 9), round(p['rho_b_all']['rho'], 9), 1e-9)\n"
    "    rho2 = spearman_avg([Sa['b_mlp'][ia[l]] for l in com], [Sb['b_mlp'][ib[l]] for l in com])\n"
    "    chk('G4r2', '%s ρ(b_mlp) 平均秩重算' % pk, abs(rho2 - p['rho_b_mlp']['rho']) <= 1e-9,\n"
    "        round(rho2, 9), round(p['rho_b_mlp']['rho'], 9), 1e-9)\n",
    "    _xa = [Sa['b_all'][ia[l]] for l in com]; _yb = [Sb['b_all'][ib[l]] for l in com]\n"
    "    rho = spearman_ord(_xa, _yb); rhoa = spearman_avg(_xa, _yb)\n"
    "    chk('G4r', '%s ρ(b_all)（主脚本文义：序数秩）' % pk,\n"
    "        rho is not None and abs(rho - p['rho_b_all']['rho']) <= 1e-9,\n"
    "        None if rho is None else round(rho, 9), round(p['rho_b_all']['rho'], 9), 1e-9)\n"
    "    if rho is not None and rhoa is not None and abs(rho - rhoa) > 1e-9:\n"
    "        w('  [WARN] %s b_all 存在并列 ⇒ 序数秩 %s vs 平均秩 %s' % (pk, round(rho, 9), round(rhoa, 9)))\n"
    "        N['WARN'] += 1\n"
    "    _xa2 = [Sa['b_mlp'][ia[l]] for l in com]; _yb2 = [Sb['b_mlp'][ib[l]] for l in com]\n"
    "    rho2 = spearman_ord(_xa2, _yb2)\n"
    "    chk('G4r2', '%s ρ(b_mlp)（主脚本文义：序数秩）' % pk,\n"
    "        rho2 is not None and abs(rho2 - p['rho_b_mlp']['rho']) <= 1e-9,\n"
    "        None if rho2 is None else round(rho2, 9), round(p['rho_b_mlp']['rho'], 9), 1e-9)\n")

# (v2) 标签按主脚本文义
rep("lab['Q9_shallow_retained'] = all(V[a]['gap'] >= FL['SHALLOWER_MIN'] for a in ARMS)\n"
    "lab['Q10_coupled_retained'] = all(V[a]['spearman_wall_ball'] > 0 for a in ARMS)\n",
    "lab['Q9_shallow_retained'] = all(QPM[k]['gap_sign_same'] for k in (PID, PID2))\n"
    "lab['Q10_coupled_retained'] = all(QPM[k]['coupled_same'] for k in (PID, PID2))\n")
rep("for k in sorted(lab):\n"
    "    chk('G7', 'JV[\"%s\"] 与重算一致' % k, bool(JV[k]) == bool(lab[k]), lab[k], JV[k])\n",
    "for k in sorted(lab):\n"
    "    chk('G7', 'JV[\"%s\"] 与重算一致（主脚本文义）' % k, bool(JV[k]) == bool(lab[k]), lab[k], JV[k])\n"
    "_strict = {'Q9': all(V[a]['gap'] >= FL['SHALLOWER_MIN'] for a in ARMS),\n"
    "           'Q10': all(V[a]['spearman_wall_ball'] > 0 for a in ARMS)}\n"
    "for _nm in ('Q9', 'Q10'):\n"
    "    _key = 'Q9_shallow_retained' if _nm == 'Q9' else 'Q10_coupled_retained'\n"
    "    if bool(_strict[_nm]) != bool(lab[_key]):\n"
    "        w('  [WARN] %s seal 严格形（四臂皆满足）=%s 与主脚本文义（逐配对同侧）=%s 不一致'\n"
    "          % (_nm, _strict[_nm], lab[_key]))\n"
    "        N['WARN'] += 1\n")

# (v3)
rep("    r3 = perm_null_share(S['b_mlp'] and {l: S['b_mlp'][idx[l]] for l in sites},\n",
    "    r3 = perm_null_share({l: S['b_mlp'][idx[l]] for l in sites},\n")

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('PATCHED %d spots' % n)
for pr in ['def spearman_ord(a, b)', 'spearman_ord(_xo, _yo)', 'spearman_ord(_xa, _yb)',
           "QPM[k]['gap_sign_same']", "QPM[k]['coupled_same']", "_strict = {'Q9'",
           "S['b_mlp'] and {"]:
    print('  chk %-30s -> %d' % (pr[:30], s.count(pr)))
