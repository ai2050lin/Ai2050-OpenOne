# -*- coding: utf-8 -*-
"""Fix P17 global rename over-reach in generated phase3201 script:
restore reg23 (sealed source v1.23) where the rename should not have applied."""
import io
P = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3201_g5e3_eline_audit.py'
t = io.open(P, encoding='utf-8').read()

def rep(old, new):
    global t
    c = t.count(old)
    assert c == 1, ('count', c, old[:80])
    t = t.replace(old, new)

rep("    reg24 = json.loads(open(SRC['registry_v123'], 'rb').read().decode('utf-8'))",
    "    reg23 = json.loads(open(SRC['registry_v123'], 'rb').read().decode('utf-8'))")
rep("    assert reg24['version'] == '1.23' and len(reg24['features']) == 26",
    "    assert reg23['version'] == '1.23' and len(reg23['features']) == 26")
rep("    reg24 = copy.deepcopy(reg24)\n    reg24['version'] = '1.24'",
    "    reg24 = copy.deepcopy(reg23)\n    reg24['version'] = '1.24'")
rep("    f24_old = [f for f in reg24['features'] if f['id'] == 'FTR-24'][0]",
    "    f24_old = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]")
rep("        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f\n                             for f in reg24['features']]",
    "        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f\n                             for f in reg23['features']]")
rep("    else:\n        reg24 = copy.deepcopy(reg24)\n        disposition = 'v1_23_unchanged",
    "    else:\n        reg24 = copy.deepcopy(reg23)\n        disposition = 'v1_23_unchanged")
rep("    for f in reg24['features']:\n        a = json.dumps(f, ensure_ascii=False, sort_keys=True)",
    "    for f in reg23['features']:\n        a = json.dumps(f, ensure_ascii=False, sort_keys=True)")
rep("    assert json.dumps(reg24['upgrade_log'], ensure_ascii=False, sort_keys=True) == \\\n        json.dumps(reg24['upgrade_log'], ensure_ascii=False, sort_keys=True)",
    "    assert json.dumps(reg24['upgrade_log'], ensure_ascii=False, sort_keys=True) == \\\n        json.dumps(reg23['upgrade_log'], ensure_ascii=False, sort_keys=True)")
rep("        assert json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True) == \\\n            json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')",
    "        assert json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True) == \\\n            json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')")
rep("    assert json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True) == \\\n        json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'",
    "    assert json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True) == \\\n        json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'")

io.open(P, 'w', encoding='utf-8', newline='').write(t)
back = io.open(P, encoding='utf-8').read()
assert back == t
print('fixed, reg23 refs now:', t.count('reg23'))
