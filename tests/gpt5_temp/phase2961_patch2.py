# -*- coding: utf-8 -*-
import io
P = r'D://AI2050//Ai2050-OpenOne//tests//glm5//phase2961_primitive_card_compression.py'
t = io.open(P, encoding='utf-8').read()
old = chr(21516)+chr(37197)+chr(32622)+' sep 84.81 '+chr(22797)+chr(29616)+chr(20294)+chr(20013)+chr(20301)+chr(31227)+chr(20301)+chr(31526)+chr(21495)+chr(32763)+chr(36716)+chr(24050)+chr(30331)+chr(35760)
new = '2944 登记的 L16 s2 sep 84.81 参照'
assert old in t, 'miss'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
print('r3m ok')
