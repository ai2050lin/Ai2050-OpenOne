# -*- coding: utf-8 -*-
"""补丁 E-offload：`mod_dev` 对 accelerate offload 的模块返回 execution_device（而非 meta 占位参数）。
   不改口径/判据；仅装置兼容性修复。修后必须重跑**全部**臂以保持同一实现版本。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\n2h1a12_quant_scheme_robustness.py'
s = io.open(p, encoding='utf-8').read()

old = ("    def mod_dev(m):\n"
       "        for p in m.parameters():\n"
       "            return p.device\n"
       "        return torch.device('cuda')")
new = ("    def mod_dev(m):\n"
       "        # [E-offload] accelerate 的 CPU-offload 用 meta 占位参数承载真实权重；此时参数 device 是 'meta'，\n"
       "        # 真实执行设备在 module._hf_hook.execution_device。若直接用参数 device 构造输入张量，\n"
       "        # 会在前向里触发 'Cannot copy out of meta tensor'。\n"
       "        h = getattr(m, '_hf_hook', None)\n"
       "        ed = getattr(h, 'execution_device', None) if h is not None else None\n"
       "        if ed is not None:\n"
       "            return ed\n"
       "        for p in m.parameters():\n"
       "            return p.device\n"
       "        return torch.device('cuda')")
assert s.count(old) == 1, 'old count=%d' % s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('has E-offload:', '[E-offload]' in s)
print('has execution_device:', 'execution_device' in s)
