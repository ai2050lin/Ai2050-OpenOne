# -*- coding: utf-8 -*-
"""补丁 3：加 loadonly 模式 —— 只加载 + 一次前向 + 记内存，不算任何研究量（保护 A1/A2 的 holdout）。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\probe_feasibility_phase19.py'
s = io.open(p, encoding='utf-8').read()

# 模块级开关
old0 = "NPAIR = int(os.environ.get('NPAIR', '0'))   # 0 = 全量（全 41 实例估 U_l，与 P17 同口径）"
new0 = old0 + "\nLOADONLY = os.environ.get('PROBE_MODE', '') == 'loadonly'"
assert s.count(old0) == 1, 'old0'
s = s.replace(old0, new0)

# main 内分支
old1 = ("            w('  加载后 RAM avail=%.1f GB ; GPU free=%.1f GB' % (ra2, gf2 if gf2 else -1))\n"
        "            meas = measure(tok, model, pairs)")
new1 = ("            w('  加载后 RAM avail=%.1f GB ; GPU free=%.1f GB' % (ra2, gf2 if gf2 else -1))\n"
        "            if LOADONLY:\n"
        "                with torch.no_grad():\n"
        "                    _ii = torch.tensor(\n"
        "                        [tok.encode(TMPL % INST_ALL[0][0], add_special_tokens=False)],\n"
        "                        device=model.get_input_embeddings().weight.device)\n"
        "                    _lg = model(input_ids=_ii).logits[0, -1].float().cpu().numpy()\n"
        "                rec['loadonly'] = True\n"
        "                rec['fwd_ok'] = bool(np.isfinite(_lg).all())\n"
        "                w('  [loadonly] 一次前向 OK=%s -> 不计算任何研究量（holdout 保护）' % rec['fwd_ok'])\n"
        "                rep['runs'][scheme] = rec\n"
        "                del model\n"
        "                gc.collect()\n"
        "                torch.cuda.empty_cache()\n"
        "                continue\n"
        "            meas = measure(tok, model, pairs)")
assert s.count(old1) == 1, 'old1'
s = s.replace(old1, new1)

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('has LOADONLY  :', 'LOADONLY = os.environ.get' in s)
print('has loadonly  :', "rec['loadonly'] = True" in s)
