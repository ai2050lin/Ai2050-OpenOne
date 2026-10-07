# -*- coding: utf-8 -*-
fp = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek\gen_q03_html_r9.py"
src = open(fp, encoding="utf-8").read()
lines = src.split("\n")

# 1) 一句话结论：整行替换（数字改为现场渲染）
idx = [i for i, l in enumerate(lines) if "一句话结论" in l]
assert len(idx) == 1, "idx=%d" % len(idx)
new1 = """_es = [PM[m]['b4_rel_readout_mean3seed_recompute'] for m in MODELS]
W('<div class="ok"><b>一句话结论：</b>K1 判决所依赖的 E_read 基线已从三份冻结载体<b>独立重算并逐位复现</b>（全部锚 <code>drift = 0.00e+00</code>）；三模型误差 <b>%.1f%% / %.1f%% / %.1f%%</b>，池化 <b>%s</b>，<b>5%% 门 0/3 过门</b>（最小值仍为门槛的 %.2f 倍）。</div>' % (100*_es[0], 100*_es[1], 100*_es[2], '%.6f' % S['pooled_mean'], S['min_E_x']))"""
lines[idx[0]] = new1
src = "\n".join(lines)

# 2) 无 %-格式化的行里 %% 会原样输出 -> 改回单 %
a = src
src = src.replace("<h2>3 · per-seed 与 bootstrap 95%% CI", "<h2>3 · per-seed 与 bootstrap 95% CI")
assert src != a, "95%% not found"
a = src
src = src.replace("<th>5%% 门</th>", "<th>5% 门</th>")
assert src != a, "5%% th not found"

open(fp, "w", encoding="utf-8", newline="\n").write(src)
chk = open(fp, encoding="utf-8").read()
print("PATCH_OK")
print("  has _es =", "_es = [PM[m]" in chk)
print("  has 95% CI =", "bootstrap 95% CI" in chk)
print("  has 5% 门</th> =", "<th>5% 门</th>" in chk)
