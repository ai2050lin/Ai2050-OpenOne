# -*- coding: utf-8 -*-
"""Append post-observation environment addendum to MEMO (after 3161 section) + daily lesson line."""
import io, os, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
NOW = time.strftime('%Y-%m-%d %H:%M')

ADD = (
    '\r\n#### 3161 执行补记（' + NOW + '，观测后环境事件；不影响冻结协议与已封存观测，磁盘复核 19/19 PASS）\r\n'
    '1. **5.14 on-the-fly bnb 加载 segfault 根因与修复**：transformers 5.14.1 的 bnb 4-bit 现场量化在 '
    'from_pretrained 内部把全量 bf16（14b=29.5GB）materialize 进 CPU RAM（资源监控实测 22.9→0.4GB 匀速冲顶后 '
    'SIGSEGV；HF_DEACTIVATE_ASYNC_LOAD 无效；= GitHub issue #43032 家族；本机可用 RAM ~22.7GB < 29.5GB 结构性不足，'
    '崩溃点随共租进程 RAM 波动浮动——此前三次「加载期崩溃」与 glm4/4b 偶发成功全部由此统一解释）。'
    '混合放置（GPU_L=20 + llm_int8_enable_fp32_cpu_offload）同样死于该 materialize（与 GPU 无关）。'
    '修复 = transformers 4.57.1 + huggingface-hub 0.36.0 隔离副本（PYTHONPATH 遮蔽，venv 本体不动，'
    '`tests/gpt5_temp/tf457/`）做一次性预量化转换 → `models/hf/Qwen3-14B-bnb-nf4`（9.26GB bnb-4bit 序列化 '
    'checkpoint，conv 54s）→ 5.14 pre-quantized 反序列化加载（5.7s、RAM 平稳、vram_alloc=9.93GB）。\r\n'
    '2. **4.57 不可直接用于正式跑（协议口径）**：4.57 的 all_hidden_states 收集时机与 5.14 不同（层调用前收集'
    '「输入」而非调用后收集「输出」）→ layer 级注入 forward-hook 的修改在 hidden_states 槽位偏移一层（diag4 实证：'
    '注入层 17 输出 → diff 出现在槽 19 而非槽 18），4b SMOKE 在 4.57 下 none share(L_mid)=0 触发协议断言。'
    '预量化转换只用 4.57 的 bnb 0.50.2 NF4 kernel；正式观测仍在 5.14（diag5 验证：slots=41、注入 delta=50.000 '
    '精确出现在槽 L_MID、d[L_MID-1]=0、is_loaded_in_4bit=True）。主脚本 MDIR_MAP[14b]→Qwen3-14B-bnb-nf4 '
    '（design dict 不含路径，execution sha 不变）。\r\n'
    '3. **4b 封存产物误覆盖与完全恢复**：无参运行主脚本 = 默认单进程正式 4b（危险默认，教训：跑 summary 必须显式 '
    'P3161_MODEL=summary）→ 08:43 覆盖 4b result.json/collect.npz；从 _parts（07:49 原始锚数据，未受损）COLLECT '
    '重跑 → collect.npz blob sha8=e54bdaed **逐位复现**（数组级证据无损）；verdict 数字三次全同'
    '（C4_0.4297|T_0.0563|ctrl_0.0005|randr_1.2312）；res_sha8 因 result 内时间戳/环境字段漂移 '
    'e261d42f→5b3ccf77（verdict 级等价）。closeout 的 seal 一致性断言同步修正为 verdict 内嵌 sha 验证'
    '（seal 哈希 pre-seal 文件态；磁盘最终文件含 seal 字段，sha8_file(最终)≠seal 属构造性；独立复核以 '
    'CRLF 字节级重构复算 seal 3/3 精确命中）。\r\n'
    '4. pre-quantized 14b 锚容差 cos=0.9903/rel=0.1398（优于 glm4 现场量化 0.196-0.241；同一 bnb kernel，'
    '加载路径不影响量化数值——同源量化数据逐位复现）。每 anchor 全链仅 ~20s（加载 6s）。\r\n'
)

# MEMO append (CRLF, keep BOM/EOL invariant)
raw = open(MEMO, 'rb').read()
assert raw[:3] == b'\xef\xbb\xbf', 'BOM lost before append'
txt = raw.decode('utf-8')
marker = '3161 执行补记'
assert marker not in txt, 'addendum already present'
txt2 = txt.rstrip('\r\n') + '\r\n' + ADD
open(MEMO, 'wb').write(txt2.encode('utf-8'))
b = open(MEMO, 'rb').read()
assert b[:3] == b'\xef\xbb\xbf'
assert b.count(b'\n') == b.count(b'\r\n'), 'EOL mixed'

# daily append
line = ('- 3161 执行补记：5.14 bnb 现场量化 = 全量 bf16 materialize→RAM 冲顶 segfault（issue #43032 家族，'
        '资源监控实证）→ 4.57 隔离副本一次性预量化 `Qwen3-14B-bnb-nf4`（9.26GB）→ 5.14 pre-quantized 加载恢复；'
        '4.57 hidden_states 收集时机偏移一层 → 不可直接正式跑（diag4/diag5 实证）；4b 无参误覆盖 → _parts 恢复 '
        'npz e54bdaed 逐位复现；教训=summary 必须显式 P3161_MODEL=summary；磁盘复核 19/19 PASS。\n')
with io.open(DAILY, 'a', encoding='utf-8') as f:
    f.write(line)

print('ADDENDUM OK: memo bytes %d -> %d' % (len(raw), len(open(MEMO, 'rb').read())))
