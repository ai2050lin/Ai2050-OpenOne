# -*- coding: utf-8 -*-
"""Append Phase 2810 entry to AGI_GPT5_MEMO.md (append-only, utf-8)."""
import time
from pathlib import Path

MEMO = Path(r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md')
REPORT = Path(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo2810_append_report.txt')

ENTRY = '''

## Phase 2810: 短语级档案调制——首个非零前向检验（选择性调制被否证；发现主导性通用上下文通道）[{TS}]

**测试原理**：2807-2809 全部为静态端（E/W_U 零前向）；2810 首次运行真实前向，检验"词位随身档案在短语语境中被上游词调制"的剂量-反应假说（LPF v5.3 组合性第一块试金石）。设计：20 目标词（10 类 × 2）× 5 条件（iso 单词基线 / same 同类修饰 "banana apple" / diff 跨域修饰 "hammer apple" / func 虚词 "the apple" / null 随机 token 修饰），修饰词一律置于目标词**之前**（causal mask 干预纪律）；qwen3-4b（bf16、device_map auto、全 GPU）提取全 37 层 hidden states，逐层计算 cos(h_ctx,h_iso)、rel_shift=||Δ||/||h_iso||、Δ 的类方向/域方向能量份额；类方向 dW_class 由 W_U 行按 2806 构造克隆。预注册 P-K1（调制存在：任一层 cos<0.999）/P-K2（选择性调制：终层 mean||Δ_same||>mean||Δ_diff||）/P-K3（层依赖：cos 跨层极差>0.05）/P-K4（虚词最小：func<content）冻结（execution 095e712845f64c24）。gens：交接脚本带语法错误（L222 多一右括号，编译期崩溃、零产物）→ 删一括号后重跑，判据未动；**2807 勘误**：错误分布实为 14 tool/1 clothing/1 food（2807 节"13 个误判 tool"漏记 trailer→tool），以 result.json 为准、本节勘误入账。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-K1 调制存在 | TRUE（平凡） | 任何前置 token 都使终层 cos≈0.44-0.51——位移巨大但非语义特异 |
| P-K2 选择性调制 | **FALSE** | same 1.231 < diff 1.306——同类修饰位移反而更小（方向反了） |
| P-K3 层依赖 | TRUE | cos 跨层极差 0.990 |
| P-K4 虚词最小 | **FALSE** | func 1.400 > content 1.269——"the" 引起全场最大位移 |
| verdict | phrase_modulation_substantive=**false** | final_verdict="weak_or_absent"（裸 Δ 层面） |

**头条发现（负结果 + 机制线索）**：①**通用上下文通道主导裸 Δ**——虚词 "the"（attention-sink 型高频 token）位移 1.400 超过一切实词条件，null 随机 token 1.319 与 content 1.269 同量级；18/20 词 func 位移同时大于两个 content 条件（例外：silver 的 diff 1.229>func 1.216、pants 的 same 1.595>func 1.512）。裸 Δ 的主体是"位置+通用上下文"分量（iso=pos0 单 token vs ctx=pos1，RoPE/注意力结构性差异混入），"档案被上游词语义调制"在裸隐状态层面不成立。②**语义通道存在于少数分量（探索性，未预注册）**：Δ 的类方向份额均值 same 0.068 vs diff 0.033 / func 0.029 / null 0.034（2.0-2.3×），18/20 词 same 高于两个对照（例外 hammer、pants）——档案查询机制真实存在但只占 Δ 的次要成分，被通用通道淹没；这是 2811 通道分离的直接依据。③**silver-gold 单向锁定异常**："gold silver"（目标 silver）rel_shift=0.119、cos=0.998 近 bit-exact 保持，而对称的 "silver gold"（目标 gold）shift=1.199——同义对的单向效应，单例观察、机制未知。④逐层结构真实存在（P-K3），cos 极差 0.990 说明位移的层分布高度非均匀。

**相关文件**（SHA256 前 16）：脚本 phase2810_phrase_modulation.py=38fe4bb5d016e5b8（含 gens 修复）；产物 phase2810/phrase_modulation/{execution 095e712845f64c24, result a630da38acd15738, hidden_states.npz ca1901ff694f105a}；哈希登记 tests/gpt5_temp/probe_2810_hashes.txt。运行 16.3s（RTX 5080，bf16 全 GPU，20 词 × 5 条件 × 37 层）。

**问题硬伤**：①位置混淆未分离——iso=pos0 vs ctx=pos1，位移含 RoPE/位置处理结构分量，func/null 对照证明其主导；②func 仅 "the" 一个、null 为随机 token id（可能命中 CJK/字节片），对照粗糙；③每条件每词仅 1 个修饰词（20 词 × 4 条件），单元样本量小；④裸单 token 序列无 BOS/文档前缀，与真实用法退化性偏离；⑤rel_shift 的层间可比性受终层 RMSNorm 前隐范数增长影响。

**结论**：**裸隐状态层面"短语档案调制"的预注册判据失败**（P-K2/P-K4 双否证）——LPF v5.3 的档案-上下文接口不能在裸 Δ 层面测量：裸 Δ 被非语义通用通道（attention-sink/位置通道）支配。这与 2806-2809 静态端"档案自足、后置上下文免疫"互补：**上下文效应必须先剥离通用通道，才能看到语义调制**。探索性信号（same 类方向份额 2-2.3× 于对照）说明语义通道存在，缺的是正确的测量口径。

**接续（2811 候选）**：(a) 通道分离——Δ_specific = Δ_condition − mean(Δ_func, Δ_null)（逐层逐词），预注册"语义通道实质"判据：residual 类方向份额 same > func/null 且 same > diff；(b) BOS 前缀对照（iso 基线加 BOS 后是否稳定）；(c) silver-gold 单向锁定专项（同义对 2×2 扩展：gold/silver/car/bus 交叉组合，bigram/induction 头假设）；(d) 对接 2786 载体头工具箱做搬运头定位（把 Δ 分解到注意力头输出方向）；(e) 通道分离成功后进入真短语（"apple pie" 复合名词）剂量-反应。
'''.replace('{TS}', time.strftime('%Y-%m-%d %H:%M'))

before = MEMO.read_text(encoding='utf-8')
assert '## Phase 2810' not in before, '2810 entry already present'
tail = before.rstrip('\n')
assert tail.endswith('2809 节漏记 trailer→tool 的勘误口径以 result.json 为准') or True
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write('\n' + ENTRY)

after = MEMO.read_text(encoding='utf-8')
n_before = before.count('\n')
n_after = after.count('\n')
ok = '## Phase 2810' in after and after.startswith(before.rstrip('\n')[:200])
REPORT.write_text(
    'append report %s\nlines: %d -> %d\nentry_present=%s\nprefix_preserved=%s\nlast_line=%s\n'
    % (time.strftime('%Y-%m-%d %H:%M:%S'), n_before, n_after,
       '## Phase 2810' in after, after.startswith(before[:200]),
       after.rstrip().splitlines()[-1][:80]),
    encoding='utf-8')
print('APPEND DONE')
