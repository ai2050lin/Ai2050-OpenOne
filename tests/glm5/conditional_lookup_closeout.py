"""Append-only record of the completed pilot; preserve all frozen evidence."""
import json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((OUT/'pilot-result.json').read_bytes());a=json.loads((OUT/'audit.json').read_bytes())
assert r['behavior_pass'] and r['technical_pass'] and a['hashes_valid']
manifest={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}
for p in (ROOT/'tests/glm5').glob('conditional_lookup*'):manifest[str(p.relative_to(ROOT))]=sha(p)
(OUT/'manifest.json').write_text(json.dumps({'files':manifest,'provenance':r['provenance'],'status':'pilot complete; discovery and confirm not executed'},indent=2),encoding='utf-8')
text=f'''

## Phase 2965: 独立八格条件查找表的预注册与 Qwen14B 自然原生试跑 [2026-11-17 11:13]

**任务与状态。** 用户授权收敛预注册并开始测试。本轮完成 CLT-20261117-v1 的冻结、16家族256输入真实试跑及独立CPU审计；未执行64家族发现或128家族正式确认、没有选择机制候选或实施干预。预注册 tests/glm5/conditional_lookup_prereg.md，机器绑定在 tests/glm5/result/conditional_lookup_native_v1/prereg.json。确认材料已冻结但未进入模型或结果分析。

**关键纠正与设计。** 检查旧 source_binding_pointer.py 的 entorder=(family+src+x)%2，旧两块行序始终相反，所以 r1=1-r0，旧表读法不能区分两个独立行位与一位加XOR。新材料覆盖00/01/10/11四表型×b=0/1，并交叉条件前置/后置共16格。名字家族隔离，代码固定在物理槽位、独立随机，标题和物理块配平。原生输入使用真实chat template、关闭thinking、固定Answer换行。后置的实体前缀逐token相同，理论上该位置不能知道未来b。外部目标 r=(1-b)r0+b*r1 是任务定义，不是已提取的网络算法。

**冻结与门槛。** pilot=16/256，discovery=64/1024，confirm=128/2048。每试跑格严格四候选margin>0与完整词表首选均须>=15/16。正式行为每格两项均>=95%。正式配对规划期待15点、p_d=.25、alpha=.025，以128家族保守计，工具计算需要106、power=.875。试跑不承担正式功效声明。全部八格用于试跑，不声称未见表型组合。后继编码门两位>=90%、联合>=85%，与最强匹配基线增益>=10点且区间下界>0；原生选择、写入预测和有限因果检验有分阶段冻结，未达到时不晋级。不能用差分搬运或真实下游状态补入替代机制提取。

**真实结果。** 原始BF16 Qwen3-14B快照 {r['provenance']['model_revision']}，CUDA RTX5080、CPU权重读取。16格各16/16，严格四候选与真实全词表top1均256/256。后置64对在实体位置的全部41残差边界逐位一致。保存实体/最终位置的所有原生残差及B20/21/22全部17408单元G/U/A；CPU BF16复算SiLU(G)*U在三个块的最大差均0、逐位相等。产物哈希、金标签、逐条margin由独立脚本核对。原场289400628字节，运行至结果保存耗时{r['elapsed_seconds']:.1f}秒。日志逐层采样cuda峰值约1.71GiB、共享80MiB；未记录finish瞬时峰值，不能当作全程峰值。进度completed表示进入该层首批回调，最终pilot-result是全量完成标记。

**判断与限制。** 扩展后00/11也全对，固定的一位行位加XOR不能统一解释这批自然行为；不排除模型分表型切换策略。后置技术校准说明未来条件未泄漏进实体状态，不证明实体已经保存完整表。身份标记后重查正文、缓存两个代码、多位置联合编码均未被排除。只有16独立试跑家族，未做自由生成、留出内部预测、原生选择或因果验收；原场可复算不是机制破解。三个MLP块的数值对账是技术证据，不是语言功能证据。无新RDC公式或AGI结论。

**第一性问题与下一大阶段。** 独立表型把“条件变化”与“可存内容”解耦，条件后置给出因果可见性约束。下一阶段按预注册用发现家族提取候选，解释记录如何通过原生QK/OV、残差和G/U/A写入，选择前状态如何组合b与行位；必须在新家族预测实际中间量并冻结小机制，再开启confirm。若只解码成功则保持信息候选；行为/数值失败停受影响步骤；若必须重查正文，缩小单向量表的命题。机制干预尚需具体索引与匹配剂量第二次冻结，不自动扩大语言或模型。

**操作与保存。** 执行 .venv/Scripts/python.exe -B tests/glm5/conditional_lookup_native.py freeze；-u -B 同脚本 pilot；-B tests/glm5/conditional_lookup_audit.py。材料/设计/脚本hash与脚本依赖、native、逐条score、日志、audit保留在conditional_lookup_native_v1，标准来源回执 results/2026-11-17/111225_conditional_lookup_native_pilot.json。统一storage_policy检查2GiB预算和20GiB余量，GPU租约互斥。只允许清理本轮空临时目录，其他研究文件保持。尚未上传正式服务器或注册客户端研究证据，避免把试跑准入作为机制成果。
'''
memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
old=memo.read_bytes();assert b'## Phase 2965:' not in old
with memo.open('ab') as f:f.write(text.encode('utf-8'))
assert memo.read_bytes().startswith(old)
idx=ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
old=idx.read_bytes();assert b'## Phase 470:' not in old
with idx.open('ab') as f:f.write('\n\n## Phase 470: 独立八格条件查找表预注册与自然原生试跑 [2026-11-17 11:13]\n\n见 GLM5 Phase 2965。Qwen3-14B 16家族256输入全部正确；条件后置64对实体全41边界逐位一致。仅完成试跑和数值审计，发现、确认与因果机制尚未执行。\n'.encode('utf-8'))
assert idx.read_bytes().startswith(old)
print('Closed pilot and appended Phase 2965 / index 470.')
