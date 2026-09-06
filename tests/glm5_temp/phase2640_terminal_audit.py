"""Append-only terminal evidence and operational correction for the complete campaign."""
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *

out=RESULT/'phase2640_paired_precision_atlas'
text_before=MEMO.read_text(encoding='utf-8-sig');phases=[int(x) for x in re.findall(r'^## Phase (\d+):',text_before,re.M)]
sources=[RESULT/'phase2637_bf16_native_numeric_control',RESULT/'phase2638_fp32_native_numeric_control',RESULT/'phase2639_expanded_paired_precision_control/bf16',RESULT/'phase2639_expanded_paired_precision_control/fp32']
conditions=[r for source in sources for r in read(source/'analysis/conditions.json')]
cfg=tomllib.loads(Path('C:/Users/Admin/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
checks={'continuous2636_to2640':phases[-5:]==list(range(2636,2641)),
        'all_five_phases_complete':all(read(next(RESULT.glob(f'phase{p}_*/analysis/final.json')))['all_checks_passed'] for p in range(2636,2641)),
        'actual7008_conditions':len(conditions)==7008,'actual6912_weight_probes':sum(r['kind']=='shared_weight' for r in conditions)==6912,
        'all_four_model_loads_restored':all(read(s/'analysis/restoration.json')['before']==read(s/'analysis/restoration.json')['after'] for s in sources),
        'production_build_and_api':read(out/'analysis/delivery_checks.json')['all_checks_passed'],
        'post_cleanup_api':read(out/'analysis/post_cleanup_checks.json')['all_checks_passed'],
        'automation_text_utf8_repaired':chr(65533) not in cfg['prompt'] and chr(65533) not in cfg['name'],
        'same_thread_continuation_active':cfg['status']=='ACTIVE' and cfg['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500' and 'Phase2636' in cfg['prompt']}
assert all(checks.values())
result={'timestamp':datetime.now().astimezone().isoformat(),'checks':checks,'all_checks_passed':True,'language_mechanism_closed':False,
    'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in list(TESTS.glob('phase2636*.py'))+list(TESTS.glob('phase2637_2639*.py'))+list(TESTS.glob('phase2640*.py'))+[ROOT/'server/native_precision_parameter_query.py',ROOT/'frontend/src/components/app/NativePrecisionParameterInspector.jsx']},
    'operational_erratum':'Previous automation update decoded non-ASCII shell JSON incorrectly, introducing916 replacement characters and corrupting the name. Replaced via official automation tool, verified zero replacement characters. Subsequent shell JSON uses ensure_ascii=True; interval and same-thread destination preserved.',
    'check_name_erratum':'Existing paired_analysis.json key six_numeric_phases_ready checks exactly2636,2637,2638,2639 (four prior Phases). Name is a typo, not additional completed phases; source now uses four_prior_phases_ready.',
    'next_work':'Crossed language-operation maps with matched material and separately heldout entities; distinguish native output pair from external common readout.'}
save(out/'analysis/terminal_audit.json',result)
marker='**Phase2640 最终交付与续研核验 ['
if marker in text_before:raise RuntimeError('terminal supplement already exists')
stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
supplement=f'\n\n{marker}{stamp}]。** 2636—2640连续五阶段完成，实数记录核验7008条件，其中6912单权重扰动和96 no-op；四次顺序加载均恢复干预矩阵哈希。新客户端134项API/构建检查与清理后133项检查通过（未做浏览器视觉验收）。清理扩大复核64原场文件共4,453,440,000字节，16前缀双精度32包保留供全部token/坐标查询。补充数值审计显示：扩大L0/V在0.2倍RMS下，用BF16伴随预测FP32有限效应的汇总误差约4.55%，FP32自身伴随约0.43%；双向参数差的中心对照误差BF16约98.42%、FP32约0.124%。这进一步区分近似梯度结构与BF16有限数值传播，不是语义闭合。检查键six_numeric_phases_ready为名称笔误，实际核验2636—2639四个前置Phase，不影响记录范围。续研提示前次经非ASCII终端JSON解码产生916个替代字符及名称乱码，本次按OpenAI Docs流程用官方工具修复并验证替代字符为0，后续采用ASCII转义JSON传输；原每10分钟同任务调度保持启用并更新到2640前沿。下一大阶段固定材料比较不同操作、独立材料验证泛化分别建立图谱，不再把同材料匹配对照一律剔除。终审文件：`{out / "analysis/terminal_audit.json"}`。\n'
with MEMO.open('a',encoding='utf-8') as stream:stream.write(supplement)
print(json.dumps({'checks':checks,'all_checks_passed':True}),flush=True)
