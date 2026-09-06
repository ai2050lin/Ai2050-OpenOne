"""Final append-only delivery and continuation audit; no additional scientific Phase."""
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *

out=RESULT/'phase2635_expanded_native_path_confirmation'
phases=[int(x) for x in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
cfg=tomllib.loads(Path('C:/Users/Admin/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
checks={'continuous_campaign':phases[-6:]==list(range(2630,2636)),
        'all_six_phases_passed':all(read(next(RESULT.glob(f'phase{p}_*/analysis/final.json')))['all_checks_passed'] for p in range(2630,2636)),
        'client_production_build':read(out/'analysis/delivery_checks.json')['all_checks_passed'],
        'post_cleanup_api':read(out/'analysis/post_cleanup_checks.json')['all_checks_passed'],
        'same_thread_continuation_active':cfg['status']=='ACTIVE' and cfg['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500',
        'ten_minute_heartbeat':cfg['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=10'}
assert all(checks.values())
report={'timestamp':datetime.now().astimezone().isoformat(),'checks':checks,'all_checks_passed':True,'mechanism_closed':False,
        'script_sha256':{str(p.relative_to(ROOT)):sha(p) for p in list(TESTS.glob('phase263[0-5]*.py'))+[ROOT/'server/native_path_parameter_query.py',ROOT/'frontend/src/components/app/NativePathParameterInspector.jsx']},
        'automation_id':cfg['id'],'cadence':'existing same-thread automation corrected from weekly Sunday09:00 to every10minutes; no duplicate task or model',
        'source':'https://learn.chatgpt.com/docs/automations?surface=app',
        'next_frontier':'separate intermediate BF16 rounding from finite nonlinear propagation; then enlarge same-output cross-family full-coordinate natural-operation maps'}
save(out/'analysis/terminal_audit.json',report)
stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
text=f'\n\n**Phase2635 交付与连续研究核验 [{stamp}]。** 2630—2635六阶段连续完成，程序与证据边界检查全部通过；客户端188项含生产构建、清理后187项API/数据检查通过。原场清理762文件共33,113,645,664字节，34个已发布前缀的68原场包保留；未展示原场未进回收站，可按保存模型/脚本/材料重建。实际核查发现旧续研调度为每周日09:00，而非此前工作摘要所称每小时；依据用户“同目标自动继续”要求，通过官方工具将同一调度修正为每10分钟唤醒，保留查重与单GPU顺序测试约束，不新建任务。OpenAI Docs核对依据：[本地续研调度与运行条件](https://learn.chatgpt.com/docs/automations?surface=app)，需电脑与应用保持运行。下一前沿仍是原生单参数跨层传播，先分离内部BF16舍入与真实非线性，再扩大严格同输出跨族材料；本轮不声称语言机制已经闭合。核验文件：`{out / "analysis/terminal_audit.json"}`。\n'
marker='**Phase2635 交付与连续研究核验 ['
if marker in MEMO.read_text(encoding='utf-8-sig'):raise RuntimeError('terminal supplement already recorded')
with MEMO.open('a',encoding='utf-8') as f:f.write(text)
print(json.dumps({'checks':checks,'all_checks_passed':True}),flush=True)
