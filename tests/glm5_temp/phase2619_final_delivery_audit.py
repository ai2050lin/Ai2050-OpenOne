"""Final build, continuous-record check, and evidence interpretation supplement."""
import json,re,subprocess,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import phase2605_c676097_c692480_singleprompt_source_patch as io
RESULT=ROOT/'tests/glm5/result';OUT=RESULT/'phase2619_fresh_bidirectional_operator_confirmation'
model_result=io.load_json(OUT/'analysis/final.json');api=io.load_json(RESULT/'phase2618_delivery_storage_and_continuation_audit/analysis/client_api_checks.json')
node=Path(r'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
build=subprocess.run([str(node),'node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';content=memo.read_text(encoding='utf-8-sig');headings=[int(v) for v in re.findall(r'^## Phase (\d+):',content,re.M)]
all_finals={p:io.load_json(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2600,2620)}
asset=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
rows=io.read_jsonl(OUT/'behavior/learned.jsonl');by_language={l:{'n':sum(r['language']==l for r in rows),'donor_answers':sum(r['language']==l and r['donor_correct'] for r in rows)} for l in ('en','zh')}
checks={'all20_phase_records': [v for v in headings if 2600<=v<=2619]==list(range(2600,2620)),
        'all20_procedural_checks':all(f['all_checks_passed'] for f in all_finals.values()),'frontend_build':build.returncode==0,
        'client_api_and_all11_panels':api['all_checks_passed'] and api['checks'].get('all11new_panels',False),
        'api_test_matches_current_asset':api['asset_sha256']==io.sha256(asset)}
result={'timestamp':datetime.now().astimezone().isoformat(),'checks':checks,'all_checks_passed':all(checks.values()),'build_stdout':build.stdout,'build_stderr':build.stderr,
        'by_language_learned':by_language,'generation_condition_runs_this_continuation':7680,'independent_samples_not_7680':True,'automatic_followups':[2615,2617,2619]}
io.save_json(OUT/'analysis/delivery_audit.json',result)
if not result['all_checks_passed']:
    print(json.dumps(result,ensure_ascii=True))
    raise RuntimeError(checks)
marker='**Phase2619交付核验与科学解读补记（append-only）**'
if marker not in content:
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=f'\n\n{marker} [{stamp}] 全部Phase2600—2619连续、20份程序检查通过；本次续跑7680次生成条件运行，不是7680个独立样本。客户端11个新增面板、HTTP范围读取、完整坐标维度和前端构建检查={json.dumps(checks,ensure_ascii=False)}。构建有原有大bundle警告，不影响构建通过。\n\n'
    text+=('扩大确认全体240pair：训练方向177/240=73.75%，三个roll为35/240、37/240、46/240；原baseline已有47/240输出另一侧答案，故必须同时看配对条件。双侧自然都正确的162pair中，训练方向126/162=77.78%，三个roll为1/162、5/162、6/162。该子集结果受行为成功筛选，不能替代全体。英语训练方向112/120=93.33%；中文65/120=54.17%且表述与方向高度不对称，不能宣布中文通用规则通过。\n\n')
    text+=('一个重要拼图是：Phase2609同义词的全向量余弦较低，并不意味着不存在可复用的功能方向；Phase2619英语三个表述的早层训练方向仍然有效。这提示“全场几何相似”与“模型使用某个方向”必须分别测量，但尚未定位产生该功能的具体坐标协同机制。FP32 source存储校准总计75,776,000个采集坐标未发现FP16导出变零或溢出，最大舍入误差约2.98e-8；不能把事前数值风险误写成已发生严重崩解。后续应围绕这一通过的早层方向追踪跨位置、跨层的条件载体，分离关系身份、词形与事件身份，不以中文或单站点失败抛弃英语阳性。\n')
    with memo.open('a',encoding='utf-8') as f:f.write(text)
print(json.dumps({'checks':checks,'all_checks_passed':result['all_checks_passed'],'by_language_learned':by_language},ensure_ascii=True))
assert result['all_checks_passed']
