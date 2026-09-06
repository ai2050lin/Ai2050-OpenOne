"""Append-only terminal proof, complete retention audit and existing-task continuation."""
import importlib.metadata as md,sys,tomllib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2669_symmetric_multitoken_delivery import OUT,ASSET


def main():
    text=MEMO.read_text(encoding='utf-8-sig');headings=[int(v) for v in re.findall(r'^## Phase (\d+):',text,re.M)]
    finals={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2662,2670)}
    clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    config=tomllib.loads(Path(r'C:\Users\Admin\.codex\automations\automation\automation.toml').read_text(encoding='utf-8'))
    checks={'eight_contiguous_phases':headings[-8:]==list(range(2662,2670)),
        'no_duplicate_campaign_heading':all(headings.count(p)==1 for p in range(2662,2670)),
        'eight_checks_passed':all(r['all_checks_passed'] for r in finals.values()),
        'no_semantic_closure_overclaim':all(not r['language_mechanism_closed'] for r in finals.values()),
        'all224_retained_hashes':len(clean['kept'])==224 and all(sha(r['path'])==r['sha256'] for r in clean['kept']),
        'all9504_deleted_manifest_targets':len(clean['targets'])==9504 and all(not Path(r['path']).exists() for r in clean['targets']),
        'asset_hash_exact':sha(ASSET)==pub['asset_sha256'],'ten_new_panels':len(pub['panels'])==10,
        'new_memo_unicode':'\ufffd' not in text[text.index('## Phase 2662:'):],
        'same_task_reads_authoritative_latest_MEMO':config['status'] in ('ACTIVE','PAUSED') and config['kind']=='heartbeat' and config['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500' and '先读' in config['prompt'] and '实际MEMO' in config['prompt'] and '\ufffd' not in config['prompt']}
    next_plan=read(OUT/'analysis/next_campaign.json');checks['next_2670_to2676_plan_present']=next_plan['frontier']==2669 and [r['phase'] for r in next_plan['plan']]==list(range(2670,2677))
    before=read(OUT/'analysis/delivery_checks.json');after=read(OUT/'analysis/post_cleanup_checks.json');live=read(OUT/'analysis/live_api_checks.json')
    checks.update(client_build=before['all_checks_passed'],post_cleanup_api=after['all_checks_passed'],live_http=live['all_checks_passed'],independent_science=read(OUT/'analysis/scientific_checks.json')['all_checks_passed'],actual_browser_checks=read(OUT/'analysis/browser_checks.json')['all_checks_passed'])
    cp=subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--',
        'frontend/src/App.jsx','frontend/src/researchKernel/heatmapResearchRoute.js','research/glm5/docs/AGI_GLM5_MEMO.md','server/research_asset_service.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    checks['scoped_diff_check']=cp.returncode==0
    packages={p:md.version(p) for p in ('torch','transformers','numpy','accelerate')}
    report={'checks':checks,'all_checks_passed':all(checks.values()),'packages':packages,'automation_id':config['id'],'automation_status':config['status'],
        'automation_prompt_update_confirmed':'2670' in config['prompt'],'automation_tool_limitation':'Two supported updates returned success but persisted prompt remained old. Do not claim prompt update succeeded. Existing prompt explicitly says read latest MEMO and use actual MEMO frontier; this verified handoff path is used without editing automation TOML or creating a duplicate.',
        'automation_prompt_sha256':hashlib.sha256(config['prompt'].encode('utf-8')).hexdigest(),
        'phase2669_memo_line':next(i for i,line in enumerate(text.splitlines(),1) if line.startswith('## Phase 2669:')),
        'diff_check':{'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}}
    save(OUT/'analysis/terminal_audit.json',report);assert report['all_checks_passed']
    marker='**Phase2669 最终复核补充**'
    if marker not in text:
        note=f'\n\n{marker} [{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}]。2662—2669八个Phase连续完整；{len(before["checks"])}项客户端与生产构建检查、{len(after["checks"])}项清理后检查、{len(live["checks"])}项真实HTTP检查通过，浏览器验收另行分账。10个新全坐标热力图面板、64个多token逐参数实例、64个短码实例和32个14B实例已发布，224个保留原包逐个哈希复核。仅删除清单内9504个未展示原包，共{clean["deleted_bytes"]:,}字节，直接删除而非回收站；模型、材料、行为、代码、全坐标派生图和重建依据保留。既有同任务自动续研以2669为完成边界、2670为下一编号，未建立重复任务。运行时='+json.dumps(packages,ensure_ascii=False)+'。\n'
        note+=f'自动任务当前状态为{config["status"]}。本轮观察到计划一度暂停，随后又恢复ACTIVE及10分钟周期；交付时保留当前实际设置，未创建重复自动任务。下一批尚未执行，不能把预定方案记作研究完成。实际桌面浏览器已验证10种类型入口、整行H/MLP末坐标及多token真实参数与分項，见analysis/browser_checks.json；并非对全部历史3D面板或移动端作视觉穷举。\n'
        note+='\n**下一次自动续研的最新执行入口。** 2662—2669已完成，下一个连续编号为2670。自动化工具两次回报成功但保存提示未更新，本记录如实注明此限制；不直接改写automation.toml或建立重复任务。现有已启用提示明确要求先读最新MEMO且以实际MEMO为准，因此采用本尾部和analysis/next_campaign.json作为交接依据，不能重复执行2662—2669或读取已清理原包。\n\n'
        note+='\n'.join(f'- Phase {p["phase"]}：{p["task"]}' for p in next_plan['plan'])+'\n\nQwen14原坐标全部62H/125MLP候选及精确掩码哈希见next_campaign.json；冻结其64组方向门的适用范围，不混同4B的q0/m0幅度确认门。自动研究仍须每批完成后依据新证据评估同目标接续，不将计划当已完成实验。\n'
        with MEMO.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
