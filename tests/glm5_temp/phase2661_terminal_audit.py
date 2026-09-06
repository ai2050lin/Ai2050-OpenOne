"""Append-only terminal proof, complete retention audit and existing-task continuation."""
import importlib.metadata as md,sys,tomllib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2661_sequence_coordinate_delivery import OUT,ASSET


def main():
    text=MEMO.read_text(encoding='utf-8-sig');headings=[int(v) for v in re.findall(r'^## Phase (\d+):',text,re.M)]
    finals={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2655,2662)}
    clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    config=tomllib.loads(Path(r'C:\Users\Admin\.codex\automations\automation\automation.toml').read_text(encoding='utf-8'))
    checks={'seven_contiguous_phases':headings[-7:]==list(range(2655,2662)),
        'no_duplicate_campaign_heading':all(headings.count(p)==1 for p in range(2655,2662)),
        'seven_checks_passed':all(r['all_checks_passed'] for r in finals.values()),
        'no_semantic_closure_overclaim':all(not r['language_mechanism_closed'] for r in finals.values()),
        'all144_retained_hashes':len(clean['kept'])==144 and all(sha(r['path'])==r['sha256'] for r in clean['kept']),
        'all8560_deleted_manifest_targets':len(clean['targets'])==8560 and all(not Path(r['path']).exists() for r in clean['targets']),
        'asset_hash_exact':sha(ASSET)==pub['asset_sha256'],'nine_new_panels':len(pub['panels'])==9,
        'new_memo_unicode':'\ufffd' not in text[text.index('## Phase 2655:'):],
        'same_task_continuation_active':config['status']=='ACTIVE' and config['kind']=='heartbeat' and config['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500' and '2661' in config['prompt'] and '2662' in config['prompt'] and '\ufffd' not in config['prompt']}
    before=read(OUT/'analysis/delivery_checks.json');after=read(OUT/'analysis/post_cleanup_checks.json');live=read(OUT/'analysis/live_api_checks.json')
    checks.update(client_build=before['all_checks_passed'],post_cleanup_api=after['all_checks_passed'],live_http=live['all_checks_passed'],independent_science=read(OUT/'analysis/scientific_checks.json')['all_checks_passed'])
    cp=subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--',
        'frontend/src/App.jsx','frontend/src/researchKernel/heatmapResearchRoute.js','research/glm5/docs/AGI_GLM5_MEMO.md','server/research_asset_service.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    checks['scoped_diff_check']=cp.returncode==0
    packages={p:md.version(p) for p in ('torch','transformers','numpy','accelerate')}
    report={'checks':checks,'all_checks_passed':all(checks.values()),'packages':packages,'automation_id':config['id'],
        'automation_prompt_sha256':hashlib.sha256(config['prompt'].encode('utf-8')).hexdigest(),
        'phase2661_memo_line':next(i for i,line in enumerate(text.splitlines(),1) if line.startswith('## Phase 2661:')),
        'diff_check':{'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}}
    save(OUT/'analysis/terminal_audit.json',report);assert report['all_checks_passed']
    marker='**Phase2661 最终复核补充**'
    if marker not in text:
        note=f'\n\n{marker} [{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}]。2655—2661七个Phase连续完整；{len(before["checks"])}项客户端与生产构建检查、{len(after["checks"])}项清理后检查、{len(live["checks"])}项真实HTTP检查通过，不冒称浏览器视觉验收。9个新全坐标热力图面板、64个完整答案逐参数实例和16个14B实例已发布，144个保留原包逐个哈希复核。仅删除清单内8560个未展示原包，共{clean["deleted_bytes"]:,}字节，直接删除而非回收站；模型、材料、行为、代码、全坐标派生图和重建依据保留。既有同任务自动续研以2661为完成边界、2662为下一编号，未建立重复任务。运行时='+json.dumps(packages,ensure_ascii=False)+'。\n'
        with MEMO.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
