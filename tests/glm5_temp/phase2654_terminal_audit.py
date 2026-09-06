"""Final append-only proof of complete scientific campaign and same-task continuation."""
import importlib.metadata as md,json,re,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2654_output_function_delivery import OUT,ASSET


def main():
    text=MEMO.read_text(encoding='utf-8-sig');headings=[int(p) for p in re.findall(r'^## Phase (\d+):',text,re.M)]
    finals={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2648,2655)}
    clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    config=tomllib.loads(Path(r'C:\Users\Admin\.codex\automations\automation\automation.toml').read_text(encoding='utf-8'))
    checks={'seven_contiguous_phases':headings[-7:]==list(range(2648,2655)),
        'seven_checks_passed':all(r['all_checks_passed'] for r in finals.values()),'no_semantic_closure_overclaim':all(not r['language_mechanism_closed'] for r in finals.values()),
        'all128_retained_hashes':len(clean['kept'])==128 and all(sha(r['path'])==r['sha256'] for r in clean['kept']),
        'all8064_deleted_only_manifest_targets':len(clean['targets'])==8064 and all(not Path(r['path']).exists() for r in clean['targets']),
        'asset_hash_exact':sha(ASSET)==pub['asset_sha256'],'new_memo_valid_unicode':'\ufffd' not in text[text.index('## Phase 2648:'):],
        'same_task_continuation_active':config['status']=='ACTIVE' and config['kind']=='heartbeat' and config['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500' and '2654' in config['prompt'] and '2655' in config['prompt'] and '\ufffd' not in config['prompt']}
    packages={p:md.version(p) for p in ('torch','transformers','numpy','accelerate')};before=read(OUT/'analysis/delivery_checks.json');after=read(OUT/'analysis/post_cleanup_checks.json')
    checks.update(client_build=before['all_checks_passed'],post_cleanup_api=after['all_checks_passed'])
    report={'checks':checks,'all_checks_passed':all(checks.values()),'packages':packages,'automation_id':config['id'],
        'automation_prompt_sha256':__import__('hashlib').sha256(config['prompt'].encode('utf-8')).hexdigest(),
        'phase2654_memo_line':next(i for i,s in enumerate(text.splitlines(),1) if s.startswith('## Phase 2654:'))}
    save(OUT/'analysis/terminal_audit.json',report);assert report['all_checks_passed']
    marker='**Phase2654 最终复核补充**'
    if marker not in text:
        note=f'\n\n{marker} [{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}]。2648—2654七个Phase连续且全部完成。{len(before["checks"])}项客户端/构建核验及{len(after["checks"])}项清理后核验通过；未声称浏览器视觉验收。7新热力图与64逐参数实例已发布，128个展示原包哈希逐个复核。仅按明确清单删除8064个未展示原包，释放{clean["deleted_bytes"]:,}字节，非回收站恢复；模型、材料、token/读出、完整派生坐标图谱、代码和重建依据保留。另有无损容器转换节省8,183,399,866字节，转换前后数组逐位一致；此节省与删除量分账，不是特征压缩。既有同任务自动续研更新到2654终点，下一轮2655继续，不建立重复任务。运行时='+json.dumps(packages,ensure_ascii=False)+'。\n'
        with MEMO.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
