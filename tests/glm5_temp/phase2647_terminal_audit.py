"""Final append-only bookkeeping and preserved scientific result checks."""
import importlib.metadata as md,json,re,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2647_matched_operation_delivery import OUT,ASSET

def main():
    text=MEMO.read_text(encoding='utf-8-sig');headings=[int(p) for p in re.findall(r'^## Phase (\d+):',text,re.M)]
    finals={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2641,2648)}
    clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    checks={'seven_contiguous_phases':headings[-7:]==list(range(2641,2648)),
        'seven_all_checks_passed':all(r['all_checks_passed'] for r in finals.values()),
        'no_semantic_closure_overclaim':all(not r['language_mechanism_closed'] for r in finals.values()),
        'retained64_raw_hashes':len(clean['kept'])==64 and all(sha(r['path'])==r['sha256'] for r in clean['kept']),
        'deleted1984_only_manifest_targets':len(clean['targets'])==1984 and all(not Path(r['path']).exists() for r in clean['targets']),
        'asset_hash_exact':sha(ASSET)==pub['asset_sha256'],'no_unicode_replacement_in_new_memo':'\ufffd' not in text[text.index('## Phase 2641:'):]}
    config=tomllib.loads(Path(r'C:\Users\Admin\.codex\automations\automation\automation.toml').read_text(encoding='utf-8'))
    checks['same_task_auto_continuation_active']=config['status']=='ACTIVE' and config['kind']=='heartbeat' and config['target_thread_id']=='01a055ce-c228-7a12-a605-9ec34f0ee500' and '2647' in config['prompt'] and '\ufffd' not in config['prompt']
    packages={p:md.version(p) for p in ('torch','transformers','numpy','accelerate')}
    report={'checks':checks,'all_checks_passed':all(checks.values()),'packages':packages,'automation_id':config['id'],
        'automation_prompt_sha256':__import__('hashlib').sha256(config['prompt'].encode('utf-8')).hexdigest(),
        'phase2647_memo_line':next(i for i,s in enumerate(text.splitlines(),1) if s.startswith('## Phase 2647:')),
        'raw_reconstruction_boundary':'Raw deletions are not reversible via Recycle Bin; recomputation requires preserved prefix/token/readout material and same model/runtime. Completed Phase entrypoints refuse rerun into original destinations; reconstruction must target a separate directory and suppress MEMO append, compare regenerated values rather than overwrite history.'}
    save(OUT/'analysis/terminal_audit.json',report);assert report['all_checks_passed']
    marker='**Phase2647 最终复核补充**'
    if marker not in text:
        with MEMO.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+' ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+']。七个Phase连续且全部完成，原生模型标量修改全部恢复；283项客户端/构建检查及282项清理后检查通过。删除仅限1984个已完成用途的未展示原场，释放52,088,297,728字节（52.09GB），保留64个示例原包且哈希逐个复核。删除非回收站恢复；可按保存材料与代码另目录重算，不覆盖历史，也不保证不同运行时字节相同。第三批1024条件采用流式全坐标汇总，无额外逐例原场需清理。既有同任务自动续研已更新到本批终点，不新建任务。运行时='+json.dumps(packages,ensure_ascii=False)+'。\n\n**智能理论与下一大任务。** 目前能确定的是：同一真实参数在多个token位置共享、条件响应幅度可较稳定复用，而有符号方向强烈依赖条件；不能由此声称固定词义神经元或普遍语义主干。下一大阶段应交叉“语言操作×输出功能”：在人名选择、自然是/否判断、事实续写之间，分别控制实体、句式和输出头行，沿冻结的全坐标包络与已校准逐参数算法测绘早中晚层；保留全部坐标和阴性，独立实体/句式确认后再做真实标量验证。优先解释方向分化来自哪里，不再次停在末层高余弦或必要性失败。\n')
    print(json.dumps({'checks':checks,'phase2647_line':report['phase2647_memo_line']},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
