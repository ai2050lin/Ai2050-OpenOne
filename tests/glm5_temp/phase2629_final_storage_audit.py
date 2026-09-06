"""Append-only delivery audit and file-only cleanup of completed unshown raw fields."""
import sys,re,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'));sys.path.insert(0,str(Path(__file__).parent))
from phase2620_native_coordinate_contract import *
from phase2629_delivery_checks import run_checks

def main():
    out=RESULT/'phase2629_expanded_native_confirmation';delivery=run_checks()
    if not delivery['all_checks_passed']:raise RuntimeError('client QA failed; do not clean raw fields')
    headings=[int(v) for v in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
    finals={p:list(RESULT.glob(f'phase{p}_*/analysis/final.json')) for p in range(2620,2630)}
    assert all(len(paths)==1 for paths in finals.values())
    checks={'all10_contiguous_phases':[p for p in headings if 2620<=p<=2629]==list(range(2620,2630)),
        'all_procedural_checks':all(read(paths[0])['all_checks_passed'] for paths in finals.values()),'client_api_and_build':delivery['all_checks_passed']}
    if not all(checks.values()):raise RuntimeError(checks)
    source=RESULT/'phase2622_unmodified_native_fields';root=(source/'field/fulltoken').resolve()
    manifest_path=out/'analysis/completed_unpublished_raw_cleanup.json'
    if manifest_path.exists():manifest=read(manifest_path)
    else:
        manifest=[]
        for row in read(source/'analysis/raw_manifest.json'):
            if row['published_exemplar']:continue
            p=Path(row['path']).resolve(strict=True)
            if p.parent!=root or not re.fullmatch(r'case_\d{4}\.float32\.npy',p.name) or p.is_symlink():raise RuntimeError(f'unsafe target {p}')
            manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'case_id':row['case_id'],
                'reason':'full-token audit completed; this raw exemplar not published; complete coordinate profiles, scalar-query fields, scripts and material retained'})
        save(manifest_path,manifest)
    for row in manifest:
        p=Path(row['path']).resolve()
        if p.parent!=root or not p.is_relative_to(RESULT.resolve()) or p.suffix!='.npy':raise RuntimeError(p)
        if p.exists():
            if sha(p)!=row['sha256']:raise RuntimeError('raw field changed after manifest')
            p.unlink()
    removed=sum(r['bytes'] for r in manifest)
    retained=[r for r in read(source/'analysis/raw_manifest.json') if r['published_exemplar']]
    checks['all64_published_fulltoken_exemplars_retained']=len(retained)==64 and all(Path(r['path']).exists() for r in retained)
    checks['all_manifest_files_removed']=all(not Path(r['path']).exists() for r in manifest)
    result={'checks':checks,'all_checks_passed':all(checks.values()),'removed_files':len(manifest),'removed_bytes':removed,
        'recovery':'not in recycle bin; regenerate from retained script/material/local BF16 model; CPU numerical library/version can affect bitwise reproducibility',
        'client_asset_sha256':delivery['asset_sha256'],'next_goal_same':True,'automation_id':'automation','mechanism_closed':False}
    save(out/'analysis/final_storage_delivery.json',result)
    r29=read(out/'analysis/final.json');native=r29['summary']['native_objective_actual_forward']
    note='\n\n**Phase2629最终交付与存储补记（append-only）** ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+'] '
    note+=f'Phase2620—2629十阶段连续；新阶段及扩大确认均有产物，程序检查={json.dumps(checks,ensure_ascii=False)}。客户端新增14面板，四模型真实权重标量查询、最后边界索引、非法索引拒绝、词嵌入/HiddenState全token单坐标、HTTP范围读取与前端构建已测试。未做浏览器手动视觉交互验收；构建已有大bundle警告。\n\n'
    note+=f'完成后按核验清单删除{len(manifest)}个未发布fulltoken原场，释放{removed} bytes，保留64个可查全token例，以及全部768+768边界/神经元全坐标、模型、脚本、行为结果和图谱。删除未进回收站，不能直接撤销；可按保留脚本重建。manifest=`{manifest_path}`。\n\n'
    note+='**科学裁决。** 本轮提供的是原生单参数可计算工具、精度边界和条件指纹，不是破解语言机制。实数公式准确是架构记账；BF16下许多小效应被舍入淹没。首token、输出标签和完整语言行为仍必须分账。native实际前向汇总='+json.dumps(native,ensure_ascii=False)+'。\n\n'
    note+='**下一完整研究方向。** 目标相同，既有每小时自动续研已更新为原生坐标/神经元/参数主线，不另起并行GPU任务。下一阶段应沿真实生成的多token轨迹选择模型自己的输出分叉，并将最后层已验证的读取公式向早中层实际Q/K/V与MLP计算延伸；优先比较相同输出身份下的不同语言操作及不同输出身份下的相同操作，建立可预测的跨层条件响应。需要位移足以越过数值分辨率而不过度偏离自然状态；低幅值单位看敏感度与有限效应两套账。不能把最终层高效应单位自动当语言概念。\n'
    with MEMO.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
