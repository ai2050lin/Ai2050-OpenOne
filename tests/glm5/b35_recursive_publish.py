"""Register B35 results and prepare an ordered production upload bundle."""
import io
import tarfile

import mlp_factor_steer_common as C
from b35_recursive_collect import BLOCK, OUT, PARTS, PUBLIC, RUN, WIDTH, identity
from deploy.research_library import Library, canonical


def main():
    lib = Library(C.ROOT/'distributed_data/research_library')
    ident = identity()
    snapshot = ident['model_snapshot']
    analysis = C.read(OUT/'analysis.json')
    units = C.read_gz(OUT/'units.json.gz')['rows']
    assert analysis['model_snapshot'] == snapshot and [r['unit'] for r in units] == list(range(WIDTH))
    objects = {}

    def put(name, doc):
        body = {**ident, **doc}
        sha = lib.ingest(body, f'{RUN}/{name}', kind='local_report')['sha256']
        objects[sha] = lib.root/'objects'/sha[:2]/f'{sha}.json'
        return sha

    size = 128
    chunks = [
        put(f'units-{start:04d}', {'schema':'b35_recursive_units.v1', 'block':BLOCK,
             'start':start, 'units':units[start:start+size]})
        for start in range(0, WIDTH, size)
    ]
    cases = {}
    old = OUT.parent/'b35_recursive_v1'
    datasets = [
        (k, C.read_gz(OUT/'inputs.json.gz')['sets'][k], C.read_gz(OUT/f'{k}-result.json.gz'))
        for k in PARTS
    ]
    datasets.insert(0, ('pilot_v1', C.read_gz(old/'inputs.json.gz')['sets']['pilot'],
                        C.read_gz(old/'pilot-result.json.gz')))
    for key, inputs, result in datasets:
        assert [r['id'] for r in inputs] == [r['id'] for r in result['rows']]
        rows = [{**r,'outcome':y} for r,y in zip(inputs,result['rows'])]
        cases[key] = {'n':len(rows),'chunk_size':32,'chunks':[
            put(f'{key}-cases-{start}', {'schema':'b35_recursive_cases.v1','part':key,'start':start,
                 'cases':rows[start:start+32]}) for start in range(0,len(rows),32)]}
    design = put('frozen-design', {'schema':'b35_recursive_design.v1',
          'prereg':C.read(OUT/'prereg.json'),'analysis_freeze':C.read(OUT/'analysis-execution.json'),
          'v1_prereg':C.read(old/'prereg.json'),'v1_pilot_case_index':cases['pilot_v1'],
          'code_ids':C.read_gz(OUT/'inputs.json.gz')['code_ids'],
          'source_code':[{ 'path':path,'sha256':C.file_sha(C.ROOT/path),'text':(C.ROOT/path).read_text(encoding='utf-8')}
                         for path in ['tests/glm5/b35_recursive_collect.py','tests/glm5/b35_recursive_analyze.py']]})
    overview = put('b35-feature-arrays', {'schema':'b35_recursive_feature_arrays.v1','block':BLOCK,
        'unit_count':WIDTH,'chunk_size':size,'chunks':chunks,'natural':analysis['natural'],
        'plots':{'natural_std':[round(r['std'],6) for r in units],
                 'lockbox_sign':[round(r['lockbox']['same_sign']/r['lockbox']['n'],6) for r in units],
                 'replicated':[int(r['replicated']) for r in units],
                 'lockbox_gain':[round(r['lockbox']['prediction']['gain_vs_mean'],6) for r in units]}})
    report = put('report', {**analysis,'schema':'b35_recursive_report.v1','feature_index_sha256':overview,
        'case_indexes':cases,'prereg_sha256':design,'boundary':'mlp.down_proj.input',
        'poles_scope':'每单元自然语料正负极值各一段，统计覆盖 19,042,014 个有效 token。',
        'evidence_scope':'方向重复来自金标准前缀的真实 B35 一阶输出贡献。逐步自由生成单独评分。相邻层投影是候选，原生数组 SHA 可追溯到采集节点。',
        'limitations':['原生模型保持未编辑。单元作用、行为必要性和开放任务语义仍待干预验证。',
                       '六步试验更换循环和文档，沿用两种提示样式、字母词表与相同转移规则。',
                       '接续预测读取原模型 B34 激活，属于局部预测；逐步贡献为连续近似，原模型在 BF16 运算。',
                       '完整 B34/B35 原生张量保存在采集节点；中心逐单元保存分析与自然极值、逐题输入与输出。']})
    cards, edges, anchors = [], [], [0, WIDTH-1]
    anchors += sorted((u for u in range(WIDTH) if units[u]['behavior_eligible']),
                      key=lambda u: (-units[u]['lockbox']['same_sign'], u))[:4]
    seen = set()

    def add_card(block, unit, level, artifact, label):
        key = f'{snapshot}:{RUN}:B{block}:U{unit}'
        if key not in seen:
            seen.add(key)
            cards.append({'id':key,'run_id':RUN,'layer':block,'boundary':'mlp.down_proj.input',
                          'index':{'space':'mlp','unit':unit},'label':label,
                          'method':'全层普查及短链发现、独立短链确认、六步留出。B34 来源按原权重余弦定位。',
                          'evidence':{'level':level,'artifact_sha256':artifact},
                          'counterexamples':[{'artifact_sha256':report,'key':'error_cases'}],
                          'scope':{'feature_index_sha256':overview,'source_is_candidate':block==34}})
        return key

    for u in dict.fromkeys(anchors):
        row = units[u]
        ref = chunks[u//size]
        level = 'heldout' if row['behavior_eligible'] else 'observation'
        target = add_card(35,u,level,ref,f'B35/U{u} · ' + ('短链到长链方向复现' if level=='heldout' else '原生单元特征'))
        src = row['upstream_prediction_unit']
        upstream = add_card(34,src,'parameter_source',ref,f'B34/U{src} · 写入对齐候选')
        edges.append({'id':C.digest([upstream,target]),'source':upstream,'target':target,'run_id':RUN,
             'relation':'adjacent_weight_alignment_candidate','evidence':{'level':'parameter_source','artifact_sha256':ref},
             'scope':{'intervened':False,'unit_prediction':row['lockbox']['prediction']}})
    graph = put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges,
         'full_native_unit_records':WIDTH,'feature_index_sha256':overview})
    binding = put('space-binding',{'schema':'research_space_binding.v1','report_sha256':report,'graph_sha256':graph,
         'traces':[],'context':{'study_type':'b35_recursive_features','native_bank':True,
                               'model_dimensions':{'n_layers':36,'d_ff':WIDTH,'d_model':2560,'n_heads':32},
                               'feature_index_sha256':overview}})
    work = put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B B35 · 全部神经元特征与递归泛化',
        'stages':{'b35_features':{'status':'done','unit_count':WIDTH,'block':BLOCK,'index_sha256':overview,
                                  'report_sha256':report},'recursive_holdout':{'status':'done',
                 'behavior_gate':analysis['behavior_gate']['passed'],'eligible_units':analysis['eligible_units']},
                 'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graph}}})
    registration = {'run_id':RUN,'model_snapshot':snapshot,'report_sha256':report,'graph_sha256':graph,
                    'binding_sha256':binding,'workpackage_sha256':work}
    out = C.ROOT/'release/ai2050-research-qwen3-4b-b35.tar.gz'
    tmp = out.with_suffix('.tmp')
    manifest = {'schema':'ai2050_research_upload.v1','objects':[{'sha256':sha} for sha in objects],
                'studies':[registration]}
    with tarfile.open(tmp,'w:gz',compresslevel=5) as tar:
        raw = canonical(manifest)
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(raw)
        tar.addfile(info,io.BytesIO(raw))
        for sha,path in objects.items():
            assert C.file_sha(path) == sha and path.stat().st_size < 16*2**20
            tar.add(path,arcname=f'objects/{sha}.json')
    tmp.replace(out)
    sha = C.file_sha(out)
    out.with_suffix('.gz.sha256').write_text(f'{sha}  {out.name}\n',encoding='ascii')
    doc = {**registration,'feature_index_sha256':overview,'objects':len(objects),'upload_sha256':sha,
           'upload_bytes':out.stat().st_size}
    PUBLIC.mkdir(parents=True,exist_ok=True)
    C.save(PUBLIC/'registration.json',doc)
    C.save(OUT/'registration-final.json',doc)
    print(doc,flush=True)


if __name__ == '__main__':
    main()
