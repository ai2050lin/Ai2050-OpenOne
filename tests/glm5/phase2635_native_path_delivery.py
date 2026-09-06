"""Publish full physical-coordinate rows; retain complete exemplar fields and scalar factors."""
import argparse
import math
import shutil
import sys
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET, panel, row
from phase2632_fulltoken_native_adjoints import INPUT_KEY, LAYERS, MODULES

SOURCE=RESULT/'phase2632_fulltoken_native_adjoints'
OUT=RESULT/'phase2635_expanded_native_path_confirmation'

def publish():
    from transformers import AutoTokenizer
    frames=read(SOURCE/'material/frames.json')
    selected=[f for f in frames if f['index']==12 and f['variant']==0]
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/hf/qwen3-4b'),local_files_only=True)
    save(OUT/'material/client_token_strings.json',{f['frame_id']:tokenizer.convert_ids_to_tokens(f['prefix_ids']) for f in selected})
    def field(name):return np.load(SOURCE/f'field/{name}.float32.npy',mmap_mode='r',allow_pickle=False)
    H=field('hidden_positions');HG=field('hidden_adjoint_positions');A=field('mlp_positions');AG=field('mlp_adjoint_positions')
    hidden=[];hgrad=[];neurons=[];ngrad=[];kv=[];q=[];wr=[];dr=[]
    weight_examples={f['frame_id'] for f in selected if f['step']==0 and f['family']=='chronology'}
    for f in selected:
        # Visual examples are declared separately from full analysis and the 34-prefix API.
        # Limit row count, never physical coordinate width; all 16 family/language groups remain.
        if f['step']!=0:continue
        i=f['frame_id'];label=f'frame{i}/{f["case_id"]}/step{f["step"]}'+('/EOS' if f['eos'] else '')
        for l in (0,1,6,18,36):
            for p,pname in ((0,'anchor'),(2,'current_boundary')):
                hidden.append(row(f'{label}/{pname}/rawH{l}',H[i,l,p],2635,'embedding' if l==0 else 'raw_hidden_coordinate',l))
                hgrad.append(row(f'{label}/{pname}/dmargin_dH{l}',HG[i,l,p],2635,'hidden_local_adjoint',l))
        for l in LAYERS:
            neurons.append(row(f'{label}/L{l}/a',A[i,l,2],2635,'mlp_intermediate_neuron',l))
            ngrad.append(row(f'{label}/L{l}/dmargin_da',AG[i,l,2],2635,'mlp_local_adjoint',l))
        with np.load(SOURCE/f'field/factors/frame_{i:04d}.npz',allow_pickle=False) as pack:
            for l in LAYERS:
                for name in ('q_proj','k_proj','v_proj'):
                    if l not in (0,35):continue
                    rr=q if name=='q_proj' else kv
                    key=f'L{l}_{name}';output=pack[key+'__value'];gradient=pack[key+'__g']
                    # Native original source and current boundary positions, not a learned alignment.
                    for t in (f['anchor_positions'][-1],len(f['prefix_ids'])-1):
                        rr.append(row(f'{label}/{key}/token{t}/projection',output[t],2635,'native_projection_output',l))
                        rr.append(row(f'{label}/{key}/token{t}/adjoint',gradient[t],2635,'native_projection_adjoint',l))
                for name,destination in (('v_proj',wr),('down_proj',dr)):
                    if i not in weight_examples:continue
                    key=f'L{l}_{name}';W=np.load(SOURCE/f'field/weights/{key}.float32.npy',mmap_mode='r',allow_pickle=False)
                    x=pack[f'L{l}_{INPUT_KEY[name]}'].astype('float64');g=pack[key+'__g'].astype('float64')
                    for j in (0,W.shape[0]-1):
                        if i==selected[0]['frame_id']:
                            destination.append(row(f'{key}/actual_W_row_j{j}',W[j],2635,'learned_scalar_weight_row',l))
                        destination.append(row(f'{label}/{key}/alltoken_dW_row_j{j}',g[:,j]@x,2635,'shared_weight_alltoken_derivative',l))
                        destination.append(row(f'{label}/{key}/lastonly_dW_row_j{j}',g[-1,j]*x[-1],2635,'shared_weight_lastonly_approximation',l))
    spec=[('hidden','Native generation: full embedding and HiddenState coordinates',2560,hidden,'Original physical j; raw block outputs, not final normalization.'),
          ('hidden_adjoint','Native output contrast: all HiddenState adjoints',2560,hgrad,'Original physical j; local derivatives, not semantic necessity.'),
          ('mlp','Native generation: all MLP intermediate neurons',9728,neurons,'Original physical k; SwiGLU activation a[k], not learned scalar weight.'),
          ('mlp_adjoint','Native generation: all MLP neuron adjoints',9728,ngrad,'Original physical k; derivative depends on native output pair and prefix.'),
          ('kv','Native K/V: source and boundary projection coordinates',1024,kv,'Original grouped-query K/V projection coordinates; no alignment across models.'),
          ('q','Native Q: source and boundary projection coordinates',4096,q,'Original Q projection coordinates, distinct from residual dimensions.'),
          ('v_weights','Real V weights: all-token vs last-position scalar derivatives',2560,wr,'Each labelled j row contains every input k; all rows j accessible through scalar query.'),
          ('down_weights','Real down weights: all-token vs last-position scalar derivatives',9728,dr,'Each labelled j row contains every MLP neuron k; early layers need all-token summation.')]
    payload=read(ASSET);previous_sha=read(OUT/'analysis/publication.json')['previous_asset_sha256'] if (OUT/'analysis/publication.json').exists() else sha(ASSET)
    panels=[panel('phase2635_'+key,name,d,rows,meaning) for key,name,d,rows,meaning in spec]
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2635_')]+panels
    payload['phase']=2635
    payload['claim_boundary']='Native all-coordinate activations and adjoints, shared scalar-weight contributions summed over every token. Exact chain-rule factors are not low-rank approximation. BF16 finite interventions do not universally match derivatives; not semantic closure.'
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']}
    payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    report={'previous_asset_sha256':previous_sha,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,
        'published_frames':[f['frame_id'] for f in selected],'new_panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in panels],
        'visual_examples':'all16 groups step0; Q/K/V display L0,L35; scalar-weight rows display chronology en/zh. The full analysis uses all415 frames and28 sites; scalar API retains all34 exemplar prefixes/28 sites and every coordinate.',
        'query':'/api/research-assets/native-path-parameter?frame=0&layer=35&module=v_proj&j=0&k=0'}
    save(OUT/'analysis/publication.json',report);print(json.dumps(report),flush=True)
    return report

def cleanup():
    # Only explicitly manifested unshown raw files, after completed confirmation and QA.
    assert read(OUT/'analysis/confirmation.json')['all_checks_passed']
    assert read(OUT/'analysis/delivery_checks.json')['all_checks_passed']
    assert read(RESULT/'phase2634_native_output_condition_maps/analysis/final.json')['all_checks_passed']
    manifest=read(SOURCE/'analysis/raw_manifest.json');published=set(read(OUT/'analysis/publication.json')['published_frames'])
    allowed={(SOURCE/'field/hidden').resolve(),(SOURCE/'field/factors').resolve()}
    targets=[];kept=[]
    for record in manifest:
        path=Path(record['path']).resolve()
        if path.parent not in allowed or not re.fullmatch(r'frame_\d{4}\.npz',path.name):raise RuntimeError(f'unexpected target {path}')
        if not path.is_relative_to(RESULT.resolve()):raise RuntimeError('target escaped result root')
        if record['published']:
            assert record['frame_id'] in published and path.is_file()
            kept.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
        else:
            assert record['frame_id'] not in published and path.is_file() and path.stat().st_size==record['bytes']
            targets.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
    assert len(targets)==762 and len(kept)==68
    audit={'timestamp':datetime.now().astimezone().isoformat(),'reason':'completed unshown raw hidden and adjoint/factor packs; all published exemplars, all-case coordinate arrays, full weights and summaries retained',
        'raw_manifest_sha256':sha(SOURCE/'analysis/raw_manifest.json'),'targets':targets,'kept':kept,
        'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'recoverability':'not in Recycle Bin; regenerate from saved local models, code and materials',
        'before_free_bytes':shutil.disk_usage(SOURCE).free}
    save(OUT/'analysis/cleanup_plan.json',audit)
    for r in targets:Path(r['path']).unlink()
    audit['after_free_bytes']=shutil.disk_usage(SOURCE).free
    audit['all_targets_absent']=all(not Path(r['path']).exists() for r in targets)
    audit['all_published_present']=all(Path(r['path']).is_file() for r in kept)
    save(OUT/'analysis/cleanup_completed.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('targets','kept')}),flush=True)

def scientific_audit():
    behavior=[json.loads(line) for line in (RESULT/'phase2631_native_generation_paths/behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()]
    punctuation=[r for r in behavior if r['family']=='punctuation']
    aliases=[]
    for r in punctuation:
        expected='?' if r['variant']==0 else '!'
        alias=r['generated'].strip().replace('？','?').replace('！','!')==expected
        aliases.append({'case_id':r['case_id'],'generated':r['generated'],'original_strict_correct':r['strict_correct'],'exact_symbol_content_correct':bool(r['answer_correct'] or alias)})
    pairs=read(RESULT/'phase2634_native_output_condition_maps/analysis/fullmatrix_pair_comparisons.json')
    cross=[r for r in pairs if r['class']=='same_output_different_family']
    frames={f['frame_id']:f for f in read(SOURCE/'material/frames.json')}
    low=read(RESULT/'phase2634_native_output_condition_maps/analysis/low_magnitude_allcoordinate.json')
    low_summary={}
    for name in ('hidden','mlp'):
        values=np.asarray([v[name+'_quartile_energy_by_layer'] for v in low.values()])
        low_summary[name]={'group_layer_min_lower_half_gradient_energy':float((values[:,:,:2].sum(-1)).min()),
                           'group_layer_max_lower_half_gradient_energy':float((values[:,:,:2].sum(-1)).max()),
                           'meaning':'Descriptive squared-adjoint energy in coordinate amplitude strata; not causal importance, information content, or deletion necessity.'}
    interventions=read(OUT/'analysis/interventions.json')
    v_groups={}
    for group in sorted({r['family']+'/'+r['language'] for r in interventions}):
        rr=[r for r in interventions if r['kind']=='shared_weight' and r['layer']==35 and r['module']=='v_proj' and r['family']+'/'+r['language']==group]
        denominator=sum(abs(r['margin32_change']) for r in rr)
        strong=[r for r in rr if abs(r['margin32_change'])>=.05]
        v_groups[group]={'n':len(rr),'full_l1_error':sum(r['full_absolute_error32'] for r in rr)/denominator if denominator else None,
                        'boundary_l1_error':sum(r['boundary_absolute_error32'] for r in rr)/denominator if denominator else None,
                        'n_effect_ge_0.05':len(strong),'sign_matches_ge_0.05':sum(np.sign(r['margin32_change'])==np.sign(r['predicted_full']) for r in strong)}
        v_groups[group]['sign_matches_ge_0.05']=int(v_groups[group]['sign_matches_ge_0.05'])
    result={'punctuation':{'n':len(aliases),'strict_correct':sum(r['original_strict_correct'] for r in aliases),
                          'exact_symbol_content_correct':sum(r['exact_symbol_content_correct'] for r in aliases),'cases':aliases,
                          'boundary':'Post-hoc separate content score; original requested-word protocol remains unchanged.'},
            'cross_family_output_matching':{'pairs':len(cross),'unique_left_frames':len({r['frame_a'] for r in cross}),
                'unique_right_frames':len({r['frame_b'] for r in cross}),
                'family_pairs':sorted({(r['family_a'],r['family_b']) for r in cross}),
                'language_pairs':sorted({(frames[r['frame_a']]['language'],frames[r['frame_b']]['language']) for r in cross}),
                'all_are_actual_different_families':all(r['family_a']!=r['family_b'] for r in cross),
                'boundary':'The code group key includes family AND language. Actual four selected cross-group pairs are word_sense versus taxonomy, sharing one right-hand frame; not four independent family replications.'},
            'low_amplitude_adjoints':low_summary,'expanded_late_v_by_group':v_groups,
            'native_output_scope':'non_eos means not EOS, not necessarily content: punctuation, formatting and subword alternatives remain. Step0 interventions do not validate full generated-sequence behavior.',
            'checks':{'all24_punctuation_content_symbols':len(aliases)==24 and all(r['exact_symbol_content_correct'] for r in aliases),
                      'all16_expanded_v_groups':len(v_groups)==16,'finite_low_energy':all(math.isfinite(v) for r in low_summary.values() for k,v in r.items() if k!='meaning')}}
    result['all_checks_passed']=all(result['checks'].values())
    save(OUT/'analysis/scientific_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('expanded_late_v_by_group','punctuation')},ensure_ascii=True),flush=True)
    return result

def finalize():
    confirmation=read(OUT/'analysis/confirmation.json');publication=read(OUT/'analysis/publication.json')
    cleanup_report=read(OUT/'analysis/cleanup_completed.json');qa=read(OUT/'analysis/delivery_checks.json');post=read(OUT/'analysis/post_cleanup_checks.json')
    audit=scientific_audit()
    summary={'expanded_single_weight_confirmation':confirmation['summary'],'client':publication,
        'scientific_audit':{k:v for k,v in audit.items() if k not in ('punctuation','expanded_late_v_by_group')},
        'punctuation_score_correction':{k:v for k,v in audit['punctuation'].items() if k!='cases'},
        'cleanup':{k:cleanup_report[k] for k in ('deleted_files','deleted_bytes','after_free_bytes','recoverability')},
        'next_same_goal_campaign':'native early-layer propagation: separate BF16 intermediate rounding from local curvature with aligned FP32 numerical controls on a bounded smaller subset; expand same-output cross-family natural operations, preserve whole-coordinate maps; no donor, no semantic-necessity-only stopping rule'}
    checks={**confirmation['checks'],'scientific_audit':audit['all_checks_passed'],'client_and_build':qa['all_checks_passed'],'post_cleanup_query':post['all_checks_passed'],
        'all_published_fields_retained':cleanup_report['all_published_present'],'all_manifested_unshown_fields_cleaned':cleanup_report['all_targets_absent'],
        'previous_phases2630_to2634_complete':all(read(next(RESULT.glob(f'phase{p}_*/analysis/final.json')))['all_checks_passed'] for p in range(2630,2635))}
    assert all(checks.values())
    finish(2635,'扩大到7200单权重条件、全token逐参数客户端与完整交付',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '冻结2633全部层、矩阵、剂量和选点算法，扩大到index32双变体32前缀；读取实际共享权重，逐token列出输入和伴随乘积。完成数值阳性与阴性复核，再发布、测试和按清单清理未展示原场。',
        r'\frac{\partial m}{\partial W_{jk}}=\sum_t\bar Y_{t,j}X_{t,k};\quad E_{L1}=\frac{\sum_i|\Delta m_i-G_i\Delta W_i|}{\sum_i|\Delta m_i|};\quad m_{32}=\langle h_{BF16},U_y-U_z\rangle_{FP32}.',
        '八族中英各两个变体，共32实际前缀；每例28矩阵×2索引规则×2幅度×2方向+no-op=225，即7200条件，另有每例一次基线前向。加2633累计10800条件/10752真实权重扰动；模型权重只内存修改并全部恢复。客户端8新面板保留全物理宽度、34个前缀全部token的28矩阵标量查询。',
        '全token求和是共享参数的真实链式算法，不是donor差分搬运。末层V与末层down分别提供跨位置和末位置退化的数值对照；较早层有限扰动不能普遍由局部导数准确预测。BF16头零变化但同状态FP32头可变，纠正了把输出舍入平台当内部完全无变化的判断。输出身份强烈影响导数相似性，不能把晚层同输出相似直接叫语言主干。2631标点24例全部输出正确符号，严格词语协议仍0/24；不得据此称为标点知识失效。2634四个同输出跨族pair实为word_sense→taxonomy且共用一个右侧前缀，不是四组独立跨族复现。',
        '本批仅Qwen3-4B，不把上一批四模型末层算术复验挪作本批全路径复验。32前缀不是独立新语义词表，最大导数选点用于数值审计而非预测语义重要性。FP32只更换诊断读出，不消除模型内部BF16误差；未测试完整FP32推理，也未分开曲率和内部舍入。客户端完成API和构建验证，不冒充浏览器视觉验收。',
        '合同2630—2635全部完成，但语言机制未破解。同目标后续应围绕已通过的晚层跨位置规律和早层测量瓶颈，分离内部舍入/曲率，补足同输出跨族材料覆盖，再验证多token生成中的稳定复用；现有持续研究调度沿MEMO最新前沿继续，不另起重复模型任务。')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['publish','cleanup','finalize','scientific_audit']);args=parser.parse_args()
    globals()[args.action]()
