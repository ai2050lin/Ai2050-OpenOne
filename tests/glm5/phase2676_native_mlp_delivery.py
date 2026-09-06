"""Publish actual physical arrays, audit conclusions and clean only allowlisted raw fields."""
import argparse,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,LAYERS,SITES
from phase2671_native_mlp_field import unbits
from phase2672_native_mlp_paths import OUT as PATHS
from phase2673_native_mlp_confirmation import OUT as MAPS
from phase2674_native_mlp_scalar import OUT as FP
from phase2675_native_mlp_crossmodel import OUT as CROSS

OUT=RESULT/'phase2676_native_mlp_delivery'
EXPAND=OUT/'expansion'

def publish():
    phases={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2670,2676)};assert all(v['all_checks_passed'] for v in phases.values())
    assert read(EXPAND/'analysis/completion.json')['all_checks_passed']
    assert read(OUT/'numeric_resolution/analysis/completion.json')['all_checks_passed']
    for precision in ('bf16','fp32'):assert read(OUT/f'prefix_replay/{precision}/analysis/completion.json')['all_checks_passed']
    short={r['case_index']:r for r in read(FIELD/'analysis/records.json')};nat={r['case_index']:r for r in read(FP/'natural/analysis/records.json')};fp={r['case_index']:r for r in read(FP/'analysis/records.json')};mat={r['case_index']:r for r in read(FP/'material/cases.json')}
    examples=[{**r,'natural_short':short[r['case_index']],'natural_structured':nat[r['case_index']],'fp_structured_base':fp[r['case_index']]['base'],'fp_structured_prompt':mat[r['case_index']]['prompt']} for r in read(CONTRACT/'material/cases.json') if r['published']]
    save(OUT/'material/published_mlp_cases.json',examples);folder=OUT/'maps/client_panels';folder.mkdir(parents=True,exist_ok=True);catalog=[]
    def panel(key,title,entries):
        metadata=[];arrays=[]
        for label,v in entries:metadata.append({'label':label});arrays.append(np.asarray(v,dtype='float64'))
        a=np.stack(arrays);assert a.ndim==2 and np.isfinite(a).all();key='phase2676_'+key;path=folder/(key+'.npz')
        assert shutil.disk_usage(folder).free>a.nbytes+8*1024**3,('publication_uncompressed_budget',key,a.nbytes)
        np.savez_compressed(path,values=a)
        catalog.append({'key':key,'title':title,'coordinate_count':a.shape[1],'rows':metadata,'matrix_sha256':sha(path)})
    groups={k:[] for k in ('h','a','gate','up','down','x','path')}
    for r in examples:
        with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
            for key in ('h','a'):
                a=unbits(z[key])
                for l in range(len(a)):
                    for pos,label in enumerate(('body','task')):groups[key].append((f'{r["case_id"]}/{key}{l}/{label}',a[l,pos]))
            for key in ('gate','up'):
                a=unbits(z[key]);groups[key].extend((f'{r["case_id"]}/{key}{l}/task',a[l]) for l in range(len(a)))
            for key in ('x','down'):
                a=unbits(z[key]);groups[key].extend((f'{r["case_id"]}/{key}{l}/task',a[i]) for i,l in enumerate(LAYERS))
        with np.load(PATHS/f'field/case_{r["case_index"]:04d}.npz') as z:groups['path'].extend((f'{r["case_id"]}/{key}/all_coordinates',z[key]) for key in z.files)
    for key,title in [('h','所有层词嵌入/HiddenState：正文与任务边界'),('a','所有层实际MLP乘积单元：正文与任务边界'),('gate','所有层 gate 单元原值'),('up','所有层 up 单元原值'),('x','四中层完整归一化输入坐标'),('down','四中层完整MLP输出坐标'),('path','五个冻结单元全部真实参数乘积路径')]:panel(key,title,groups[key])
    del groups
    with np.load(FIELD/'weights/native_candidate_vectors.npz') as z:panel('weights','真实学习权重：完整gate/up输入行与down输出列',[(key,z[key]) for key in z.files])
    for metric in ('h','a'):
        with np.load(MAPS/'maps/old_gate_counts.npz') as z:panel('confirmation_'+metric,'全坐标旧门通过组数（0–64）：'+metric,[(f'{metric}{l}/q0m0/64groups',z[metric][l]) for l in range(len(z[metric]))])
    terms=[]
    for p in sorted((MAPS/'maps').glob('product_*.npz')):
        with np.load(p) as z:
            for key in z.files:
                full=z[key]
                for l in LAYERS:terms.append((f'{p.stem}/{key}/MLP{l}',full[l].copy()))
    panel('product_accounting','全单元乘积展开 v1−v0（对照图为v0−v1，选定层，非唯一因果分配）',terms);del terms
    for model in ('qwen14','glm4','ds7'):
        mcases=[r for r in read(CROSS/model/'material/cases.json') if r['published']];arrays={'h':[],'a':[]};info=read(CROSS/model/'protocol/model.json')
        for r in mcases:
            with np.load(CROSS/model/f'field/case_{r["case_index"]:04d}.npz') as z:
                for key in ('h','a'):
                    a=unbits(z[key]);arrays[key].extend((f'{model}/{r["case_id"]}/{key}{l}/task',a[l,-1]) for l in range(len(a)))
        for key in ('h','a'):panel(model+'_'+key,f'{model} 本模型全部原生'+('词嵌入/H坐标' if key=='h' else 'MLP单元'),arrays[key])
    derivatives=[]
    for r in examples:
        with np.load(FP/f'field/case_{r["case_index"]:04d}.npz') as z:
            for l,j in SITES:
                for kind in ('gate','up','down'):
                    for part in ('all','content','format','eos'):
                        vectors=[]
                        for branch in ('Y','N'):
                            if kind=='down':terms=z[f'{branch}__L{l}_J{j}_a'].astype('float64')[:,None]*z[f'{branch}__L{l}_down_g_{part}'].astype('float64')
                            else:terms=z[f'{branch}__L{l}_x'].astype('float64')*z[f'{branch}__L{l}_J{j}_{kind}_g_{part}'].astype('float64')[:,None]
                            vectors.append(terms.sum(0))
                        derivatives.append((f'{r["case_id"]}/L{l}J{j}/{kind}/{part}/all_scalar_coordinates',vectors[0]-vectors[1]))
    panel('scalar_derivatives','完整结构化答案：每个真实gate/up/down标量的全部坐标导数',derivatives)
    expanded={'h':[],'a':[]}
    for r in read(EXPAND/'material/cases.json'):
        if not r['published']:continue
        with np.load(EXPAND/f'field/case_{r["case_index"]:04d}.npz') as z:
            for key in ('h','a'):
                a=unbits(z[key])
                for l in range(len(a)):
                    for pos,label in enumerate(('body','task')):expanded[key].append((f'{r["case_id"]}/{key}{l}/{label}',a[l,pos]))
    for key in ('h','a'):panel('expanded_'+key,'4096多动词时序扩大复核：全部原始'+('H/词嵌入坐标' if key=='h' else 'MLP单元'),expanded[key])
    with np.load(OUT/'maps/source_prefix_all_coordinates.npz') as z:
        for key in ('h','a'):panel('source_prefix_'+key,'相同正文前缀的原场最大绝对差：'+key+'（测量审计，非语义）',[(f'{key}{l}/max_abs_over7168same_source_pairs',a) for l,a in enumerate(z[key+'__maximum_abs_difference'])])
    replay={'h':[],'a':[]}
    diagnostic=[g for g in read(OUT/'prefix_replay/material/groups.json') if g['published']]
    for precision in ('bf16','fp32'):
        for g in diagnostic:
            r=g['base']
            with np.load(OUT/f'prefix_replay/{precision}/field/group_{g["index"]:02d}.npz') as z:
                for variant in ('original','base','repeat','other','base_padded','other_padded'):
                    for key in ('h','a'):
                        origin='saved_BF16_reference' if variant=='original' else precision
                        replay[key].extend((f'case{r["case_index"]}/body_token{r["body_end_token"]}/{origin}/{variant}/{key}{l}/source={r["body"]}',a) for l,a in enumerate(z[variant+'__'+key]))
    for key in ('h','a'):panel('prefix_replay_'+key,'BF16/FP32同正文前缀：原样重复与等长填充全部'+('H/词嵌入坐标' if key=='h' else 'MLP单元'),replay[key])
    boundary='Allphysicalcolumns retained; rows may select clearly identifiedlayers. Fixedcandidatewindows not sparse representation. NativeBF16 fields and samevaluedFP32 derivatives from structuredanswer conditions separatelylabelled. No semantic closure.'
    save(OUT/'material/client_panel_catalog.json',{'phase':2676,'panels':catalog,'boundary':boundary,'display':'Every physical column; noTopK, averaging or dimensional projection. Paginated rows for display only.'})
    save(OUT/'analysis/publication.json',{'panels':[{k:v for k,v in p.items() if k!='rows'}|{'row_count':len(p['rows'])} for p in catalog],'published_bf':[r['case_index'] for r in examples],'phase_summaries':{str(p):v['summary'] for p,v in phases.items()},'old_asset_untouched':True})
    print('PUBLISHED',len(catalog),'native MLP panels',flush=True)

def cleanup():
    assert not (OUT/'analysis/cleanup_completed.json').exists()
    for p in ('scientific_checks','delivery_checks','live_api_checks','browser_checks'):assert read(OUT/f'analysis/{p}.json')['all_checks_passed'],p
    kept=[];targets=[]
    for source in [FIELD,EXPAND]+[CROSS/k for k in ('qwen14','glm4','ds7')]:
        for r in read(source/'analysis/raw_manifest.json'):
            path=Path(r['path']).resolve();assert path.parent==(source/'field').resolve() and path.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',path.name) and path.stat().st_size==r['bytes']
            entry={**r,'path':str(path),'sha256':sha(path)};(kept if r['published'] else targets).append(entry)
    for path in sorted((FP/'field').glob('case_*.npz')):kept.append({'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    for precision in ('bf16','fp32'):
        folder=OUT/f'prefix_replay/{precision}'
        for r in read(folder/'analysis/raw_manifest.json'):
            path=Path(r['path']).resolve();assert r['published'] and path.parent==(folder/'field').resolve() and path.is_relative_to(RESULT.resolve()) and re.fullmatch(r'group_\d{2}\.npz',path.name) and path.stat().st_size==r['bytes']
            kept.append({**r,'path':str(path),'sha256':sha(path)})
    assert len(targets)==9700 and len(kept)==52
    paths=[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((PATHS/'field').glob('case_*.npz'))];assert len(paths)==16
    weight_path=FIELD/'weights/native_candidate_vectors.npz';weights={'path':str(weight_path.resolve()),'sha256':sha(weight_path)}
    assert weights['sha256']==read(FIELD/'protocol/native_weights.json')['candidate_vector_sha256']
    plan={'kept':kept,'kept_native_path_packs':paths,'kept_learned_weight_vectors':weights,'targets':targets,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'before_free_bytes':shutil.disk_usage(OUT).free,
        'recoverability':'Direct deletion, notRecycleBin. Retain52published BF/FP/cross/expanded/prefix-replay raw packs,16native path packs, fullcoordinate derivedmaps, allmaterials/behavior/conditions, actualweightvectors/code andhashes. Models and learnedcheckpoints never modified.'}
    save(OUT/'analysis/cleanup_plan.json',plan)
    for r in targets:
        path=Path(r['path']);assert path.stat().st_size==r['bytes'];path.unlink()
    plan.update(all_deleted=all(not Path(r['path']).exists() for r in targets),all_kept=all(sha(r['path'])==r['sha256'] for r in kept+paths+[weights]),after_free_bytes=shutil.disk_usage(OUT).free)
    save(OUT/'analysis/cleanup_completed.json',plan);print('CLEANED',plan['deleted_files'],plan['deleted_bytes'],flush=True)

def finalize():
    checks={k:read(OUT/f'analysis/{k}.json')['all_checks_passed'] for k in ('scientific_checks','delivery_checks','live_api_checks','browser_checks','post_cleanup_checks')};clean=read(OUT/'analysis/cleanup_completed.json');checks.update(only_allowlisted_raw_deleted=clean['all_deleted'],all_published_hashes_retained=clean['all_kept'])
    assert all(checks.values());science=read(OUT/'analysis/scientific_checks.json');nextplan=read(OUT/'analysis/next_campaign.json')
    finish(2676,'原生MLP条件路径、标量验证与三模型完整审计交付',OUT,{'provenance':str(Path(__file__)),'summary':{'science':science,'publication':read(OUT/'analysis/publication.json'),'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},'next_campaign':nextplan},'checks':checks},
        '从独立实体/内容交叉观察到真实MLP单元每项权重计算，再到全部token标量/联合小剂量和三个模型本地坐标复验。额外审计相同token前缀的正文状态：原样重复、后续文本改变、两者等长遮罩填充和同权重FP32回放。所有机制判断与已知代数恒等、浮点执行形状、数值精度分账。',
        r'(e,c,o,f,p,q,m,v)\to x\to(g,u)\to a\to W_da\to h\to P(y);\quad G^{all}=G^{content}+G^{format}+G^{EOS}.\qquad H^{ideal}_{\ell,t}(p\Vert s_1)=H^{ideal}_{\ell,t}(p\Vert s_2)\ (t<|p|);\quad E_{prefix}=\max_{\ell,j}|H^{(T)}_{\ell,t,j}-H^{(T^\prime)}_{\ell,t,j}|.\qquad R_{L1}=\frac{\sum|\Delta L-\widehat{\Delta L}|}{\sum|\Delta L|}\quad\text{only if }\sum|\Delta L|>0.',
        '8192八族双语全场；40960原生单元路径观察；4096全MLP乘积对与256全坐标对照组；128格式化自然/FP32前缀；7680单标量、1280联合、480半剂量和128no-op；三模型各512条件；另4096中英多动词/八新实体对时序扩大复核（后验选族，非全族泛化样本）；16预定展示前缀上1440真实参数剂量/FP64读出复核（核心仍FP32，最大剂量不称无穷小）。复用既有8192原场的7168同正文前缀配对审计，以及64组×5对照×2精度=640次Transformer重放（不重复计算为自然生成样本）；新增实际权重/激活/导数及前缀测量误差的全坐标热力图、精确查询和审计。',
        '原始语言操作怎样变成具体gate/up输入贡献、怎样写到每个残差坐标已可定位和复查，但这是有限模型与有限材料的计算拼图，不是完整语言数学理论。重要旧阳性已在8192实体—内容独立新条件上扩大范围复核，未把确认失败条件删除。',
        '同一单元贡献与条件相符不等于独立语义符号；交互分解依基点，联合三参数不是路径最小割。两内容实例和少量句式不能代表无限语言。规范序列概率与自然回答严格区分，精度未分辨的微效应不能报作成功。',
        '当前大阶段全部完成；下一目标仍为外部操作到原生坐标条件复用/分化，依据实际通过与失败模式的完整下一批计划见analysis/next_campaign.json并继续同目标研究，不复写已完成Phase。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['publish','cleanup','finalize']);args=p.parse_args();globals()[args.action]()
