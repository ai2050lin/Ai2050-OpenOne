"""Zero-copy native artifact publication; full campaign completion is separately gated."""
import argparse,itertools,shutil
import numpy as np
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2684_source_campaign_delivery'
CONTRACT=RESULT/'phase2677_source_role_contract';FIELD=RESULT/'phase2678_padded_source_field';SOURCE=RESULT/'phase2679_native_source_ledger'
PATHS=RESULT/'phase2680_native_mlp_source_paths';FRESH=RESULT/'phase2681_fresh_source_confirmation';SCALAR=RESULT/'phase2682_resolved_scalar_paths';CROSS=RESULT/'phase2683_crossmodel_function_atlas'
LAYERS=(23,26,27,28)


def publish(preview=False):
    phases={p:read(path/'analysis/final.json') for p,path in ((2677,CONTRACT),(2678,FIELD),(2679,SOURCE),(2680,PATHS),(2681,FRESH))}
    if not preview:phases.update({2682:read(SCALAR/'analysis/final.json'),2683:read(CROSS/'analysis/final.json')})
    assert all(p['all_checks_passed'] for p in phases.values())
    catalog=[];retained={};examples=[]
    def register(path):
        path=path.resolve();assert path.is_relative_to(RESULT.resolve());retained[str(path)]={'path':str(path),'bytes':path.stat().st_size}
        return path.relative_to(RESULT.resolve()).as_posix()
    def panel(key,title,entries,boundary):
        assert entries
        D=entries[0].pop('_D');assert all(e.pop('_D')==D for e in entries[1:])
        catalog.append({'key':'phase2684_'+key,'title':title,'coordinate_count':D,'rows':entries,'storage':'native_descriptor','boundary':boundary})
    def describe(path,key,index,label,shape,encoding=None):
        row={'file':register(path),'array':key,'index':list(index),'label':label,'_D':shape[-1]}
        if encoding:row['encoding']=encoding
        return row
    for dataset,material,native,source in (('original',CONTRACT,FIELD,SOURCE),('fresh',FRESH,FRESH,FRESH)):
        records={r['case_index']:r for r in read(native/'analysis/records.json')};rows=[r for r in read(material/'material/cases.json') if r['published']]
        assert len(rows)==64
        panels={'h':[],'a':[]}
        for r in rows:
            ci=r['case_index'];path=native/f'field/case_{ci:04d}.npz';sp=source/('source_field' if dataset=='fresh' else 'field')/f'case_{ci:04d}.npz'
            pp={k:r[k] for k in ('case_index','case_id','prompt','token_strings','body_end_token','token_regions')}
            pp.update(dataset=dataset,native_path=register(path),source_path=register(sp),natural={k:records[ci][k] for k in ('generated','generated_ids','content_correct','strict_correct','eos')})
            examples.append(pp)
            with np.load(path) as z:
                for metric in ('h','a'):
                    shape=z[metric].shape
                    for l,q in itertools.product(range(shape[0]),range(2)):
                        panels[metric].append(describe(path,metric,(l,q),f'{dataset}/{r["case_id"]}/{metric}{l}/{("body","task")[q]}',shape,'native_bf16'))
        for metric in ('h','a'):panel(dataset+'_'+metric,f'{dataset} 64展示条件全部'+('词嵌入/H坐标' if metric=='h' else 'MLP神经元'),panels[metric],'Actual native BF16, body/task separate. H0 embedding; H36 before final RMSNorm. No same-index crossmodel claim.')
    weight=FIELD/'weights/native_candidate_vectors.npz';rr=[]
    with np.load(weight) as z:
        for k in z.files:rr.append(describe(weight,k,(),k,z[k].shape))
    panel('actual_weights','五神经元真实 gate/up/down 全输入行与输出列',rr,'Learned weights, not gradients or activations. Candidate windows do not define a sparse semantic basis.')
    for metric in ('h','a'):
        entries=[]
        for filename,scope in [('all_native_four_function_global_counts.npz','64 bases'),('all_native_four_function_family_counts.npz','4 bases/family-language')]:
            path=PATHS/'maps'/filename
            with np.load(path) as z:
                for k in z.files:
                    if not (k.startswith(metric+'__') or '__'+metric+'__' in k):continue
                    shape=z[k].shape
                    for l,q in itertools.product(range(shape[0]),range(2)):entries.append(describe(path,k,(l,q),f'{scope}/{k}/L{l}/{("body","task")[q]}',shape))
        panel('original_counts_'+metric,'原始四功能全坐标方向计数（含全部部分通过/零/反向）：'+metric,entries,'Counts are integers, not BF16. Four function same sign is withinbase; body agreement is causal-prefix control. Anyzero and opposed can overlap.')
        counts=[];amplitudes=[]
        for path in sorted((FRESH/'maps').glob('fresh_*.npz')):
            with np.load(path) as z:
                for k in z.files:
                    if not k.startswith(metric+'__'):continue
                    shape=z[k].shape
                    for f,o,l,q in itertools.product(range(2),range(2),range(shape[1]),range(2)):
                        entry=describe(path,k,(f*2+o,l,q),f'{path.stem}/{k}/f{f}o{o}/L{l}/{("body","task")[q]}',shape)
                        (amplitudes if 'abs_delta_sum' in k else counts).append(entry)
        panel('fresh_counts_'+metric,'4096确认：两句式两顺序各自全坐标方向计数 '+metric,counts,'8bases per form/order cell, four output functions. Everypartialcount retained, notsemanticclosure.')
        panel('fresh_amplitudes_'+metric,'4096确认：四功能目标差最小/最大绝对幅值逐坐标和 '+metric,amplitudes,'Minimum/maximum across fourfunctions, summed over8basegroups percell; not individual-value minima/maxima overthewholecorpus. No selected-coordinate compression.')
    source_rows=[];path_rows=[]
    for path in sorted((SOURCE/'maps').glob('source_*.npz')):
        with np.load(path) as z:
            for k in z.files:
                shape=z[k].shape
                if len(shape)!=3 or shape[-1]!=2560:continue
                for q,j in itertools.product(range(shape[0]),range(shape[1])):
                    source_rows.append(describe(path,k,(q,j),f'{path.stem}/{k}/{("body","task")[q]}/head_or_role{j}',shape))
    panel('head_role_sources','全部head/外部来源角色 → 全部残差坐标：有符号项、绝对项、目标差',source_rows,'Source maps raw8case sums or4targetpair counts as named. Head/role axes differ by key. V is contextualized; arithmeticaccounting notsemanticattribution. Native AV/Wo roundoff separate.')
    for path in sorted((PATHS/'maps').glob('path_*.npz')):
        with np.load(path) as z:
            for k in z.files:
                shape=z[k].shape
                for idx in np.ndindex(shape[:-1]):path_rows.append(describe(path,k,idx,f'{path.stem}/{k}/index{idx}',shape))
    panel('source_to_weight','条件来源 → RMSNorm → gate/up 全参数输入项与 down 全输出项',path_rows,'8case raw sums. Observed RMS denominator is endogenous; conditional allocation does notsimulate source ablation. Signedandabsoluteterms differ; rounding paths explicit.')
    if not preview:
        q14_weights=read(CROSS/'qwen14/analysis/candidate_weight_audit.json')
        assert q14_weights['all_checks_passed']
        for filename,key,title,boundary in (
            ('candidate_native_vectors.npz','qwen14_native_weights','Qwen14B 两个候选神经元：完整真实 gate/up/down 权重向量','Post-discovery windows from everyQ14global64-gateMLPunit. ActualcheckpointBF16 values stored exactly inFP32. Every5120coordinate, no TopK; weight is not activation.'),
            ('candidate_single_unit_terms.npz','qwen14_single_unit_terms','Qwen14B 两展示例：单神经元 down 投影全部输出坐标','FP64 product of actualWdown and nativea; one ideal term, notwholeMLP/HiddenState or newcausaltest. Body/task separatelylabelled. Positive term can coexistwithnegativewholeH targetdifference.'),
            ('published_native_embeddings.npz','qwen14_embeddings','Qwen14B 展示例：实际检查点词嵌入全部5120参数','All76uniqueinputtokens fromtwoactualpublishedchronologyEN examples;165occurrences independentlyequalactualH0. Notall512cases. ExactnativeBF16weights storedinFP32.')):
            rr=[];path=CROSS/'qwen14/weights'/filename
            with np.load(path) as z:
                for k in z.files:rr.append(describe(path,k,(),k,z[k].shape))
            panel(key,title,rr,boundary)
        # All prospectively retained full-token MLP fields remain inspectable,
        # not just the five unit windows. Native axis labels distinguish H/a.
        fullrows={}
        for dataset,material,native in (('original',CONTRACT,FIELD),('fresh',FRESH,FRESH)):
            for r in read(material/'material/cases.json'):
                if not r.get('parameter_published',r['published'] and r['output_function']=='truth'):continue
                path=native/f'field/case_{r["case_index"]:04d}.npz'
                with np.load(path) as z:
                    for k in z.files:
                        if not k.startswith('full__') or k=='full__h':continue
                        shape=z[k].shape;rr=fullrows.setdefault(k,[])
                        for li,t in np.ndindex(shape[:-1]):rr.append(describe(path,k,(li,t),f'{dataset}/{r["case_id"]}/{k}/L{LAYERS[li]}/token{t}',shape,'native_bf16'))
        for k,rr in fullrows.items():panel(k,'32展示例：全部实际 token '+k+' 原生全坐标',rr,'16original+16fresh truth exemplars; all units/coordinates in4explicit native layers, no TopK. NativeBF16. Query/embedding H is separately addressable.')
        for metric in ('h','a'):
            for model in ('qwen14','glm4','ds7','ds7_answer'):
                rr=[]
                for path in sorted((CROSS/model/'maps').glob('counts_*.npz')):
                    with np.load(path) as z:
                        for k in z.files:
                            if not k.startswith(metric+'__'):continue
                            shape=z[k].shape
                            for l,q in itertools.product(range(shape[0]),range(2)):rr.append(describe(path,k,(l,q),f'{model}/{path.stem}/{k}/L{l}/{("body","task")[q]}',shape))
                panel(model+'_counts_'+metric,f'{model} 四功能方向计数：本模型完整 '+metric,rr,'4bases per family/language. Native coordinate indices not aligned acrossmodels. DS generation/calibration limits in model report.')
        rr=[];cases=read(SCALAR/'material/cases.json')
        for r in cases:
            path=SCALAR/f'maps/case_{r["case_index"]:04d}.npz';shape=(120,3,2560)
            for ci,kind in itertools.product(range(120),range(3)):rr.append(describe(path,'local_coordinate_sums',(ci,kind),f'{r["case_id"]}/C{ci:03d}/{("actual_signed_sum","prediction_signed_sum","absolute_error_sum")[kind]}',shape))
        panel('scalar_local_validation','15360真实标量改动：全token逐输出坐标预测/实测/误差',rr,'FP32core. Sumover actualtokens, notnonlinear final-scoreprediction. One real weight changed, finallyrestored; known local formulas notsemanticclosure.')
        # Crossmodel and scalar raw fields remain accessible through row views.
        for model in ('qwen14','glm4','ds7','ds7_answer'):
            rr=[];aa=[]
            for r in read(CROSS/model/'material/cases.json'):
                if not r['published']:continue
                path=CROSS/model/f'field/case_{r["case_index"]:04d}.npz'
                with np.load(path) as z:
                    shape=z['full__h'].shape
                    for l,t in np.ndindex(shape[:-1]):rr.append(describe(path,'full__h',(l,t),f'{model}/{r["case_id"]}/H{l}/token{t}:{r["token_strings"][t]}',shape,'native_bf16'))
                    shape=z['a'].shape
                    for l,q in np.ndindex(shape[:-1]):aa.append(describe(path,'a',(l,q),f'{model}/{r["case_id"]}/MLP{l}/{("body","task")[q]}',shape,'native_bf16'))
            panel(model+'_full_H',f'{model} 两展示例全部token词嵌入与HiddenState',rr,'H0embedding; subsequent residual block outputs, before final norm. Two chronologyEN examples, not all-family raw display. Everycolumn unchanged.')
            panel(model+'_raw_a',f'{model} 两展示例全部原生MLP神经元',aa,'ActualBF16 body/task boundary units, not directioncounts. Originalnativeindices, no crossmodelindexsemanticmatch.')
        scalar_rows=[]
        for r in cases:
            if not r['published_numeric']:continue
            path=SCALAR/f'field/case_{r["case_index"]:04d}.npz'
            with np.load(path) as z:
                for k in z.files:
                    shape=z[k].shape
                    if not shape or shape[-1]!=2560:continue
                    for idx in np.ndindex(shape[:-1]):scalar_rows.append(describe(path,k,idx,f'{r["case_id"]}/{k}/index{idx}',shape))
        panel('scalar_raw','FP32数值对照：真实词嵌入/H/MLP输入与输出全部坐标',scalar_rows,'16published numerical prefixes, actualBF16 output IDs teacherforced inFP32. H usesbody/task boundaries; x/down/E actualallinputtokens. Not originalBF16 or FP64core.')
    boundary='Native all-coordinate source/function/scalar charts. No donor-state transplant; accounting/finitepattern/semantic mechanism are separate claims.'
    prefix='staged_' if preview else ''
    save(OUT/f'material/{prefix}client_panel_catalog.json',{'phase':2684,'panels':catalog,'boundary':boundary,'display':'Original coordinates, native array views; no duplicate dense heatmap copies, no TopK/averaged bins.'})
    save(OUT/f'material/{prefix}published_source_cases.json',examples)
    save(OUT/f'analysis/{prefix}publication.json',{'preview_only':preview,'panels':len(catalog),'source_cases':len(examples),'referenced_files':list(retained.values()),
         'completed_phases':list(phases),'raw_arrays_copied':False,'publication_manifest_sha256':sha(OUT/f'material/{prefix}client_panel_catalog.json')})
    print('SOURCE PUBLICATION',prefix,len(catalog),len(examples),flush=True)


def cleanup():
    assert not (OUT/'analysis/cleanup_completed.json').exists()
    for name in ('scientific_checks','delivery_checks','live_api_checks','browser_checks'):
        r=read(OUT/f'analysis/{name}.json');assert r['all_checks_passed'] and not r.get('preview_only',False),name
    publication=read(OUT/'analysis/publication.json');assert not publication['preview_only']
    referenced={Path(r['path']).resolve() for r in publication['referenced_files']};targets=[];kept=[]
    for folder,total in ((FIELD,8448),(SOURCE,512)):
        manifest=read(folder/'analysis/raw_manifest.json');assert len(manifest)==total
        for r in manifest:
            path=Path(r['path']).resolve()
            assert path.parent==(folder/'field').resolve() and path.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',path.name)
            assert path.stat().st_size==r['bytes'];entry={**r,'path':str(path),'sha256':sha(path)}
            if r['published']:assert path in referenced;kept.append(entry)
            else:assert path not in referenced;targets.append(entry)
    assert len(targets)==8832 and len(kept)==128
    # Hash every referenced actual array (native fields and fullcoordinate maps),
    # not only the two folders that have unpublished raw trajectories.
    retained=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(referenced)]
    plan={'targets':targets,'retained_referenced_files':retained,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),
          'before_free_bytes':shutil.disk_usage(OUT).free,'recoverability':'Direct deletion, not RecycleBin. Only unshown2678rawand2679sourcepacks. Publishedpacks/fullcoordinatecharts/material/behavior/weights/code remain. Regeneration needs retained checkpoints and exact numerical protocol; no promiseofbyte-identicaldifferenthardware.'}
    save(OUT/'analysis/cleanup_plan.json',plan)
    for r in targets:
        path=Path(r['path']);assert path.stat().st_size==r['bytes'];path.unlink()
    plan.update(all_deleted=all(not Path(r['path']).exists() for r in targets),all_retained=all(sha(r['path'])==r['sha256'] for r in retained),after_free_bytes=shutil.disk_usage(OUT).free)
    assert plan['all_deleted'] and plan['all_retained'];save(OUT/'analysis/cleanup_completed.json',plan);print('2684 CLEANED',len(targets),plan['deleted_bytes'],flush=True)


def review_and_plan():
    # Read complete actual results. Never use this draft implementation as a
    # substitute for finishing the source campaign or as a future test result.
    fresh=read(FRESH/'analysis/final.json');scalar=read(SCALAR/'analysis/final.json');cross=read(CROSS/'analysis/final.json')
    assert all(r['all_checks_passed'] for r in (fresh,scalar,cross))
    observations={'coordinate_confirmation':fresh['summary']['coordinate_confirmation'],'scalar_local_arithmetic':scalar['summary'],
                  'DS_actual_paired_protocol':read(CROSS/'analysis/DS_protocol_pair.json'),
                  'Qwen14_actual_candidate_weight_links':read(CROSS/'qwen14/analysis/candidate_weight_audit.json')['direct_same_layer_links']}
    cross_candidates={}
    for model in ('qwen14','glm4','ds7','ds7_answer'):
        cross_candidates[model]={}
        with np.load(CROSS/model/'maps/global_counts.npz') as z:
            for metric in ('h','a'):
                # Enumerate every coordinate meeting the frozen64-base gate;
                # keep all background maps. This is not a TopK representation.
                addresses=np.argwhere(z[metric+'__all4_same_nonzero'][:,1]==64)
                cross_candidates[model][metric]=[{'layer_or_checkpoint':int(l),'coordinate':int(j),
                    'positive_base_groups':int(z[metric+'__all4_positive'][l,1,j]),
                    'negative_base_groups':int(z[metric+'__all4_negative'][l,1,j])} for l,j in addresses]
    observations['all_native_global64_task_gate_addresses']=cross_candidates
    plan={'same_goal':True,'created_after_complete_results':True,'current_campaign_complete_only_after_delivery_checks':True,
          'objective':'外部语言操作如何经真实Q/K/V单标量、head归一化、位置变换和全来源softmax形成条件路由，再进入已测MLP坐标路径；积累可预测的条件拼图，而不是重复把归因恒等式称为语义齿轮。',
          'actual_evidence':observations,
          'why_change_focus':['2680/2681的完整图谱把分族条件纹理与全局普适单坐标假说分开；新词汇/形式/顺序确认必须用实际计数，而不是扩大叙述。',
             '2682已把局部MLP公式误差与末端微小输出变化分账；再重复down/gate同一恒等式不会自动解释语言操作怎样激活它。',
             '现在的source分账从已经上下文化的V开始；尚未在本轮原生标量范式下解释前序残差如何通过具体Wq/Wk/Wv与headnorm/RoPE产生条件路由。',
             'DS原生/答案区配对属于接口条件研究，不得按较高正确率挑一组假装原生跨模型机制已复现。'],
          'phases':{
              '2685':'完整结果复审、原生Q/K/V单参数与角色/词汇独立对照合同；冻结基本算术规则、精度账本和外推边界。',
              '2686':'八族双语的大规模词汇×语义角色×表达形式×输出功能材料；至少4096正式条件；分词/同token前缀/真实最终答案边界预检，独立内容和实体留给确认。每族包含意义改变与表面改变对照，不仅换名字。',
              '2687':'全部层H/MLP背景与预定层原生attention输入、线性Q/K/V、headnorm前后、RoPE前后、全部head/source概率的完整坐标测绘；全部真实token参与，按数值存储预算流式处理。',
              '2688':'具体Wq/Wk/Wv的全输入坐标项与条件响应表，保留幅值、符号、相消和全部低值坐标；坐标联合响应分组只是附加索引，不代替全场，也不自动命名为语义。',
              '2689':'至少128分层前缀的普通/低值真实单标量±多剂量测试；显式重算headnorm内生分母、位置旋转、全来源softmax，而不是固定分母删除或搬运其他样本；局部路由预测误差与完整生成串概率另列。',
              '2690':'在冻结算法和判据下以新语义关系实例/实体/表达组合再做至少4096条件扩大确认；重复的精确token与新的抽象关系分别声明，记录全部失败及部分复用。',
              '2691':'Qwen14B、GLM4、DS7B顺序非量化本模型坐标复验。2683的Qwen14B完整图谱实际出现3个H坐标和2个MLP神经元通过64基础组四功能方向门，因此Qwen14B须用至少4096个独立新条件扩大确认：新实体/关系实例、两表达形式、两提及顺序以及实体槽位和语义角色分开反平衡。冻结全部旧坐标计数背景和全部通过门的地址，不只计算5个地址，不按结果再改门。GLM4和DS7B每模型至少512条件，先校准接口/预算；不借用4B下标命名跨模型语义。',
              '2692':'把原生attention参数路径与MLP路径连接成有舍入残差的可查询计算账本，审查词汇/实体绑定/任务模板等解释；不把两条分开实验的阳性拼成自然机制闭环。',
              '2693':'重要完整坐标热力图、真实参数/源token查询、独立数值与真实客户端验收，之后清理未展示原场；再依据结果审查同目标后续。'},
          'first_principles':'固定权重可在不同token、位置和上下文反复应用；研究难点是提取这种条件复用怎样对应语言关系，而不是证明有限参数能执行已知矩阵运算。有限上下文模型的巨大组合空间不等于已经证明无条件无限语言能力。',
          'boundaries':['先观察，再找结构，再检验机制；不以删除/救援/必要性失败作为唯一路线关闭门。','不用统计显著性、PCA/TopK或高级数学竞赛替代基础全坐标证据。','同值FP32与FP64分析仅用于数值可分辨性，不能冒称原生BF16或整个FP64模型。','Qwen14B的5个坐标是固定形式/顺序、有限材料上的条件方向候选，可能编码实体槽位、提及顺序或回答指向，不能称为普适语义齿轮。重命名、表面顺序与关系翻转分开操纵；相对编码可表现为可预测符号改变，不能只把绝对值不变当成功。自然长度与padding前向均数值不同，不能拼接成同一条自然生成轨迹。','具体材料、存储预算、实际候选层与标量在2685/2686独立冻结；本计划不是完成记录。','若资源不够，先精确流式/分块而不截断坐标；不得未经范围核验删除历史场或改系统设置。']}
    save(OUT/'analysis/next_campaign.json',plan);print('ACTUAL RESULTS REVIEWED; SAME-GOAL NEXT PLAN SAVED',flush=True)


def finalize():
    checks={k:read(OUT/f'analysis/{k}.json')['all_checks_passed'] for k in ('scientific_checks','delivery_checks','live_api_checks','browser_checks','post_cleanup_checks','terminal_audit')}
    clean=read(OUT/'analysis/cleanup_completed.json');checks.update(unshown_allowlist_deleted=clean['all_deleted'],all_published_arrays_retained=clean['all_retained'])
    nextplan=read(OUT/'analysis/next_campaign.json');assert all(checks.values()) and nextplan['same_goal'] is True
    science=read(OUT/'analysis/scientific_checks.json');pub=read(OUT/'analysis/publication.json')
    finish(2684,'来源条件全坐标图谱、真实单标量验证与三模型完整交付',OUT,{'provenance':str(Path(__file__)),'summary':{'science':science,'published_heatmap_types':pub['panels'],
        'source_parameter_cases':pub['source_cases'],'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},'next_campaign':nextplan},'checks':checks},
        '把实际语言条件、所有原生坐标背景、全部来源分账和真实学习标量对应起来。数值恒等式、有限样本复用、完整输出概率和语言机制四者分别审查，不因普适单坐标门为空停止研究。',
        r'c_{h,s,k}=P_{q,h,s}\sum_d W_{o,k,hd}V_{s,kv(h),d};\quad z_{s,k}=\frac{\gamma_k c_{s,k}}{\sqrt{D^{-1}\sum_i u_i^2+\epsilon}};\quad g_j=\sum_kW_{g,jk}x_k.\qquad \Delta m_{t,:}=W_{d,:,j}\Delta a_{t,j}.',
        'C0018448固定执行形状八族双语条件；C002512全部head/source/原坐标账本；C003全部H/MLP四功能复用和五MLP输入来源全坐标路径；C0044096新实体/新词汇/形式顺序确认、其中512来源路径复验；C005128前缀15360真实单标量有限改动及完整实际生成串FP32计分，16例FP64读出对照；C006Qwen14B/GLM4各512，DS7B原生与显式进入答案区协议各512，DS各64独立实体预算校准（3模型4协议，2048正式条件）；C007无模型只读来源/参数查询与原坐标热力图、直接/HTTP/浏览器/构建验证；C008仅清理8832未展示原场。',
        '得到的是可寻址、可复查的条件图谱：一个来源位置经哪些head写到哪个坐标、观察到的归一化状态下如何进入某个gate/up输入项、真实权重改变怎样影响局部计算。全局固定语义坐标的强假说与更细条件化纹理必须分开，仍须靠新语言操作检验复用/分化。',
        'V来源已被上下文化，外部角色是注释不是发现的模块；RMS分母内生，所以来源分账非删除效果。跨功能末尾token、问法和prefill不同，形式泛化不充分。FP32数值验证不是BF16自主生成；FP64仅读出。2682首次加载0样本时遇到内存上限，释放本轮后端并关闭异步载入后重试，未改变模型/系统分页设置，见loading_incident.json。数千Phase不闭合只说明现有证据未完成解释，不能逻辑推出必需新数学。',
        '本轮2677–2684全部完成；根据实际完整结果冻结下一同目标大任务（next_campaign.json），继续研究条件坐标如何随角色/表达/组合变化，保留全部阴性和部分规律。没有宣称破解无限组合语言机制。')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--action',choices=['publish','review_and_plan','cleanup','finalize'],default='publish');a=ap.parse_args()
    if a.action=='publish':publish(a.preview)
    else:assert not a.preview;globals()[a.action]()
