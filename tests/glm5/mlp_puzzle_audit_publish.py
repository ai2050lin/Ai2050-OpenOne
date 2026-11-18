"""Audit retained raw data, describe limitations, and publish bounded client evidence."""
import io,json,tarfile,time
from collections import Counter
import numpy as np
import mlp_puzzle_campaign as P
from deploy.research_library import Library,canonical
from tools.power_analysis import check_phase

def audit():
    p=P.checked();fr=P.read(P.OUT/'frozen.json');ef=P.read(P.OUT/'effect-frozen.json')
    assert fr['plan_sha256']==P.sha(P.OUT/'plan.json') and fr['predictors_sha256']==P.sha(P.OUT/'predictors.npz')
    assert ef['frozen_sha256']==P.sha(P.OUT/'frozen.json') and ef['effect_sha256']==P.sha(P.OUT/'effect-predictor.npz')
    assert fr['holdout_unopened'] and ef['holdout_unopened']
    report=P.read(P.OUT/'report.json');co=np.load(P.OUT/'predictors.npz');hashes=[];identities=set();embedding=None
    means={b:np.zeros((8,9728)) for b in P.BLOCKS};constant={b:np.zeros(9728) for b in P.BLOCKS};frameerrors={}
    start=time.monotonic()
    for split,n in [('discovery',128),('holdout',512)]:
        rows=P.read(P.OUT/f'{split}-materials.json')
        for f in range(n):
            path=P.OUT/f'{split}-{f:04d}.npz';meta=P.read(path.with_suffix('.json'));assert P.sha(path)==meta['sha256']
            assert meta['plan_sha256']==P.sha(P.OUT/'plan.json');hashes.append({'path':path.name,'sha256':meta['sha256'],'bytes':path.stat().st_size})
            with np.load(path) as z:
                h=z['hidden'];assert h.shape==(8,37,2,2560)
                if embedding is None:embedding=h[0,0,-1]
                assert np.all(h[:,0,-1]==embedding),'Final token embedding changes'
                for row in rows[f*8:f*8+8]:
                    key=tuple(row['tokens']);assert key not in identities;identities.add(key)
                    assert row['positions'][-1]==len(row['tokens'])-1
                if split=='holdout':
                    X=np.stack([P.features(r) for r in rows[f*8:f*8+8]])
                    for b in P.BLOCKS:
                        Y=P.unbits(z[f'A{b}']);means[b]+=Y/512;constant[b]+=((Y-co[f'A{b}'][0])**2).sum(0)/(512*8)
                    for g in fr['groups']:
                        b=g['block'];u=g['units'];Y=P.unbits(z[f'A{b}'])[:,u];c=co[f'A{b}'][:,u]
                        cols=[0,1,2,3,4,5] if g['kind']=='binding' else [0,1,2,6]
                        vals=[float(((Y-X@c)**2).mean()),float(((Y-X[:,cols]@c[cols])**2).mean())]
                        frameerrors.setdefault(g['id'],{}).setdefault(str(rows[f*8]['frame']),[]).append(vals)
            if (f+1)%64==0:P.log('audit_'+split,f+1,n,start)
    X=np.stack([P.features(r) for r in P.read(P.OUT/'holdout-materials.json')[:8]])
    tables={};failures={};frame_summary={};contrast=[]
    for g in report['groups']:
        b=g['block'];u=g['units'];coef=co[f'A{b}'][:,u];cols=[0,1,2,3,4,5] if g['kind']=='binding' else [0,1,2,6]
        tables[g['id']]={'conditions':X[:,1:4].tolist(),'observed':means[b][:,u].tolist(),'predicted':(X@coef).tolist(),
            'baseline':(X[:,cols]@coef[cols]).tolist(),'coefficients':coef.tolist(),'constant_embedding_baseline_mse':float(constant[b][u].mean())}
        reason=[]
        if g['relative_mse']>.8:reason.append('误差未比预注册基线降低至少20%')
        if g['family_win_rate']<.65:reason.append('改善家族不足65%')
        if g['gain_ci'][0]<=0:reason.append('多重比较校正后的改善区间包含零')
        if g['random_group_relative_mse']<=g['relative_mse']:reason.append('同层随机组的相对误差不高于候选组，独特性未建立')
        if not any(e['group']==g['id'] for e in report['effects']):reason.append('未安排本组作用检验；保留为响应拼图')
        else:
            primary=next(e for e in report['effects'] if e['group']==g['id'] and e['dose']==1)
            if not primary['use_supported']:reason.append('置零作用未同时通过实质效应、等范数对照和有效对照数量门')
        failures[g['id']]=reason or ['范围内响应与作用通过；机制完整性及独立复现仍待检']
        frame_summary[g['id']]={k:float(np.mean(v,0)[0]/np.mean(v,0)[1]) for k,v in frameerrors[g['id']].items()}
        contrast.append({'group':g['id'],'median_abs_qr':float(np.median(np.abs(coef[6]))),'median_abs_L':float(np.median(np.abs(coef[3]))),
                         'median_abs_qrL':float(np.median(np.abs(coef[7]))),'scope':'descriptive coexistence of fitted terms;not separate fixed modules'})
    causal=[]
    for g in fr['causal_groups']:
        field='language_margin' if g['kind']=='language' else 'content_margin'
        for dose in [.5,1.]:
            losses=[];rand=[];null=[];correctlost=0;strata={}
            for f in range(256):
                row=P.read(P.OUT/f'causal-{f:04d}.json');assert row['commit_sha256']==P.sha(P.OUT/f'commit-{f:04d}.json')
                r=next(v for v in row['records'] if v['group']==g['id'] and v['dose']==dose)
                assert r['norm_valid']==(r['norm_relative_error']<=.02)
                loss=row['baseline'][field]-r['edited'][field];losses.append(loss)
                rand.append(row['baseline'][field]-r['random_group'][field]);null.append(row['baseline'][field]-r['norm_control'][field])
                correctlost+=row['baseline']['top1_correct'] and not r['edited']['top1_correct']
                key=f"semantic{row['sample']['semantic']}_language{row['sample']['language']}"
                strata.setdefault(key,[]).append(loss)
            causal.append({'group':g['id'],'dose':dose,'random_group_mean_margin_loss':float(np.mean(rand)),
                'random_group_excess_ci':P.bootstrap_gain(np.array(rand),np.array(losses),.0125),
                'lost_acceptable_first_token':int(correctlost),'strata':{k:{'n':len(v),'mean_loss':float(np.mean(v))} for k,v in strata.items()}})
    generations=report['generation'];strict=sum(r['answer'].strip().lower() in ('red','blue','红','红色','蓝','蓝色') for r in generations)
    limits=[
      '本实验覆盖记录绑定和输出语言选择，没有检验聊天/辩论风格、开放翻译、递归深度或训练如何形成编码。',
      'q×r 与答案的关系由人工设计提供；拟合得到的是条件响应规律，不是从无标签语料发现的内部算法。',
      '下一词预测器直接拟合输出 logits，未通过选定单元组和真实下游参数生成；不能将其保真度称作单元图谱破解率。',
      '输入末 token 的 embedding 完全相同，排除了该读出位置的单 token 身份；整段词序、位置、句长、语用仍可参与编码。',
      '每个留出家族八条件完整交叉；但只有四种新句式，名词与句式未独立交叉，数字 ID 不能当成独立语义领域。',
      '因果子集每家族取一条件，条件与句式相关；成对编辑的局部作用仍可比较，但分层语言效应不能跨句式独立归因。',
      '置信区间按记录家族重采样，未按四个句式簇估计；不支持宽泛句式总体的置信声明。',
      '效应预测使用原始 logits 与冻结校准增量；它不是不读原模型状态的下一词生成器。',
      '能力材料来自已暴露题库，且不在 plan.materials 的冻结哈希集合内；仅做有原文可追溯的描述性副作用检查。',
      '随机单元组不是唯一负对照；等范数对照更公平但不是语义匹配路径。单元组冗余和其他补偿路线仍未解释。',
      '未同时比较 SAE、transcoder 或无监督特征提取；没有新的特征字典，更未解释 embedding→attention→MLP 全链路。',
      '全部运行在一个真实节点；技术复算与逐位恢复不是独立节点科学复现。共享显存性能计数不可读，null 不能当成零。']
    result={'schema':'native_puzzle_audit.v1','verified_files':len(hashes),'unique_inputs':len(identities),'final_embedding_bitwise_constant':True,
        'group_failures':failures,'frame_response_ratios':frame_summary,'factor_coexistence':contrast,'causal':causal,
        'strict_single_color_generation_format':{'n':len(generations),'format_pass':strict,'scope':'posthoc format screen; correctness retained separately'},
        'implementation':P.read(P.OUT/'causal-implementation-audit.json'),'control_amendment':P.read(P.OUT/'control-amendment.json'),
        'limits':limits,'native_manifest':hashes}
    P.save('audit.json',result);P.save('response-tables.json',tables)
    print(json.dumps({'audit_files':len(hashes),'embedding_constant':True,'generation_format':strict}),flush=True)

def publish():
    p=P.checked();report=P.read(P.OUT/'report.json');a=P.read(P.OUT/'audit.json');fr=P.read(P.OUT/'frozen.json')
    a['baselines']=P.read(P.OUT/'baseline-audit.json')
    a['admissibility']={'formal_evidence_promotion':False,
      'missing_prerun_gate':'Plan recorded continuous-effect normal-approximate MDE but did not call the repository-required paired-binary power gate before inference. Posthoc calculations do not repair preregistration.',
      'posthoc_binary_sensitivity':[{'alpha':.05/4,**check_phase(d,256,alpha=.05/4)} for d in [.02,.03,.05]],
      'effect_gate_wording':'Plan key says absolute_mean_margin_change_min, frozen executable tests positive mean loss >=0.2. We report the executable directional test; opposite signed effects are descriptive counterexamples, not passed primary hypotheses.',
      'raw_provenance':'Raw per-family JSON links to frozen plan; that plan omitted git_dirty in its provenance object. Original files are retained unchanged; publication metadata and the final helper-written audit include all five fields, without retroactive formal promotion.'}
    a['limits'].append('运行前缺少仓库要求的二值行为功效准入，且原始溯源缺git_dirty；本轮不晋级正式证据。连续响应统计与局部作用按实际执行口径保留。')
    causal_rows=[P.read(P.OUT/f'causal-{f:04d}.json') for f in range(256)]
    a['causal_counterexamples']={}
    for g in fr['causal_groups']:
        field='language_margin' if g['kind']=='language' else 'content_margin';examples=[]
        for row in causal_rows:
            rec=next(r for r in row['records'] if r['group']==g['id'] and r['dose']==1)
            examples.append({'family':row['sample']['family'],'prompt':row['sample']['prompt'],
                 'semantic':row['sample']['semantic'],'language':row['sample']['language'],
                 'margin_loss':row['baseline'][field]-rec['edited'][field],'baseline':row['baseline'],'edited':rec['edited']})
        examples.sort(key=lambda r:r['margin_loss'])
        a['causal_counterexamples'][g['id']]=[examples[0],examples[-1]]
    a['strong_baseline_summary']={'four_state_beats_eight_groups':sum(v['eight_over_four']>1 for v in a['baselines']['groups'].values()),
        'conclusion':'Stable response is retained; eight-condition complexity is not justified by the four-state challenge. Trace upstream formation of color × language states next.'}
    libs=[Library(P.ROOT/v) for v in ['distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library']]
    objects={}
    def put(name,body):
        doc={**body,'kind':'real','run_id':P.RUN,'model_snapshot':P.SNAPSHOT,
             'provenance':{**p['provenance'],'git_dirty':True,'created_at':p['created_at']}};raw=canonical(doc)
        hashes={lib.ingest(doc,f'{P.RUN}/{name}',kind='local_report')['sha256'] for lib in libs};assert len(hashes)==1
        h=hashes.pop();objects[h]=raw;return h
    artifacts={'冻结设计':put('plan',p),'冻结选择':put('selection',{'schema':'native_puzzle_selection.v1',**fr}),
        '技术与结论审计':put('audit',a),'权重逐张量加载审计':put('loader',P.read(P.OUT/'loader-audit.json'))}
    artifacts['四状态强基线事后挑战']=put('baseline-audit',{'schema':'native_puzzle_baselines.v1',**a['baselines'],'protocol':P.read(P.OUT/'baseline-audit-plan.json')})
    # Selected evidence is small; full native fields remain in the hashed local archive.
    chunks=[];materials=P.read(P.OUT/'holdout-materials.json');coeff=np.load(P.OUT/'predictors.npz')
    for start in range(0,256,32):
        chunks.append(put(f'causal-{start}',{'schema':'native_puzzle_effect_evidence.v1','records':[P.read(P.OUT/f'causal-{f:04d}.json') for f in range(start,start+32)]}))
    artifacts['全部干预证据索引']=put('effect-index',{'schema':'native_puzzle_evidence_index.v1','parts':chunks})
    artifacts['完整发现与留出文本']=put('materials',{'schema':'native_puzzle_materials.v1','discovery':P.read(P.OUT/'discovery-materials.json'),'holdout':materials})
    artifacts['能力题原文与结果']=put('capability',{'schema':'native_puzzle_capability.v1','materials':P.read(P.OUT/'capability-materials.json'),'measurements':P.read(P.OUT/'capability.json')})
    samples=[]
    for f in sorted({0,1,2,3,*[c['family'] for g in report['groups'] for c in g['counterexamples']]}):
        with np.load(P.OUT/f'holdout-{f:04d}.npz') as z:
            selected={}
            for g in fr['groups']:
                b=g['block'];units=g['units'];index=z[f'GV_units{b}'].tolist();pos=[index.index(u) for u in units]
                selected[g['id']]={'units':units,'G':P.unbits(z[f'G{b}'])[:,pos].tolist(),'V':P.unbits(z[f'V{b}'])[:,pos].tolist(),
                    'A':P.unbits(z[f'A{b}'])[:,units].tolist(),'coefficients':coeff[f'A{b}'][:,units].tolist()}
            samples.append({'family':f,'samples':materials[f*8:f*8+8],'selected':selected})
    artifacts['实例与反例原生GVA']=put('examples',{'schema':'native_puzzle_examples.v1','families':samples,'layout':'eight balanced conditions × selected native units; BF16 decoded to exact float32'})
    unit_examples={}
    for g in fr['groups']:
        values=samples[0]['selected'][g['id']]
        unit_examples[g['id']]={'family':samples[0]['family'],'unit':g['units'][0],
           'rows':[{'query':r['query'],'record':r['record'],'language':r['language'],
                    **{k:values[k][i][0] for k in ['G','V','A']}} for i,r in enumerate(samples[0]['samples'])]}
    report.update(artifacts=artifacts,audit={k:v for k,v in a.items() if k!='native_manifest'},response_tables=P.read(P.OUT/'response-tables.json'),
                  unit_examples=unit_examples,
                  provenance={**p['provenance'],'git_dirty':True,'created_at':p['created_at']},
                  limits=[*report['limits'],*a['limits']],formal_evidence_promotion=False)
    ref=put('report',report);cards=[];edges=[]
    for g in report['groups']:
        cid=f'{P.RUN}:{g["id"]}'
        cards.append({'id':cid,'run_id':P.RUN,'layer':g['block'],'boundary':'mlp.down_proj.input',
          'label':f'B{g["block"]} · {"对象属性组合" if g["kind"]=="binding" else "输出语言"} · 8单元组',
          'index':{'space':'mlp','unit':g['units'][0],'units':g['units'],'anchor_only':True},
          'method':'discovery-only selection; frozen factorial predictor; eight-condition heldout; separate native interventions',
          'evidence':{'level':'observation' if g['passed'] else 'negative','artifact_sha256':ref},
          'scope':{'group_units':g['units'],'response_gate':g['passed'],'formal_promotion':False,'reason':a['group_failures'][g['id']]},
          'counterexamples':[{'artifact_sha256':ref,'field':f'groups.{g["id"]}.counterexamples'}]})
        edges.append({'id':cid+':mapping','source':'language:record_binding' if g['kind']=='binding' else 'language:requested_output_language',
            'target':cid,'relation':'留出条件响应关联；作用单独计分','run_id':P.RUN,'evidence':{'level':'observation' if g['passed'] else 'negative','artifact_sha256':ref}})
    for r in report['representatives']:
        cards.append({'id':f'{P.RUN}:B{r["block"]}:U{r["unit"]}','run_id':P.RUN,'layer':r['block'],'boundary':'mlp.down_proj.input',
           'label':f'B{r["block"]}/U{r["unit"]} · 旧档案分层代表','index':{'space':'mlp','unit':r['unit']},'method':r['kind'],
           'evidence':{'level':'observation','artifact_sha256':ref},'scope':r,'counterexamples':[{'artifact_sha256':ref,'field':'audit.limits'}]})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
       'context':{'study_type':'native_puzzles','native_bank':True,'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 单元拼图：解释、反例、预测与作用',
       'stages':{'behavior':{'status':'done','heldout':report['behavior']},'prediction':{'status':'done','groups_passed':sum(g['passed'] for g in report['groups'])},
       'native_effect':{'status':'done','formal_evidence_promotion':False},'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    reg={'run_id':P.RUN,'model_snapshot':P.SNAPSHOT,'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    bundle=P.OUT/'ai2050-research-native-puzzles-v1.tar.gz'
    assert sum(map(len,objects.values()))<128*2**20
    with tarfile.open(bundle,'w:gz',compresslevel=5) as t:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':h} for h in objects],'studies':[reg]}))
        for h,raw in objects.items():add(f'objects/{h}.json',raw)
    assert bundle.stat().st_size<64*2**20
    reg.update(bundle=str(bundle),bundle_sha256=P.sha(bundle),objects=len(objects),cards=len(cards),edges=len(edges),local_libraries=[str(l.root) for l in libs],remote_receipt=None)
    P.save('registration.json',reg);P.save('client-report.json',report);print(json.dumps(reg,ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    (audit if sys.argv[-1]=='audit' else publish)()
