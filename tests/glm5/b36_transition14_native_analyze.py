"""Audit the frozen native path experiment; no selection or model execution."""
import json, math, html, time
import numpy as np
import b36_transition14 as M
from b36_transition14_native_math import mobius
S = M.S


def paired(left, right):
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    gain, loss = int((~left & right).sum()), int((left & ~right).sum())
    d = gain + loss
    p = min(1., 2 * sum(math.comb(d, k) for k in range(min(gain, loss)+1)) / 2**d) if d else 1.
    if d:
        from scipy.stats import binomtest
        assert abs(p - binomtest(min(gain, loss), d, .5).pvalue) < 1e-12
    return dict(n=len(left), left_correct=int(left.sum()), right_correct=int(right.sum()),
                gained=gain, lost=loss, delta=float(right.mean()-left.mean()), p_exact=p)


def interval(x, rng):
    x = np.asarray(x, float)
    return np.quantile(x[rng.integers(0, len(x), (20000, len(x)))].mean(1), [.025, .975]).tolist()


def plot_fields(mean, rms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale=np.sqrt((rms*rms).mean(1,keepdims=True)).clip(1e-12)
    fig,axes=plt.subplots(2,1,figsize=(16,7),layout='constrained')
    for ax,data,title in zip(axes,[mean,mean/scale],['Native mean (BF16 values decoded to float)','Mean divided by per-layer population RMS']):
        bound=float(np.abs(data).max())
        im=ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-bound,vmax=bound)
        ax.set_title(title);ax.set_xlabel('Native residual coordinate (unchanged order, all 5120)');ax.set_ylabel('Raw boundary (0 = embedding)');fig.colorbar(im,ax=ax)
    fig.suptitle('Before Code: all 41 raw boundaries, mean over 184 views / 92 families\nDescriptive only; averaging may cancel condition-specific signals')
    fig.savefig(S.OUT/'native-all-coordinates.png',dpi=180);plt.close(fig)


def run():
    S.configure(60*2**20)
    p = S.read(S.OUT/'native-plan.json')
    for name, sha in p['source_hashes'].items():
        assert S.file_sha(S.ROOT/'tests/glm5'/name) == sha, name
    assert p['behavior_analysis_sha256'] == S.file_sha(S.OUT/'analysis.json')
    ex = S.read(S.OUT/'native-execution.json')
    assert ex['source_hashes'] == p['source_hashes']
    assert ex['numeric_sha256'] == S.file_sha(S.OUT/'native-numeric.json')
    assert S.read(S.OUT/'native-numeric.json')['passed']
    rows = [r for r in p['rows'] if r['split']=='confirm']
    by = {r['id']:r for r in rows}; families = sorted({r['family'] for r in rows})
    original_plan=S.read(S.OUT/'plan.json')
    original_rows={r['id']:r for r in original_plan['rows']}
    originals={r['id']:r for arm in ['counted0','counted2'] for r in S.read(S.OUT/(arm+'.json'))['results']}
    history_codes={}
    for row in rows:
        slot=originals[row['id']]['parsed']['slot']
        if slot is not None:
            code=original_rows[row['id']]['codes'][slot-1]
            history_codes[row['id']]=row['choice_ids']['ABCD'.index(code)]
    assert len(rows)==184 and len(families)==92
    effects, controls, seen = [], [], []
    field_sum = np.zeros((41, S.DIM), np.float64)
    field_square = np.zeros_like(field_sum)
    array_hashes = {}; embeddings={};start=time.monotonic()
    for index, entry in enumerate(ex['records']):
        path = S.OUT/entry['path']; assert S.file_sha(path)==entry['sha256']
        rec = S.read(path); assert rec['source_hashes']==p['source_hashes']
        assert rec['arms']==p['arms'] and rec['restore_bitwise'] and rec['history_attention_hold_current_bitwise'] and rec['native_product_bitwise']
        arr = path.parent/'native.npz'; assert S.file_sha(arr)==rec['arrays_sha256']
        array_hashes[str(arr.relative_to(S.OUT))]=rec['arrays_sha256']
        n=len(rec['row_ids']); seen += rec['row_ids']
        with np.load(arr, allow_pickle=False) as z:
            assert z['hidden'].shape==(n,41,4,S.DIM) and z['hidden'].dtype==np.uint16
            assert z['B36_A'].shape==(n,4,S.WIDTH)
            assert z['down_columns'].shape==(64,S.DIM)
            for key in z.files:
                assert z[key].dtype==np.uint16 and np.isfinite(S.unbits(z[key])).all(), key
            field=S.unbits(z['hidden'])[:,:,3,:].astype(np.float64)
            for i,rid in enumerate(rec['row_ids']):
                token=by[rid]['tokens'][-1];emb=z['hidden'][i,0,3].copy()
                if token in embeddings:assert np.array_equal(embeddings[token],emb)
                else:embeddings[token]=emb
            field_sum+=field.sum(0);field_square+=(field*field).sum(0)
            assert np.array_equal(z['final_raw'][0],z['final_raw'][p['arms'].index('restore')])
            assert np.array_equal(z['final_raw'][0],z['final_raw'][p['arms'].index('history_hold_attention')])
            baseline_logits=S.unbits(z['baseline_logits'])
            baseline_records={r['id']:r for r in rec['effects'] if r['arm']=='mask0'}
            for i,rid in enumerate(rec['row_ids']):
                rr=baseline_records[rid]
                assert int(baseline_logits[i].argmax())==rr['top1']
                assert np.array_equal(baseline_logits[i,by[rid]['choice_ids']],np.asarray(rr['scores'],np.float32))
        effects+=rec['effects'];controls+=rec['norm_controls']
        S.log('native_audit',index+1,len(ex['records']),start)
    assert len(seen)==len(set(seen))==184 and set(seen)==set(by)
    lookup={(r['id'],r['arm']):r for r in effects}
    assert len(lookup)==len(effects)==184*len(p['arms'])
    for row in rows:
        for arm in p['arms']:
            r=lookup[row['id'],arm];assert r['gold']==row['gold'] and r['choice_ids']==row['choice_ids']
            assert r['correct']==(r['top1']==row['choice_ids'][row['gold']])
            assert abs(r['fixed_margin']-(r['scores'][r['gold']]-r['scores'][r['fixed_rival']]))<1e-7
            assert np.isfinite([r['fixed_margin'],r['strict_margin'],r['KL'],r['b36_displacement_norm']]).all()
    assert len(controls)==184 and {r['id'] for r in controls}==set(by)
    agreement=sum(lookup[r['id'],'mask0']['top1']==r['observed_token'] for r in rows)/len(rows)
    admitted=agreement>=p['numerical']['source_generation_top1_agreement_min']
    norm_admitted=all(r['passed'] and r['relative_error']<=.02 for r in controls)
    rng=np.random.default_rng(S.SEED+3601)
    affine={}
    for f in original_plan['families']:
        aa,bb=1,0
        for op,k in f['ops']:
            if op=='mul':aa*=k;bb*=k
            elif op=='add':bb+=k
            else:bb-=k
        affine[f['id']]=(aa,bb)
    clusters=sorted({affine[f] for f in families})
    def clustered_interval(delta):
        totals=np.array([sum(delta[i] for i,f in enumerate(families) if affine[f]==key) for key in clusters])
        sizes=np.array([sum(affine[f]==key for f in families) for key in clusters])
        draws=rng.integers(0,len(clusters),(20000,len(clusters)))
        return np.quantile(totals[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()
    family_rows={f:[r for r in rows if r['family']==f] for f in families}
    assert all(len(rr)==2 and {r['arm'] for r in rr}=={'counted0','counted2'} for rr in family_rows.values())
    joint={arm:np.array([all(lookup[r['id'],arm]['correct'] for r in family_rows[f]) for f in families]) for arm in p['arms']}
    margin={arm:np.array([np.mean([lookup[r['id'],arm]['fixed_margin'] for r in family_rows[f]]) for f in families]) for arm in p['arms']}
    primary=[]
    for left,right in p['primary']:
        stat=paired(joint[left],joint[right]);delta=joint[right].astype(float)-joint[left].astype(float)
        primary.append(dict(left=left,right=right,**stat,admitted=admitted,
                            p_for_correction=stat['p_exact'] if admitted else 1.,descriptive_family95=interval(delta,rng),
                            affine_clusters=len(clusters),exploratory_affine_cluster95=clustered_interval(delta)))
    carry=0.
    for rank,item in enumerate(sorted(primary,key=lambda r:r['p_for_correction'])):
        carry=max(carry,(len(primary)-rank)*item['p_for_correction']);item['p_holm']=min(1.,carry)
        item['reject']=admitted and item['p_holm']<.05
    summaries=[]
    for arm in p['arms']:
        rr=[lookup[r['id'],arm] for r in rows];change=margin[arm]-margin['mask0']
        summaries.append(dict(arm=arm,view_correct=sum(r['correct'] for r in rr),views=184,
                             joint_family_correct=int(joint[arm].sum()),families=92,
                             printed_slot_consistent=sum(r['id'] in history_codes and r['top1']==history_codes[r['id']] for r in rr),printed_slot_parseable=len(history_codes),
                             mean_fixed_margin_change=float(change.mean()),descriptive_family95=interval(change,rng),
                             mean_KL=float(np.mean([r['KL'] for r in rr])),mean_displacement_norm=float(np.mean([r['b36_displacement_norm'] for r in rr]))))
    coeff=mobius(np.stack([margin[f'mask{i}'] for i in range(8)]))
    interactions=[dict(mask=i,mean=float(coeff[i].mean()),descriptive_family95=interval(coeff[i],rng)) for i in range(1,8)]
    stats=dict(primary=primary,summaries=summaries,interactions=interactions,power=p['power'],
               source_position_counts={name:dict(min=int(np.min(v)),median=float(np.median(v)),max=int(np.max(v))) for name,v in [('prompt',[r['prompt_length'] for r in rows]),('history',[len(r['tokens'])-r['prompt_length']-1 for r in rows]),('current',[1]*len(rows))]},
               numeric_generation_agreement=agreement,numeric_generation_mismatch_ids=[r['id'] for r in rows if lookup[r['id'],'mask0']['top1']!=r['observed_token']],numeric_admitted=admitted,norm_control_admitted=norm_admitted,
               norm_controls_failed=[r for r in controls if not r['passed']],array_hashes=array_hashes,
               embedding_current_token_ids=sorted(embeddings),embedding_identity_bitwise=True,
               native_plan_sha256=S.file_sha(S.OUT/'native-plan.json'),execution_sha256=S.file_sha(S.OUT/'native-execution.json'),
               source_sha256=S.file_sha(__file__),nominal_families=92,views=184,
               limits='Self-history already contains Value and Slot. Original generated token IDs are held fixed while the prefix is recomputed: no new free-generation rollout under neural interventions. Conditional Code-token routing only; no original arithmetic necessity, autonomous recomputation, uniqueness, or first-formation claim. Six family-joint binary comparisons use Holm. Continuous margins, KL and Boolean interactions are descriptive, not additional confirmatory tests. Random direction control is one frozen direction, not a population null. No broad unrelated-task damage battery. Bootstrap zero-discordance intervals do not establish equivalence.')
    S.save('native-analysis.json',stats)
    np.savez_compressed(S.OUT/'native-field-summary.npz',mean=field_sum/184,rms=np.sqrt(field_square/184))
    plot_fields(field_sum/184,np.sqrt(field_square/184))
    lines='；'.join(f"{r['arm']}：双轮换均正确{r['joint_family_correct']}/92，单视图正确{r['view_correct']}/184，固定竞争项间隔变化{r['mean_fixed_margin_change']:.4f}" for r in summaries)
    tests='；'.join(f"{r['left']}→{r['right']}：新增{r['gained']}、丢失{r['lost']}，Holm p={r['p_holm']:.6g}" for r in primary)
    memo=f"冻结32单元及匹配32单元，4家族仅用于数值校准，92家族×2代码轮换确认。原生运行器与原生成代码token一致率{agreement:.4%}；范数匹配控制{'全部合格' if norm_admitted else '有不合格项，不支持选择性结论'}。原样剂量1和历史扰动后恢复后续attention的当前位置均逐位一致。{lines}。六项冻结双轮换联合正确率配对比较：{tests}。所有41个残差边界与4个因果可见位置保存原生BF16全坐标；B36的G、V、A及固定64单元全前缀轨迹、真实down列同时保存。对P题目、H自产历史、C当前位置的8种移除组合，令m(S)为真值相对自然基线最强竞争代码的logit间隔，I(T)=Σ_(S⊆T)(−1)^(|T|−|S|)m(S)为描述性布尔分解；该恒等分解和非线性项不是新的语言机制定律。后续attention/MLP恢复是受干预状态与自然模块输出的混合路径实验，不证明唯一或充分机制。读出时Value/Slot已存在，结论仅涉及Code接续；无法凭本实验确定复杂语义首次在哪层或哪个生成时刻形成。没有广泛无关任务损伤测试或同预算字典比较。"
    if not admitted:
        memo='原生结果与原自然生成的代码token一致率未满足冻结99%准入，所有路径数据仅作数值诊断，不计正式机制确认。'+memo
    memo+=' 全部神经干预固定原始自产历史token并重算前缀，没有重新自由生成受干预的历史；模块输出恢复覆盖后三层全部位置，未细分注意力头。B36指零起始block索引36。'
    memo+=f' 程序家族独立只是统计近似，确认材料有{len(clusters)}种终态仿射函数；附加同仿射函数聚类区间仅作敏感性分析，不能证明排除了所有相关性。'
    memo+=' 少数自产Value/Slot可能错误，主指标始终按原题真值评分，同时描述代码与已打印Slot是否一致；二者不合并，不能把迎合错误历史计作原题计算正确。'
    memo+=f' 代码读出位置共{len(embeddings)}种当前token，所有相同token的hs[0]逐位相同；跨层同坐标索引仍不自动表示功能相同。'
    memo+=f" 原生执行耗时{ex['seconds']:.1f}秒；完整基线logits的首选token与四代码分数从原生数组独立核对，干预状态按原生位模式保存。"
    memo+=' 模型计算定义为z=RMSNorm(r)、G=W_gate z、V=W_up z、A=SiLU(G)⊙V、MLP=W_down A；G/V/A在相同自然前向采集，原生乘积逐位核对。固定单元集合J的理想实数写入差为Δr=−Σ_(j∈J) A_j W_down[:,j]，实际BF16干预重新执行投影并计入舍入，不能要求浮点输出满足精确加法。上述均为可追溯的模型定义，不是新发现的语言数学定律。题目、历史、当前三个分区包含的位置数不相等，较大效应不能单凭数值大小解释成更关键的信息来源；同时保存各条件实际残差位移范数。'
    memo+=f" 功效规划按20个百分点效应、30%不一致率、六比较α=.05/6，92家族功效{p['power']['achieved_power']:.3f}；小效应未显著不等于没有作用。"
    memo+=' 本内部实验作用于明确计数接口的Code读出，尚不是原full接口多打印一步的直接中介检验；保存更早位置的场不等于已经在那些生成时刻完成干预。'
    memo+=' 数值校准在4家族的8个视图上同时比较标准HF的4个读出位置×41个原生残差边界（逐向量相对L2最大值≤.02）、Code位置最终归一化状态和全词表logits（最大绝对差≤.5）。这些是固定自产前缀重新前向的场，并非原始逐步生成时保存的KV缓存状态。'
    S.save('native-summary.json',dict(status='completed' if admitted else 'numeric_admission_failed',memo_text=memo,analysis_sha256=S.file_sha(S.OUT/'native-analysis.json'),confirmation_families=92,numeric_admitted=admitted,norm_control_admitted=norm_admitted))
    S.save('native-feature-card.json',dict(feature_id=S.SNAPSHOT+':block36:post_gate_times_up:fixed32:'+p['candidate_sha256'][:12],
        object_type='Previously frozen group of32 native MLP units, not a single residual coordinate or newly discovered concept',
        model_snapshot=S.SNAPSHOT,block_index=36,boundary='post SiLU(gate_proj) times up_proj, before down_proj',
        unit_ids=p['unit_ids'],matched_unit_ids=p['matched_unit_ids'],selection_source='native-candidate.json',
        method_version=S.file_sha(S.OUT/'native-plan.json'),controlled_activation_fields=array_hashes,
        representative_contexts=[dict(id=r['id'],generated_prefix=r['generated_prefix']) for r in rows[:2]],
        representative_selection='First2 frozen confirmation views, not chosen for effect magnitude',
        natural_corpus_statistics='Not newly sampled; this stage uses controlled arithmetic templates only',
        template_selectivity='No new selectivity search; all frozen families retained regardless of baseline correctness',
        generalization='92 nominal new registered program families at the counted output interface;4 numerical pilot families excluded',
        unembedding_readout='Final full-vocabulary native logits and code margins retained; direct linear readout of isolated group not separately tested',
        causal_results=summaries,primary_comparisons=primary,source_partitions=p['source_partition'],downstream_protocol=p['paths'],
        evidence_level='Scoped original-model interventions executed; interpret outcome and controls separately' if admitted else 'Numerical diagnosis only; confirmatory admission failed',
        limitations=stats['limits'],counterexamples='All original failures retained in counted0/2.json and all intervention outcomes in native/*/record.json',
        formal_center_registration=False))
    decision=S.read(S.OUT/'stage-decision.json')
    S.save('native-stage-decision.json',dict(behavior_status=decision['status'],native_status='completed' if admitted else 'numeric_admission_failed',native_effect_records=len(effects),confirmatory_admitted=admitted,formation_claim='Not established. Conditional output routing only.'))
    section='<h2>B36原生来源与后续路径</h2><p>'+html.escape(memo)+'</p><img src="native-all-coordinates.png" alt="代码读出前全部原生坐标，原值均值与逐层RMS缩放" style="max-width:100%;height:auto"><p>41个残差边界、5120个坐标均保留原始顺序；第一图为原值均值，第二图除以各层总体RMS，分别使用对称完整色标。跨样本均值可能抵消条件差异，全部逐样本原生场仍保留；图形相似本身不计机制证据。</p><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre><p><a href="native-analysis.json">原生分析</a> · <a href="native-plan.json">冻结原生方案</a> · <a href="native-stage-decision.json">原生阶段状态</a></p>'
    report=S.OUT/'report.html';text=report.read_text(encoding='utf8');report.write_text(text.replace('</html>',section+'</html>'),encoding='utf8')
    print(json.dumps(dict(admitted=admitted,norm_control_admitted=norm_admitted,primary=primary,summaries=summaries),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
