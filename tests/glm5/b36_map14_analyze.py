"""Independent artifact audit and preregistered paired behavioral comparisons."""
import math,re,ast,operator,collections,json,html
import numpy as np
import b36_map14 as M
S=M.S

def wilson(k,n):
    z=1.95996398454;p=k/n;a=1+z*z/n;mid=(p+z*z/(2*n))/a;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [mid-half,mid+half]

def run():
    S.configure();plan=S.read(S.OUT/'plan.json');by={r['id']:r for r in plan['rows']};results=[]
    for name,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name
    root=S.ROOT/'tests/glm5/result'
    prior_paths=[root/'b36_generated14_20261113_v1/plan.json',root/'b36_semantic14_20261113_v1/arithmetic-families.json',root/'b36_semantic14_20261113_v1/expanded-plan.json']
    prior=list(S.read(prior_paths[0])['families']);ar=S.read(prior_paths[1]);used=set(ar['diagnostic_family_ids'])|{g['id'] for g in S.read(prior_paths[2])['families']};prior += [g for g in ar['families'] if g['id'] in used]
    recipes=[S.digest(g['ops'][:g['depth']]) for g in plan['families']]
    prior_recipes={S.digest(g['ops'][:d]) for g in prior for d in [2,3]}
    prior_prefixes={S.digest([g['start'],g['ops'][:d]]) for g in prior for d in [1,2,3]}
    overlaps=sum(S.digest([g['start'],g['ops'][:d]]) in prior_prefixes for g in plan['families'] for d in range(1,g['depth']+1))
    assert overlaps==0 and len({S.digest([g['start'],g['ops'][:g['depth']]]) for g in plan['families']})==96
    novelty=dict(distinct_visible_operation_recipes=len(set(recipes)),families_with_recipe_seen_previously=sum(r in prior_recipes for r in recipes),semantic_prefix_overlaps=overlaps,prior_files={str(p.relative_to(S.ROOT)):S.file_sha(p) for p in prior_paths},scope='All96 start+visible-operation instances and semantic prefixes are new; recipe-level novelty is not guaranteed. Family IDs use full three-operation candidates even when only two operations are shown. Frozen-plan wording about new whole programs must not be read as unseen visible recipes.')
    from transformers import AutoTokenizer
    from tools.provenance import check_provenance
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    for arm in M.ARMS:
        d=S.read(S.OUT/(arm+'.json'));assert d['plan_sha256']==S.file_sha(S.OUT/'plan.json');assert not check_provenance(d);assert len(d['results'])==96;results+=d['results']
    assert len({r['id'] for r in results})==len(by)==672
    for r in results:
        row=by[r['id']];assert tok.encode(row['prompt'],add_special_tokens=False)==row['tokens']
        assert tok.decode(r['tokens'],skip_special_tokens=False)==r['text']
        eos=tok.eos_token_id in r['tokens'];assert M.evaluate(r['text'],row,eos)==r['parsed']
        if '```python' in row['prompt']:
            code=row['prompt'].split('```python\n')[1].split('\n```')[0];state=None;truth=[]
            for node in ast.parse(code).body:
                if isinstance(node,ast.Assign):assert node.targets[0].id=='x';state=ast.literal_eval(node.value)
                else:assert isinstance(node,ast.AugAssign) and node.target.id=='x';state={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul}[type(node.op)](state,ast.literal_eval(node.value));truth.append(state)
            assert truth==row['truth']
        if row['arm']!='compute':
            table=re.findall(r'Row ([1-4]): value (\d+), code ([A-D])',row['prompt']);assert len(table)==4
            assert [int(t[1]) for t in table]==row['values'] and [t[2] for t in table]==row['codes']
        if row['arm'].startswith('match'):assert f"Target value: {row['truth'][-1]}\n" in row['prompt']
        if row['arm'].startswith('encode'):assert f"Selected row: {row['slot']}\n" in row['prompt']
    groups=collections.defaultdict(dict)
    for r in results:groups[by[r['id']]['family']][by[r['id']]['arm']]=r
    cells=[]
    for arm in M.ARMS:
      for depth in [2,3]:
        rr=[r for r in results if by[r['id']]['arm']==arm and by[r['id']]['depth']==depth]
        fields=['steps_correct','value_correct','slot_correct','code_correct','format_correct','eos','joint'];counts={k:sum(r['parsed']['metrics'][k] for r in rr) for k in fields}
        cells.append(dict(arm=arm,depth=depth,n=len(rr),**counts,passed=counts['joint']>=44,joint_wilson95=wilson(counts['joint'],len(rr))))
    primary=[];rng=np.random.default_rng(S.SEED)
    # Exploratory dependence sensitivity: different recipes can compute the same affine function.
    affine={}
    for g in plan['families']:
        aa,bb=1,0
        for op,k in g['ops'][:g['depth']]:
            if op=='mul':aa*=k;bb*=k
            elif op=='add':bb+=k
            else:bb-=k
        affine[g['id']]=(aa,bb)
    cluster_keys=sorted(set(affine.values()));family_order=list(groups)
    for left,right in [('integrated0','match0'),('match0','encode0')]:
        aa=np.array([g[left]['parsed']['metrics']['code_correct'] for g in groups.values()]);bb=np.array([g[right]['parsed']['metrics']['code_correct'] for g in groups.values()]);gain=int((~aa&bb).sum());loss=int((aa&~bb).sum());disc=gain+loss
        p=min(1.,2*sum(math.comb(disc,k) for k in range(min(gain,loss)+1))/2**disc) if disc else 1.
        if disc:
            from scipy.stats import binomtest
            assert abs(p-float(binomtest(min(gain,loss),disc,.5).pvalue))<1e-12
        delta=bb.astype(float)-aa;boot=delta[rng.integers(0,96,(20000,96))].mean(1)
        totals=np.array([sum(delta[i] for i,f in enumerate(family_order) if affine[f]==key) for key in cluster_keys]);sizes=np.array([sum(affine[f]==key for f in family_order) for key in cluster_keys]);draws=rng.integers(0,len(cluster_keys),(20000,len(cluster_keys)));cluster_boot=totals[draws].sum(1)/sizes[draws].sum(1)
        primary.append(dict(left=left,right=right,n=96,gained=gain,lost=loss,delta=float(delta.mean()),p_exact=p,descriptive_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),exploratory_affine_cluster_count=len(cluster_keys),exploratory_affine_cluster95=np.quantile(cluster_boot,[.025,.975]).tolist(),cluster_note='Sensitivity only: group recipes sharing final x->a*x+b, without using responses to define groups. Primary preregistered family-level test unchanged.'))
    carry=0.
    for rank,p in enumerate(sorted(primary,key=lambda x:x['p_exact'])):carry=max(carry,(2-rank)*p['p_exact']);p['p_holm']=min(1.,carry);p['reject']=p['p_holm']<.05
    breakdown=[];examples=[]
    for arm in M.ARMS:
        counts=collections.Counter()
        for g in groups.values():
            r=g[arm];q=r['parsed'];m=q['metrics'];row=by[r['id']]
            needs_steps=arm=='compute' or arm.startswith('integrated')
            needs_slot=arm.startswith(('integrated','match'))
            needs_code=arm!='compute'
            if needs_steps and q['steps'] is None:kind='steps_unparseable'
            elif needs_steps and len(q['steps'])!=row['depth']:kind='wrong_number_of_printed_steps'
            elif needs_steps and not m['steps_correct']:kind='printed_intermediate_error'
            elif needs_steps and q['value'] is None:kind='value_unparseable'
            elif needs_steps and not m['value_correct']:kind='printed_value_error_after_correct_steps'
            elif needs_slot and q['slot'] is None:kind='slot_unparseable'
            elif needs_slot and not m['slot_correct']:kind='wrong_slot_after_supplied_value' if arm.startswith('match') else 'wrong_slot_after_correct_value'
            elif needs_code and q['code'] is None:kind='code_unparseable'
            elif needs_code and not m['code_correct']:kind='wrong_code_after_supplied_slot' if arm.startswith('encode') else 'wrong_code_after_correct_slot'
            elif not m['format_correct']:kind='format_only'
            elif not m['eos']:kind='no_EOS'
            else:kind='joint_success'
            counts[kind]+=1
            if not m['joint'] and sum(e['arm']==arm for e in examples)<5:examples.append(dict(id=r['id'],arm=arm,kind=kind,truth=row['truth'],values=row['values'],codes=row['codes'],expected_slot=row['slot'],expected_code=row['code'],text=r['text'],parsed=q))
        subset=[g[arm] for g in groups.values()]
        marginal={}
        if arm.startswith('integrated'):
            marginal.update(final_code_correct_despite_wrong_printed_steps=sum(r['parsed']['metrics']['code_correct'] and not r['parsed']['metrics']['steps_correct'] for r in subset),final_code_correct_despite_wrong_value=sum(r['parsed']['metrics']['code_correct'] and not r['parsed']['metrics']['value_correct'] for r in subset),value_matches_printed_slot=sum(r['parsed']['metrics']['slot_matches_generated_value'] is True for r in subset))
            marginal.update(printed_step_count_offsets=dict(collections.Counter(len(r['parsed']['steps'])-by[r['id']]['depth'] for r in subset if r['parsed']['steps'] is not None)),correct_value_n=sum(r['parsed']['metrics']['value_correct'] for r in subset),wrong_slot_given_correct_value=sum(r['parsed']['metrics']['value_correct'] and not r['parsed']['metrics']['slot_correct'] for r in subset),correct_slot_n=sum(r['parsed']['metrics']['slot_correct'] for r in subset),wrong_code_given_correct_slot=sum(r['parsed']['metrics']['slot_correct'] and not r['parsed']['metrics']['code_correct'] for r in subset))
        if needs_slot:marginal['code_matches_printed_slot']=sum(r['parsed']['metrics']['code_consistent_with_generated_slot'] is True for r in subset)
        breakdown.append(dict(arm=arm,counts=dict(counts),marginal_consistency=marginal))
    rotations=[]
    for stem in ['integrated','match','encode']:
        bothwrong=stablewrong=slotsame=0
        for g in groups.values():
            a,b=g[stem+'0'],g[stem+'2'];ra,rb=by[a['id']],by[b['id']];qa,qb=a['parsed'],b['parsed']
            if qa['code'] and qb['code']:
                va=ra['values'][ra['codes'].index(qa['code'])];vb=rb['values'][rb['codes'].index(qb['code'])]
                if not qa['metrics']['code_correct'] and not qb['metrics']['code_correct']:bothwrong+=1;stablewrong+=va==vb
            if qa['slot'] is not None and qb['slot'] is not None:slotsame+=qa['slot']==qb['slot']
        rotations.append(dict(arm=stem,both_codes_wrong=bothwrong,same_wrong_numeric_choice=stablewrong,same_printed_slot=slotsame,scope='Posthoc descriptive paired mappings; no extra independentN.'))
    passed=all(c['passed'] for c in cells)
    totals={arm:{k:sum(c[k] for c in cells if c['arm']==arm) for k in ['n','steps_correct','value_correct','slot_correct','code_correct','joint']} for arm in M.ARMS}
    interpretation=f"仅计算时，步骤全对{totals['compute']['steps_correct']}/96，最终数值正确{totals['compute']['value_correct']}/96。加入选项及行号、代码输出后，两种轮换的步骤全对分别{totals['integrated0']['steps_correct']}/96与{totals['integrated2']['steps_correct']}/96，最终代码正确分别{totals['integrated0']['code_correct']}/96与{totals['integrated2']['code_correct']}/96。给定正确数值时，行号正确分别{totals['match0']['slot_correct']}/96与{totals['match2']['slot_correct']}/96，代码正确分别{totals['match0']['code_correct']}/96与{totals['match2']['code_correct']}/96；给定正确行号时，代码正确分别{totals['encode0']['code_correct']}/96与{totals['encode2']['code_correct']}/96。"
    interpretation+=' 这些配对结果缩小的是行为失败范围：端到端条件同时改变选项上下文、要求的字段和自产历史，尚未把三者分开。输出步骤的错误不能直接定位内部语义计算的错误，也不能把提供上游真值后的恢复当作神经元中介证据。'
    descriptive='本轮端到端两组分别32和38条未联合通过，均恰好多打印一步；正确Value之后选错Slot为0/69与0/62，正确Slot之后写错Code为0/70与0/63。相反，6条和5条打印步骤有误却仍给对最终代码。两种轮换都选错代码的22个家族，代码所指向的错误数值全部相同。这些是事后可见输出描述，不能把零次观察错误提升成绝对不出错，也不能把额外步骤等同于内部一定多算了一步。'
    interpretation+=' '+descriptive
    S.save('analysis.json',dict(cells=cells,totals=totals,interpretation=interpretation,novelty_audit=novelty,gate_passed=passed,primary_comparisons=primary,output_error_breakdown=breakdown,rotation_descriptions=rotations,examples=examples,n_families=96,n_generated=672,audit='All frozen source hashes, input tokens, rendered program truth, visible tables, output token decoding, metrics and result provenance verified.',source_sha256=S.file_sha(__file__),limit='Visible output-stage decomposition and supplied-input controls, not identification of hidden causal modules. Printed traces need not be faithful to internal computation.'))
    headers=['任务','步数','N','步骤全对','数值正确','行号正确','代码正确','格式','EOS','联合']
    lines=[]
    for c in cells:
        irrelevant={'steps_correct','value_correct'} if c['arm'].startswith(('match','encode')) else set()
        if c['arm'].startswith('encode'):irrelevant.add('slot_correct')
        if c['arm']=='compute':irrelevant|={'slot_correct','code_correct'}
        lines.append('<tr>'+''.join('<td>'+('—' if k in irrelevant else str(c[k]))+'</td>' for k in ['arm','depth','n','steps_correct','value_correct','slot_correct','code_correct','format_correct','eos','joint'])+'</tr>')
    body='<h1>数值 → 选项行 → 答案代码</h1><p>'+S.RUN+' · '+S.now()+'</p><p>96个独立新程序家族，672次配对生成。两步与三步各48个；各代码视图不是额外独立样本。14个格各需至少44/48联合成功。</p><p><strong>'+('行为过门，需按独立机制计划续研。' if passed else '联合行为门未通过；不启动B36机制正例实验。')+'</strong></p><div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(lines)+'</table></div>'
    body+='<p>compute自行计算；integrated计算后输出数值、行号和代码；match提供正确数值；encode提供正确行号。0/2为配对的代码轮换，选项数值和行号保持不变。后两项是人为提供上游结果的对照，不计为模型自身完成了原始计算。</p><h2>主要发现</h2><p>'+html.escape(interpretation)+'</p><h2>冻结的主要比较</h2><pre>'+html.escape(json.dumps(primary,ensure_ascii=False,indent=2))+'</pre><p>两个配对精确McNemar检验按Holm校正。功效按16个百分点差异、20%翻转率、N96规划；不保证小效应有充分功效。区间为描述性家族重采样95%。家族独立是推断近似，仿射聚类只是敏感性检查。零不一致对时重采样区间退化为[0,0]，不能证明总体等价或不存在小差异。</p><h2>可见错误分布</h2><pre>'+html.escape(json.dumps(breakdown,ensure_ascii=False,indent=2))+'</pre><p>正确数值后错误行号与正确行号后错误代码分别记账。它们是输出上的分离；提供数值/行号同时改变输入条件，不能直接推出独立的内部模块。</p><h2>真实反例</h2>'
    body+='<p>材料独立性审计：96个初始值与实际操作组成的新实例，语义前缀无旧材料重合；实际显示的操作序列只有93种，其中9个家族沿用旧序列。家族ID包含完整三操作候选，两步题只显示前两项。因此不能将冻结计划中的新程序措辞理解为全部未见操作组合。</p>'
    for e in examples:body+='<details><summary>'+html.escape(e['id']+' / '+e['kind'])+'</summary><pre>'+html.escape(json.dumps(e,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<h2>证据边界与下一步</h2><p>打印中间步骤不等于直接观察内部推理。程序长度不等于神经层数。两步与三步由不同家族承担，深度间差异不是同程序配对因果效应。本轮确认的是可见计算/选择/代码行为的范围；任何B36干预、层间与生成时间比较须先满足冻结准入并另存原生位置、模型自身历史和匹配对照。</p><p><a href="plan.json">冻结计划</a> · <a href="analysis.json">完整审计与统计</a> · <a href="source.zip">源码</a> · <a href="manifest.json">文件清单</a></p>'
    (S.OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:1150px;margin:32px auto;padding:20px;font:16px/1.7 sans-serif;color:#293d48}.scroll{overflow:auto}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#edf2f4}pre{background:#f4f6f7;white-space:pre-wrap;overflow-wrap:anywhere;padding:18px}details{border-bottom:1px solid #ddd;padding:12px}summary{cursor:pointer}a{color:#176581}</style>'+body+'</html>',encoding='utf8')
    print(json.dumps(dict(gate_passed=passed,passed_cells=sum(c['passed'] for c in cells),cells=cells,primary=primary,breakdown=breakdown),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
