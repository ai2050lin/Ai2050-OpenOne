"""Audit all frozen artifacts and test six predeclared behavioral contrasts."""
import ast,operator,re,math,collections,json,html
import numpy as np
import b36_transition14 as M
S=M.S

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-radius,mid+radius]

def run():
    S.configure();plan=S.read(S.OUT/'plan.json');supp=S.read(S.OUT/'length-plan.json')
    assert supp['main_plan_sha256']==S.file_sha(S.OUT/'plan.json') and supp['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_transition14_length.py')
    natural_arms=M.ARMS+['length_control'];natural_rows=plan['rows']+supp['rows'];by={r['id']:r for r in natural_rows}
    for name,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name
    for name,h in plan['prior_files'].items():assert S.file_sha(S.ROOT/name)==h,name
    from transformers import AutoTokenizer
    from tools.provenance import check_provenance
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);all_results={};source_hashes={}
    for arm in natural_arms+M.HISTORY:
        path=S.OUT/(arm+'.json');d=S.read(path);assert not check_provenance(d) and d['plan_sha256']==S.file_sha(S.OUT/'plan.json');all_results[arm]=d['results'];source_hashes[arm]=S.file_sha(path)
        if arm in natural_arms:assert len(d['results'])==96
        if arm=='length_control':assert d['supplement_plan_sha256']==S.file_sha(S.OUT/'length-plan.json')
    hp=S.read(S.OUT/'history-plan.json');assert hp['plan_sha256']==S.file_sha(S.OUT/'plan.json') and hp['source_sha256']==source_hashes['full']
    for arm in M.HISTORY:
        assert S.read(S.OUT/(arm+'.json'))['history_plan_sha256']==S.file_sha(S.OUT/'history-plan.json')
        by.update({r['id']:r for r in hp['rows'][arm]})
    seen=set()
    for arm,rr in all_results.items():
        expected={r['id'] for r in natural_rows if r['arm']==arm} if arm in natural_arms else {r['id'] for r in hp['rows'][arm]}
        assert {r['id'] for r in rr}==expected
        for r in rr:
            assert r['id'] not in seen;seen.add(r['id']);row=by[r['id']]
            tail=tok.decode(r['tokens'],skip_special_tokens=False);assert tail==r['tail_text'] and row.get('prefix','')+tail==r['text']
            eos=tok.eos_token_id in r['tokens'];assert M.evaluate(r['text'],row,eos)==r['parsed']
            if arm in natural_arms:
                assert tok.encode(row['prompt'],add_special_tokens=False)==row['tokens']
                code=row['prompt'].split('```python\n')[1].split('\n```')[0];truth=[];parsed_ops=[];state=None;start=None
                for node in ast.parse(code).body:
                    if isinstance(node,ast.Assign):assert node.targets[0].id=='x';state=start=ast.literal_eval(node.value)
                    else:
                        assert isinstance(node,ast.AugAssign) and node.target.id=='x';k=ast.literal_eval(node.value);name={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul'}[type(node.op)];parsed_ops.append([name,k]);state={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[name](state,k);truth.append(state)
                assert truth==row['truth'] and parsed_ops==row['ops'] and start==row['start']
                table=re.findall(r'Row ([1-4]): value (\d+), code ([A-D])',row['prompt'])
                if arm not in ['no_table','length_control']:assert [int(t[1]) for t in table]==row['values'] and [t[2] for t in table]==row['codes']
                else:assert not table
            else:
                assert tok.decode(row['prefix_tokens'],skip_special_tokens=False)==row['prefix']
                original=by[row['source_id']];assert row['tokens']==original['tokens']+row['prefix_tokens']
                if arm=='replay':assert r['replay_exact']==(r['tokens']==row['original_tail'])
    full={by[r['id']]['family']:r for r in all_results['full']}
    for arm in M.HISTORY:
        for row in hp['rows'][arm]:
            origin=full[row['family']];prefix=M.history_prefix(origin['text'],origin['tokens'],row['depth'],tok)
            assert prefix==row['source_prefix'] and row['original_tail']==origin['tokens'][prefix['cut']:]
            expected=prefix['text'] if arm in ['replay','boundary'] else M.replace_steps(prefix['text'],row['truth'] if arm=='gold' else row['truth'][:-1]+[row['foil']])
            if arm!='replay':expected+='\nValue: '
            assert expected==row['prefix']
            if arm=='replay':assert row['prefix_tokens']==prefix['tokens']
    for gold,foil in zip(hp['rows']['gold'],hp['rows']['foil']):assert gold['family']==foil['family'] and len(gold['prefix_tokens'])==len(foil['prefix_tokens'])
    regenerated,prior_paths=M.families();assert S.digest(regenerated)==S.digest(plan['families'])
    prior,_=M.prior_materials();oldrecipes={S.digest(g['ops'][:d]) for g in prior for d in [2,3]};oldprefixes={S.digest([g['start'],g['ops'][:d]]) for g in prior for d in [1,2,3]}
    assert len({S.digest(g['ops']) for g in plan['families']})==96
    assert not any(S.digest(g['ops']) in oldrecipes or any(S.digest([g['start'],g['ops'][:d]]) in oldprefixes for d in range(1,g['depth']+1)) for g in plan['families'])
    maps={a:{by[r['id']]['family']:r for r in rr} for a,rr in all_results.items()};cells=[]
    metrics=['step_count_correct','steps_correct','value_correct','slot_correct','code_correct','format_correct','eos','joint']
    for arm,rr in all_results.items():
        for depth in [2,3]:
            q=[r for r in rr if by[r['id']]['depth']==depth];counts={k:sum(r['parsed']['metrics'][k] for r in q) for k in metrics}
            cells.append(dict(arm=arm,depth=depth,n=len(q),**counts,joint_wilson95=wilson(counts['joint'],len(q)),autonomous=arm in natural_arms,mechanism_target=arm in plan['prereg']['mechanism_arms']))
    replay_n=len(all_results['replay']);replay_exact=sum(r['replay_exact'] for r in all_results['replay']);history_admitted=replay_n==96 and replay_exact==96 and not hp['missing']
    affine={}
    for g in plan['families']:
        aa,bb=1,0
        for op,k in g['ops']:
            if op=='mul':aa*=k;bb*=k
            elif op=='add':bb+=k
            else:bb-=k
        affine[g['id']]=(aa,bb)
    primary=[];rng=np.random.default_rng(S.SEED)
    for test in plan['prereg']['primary']+[supp['test']]:
        left,right,key=test['left'],test['right'],test['metric'];ids=sorted(set(maps[left])&set(maps[right]));n=len(ids)
        aa=np.asarray([maps[left][i]['parsed']['metrics'][key] for i in ids],dtype=bool);bb=np.asarray([maps[right][i]['parsed']['metrics'][key] for i in ids],dtype=bool)
        gain=int((~aa&bb).sum());loss=int((aa&~bb).sum());discordant=gain+loss;delta=bb.astype(float)-aa.astype(float)
        admitted=n==96 and (left not in M.HISTORY or history_admitted)
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gain,loss)+1))/2**discordant) if discordant else 1.
        if discordant:
            from scipy.stats import binomtest
            assert abs(p-float(binomtest(min(gain,loss),discordant,.5).pvalue))<1e-12
        ci=cluster_ci=None;keys=sorted({affine[i] for i in ids})
        if n:
            boot=delta[rng.integers(0,n,(20000,n))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
            totals=np.array([sum(delta[k] for k,i in enumerate(ids) if affine[i]==a) for a in keys]);sizes=np.array([sum(affine[i]==a for i in ids) for a in keys]);draw=rng.integers(0,len(keys),(20000,len(keys)));cluster_ci=np.quantile(totals[draw].sum(1)/sizes[draw].sum(1),[.025,.975]).tolist()
        primary.append(dict(**test,n=n,left_correct=int(aa.sum()),right_correct=int(bb.sum()),gained=gain,lost=loss,delta=float(delta.mean()) if n else None,admitted=admitted,p_exact=p if admitted else None,p_for_correction=p if admitted else 1.,descriptive_family95=ci,exploratory_affine_cluster95=cluster_ci,affine_clusters=len(keys)))
    carry=0.
    for rank,item in enumerate(sorted(primary,key=lambda q:q['p_for_correction'])):
        carry=max(carry,(len(primary)-rank)*item['p_for_correction']);item['p_holm']=min(1.,carry);item['reject']=item['admitted'] and item['p_holm']<.05
    breakdown=[];examples=[]
    for arm,rr in all_results.items():
        item=dict(arm=arm,n=len(rr),step_patterns=dict(collections.Counter(r['parsed']['step_pattern'] for r in rr)) if 'Steps' in (by[rr[0]['id']]['fields'] if rr else []) else None,value_correct=sum(r['parsed']['metrics']['value_correct'] for r in rr),correct_value_wrong_slot=sum(r['parsed']['metrics']['value_correct'] and not r['parsed']['metrics']['slot_correct'] for r in rr) if arm not in ['no_table','table','code'] else None,correct_slot_wrong_code=sum(r['parsed']['metrics']['slot_correct'] and not r['parsed']['metrics']['code_correct'] for r in rr) if arm not in ['no_table','table','code'] else None,value_tracks_last_printed=sum(r['parsed']['metrics']['value_matches_last_printed_step'] is True for r in rr),code_correct_despite_wrong_steps=sum(r['parsed']['metrics']['code_correct'] and not r['parsed']['metrics']['steps_correct'] for r in rr) if arm!='no_steps' else None)
        if arm=='length_control':item['correct_value_wrong_slot']=item['correct_slot_wrong_code']=None
        if arm in M.HISTORY:
            item.update(value_equals_foil=sum(r['parsed']['value']==by[r['id']]['foil'] for r in rr),supplied_prefix_evidence=True)
        if arm!='no_steps':
            terminal_bad=[r for r in rr if r['parsed']['metrics']['steps_correct'] and not r['parsed']['metrics']['value_correct']]
            repeats=0
            for r in terminal_bad:
                row=by[r['id']];op,k=row['ops'][-1];again={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](row['truth'][-1],k)
                repeats+=r['parsed']['value']==again
            item.update(correct_steps_wrong_value=len(terminal_bad),correct_steps_wrong_value_equals_repeated_last_update=repeats)
        breakdown.append(item)
        failures=[r for r in rr if not r['parsed']['metrics']['joint']]
        for r in failures[:3]:
            row=by[r['id']];examples.append(dict(arm=arm,id=r['id'],start=row['start'],ops=row['ops'],truth=row['truth'],values=row['values'],codes=row['codes'],expected_slot=row['slot'],expected_code=row['code'],prefix=row.get('prefix'),text=r['text'],parsed=r['parsed']))
    target=[c for c in cells if c['mechanism_target']];gate=len(target)==4 and all(c['n']==48 and c['joint']>=44 for c in target)
    rows_by_arm={a:[r for r in natural_rows if r['arm']==a] for a in natural_arms}
    for a,b in zip(rows_by_arm['no_table'],rows_by_arm['table']):assert re.sub(r'\nOptions:\n(?:Row[^\n]+\n){3}Row[^\n]+','',b['prompt'])==a['prompt']
    for r,b in zip(rows_by_arm['length_control'],rows_by_arm['table']):assert r['family']==b['family'] and len(r['tokens'])==len(b['tokens']) and r['prompt'].count('\n')==b['prompt'].count('\n')
    a=dict(cells=cells,primary=primary,breakdown=breakdown,examples=examples,history_audit=dict(n=replay_n,exact=replay_exact,missing=hp['missing'],admitted=history_admitted,gold_prefix_already_equal=sum(r['source_prefix']['values']==r['truth'] for r in hp['rows']['gold']),replay_mismatch_ids=[r['id'] for r in all_results['replay'] if not r['replay_exact']]),gate_passed=gate,gate_scope=plan['prereg']['mechanism_gate'],n_families=96,n_natural=sum(len(all_results[a]) for a in M.ARMS),n_history=sum(len(all_results[a]) for a in M.HISTORY),source_sha256=S.file_sha(__file__),input_plan_sha256=S.file_sha(S.OUT/'plan.json'),raw_artifact_hashes=source_hashes,audit='Frozen sources and prior identities,96 visible-recipe novelty, AST truths, input padding/token identity, all generated tokens/metrics and exact history edits checked. Nonzero-discordance p values checked against scipy.',limits='Outputs are observable traces, not guaranteed faithful internal calculations. Gold/foil/boundary are supplied-history interventions. Family independence is approximate; affine clustering is exploratory. Zero-discordance bootstrap degeneracy does not establish equivalence. Counted interface changes wording and length; no isolated numerical-count claim.')
    a['n_natural']=sum(len(all_results[k]) for k in natural_arms)
    a['length_supplement']=dict(plan_sha256=S.file_sha(S.OUT/'length-plan.json'),exposure=supp['exposure'],correction=supp['correction_amendment'],power=supp['power'],token_and_newline_matching=True)
    S.save('analysis.json',a)
    S.save('stage-decision.json',dict(status='counted_interface_behavior_passed_requires_frozen_native_protocol' if gate else 'counted_interface_behavior_gate_failed',target_cells=target,old_raw_interface_gate='Not retroactively changed.',candidate_file='native-candidate.json',new_native_interventions=0,layer_vs_generation_formation='Not established by this behavior analysis.'))
    headers=['条件','步数','N','步数正确','步骤全对','数值','行号','代码','格式','EOS','联合'];table=[]
    for c in cells:
        arm=c['arm'];needs=['Steps','Value'] if arm=='length_control' else M.FIELDS.get(arm,['Steps','Value','Slot','Code']);irrelevant=set()
        if 'Steps' not in needs:irrelevant|={'step_count_correct','steps_correct'}
        if 'Slot' not in needs:irrelevant.add('slot_correct')
        if 'Code' not in needs:irrelevant.add('code_correct')
        table.append('<tr>'+''.join('<td>'+('—' if k in irrelevant else html.escape(str(c[k])))+'</td>' for k in ['arm','depth','n']+metrics)+'</tr>')
    body='<h1>多打印一步：选项、字段与自产历史</h1><p>'+S.RUN+' · '+S.now()+'</p><p>96个新实例、96种未见实际操作序列；两步和三步各48个，代码/历史视图不增加独立家族数。</p><p><strong>'+('预注册的明确计数接口已过行为门；还需要原生路径实验。' if gate else '预注册的明确计数接口未过行为门；不启动B36机制正例实验。')+'</strong></p><p>no_table/table只改变选项表；table/code/full逐项增加代码和行号字段；no_steps移除步骤字段；counted0/2额外明确更新次数并轮换代码。replay是原样历史回放；boundary强制进入Value字段；gold替换正确步骤；foil只扰动最后一步且保持数字token长度。后三者不能算自主计算成功。</p><div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(table)+'</table></div>'
    body+='<h2>七项配对比较（含长度补充）</h2><pre>'+html.escape(json.dumps(primary,ensure_ascii=False,indent=2))+'</pre><p>七检验Holm校正；缺失或回放未准入的比较仍占校正家族，按p=1保守处理。功效按20个百分点、30%不一致率规划，N96约0.813。区间是描述性重采样，零不一致对下的[0,0]不证明总体等价。</p><p>length_control与table逐题匹配输入token数和换行数，保持Steps/Value要求，使用无关文字替代选项表。补充方案在no_table响应已查看、table响应已运行但未打开内容时冻结；保留原主方案，校正范围从6项保守扩大到7项。无关文字可能重复完整句子，不能断言完全没有语义影响。</p><h2>历史回放审计</h2><pre>'+html.escape(json.dumps(a['history_audit'],ensure_ascii=False,indent=2))+'</pre><h2>错误分布</h2><pre>'+html.escape(json.dumps(breakdown,ensure_ascii=False,indent=2))+'</pre><h2>实际反例</h2>'
    for e in examples:body+='<details><summary>'+html.escape(e['id'])+'</summary><pre>'+html.escape(json.dumps(e,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<h2>证据边界</h2><p>多打印一步不等于内部一定多算一步。人工边界、金标准和扰动历史是输入干预，不能替代原模型神经元路径干预。不同深度是不同家族，不能把深度差当配对因果效应。明确计数接口是预注册的新范围，其通过不替原始易错接口补过。</p><p><a href="plan.json">冻结方案</a> · <a href="history-plan.json">实际历史编辑</a> · <a href="analysis.json">完整审计</a> · <a href="stage-decision.json">阶段判断</a></p>'
    (S.OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 sans-serif;color:#253d47}.scroll{overflow:auto}table{border-collapse:collapse}td,th{white-space:nowrap;border-bottom:1px solid #ddd;padding:9px}th{background:#edf3f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:16px}details{padding:10px;border-bottom:1px solid #ddd}summary{cursor:pointer}a{color:#196482}</style>'+body+'</html>',encoding='utf8')
    print(json.dumps(dict(gate_passed=gate,target_cells=target,history=a['history_audit'],primary=primary,breakdown=breakdown),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
