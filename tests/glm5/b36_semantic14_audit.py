"""Recompute material truth, tokens, frozen identities and all behavioral counts."""
import ast,collections,operator,re,math,time
import b36_semantic14 as S

STAGES=['pilot','interface','arithmetic-interface','fewshot','order-pilot','python','expanded']
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;a=1+z*z/n;c=(p+z*z/(2*n))/a;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [c-r,c+r]

def run():
    S.configure();start=time.monotonic()
    from transformers import AutoTokenizer
    from tools.provenance import check_provenance
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    graphs={g['id']:g for g in S.read(S.OUT/'graph-families.json')['graphs']}
    arith={g['id']:g for g in S.read(S.OUT/'arithmetic-families.json')['families']}
    stages=[];observed_prompts=set();observed_families=set();sources={};truth_count=0;examples=[]
    for stage in STAGES:
        plan=S.read(S.OUT/(stage+'-plan.json'));by={r['id']:r for r in plan['rows']};assert len(by)==len(plan['rows'])
        for name,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h; sources[name]=h
        if 'graph_sha256' in plan:assert S.file_sha(S.OUT/'graph-families.json')==plan['graph_sha256']
        for r in plan['rows']:
            assert tok.encode(r['prompt'],add_special_tokens=False)==r['tokens']
            assert r['answer_pos']==len(r['tokens'])-1 and 0<=r['content_end']<=r['answer_pos']<S.LENGTH
            assert len(set(r['choice_ids']))==len(r['choice_ids'])
            for word,tid in zip(r['choices'],r['choice_ids']):assert tok.encode(r['prompt']+r.get('leading','')+word,add_special_tokens=False)==r['tokens']+[tid]
            assert r['expected_answer']==r['choices'][r['gold']]
            if r['family'] in graphs:
                g=graphs[r['family']];x=g['start'];h=[x]
                for op in g['sequence'][:r['depth']]:x=g[op][x];h.append(x)
                assert h==r['intermediate_states'] and x==r['gold']
            elif r['family'] in arith:
                g=arith[r['family']];x=g['start'];h=[x]
                for kind,k in g['ops'][:r['depth']]:x=dict(add=operator.add,sub=operator.sub,mul=operator.mul,mod=operator.mod)[kind](x,k);h.append(x)
                assert h==r['intermediate_states'] and x==r['semantic_gold']==r['semantic_choices'][r['gold']]
                if '```python' in r['core']:
                    body=r['core'].split('```python\n')[1].split('\n```')[0];tree=ast.parse(body);value=None
                    for node in tree.body:
                        if isinstance(node,ast.Assign):assert len(node.targets)==1 and node.targets[0].id=='x';value=ast.literal_eval(node.value)
                        elif isinstance(node,ast.AugAssign):assert node.target.id=='x';value={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Mod:operator.mod}[type(node.op)](value,ast.literal_eval(node.value))
                        else:assert isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and node.value.func.id=='print'
                    assert value==x
            else:
                assert stage=='order-pilot';edges=set()
                for a,rel,b in re.findall(r'(\w+) is (taller|shorter|older|younger) than (\w+)\.',r['core']):edges.add((a,b) if rel in ['taller','older'] else (b,a))
                assert len(edges)==5
                for _ in range(6):edges|={(a,d) for a,b in list(edges) for c,d in list(edges) if b==c}
                a,b=r['query'];winner=a if (a,b) in edges else b;assert (a,b) in edges or (b,a) in edges
                assert winner==r['semantic_answer']==r['semantic_choices'][r['gold']]
            truth_count+=1
        results=[];dest=S.OUT/(stage+'-score');ph=S.file_sha(S.OUT/(stage+'-plan.json'))
        for path in sorted(dest.glob('chunk-*.json')):
            chunk=S.read(path);assert not check_provenance(chunk);assert chunk['plan_sha256']==ph
            assert chunk['row_ids']==[r['id'] for r in chunk['results']];results+=chunk['results']
        assert len({r['id'] for r in results})==len(results)
        cells=[]
        for key in sorted({(r['frame'],r['branch'],r['depth'],r['format']) for r in plan['rows']}):
            rows=[r for r in results if tuple(by[r['id']][k] for k in ['frame','branch','depth','format'])==key]
            planned=sum(tuple(r[k] for k in ['frame','branch','depth','format'])==key for r in plan['rows'])
            n=len(rows);strict=sum(r['strict_correct'] for r in rows);top=sum(r['top1_correct'] for r in rows)
            cells.append(dict(zip(['frame','branch','depth','format'],key),n=n,planned=planned,strict_correct=strict,top1_correct=top,top1_wilson95=wilson(top,n),passed=(n==planned and min(strict,top)>=math.ceil(.9*planned)),best_possible_accuracy=(planned-n+min(strict,top))/planned))
        for r in results:
            row=by[r['id']];score=r['scores'];assert len(score)==len(row['choice_ids']) and all(math.isfinite(v) for v in score)
            assert r['strict_correct']==(score[row['gold']]>max(v for i,v in enumerate(score) if i!=row['gold']))
            assert r['top1_correct']==(r['top1']==row['choice_ids'][row['gold']]) and r['top1']==r['top10_ids'][0]
            observed_prompts.add(S.digest(row['tokens']));observed_families.add(row['family'])
            if stage=='expanded' and not r['top1_correct'] and row['format']=='direct' and len(examples)<6:examples.append(dict(id=row['id'],core=row['core'],truth=row['intermediate_states'],expected=row['expected_answer'],actual=tok.decode([r['top1']]),forced_correct=r['strict_correct']))
        summary=dest/'summary.json'
        if stage!='pilot':
            assert summary.exists() and len(results)==len(plan['rows']);d=S.read(summary);assert not check_provenance(d) and d['results']==results
            original={(c['frame'],c['branch'],c['depth'],c['format']):c for c in d['report']['cells']}
            for c in cells:
                o=original[tuple(c[k] for k in ['frame','branch','depth','format'])]
                assert all(o[k]==c[k] for k in ['n','strict_correct','top1_correct','passed'])
        else:assert len(results)==S.read(S.OUT/'pilot-futility-stop.json')['completed_rows']
        frames=[]
        for f in sorted({r['frame'] for r in plan['rows']}):
            cc=[c for c in cells if c['frame']==f]
            frames.append(dict(frame=f,n=sum(c['n'] for c in cc),planned=sum(c['planned'] for c in cc),strict=sum(c['strict_correct'] for c in cc),top1=sum(c['top1_correct'] for c in cc),passed=all(c['passed'] for c in cc),pass_cells=sum(c['passed'] for c in cc),total_cells=len(cc)))
        stages.append(dict(stage=stage,planned=len(plan['rows']),completed=len(results),families=len({by[r['id']]['family'] for r in results}),cells=cells,frames=frames,plan_sha256=ph))
        S.log('audit',len(stages),len(STAGES),start)
    expanded=S.read(S.OUT/'expanded-plan.json');old=set(S.read(S.OUT/'arithmetic-families.json')['diagnostic_family_ids'])
    key=lambda g,d:S.digest([g['start'],g['ops'][:d]])
    prior={key(arith[f],d) for f in old for d in [1,2,3]};new=[key(g,d) for g in expanded['families'] for d in [1,2,3]]
    assert len(set(new))==144 and not prior.intersection(new)
    assert not {g['id'] for g in graphs.values() if g['split'] in ['confirm','lockbox','discovery']}.intersection(observed_families)
    assert all(arith[f]['partition']=='discovery' for f in observed_families if f in arith)
    hf=S.read(S.OUT/'hf-independent-check.json');assert hf['native_vs_HF_bitwise'] and all(r['max_logit_difference']==0 for r in hf['rows'])
    assert S.read(S.OUT/'cached_source_identity_check.json')['all_matched']
    progress=[__import__('json').loads(line) for line in (S.OUT/'progress.jsonl').read_text().splitlines() if line.strip()]
    resources=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in progress),dxgi_shared_peak_bytes=max(r.get('shared_gpu_bytes',0) for r in progress))
    S.save('audit.json',dict(status='passed',scope='Technical recomputation only; one local node, no independent scientific replication.',stages=stages,material_rows_checked=truth_count,completed_14b_measurements=sum(s['completed'] for s in stages),unique_prompt_hashes=len(observed_prompts),unique_construction_families=len(observed_families),executed_sources=sources,examples=examples,resources=resources,
        heldout_responses_opened=0,new_mechanism_interventions=0,new_hidden_field_collection=0,expanded_prefix_disjoint=True,
        model_audit='14 original model/tokenizer files rehashed; match local cached HF tree revision40c069824f4251a91eefaf281ebe4c544efd3e18. No remote source attestation.',
        boundary='Whole-program splitting alone could share short arithmetic prefixes. Expanded pilot explicitly removes them. Any future confirm must enforce the same constraint across all exposed prefixes. All interface selection is exploratory. Small-cell percentages are not population accuracy.'))
    print('Audit passed',sum(s['completed'] for s in stages),'measurements',len(observed_families),'families',flush=True)

if __name__=='__main__':run()
