"""Publish immutable research cases, native indexes and typed evidence cards."""
import argparse

import mlp_functional_common as C
from mlp_atlas_publish_final import Archive


def without(data,*keys):
    return {k:v for k,v in data.items() if k not in keys}


def archive_cases(archive,name,rows):
    size=64
    chunks=[archive.put(f'{name}-cases-{i}',{'schema':'functional_cases.v1','collection':name,'start':i,'cases':rows[i:i+size]})
            for i in range(0,len(rows),size)]
    return {'n':len(rows),'chunk_size':size,'chunks':chunks}


def main(final,observed=False):
    ident=C.identity(); archive=Archive(C.PUBLIC,C.RUN,ident)
    trained=C.read(C.OUT/'dictionary-training.json');dictionary=C.read(C.OUT/'dictionary-evaluation.json')
    documents=C.read(C.OUT/'fresh-documents.json'); pilot=C.read(C.OUT/'pilot-result.json')
    refs={};cases={};cards=[];edges=[]
    refs['dictionary']=archive.put('dictionary-evaluation',without(dictionary,'rows'))
    refs['training']=archive.put('dictionary-training',trained)
    refs['pilot']=archive.put('behavior-pilot',pilot)
    cases['natural']=archive_cases(archive,'natural',[{'source':r,**v} for r,v in zip(documents['natural'],dictionary['rows'])])
    refs['dictionary_states']=archive.put('dictionary-states',{'schema':'research_native_tensor_index.v1',
               'weights':{k:archive.native(C.OUT/r['path'],r['sha256']) for k,r in trained['outputs'].items()}})
    prereg={name:C.read(C.OUT/f'{name}-prereg.json') for name in ['dictionary-training','dictionary-evaluation','controlled','pilot-execution']}

    def card(key,layer,boundary,index,label,method,level,ref,counterexamples,scope):
        id=f'{archive.snap}:{C.RUN}:{key}'
        cards.append({'id':id,'layer':layer,'boundary':boundary,'index':index,'run_id':C.RUN,'label':label,'method':method,
                      'evidence':{'level':level,'artifact_sha256':ref},'counterexamples':counterexamples,'scope':scope})
        return id

    def edge(source,target,relation,level,ref):
        edges.append({'id':C.digest([C.RUN,source,target,relation]),'source':source,'target':target,'relation':relation,'run_id':C.RUN,
                      'evidence':{'level':level,'artifact_sha256':ref}})

    lock=dictionary['summary']['lockbox']['operations']
    card('behavior-sae',35,'mlp.output_sae',{'space':'dictionary','dictionary_size':4096,'active_budget':256},
         'B35 字典行为校准与同预算对照','完整词表 KL、真实 next-token CE 及输出重建联合训练，两个新留出比较',
         'heldout',refs['dictionary'],[{'native256':lock['native256'],'functional_sae':lock['functional_sae'],'comparison_passed':dictionary['both_passed']}],
         {'observe':'native MLP output','kind':'representation','native_units_not_equated':True})
    trace=paths=generated=capability=revision=flow=None;counterexamples=[];behavior_errors=[]
    if final or observed:
        trace=C.read(C.OUT/'trace-result.json');paths=C.read(C.OUT/'path-analysis.json')
        prereg['trace-execution']=C.read(C.OUT/'trace-execution-prereg.json')
        refs['paths']=archive.put('all-writer-effects',paths)
        refs['native']=archive.put('native-fields',{'schema':'research_native_tensor_index.v1','manifest':without(trace,'rows'),
                 'files':[archive.native(C.OUT/r['path'],r['sha256']) for r in trace['native']]+[archive.native(C.OUT/'writer-projections.npz',paths['projection_sha256'])]})
        corpus=C.read(C.OUT/'controlled-inputs.json');byid={r['id']:r for r in trace['rows']}
        cases['controlled']=archive_cases(archive,'controlled',[{**r,'source':without(documents['controlled'][r['b']],'tokens'),
                    'outcome':byid[r['id']]} for r in corpus['rows']])
        behavior_errors=[{'index':i,'expected':r['expected'],'word':byid[r['id']]['word'],'generated':byid[r['id']]['generated']}
                         for i,r in enumerate(corpus['rows']) if not byid[r['id']]['correct']]
        refs['original_gate_disposition']=archive.put('original-gate-disposition',C.read(C.OUT/'generated-causal.json'))
        if final:
            rev=C.OUT/'revision'
            generated=C.read(rev/'generated-causal.json');capability=C.read(rev/'capability-causal.json')
            obs=C.read(rev/'confirm-result.json');pilot2=C.read(rev/'pilot-result.json')
            revision=without(obs,'rows') | {'pilot_cells':pilot2['cells']}
            remainder=rev/'remainder.json'
            if remainder.is_file():
                parts=C.read(remainder);revision['remainder']=parts
                refs['revision_remainder']=archive.put('revision-residual-remainder',parts)
            rev_corpus=C.read(rev/'inputs.json')['rows'];rev_docs=C.read(rev/'documents.json')['docs']
            byid2={r['id']:r for r in pilot2['rows']+obs['rows']}
            cases['revision']=archive_cases(archive,'revision',[{**r,'source':without(rev_docs[r['b']],'tokens'),'outcome':byid2[r['id']]} for r in rev_corpus])
            cases['generated']=archive_cases(archive,'generated',generated['rows'])
            cap=C.read(C.CAP);assert len(cap)==len(capability['rows'])
            cases['capability']=archive_cases(archive,'capability',[{'input':r,**s} for r,s in zip(cap,capability['rows'])])
            refs['revision']=archive.put('independent-revision',{'schema':'functional_revision_evidence.v2','revision':revision,
                         'native':archive.native(C.OUT/obs['native']['path'],obs['native']['sha256'])})
            refs['generated']=archive.put('generated-causal',without(generated,'rows'))
            refs['capability']=archive.put('capability-causal',without(capability,'rows'))
            if (rev/'flow-check.json').is_file():
                flow=C.read(rev/'flow-check.json');refs['flow_boundary']=archive.put('last-block-flow-boundary',flow)
            prereg['revision']=C.read(rev/'prereg.json')
            for name in ['pilot','confirm','generated','capability']:
                prereg[f'revision-{name}']=C.read(rev/f'{name}-execution-prereg.json')
            counterexamples=[{'index':r['index'],'case_id':r['id'],'expected':r['expected'],'clean':r['conditions']['clean']['word'],
                              'edited':r['conditions']['joint_1']['word']}
                             for r in generated['rows'] if r['conditions']['clean']['correct'] and not r['conditions']['joint_1']['correct']]
        rids={};writer_ids={}
        for name,picks in paths['selected'].items():
            title={'location':'地理属性','industry':'生计属性'}[name]
            obs=paths['observations'][name]
            rids[name]=card('incoming-'+name,35,'pre_mlp_residual',{'space':'residual','dimensions':2560},
                            f'问题前 B35 输入 · {title}', '发现集固定方向与阈值，两个留出预测；Q 后缀不可见',
                            'observation',refs['paths'],[{'lockbox':obs['incoming']['lockbox'],'scope':'document-defined attributes'}],obs)
            used_picks=revision['replication'][name]['writers'] if revision else picks
            for p in used_picks:
                b,h=p['block'],p['head'];label=f'B{b}/H{h}' if h is not None else f'B{b} MLP'
                index={'space':'attention','head':h} if h is not None else {'space':'mlp','field':'residual_write'}
                level='heldout' if p['replicated'] else 'observation'
                proof=refs['revision'] if revision else refs['paths']
                id=card(f'{name}-{b}-{h}',b,'attention_o_proj_head' if h is not None else 'mlp_output',index,
                        f'{label} · 问题前{title}写入','真实输出投影，两个 R 方向、两个留出独立检验',level,proof,
                        [{'heldouts':p['heldouts'],'revision':p.get('revision'),'scope':'writer correspondence and individual necessity are distinct'}],p)
                edge(f'language:document_{name}_selection',id,'prequestion_attribute_response',level,proof)
                edge(id,rids[name],'native_residual_write_projection','parameter_source',refs['native'])
                writer_ids[name,b,h]=id
        if generated:
            causal=generated.get('summary',{}).get('industry',{})
            support=causal.get('passed',False)
            cid=card('upstream-intervention',35,'prequestion_residual_write_group',{'space':'residual','field':'writer_group'},
                     '问题前生计路径联合移除 · '+('主效应达门' if support else '主效应未达门'),
                     '在原生前缀一个位置衰减整 Head/MLP 写入，下游和 KV 真前向',
                     'intervention' if support else 'negative',refs['generated'],counterexamples[:20],
                     {'native_parameters_edited':0,'min_edit_params':None,'primary':causal,'capability':capability['summary']})
            edge(cid,rids['industry'],'joint_writer_ablation' if support else 'joint_writer_ablation_gate_not_met',
                 'intervention' if support else 'negative',refs['generated'])
        if flow:
            checked=flow['summary'];ref=refs['flow_boundary']
            fid=card('last-block-temporal-boundary',35,'future_query_logits',{'space':'residual','field':'fixed_prefix_future_query'},
                     'B35/H25 问题前写入的时间边界','单独移除 B35/H25，比较局部残差、后续位置词表 logits 与 36 层 KV',
                     'negative',ref,[checked],{'scope':'fixed-prefix recomputation on exposed cases'})
            edge(writer_ids['industry',35,25],fid,'past_last_block_write_no_future_transfer','negative',ref)
    names=[p for p in C.ROOT.glob('tests/glm5/mlp_functional*.py')]
    refs['source_code']=archive.sources([p.relative_to(C.ROOT).as_posix() for p in sorted(names)])
    refs['case_index']=archive.put('case-index',{'schema':'functional_case_index.v1','collections':cases})
    gr=archive.put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges,'semantic_mechanisms_closed':0})
    conclusion=(f'行为校准 SAE 锁箱 KL {lock["functional_sae"]["KL_clean"]:.4f}，损失恢复 {lock["functional_sae"]["loss_recovered"]:.1%}。'
                f'同预算原生对照 KL {lock["native256"]["KL_clean"]:.4f}，损失恢复 {lock["native256"]["loss_recovered"]:.1%}。'
                +('两个留出的同预算门通过。' if dictionary['both_passed'] else '两个留出的同预算门未通过。'))
    d={'schema':'mlp_functional_report.v1','state':'complete' if final else 'running','conclusion':conclusion,
       'training':without(trained,'training_doc_indices','development_check_indices'),'dictionary':without(dictionary,'rows'),
       'paths':without(paths,'all_writer_effects','directions') if paths else None,'native_count':trace['backgrounds']*4 if trace else 0,
       'generated':without(generated,'rows') if generated else None,'capability':without(capability,'rows') if capability else None,
       'behavior':{'pilot':pilot['cells'],'formal':trace['cells'] if trace else None,'passed':trace['passed'] if trace else None},
       'revision':revision,'flow_boundary':flow['summary'] if flow else None,'cases':cases,'prereg':prereg,'refs':refs,
       'counterexamples':counterexamples,'behavior_errors':behavior_errors}
    version='final' if final else 'observed' if observed else 'calibration'
    rr=archive.put(f'report-{version}',d)
    title='行为保真字典与问题前属性路径'
    archive.bind(rr,gr,{'study_type':'mlp_functional_upstream','native_bank':bool(trace),'case_index':refs['case_index']},title,
                 {'dictionary':{'status':'done','gate':dictionary['both_passed']},
                  'upstream':{'status':'done' if paths else 'running','behavior_gate':revision['passed'] if revision else trace['passed'] if trace else None},
                  'causal':{'status':generated['status'] if generated else 'pending'},
                  'capability':{'status':'done' if capability else 'pending'}})
    C.save(C.OUT/f'registration-{version}.json',
           {**ident,'report_sha256':rr,'graph_sha256':gr,'refs':refs,'case_index':cases,'cards':len(cards),'edges':len(edges)},False)
    print({'report':rr,'graph':gr,'cards':len(cards),'edges':len(edges),'complete':final},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');parser.add_argument('--observed',action='store_true')
    args=parser.parse_args();main(args.final,args.observed)
