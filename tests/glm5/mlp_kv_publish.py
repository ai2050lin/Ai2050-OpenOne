"""Append the native KV mediation results and their case-level evidence to the research graph."""
import argparse

import numpy as np

import mlp_kv_common as C
from mlp_atlas_publish_final import Archive
from mlp_functional_publish import archive_cases, without


def native_chunks(archive,name,result,folder):
    meta=result['native'];path=folder/meta['path'];assert C.file_sha(path)==meta['sha256']
    chunks=[];size=8
    with np.load(path) as state:
        arrays={k:state[k] for k in meta['fields']}
        assert all(a.dtype==np.uint16 and list(a.shape)==meta['fields'][k] for k,a in arrays.items())
        for start in range(0,len(result['rows']),size):
            rows=[{'id':result['rows'][i]['id'],'index':i,'fields':{k:v[i].tolist() for k,v in arrays.items()}}
                  for i in range(start,min(start+size,len(result['rows'])))]
            chunks.append(archive.put(f'{name}-native-{start}',{'schema':'kv_slot_native_cases.v1','dtype':meta['dtype'],'cases':rows,'start':start}))
    return {'n':len(result['rows']),'chunk_size':size,'chunks':chunks,'provenance':meta}


def main(final):
    archive=Archive(C.PUBLIC,C.RUN,C.identity());refs={};cases={};native={};cards=[];edges=[]
    plan=C.read(C.OUT/'prereg.json');pilot=C.read(C.OUT/'pilot-behavior.json');formal=C.read(C.OUT/'formal-behavior.json')
    med=C.read(C.OUT/'mediation-result.json');earlier=C.read(C.OUT/'earlier-result.json');cap=C.read(C.OUT/'capability-result.json')
    corpus=C.read(C.OUT/'corpus.json')['rows'];items={r['id']:r for r in corpus};documents=C.read(C.OUT/'documents.json')['docs']
    def with_input(rows):
        return [{**r,'input':items[r['id']],'source':without(documents[r['b']],'tokens')} for r in rows]
    for key,doc,include in [('pilot',pilot,True),('formal',formal,True),('mediation',med,False),('earlier',earlier,False)]:
        refs[key]=archive.put(key,without(doc,'rows'))
        cases[key]=archive_cases(archive,key,with_input(doc['rows']) if include else doc['rows'])
    refs['capability']=archive.put('capability',without(cap,'rows'))
    cases['capability']=archive_cases(archive,'capability',[{'input':x,**r} for x,r in zip(C.read(C.CAP),cap['rows'])])
    native['mediation']=native_chunks(archive,'mediation',med,C.OUT)
    refs['raw_slots']=archive.put('raw-slot-tensors',{'schema':'research_native_tensor_index.v1',
                      'files':[archive.native(C.OUT/med['native']['path'],med['native']['sha256'])]})
    targeted=screen=target_plan=None
    if final:
        folder=C.OUT/'targeted';target_plan=C.read(folder/'prereg.json');targeted=C.read(folder/'result-layout-v2.json');screen=C.read(folder/'screen.json')
        sources=C.read(folder/'inputs.json')
        refs['targeted']=archive.put('targeted',without(targeted,'rows'))
        refs['targeted_initial_layout']=archive.put('targeted-initial-layout',C.read(folder/'result.json'))
        refs['screen']=archive.put('targeted-enrollment',without(screen,'rows'))
        cases['screen']=archive_cases(archive,'screen',[{**r,'input':x,'source':without(d,'tokens')} for r,x,d in zip(screen['rows'],sources['rows'],sources['docs'])])
        cases['targeted']=archive_cases(archive,'targeted',targeted['rows'])
        if targeted.get('native'):
            native['targeted']=native_chunks(archive,'targeted',targeted,folder)
            refs['targeted_raw_slots']=archive.put('targeted-raw-slots',{'schema':'research_native_tensor_index.v1',
                    'files':[archive.native(folder/targeted['native']['path'],targeted['native']['sha256'])]})

    def card(key,layer,boundary,index,label,method,level,ref,scope,counterexamples=()):
        id=f'{archive.snap}:{C.RUN}:{key}'
        cards.append({'id':id,'layer':layer,'boundary':boundary,'index':index,'run_id':C.RUN,'label':label,'method':method,
                      'evidence':{'level':level,'artifact_sha256':ref},'counterexamples':list(counterexamples),'scope':scope})
        return id

    def edge(source,target,relation,level,ref):
        edges.append({'id':C.digest([C.RUN,source,target,relation]),'source':source,'target':target,'relation':relation,
                      'run_id':C.RUN,'evidence':{'level':level,'artifact_sha256':ref}})

    verified=(med['checks']['only_B35_and_source'] and med['checks']['replay_logits_bitwise_equal']==len(med['rows'])
              and med['checks']['rescue_logits_bitwise_equal']==len(med['rows']))
    source=card('B34-H2',34,'attention_o_proj_head',{'space':'attention','head':2},'B34/H2 · 问题前写入',
                'native head z attenuation before o_proj at token #991','parameter_source',refs['mediation'],{'position':991})
    residual=card('B35-input',35,'input_layernorm_input',{'space':'residual','dimensions':2560},'B35 注意力前残差',
                  'BF16 residual read before B35 attention norm','parameter_source',refs['raw_slots'],{'position':991})
    k=card('B35-K',35,'cache_key_post_rope',{'space':'attention','field':'K','kv_groups':8},'B35 K · 问题前缓存槽',
           'actual post-RoPE K at source slot; eight native GQA groups','parameter_source',refs['raw_slots'],{'position':991})
    v=card('B35-V',35,'cache_value',{'space':'attention','field':'V','kv_groups':8},'B35 V · 问题前缓存槽',
           'actual V at source slot; eight native GQA groups','parameter_source',refs['raw_slots'],{'position':991})
    answer=card('later-query',35,'assistant_logits',{'space':'residual','field':'later_query_logits'},'后续回答分布 · 回放和恢复',
                'replay edited slot into clean cache and restore clean slot into edited cache','intervention' if verified else 'observation',
                refs['mediation'],med['checks'])
    edge(source,residual,'native_residual_write','parameter_source',refs['raw_slots'])
    for node,kind in [(k,'key'),(v,'value')]:
        edge(residual,node,'native_cache_'+kind,'parameter_source',refs['raw_slots'])
        edge(node,answer,'slot_replay_'+kind,'intervention' if verified else 'observation',refs['mediation'])
    m=card('B33-MLP',33,'mlp_output',{'space':'mlp','field':'residual_write'},'B33 MLP · 早一块的缓存路径',
           'whole MLP write removal and two-layer cache replay','intervention',refs['earlier'],earlier['checks'])
    b34=card('B34-KV',34,'source_cache_slots',{'space':'attention','field':'KV','kv_groups':8},'B34 原生 K/V · 源 token',
             'replayed B34 and B35 independently and jointly','parameter_source',refs['earlier'],{'cases':32})
    for target in [b34,k,v]:edge(m,target,'earlier_mlp_cache_path','parameter_source',refs['earlier'])
    edge(b34,answer,'earlier_two_layer_cache_replay','intervention',refs['earlier'])
    last=card('B35-H25',35,'attention_o_proj_head',{'space':'attention','head':25},'B35/H25 · 末层历史写入边界',
              'same-position Head attenuation after its own KV creation','negative',refs['earlier'],{'cases':8},earlier['negative_controls'])
    edge(last,answer,'no_future_transfer_fixed_prefix','negative',refs['earlier'])
    main_result=(targeted or med)['summary'].get('all',{})
    behavioral=bool(main_result.get('passed'));eligible=bool(main_result.get('eligible'))
    effect_ref=refs['targeted'] if targeted else refs['mediation']
    outcome=card('generation-gate',35,'generated_first_word',{'space':'residual','field':'natural_generation_gate'},
                 '生成干预 · '+('通过效应门' if behavioral else '未达到效应门' if eligible else '确认资格未达'),
                 'paired natural first-word accuracy and matched-norm cache control',
                 'intervention' if behavioral else 'negative' if eligible else 'observation',effect_ref,main_result)
    edge(answer,outcome,'generated_behavior_scope','intervention' if behavioral else 'negative' if eligible else 'observation',effect_ref)
    refs['source_code']=archive.sources([p.relative_to(C.ROOT).as_posix() for p in sorted(C.ROOT.glob('tests/glm5/mlp_kv_*.py'))])
    refs['case_index']=archive.put('case-index',{'schema':'mlp_kv_case_index.v1','cases':cases,'native':native})
    prior=C.read(C.PREVIOUS/'registration-final.json')
    graph=archive.put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    report={'schema':'mlp_kv_mediation_report.v1','state':'complete' if final else 'running','prereg':plan,'targeted_prereg':target_plan,
            'previous_report_sha256':prior['report_sha256'],'pilot':without(pilot,'rows'),'formal':without(formal,'rows'),
            'mediation':without(med,'rows'),'earlier':without(earlier,'rows'),'capability':without(cap,'rows'),
            'targeted':without(targeted,'rows') if targeted else None,'screen':without(screen,'rows') if screen else None,
            'cases':cases,'native':native,'refs':refs}
    rr=archive.put('report-final' if final else 'report-initial',report)
    archive.bind(rr,graph,{'study_type':'mlp_kv_mediation','native_bank':True,'case_index':refs['case_index']},'问题前写入怎样经 KV 到达后续回答',
                 {'behavior':{'status':'done','passed':formal['passed']},'kv_replay':{'status':'done','bitwise_verified':verified},
                  'low_margin':{'status':'done' if final else 'running','eligible':eligible},'capability':{'status':'done'}})
    C.save(C.OUT/('registration-final.json' if final else 'registration-initial.json'),
           {**C.identity(),'report_sha256':rr,'graph_sha256':graph,'refs':refs,'cases':cases,'native':native,'cards':len(cards),'edges':len(edges)},False)
    print('registered',rr,graph,len(cards),len(edges),'complete',final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
