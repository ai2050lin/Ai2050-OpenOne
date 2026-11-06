"""Archive early-token reads, writer interventions, raw cache slices and counterexamples."""
import mlp_read_route_common as C
from mlp_atlas_publish_final import Archive
from mlp_functional_publish import archive_cases, without


def main():
    out=C.OUT;archive=Archive(C.PUBLIC,C.RUN,C.identity())
    route=C.read(out/'route-freeze.json');plan=C.read(out/'prereg.json');screen=C.read_gz(out/'screen.json.gz')
    causal=C.read_gz(out/'causal-result-strict-v2.json.gz');cap=C.read_gz(out/'capability-result-strict-v2.json.gz')
    inputs=C.read_gz(out/'inputs.json.gz')['rows'];docs=C.read_gz(out/'documents.json.gz')['docs']
    byid={r['id']:r for r in inputs};refs={};cases={}
    for name,value in [('route',route),('screen',without(screen,'rows')),('causal',without(causal,'rows')),('capability',without(cap,'rows'))]:
        refs[name]=archive.put(name,value)
    refs['strict_control']=archive.put('strict-control-prereg',C.read(out/'strict-match-prereg.json'))
    def with_source(rows):
        return [{**r,'input':byid[r['id']],'source':without(docs[byid[r['id']]['b']],'tokens')} for r in rows]
    discovery=C.read_gz(out/'discovery-readings.json.gz')
    cases['discovery']=archive_cases(archive,'discovery',with_source(discovery['rows']))
    refs['discovery_derivatives']=archive.put('discovery-derivatives',{'schema':'early_writer_discovery.v1','derivatives':discovery['derivative']})
    cases['screen']=archive_cases(archive,'screen',with_source(screen['rows']))
    cases['causal']=archive_cases(archive,'causal',with_source(causal['rows']))
    cases['capability']=archive_cases(archive,'capability',[{'input':r,**v} for r,v in zip(C.read(C.CAP),cap['rows'])])
    by_role=C.read(out/'by-role-result.json') if (out/'by-role-result.json').exists() else None
    if by_role:
        refs['by_role']=archive.put('by-role-audit',without(by_role,'rows'))
        cases['by_role']=archive_cases(archive,'by_role',by_role['rows'])
    native={'n':0,'chunks':[],'chunk_size':4};raw_refs=[archive.native(out/'discovery-reads.npz',route['raw_weight_sha256'])]
    if causal['rows']:
        rows=C.read_gz(out/'holdout-native.json.gz')['rows'];meta=causal['native'];size=4
        native={'n':len(rows),'chunk_size':size,'chunks':[archive.put(f'native-{j}',{'schema':'early_read_route_raw_slots.v1',
                   'dtype':'BF16 uint16','start':j,'cases':rows[j:j+size]}) for j in range(0,len(rows),size)],'provenance':meta}
        raw_refs.extend([archive.native(out/'holdout-native.json.gz',meta['raw_slots_sha256']),archive.native(out/'holdout-weights.npz',meta['full_weight_sha256'])])
    refs['native']=archive.put('native-tensor-index',{'schema':'research_native_tensor_index.v1','files':raw_refs})
    refs['code']=archive.sources([p.relative_to(C.ROOT).as_posix() for p in sorted(C.ROOT.glob('tests/glm5/mlp_read_route_*.py'))])
    refs['case_index']=archive.put('case-index',{'schema':'mlp_early_route_case_index.v1','cases':cases,'native':native})
    cards=[];edges=[]
    def card(key,layer,boundary,index,label,method,level,ref,scope,counterexamples=()):
        id=f'{archive.snap}:{C.RUN}:{key}'
        cards.append({'id':id,'layer':layer,'boundary':boundary,'index':index,'run_id':C.RUN,'label':label,'method':method,
                      'evidence':{'level':level,'artifact_sha256':ref},'counterexamples':list(counterexamples),'scope':scope})
        return id
    def edge(source,target,relation,level,ref):
        edges.append({'id':C.digest([C.RUN,source,target,relation]),'source':source,'target':target,'relation':relation,'run_id':C.RUN,
                      'evidence':{'level':level,'artifact_sha256':ref}})
    n=len(causal['rows']);checks=causal['checks'];summary=causal['summary']['all']
    verified=bool(n and checks.get('replay_logits_bitwise_equal')==n and checks.get('rescue_logits_bitwise_equal')==n)
    group=card('writer-group',34,'source_token_joint_write',{'space':'residual','field':'source_writer_group'},'属性词与选择键的联合写入',
               'rank on discovery derivatives; intervene on native Head and MLP output at frozen token roles',
               'intervention' if verified else 'observation',refs['causal'],{'roles':route['roles'],'heads':route['heads'],'mlps':route['mlps']})
    for b,h in route['heads']:
        id=card(f'head-{b}-{h}',b,'attention_o_proj_input',{'space':'attention','head':h},f'B{b}/H{h} 原生写入',
                'selected on discovery; actual pre-o_proj edit as member of joint group','parameter_source',refs['causal'],{'roles':route['roles']})
        edge(id,group,'joint_writer_member','parameter_source',refs['causal'])
    for b in route['mlps']:
        id=card(f'mlp-{b}',b,'mlp_output',{'space':'mlp','field':'residual_write'},f'B{b} MLP 原生写入',
                'selected on discovery; actual MLP output edit as member of joint group','parameter_source',refs['causal'],{'roles':route['roles']})
        edge(id,group,'joint_writer_member','parameter_source',refs['causal'])
    readouts=[]
    for b,data in causal.get('read_summary',{}).items():
        confirmed=data['Q_unchanged']==n and data['nonzero_read']==n
        id=card('reader-'+b,int(b),'answer_query_attention_output',{'space':'attention','field':'source_values'},f'B{b} 回答查询的源槽读取',
                'post-RoPE Q/K and native V reconstruction; actual V zero with unchanged query and K',
                'intervention' if confirmed else 'observation',refs['causal'],data)
        readouts.append((id,confirmed))
    cache=card('changed-cache',35,'prefill_cache_changed_ranges',{'space':'attention','field':'kv_changed_ranges'},'原生 B34/B35 缓存改动范围',
               'compare every native cache slot and replay all changed slices, source-only and propagated-slot-only',
               'parameter_source' if verified else 'observation',refs['causal'],checks)
    edge(group,cache,'native_writer_to_changed_cache','parameter_source' if verified else 'observation',refs['causal'])
    for id,confirmed in readouts:edge(cache,id,'source_value_read_validation','intervention' if confirmed else 'observation',refs['causal'])
    readout=card('readout',35,'assistant_logits',{'space':'residual','field':'assistant_distribution'},'后续词表分布的逐位回放与恢复',
                 'counterfactual replay on clean prefix; restoration on edited prefix',
                 'intervention' if verified else 'observation',refs['causal'],checks)
    edge(cache,readout,'full_cache_replay_and_rescue','intervention' if verified else 'observation',refs['causal'])
    counters=[{'index':r['index'],'id':r['id'],'expected':r['expected'],'after':r['conditions']['joint_1']['word'],
               'KL':r['conditions']['joint_1']['KL'],'changed':r['conditions']['joint_1']['word']!=r['conditions']['clean']['word']}
              for r in causal['rows']]
    ranked=sorted(counters,key=lambda r:(not r['changed'],-r['KL'],r['index']))[:24]
    level='intervention' if summary['passed'] else 'negative' if summary['eligible'] else 'observation'
    outcome=card('word-outcome',35,'generated_first_word',{'space':'residual','field':'greedy_first_word'},'未见低余量组合的生成首词效应',
                 'paired natural first word, fixed norm-matched writer control, preregistered effect gate',level,refs['causal'],summary,ranked)
    edge(readout,outcome,'generated_word_effect',level,refs['causal'])
    graph=archive.put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    report={'schema':'mlp_early_read_route_report.v1','state':'complete','prereg':plan,'route':route,'screen':without(screen,'rows'),
            'causal':without(causal,'rows'),'capability':without(cap,'rows'),'by_role':without(by_role,'rows') if by_role else None,
            'native':native,'cases':cases,'refs':refs,'counterexamples':ranked,
            'previous_report_sha256':C.read(C.PREVIOUS/'registration-final.json')['report_sha256']}
    rr=archive.put('report',report)
    archive.bind(rr,graph,{'study_type':'mlp_early_read_route','native_bank':bool(n),'case_index':refs['case_index']},
                 '回答读取的早期源 token 与联合写入路径',
                 {'discovery':{'status':'done','n':route['discovery_n']},'holdout':{'status':'done','n':n},
                  'replay':{'status':'done','bitwise_verified':verified},'generation':{'status':'done','passed':summary['passed']},
                  'capability':{'status':'done'}})
    C.save(out/'registration-final.json',{**C.identity(),'report_sha256':rr,'graph_sha256':graph,'refs':refs,'cases':cases,
                                         'native':native,'cards':len(cards),'edges':len(edges)},False)
    print('registered',rr,graph,len(cards),len(edges),flush=True)


if __name__=='__main__':main()
