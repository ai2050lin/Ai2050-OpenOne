"""Persist every native unit, negative result and ordered protocol upload metadata."""
import io
import tarfile
import mlp_tail_goodfire as C
from deploy.research_library import Library, canonical


def main():
    man=C.read(C.OUT/'manifest.json');assert man['state']=='complete'
    lib=Library(C.ROOT/'distributed_data/research_library');objects={}
    def put(name,doc):
        body={'kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
              'provenance':C.read(C.OUT/'admissibility-audit.json')['provenance'],**doc}
        sha=lib.ingest(body,f'{C.RUN}/{name}',kind='local_report')['sha256']
        objects[sha]=lib.root/'objects'/sha[:2]/(sha+'.json')
        C.save_json(C.PUBLIC/(name+'.research.json'),body)
        return sha
    plan=put('design',{'schema':'mlp_tail_design_bundle.v1','plans':{n:C.read(C.OUT/n) for n in ['plan.json','prediction-plan.json','path-plan.json','readout-plan.json','profile-implementation.json']}})
    sources=put('sources',{'schema':'research_source_bundle.v1','files':[{'path':f'tests/glm5/{n}','sha256':C.file_sha(C.ROOT/'tests/glm5'/n),'text':(C.ROOT/'tests/glm5'/n).read_text(encoding='utf-8')} for n in ['mlp_tail_goodfire.py','mlp_tail_profiles.py','mlp_tail_predict.py','mlp_tail_path_test.py','mlp_tail_finalize.py','mlp_tail_readout.py']]})
    manifest=put('native-manifest',man)
    report=put('report',C.read(C.OUT/'report.json'))
    graph_refs=[]
    for b in C.BLOCKS:
        profile=C.read(C.OUT/f'B{b}-profile.json');cues=C.read(C.OUT/f'B{b}-cues.json')['units']
        evidence=put(f'layer-{b}',{'schema':'mlp_tail_layer_index.v1','block':b,'width':C.WIDTH,'manifest_sha256':manifest,'native_manifest_file_sha256':C.file_sha(C.OUT/'manifest.json'),
            'arrays':{f'{f}_{k}':profile['fields'][f][k] for f in ['G','V','A'] for k in ['mean','std','min','max']},
            'scope':'Full profiles, poles, all pair matrices and parameters are in the verified external dataset; semantic labels unconfirmed.'})
        for lo in range(0,C.WIDTH,512):
            cards=[{'id':f'{C.SNAPSHOT}:{C.RUN}:B{b}:U{u}','run_id':C.RUN,'layer':b,'boundary':'mlp.down_proj.input',
                'index':{'space':'mlp','unit':u},'method':'Native SwiGLU replay, long-context gate/value/product poles and complete adjacent pair matrices',
                'label':f'B{b}/U{u} · '+cues[u]['description'],'evidence':{'level':'observation','artifact_sha256':evidence},
                'counterexamples':[{'artifact_sha256':report,'reason':'Sparse cross-layer rollout failed; lexical/context cue hypotheses remain unconfirmed.'}],
                'scope':{'semantic_confirmed':False,'unit_api':f'/api/research/mlp-tail/units/{b}/{u}','relation_api':f'/api/research/mlp-tail/relations/{b}/{u}'}}
                for u in range(lo,min(C.WIDTH,lo+512))]
            graph_refs.append(put(f'graph-{b}-{lo:04}',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':[],
                'note':'Observation cards for all native units. Dense correlations and parameter cosines are not promoted to causal edges.'}))
        print(f'published B{b}: {C.WIDTH} units',flush=True)
    graph=graph_refs[0]
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':report,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'mlp_tail_atlas','native_bank':True,'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32},
            'all_graph_sha256':graph_refs,'native_manifest_sha256':manifest,'read_api':'/api/research/mlp-tail'}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B最后四层 · 38,912单元全量特性与上下游关联',
        'stages':{'collection':{'status':'done','units':38912,'documents':2048,'matrices':15},
            'native_path':{'status':'done','source_units':12,'documents':96,'observations':9216,'scope':'Finite effects, paired controls and restoration; not semantic proof.'},
            'cross_layer_prediction':{'status':'not_supported','free_rollout_generalization':False},
            'semantic_explanation':{'status':'not_complete','confirmed':0},
            'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graph,'all_graph_sha256':graph_refs}}})
    reg={'run_id':C.RUN,'model_snapshot':C.SNAPSHOT,'report_sha256':report,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work,'all_graph_sha256':graph_refs}
    out=C.ROOT/'release/ai2050-research-mlp-tail-20261108.tar.gz';out.parent.mkdir(exist_ok=True)
    with tarfile.open(out.with_suffix('.tmp'),'w:gz',compresslevel=5) as tar:
        raw=canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objects],'studies':[reg]})
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        for sha,p in objects.items():assert C.file_sha(p)==sha;tar.add(p,arcname=f'objects/{sha}.json')
    out.with_suffix('.tmp').replace(out)
    reg.update(upload_file=str(out.relative_to(C.ROOT)),upload_sha256=C.file_sha(out),objects=len(objects),native_manifest_sha256=C.file_sha(C.OUT/'manifest.json'))
    C.save_json(C.PUBLIC/'registration.json',reg)
    print({'registration':reg,'upload_bytes':out.stat().st_size},flush=True)


if __name__=='__main__':main()
