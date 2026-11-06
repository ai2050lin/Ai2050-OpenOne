"""Every native unit has a durable identity and separate evidence axes, never a solved % ."""
import sys,json,subprocess,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials as P
from shared.dist_protocol.mechanism_trace import file_sha,digest
from deploy.research_library import Library
RUN='MLP-ATLAS-INVENTORY-20261104-v1';OUT=P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1';PUBLIC=P.ROOT/'research/mlp_atlas/inventory_v1'
CENSUS=Path('D:/ai2050_native/_census/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-20000000')

def main():
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);man=P.read(CENSUS/'manifest.json');snap=man['model_snapshot']['sha256'];assert snap==P.read(P.OUT/'corpus.json')['model_snapshot']
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={P.ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),'git_dirty':True,'model_path':str(P.MODEL),'model_revision':snap,'seed':P.SEED}
    lib=Library(P.ROOT/'distributed_data/research_library');layers=[];layerrefs={};audit=[];start=time.time()
    def archive(name,d):
        body={'schema':'mlp_reverse_inventory.v1','kind':'real','run_id':RUN,'model_snapshot':snap,'provenance':prov,**d};path=PUBLIC/(name+'-'+digest(body)[:12]+'.research.json');P.save(path,body);return lib.ingest(body,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    attempts=P.read(P.ROOT/'tests/glm5/result/native_mlp_long_v1/documents.json')['units'];cardmap={}
    oldctx=P.read(P.ROOT/'research/native_mlp_long/v1/unified-evidence.research.json');fn=P.read(P.ROOT/'tests/glm5/result/native_mlp_function_v4/delivery-registration-stream.json');binding=None
    with lib.db() as db:rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    for row in rows:
        b=lib.artifact(row['sha'])
        if b['run_id']=='NMLP-FUNCTION-20261104-v4':binding=b;break
    assert binding;answer_units={(28,4371),(32,5610),(32,3952),(29,8463),(34,446),(27,2658)}
    for block in range(man['n_layers']):
        path=CENSUS/f'L{block}.npz';meta=next(f for f in man['files'] if f['path']==path.name);assert file_sha(path)==meta['sha256']
        with np.load(path) as z:
            n=int(z['n']);assert n==man['n_tokens_counted'];hist=z['hist'];assert np.all(hist.sum((1,2))==n);mean=z['s1']/n;var=np.maximum(z['s2']/n-mean**2,0);rms=np.sqrt(z['s2']/n)
            assert all(np.isfinite(z[key]).all() for key in ['s1','s2','amin','amax','top_val','bot_val'])
            stats={'n_tokens':n,'A_mean':mean.tolist(),'A_std':np.sqrt(var).tolist(),'A_RMS':rms.tolist(),'A_min':z['amin'].tolist(),'A_max':z['amax'].tolist(),'GV_sign_counts':hist.sum(1).tolist(),
                'direct_readout_top_token_ids':z['readout_top'].tolist(),'direct_readout_bottom_token_ids':z['readout_bot'].tolist()}
        width=man['d_ff'];flags={'parameter_locator':[True]*width,'natural_statistics':[True]*width,'natural_context_prediction_attempted':[False]*width,'natural_context_prediction_passed':[False]*width,'controlled_response_recorded':[True]*width,
            'heldout_answer_direction':[False]*width,'individual_causal_mechanism':[False]*width,'graph_predictive_closed':[False]*width,'semantic_monosemanticity_confirmed':[False]*width}
        for a in attempts:
            if a['layer']==block:flags['natural_context_prediction_attempted'][a['unit']]=True
        for l,u in answer_units:
            if l==block:flags['heldout_answer_direction'][u]=True
        cards={k:v for k,v in binding['context']['unit_cards'].items() if k.startswith(str(block)+':')};naturalcards={k:v for k,v in oldctx['unit_cards'].items() if k.startswith(str(block)+':')}
        d={'schema':'mlp_reverse_inventory_layer.v1','block':block,'d_ff':width,'unit_ids':[f'{snap}:B{block}:mlp.down_proj.input:U{u}' for u in range(width)],'native_index_order':True,'flags':flags,'natural_statistics':stats,
            'case_evidence_cards':cards,'natural_attempt_cards':naturalcards,'raw_census_manifest_sha256':file_sha(CENSUS/'manifest.json'),'raw_census_layer_sha256':meta['sha256'],
            'scope':'Natural statistics anddirect readout labels are not semantic interpretations. Answer-direction evidence is only the registeredYes/No scopes. Lack of an individual causal result is not proof of no contribution.'}
        layerrefs[str(block)]=archive(f'layer-B{block:02d}',d);layers.append({'block':block,'n_units':width,'counts':{k:sum(v) for k,v in flags.items()}});audit.append({'path':str(path),'sha256':meta['sha256'],'bytes':path.stat().st_size})
        print(json.dumps({'layer':block,'units':width}),flush=True)
    counts={k:sum(l['counts'][k] for l in layers) for k in layers[0]['counts']};report={'schema':'mlp_reverse_inventory_report.v1','n_layers':36,'d_ff':9728,'total_units':350208,'evidence_counts':counts,'layer_refs':layerrefs,'layers':layers,'raw_census_tokens':man['n_tokens_counted'],
        'census_manifest_sha256':file_sha(CENSUS/'manifest.json'),'all_source_layer_hashes_verified':True,'source_layer_files':audit,'seconds':time.time()-start,
        'definition_of_completion':{'coverage':'Every350208unit has exact native locator and queryable evidence record.','functional_scope':'Reproducible conditional response explanation, explicit counterexamples and tested generalization.','mechanism':'Executable computation/path andheldout activation/output predictions against lexical/position baselines, scoped intervention where claimed.','global_claim':'Neither complete coverage nor a list of labels establishes all language mechanisms or global neuron monosemanticity.'},
        'next_run_id':'MLP-ATLAS-CODE-20261104-v2','dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32,'n_kv_heads':8,'head_dim':128},'local_model_id':'9a784350de7f8d690b04ffef','limitations':['Censusmax_tokens20M is a nominal input budget; actual countedtokens afterhead exclusion is19,042,014.','466native units attemptednatural contextual prediction; none passedbothheldouts. Counts use actual unit identities.','Sixanswer-direction candidates do not imply sixfully cracked neurons; B24/U8443 has readout-dependent evidence andis retained in source cards.','Whole-array controlled profiling is observation, not whole-array confirmatory tests. Newanswer-code study remains separately gated.']}
    rr=archive('report',report);P.save(OUT/'registration-v2.json',{'report_sha256':rr,'layer_refs':layerrefs,'provenance':prov,'model_snapshot':snap,'run_id':RUN})
    archive('workpackage',{'schema':'research_workpackage.v1','title':'350208个原生MLP单元：覆盖、证据与未破解项账本','stages':{'inventory':{'status':'done','units':350208,'natural_tokens':man['n_tokens_counted']},'registration_validated':{'status':'done','report_sha256':rr},'semantic_decoding':{'status':'not_complete','mechanisms_closed':0,'note':'No totaldeciphering percentage; evidence dimensions kept separately.'},'next_research':{'status':'planned','run_id':'MLP-ATLAS-CODE-20261104-v2'}}})
    print(json.dumps({'report_sha256':rr,'counts':counts,'seconds':time.time()-start}))
if __name__=='__main__':main()
