"""Unlabelled natural MLP input/output bank for dictionaries at every layer."""
import sys,time,json,sqlite3,os,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as BASE
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot
from deploy.research_library import Library
from tools.power_analysis import check_phase
ROOT=BASE.ROOT;MODEL=BASE.MODEL;OUT=ROOT/'tests/glm5/result/mlp_atlas_dictionary_v1';PUBLIC=ROOT/'research/mlp_atlas/dictionary_v1';RUN='MLP-ATLAS-DICTIONARY-20261104-v1';SEED=BASE.SEED+700
read=BASE.read;save=BASE.save

def status(stage,state='done',**kw):
    p=PUBLIC/'00-progress.research.json';d=read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':RUN,'title':'全36层自然MLP字典：无标注提取与留出重建','stages':{}}
    d['stages'][stage]={'status':state,'time':time.time(),**kw};save(p,d,False);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report');print(json.dumps({'stage':stage,'status':state,**kw}),flush=True)

def main():
    import torch
    from transformers import AutoModelForCausalLM
    assert read(BASE.OUT/'bank/manifest.json')['state']=='complete','Sequential GPU rule: finish controlledbank first'
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);data=read(BASE.OLD/'documents.json');tokens=np.load(BASE.OLD/'input-tokens.npy',mmap_mode='r');assert tokens.shape[0]==2048
    prov={**read(BASE.OUT/'pilot-result.json')['provenance'],'seed':SEED};rng=np.random.default_rng(SEED);positions=np.sort(rng.choice(np.arange(512,1024),16,replace=False));shapes={f'B{b:02d}-{kind}.npy':(2048,16,2560) for b in range(36) for kind in ['input','output']}
    nbytes=sum(np.prod(s)*2+256 for s in shapes.values());assert nbytes<12*2**30 and shutil.disk_usage(OUT).free>28*2**30
    contract={'schema':'natural_mlp_dictionary_plan.v1','kind':'real','run_id':RUN,'model_snapshot':prov['model_revision'],'provenance':prov,'positions':positions.tolist(),'n_docs':2048,'tokens_per_doc':1024,'n_positions_per_doc':16,'native_fields':['MLP.input','MLP.output'],
        'all_layers':36,'native_dtype':'BF16_bits','estimated_raw_bytes':int(nbytes),'budget_GiB':15,'source_documents_sha256':file_sha(BASE.OLD/'documents.json'),'source_tokens_sha256':file_sha(BASE.OLD/'input-tokens.npy'),'source_sha256':file_sha(__file__),
        'split':'prior1024FineWebdiscovery docs forunsupervisedfit; prior512confirm and512lockbox Wikipedia docs fordictionaryholdout. Thosebackgrounds appeared inearlierresearch, notclaimedgloballyunseen.',
        'algorithm_plan':{'kind':'TopKtranscoder','m':1024,'k':32,'steps':1000,'batch':256,'lr':.0002,'dead_after_tokens':16384,'aux_k':128,'aux_coef':1/32,'fit':'discoveryonly, no language labels','baseline':'same32native unit contributions, andconstantdiscoverymean','claims':'FVU reconstruction alone is notfeaturesemanticinterpretation, necessity orlanguage-decoding completion'},
        'native_dictionary_mapping':'retain all9728×1024discovery correlations perblock, nativeindexorder. Eachunit nominates maxabsdiscoveryfeature, thenrecordsamepartner correlationonconfirm/lockbox. This isconditionalassociation, notsemanticexplanation; discoveryranking isnotabackbonedefinition.',
        'power':{'endpoint':'pairedper-document squared-error difference normalizedbyheldoutvariance; planningvarianceproxy.25, notaposteriori guarantee','expected_FVU_advantage':.1,'n_docs_each_heldout':512,'alpha_per_layer':.05/36,'calculation':check_phase(.1,512,p_d=.25,alpha=.05/36)},
        'capability':'Original modelweights notedited. Splice orsteering would require itsown heldout/capability gate; notclaimed byobservertraining.',
        'resource_scope':'finite once-throughcollect; sequentialGPU; stopifbudget exceeded; per-layerdictionaries trainsequentially aftermodelunload'}
    assert contract['power']['calculation']['pass'];save(PUBLIC/'prereg.research.json',contract);save(OUT/'source-documents.json',{'kind':'real','run_id':RUN,'provenance':prov,'docs':data['docs'],'positions':positions.tolist()})
    folder=OUT/'bank';folder.mkdir(exist_ok=True);maps={name:np.lib.format.open_memmap(folder/name,mode='w+',dtype=np.uint16,shape=shape) for name,shape in shapes.items()};ctx={};handles=[];checks=[];canaries={};started=time.time()
    with GpuLease():
        assert snapshot(MODEL)==prov['model_revision'];torch.set_num_threads(4);m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        def bits(x):return x.detach().contiguous().view(torch.uint16).cpu().numpy()
        for b,l in enumerate(m.model.layers):
            def pre(_m,args,b=b):maps[f'B{b:02d}-input.npy'][ctx['indices']]=bits(args[0][:,positions])
            def post(_m,args,out,b=b):maps[f'B{b:02d}-output.npy'][ctx['indices']]=bits(out[:,positions])
            handles.extend([l.mlp.register_forward_pre_hook(pre),l.mlp.register_forward_hook(post)])
        status('natural_collection','running',done=0,total=2048,raw_GiB=nbytes/2**30)
        for i in range(0,2048,2):
            ctx['indices']=list(range(i,i+2));ids=torch.tensor(np.asarray(tokens[i:i+2,:1024]),device='cuda')
            with torch.inference_mode():o=m(input_ids=ids,use_cache=False,logits_to_keep=1)
            if i in [0,1024,1536]:
                # Forward observers merely copy tensors; verify without changing execution shape.
                baseline=o.logits.detach().clone()
                with torch.inference_mode():again=m(input_ids=ids,use_cache=False,logits_to_keep=1)
                canaries[i]=baseline.clone();checks.append({'doc_index':i,'repeat_logits_bitwise_equal':torch.equal(baseline,again.logits)});assert checks[-1]['repeat_logits_bitwise_equal'];del again
            del o
            if (i+2)%32==0:
                for a in maps.values():a.flush()
                status('natural_collection','running',done=i+2,total=2048,seconds=round(time.time()-started,1))
        for h in handles:h.remove()
        for rec in checks:
            i=rec['doc_index']
            with torch.inference_mode():clean=m(input_ids=torch.tensor(np.asarray(tokens[i:i+2,:1024]),device='cuda'),use_cache=False,logits_to_keep=1)
            rec['observer_vs_clean_bitwise_equal']=torch.equal(canaries[i],clean.logits);assert rec['observer_vs_clean_bitwise_equal'];del clean
        assert snapshot(MODEL)==prov['model_revision']
    for a in maps.values():a.flush()
    manifest={'schema':'natural_mlp_dictionary_bank.v1','kind':'real','run_id':RUN,'model_snapshot':prov['model_revision'],'provenance':prov,'state':'complete','n_docs':2048,'positions':positions.tolist(),'layers':36,'d_model':2560,'checks':checks,'seconds':time.time()-started,
        'files':[{'path':name,'shape':list(shapes[name]),'bytes':(folder/name).stat().st_size,'sha256':file_sha(folder/name)} for name in sorted(shapes)]}
    save(folder/'manifest.json',manifest);status('natural_collection','done',docs=2048,positions_per_layer=32768,raw_bytes=sum(f['bytes'] for f in manifest['files']),seconds=manifest['seconds'])
if __name__=='__main__':main()
