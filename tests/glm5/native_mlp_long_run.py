"""Real single-GPU natural long-context collection; selected raw arrays lossless."""
import json,sys,time,subprocess,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from shared.dist_protocol.mechanism_trace import GpuLease,digest,file_sha
from shared.dist_protocol.natural_register_runner import snapshot
from deploy.research_library import Library
def status(stage,state='done',**kw):
    path=P.PUBLIC/'00-progress.research.json';d=P.read(path) if path.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':P.RUN,'title':'原生MLP长语料 · 全层普查与432随机单元留出说明书','stages':{}}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time();P.save(path,d,False)
    Library(P.ROOT/'distributed_data/research_library').ingest(d,str(path.relative_to(P.ROOT)),kind='local_report');print(json.dumps({'stage':stage,'state':state,**kw},ensure_ascii=False),flush=True)

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    data=P.read(P.OUT/'documents.json');pr=P.read(P.PUBLIC/'prereg.research.json');docs=data['docs'];units=data['units'];raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r')
    files=['tests/glm5/native_mlp_long_materials.py','tests/glm5/native_mlp_long_features.py','tests/glm5/native_mlp_long_run.py','tests/glm5/native_mlp_long_analyze.py']
    freeze={'run_id':P.RUN,'prereg_sha256':digest(pr),'dependencies':{n:file_sha(P.ROOT/n) for n in files},'documents_sha256':file_sha(P.OUT/'documents.json'),'model_snapshot':pr['model_snapshot']}
    P.save(P.OUT/'execution-freeze.json',freeze)
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={P.ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),'git_dirty':True,'model_path':str(P.MODEL),'model_revision':pr['model_snapshot'],'seed':P.SEED}
    with GpuLease():
        assert torch.cuda.is_available() and snapshot(P.MODEL)==pr['model_snapshot'];assert shutil.disk_usage(P.OUT).free>50*2**30
        torch.set_num_threads(4);torch.manual_seed(P.SEED);started=time.time()
        model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        specs={l:[r['unit'] for r in units if r['layer']==l] for l in range(36)}
        mm={l:{key:np.lib.format.open_memmap(P.OUT/f'L{l:02d}-{key}.npy',mode='w+',dtype=np.uint16,shape=(2048,1024,len(specs[l]))) for key in ['G','V','A']} for l in specs}
        hidden=np.lib.format.open_memmap(P.OUT/'hidden_last.npy',mode='w+',dtype=np.uint16,shape=(2048,37,2560))
        sums=torch.zeros((3,36,3,9728),dtype=torch.float64,device='cuda');squares=torch.zeros_like(sums)
        mins=torch.full_like(sums,float('inf'));maxs=torch.full_like(sums,float('-inf'));regimes=torch.zeros((3,36,9728,4),dtype=torch.int64,device='cuda')
        hsum=torch.zeros((3,36,2560),dtype=torch.float64,device='cuda');hnorm=torch.zeros((3,36),dtype=torch.float64,device='cuda')
        count=np.zeros(3,np.int64);ctx={};pools={};handles=[];K=64
        for l in range(36):
            n=len(specs[l])
            for fold in range(3):
                for pole in ['Gpos','Gneg','Apos','Aneg']:
                    pools[(fold,l,pole)]={'value':np.full((K,n),-np.inf,np.float32),'doc':np.full((K,n),-1,np.int32),'position':np.zeros((K,n),np.int16),
                        'h':np.zeros((K,n,2560),np.uint16)}
        def bits(x):return x.detach().contiguous().view(torch.uint16).cpu().numpy()
        def pool_update(fold,l,pole,values,positions,h):
            pool=pools[(fold,l,pole)];sign=-1 if pole.endswith('neg') else 1
            values=(values*sign).float().cpu().numpy();positions=positions.cpu().numpy();hs=bits(h)
            for b in range(len(ctx['indices'])):
                for j in range(values.shape[1]):
                    k=int(np.argmin(pool['value'][:,j]))
                    if values[b,j]>pool['value'][k,j]:
                        pool['value'][k,j]=values[b,j];pool['doc'][k,j]=ctx['indices'][b];pool['position'][k,j]=positions[b,j];pool['h'][k,j]=hs[b,j]
        for l,layer in enumerate(model.model.layers):
            def pre(_m,args,l=l):hidden[ctx['indices'],l]=bits(args[0][:,-1])
            def prenorm(_m,args,l=l):ctx[(l,'h')]=args[0].detach()
            def gate(_m,args,out,l=l):ctx[(l,'G')]=out.detach()
            def value(_m,args,out,l=l):ctx[(l,'V')]=out.detach()
            def act(_m,args,l=l):
                a=args[0].detach();g=ctx.pop((l,'G'));v=ctx.pop((l,'V'));h=ctx.pop((l,'h'));fold=ctx['fold'];ix=ctx['indices'];sel=specs[l]
                for key,x in [('G',g),('V',v),('A',a)]:
                    mm[l][key][ix]=bits(x[:,:,sel]);z=x[:,32:].float().flatten(0,1)
                    sums[fold,l,['G','V','A'].index(key)]+=z.sum(0,dtype=torch.float64)
                    squares[fold,l,['G','V','A'].index(key)]+=(z*z).sum(0,dtype=torch.float64)
                    mins[fold,l,['G','V','A'].index(key)]=torch.minimum(mins[fold,l,['G','V','A'].index(key)],z.amin(0).double())
                    maxs[fold,l,['G','V','A'].index(key)]=torch.maximum(maxs[fold,l,['G','V','A'].index(key)],z.amax(0).double())
                gg=g[:,32:]>0;vv=v[:,32:]>0
                for k,mask in enumerate([gg&vv,(~gg)&(~vv),gg&(~vv),(~gg)&vv]):regimes[fold,l,:,k]+=mask.sum((0,1))
                hh=h[:,32:].float();hsum[fold,l]+=hh.sum((0,1),dtype=torch.float64);hnorm[fold,l]+=hh.norm(dim=-1).sum(dtype=torch.float64)
                for key,x in [('G',g),('A',a)]:
                    picked=x[:,32:,sel].float()
                    for pole,sign in [('pos',1),('neg',-1)]:
                        values,positions=(picked*sign).max(1);positions+=32
                        hp=h[torch.arange(len(ix),device='cuda')[:,None],positions]
                        pool_update(fold,l,key+pole,values*sign,positions,hp)
            handles += [layer.register_forward_pre_hook(pre),layer.post_attention_layernorm.register_forward_pre_hook(prenorm),
                layer.mlp.gate_proj.register_forward_hook(gate),layer.mlp.up_proj.register_forward_hook(value),layer.mlp.down_proj.register_forward_pre_hook(act)]
        def final(_m,args,out):hidden[ctx['indices'],36]=bits(out[:,-1])
        handles.append(model.model.norm.register_forward_hook(final));results=[]
        status('collection','running',done=0,total=2048,random_units=432,all_unit_statistics=350208)
        for start in range(0,len(docs),2):
            ix=list(range(start,min(start+2,len(docs))));fold={'discovery':0,'confirm':1,'lockbox':2}[docs[start]['split']]
            assert all(docs[i]['split']==docs[start]['split'] for i in ix);ctx.update(indices=ix,fold=fold)
            with torch.inference_mode():out=model(input_ids=torch.tensor(np.asarray(raw[ix,:1024]),device='cuda'),use_cache=False,logits_to_keep=1)
            probs=out.logits[:,-1].float().log_softmax(-1)
            for j,i in enumerate(ix):results.append({'doc_index':i,'actual_next_token_id':int(raw[i,1024]),'predicted_next_token_id':int(probs[j].argmax()),'nll':-float(probs[j,int(raw[i,1024])])})
            count[fold]+=len(ix)*992;del out,probs
            if (start+2)%32==0:status('collection','running',done=start+2,total=2048,seconds=round(time.time()-started,1))
        for h in handles:h.remove()
        for l,bykey in mm.items():
            for mmap in bykey.values():mmap.flush()
        hidden.flush()
        np.savez(P.OUT/'population-statistics.npz',count=count,sums=sums.cpu().numpy(),squares=squares.cpu().numpy(),mins=mins.cpu().numpy(),maxs=maxs.cpu().numpy(),regimes=regimes.cpu().numpy(),hsum=hsum.cpu().numpy(),hnorm=hnorm.cpu().numpy())
        for (fold,l,pole),pool in pools.items():
            pool['value']*=(-1 if pole.endswith('neg') else 1);np.savez(P.OUT/f'pool-{fold}-L{l:02d}-{pole}.npz',**pool)
        write_norm=np.stack([layer.mlp.down_proj.weight.float().norm(dim=0).cpu().numpy() for layer in model.model.layers]);np.save(P.OUT/'write-norm.npy',write_norm)
        for l,layer in enumerate(model.model.layers):
            sel=specs[l];np.savez(P.OUT/f'parameters-L{l:02d}.npz',gate=bits(layer.mlp.gate_proj.weight[sel]),up=bits(layer.mlp.up_proj.weight[sel]),
                down=bits(layer.mlp.down_proj.weight[:,sel].T),gamma=bits(layer.post_attention_layernorm.weight))
        assert snapshot(P.MODEL)==pr['model_snapshot']
        manifest={'kind':'real','run_id':P.RUN,'model_snapshot':pr['model_snapshot'],'docs':2048,'tokens':2097152,'counted_per_split':count.tolist(),
            'units':units,'selected_raw_arrays':[{'path':p.name,'sha256':file_sha(p),'bytes':p.stat().st_size} for p in sorted(P.OUT.glob('L??-?.npy'))],
            'other_files':[{'path':p.name,'sha256':file_sha(p),'bytes':p.stat().st_size} for p in [P.OUT/'population-statistics.npz',P.OUT/'hidden_last.npy',P.OUT/'write-norm.npy']],
            'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'snapshot_unchanged':True,'provenance':prov}
        P.save(P.OUT/'collection.json',manifest);P.save(P.OUT/'next-token-observations.json',{'rows':results,'provenance':prov});status('collection',docs=2048,counted_tokens=int(count.sum()),seconds=manifest['seconds'])
if __name__=='__main__':main()
