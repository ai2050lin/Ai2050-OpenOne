"""Calibrate B35 SAE against the real vocabulary distribution and teacher-forced next token."""
import argparse
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

import mlp_functional_common as C
from mlp_fidelity_v2_tail_v2 import tail_logits
from mlp_fidelity_v2_tail import cluster_interval, metric
from server.mechanism_trace_service import parameter
from shared.dist_protocol.mechanism_trace import GpuLease, unbits
from shared.dist_protocol.natural_register_runner import snapshot
from shared.dist_protocol.transcoder import TopKTranscoder, neuron_topk_recon

STEPS, BATCH = 640, 32
NAMES = ['clean','zero','archived_sae','mse_control','functional_sae','native256']


def archived():
    state=C.read(C.PREVIOUS/'trained/B35-result.json')
    path=C.ROOT/state['state_path']
    assert C.file_sha(path)==state['state_sha256']
    with np.load(path) as z:
        return TopKTranscoder.from_state(z,device='cuda')


def fit():
    result=C.OUT/'dictionary-training.json'
    if result.exists():return
    C.OUT.mkdir(parents=True,exist_ok=True)
    source=C.PREVIOUS/'fresh-documents.json'
    bank=C.PREVIOUS/'tail-natural'
    docs=C.read(source)['natural']
    manifest=C.read(bank/'result.json')
    positions=C.read(C.PREVIOUS/'natural-bank/manifest.json')['positions']
    source_arrays={k:bank/f'{k}.npy' for k in ['residual','output','final-hidden']}
    for k,p in source_arrays.items():assert C.file_sha(p)==next(r['sha256'] for r in manifest['native_files'] if r['path']==p.name)
    groups=[sorted([i for i,r in enumerate(docs) if r['language']==lang],key=lambda i:C.digest([C.SEED,docs[i]['id']])) for lang in ['en','zh']]
    train=np.asarray([i for g in groups for i in g[:192]])
    valid=np.asarray([i for g in groups for i in g[192:]])
    contract={**C.identity(),'schema':'functional_dictionary_train_prereg.v1','steps':STEPS,'batch':BATCH,
              'heldouts_used_for_fit':False,'previous_source_sha256':C.file_sha(source),'previous_tail_result_sha256':C.file_sha(bank/'result.json'),
              'training_doc_indices':train.tolist(),'development_check_indices':valid.tolist(),
              'positions':positions,'objectives':{'mse_control':'normalized output MSE','functional_sae':'KL(p_clean||p_replacement) + 0.15 true-next-token CE + 0.10 normalized MSE'},
              'same_seed_init_order_budget':True,'selection':'fixed last step, no selection by development or fresh holdout metrics',
              'capacity':'4096 features, 256 active, archived B35 SAE initialization','lr':'3e-5 to 3e-6 cosine Adam',
              'train_numerics':'BF16 compact vocabulary GEMM over every vocabulary token. Clean final hidden states come from audited previous native tail. Evaluation uses original sequence layout and a true hook canary.',
              'source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'dictionary-training-prereg.json',contract)
    started=time.time();torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(C.MODEL)==contract['model_snapshot']
        config=C.read(C.MODEL/'config.json')
        head_name='model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
        head=torch.from_numpy(parameter(C.MODEL,head_name,(slice(None),slice(None)))).to('cuda',torch.bfloat16)
        scale=torch.from_numpy(parameter(C.MODEL,'model.norm.weight',slice(None))).to('cuda',torch.bfloat16)
        eps=config['rms_norm_eps']
        r,y,h=[torch.from_numpy(unbits(np.load(source_arrays[k])).reshape(-1,2560)).to('cuda') for k in ['residual','output','final-hidden']]
        targets=torch.tensor([d['tokens'][p+1] for d in docs for p in positions],device='cuda')
        ti=(train[:,None]*len(positions)+np.arange(len(positions))).ravel()
        vi=(valid[:,None]*len(positions)+np.arange(len(positions))).ravel()
        variance=((y[ti]-y[ti].mean(0)).square().sum(-1)).mean().item()

        def replacement_logprobs(residual, output):
            z=(residual.to(torch.bfloat16)+output.to(torch.bfloat16)).float()
            hhat=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)).to(torch.bfloat16)*scale
            return F.linear(hhat,head).float().log_softmax(-1)

        def losses(t,index):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                reconstructed=t(y[index])
            lp=replacement_logprobs(r[index],reconstructed)
            with torch.no_grad():
                clean=F.linear(h[index].to(torch.bfloat16),head).float().log_softmax(-1)
            kl=(clean.exp()*(clean-lp)).sum(-1).mean()
            ce=F.nll_loss(lp,targets[index])
            mse=(reconstructed.float()-y[index]).square().sum(-1).mean()/variance
            return kl,ce,mse

        outputs={}
        for name in ['mse_control','functional_sae']:
            dictionary=archived()
            for p in dictionary.params():p.requires_grad_(True)
            opt=torch.optim.Adam(dictionary.params(),lr=3e-5,fused=True)
            rng=np.random.default_rng(C.SEED)
            curve=[]
            for step in range(1,STEPS+1):
                ids=ti[rng.integers(len(ti),size=BATCH)]
                opt.param_groups[0]['lr']=3e-6+1.35e-5*(1+math.cos(math.pi*(step-1)/(STEPS-1)))
                kl,ce,mse=losses(dictionary,ids)
                loss=mse if name=='mse_control' else kl+.15*ce+.10*mse
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                with torch.no_grad():
                    dictionary.W_dec.div_(dictionary.W_dec.norm(dim=1,keepdim=True).clamp_min(1e-8))
                if step in [1,160,320,480,STEPS]:
                    with torch.no_grad():
                        values=np.asarray([[float(v) for v in losses(dictionary,vi[j:j+32])] for j in range(0,len(vi),32)])
                    row={'step':step,'validation_KL':float(values[:,0].mean()),'validation_CE':float(values[:,1].mean()),'validation_FVU_dev_scale':float(values[:,2].mean())}
                    assert np.isfinite(values).all()
                    curve.append(row);print(name,row,flush=True)
            for p in dictionary.params():p.requires_grad_(False)
            path=C.OUT/f'{name}.npz'
            np.savez(path,**dictionary.state())
            outputs[name]={'path':path.name,'sha256':C.file_sha(path),'curve':curve}
        assert snapshot(C.MODEL)==contract['model_snapshot']
    C.save(result,{**contract,'outputs':outputs,'seconds':time.time()-started})


def evaluate():
    from scipy.stats import t as student_t
    from transformers import AutoModelForCausalLM

    result=C.OUT/'dictionary-evaluation.json'
    if result.exists():return
    source=C.OUT/'fresh-documents.json';training=C.OUT/'dictionary-training.json'
    docs=C.read(source)['natural'];trained=C.read(training)
    positions=[511,583,655,727,799,871,943,1022]
    contract={**C.identity(),'schema':'functional_dictionary_evaluation_prereg.v1','source_sha256':C.file_sha(__file__),
              'fresh_documents_sha256':C.file_sha(source),'training_sha256':C.file_sha(training),'operations':NAMES,'positions':positions,
              'comparison':'same per-token active budget k=256, ranked by abs(a_j)*norm(W_down[:,j]) for native neurons',
              'gate':'On both heldouts, paired document CI for native256 minus functional SAE is >0 for KL and CE. Family alpha .05/4. All other contrasts and FVU descriptive.',
              'power_plan':'256 paired documents per split, paired t planning d=0.25, family alpha .0125, power >.8 under iid document-level approximation.'}
    C.save(C.OUT/'dictionary-evaluation-prereg.json',contract)
    torch.set_num_threads(4);rows=[];started=time.time();fields={};canary=None
    with GpuLease(),torch.inference_mode():
        assert snapshot(C.MODEL)==contract['model_snapshot']
        model=AutoModelForCausalLM.from_pretrained(C.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
        dictionaries={'archived_sae':archived()}
        for name,entry in trained['outputs'].items():
            p=C.OUT/entry['path'];assert C.file_sha(p)==entry['sha256']
            with np.load(p) as z:dictionaries[name]=TopKTranscoder.from_state(z,device='cuda')
        l=model.model.layers[35]
        take=lambda x:x[:,positions].reshape(-1,x.shape[-1]).detach().clone()
        hooks=[l.post_attention_layernorm.register_forward_pre_hook(lambda _,a:fields.update(r=take(a[0]))),
               l.mlp.register_forward_hook(lambda _,a,y:fields.update(y=take(y))),
               l.mlp.down_proj.register_forward_pre_hook(lambda _,a:fields.update(a=take(a[0]))),
               model.model.norm.register_forward_pre_hook(lambda _,a:fields.update(full=a[0].detach().clone()))]
        keep=torch.tensor(positions,device='cuda');down=l.mlp.down_proj.weight.float().T.contiguous()
        for lo in range(0,len(docs),4):
            batch=docs[lo:lo+4]
            x=torch.tensor([r['tokens'][:1024] for r in batch],device='cuda')
            target=torch.tensor([r['tokens'][p+1] for r in batch for p in positions],device='cuda')
            output=model(input_ids=x,logits_to_keep=keep,use_cache=False)
            clean=output.logits.reshape(-1,output.logits.shape[-1]);r,y,a,full=[fields[k] for k in ['r','y','a','full']]
            ops={'clean':y,'zero':torch.zeros_like(y),'native256':neuron_topk_recon(a.float(),down,256)}
            ops.update({name:d(y.float()) for name,d in dictionaries.items()})
            assert torch.equal(clean,tail_logits(model,r,y,full,positions,keep)),('native tail mismatch',lo)
            if lo==0:
                def edit(_,args,value):
                    value=value.clone();value[:,positions]=ops['functional_sae'].reshape(4,len(positions),2560).to(value.dtype)
                    return value
                h=l.mlp.register_forward_hook(edit)
                test=model(input_ids=x,logits_to_keep=keep,use_cache=False).logits.reshape_as(clean);h.remove()
                canary=torch.equal(test,tail_logits(model,r,ops['functional_sae'],full,positions,keep));assert canary
            yy=y.float().reshape(len(batch),len(positions),-1)
            per={}
            for name,op in ops.items():
                logits=clean if name=='clean' else tail_logits(model,r,op,full,positions,keep)
                per[name]={**metric(logits,clean,target),'sse':(op.float()-y.float()).square().sum(-1).cpu().numpy()}
            for j,d in enumerate(batch):
                sl=slice(j*len(positions),(j+1)*len(positions))
                rows.append({'document':lo+j,'source_id':d['id'],'language':d['language'],'split':d['split'],
                             'target_ids':target[sl].tolist(),'output_sum':yy[j].double().sum(0).cpu().numpy(),
                             'output_sq':float(yy[j].double().square().sum()),
                             'conditions':{name:{k:values[sl].tolist() for k,values in parts.items()} for name,parts in per.items()}})
            if (lo+4)%64==0:print('fresh evaluation',lo+4,'/',len(docs),flush=True)
        for hook in hooks:hook.remove()
        assert snapshot(C.MODEL)==contract['model_snapshot']

    def mean_doc(row,name,key):return float(np.mean(row['conditions'][name][key]))
    summary={}
    for split in ['confirm','lockbox']:
        subset=[r for r in rows if r['split']==split];n=len(subset)
        total=np.sum([r['output_sum'] for r in subset],axis=0)
        den=sum(r['output_sq'] for r in subset)-float(total@total)/(n*len(positions))
        means={name:{key:float(np.mean([mean_doc(r,name,key) for r in subset])) for key in ['CE','KL_clean']} for name in NAMES}
        for name in NAMES:
            v=means[name];v['FVU']=sum(sum(r['conditions'][name]['sse']) for r in subset)/den
            recovery_den=means['zero']['CE']-means['clean']['CE']
            v['loss_recovered']=(means['zero']['CE']-v['CE'])/recovery_den if recovery_den>0 else None
        comparisons={}
        for base in ['native256','archived_sae','mse_control']:
            comparisons[base]={}
            for key in ['KL_clean','CE']:
                dif=np.asarray([mean_doc(r,base,key)-mean_doc(r,'functional_sae',key) for r in subset])
                mean=float(dif.mean());half=float(student_t.ppf(1-.05/8,n-1)*dif.std(ddof=1)/np.sqrt(n))
                comparisons[base][key]={'base_minus_functional':mean,'adjusted_CI':[mean-half,mean+half]}
        summary[split]={'documents':n,'operations':means,'comparisons':comparisons,
                        'passed':all(v['adjusted_CI'][0]>0 for v in comparisons['native256'].values())}
    for row in rows:del row['output_sum'],row['output_sq']
    C.save(result,{**contract,'status':'complete','summary':summary,'native_layout_logit_exact':True,'true_hook_logit_exact':canary,
                   'both_passed':all(r['passed'] for r in summary.values()),'rows':rows,'seconds':time.time()-started})
    for split,r in summary.items():print(split,r['operations'],'pass',r['passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['fit','evaluate'])
    {'fit':fit,'evaluate':evaluate}[parser.parse_args().part]()
