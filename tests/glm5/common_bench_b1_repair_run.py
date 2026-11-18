"""Frozen B1 repair runner: native BF16, bounded CUDA, preserved original D."""
from __future__ import annotations
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
B=R.B


def effective_norm(h, delta):
    return ((h.float()+delta).to(h.dtype).float()-h.float()).norm(dim=-1)


def match_norm(h, direction, target):
    d=direction.float()/direction.float().norm(dim=-1,keepdim=True).clamp_min(1e-12)*target[:,None]
    for _ in range(5):
        d=d*(target/effective_norm(h,d).clamp_min(1e-12))[:,None]
    return d


class Runner:
    def __init__(self,stage):
        import torch
        from transformers import AutoTokenizer
        from shared.dist_protocol.natural_register_runner import snapshot
        import mlp_condition_design as MD, mlp_condition_run as MR, mlp_condition_stream_load as SL
        self.t=torch;self.start=time.monotonic();self.stage=stage
        self.pre=R.checked();self.plan=R.read(R.OLD/'plan.json')
        R.log(stage+'-identity',0,1,self.start)
        assert snapshot(R.MODEL)==self.pre['model_snapshot']
        MD.checked=lambda:self.plan
        MR.save=lambda name,body,plan:R.save(stage+'-'+name,body)
        self.m=SL.load(R.MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        self.tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
        self.tok.padding_side='left'
        if self.tok.pad_token_id is None:self.tok.pad_token_id=self.tok.eos_token_id
        self.ids=[self.plan['choice_ids'][k] for k in B.CLASSES]
        z=np.load(R.OLD/'staged-centres.npz')
        self.cent={k.split('__')[1]:torch.tensor(z[k],device='cuda') for k in z.files if k.startswith('32__')}
        self.u=torch.stack([self.m.lm_head.weight[sorted(set(self.ids[k]+self.ids[k+5]))].float().mean(0) for k in range(5)])
        self.heads=self.m.config.num_attention_heads
        self.hdim=self.m.config.head_dim
        self.s={};self.grab={};self.handles=[]
        for b,layer in enumerate(self.m.model.layers):
            self.handles.append(layer.register_forward_hook(self.block_hook(b)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.post_hook(b)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self.o_hook(b)))
        self.handles.append(self.m.model.norm.register_forward_hook(self.norm_hook))

    def block_hook(self,b):
        def fn(mod,args,out):
            h=out[0] if isinstance(out,tuple) else out
            if self.s.get('capture'):self.grab['raw',b]=h[:,-1].clone()
            if self.s.get('block')!=b or self.s.get('mode') not in ('add','set') or (self.s.get('prefill') and h.shape[1]==1):return
            h=h.clone();d=self.s['delta']
            h[:,-1]=(h[:,-1].float()+d).to(h.dtype) if self.s['mode']=='add' else d.to(h.dtype)
            return (h,*out[1:]) if isinstance(out,tuple) else h
        return fn

    def post_hook(self,b):
        def fn(mod,args):
            if self.s.get('capture'):self.grab['post',b]=args[0][:,-1].clone()
        return fn

    def o_hook(self,b):
        def fn(mod,args):
            if self.s.get('capture'):self.grab['oin',b]=args[0][:,-1].clone()
            heads=self.s.get('head_group',{}).get(b,[])
            if heads:
                a=args[0].clone()
                for h in heads:a[:,-1,h*self.hdim:(h+1)*self.hdim]=0
                return (a,*args[1:])
        return fn

    def norm_hook(self,mod,args,out):
        self.final=out[:,-1].clone()

    def batch(self,rows):
        return self.tok.pad({'input_ids':[r['tokens'] for r in rows]},return_tensors='pt').to('cuda')

    def run(self,batch,**state):
        self.s=state
        out=self.m(**batch,use_cache=False,logits_to_keep=1,output_hidden_states=state.get('capture',False))
        lg=out.logits[:,0].float();final=self.final.clone()
        self.s={}
        return lg,final,out.hidden_states

    def scores(self,lg):
        return self.t.stack([lg[:,j].max(1).values for j in self.ids],1)

    def vectors(self,rows,targets,h):
        t=self.t; ds={k:[] for k in ['pooled','cross_language','cross_source','unembed','random0','random1','random2','wrong_donor']}
        for r in rows:
            k=B.COLORS.index(r['gold_color']);kt=targets[r['id']]
            for arm,key in [('pooled','pooled'),('cross_language',f"L{-r['language']}"),('cross_source',f"S{-r['source']}")]:
                ds[arm].append(self.cent[key][kt]-self.cent[key][k])
            ds['unembed'].append(self.u[kt]-self.u[k])
            for j in range(3):ds[f'random{j}'].append(t.tensor(R.rng_for(r['id'],f'random{j}').standard_normal(h.shape[1]),device='cuda',dtype=t.float32))
            others=[x for x in range(5) if x not in (k,kt)]
            a,b=R.rng_for(r['id'],'wrong-donor').choice(others,2,replace=False)
            ds['wrong_donor'].append(self.cent['pooled'][b]-self.cent['pooled'][a])
        ds={k:t.stack(v)*R.ALPHA for k,v in ds.items()}
        norm=effective_norm(h,ds['pooled'])
        for arm in ['unembed','random0','random1','random2','wrong_donor']:ds[arm]=match_norm(h,ds[arm],norm)
        ds.update(direct_pooled=ds['pooled'],direct_cross_language=ds['cross_language'],direct_unembed=ds['unembed'])
        return ds,norm

    def measure(self,lg,base,rows,targets):
        sc=self.scores(lg);bs=self.scores(base);pred=sc.argmax(1);top=lg.argmax(1)
        kl=(base.softmax(-1)*(base.log_softmax(-1)-lg.log_softmax(-1))).sum(-1)
        out=[]
        for i,r in enumerate(rows):
            k=B.COLORS.index(r['gold_color']);kt=targets[r['id']];lang=0 if r['language']==1 else 5
            out.append({'pred':int(pred[i]),'top1':int(top[i]),'success':bool(pred[i]==lang+kt),
                'top1_success':int(top[i]) in self.ids[lang+kt],'unchanged':bool(pred[i]==lang+k),
                'language_kept':int(pred[i])//5==lang//5,'other_language_target':int(pred[i])==(5-lang)+kt,
                'margin':float(sc[i,lang+kt]-sc[i,lang+k]),'base_margin':float(bs[i,lang+kt]-bs[i,lang+k]),'kl_base_to_arm':float(kl[i])})
        return out

    def group_write(self,group):
        out=self.t.zeros_like(self.grab['raw',35],dtype=self.t.float32)
        for b,heads in group.items():
            a=self.grab['oin',b].reshape(-1,self.heads,self.hdim).float()
            w=self.m.model.layers[b].self_attn.o_proj.weight.float().reshape(-1,self.heads,self.hdim)
            for h in heads:out+=a[:,h]@w[:,h].T
        return out

    def deadline(self):
        assert time.monotonic()-self.start<3600,'Stage GPU budget exhausted'
        R.check_write(R.OUT,1<<30,operation='B1批次磁盘检查')


def collect(split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    pre=R.checked();pilot=split=='discovery'
    assert not (R.OUT/f'{split}-started.json').exists(),'Split already started; no silent replay'
    if not pilot:
        fz=R.read(R.OUT/'path-frozen.json')
        assert fz['prereg_sha256']==R.sha(R.OUT/'prereg.json')
        assert fz['pilot_items_sha256']==R.sha(R.OUT/'discovery-items.json')
    assert R.sha(R.OLD/f'{split}-materials.json')==pre['materials'][split]['sha256']
    rows=R.read(R.OLD/f'{split}-materials.json')
    if pilot:rows=[r for r in rows if r['family']<pre['pilot_families']]
    assert len(rows)%8==0
    targets=R.target_map(rows)
    R.save(f'{split}-started.json',{'split':split,'n':len(rows),'material_sha256':pre['materials'][split]['sha256'],'targets':targets},True)
    folder=R.OUT/f'{split}-native';folder.mkdir()
    arrays={};items=[];score=np.zeros((2,36,32));count=np.zeros(2)
    with GpuLease(),torch.inference_mode():
        rr=Runner(split);n=len(rows);d=rr.m.config.hidden_size
        groups={} if pilot else {k:{int(b):v for b,v in g.items()} for k,g in fz['groups'].items()}
        arms=R.BASE_ARMS+list(groups)
        shapes={'hidden':(n,37,3,d),'postattn':(n,36,d),'oin':(n,36,rr.heads*rr.hdim),'final_raw':(n,d),'final_states':(n,len(arms)+1,d)}
        if groups:shapes['path_pred_final']=(n,len(groups),d)
        for k,shape in shapes.items():arrays[k]=np.lib.format.open_memmap(folder/f'{k}.npy',mode='w+',dtype=np.uint16,shape=shape)
        bits=lambda x:x.contiguous().view(torch.uint16).cpu().numpy()
        R.log(split,0,n,rr.start)
        calibration=[]
        for start in range(0,n,8):
            rs=rows[start:start+8];batch=rr.batch(rs);base,final,hs=rr.run(batch,capture=True)
            saved=rr.grab['raw',31].clone();finalraw=rr.grab['raw',35].clone()
            ds,norm=rr.vectors(rs,targets,saved)
            if start==0:
                no,_,_=rr.run(batch,mode='add',block=31,delta=torch.zeros_like(saved).float())
                restored,_,_=rr.run(batch)
                cal={'noop_equal':bool(torch.equal(no,base)),'restore_equal':bool(torch.equal(restored,base))}
                calibration.append(cal);assert all(cal.values()),'No-op calibration failed'
            width=batch['input_ids'].shape[1]
            for i,r in enumerate(rs):
                pos=[width-len(r['tokens'])+r['positions'][p] for p in ['doc_end','question_end','answer_prefix']]
                arrays['hidden'][start+i]=bits(torch.stack([h[i,pos] for h in hs]))
            arrays['postattn'][start:start+8]=bits(torch.stack([rr.grab['post',b] for b in range(36)],1))
            arrays['oin'][start:start+8]=bits(torch.stack([rr.grab['oin',b] for b in range(36)],1))
            arrays['final_raw'][start:start+8]=bits(finalraw)
            arrays['final_states'][start:start+8,0]=bits(final)
            pred=rr.scores(base).argmax(1);top=base.argmax(1)
            rec=[{'id':r['id'],'family':r['family'],'gold':B.CLASSES.index(r['gold']),'base':int(pred[i]),
                  'base_top1':int(top[i]),'base_top1_correct':int(top[i]) in rr.ids[B.CLASSES.index(r['gold'])],
                  'target_color':targets[r['id']],'effective_pooled_norm':float(norm[i]),'arms':{}} for i,r in enumerate(rs)]
            if pilot:
                # Local algebraic projection onto the frozen discovery colour contrast.
                k=torch.tensor([B.COLORS.index(r['gold_color']) for r in rs],device='cuda')
                direction=rr.cent['pooled'][k]-rr.cent['pooled'].mean(0)
                direction=direction/direction.norm(dim=-1,keepdim=True)
                for b in range(14,32):
                    a=rr.grab['oin',b].float().reshape(8,rr.heads,rr.hdim)
                    w=rr.m.model.layers[b].self_attn.o_proj.weight.float().reshape(d,rr.heads,rr.hdim)
                    proj=torch.einsum('bd,dhv,bhv->bh',direction,w,a).cpu().numpy()
                    for si,s in enumerate((1,-1)):
                        mask=np.array([r['source']==s for r in rs]);score[si,b]+=proj[mask].sum(0)
                for si,s in enumerate((1,-1)):count[si]+=sum(r['source']==s for r in rs)
            predictions={}
            for pi,(arm,g) in enumerate(groups.items()):
                write=rr.group_write(g)
                pf=rr.m.model.norm((finalraw.float()-write).to(finalraw.dtype))
                pl=rr.m.lm_head(pf).float()
                predictions[arm]=rr.measure(pl,base,rs,targets)
                arrays['path_pred_final'][start:start+8,pi]=bits(pf)
            for ai,arm in enumerate(arms):
                if arm.startswith('donor_'):
                    perm=[i^(2 if arm=='donor_q' else 1) for i in range(8)]
                    lg,fn,_=rr.run(batch,mode='set',block=31,delta=saved[perm])
                elif arm in groups:lg,fn,_=rr.run(batch,head_group=groups[arm])
                else:lg,fn,_=rr.run(batch,mode='add',block=35 if arm.startswith('direct_') else 31,delta=ds[arm])
                measures=rr.measure(lg,base,rs,targets)
                for i,v in enumerate(measures):
                    if arm.startswith('donor_'):v['donor_gold']=B.CLASSES.index(rs[perm[i]]['gold'])
                    elif arm in groups:v['prediction']=predictions[arm][i]
                    else:
                        actual=effective_norm(finalraw if arm.startswith('direct_') else saved,ds[arm])
                        v['effective_norm']=float(actual[i]);v['relative_norm_error']=float(abs(actual[i]/norm[i]-1))
                    rec[i]['arms'][arm]=v
                arrays['final_states'][start:start+8,ai+1]=bits(fn)
            items.extend(rec)
            if start%64==0 or start+8==n:
                for a in arrays.values():a.flush()
                rr.deadline();R.log(split,start+8,n,rr.start)
        R.save(f'{split}-items.json',{'rows':items,'calibration':calibration,'arms':arms,'elapsed_seconds':time.monotonic()-rr.start},True)
        gate,_=R.behavior_gate(rows,items)
        R.save(f'{split}-gate.json',gate,True)
        if pilot:
            groups={}
            for si,source in enumerate(('document','parametric')):
                ranked=sorted([(float(score[si,b,h]/count[si]),b,h) for b in range(14,32) for h in range(rr.heads)],reverse=True)
                chosen=[(b,h) for v,b,h in ranked if v>0][:4];assert len(chosen)==4
                group={}
                for b,h in chosen:group.setdefault(b,[]).append(h)
                random={b:R.rng_for(f'{source}-{b}','head-control').choice([h for h in range(rr.heads) if h not in hs],len(hs),replace=False).tolist() for b,hs in group.items()}
                groups['heads_'+source]=group;groups['heads_'+source+'_random']=random
            R.save('path-frozen.json',{'groups':groups,'scores':(score/count[:,None,None]).tolist(),
                'pilot_gate':gate,'prereg_sha256':R.sha(R.OUT/'prereg.json'),'pilot_items_sha256':R.sha(R.OUT/'discovery-items.json'),
                'selection':'discovery natural head-write projection; local source evidence only'},True)
    for a in arrays.values():a.flush()
    arrays.clear()
    R.save(f'{split}-manifest.json',{'schema':'b1_repair_native.v2','positions':['doc_end','question_end','answer_prefix'],
        'encoding':'uint16 BF16 bits; hs36 is final normalized, final_raw is block35 output',
        'arms':['base']+arms,'groups':groups,'row_ids':[r['id'] for r in rows],
        'files':{p.name:{'sha256':R.sha(p),'bytes':p.stat().st_size} for p in folder.glob('*.npy')},
        'purpose':'reusable native evidence and full-logit reconstruction','review_by':'2026-12-10'},True)


def generation():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    assert not (R.OUT/'generation.json').exists()
    outputs=[]
    with GpuLease(),torch.inference_mode():
        rr=Runner('generation')
        for split in ['lockbox','lockbox_frame']:
            allrows=R.read(R.OLD/f'{split}-materials.json');targets=R.read(R.OUT/f'{split}-started.json')['targets']
            fams=sorted({r['family'] for r in allrows})[:16];rows=[r for r in allrows if r['family'] in fams]
            for start in range(0,len(rows),8):
                rs=rows[start:start+8];batch=rr.batch(rs);rr.run(batch,capture=True)
                ds,_=rr.vectors(rs,targets,rr.grab['raw',31])
                for arm in ['base','pooled','unembed','cross_language']:
                    rr.s={} if arm=='base' else {'mode':'add','block':31,'delta':ds[arm],'prefill':True}
                    ids=rr.m.generate(**batch,do_sample=False,max_new_tokens=6,use_cache=True,pad_token_id=rr.tok.pad_token_id)
                    rr.s={}
                    for i,r in enumerate(rs):
                        new=ids[i,batch['input_ids'].shape[1]:].tolist()
                        outputs.append({'id':r['id'],'split':split,'arm':arm,'target_color':targets[r['id']],
                                        'tokens':new,'text':rr.tok.decode(new,skip_special_tokens=True)})
                if start%32==0:rr.deadline();R.log('generation-'+split,start+8,len(rows),rr.start)
        R.save('generation.json',{'rows':outputs,'max_new_tokens':6,'hook':'prompt last token only; cache contains the intervened prompt states'},True)


def capability():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    assert not (R.OUT/'capability.json').exists()
    rows=R.read(R.CAP);outputs=[]
    assert R.sha(R.CAP)==R.checked()['capability']['sha256']
    with GpuLease(),torch.inference_mode():
        rr=Runner('capability')
        for start in range(0,len(rows),8):
            rs=rows[start:start+8];fake=[];targets={}
            for r in rs:
                prompt='Answer the multiple-choice question with its letter only.\n'+r['question']+'\n'+'\n'.join(f'{chr(65+j)}. {s}' for j,s in enumerate(r['choices']))+'\nAnswer:'
                k=int(R.rng_for(r['id'],'cap-colour').integers(5));kt=(k+1+int(R.rng_for(r['id'],'cap-target').integers(4)))%5
                fake.append({'id':r['id'],'tokens':rr.tok.encode(prompt,add_special_tokens=False),'gold_color':B.COLORS[k],'language':1,'source':1})
                targets[r['id']]=kt
            batch=rr.batch(fake);base,final,_=rr.run(batch,capture=True);ds,_=rr.vectors(fake,targets,rr.grab['raw',31])
            records=[{'id':r['id'],'subject':r['subject'],'gold':r['answer'],'arms':{}} for r in rs]
            for arm in ['base','pooled','unembed','random0']:
                lg=base if arm=='base' else rr.run(batch,mode='add',block=31,delta=ds[arm])[0]
                kl=(base.softmax(-1)*(base.log_softmax(-1)-lg.log_softmax(-1))).sum(-1)
                for i,r in enumerate(rs):
                    letters=[chr(65+j) for j in range(len(r['choices']))]
                    letter_ids=[sorted({rr.tok.encode(x+s,add_special_tokens=False)[0] for x in ('',' ')}) for s in letters]
                    pred=letters[int(torch.stack([lg[i,ids].max() for ids in letter_ids]).argmax())]
                    records[i]['arms'][arm]={'pred':pred,'correct':pred==r['answer'],'kl':float(kl[i])}
            outputs+=records
            if start%64==0 or start+8==len(rows):rr.deadline();R.log('capability',start+8,len(rows),rr.start)
        R.save('capability.json',{'rows':outputs,'scope':'historical exposed MCQ pool; arbitrary balanced colour edits, forced letter scoring'},True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['discovery','lockbox','lockbox_frame','generation','capability']);a=ap.parse_args()
    if a.stage in ('generation','capability'):globals()[a.stage]()
    else:collect(a.stage)
