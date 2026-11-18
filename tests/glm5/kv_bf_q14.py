"""BF16 layer-streaming 14B reference with parallel answer-query counterfactuals."""

from contextlib import ExitStack, contextmanager
import gc
from itertools import batched
import json
import time
from pathlib import Path
import sys

import numpy as np

import kv_bf_run as C
import kv_bf_study as Study
import kv_bf_q4 as Q
from kv_bf_freeze import json_counts
from kv_relay_native import CAP, CONDITIONS


A=C.A
R=A.R
BATCH=16
SOURCE_LAYERS=tuple(range(19,28))
RESIDENT=int(4.75*2**30)


def f32(x):
    return x.detach().cpu().float().numpy().copy()


class Engine:
    """Load each streamed shard tensor once per block, using the original files."""
    def __init__(self):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding,apply_rotary_pos_emb
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        self.t=torch
        self.rotate=apply_rotary_pos_emb
        self.sdpa=sdpa_attention_forward
        self.folder=C.ROOT/"models/hf"/R.Base.MODELS["q14"]
        cfg=AutoConfig.from_pretrained(self.folder,local_files_only=True)
        with torch.device("meta"):
            self.model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation="sdpa").eval()
        self.tok=AutoTokenizer.from_pretrained(self.folder,local_files_only=True)
        self.tok.padding_side="left"
        self.model.generation_config=GenerationConfig.from_pretrained(self.folder,local_files_only=True)
        self.index=json.loads((self.folder/"model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.open_tensor_file=safe_open
        self.decoder=self.model.model
        self.rope=Qwen3RotaryEmbedding(cfg,device="cpu").cuda()
        self.decoder.rotary_emb=self.rope
        self.layers=list(self.decoder.layers)
        self.heads=cfg.num_attention_heads
        self.groups=cfg.num_attention_heads//cfg.num_key_value_heads
        self.width=cfg.hidden_size
        self.bytes=0
        for name in ("model.embed_tokens","model.norm","lm_head"):
            mod=self.model.get_submodule(name)
            src=self.weight(name+".weight")
            mod.load_state_dict({"weight":src},assign=True,strict=True)
            self.bytes+=src.numel()*2
        self.keep=set()
        for i in range(len(self.layers)):
            cost=sum(p.numel()*2 for p in self.layers[i].parameters())
            if self.bytes+cost>RESIDENT:
                break
            self.load(i)
            self.keep.add(i)
            self.bytes+=cost
        self.letters=[self.tok.encode(" "+x,add_special_tokens=False)[0] for x in "PQ"]

    def weight(self,n):
        with self.open_tensor_file(self.folder/self.index[n],framework="pt",device="cpu") as tensors:
            x=tensors.get_tensor(n)
            assert x.dtype==self.t.bfloat16
            return x.to("cuda")

    def load(self,i):
        block=self.layers[i]
        by_file={}
        for name in block.state_dict():
            fullname=f"model.layers.{i}."+name
            by_file.setdefault(self.index[fullname],[]).append((name,fullname))
        state={}
        for filename,names in by_file.items():
            with self.open_tensor_file(self.folder/filename,framework="pt",device="cpu") as data:
                for name,full in names:
                    x=data.get_tensor(full)
                    assert x.dtype==self.t.bfloat16
                    state[name]=x.to("cuda")
        block.load_state_dict(state,assign=True,strict=True)

    @contextmanager
    def block(self,i):
        if i not in self.keep:
            self.load(i)
        try:
            yield self.layers[i]
        finally:
            if i not in self.keep:
                self.layers[i].to_empty(device="meta")

    def close(self):
        del self.layers,self.decoder,self.model,self.rope
        gc.collect()
        self.t.cuda.empty_cache()

    def sample(self,rows,observed):
        t=self.t
        sequences=[]
        donors=[]
        places=[]
        labels=[]
        first=[]
        for r,o in zip(rows,observed,strict=True):
            ids,opts=A.encoded(self.tok,r)
            assert ids==o["input_ids"] and opts==o["options"]
            if o["joint"]:
                ix=o["index"]
                continued=o["tokens"][:ix["code"]]
                s,n=ix["state"],ix["note"]
            else:
                second="State" if r["order"] else "Note"
                start=r["note"] if r["order"] else r["state"]
                follow=r[second.lower()]
                txt=f" {start}\n{second}: {follow}\nCode:"
                continued=self.tok.encode(txt,add_special_tokens=False)
                s=next(j for j,i in enumerate(continued) if i==opts[r["state"]])
                n=next(j for j,i in enumerate(continued) if i==opts[r["note"]])
            full=ids+continued
            base=full.copy()
            base[len(ids)+s]=opts[r["alt"]]
            assert full[len(ids)+s]==opts[r["state"]] and full[len(ids)+n]==opts[r["note"]]
            sequences.append(full)
            donors.append(base)
            places.append((len(ids)+s,len(ids)+n))
            labels.append(self.letters[r["code"]=="Q"])
            first.append(len(ids))
        n=len(rows)
        width=max(map(len,sequences))
        pads=[width-len(x) for x in sequences]
        def padded(xs):
            return [[self.tok.pad_token_id]*p+x for p,x in zip(pads,xs,strict=True)]
        ids=t.as_tensor(padded(sequences)+padded(donors),device="cuda")
        real=t.arange(width,device="cuda")[None]>=t.as_tensor(pads,device="cuda")[:,None]
        idx=t.arange(width,device="cuda")
        visible=real[:,None,None,:] & (idx[None,:]<=idx[:,None])[None,None]
        pos=(real.cumsum(-1)-1).masked_fill(~real,1).repeat(2,1)
        emb=self.decoder.embed_tokens(ids)
        cs=self.rope(emb,pos)
        slots=t.as_tensor(places,device="cuda")+t.as_tensor(pads,device="cuda")[:,None]
        return dict(n=n,ids=ids,emb=emb,cs=cs,mask=visible.repeat(2,1,1,1),last=visible[:,:,-1:],
                    slots=slots,target=t.as_tensor(labels,device="cuda"),first=np.asarray(first)+pads)

    def qkv(self,l,x,angles):
        t=self.t
        v=l.input_layernorm(x)
        s=(*v.shape[:-1],-1,128)
        m=l.self_attn
        q=m.q_norm(m.q_proj(v).view(s)).transpose(1,2)
        k=m.k_norm(m.k_proj(v).view(s)).transpose(1,2)
        val=m.v_proj(v).view(s).transpose(1,2)
        q,k=self.rotate(q,k,*angles)
        return q,k,val

    def logits(self,h):
        return self.model.lm_head(self.decoder.norm(h))

    def attention(self,l,q,k,v,mask):
        return self.sdpa(l.self_attn,q,k,v,mask,scaling=128**-.5,dropout=0.)[0]

    def residual(self,l,x,o):
        a=x+l.self_attn.o_proj(o.reshape(*x.shape[:-1],self.width).contiguous())
        return a+l.mlp(l.post_attention_layernorm(a))

    def read(self,q,k,visible):
        t=self.t
        kk=k.repeat_interleave(self.groups,dim=1)
        s=(q.float()*kk.float()).sum(-1)*(128**-.5)
        return s.masked_fill(~visible[:,0],-t.inf).softmax(-1)

    def slots(self,obj,idx):
        t=self.t
        return obj.gather(-1,idx[:,None].expand(-1,obj.shape[1],-1))

    def slot_vectors(self,x,j):
        return x.gather(2,j[:,None,:,None].expand(-1,8,-1,128))

    def foreground(self,l,h,q,k,v,mask):
        # One physical query head from its own GQA group.
        g=h//self.groups
        y=self.t.nn.functional.scaled_dot_product_attention(q[:,h:h+1],k[:,g:g+1],v[:,g:g+1],
                                                            attn_mask=mask,scale=128**-.5,is_causal=False)
        return y[:,0,0]

    def forced_behavior(self,x,s,b_rows,check_prefix=True):
        t=self.t
        n=s["n"]
        last=self.logits(x[:n,-1])
        ans=last.argmax(-1)
        result=[]
        if check_prefix:
            width=s["ids"].shape[1]
            start=int(min(s["first"]))-1
            pred=self.logits(x[:n,start:-1]).argmax(-1).cpu().numpy()
        else:
            start=0
            pred=None
        ids=s["ids"][:n].cpu().numpy()
        for i,(r,a,e) in enumerate(zip(b_rows,ans.tolist(),s["target"].tolist(),strict=True)):
            if pred is None:
                same,first=None,None
            else:
                left=s["first"][i]
                diff=np.flatnonzero(pred[i,left-1-start:]!=ids[i,left:])
                same=not len(diff)
                first=dict(offset=int(diff[0]),token=int(pred[i,left-1-start+diff[0]]),
                           rendered=self.tok.decode(int(pred[i,left-1-start+diff[0]]))) if len(diff) else None
            result.append(dict(id=r["id"],conditional=bool(a==e),greedy_prefix_and_code=bool(same and a==e) if check_prefix else None,
                               predicted=a,expected=e,first_mismatch=first))
        return last[:,self.letters],result

    def scan(self,rows,obs):
        """Score upstream writes on discovery, without using any confirmation responses."""
        t=self.t
        s=self.sample(rows,obs)
        n=s["n"]
        x=s["emb"]
        actions={}
        for i in range(max(SOURCE_LAYERS)+1):
            with self.block(i) as l:
                q,k,v=self.qkv(l,x,s["cs"])
                if i in SOURCE_LAYERS:
                    p=self.slots(self.read(q[:n,:,-1:],k[:n],s["last"]),s["slots"])
                    y=self.slot_vectors(v,s["slots"].repeat(2,1))
                    d=(y[n:,:,0].float()-y[:n,:,0].float()).repeat_interleave(self.groups,1)
                    target=t.minimum(p.amax(-1),2*p.amin(-1))
                    w=l.self_attn.o_proj.weight.reshape(self.width,self.heads,128)
                    length=t.linalg.vector_norm(t.einsum("bhd,mhd->bhm",d,w.float()),dim=-1)
                    actions[i]=f32(target*length)
                x=self.residual(l,x,self.attention(l,q,k,v,s["mask"]))
        return actions

    def one_v(self,l,h,source_q,k,v,vis,slots,alpha,d,cell):
        """Native BF16 source edit to one head using the frozen target and cap."""
        t=self.t
        slot,signal=CONDITIONS[cell]
        if not signal:
            d=t.roll(d,(13+7*l.self_attn.layer_idx)%127+1,dims=-1)
        n=len(v)
        idx=t.arange(n,device="cuda")
        loc=slots[:,slot]
        g=h//self.groups
        target=t.minimum(alpha.max(-1).values,2*alpha.min(-1).values)
        gain=target/alpha[:,slot].clamp_min(1e-12)
        old=v[idx,g,loc]
        put=(old.float()+gain.clamp_max(CAP)[:,None]*d).to(v.dtype)
        new=v.clone()
        new[idx,g,loc]=put
        base=self.foreground(l,h,source_q,k,v,vis)
        z=self.foreground(l,h,source_q,k,new,vis)
        assay=dict(want=target[:,None]*d,gain=gain,analytic=alpha[:,slot,None]*(put.float()-old.float()),
                   actual=z.float()-base.float())
        return z,assay

    def branch_qkv(self,l,last,k,v,angles):
        q,k1,v1=self.qkv(l,last[:,None],angles)
        kk,vv=k.clone(),v.clone()
        kk[:,:,-1:]=k1
        vv[:,:,-1:]=v1
        return q,kk,vv

    def read_receiver(self,l,branch,sources,vis,slots,cfg,arms,discover=False):
        t=self.t
        b,q0,k0,v0,z0,don,dsrc,cs=sources
        n=len(b)
        idx=t.arange(n,device="cuda")
        pk=self.slots(self.read(q0,k0,vis),slots)
        if discover:
            delta=[]
            for x in branch:
                q,k,v=self.branch_qkv(l,x,k0,v0,cs)
                p=self.slots(self.read(q,k,vis),slots)
                delta.append(f32((p.clamp_min(1e-12).log()-pk.clamp_min(1e-12).log()).square().mean(-1)))
            return np.stack(delta,1),f32(pk)
        j0,j1=cfg["receiver"]
        d0=self.slot_vectors(v0,slots)
        donor=[don[:,h//self.groups].float()-d0[:,h//self.groups,0].float() for h in (j0,j1)]
        outgoing={}
        result={}
        for cell,x in enumerate(branch):
            for variant in arms:
                q,k,v=self.branch_qkv(l,x,k0,v0,cs)
                if variant!="dynamic":
                    q[:,[j0,j1]]=q0[:,[j0,j1]]
                p=self.slots(self.read(q,k,vis),slots)
                visible=vis
                mixed=self.attention(l,q,k,v,visible)
                added=None
                assay=[]
                for ii,h in enumerate((j0,j1)):
                    natural=pk[:,h]
                    target=t.minimum(natural.max(-1).values,2*natural.min(-1).values)
                    slot,signal=CONDITIONS[cell]
                    alpha=p[:,h,slot]
                    gain=target/alpha.clamp_min(1e-12)
                    d=donor[ii]
                    if not signal:
                        d=t.roll(d,(13+7*l.self_attn.layer_idx)%127+1,dims=-1)
                    group=h//self.groups
                    take=slots[:,slot]
                    prev=v[idx,group,take]
                    scaled=gain.clamp_max(CAP)[:,None]*d
                    put=(prev.float()+scaled).to(v.dtype)
                    rounded=put.float()-prev.float()
                    base=mixed[:,0,h].float()
                    if ii==0 or variant in ("native","dynamic"):
                        alt=v.clone()
                        alt[idx,group,take]=put
                        changed=self.foreground(l,h,q,k,alt,visible)
                        mixed[:,0,h]=changed
                        actual=changed.float()-base
                        analytic=alpha[:,None]*rounded
                    elif variant=="once":
                        analytic=alpha[:,None]*scaled
                        changed=(base+analytic).to(v.dtype)
                        mixed[:,0,h]=changed
                        actual=changed.float()-base
                    else:
                        analytic=alpha[:,None]*(rounded if variant=="late-round" else scaled)
                        actual=analytic
                        added=analytic
                    assay.append(dict(want=target[:,None]*d,analytic=analytic,actual=actual,gain=gain))
                projection=l.self_attn.o_proj(mixed.reshape(n,1,self.width))
                if added is not None:
                    w=l.self_attn.o_proj.weight[:,j1*128:(j1+1)*128].float()
                    projection=(projection.float()+(added@w.T)[:,None]).to(projection.dtype)
                y=x[:,None]+projection
                y=y+l.mlp(l.post_attention_layernorm(y))
                key=variant,cell
                outgoing[key]=y[:,0]
                result[key]=dict(alpha=f32(p[:,[j0,j1]]),dose={k:np.stack([f32(a[k]) for a in (*assay,dsrc[cell])],1)
                                                            for k in assay[0]})
        groups=[j0//self.groups,j1//self.groups]
        states=d0[:,groups]
        return outgoing,result,f32(pk[:,[j0,j1]]),dict(value_state=f32(states[:,:,0]),
                                                         value_note=f32(states[:,:,1]),value_donor=f32(don[:,groups]))

    def calculate(self,rows,obs,design=None,scan_receivers=False,variants=None,behavior_prefix=True):
        t=self.t
        s=self.sample(rows,obs)
        n=s["n"]
        x=s["emb"]
        arms={}
        source=[]
        source_dose=[]
        rec={}
        pk=None
        for i in range(len(self.layers)):
            if scan_receivers and i>design["source"][0]+1:
                break
            with self.block(i) as l:
                angles=tuple(j[:len(x)] for j in s["cs"])
                q,k,v=self.qkv(l,x,angles)
                mask=s["mask"][:len(x)]
                o=self.attention(l,q,k,v,mask)
                qq,kk,vv=q[:n,:,-1:],k[:n],v[:n]
                last_cs=tuple(j[:n,-1:] for j in s["cs"])
                if design is not None and i==design["source"][0]:
                    h=design["source"][1]
                    g=h//self.groups
                    p=self.slots(self.read(qq,kk,s["last"]),s["slots"])[:,h]
                    vv2=self.slot_vectors(v,s["slots"].repeat(2,1))
                    d=(vv2[n:,g,0].float()-vv2[:n,g,0].float())
                    for cell in range(4):
                        y,z=self.one_v(l,h,qq,kk,vv,s["last"],s["slots"],p,d,cell)
                        duplicate=o[:n,-1:].clone()
                        duplicate[:,0,h]=y
                        source.append(self.residual(l,x[:n,-1:],duplicate)[:,0])
                        source_dose.append(z)
                if design is not None and i==design["source"][0]+1:
                    donor=self.slot_vectors(v[n:],s["slots"])[:,:,0]
                    r=self.read_receiver(l,source,(x[:n,-1],qq,kk,vv,o[:n],donor,source_dose,last_cs),
                                         s["last"],s["slots"],design,variants,discover=scan_receivers)
                    if scan_receivers:
                        return r
                    arms,rec,pk,states=r
                elif arms:
                    # Reuse this native layer and immutable history for every one-token arm.
                    for j,z in arms.items():
                        q1,k1,v1=self.branch_qkv(l,z,kk,vv,last_cs)
                        arms[j]=self.residual(l,z[:,None],self.attention(l,q1,k1,v1,s["last"]))[:,0]
                x=self.residual(l,x,o)
                if design is None or i>=design["source"][0]+1:
                    x=x[:n]
        native,behavior=self.forced_behavior(x,s,rows,check_prefix=behavior_prefix)
        if not arms:
            return behavior
        variants_out={}
        for (mode,cell),last in arms.items():
            r=rec[mode,cell]
            r["scores"]=f32(self.logits(last)[:,self.letters])
            variants_out.setdefault(mode,[]).append(r)
        return dict(ids=np.array([r["id"] for r in rows]),native_score=f32(native),natural_alpha=pk,**states),variants_out

    @contextmanager
    def stock_forward(self):
        handles=[]
        for i,m in enumerate(self.layers):
            if i in self.keep:
                continue
            def attach(l,args,j=i):
                self.load(j)
            def free(l,args,out):
                l.to_empty(device="meta")
            handles.extend((m.register_forward_pre_hook(attach),m.register_forward_hook(free)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def native_check(self,rows,obs):
        t=self.t
        s=self.sample(rows,obs)
        n=s["n"]
        x=s["emb"][:n]
        angles=tuple(y[:n] for y in s["cs"])
        for i in range(len(self.layers)):
            with self.block(i) as l:
                q,k,v=self.qkv(l,x,angles)
                x=self.residual(l,x,self.attention(l,q,k,v,s["mask"][:n]))
        mine=self.logits(x[:,-1])
        mask=s["last"][:,0,0].long()
        pos=(mask.cumsum(-1)-1).masked_fill(mask==0,1)
        with self.stock_forward():
            native=self.model(input_ids=s["ids"][:n],attention_mask={"full_attention":s["mask"][:n]},position_ids=pos,use_cache=False,
                              logits_to_keep=1).logits[:,0]
        err=(mine.float()-native.float()).abs()
        result=dict(max_error=float(err.max()),pq_error=float(err[:,self.letters].max()),
                    full_top1_agreement=bool(t.equal(native.argmax(-1),mine.argmax(-1))),
                    n=n,ids=[r["id"] for r in rows],mask="Both runs use the same explicit causal 4-D mask.")
        assert result["pq_error"]<=.25 and result["full_top1_agreement"],result
        return result

    def pipeline(self,packs,mode,progress,design=None,variants=None,check_prefix=True):
        """Layer-major execution. Keep activations on CPU while the next block is loaded."""
        t=self.t
        packets=[]
        for rows,obs in packs:
            state=self.sample(rows,obs)
            current=state.pop("emb")
            if mode=="behavior":
                current=current[:len(rows)]
            vectors={k:state[k].cpu() for k in ("mask","last","slots","ids","target")}
            vectors["cs"]=tuple(x.cpu() for x in state["cs"])
            vectors.update(n=state["n"],first=state["first"])
            packets.append(dict(rows=rows,s=vectors,x=current.cpu(),scans={},arms={}))
        del state,current,vectors
        if mode=="source":
            stop=max(SOURCE_LAYERS)+1
        elif mode=="receiver":
            stop=design["source"][0]+2
        else:
            stop=len(self.layers)
        start=time.monotonic()
        progress(mode+"-layer-major",0,stop,start,slabs=len(packets))
        for i in range(stop):
            with self.block(i) as l:
                for p in packets:
                    self.pipeline_step(l,i,p,mode,design,variants)
            progress(mode+"-layer-major",i+1,stop,start,slabs=len(packets))
        out=[]
        for p in packets:
            if mode=="source":
                out.append(p["scans"])
                continue
            if mode=="receiver":
                out.append(p["receiver"])
                continue
            s={k:v.cuda() if t.is_tensor(v) else v for k,v in p["s"].items()}
            x=p["x"].cuda()
            scores,records=self.forced_behavior(x,s,p["rows"],check_prefix=check_prefix)
            if mode=="behavior":
                out.append(records)
                continue
            res={}
            for (method,cell),last in p["arms"].items():
                info=p["record"][method,cell]
                info["scores"]=f32(self.logits(last.cuda())[:,self.letters])
                res.setdefault(method,[]).append(info)
            out.append((dict(ids=np.array([r["id"] for r in p["rows"]]),native_score=f32(scores),
                             natural_alpha=p["alpha"],**p["values"]),res))
        return out

    def pipeline_step(self,l,i,p,mode,design,variants):
        t=self.t
        s=p["s"]
        x=p["x"].cuda()
        n=s["n"]
        angles=tuple(j[:len(x)].cuda() for j in s["cs"])
        mask=s["mask"][:len(x)].cuda()
        vis=s["last"].cuda()
        slots=s["slots"].cuda()
        q,k,v=self.qkv(l,x,angles)
        o=self.attention(l,q,k,v,mask)
        qq,kk,vv=q[:n,:,-1:],k[:n],v[:n]
        last_cs=tuple(j[:n,-1:] for j in angles)
        if mode=="source" and i in SOURCE_LAYERS:
            sq=self.slots(self.read(qq,kk,vis),slots)
            y=self.slot_vectors(v,slots.repeat(2,1))
            d=(y[n:,:,0].float()-y[:n,:,0].float()).repeat_interleave(self.groups,1)
            target=t.minimum(sq.amax(-1),2*sq.amin(-1))
            w=l.self_attn.o_proj.weight.reshape(self.width,self.heads,128)
            length=t.linalg.vector_norm(t.einsum("bhd,mhd->bhm",d,w.float()),dim=-1)
            p["scans"][i]=f32(target*length)
        if design is not None and i==design["source"][0]:
            h=design["source"][1]
            g=h//self.groups
            weights=self.slots(self.read(qq,kk,vis),slots)[:,h]
            va=self.slot_vectors(v,slots.repeat(2,1))
            d=va[n:,g,0].float()-va[:n,g,0].float()
            p["early"]=[]
            p["early_dose"]=[]
            for cell in range(4):
                y,z=self.one_v(l,h,qq,kk,vv,vis,slots,weights,d,cell)
                clone=o[:n,-1:].clone()
                clone[:,0,h]=y
                p["early"].append(self.residual(l,x[:n,-1:],clone)[:,0].cpu())
                p["early_dose"].append({key:val.cpu() for key,val in z.items()})
        if design is not None and i==design["source"][0]+1:
            donor=self.slot_vectors(v[n:],slots)[:,:,0]
            source=[j.cuda() for j in p.pop("early")]
            dsrc=[{k:v.cuda() for k,v in z.items()} for z in p.pop("early_dose")]
            r=self.read_receiver(l,source,(x[:n,-1],qq,kk,vv,o[:n],donor,dsrc,last_cs),vis,slots,
                                 design,variants,discover=mode=="receiver")
            if mode=="receiver":
                p["receiver"]=r
                return
            branch,p["record"],p["alpha"],p["values"]=r
            p["arms"]={j:y.cpu() for j,y in branch.items()}
        elif p["arms"]:
            for j,b in p["arms"].items():
                y=b.cuda()
                q1,k1,v1=self.branch_qkv(l,y,kk,vv,last_cs)
                p["arms"][j]=self.residual(l,y[:,None],self.attention(l,q1,k1,v1,vis))[:,0].cpu()
        ans=self.residual(l,x,o)
        if mode=="behavior" or (design is not None and i>=design["source"][0]+1):
            ans=ans[:n]
        p["x"]=ans.cpu()


def checked_packs(key,n_family):
    p=Study.verify()
    take=set(p["materials"][key]["compare_families"][:n_family])
    rows=A.load(p["materials"][key]["name"])
    behavior=A.load(f"q4-behavior-{key}.json")
    batch=[(r,o) for r,o in zip(rows,behavior["rows"],strict=True) if r["family"] in take]
    return [([x[0] for x in b],[x[1] for x in b]) for b in batched(batch,BATCH)]


def run():
    import torch
    assert (C.OUT/"q4-confirm.json").exists()
    assert A.load("q4-confirm.json")["status"]=="finished"
    p=Study.verify()
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    def progress(stage,j,n,t0,**attrs):
        R.Base.Previous.stage_log("q14",stage,j,n,t0,**attrs)
    t0=time.monotonic()
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        progress("weight-start",0,1,t0)
        m=Engine()
        progress("weight-ready",1,1,t0,resident=m.bytes)
        try:
            if not (C.OUT/"q14-plan.json").exists():
                identity=Study.model_identity("q14")
                A.save("q14-plan.json",dict(parent=Study.sha("plan.json"),sources={Path(__file__).name:A.digest(Path(__file__))},
                    model=identity,stamp=C.stamp(),condition="4B observed answer prefix exactly token-matched where available. Canonical fallback separately flagged by 4B behavior. 14B conditional Code and exact greedy prefix+Code evaluated.",
                    behavior_gate="Conditional P/Q >=90% in all eight cells, use all-eligible eight-view complete families.",
                    search=dict(source_layers=SOURCE_LAYERS,source_score="mean original O write norm at natural common target",next_receivers="top two next-layer Q/K attention changes under original source Value edit, distinct GQA groups",
                                budget="16 discovery families and 48 confirmation families; batch 16. Layer-major original weights, CPU activation staging.")))
            qp=A.load("q14-plan.json")
            assert qp["parent"]==Study.sha("plan.json") and qp["sources"][Path(__file__).name]==A.digest(Path(__file__))
            discover=checked_packs("discover",16)
            confirm=checked_packs("confirm",48)
            if not (C.OUT/"q14-calibration.json").exists():
                r,o=discover[0]
                t0=time.monotonic()
                progress("q14-stock-check",0,1,t0)
                A.save("q14-calibration.json",m.native_check(r[:2],o[:2]))
                progress("q14-stock-check",1,1,t0)
            if not (C.OUT/"q14-selection.json").exists():
                assert not (C.OUT/"q14-behavior.json").exists()
                scans={}
                for data in m.pipeline(discover,"source",progress):
                    for k,val in data.items():
                        scans.setdefault(k,[]).append(val)
                axes={k:np.concatenate(v).mean(0) for k,v in scans.items()}
                supply=max(axes,key=lambda k:axes[k].max())
                h=int(axes[supply].argmax())
                target=[]
                alpha=[]
                for b,p0 in m.pipeline(discover,"receiver",progress,design=dict(source=(supply,h))):
                    target.append(b)
                    alpha.append(p0)
                w=np.concatenate(target).mean((0,1))**.5
                ranking=np.argsort(-w,kind="stable")
                first=int(ranking[0])
                second=next(int(x) for x in ranking if x//m.groups!=first//m.groups)
                A.save("q14-selection.json",dict(stamp=C.stamp(),source=(supply,h),receiver=(first,second),receiver_layer=supply+1,
                                                source_scores={k.tolist() if hasattr(k,"tolist") else str(k):v.tolist() for k,v in axes.items()},
                                                receiver_score=w.tolist(),receiver_median_alpha=np.median(np.concatenate(alpha),axis=0).tolist(),
                                                plan=Study.sha("q14-plan.json")))
            chosen=A.load("q14-selection.json")
            assert chosen["plan"]==Study.sha("q14-plan.json")
            methods=(*Q.METHODS,"dynamic")
            if not (C.OUT/"freeze-q14.json").exists():
                found={}
                dst=m.pipeline(discover,"discover-precision",progress,design=chosen,variants=methods,check_prefix=False)
                for j,((r,o),(records,out)) in enumerate(zip(discover,dst,strict=True),1):
                    name=f"q14-discover-{j:03d}.npz"
                    if not (C.OUT/name).exists():
                        records["good"]=np.ones(len(r),bool)
                        Study.write_np(name,Study.arms_summary(records,out,p["dose"]["limits"]))
                    found[name]=Study.sha(name)
                del dst
                primary,score=json_counts(Study.load_data(dict(shards=found)))
                A.save("freeze-q14.json",dict(stamp=C.stamp(),primary=primary,choices=score,shards=found,selection=Study.sha("q14-selection.json")))
            frozen=A.load("freeze-q14.json")
            assert frozen["selection"]==Study.sha("q14-selection.json")
            if not (C.OUT/"q14-behavior.json").exists():
                seen=[x for slab in m.pipeline(confirm,"behavior",progress) for x in slab]
                roster=[r for b,_ in confirm for r in b]
                groups={}
                family={}
                for r,o in zip(roster,seen,strict=True):
                    assert r["id"]==o["id"]
                    key=f"{r['context']}{r['order']}{r['identity']}"
                    groups.setdefault(key,[]).append(o["conditional"])
                    family.setdefault(r["family"],[]).append(o["conditional"])
                checks={k:dict(n=len(v),correct=sum(v),passed=bool(np.mean(v)>=.90)) for k,v in groups.items()}
                A.save("q14-behavior.json",dict(rows=seen,cells=checks,eligible=[k for k,v in family.items() if all(v)],
                                             passed=all(z["passed"] for z in checks.values()),freeze=Study.sha("freeze-q14.json")))
            tape=A.load("q14-behavior.json")
            assert tape["freeze"]==Study.sha("freeze-q14.json")
            if not tape["passed"]:
                A.save("q14-confirm.json",dict(status="behavior-gate-failed",shards={},behavior=Study.sha("q14-behavior.json")))
                return
            written={}
            allowed=set(tape["eligible"])
            dst=m.pipeline(confirm,"confirm-precision",progress,design=chosen,variants=methods,check_prefix=False)
            for j,((r,o),(records,out)) in enumerate(zip(confirm,dst,strict=True),1):
                name=f"q14-confirm-{j:03d}.npz"
                if not (C.OUT/name).exists():
                    records["good"]=np.array([x["family"] in allowed for x in r])
                    Study.write_np(name,Study.arms_summary(records,out,p["dose"]["limits"]))
                written[name]=Study.sha(name)
            del dst
            A.save("q14-confirm.json",dict(status="finished",shards=written,behavior=Study.sha("q14-behavior.json"),freeze=Study.sha("freeze-q14.json")))
        finally:
            m.close()


if __name__=="__main__":
    C.setup()
    run()
