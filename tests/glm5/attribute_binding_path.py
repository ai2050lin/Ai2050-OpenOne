"""Archive exact BF16 KV slots and inspect upstream of qualified B30 groups."""
import argparse
import base64
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_1samp

import attribute_binding_benchmark as B
import attribute_binding_circuit as C
import mlp_factor_steer_common as prior
from mlp_fidelity_v2_tail import capability_interval
from mlp_functional_trace import load_model
from mlp_read_route_causal import branch, changed
from shared.dist_protocol.mechanism_trace import GpuLease
from shared.dist_protocol.transcoder import TopKTranscoder


def bits(value):
    a=value.detach().contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False)
    return {'shape':list(a.shape),'dtype':'bf16','encoding':'base64-u16-le',
            'values':base64.b64encode(a.tobytes()).decode('ascii')}


def kv_sample(model, tok, eos, tc, data, selected, row, expected):
    x,other,ij=C.batch_inputs([row]);bank,cache=C.prefix(model,x,ij,True)
    donor,other_cache=C.prefix(model,other,ij,False);del other_cache
    clean,clean_seq=branch(model,x,cache,eos)
    weight=model.model.layers[B.BLOCK].mlp.down_proj.weight.detach().T.float()
    vectors=C.edited_vectors(*C.delta_bank(bank,donor,tc),weight,tc.W_dec*tc.y_scale,selected)
    output={'id':row['id'],'positions':row['sites'],'selected_slot':row['selected_slot'],'methods':{}}
    for m,vec in vectors.items():
        edited,_=C.changed_cache(model,cache,bank,ij,vec)
        actual,actual_seq=branch(model,x,edited,eos)
        forward,reverse,mask=changed(cache,edited)
        replay,rseq=branch(model,x,cache,eos,forward);restore,sseq=branch(model,x,edited,eos,reverse)
        replayed=bool(torch.equal(replay,actual) and rseq==actual_seq)
        restored=bool(torch.equal(restore,clean) and sseq==clean_seq)
        fields={}
        for (block,name),changed_tokens in mask.items():
            positions=sorted(set(row['sites']+[int(changed_tokens[0].nonzero()[0])]))
            first=getattr(cache.layers[block],name)[0,:,positions]
            final=getattr(edited.layers[block],name)[0,:,positions]
            fields[f'{block}:{name}']={'positions':positions,'changed_slots':changed_tokens[0].nonzero().flatten().tolist(),
                                      'clean':bits(first),'edited':bits(final)}
        output['methods'][m]={'replay_exact':replayed,'restore_exact':restored,'fields':fields,
            'clean_text':tok.decode(clean_seq[0]),'edited_text':tok.decode(actual_seq[0]),
            'heldout_generation_word_agrees':B.first_word(tok.decode(actual_seq[0]))==expected['methods'][m]['doses']['1.0']['word']}
        del edited
    return output


@contextmanager
def head_states(model, ij, delta=None, head=None):
    module=model.model.layers[B.BLOCK].self_attn
    output={}

    def sample(_,args):
        x=args[0];h=x[ij].reshape(len(ij[0]),3,32,128)
        output['values']=h.detach().clone()
        if delta is not None:
            new=x.clone();h=h.clone();h[:,:,head]=(h[:,:,head].float()+delta).to(x.dtype);new[ij]=h.flatten(-2)
            return (new,*args[1:])

    def query(_,args,value):
        output['q_norm']=value[ij].detach().clone()

    h=module.o_proj.register_forward_pre_hook(sample);q=module.q_norm.register_forward_hook(query)
    try:
        yield output
    finally:
        h.remove();q.remove()


def head_fields(model,x,ij,delta=None,head=None):
    with head_states(model,ij,delta,head) as h:
        bank,cache=C.prefix(model,x,ij,False)
    return h,bank,cache


def fit_head(model, discovery, ids):
    layer=model.model.layers[B.BLOCK]
    w=layer.self_attn.o_proj.weight.T.float().view(32,128,2560)
    scores=[]
    for i in range(0,32,4):
        x,other,ij=C.batch_inputs(discovery[i:i+4]);h,a,cache=head_fields(model,x,ij);del cache
        d,b,cache=head_fields(model,other,ij);del cache
        direction=(b['a']-a['a']).gather(-1,ids.expand(len(x),-1,-1))
        change=torch.einsum('bshc,hcd->bshd',(d['values']-h['values']).float(),w)
        r=(a['residual'].float()[:,:,None]+change).to(torch.bfloat16)
        z=layer.post_attention_layernorm(r)
        act=(layer.mlp.act_fn(layer.mlp.gate_proj(z))*layer.mlp.up_proj(z)).float()
        act=act.gather(-1,ids[None,:,None].expand(len(x),-1,32,-1))
        difference=act-a['a'].gather(-1,ids.expand(len(x),-1,-1))[:,:,None]
        scores.append((difference*direction[:,:,None]).sum(-1).mean(1))
    strength=torch.cat(scores).mean(0)
    return int(strength.argmax()),strength.tolist()


def inspect_upstream(model,tok,eos,data,eligible,selection,out):
    methods=[m for m in ('native','attribution') if any(m in v for v in eligible.values())]
    if not methods:
        return [],'no_qualified_native_group'
    chosen=methods[0];ids=selection[chosen]
    groups=[s for s,v in eligible.items() if chosen in v]
    discovery=[r for r in data['rows'] if r['split']=='discovery']
    head,rank=fit_head(model,discovery,ids)
    # Additional documents generated only after the subgroup is frozen, at a separate seed index.
    frozen={'method':chosen,'head':head,'scores':rank,'source_n':32,'fresh_n_per_split':32,'splits':groups,
            'seed_offsets':[956,1195,1434], 'gate':'difference in native group alignment vs sign/permutation control; alpha .05/3; diagnostic only',
            'source_sha256':prior.file_sha(__file__)}
    prior.save(out/'upstream-freeze.json',frozen)
    material=B.make_material(tok,{**{f'offset{i}':0 for i in range(4)},**{s:32 for s in groups}})
    rows=material['rows'];result=[];transform=C.control_transform('cuda',128)
    for lo in range(0,len(rows),4):
        batch=rows[lo:lo+4];x,other,ij=C.batch_inputs(batch)
        h,a,cache=head_fields(model,x,ij);original,words=branch(model,x,cache,eos)
        d,b,other_cache=head_fields(model,other,ij);del other_cache
        delta=(d['values']-h['values'])[:,:,head].float()
        _,edited,ec=head_fields(model,x,ij,delta,head);change,seq=branch(model,x,ec,eos);del ec
        _,matched,mc=head_fields(model,x,ij,transform(delta),head);ctrl,cseq=branch(model,x,mc,eos);del mc
        direction=(b['a']-a['a']).gather(-1,ids.expand(len(x),-1,-1))
        alignment=lambda bank:((bank['a']-a['a']).gather(-1,ids.expand(len(x),-1,-1))*direction).sum((-1,-2))
        vals=(alignment(edited)-alignment(matched)).tolist()
        for i,r in enumerate(batch):
            result.append({'id':'upstream-'+r['id'],'split':r['split'],'alignment_advantage':vals[i],
                'word':[B.first_word(tok.decode(q[i])) for q in (words,seq,cseq)],
                'margin':[float(C.logits_gap(z[i:i+1],[r],data['surfaces'])[0]) for z in (original,change,ctrl)]})
        if lo==0:
            attn=model.model.layers[B.BLOCK].self_attn;pos=ij[1][0]
            q=h['q_norm'][0,:,head]
            cos,sin=model.model.rotary_emb(q[None],pos[None]);rot=torch.cat((-q[:,64:],q[:,:64]),-1)
            q_rot=(q*cos[0]+rot*sin[0]);k=cache.layers[B.BLOCK].keys[0,head//4]
            dot=q_rot.float()@k.float().T*attn.scaling
            dot.masked_fill_(torch.arange(B.CUT,device=dot.device)[None]>pos[:,None],float('-inf'))
            probs=dot.softmax(-1);p,v=probs.topk(5)
            original_embedding=model.model.embed_tokens(x[0,pos]);donor_embedding=model.model.embed_tokens(other[0,pos])
            frozen['native_example']={'id':batch[0]['id'],'embedding_token_ids':[x[0,pos].tolist(),other[0,pos].tolist()],
                 'embedding_original':bits(original_embedding),'embedding_donor':bits(donor_embedding),
                 'query_positions':pos.tolist(),'RoPE_q':bits(q_rot),'source_positions':v.tolist(),'attention_weights':p.tolist(),
                 'native_pre_o_proj':bits(h['values'][0,:,head])}
        del cache
    score={}
    for split in groups:
        v=[r['alignment_advantage'] for r in result if r['split']==split]
        p=float(ttest_1samp(v,0.,alternative='greater').pvalue)
        score[split]={'n':len(v),'mean':float(np.mean(v)),'p':p if np.isfinite(p) else None}
    return [{**frozen,'diagnostic_summary':score,'cases':result}],'native_group_upstream_diagnostic_completed'


@torch.inference_mode()
def run(out):
    data=prior.read_gz(out/'inputs.json.gz');frozen=prior.read(out/'frozen-candidates.json')
    selection={m:torch.tensor(v,device='cuda') for m,v in frozen['indices'].items()}
    reports={s:prior.read_gz(out/f'{s}.json.gz') for s in list(B.COUNTS)[1:]}
    cap=prior.read_gz(out/'capability.json.gz')['rows'];mask=np.asarray([r['eligible'] for r in cap])
    correct=lambda m:[r['choice'][m]==r['answer'] for r in cap]
    eligible={s:[m for m,v in r['summary']['methods'].items() if v['anchor'] and
                  capability_interval(correct('clean'),correct(m),mask,alpha=.05/3)['status']=='noninferior'] for s,r in reports.items()}
    with GpuLease():
        model,tok,eos=load_model()
        with np.load(B.TC/'transcoder.npz') as arrays:tc=TopKTranscoder.from_state(arrays,device='cuda')
        records={}
        for split,report in reports.items():
            first=report['rows'][0];row=next(r for r in data['rows'] if r['id']==first['id'])
            records[split]=kv_sample(model,tok,eos,tc,data,selection,row,first)
        upstream,status=inspect_upstream(model,tok,eos,data,eligible,selection,out)
    prior.save(out/'native-path.json',{'sampled_native_kv':records,'upstream':upstream,'upstream_status':status,'qualified':eligible,
        'sample_note':'One fixed first case per split. Single-row replay uses a different batch layout than the main eight-row study.'})
    print(status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);run(parser.parse_args().out)
