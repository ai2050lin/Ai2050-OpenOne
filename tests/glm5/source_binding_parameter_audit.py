"""Independently replay native parameter projections from stored incoming states."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime
from source_binding_analysis import verify_code

def run(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    verify_code();D.admission(model,split);root=D.OUT/model;c=D.read(root/'candidates.json');manifest=D.read(root/f'{split}-native-manifest.json')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];folder=root/f'{split}-native'
    arrays={n:np.load(folder/f'{n}.npy',mmap_mode='r') for n in ['hidden','query','key','vproj','gate','value','units','mlp_input']}
    def tensor(a):return torch.from_numpy(np.array(a)).view(torch.bfloat16).cuda()
    def rel(a,b):return ((a.float()-b.float()).flatten(1).norm(dim=1)/b.float().flatten(1).norm(dim=1).clamp_min(1e-8)).cpu().numpy()
    results=[];start=time.monotonic()
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model)
        for candidate in c['heads']:
            b,h=candidate['block'],candidate['head'];bi=manifest['head_blocks'].index(b);layer=rt.block(b);errs={k:[] for k in ['Q','K','V']}
            for lo in range(0,len(rows),128):
                rs=rows[lo:lo+128];x=tensor(arrays['hidden'][lo:lo+len(rs),b]);x=layer.input_layernorm(x)
                q=layer.self_attn.q_proj(x).reshape(len(rs),5,rt.heads,128);k=layer.self_attn.k_proj(x).reshape(len(rs),5,rt.kv,128)
                if hasattr(layer.self_attn,'q_norm'):q=layer.self_attn.q_norm(q);k=layer.self_attn.k_norm(k)
                v=layer.self_attn.v_proj(x).reshape(len(rs),5,rt.kv,128).transpose(1,2);q=q.transpose(1,2);k=k.transpose(1,2)
                pos=torch.tensor([r['positions'] for r in rs],device='cuda');rot=rt.Rotary(rt.cfg,device='cuda');pe=rot(x,pos);q,k=rt.mod.apply_rotary_pos_emb(q,k,*pe)
                q0=tensor(arrays['query'][lo:lo+len(rs),bi,h]);k0=[];v0=[]
                for i,r in enumerate(rs):
                    idx=np.asarray(r['positions'])+manifest['length']-len(r['tokens']);k0.append(np.asarray(arrays['key'][lo+i,bi,candidate['kv_head'],idx]));v0.append(np.asarray(arrays['vproj'][lo+i,bi,candidate['kv_head'],idx]))
                errs['Q'].extend(rel(q[:,h,-1],q0).tolist());errs['K'].extend(rel(k[:,candidate['kv_head']],tensor(np.stack(k0))).tolist());errs['V'].extend(rel(v[:,candidate['kv_head']],tensor(np.stack(v0))).tolist())
            results.append({'id':f'B{b}H{h}','incoming_positions':'five stored native positions; Q evaluated final position, K/V allfive',
                'max_relative_error':{k:max(v) for k,v in errs.items()},'passed':all(max(v)<.02 for v in errs.values()),
                'parameter_modules':[f'model.layers.{b}.input_layernorm',f'model.layers.{b}.self_attn.q_proj',f'model.layers.{b}.self_attn.k_proj',f'model.layers.{b}.self_attn.v_proj']})
            del layer;gc.collect();torch.cuda.empty_cache();D.log(model+'/'+split+'/parameter-replay',len(results),6,start)
        for candidate in c['mlps']:
            b=candidate['block'];bi=manifest['mlp_blocks'].index(b);layer=rt.block(b);indices=candidate['selected'];errs={k:[] for k in ['G','V','A']}
            for lo in range(0,len(rows),128):
                hi=min(lo+128,len(rows));x=tensor(arrays['mlp_input'][lo:hi,bi])
                if hasattr(layer.mlp,'gate_up_proj'):g,v=layer.mlp.gate_up_proj(x).chunk(2,-1)
                else:g=layer.mlp.gate_proj(x);v=layer.mlp.up_proj(x)
                a=torch.nn.functional.silu(g)*v
                for name,actual,stored in [('G',g,'gate'),('V',v,'value'),('A',a,'units')]:errs[name].extend(rel(actual[:,indices],tensor(arrays[stored][lo:hi,bi][:,indices])).tolist())
            results.append({'id':f'B{b}MLP','incoming_positions':'actual normalized postattention input, selected256unit G/V/A replay',
                'max_relative_error':{k:max(v) for k,v in errs.items()},'passed':all(max(v)<.02 for v in errs.values()),
                'parameter_module':f'model.layers.{b}.mlp','selected_units':indices})
            del layer;gc.collect();torch.cuda.empty_cache();D.log(model+'/'+split+'/parameter-replay',len(results),6,start)
    D.save(f'{model}/{split}-parameter-replay.json',{'checks':results,'passed':all(x['passed'] for x in results),
        'code_sha256':D.sha(Path(__file__)),'scope':'native parameter lineage calibration, not new mechanistic evidence or causal sufficiency'},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);a=ap.parse_args();run(a.model,a.split)
