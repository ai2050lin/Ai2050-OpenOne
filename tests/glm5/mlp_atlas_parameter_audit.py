"""Whole-unit parameter provenance and symmetric product identity, not semantic decoding."""
import sys,time,gc,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as C
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits
from shared.dist_protocol.natural_register_runner import snapshot
ROOT=C.ROOT;OUT=ROOT/'tests/glm5/result/mlp_atlas_parameter_audit_v1';PUBLIC=ROOT/'research/mlp_atlas/parameter_audit_v1';RUN='MLP-ATLAS-PARAMETER-20261104-v1'

def main():
    import torch
    from server.mechanism_trace_service import parameter
    from mlp_atlas_natural_collect import status as unused
    man=C.read(C.OUT/'bank/manifest.json');assert man['state']=='complete';OUT.mkdir(exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    plan={'schema':'mlp_parameter_audit_plan.v1','kind':'real','run_id':RUN,'model_snapshot':man['model_snapshot'],'provenance':man['provenance'],'source_bank_manifest_sha256':file_sha(C.OUT/'bank/manifest.json'),'script_sha256':file_sha(__file__),
        'scope':'Exploratory arithmetic audit after code-response results; no new confirmatory semantic claim, no weights edited, no causal promotion.',
        'n_inputs':9984,'readouts':2,'all_native_units':350208,'projection':'Actual BF16 x @ W_gate/up, SiLU(G)*V, and A @ W_down, compared against captured native arrays; all coordinates retained. Batch shape may change BF16 rounding.',
        'product_identity':'For two observed contexts 1,0: S1*V1-S0*V0 = .5*(S1-S0)*(V1+V0)+.5*(V1-V0)*(S1+S0), S=BF16 SiLU(G). Apply the registered four-cell contrast. Quantized A residual kept explicitly.',
        'interpretation':'An exact symmetric algebraic allocation, not a unique causal decomposition and not capacity-times-gating theory. Gate/Value are implementation branches, not semantic categories.'}
    C.save(PUBLIC/'prereg.research.json',plan);start=time.time();layers=[];torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(C.MODEL)==man['model_snapshot']
        for b in range(36):
            maps={k:np.load(C.OUT/f'bank/B{b:02d}-{k}.npy',mmap_mode='r') for k in ['G','V','A','input','output']};prefix=f'model.layers.{b}.mlp.'
            weights={k:torch.from_numpy(parameter(C.MODEL,prefix+name+'.weight',(slice(None),slice(None)))).to('cuda',dtype=torch.bfloat16) for k,name in [('G','gate_proj'),('V','up_proj'),('output','down_proj')]}
            accum={k:{'sse':np.zeros(9728 if k!='output' else 2560),'s2':np.zeros(9728 if k!='output' else 2560),'max_abs':np.zeros(9728 if k!='output' else 2560),'exact':np.zeros(9728 if k!='output' else 2560,dtype=np.int64)} for k in ['G','V','A','output']}
            n=9984*2
            with torch.inference_mode():
                for lo in range(0,n,512):
                    hi=min(n,lo+512);x=torch.from_numpy(np.asarray(maps['input']).reshape(n,2560)[lo:hi].copy()).view(torch.bfloat16).to('cuda');gg=torch.nn.functional.linear(x,weights['G']);vv=torch.nn.functional.linear(x,weights['V']);aa=torch.nn.functional.silu(gg)*vv
                    # Down-projection receives recorded A to isolate its source from upstream rounding.
                    original_a=torch.from_numpy(np.asarray(maps['A']).reshape(n,9728)[lo:hi].copy()).view(torch.bfloat16).to('cuda');yy=torch.nn.functional.linear(original_a,weights['output'])
                    for key,pred in [('G',gg),('V',vv),('A',aa),('output',yy)]:
                        observed=unbits(np.asarray(maps[key]).reshape(n,-1)[lo:hi]);p=pred.float().cpu().numpy();err=(p-observed).astype(np.float64);r=accum[key];r['sse']+=(err*err).sum(0);r['s2']+=(observed.astype(np.float64)**2).sum(0);r['max_abs']=np.maximum(r['max_abs'],np.abs(err).max(0));r['exact']+=(p==observed).sum(0)
            arrays={}
            for key,r in accum.items():
                arrays[key+'_relative_RMS']=np.sqrt(r['sse']/np.maximum(r['s2'],1e-20));arrays[key+'_absolute_RMS']=np.sqrt(r['sse']/n);arrays[key+'_max_abs']=r['max_abs'];arrays[key+'_exact_fraction']=r['exact']/n;arrays[key+'_zero_reference_energy']=r['s2']<=1e-20
            # Use actual observed branches, not recomputed projections, for the identity.
            S=torch.nn.functional.silu(torch.from_numpy(np.asarray(maps['G']).copy()).view(torch.bfloat16)).float().numpy().reshape(416,3,2,2,2,2,9728)
            V=unbits(maps['V']).reshape(S.shape);actual=unbits(maps['A']).reshape(S.shape)
            def alloc(t1,q1,t0,q0):
                s1,s0=S[:,:,:,t1,q1],S[:,:,:,t0,q0];v1,v0=V[:,:,:,t1,q1],V[:,:,:,t0,q0]
                return .5*(s1-s0)*(v1+v0),.5*(v1-v0)*(s1+s0)
            g0,v0=alloc(0,0,1,0);g1,v1=alloc(1,1,0,1);gt=.5*(g0+g1);vt=.5*(v0+v1);delta=.5*(actual[:,:,:,0,0]-actual[:,:,:,1,0]+actual[:,:,:,1,1]-actual[:,:,:,0,1]);residual=delta-gt-vt
            for key,a in [('Gate_term',gt),('Value_term',vt),('rounding_residual',residual)]:
                arrays[key+'_split_code_mapping_mean']=np.stack([a[lo:hi].mean(0,dtype=np.float64) for lo,hi in [(0,32),(32,224),(224,416)]]).astype(np.float32)
                arrays[key+'_split_code_mapping_RMS']=np.stack([np.sqrt((a[lo:hi].astype(np.float64)**2).mean(0)) for lo,hi in [(0,32),(32,224),(224,416)]]).astype(np.float32)
                arrays[key+'_RMS']=np.sqrt((a.astype(np.float64)**2).mean((0,1,2))).astype(np.float32)
            for key,w in weights.items():arrays[key+'_weight_axis_norm']=w.float().norm(dim=0 if key=='output' else 1).cpu().numpy()
            path=OUT/f'B{b:02d}.npz';np.savez_compressed(path,**arrays);row={'block':b,'sha256':file_sha(path),'projection_max_relative_RMS':{k:float(arrays[k+'_relative_RMS'].max()) for k in accum},'fraction_every_value_exact':{k:float((arrays[k+'_exact_fraction']==1).mean()) for k in accum},'all_9728_units_checked':True};layers.append(row);print(json.dumps({'block':b,**row}),flush=True)
            del maps,weights,S,V,actual,g0,v0,g1,v1,gt,vt,delta,residual,arrays;gc.collect();torch.cuda.empty_cache()
        assert snapshot(C.MODEL)==man['model_snapshot']
    C.save(OUT/'audit-result.json',{'schema':'mlp_parameter_audit_report.v1','kind':'real','run_id':RUN,'model_snapshot':man['model_snapshot'],'provenance':man['provenance'],'n_inputs':9984,'all_native_units':350208,'readouts':2,'layers':layers,'plan':plan,'seconds':time.time()-start,'semantic_causal_closure':False});print(json.dumps({'done':36,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
