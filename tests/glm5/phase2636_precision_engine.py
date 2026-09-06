"""Native single-weight arithmetic controls. CPU token/head storage, all blocks on CUDA."""
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr
from phase2624_scalar_forward_validation import digest_tensor
from phase2632_fulltoken_native_adjoints import AdjointCapture, INPUT_KEY, LAYERS, MODULES, module_at
from phase2636_precision_contract import PROBE_SITES,SCALES

SOURCE=RESULT/'phase2632_fulltoken_native_adjoints'

def load_precision(precision):
    assert precision in ('bf16','fp32')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=torch.bfloat16 if precision=='bf16' else torch.float32
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/hf/qwen3-4b',dtype=dtype,device_map={'':'cpu'},
        local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    for p in model.parameters():p.requires_grad_(False)
    model.model.layers.to('cuda:0');model.model.norm.to('cuda:0');model.model.rotary_emb.to('cuda:0')
    assert model.get_input_embeddings().weight.device.type=='cpu' and model.lm_head.weight.device.type=='cpu'
    assert all(next(b.parameters()).device.type=='cuda' for b in model.model.layers)
    checks={}
    for l in LAYERS:
        for name in MODULES:
            expected=np.load(SOURCE/f'field/weights/L{l}_{name}.float32.npy',mmap_mode='r')
            checks[f'L{l}_{name}']=np.array_equal(arr(module_at(model,l,name).weight),expected)
    assert all(checks.values())
    return model,{'precision':precision,'all28_weight_values_exact':checks,'core_cuda_bytes':sum(p.numel()*p.element_size() for p in model.parameters() if p.device.type=='cuda'),
        'device_rule':'embedding/head weights CPU, all36 blocks and final norm CUDA, rotary CUDA; identical in both modes',
        'tf32':False,'quantized':bool(getattr(model,'is_quantized',False))}

def embedding(model,frame,grad=False):
    with torch.no_grad():x=model.get_input_embeddings()(torch.tensor([frame['prefix_ids']],device='cpu')).to('cuda:0')
    return x.detach().requires_grad_(grad)

@torch.inference_mode()
def score(model,frame,contrast):
    state=model.model(inputs_embeds=embedding(model,frame),use_cache=False).last_hidden_state[0,-1]
    return float((state.float()*contrast).sum()),arr(state)

def run_precision(precision,frames,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    completed=out/'analysis/completion.json'
    if completed.exists():
        prior=read(completed)
        assert prior['frame_ids']==[f['frame_id'] for f in frames] and prior['precision']==precision and all(prior['checks'].values())
        return prior['summary'],prior['checks']
    model,info=load_precision(precision);save(out/'protocol/model.json',info)
    old={r['frame_id']:r for r in read(SOURCE/'analysis/records.json')};weight_info=read(SOURCE/'protocol/weights.json')
    before={f'L{l}_{n}':digest_tensor(module_at(model,l,n).weight) for l,n in PROBE_SITES}
    U=model.lm_head.weight.detach().float();records=[];conditions=[];manifest=[]
    oldhidden=np.load(SOURCE/'field/hidden_positions.float32.npy',mmap_mode='r')
    for fi,frame in enumerate(frames):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();fid=frame['frame_id']
        contrast=(U[frame['chosen_id']]-U[frame['runnerup_id']]).to('cuda:0')
        capture=AdjointCapture(model);em=embedding(model,frame,True)
        result=model.model(inputs_embeds=em,use_cache=False);state=result.last_hidden_state[0,-1]
        loss=(state.float()*contrast).sum();loss.backward();baseline=float(loss.detach());base_state=arr(state)
        with torch.no_grad():
            # Common FP32 diagnostic head ranking, not each precision's historical generation interface.
            native_ids=(U@state.detach().cpu().float()).topk(2).indices.tolist()
        pack={};sites=[]
        hs=[em]+capture.hidden
        pack['hidden_boundary']=np.stack([arr(v[0,-1]) for v in hs]);pack['hidden_adjoint_boundary']=np.stack([arr(v.grad[0,-1]) for v in hs])
        pack['mlp_boundary']=np.stack([arr(v[0,-1]) for v in capture.a]);pack['mlp_adjoint_boundary']=np.stack([arr(v.grad[0,-1]) for v in capture.a])
        pack['normalized_boundary']=base_state
        for l in LAYERS:
            for name in MODULES:
                x,y=capture.linears[(l,name)];key=f'L{l}_{name}';xx=arr(x[0]);gg=arr(y.grad[0])
                pack[f'L{l}_{INPUT_KEY[name]}']=xx;pack[key+'__g']=gg;pack[key+'__value']=arr(y[0])
                prior=next(s for s in old[fid]['sites'] if s['layer']==l and s['module']==name)
                for selector in ('diagnostic_max','matched'):
                    j=prior['diagnostic_max_j'] if selector=='diagnostic_max' else prior['matched_index_j']
                    k=prior['diagnostic_max_k'] if selector=='diagnostic_max' else prior['matched_index_k']
                    terms=xx[:,k].astype('float64')*gg[:,j].astype('float64')
                    sites.append({'layer':l,'module':name,'selector':selector,'j':j,'k':k,'gradient_full':float(terms.sum()),'gradient_last':float(terms[-1])})
        path=out/f'field/frame_{fid:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**pack)
        manifest.append({'path':str(path),'bytes':path.stat().st_size,'frame_id':fid})
        records.append({'frame_id':fid,'case_id':frame['case_id'],'family':frame['family'],'language':frame['language'],'variant':frame['variant'],
            'margin':baseline,'original_bf16_margin':old[fid]['fp32_loss'],'common_fp32_head_top2':native_ids,
            'old_native_top2':[frame['chosen_id'],frame['runnerup_id']],
            'old_hidden_max_difference':float(np.max(np.abs(pack['hidden_boundary']-oldhidden[fid,:,2]))),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'sites':sites})
        capture.close();capture.reset();del capture,result,state,loss,em,hs,pack,x,y,xx,gg;gc.collect();torch.cuda.empty_cache()
        with torch.no_grad():
            noop,nstate=score(model,frame,contrast)
            conditions.append({'frame_id':fid,'case_id':frame['case_id'],'family':frame['family'],'language':frame['language'],'kind':'noop',
                'margin_change':noop-baseline,'state_l2_change':float(np.linalg.norm(nstate-base_state))})
            for site in sites:
                l,name=site['layer'],site['module']
                if (l,name) not in PROBE_SITES:continue
                W=module_at(model,l,name).weight;j,k=site['j'],site['k'];rms=weight_info[f'L{l}_{name}']['rms']
                for scale in SCALES:
                    for sign in (-1,1):
                        original=W[j,k].clone();original_value=float(original)
                        target=float(torch.tensor(original_value+sign*scale*rms,dtype=torch.bfloat16).float())
                        try:
                            W[j,k]=target;delta=float(W[j,k])-original_value
                            changed,cstate=score(model,frame,contrast);effect=changed-baseline
                            conditions.append({'frame_id':fid,'case_id':frame['case_id'],'family':frame['family'],'language':frame['language'],
                                'kind':'shared_weight',**site,'scale':scale,'sign':sign,'original_weight':original_value,'target_weight':target,
                                'actual_delta':delta,'effect':effect,'predicted_full':delta*site['gradient_full'],'predicted_last':delta*site['gradient_last'],
                                'state_l2_change':float(np.linalg.norm(cstate-base_state)),'precision':precision})
                        finally:W[j,k].copy_(original)
        save(out/'analysis/progress.json',{'completed_frames':fi+1,'total_frames':len(frames),'conditions':len(conditions)})
        print(precision,'numeric control',fi+1,'/',len(frames),'conditions',len(conditions),flush=True)
    after={f'L{l}_{n}':digest_tensor(module_at(model,l,n).weight) for l,n in PROBE_SITES};assert before==after
    save(out/'analysis/conditions.json',conditions);save(out/'analysis/records.json',records);save(out/'analysis/raw_manifest.json',manifest)
    save(out/'analysis/restoration.json',{'before':before,'after':after,'disk_model_changed':False});save(out/'material/frames.json',frames)
    del model,U;gc.collect();torch.cuda.empty_cache()
    checks={'all_expected_conditions':len(conditions)==len(frames)*73,'all_noops_zero':all(r['margin_change']==0 and r['state_l2_change']==0 for r in conditions if r['kind']=='noop'),
        'all_six_matrices_restored':before==after,'same_weight_values':all(info['all28_weight_values_exact'].values()),
        'bf16_old_hidden_exact':precision!='bf16' or all(r['old_hidden_max_difference']==0 for r in records)}
    summary=summarize(conditions)
    save(completed,{'precision':precision,'frame_ids':[f['frame_id'] for f in frames],'summary':summary,'checks':checks})
    return summary,checks

def summarize(conditions):
    output={}
    for l,name in PROBE_SITES:
        for scale in SCALES:
            rows=[r for r in conditions if r['kind']=='shared_weight' and r['layer']==l and r['module']==name and r['scale']==scale]
            den=sum(abs(r['effect']) for r in rows)
            active=[r for r in rows if abs(r['effect'])>=1e-5]
            output[f'L{l}/{name}/scale{scale}']={'n':len(rows),'zero_delta':sum(r['actual_delta']==0 for r in rows),
                'mean_abs_effect':den/len(rows),'full_l1_error':sum(abs(r['effect']-r['predicted_full']) for r in rows)/den if den else None,
                'last_l1_error':sum(abs(r['effect']-r['predicted_last']) for r in rows)/den if den else None,
                'n_effect_ge_1e-5':len(active),'sign_agreement':float(np.mean([np.sign(r['effect'])==np.sign(r['predicted_full']) for r in active])) if active else None}
    return output
