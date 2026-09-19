"""Actual middle-MLP continued learning of natural content, not prompt-final digits."""
import argparse
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'formation'

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    detailed=set(read(BASE/'material/protocol.json')['detailed_prefix_ids'])
    rows=[r for r in gzread(BASE/'material/natural.json.gz') if r['sample_id'] in detailed]
    train=[];panel=[]
    for r in rows:
      if r['split']=='validation':continue
      for j,pos in enumerate([len(r['prompt_ids'])//2,2*len(r['prompt_ids'])//3]):
        rec={'sample_id':r['sample_id']+f'_content{j}','source_group':r['source_group'],'cohort':r['cohort'],'split':r['split'],
          'ids':r['prompt_ids'][:pos+1],'target':r['prompt_ids'][pos+1],'position':pos,'kind':'natural_content'}
        (train if r['split']=='train' else panel).append(rec)
    # Program scoring is after a real emitted terminal marker, not initial prompt-final.
    events=gzread(BASE/'events/material.json.gz')['trajectories'];digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    missing=[]
    for item in events:
        row=item['row'];native=read(ROOT/item['native_record']);markers=[e['emitted_token_step'] for e in item['events'] if e['type']=='terminal_marker']
        eligible=[j for j,t in enumerate(native['generated_ids']) if markers and j>max(markers) and t in digits]
        if not eligible:missing.append(row['sample_id']);continue
        j=eligible[0]
        panel.append({'sample_id':row['sample_id']+'_actual_terminal','source_group':row['source_group'],'cohort':row['representation'],'split':'old_program_test',
          'ids':row['prompt_ids']+native['generated_ids'][:j],'target':row['target_ids'][0],'position':len(row['prompt_ids'])+j-1,
          'actual_emitted_token_id':native['generated_ids'][j],'kind':'actual_terminal_digit','native_digit_step':j})
    assert len(train)==576 and sum(r['kind']=='natural_content' for r in panel)==384
    # Permutation is within natural language/cohort; it never uses heldout targets.
    perm=np.arange(len(train));rng=np.random.default_rng(2742)
    for c in ['gum','ewt','cmrc']:
        ix=np.array([i for i,r in enumerate(train) if r['cohort']==c]);perm[ix]=rng.permutation(ix)
    for i,r in enumerate(train):r['permuted_target']=train[int(perm[i])]['target']
    value={'timestamp':stamp(),'source':snapshot(__file__),'train_examples':len(train),'test_natural_examples':384,
      'old_actual_terminal_examples':len(panel)-384,'old_terminal_examples_without_frozen_marker':missing,
      'parameters':'All74711040 original block16 gate/up/down scalars; other weights fixed; complete native17..35 suffix and full vocabulary backpropagated.',
      'conditions':['natural_target','within_cohort_permuted_target'],'seeds':[2742,2743],'steps':32,'batch_examples':4,'step_FP32_norm':0.02,'checkpoints':[1,8,32],
      'precision':'Block16 trainable FP32 with BF16 boundary casts; baseline bridge measured separately; original checkpoint immutable; final BF16-deployment separately measured.',
      'material_scope':'Inputs truncated at authentic content boundary. Correct next token is only a training/scoring label, not an input. Old actual terminal prefixes were model-generated and are scoring-only, historically exposed diagnostics.',
      'selection':'All288 detailed train prefixes times2positions; all192 test prefixes times2positions; no correctness selection.',
      'observed_fields':'24 predeclared natural heldout examples, block16gate/up/activation and all final coordinates before/after; noTopK.',
      'uncertainty':'Source-cluster intervals separately for real/permuted targets and both draw orders. No inference to historical pretraining or universal semantic subspace.'}
    compressed(OUT/'material.json.gz',{'train':train,'panel':panel,'permutation':perm.tolist()});immutable(p,value);return value,{'train':train,'panel':panel}

def loss(model,row,target_key='target'):
    import torch
    post=model.model(input_ids=torch.tensor([row['ids']],device='cuda'),use_cache=False).last_hidden_state[0,-1]
    logits=model.lm_head(post).float();lp=logits.double().log_softmax(-1)
    return -lp[row[target_key]],logits

def evaluate(model,rows):
    import torch
    values=[];top=[]
    with torch.no_grad():
        for r in rows:
            ll,z=loss(model,r);values.append(float(ll));top.append(int(z.argmax()))
    return {'loss':np.array(values),'argmax':np.array(top)}

def report(rows,packet,baseline):
    result=[]
    for kind,cohort in sorted({(r['kind'],r['cohort']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['kind'],r['cohort'])==(kind,cohort)]
        result.append({'kind':kind,'cohort':cohort,'examples':len(ix),'loss_delta':clustered(packet['loss'][ix]-baseline['loss'][ix],[rows[i]['source_group'] for i in ix]),
          'argmax_accuracy':float(np.mean(packet['argmax'][ix]==[rows[i]['target'] for i in ix]))})
    return result

def observe(model,rows,file):
    import torch
    data={};layer=model.model.layers[16];handles=[]
    for name in ['gate_proj','up_proj']:
        handles.append(getattr(layer.mlp,name).register_forward_hook(lambda m,a,o,name=name:data.__setitem__(name,o[0,-1].detach().float().cpu().numpy())))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:data.__setitem__('activation',a[0][0,-1].detach().float().cpu().numpy())))
    collected=defaultdict(list)
    try:
      with torch.no_grad():
        for r in rows:
            v=model.model(input_ids=torch.tensor([r['ids']],device='cuda'),use_cache=False)
            for k,a in data.items():collected[k].append(a)
            collected['postnorm'].append(bits(v.last_hidden_state[0,-1]))
    finally:
        for h in handles:h.remove()
    npz(file,**{k:np.stack(v) for k,v in collected.items()})

def main(pilot=False):
    import torch
    from torch.utils.checkpoint import checkpoint
    protocol,data=freeze();train=data['train'];panel=data['panel'];small=panel[:4] if pilot else panel
    path=OUT/('pilot.json' if pilot else 'result.json')
    if path.exists():return
    if not pilot:assert read(OUT/'pilot.json')['all_passed']
    start=time.monotonic();model=None;handles=[];guard(1600*1024**2)
    try:
      model,tok=load('qwen4',OUT)
      for p in model.parameters():p.requires_grad_(False)
      native=evaluate(model,small);target=model.model.layers[16].mlp
      original={n:p.detach().float().cpu().clone() for n,p in target.named_parameters()};target.float();named=dict(target.named_parameters());params=list(named.values())
      for p in params:p.requires_grad_(True)
      handles=[target.register_forward_pre_hook(lambda m,a:(a[0].float(),)+a[1:]),target.register_forward_hook(lambda m,a,o:o.to(torch.bfloat16))]
      for layer in model.model.layers[17:]:
        original_forward=layer.forward
        def forward(*args,_forward=original_forward,**kwargs):
            return checkpoint(_forward,*args,use_reentrant=False,**kwargs) if torch.is_grad_enabled() else _forward(*args,**kwargs)
        layer.forward=forward
      bridge=evaluate(model,small);tick=time.monotonic();ll,z=loss(model,train[0]);gradient=torch.autograd.grad(ll,params);torch.cuda.synchronize()
      gradient_seconds=time.monotonic()-tick;assert all(torch.isfinite(g).all() for g in gradient);del gradient,ll,z
      if pilot:
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':gradient_seconds*512+800<7200,'backward_seconds':gradient_seconds,
          'projected_seconds':gradient_seconds*512+800,'bridge_minus_native_loss':float((bridge['loss']-native['loss']).mean()),
          'train_examples':len(train),'panel_examples':len(panel),'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        save(path,result);ledger('natural_formation_pilot',result['seconds']);print('NATURAL_FORMATION_PILOT',result,flush=True);return
      npz(OUT/'native_baseline.npz',**native);npz(OUT/'bridge_baseline.npz',**bridge)
      field_rows=[r for c in ['gum','ewt','cmrc'] for r in [x for x in panel if x['cohort']==c and x['kind']=='natural_content'][:8]]
      save(OUT/'field_rows.json',[r['sample_id'] for r in field_rows]);observe(model,field_rows,OUT/'bridge_fields.npz');runs=[]
      for seed in protocol['seeds']:
        draws=np.random.default_rng(seed).integers(len(train),size=(32,4));npz(OUT/f'draws_{seed}.npz',indices=draws)
        for condition in protocol['conditions']:
          folder=OUT/f'{condition}_{seed}'
          if (folder/'result.json').exists():runs.append(read(folder/'result.json'));continue
          with torch.no_grad():
            for n,p in named.items():p.copy_(original[n].to(p.device))
          trace=[];checkpoints=[];runstart=time.monotonic()
          for step,indices in enumerate(draws,1):
            acc=[torch.zeros_like(p) for p in params];losses=[];key='target' if condition=='natural_target' else 'permuted_target'
            for index in indices:
                ll,z=loss(model,train[index],key);grads=torch.autograd.grad(ll,params);losses.append(float(ll.detach()))
                for a,g in zip(acc,grads):a.add_(g,alpha=.25)
                del ll,z,grads
            norm=torch.stack([a.square().sum() for a in acc]).sum().sqrt();assert norm>0 and torch.isfinite(norm)
            with torch.no_grad():
                for p,a in zip(params,acc):p.add_(a,alpha=-.02/float(norm))
            trace.append({'step':step,'drawn_loss':float(np.mean(losses)),'full_gradient_norm':float(norm),'parameter_step_norm':.02})
            del acc
            if step in protocol['checkpoints']:
                packet=evaluate(model,panel);npz(folder/f'checkpoint{step}.npz',**packet);checkpoints.append({'step':step,'reports':report(panel,packet,bridge)})
                save(folder/'progress.json',{'trace':trace,'checkpoints':checkpoints});print('NATURAL_FORMATION',condition,seed,step,round(time.monotonic()-runstart,1),flush=True)
            assert time.monotonic()-start<7200
          observe(model,field_rows,folder/'final_FP32_bridge_fields.npz')
          delta={n:p.detach().cpu().numpy()-original[n].numpy() for n,p in named.items()};npz(folder/'parameter_delta_FP32.npz',**delta)
          normdelta=float(np.sqrt(sum(np.sum(v.astype(float)**2) for v in delta.values())))
          # Evaluate true BF16 scalar deployment while preserving the FP32 trained state.
          final={n:p.detach().cpu().clone() for n,p in named.items()}
          with torch.no_grad():
            for n,p in named.items():p.copy_(p.to(torch.bfloat16).float())
          for h in handles:h.remove()
          handles=[];target.bfloat16()
          deployed=evaluate(model,panel);npz(folder/'deployed_BF16.npz',**deployed);observe(model,field_rows,folder/'deployed_BF16_fields.npz')
          deployed_norm=float(np.sqrt(sum(float((p.detach().float().cpu()-original[n]).double().square().sum()) for n,p in named.items())))
          target.float();handles=[target.register_forward_pre_hook(lambda m,a:(a[0].float(),)+a[1:]),target.register_forward_hook(lambda m,a,o:o.to(torch.bfloat16))]
          run={'condition':condition,'seed':seed,'trace':trace,'checkpoints':checkpoints,'deployed_reports_vs_native':report(panel,deployed,native),
            'delta_FP32_norm':normdelta,'delta_native_BF16_norm':deployed_norm,'delta_sha256':sha(folder/'parameter_delta_FP32.npz'),
            'seconds':time.monotonic()-runstart,'distinct_drawn_examples':len(set(draws.ravel().tolist()))}
          save(folder/'result.json',run);runs.append(run);del delta,final;guard()
      with torch.no_grad():
        for n,p in named.items():p.copy_(original[n].to(p.device))
      reset=evaluate(model,panel[:4]);assert np.array_equal(reset['loss'],bridge['loss'][:4])
      result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'runs':runs,'seconds':time.monotonic()-start,
        'original_checkpoint_unmodified':True,'restored_bridge_exact_first4':True,'scope':'Actual restricted continued training of a pretrained block, not the historical emergence of language. Terminal-digit scoring uses actual old generated prefixes and does not grade a newly generated reasoning chain.'}
      save(path,result);ledger('natural_content_actual_formation',result['seconds']);print('NATURAL_FORMATION_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.freeze:print('FORMATION_PROTOCOL',freeze()[0])
    else:main(a.pilot)
