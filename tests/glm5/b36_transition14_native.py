"""Conditional B36 source-position/path study; refuses execution before behavior admission."""
import sys,time,gc,itertools
import numpy as np
import b36_transition14 as M
from b36_transition14_native_math import partitions,subset_mask,locate_sites
S=M.S
ARMS=[f'mask{i}' for i in range(8)]+['restore','matched','random_norm','hold_attention','hold_mlp','hold_both','history_hold_attention']
LENGTH=256;BATCH=4

def sources():
    names=['b36_transition14_native.py','b36_transition14_native_analyze.py','b36_transition14_native_math.py','test_b36_transition14_native_math.py','b36_transition14.py','b36_semantic14.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']
    return {n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in names}

def prepare():
    S.configure(750*2**20);a=S.read(S.OUT/'analysis.json');assert a['gate_passed'],'Behavior gate not passed'
    assert not (S.OUT/'native-plan.json').exists()
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);p=S.read(S.OUT/'plan.json');candidate=S.read(S.OUT/'native-candidate.json');assert S.file_sha(candidate['source_path'])==candidate['source_sha256']
    rows=[];missing=[]
    for arm in ['counted0','counted2']:
        source={r['id']:r for r in p['rows'] if r['arm']==arm}
        for result in S.read(S.OUT/(arm+'.json'))['results']:
            row=locate_sites(source[result['id']],result,tok)
            if row is None:missing.append(result['id'])
            else:rows.append(row)
    if missing:
        S.save('native-summary.json',dict(status='numeric_admission_failed',reason='Required causal readout positions could not be located for all192 predeclared views.',missing=missing,memo_text='明确计数接口通过行为门，但全部冻结样本的真实读出位置不齐全，因此停止原生路径步骤，没有以可解析子集替代完整样本。'))
        return
    assert len(rows)==192 and max(len(r['tokens']) for r in rows)<=LENGTH
    pilot=[]
    for depth in [2,3]:
        ids=sorted({r['family'] for r in rows if r['depth']==depth},key=lambda x:S.digest([S.SEED,'numeric-pilot',x]));pilot+=ids[:2]
    n=96-len(pilot);power=check_phase(.2,n,p_d=.3,alpha=.05/6);assert power['pass']
    for r in rows:r['split']='numeric' if r['family'] in pilot else 'confirm'
    S.save('native-plan.json',dict(rows=rows,arms=ARMS,source_hashes=sources(),behavior_analysis_sha256=S.file_sha(S.OUT/'analysis.json'),candidate_sha256=S.file_sha(S.OUT/'native-candidate.json'),source_generations={arm:S.file_sha(S.OUT/(arm+'.json')) for arm in ['counted0','counted2']},model_files=p['model_files'],pilot_families=pilot,n_confirm_families=n,unit_ids=candidate['unit_ids'],matched_unit_ids=candidate['matched_unit_ids'],block_index=36,length=LENGTH,batch=BATCH,chunk=92,power=power,primary=[['mask0','mask7'],['mask0','matched'],['mask7','hold_attention'],['mask7','hold_mlp'],['mask7','hold_both'],['mask0','mask2']],primary_metric='Per-family joint correctness across BOTH code rotations using original-model full-vocabulary top1. Exact paired McNemar Holm6; effects below20 points may be underpowered.',source_partition='P=all original prompt tokens; H=all self-generated prefix tokens excluding the current last token; C=current last token. mask bits1/2/4 remove fixed32 B36 post-gate*up activations on those disjoint source positions; never change weights or input token IDs.',paths='hold_attention/hold_mlp/hold_both remove group on all valid B36 positions, then restore corresponding natural o_proj/down_proj outputs at ALL positions of B37..39. history_hold_attention is mask2 plus later attention restoration, expected to leave current-token output bitwise unchanged by architecture.',numerical=dict(dose1_bitwise=True,history_attention_hold_current_bitwise=True,hf_reference_relative_l2_max=.02,hf_reference_logit_max_abs=.5,source_generation_top1_agreement_min=.99,norm_control_relative_error_max=.02),controls='Fixed matched32, no-op dose1, random residual direction matched iteratively to actual post-B36 BF16 displacement; invalid norm controls retained and not used to claim selective effects.',readout='Causal prefix before actual Code token; candidate code tokens validated single-token including their real leading space. Code itself excluded. Value and Slot are already in self-history: this is conditional output routing, not proof of original arithmetic computation.',fields='All41 raw residual boundaries at prompt end,before the d-th printed Step (d=program update count, even if extra values are printed),before Value,before Code. Native BF16; final normalization separately. Full B36G/V/A at those sites; selected64 activation trajectories over all valid prefix tokens; exact native down-weight columns. No hidden-coordinate sorting/PCA.',interaction='Boolean-lattice differences of fixed-baseline-competitor logit margin; nonlinear interaction can arise from downstream attention/MLP/RMSNorm and does not itself establish a linguistic computational unit.',budget=dict(new_peak_bytes=750*2**20,reserve_bytes=20*2**30,expected_minutes=30,hard_runtime_minutes=90),limits='Numeric pilot4 families excluded from confirmation. No success-subset filtering. No sufficient/unique-mechanism claim. No new SAE/transcoder or broad side-task intervention battery. Source-location effects do not uniquely determine first formation of a complex semantic feature.'))
    print('Native protocol frozen:8 numerical views;184 confirmation views/92families',flush=True)

def admit():
    p=S.read(S.OUT/'native-plan.json');assert S.read(S.OUT/'analysis.json')['gate_passed'];assert p['behavior_analysis_sha256']==S.file_sha(S.OUT/'analysis.json') and p['candidate_sha256']==S.file_sha(S.OUT/'native-candidate.json')
    for n,h in p['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/n)==h,n
    for n,h in p['source_generations'].items():assert S.file_sha(S.OUT/(n+'.json'))==h,n
    return p

def reference(rows,expected,R):
    """HF automatic 2D-mask orchestration, independent of custom 4D mask construction."""
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cfg=AutoConfig.from_pretrained(S.MODEL,local_files_only=True);hooks=[];seen=[]
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    for obj,key in [(model.model.embed_tokens,'model.embed_tokens.weight'),(model.model.norm,'model.norm.weight'),(model.lm_head,'lm_head.weight')]:obj.weight=torch.nn.Parameter(R.C.weight(key).cuda(),requires_grad=False)
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda()
    for b,layer in enumerate(model.model.layers):
        def load(m,a,b=b):m.load_state_dict(R.C.load_block(b,cfg).state_dict(),assign=True,strict=True)
        def release(m,a,y):m.to_empty(device='meta')
        hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
    checks=[];last_raw={}
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m,a:last_raw.update(value=a[0])))
    for lo in range(0,len(rows),BATCH):
        rr=rows[lo:lo+BATCH];ids=torch.from_numpy(R.padded(rr)).cuda();mask=torch.arange(LENGTH,device='cuda')[None,:]<torch.tensor([len(r['tokens']) for r in rr],device='cuda')[:,None]
        # Omit position_ids and all4D masks so HF constructs them.
        output=model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=True)
        at=torch.tensor([len(r['tokens'])-1 for r in rr],device='cuda');batch=torch.arange(len(rr),device='cuda');actual=output.hidden_states[-1][batch,at];ref=torch.from_numpy(expected['final_normalized'][lo:lo+len(rr)]).view(torch.bfloat16).cuda()
        relative=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20));logits=output.logits[batch,at].float();wanted=torch.from_numpy(expected['baseline_logits'][lo:lo+len(rr)]).view(torch.bfloat16).cuda().float()
        sites=torch.tensor([r['sites'] for r in rr],device='cuda');bb=batch[:,None]
        raw_fields=torch.stack([h[bb,sites] for h in output.hidden_states[:-1]]+[last_raw['value'][bb,sites]],dim=1)
        expected_fields=torch.from_numpy(expected['hidden'][lo:lo+len(rr)]).view(torch.bfloat16).cuda()
        field_relative=(raw_fields.float()-expected_fields.float()).norm(dim=-1)/expected_fields.float().norm(dim=-1).clamp_min(1e-20)
        checks.append(dict(ids=[r['id'] for r in rr],normalized_relative_l2=relative,normalized_bitwise=bool(torch.equal(actual,ref)),raw_field_max_vector_relative_l2=float(field_relative.max()),raw_fields_bitwise=bool(torch.equal(raw_fields,expected_fields)),logit_max_abs=float((logits-wanted).abs().max()),top1_equal=bool(torch.equal(logits.argmax(-1),wanted.argmax(-1)))))
        del output,ids,mask,actual,ref,logits,wanted,raw_fields,expected_fields,field_relative;last_raw.clear();gc.collect();torch.cuda.empty_cache()
    for h in hooks:h.remove()
    del model;gc.collect();torch.cuda.empty_cache();return checks

def collect(rows,folder,p,R,start):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    if (folder/'record.json').exists():
        rec=S.read(folder/'record.json');assert rec['row_ids']==[r['id'] for r in rows] and rec['source_hashes']==p['source_hashes'] and rec['arrays_sha256']==S.file_sha(folder/'native.npz')
        with np.load(folder/'native.npz') as z:refdata=dict(final_normalized=z['final_normalized'].copy(),baseline_logits=z['baseline_logits'].copy(),hidden=z['hidden'].copy())
        return rec,refdata
    assert not (folder/'native.npz').exists(),'Uncommitted native array exists; inspect before any overwrite.'
    folder.mkdir(parents=True,exist_ok=True);cfg,pos,pe,causal=R.context();n=len(rows);indices=torch.tensor([r['sites'] for r in rows],device='cuda');batch=torch.arange(n,device='cuda');at=indices[:,-1]
    masks=torch.tensor(np.stack([partitions(r['prompt_length'],len(r['tokens']),LENGTH) for r in rows]),device='cuda');valid=masks.any(1);units=p['unit_ids'];matched=p['matched_unit_ids'];tracked=units+matched
    emb=R.C.weight('model.embed_tokens.weight').cuda();stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(rows)).cuda(),emb);del emb;torch.cuda.empty_cache();mask=R.mask_for(rows,causal)
    hidden=np.empty((n,41,4,S.DIM),np.uint16);hidden[:,0]=R.bits(stage[batch[:,None],indices]);native={k:np.empty((n,4,S.WIDTH),np.uint16) for k in ['G','V','A']};trajectory=np.zeros((n,LENGTH,64),np.uint16);weights=None;states=None;norm_records=[];applied_norms={}
    rng=np.random.default_rng(S.SEED+3600);direction=torch.from_numpy(rng.normal(size=S.DIM).astype(np.float32)).cuda();direction/=direction.norm();global_norm_target=None
    for block in range(40):
        layer=R.C.load_block(block,cfg)
        if block==36:states=stage[None].repeat(len(ARMS),1,1,1);del stage;weights=R.bits(layer.mlp.down_proj.weight[:,tracked].T)
        active=['prefix'] if block<36 else ARMS
        base_attention=torch.empty((n,LENGTH,S.DIM),device='cuda',dtype=torch.bfloat16) if block>=36 else None
        base_mlp=torch.empty_like(base_attention) if block>=36 else None
        base_residual=torch.empty_like(base_attention) if block==36 else None
        for ai,arm in enumerate(active):
            for lo in range(0,n,BATCH):
                hi=min(n,lo+BATCH);count=hi-lo;current=stage if block<36 else states[ai];cap={};hooks=[]
                if block>=36:
                    hooks+=[layer.self_attn.o_proj.register_forward_hook(lambda m,a,y:cap.update(attention=y)),layer.mlp.down_proj.register_forward_hook(lambda m,a,y:cap.update(mlp=y))]
                if block==36 and arm=='mask0':
                    hooks+=[layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:cap.update(residual=a[0])),layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:cap.update(G=y)),layer.mlp.up_proj.register_forward_hook(lambda m,a,y:cap.update(V=y)),layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:cap.update(A=a[0]))]
                if block==36 and arm!='mask0':
                    bits=int(arm[4:]) if arm.startswith('mask') else 2 if arm=='history_hold_attention' else 7
                    editmask=masks[lo:hi,[j for j in range(3) if bits&(1<<j)]].any(1);which=matched if arm=='matched' else units
                    dose=1. if arm=='restore' else 0.
                    def edit(m,args,y):
                        if arm=='random_norm':
                            original=states[0,lo:hi];target=(states[7,lo:hi].float()-original.float()).norm(dim=-1);scale=target.clone();best=y.clone();err=torch.full_like(target,float('inf'))
                            for iteration in range(8):
                                trial=(y.float()+scale[:,:,None]*direction).to(y.dtype);post=base_residual[lo:hi]+trial;got=(post.float()-original.float()).norm(dim=-1);relative=(got-target).abs()/target.clamp_min(1e-12);better=relative<err;best=torch.where(better[:,:,None],trial,best);err=torch.minimum(err,relative);scale*=target/got.clamp_min(1e-12)
                            actual=((base_residual[lo:hi]+best).float()-original.float()).norm(dim=-1);ok=valid[lo:hi];tn=(target.square()*ok).sum(-1).sqrt();an=(actual.square()*ok).sum(-1).sqrt();rel=(an-tn).abs()/tn.clamp_min(1e-12)
                            for j,row in enumerate(rows[lo:hi]):norm_records.append(dict(id=row['id'],target_norm=float(tn[j]),actual_norm=float(an[j]),relative_error=float(rel[j]),passed=bool(rel[j]<=.02)))
                            return torch.where(editmask[:,:,None],best,y)
                        value=args[0].clone();scales=torch.where(editmask,dose,1.).to(value.dtype);value[:,:,which]*=scales[:,:,None]
                        return torch.nn.functional.linear(value,m.weight)
                    hooks.append(layer.mlp.down_proj.register_forward_hook(edit))
                if block>=37 and arm in ['hold_attention','hold_both','history_hold_attention']:
                    hooks.append(layer.self_attn.o_proj.register_forward_hook(lambda m,a,y:base_attention[lo:hi].clone()))
                if block>=37 and arm in ['hold_mlp','hold_both']:
                    hooks.append(layer.mlp.down_proj.register_forward_hook(lambda m,a,y:base_mlp[lo:hi].clone()))
                current[lo:hi]=layer(current[lo:hi],attention_mask=mask[lo:hi],position_ids=pos,position_embeddings=pe,use_cache=False)
                for h in hooks:h.remove()
                if block>=36 and arm=='mask0':
                    base_attention[lo:hi]=cap['attention'];base_mlp[lo:hi]=cap['mlp']
                    if block==36:
                        base_residual[lo:hi]=cap['residual'];ix=indices[lo:hi];bb=torch.arange(count,device='cuda')[:,None]
                        for k in native:native[k][lo:hi]=R.bits(cap[k][bb,ix])
                        assert torch.equal(torch.nn.functional.silu(cap['G'])*cap['V'],cap['A'])
                        trajectory[lo:hi]=R.bits(cap['A'][:,:,tracked])
                del cap
        baseline=stage if block<36 else states[0];hidden[:,block+1]=R.bits(baseline[batch[:,None],indices])
        if block==36:
            for ai,arm in enumerate(ARMS):applied_norms[arm]=((states[ai].float()-states[0].float()).square().sum(-1)*valid).sum(-1).sqrt().cpu().tolist()
        del layer,base_attention,base_mlp,base_residual;gc.collect();torch.cuda.empty_cache();S.log('native_'+folder.name,block+1,40,start,block=block)
        if time.monotonic()-start>p['budget']['hard_runtime_minutes']*60:raise RuntimeError('Frozen native runtime cap reached')
    assert torch.equal(states[ARMS.index('restore')],states[0]),'Dose1 numerical no-op failed'
    assert torch.equal(states[ARMS.index('history_hold_attention'),batch,at],states[0,batch,at]),'Causal attention-routing sanity failed'
    raw=states[:,batch,at].clone();del states;gc.collect();torch.cuda.empty_cache()
    norm=Qwen3RMSNorm(S.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False);final=norm(raw);head=R.C.weight('lm_head.weight').cuda();effects=[];baseline_logits=np.empty((n,head.shape[0]),np.uint16)
    for lo in range(0,n,BATCH):
        hi=min(n,lo+BATCH);base_native=torch.nn.functional.linear(final[0,lo:hi],head);baseline_logits[lo:hi]=R.bits(base_native);base=base_native.float();lp=base.log_softmax(-1);prob=lp.exp()
        rivals=[]
        for j,row in enumerate(rows[lo:hi]):rivals.append(max((k for k in range(4) if k!=row['gold']),key=lambda k:float(base[j,row['choice_ids'][k]])))
        for ai,arm in enumerate(ARMS):
            logits=torch.nn.functional.linear(final[ai,lo:hi],head).float();kl=(prob*(lp-logits.log_softmax(-1))).sum(-1)
            for j,row in enumerate(rows[lo:hi]):
                scores=logits[j,row['choice_ids']].cpu().tolist();g=row['gold'];effects.append(dict(id=row['id'],family=row['family'],rotation=row['arm'],depth=row['depth'],arm=arm,gold=g,choice_ids=row['choice_ids'],scores=scores,fixed_rival=rivals[j],fixed_margin=scores[g]-scores[rivals[j]],strict_margin=scores[g]-max(v for k,v in enumerate(scores) if k!=g),top1=int(logits[j].argmax()),correct=int(logits[j].argmax())==row['choice_ids'][g],observed_token=row['observed_token'],KL=float(kl[j]),b36_displacement_norm=applied_norms[arm][lo+j]))
    arrays=dict(hidden=hidden,final_raw=R.bits(raw),final_normalized=R.bits(final[0]),baseline_logits=baseline_logits,B36_G=native['G'],B36_V=native['V'],B36_A=native['A'],B36_selected_trajectory=trajectory,down_columns=weights)
    file=folder/'native.npz';np.savez_compressed(file,**arrays);refdata=dict(final_normalized=arrays['final_normalized'],baseline_logits=arrays['baseline_logits'],hidden=arrays['hidden'])
    record=dict(row_ids=[r['id'] for r in rows],arms=ARMS,effects=effects,norm_controls=norm_records,arrays_sha256=S.file_sha(file),restore_bitwise=True,history_attention_hold_current_bitwise=True,native_product_bitwise=True,source_hashes=p['source_hashes'])
    S.save(str(folder.relative_to(S.OUT))+'/record.json',record)
    del raw,final,head,norm,mask,hidden,native,trajectory,arrays;gc.collect();torch.cuda.empty_cache();return record,refdata

def run():
    S.configure(750*2**20);p=admit();R=S.bind();R.LENGTH=LENGTH;R.BATCH=BATCH;start=time.monotonic();records=[]
    with S.GpuLease():
        import torch
        with torch.inference_mode():
            for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('native_identity',i+1,len(p['model_files']),start)
            pilot=[r for r in p['rows'] if r['split']=='numeric'];rec,refdata=collect(pilot,S.OUT/'native/numeric',p,R,start)
            if (S.OUT/'native-numeric.json').exists():
                numeric=S.read(S.OUT/'native-numeric.json');assert numeric['source_hashes']==p['source_hashes'] and numeric['pilot_arrays_sha256']==rec['arrays_sha256'];checks=numeric['checks'];passed=numeric['passed']
            else:
                checks=reference(pilot,refdata,R)
                generation_agreement=sum(r['top1']==r['observed_token'] for r in rec['effects'] if r['arm']=='mask0')/len(pilot)
                checks.append(dict(kind='source_generation',agreement=generation_agreement,normalized_relative_l2=0.,logit_max_abs=0.,top1_equal=generation_agreement>=p['numerical']['source_generation_top1_agreement_min']))
                passed=all(c['normalized_relative_l2']<=.02 and c.get('raw_field_max_vector_relative_l2',0.)<=.02 and c['logit_max_abs']<=.5 and c['top1_equal'] for c in checks)
                S.save('native-numeric.json',dict(checks=checks,passed=passed,pilot_family_count=4,excluded_from_confirm=True,source_hashes=p['source_hashes'],pilot_arrays_sha256=rec['arrays_sha256']))
            if not passed:
                S.save('native-summary.json',dict(status='numeric_admission_failed',checks=checks,memo_text='明确计数行为门通过，但原生运行器的标准HF自动掩码/位置数值复核或自然生成代码身份核对未满足冻结准入，停止正式路径确认；仅保留数值校准数据，不把它计为机制证据。'));return
            del refdata;gc.collect();torch.cuda.empty_cache()
            rows=[r for r in p['rows'] if r['split']=='confirm']
            for lo in range(0,len(rows),p['chunk']):
                folder=S.OUT/f'native/confirm-{lo:03d}';rec,_=collect(rows[lo:lo+p['chunk']],folder,p,R,start);records.append(rec);gc.collect();torch.cuda.empty_cache()
    admit();S.save('native-execution.json',dict(records=[dict(path=f'native/confirm-{lo:03d}/record.json',sha256=S.file_sha(S.OUT/f'native/confirm-{lo:03d}/record.json')) for lo in range(0,len(rows),p['chunk'])],numeric_sha256=S.file_sha(S.OUT/'native-numeric.json'),seconds=time.monotonic()-start,confirmation_families=92,confirmation_views=184,source_hashes=p['source_hashes']))
    print('Native confirmation complete; analyze before closeout',flush=True)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
