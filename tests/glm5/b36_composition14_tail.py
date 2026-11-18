"""Original-model path clamps and state-conditioned local propagation tests."""
import argparse
import gc
import time
import numpy as np
import b36_composition14 as B
import b36_composition14_analysis as A
import b36_composition14_collect as C
from b36_composition14_math import local_step,rms_jvp

ARMS=['baseline','restore','zero','double','matched_zero','matched_double','random_norm','radial_norm',
      'zero_hold_attention','zero_hold_mlp','zero_hold_both','zero_restore_roots','zero_restore_random_roots']
OPERATORS=['full0','no_attention0','no_mlp0','residual0','full2','no_attention2','no_mlp2','residual2']


def source_hashes():
    names=['b36_composition14_tail.py','b36_composition14_math.py','test_b36_composition14_math.py','b36_composition14_analysis.py',
           'b36_composition14_tail_analysis.py','b36_composition14_output.py','b36_composition14_campaign.py',
           'b36_composition14_behavior_only.py','b36_composition14_calibrate.py','test_b36_composition14_admission.py',
           'recursive_branch14_analysis.py','branch_reuse14_analysis.py','../../tools/provenance.py','../../tools/power_analysis.py']
    return {n:B.file_sha(B.ROOT/'tests/glm5'/n) for n in names}


def freeze():
    B.configure();A.admit();assert not (B.OUT/'execution-freeze.json').exists()
    assert (B.OUT/'selection.json').exists()
    assert 'OK' in (B.OUT/'math-tests.log').read_text(encoding='utf8')
    assert 'OK' in (B.OUT/'admission-tests.log').read_text(encoding='utf8')
    numerical_pilot=B.read(B.OUT/'tail-pilot/summary.json');assert numerical_pilot['numeric_passed']
    assert numerical_pilot['execution_source_hashes']==source_hashes()
    B.save('execution-freeze.json',dict(source_hashes=source_hashes(),selection_sha256=B.file_sha(B.OUT/'selection.json'),
        numeric_pilot_sha256=B.file_sha(B.OUT/'tail-pilot/summary.json'),arms=ARMS,operators=OPERATORS,
        edit='Scale B36 actual down-projection input for fixed32 at the final answer-prompt token. Component clamps at B37..39 only the same token.',
        norm='Random and negative-radial directions match actual BF16 post-B36 residual displacement within2%; retain invalid cases.',
        derivative='Smooth FP32 derivative anchored on original native values, native prefix QK/V. Actual applied B36 residual displacement is seed. No intervened downstream state feeds prediction.',
        same_material_counterfactual_baselines='Use already frozen primitive-fit F32(A36), Fh(h37), Fhp(h37,prefix mean) to predict root changes on the identical intervention rows. F32 gets A36 scaled0/2; Fh/Fhp get the measured applied B36 residual displacement, exactly the same seed as the native Jacobian. Historical prefix mean unchanged by a last-token intervention. No refitting or changed gates.',
        baseline_future='Natural unedited late states permitted as conditioning for local counterfactual response. Does not predict the unedited model from text.',
        limits=['Whole-prefix model replay is an execution reference, not extracted theory.','Root restoration is a mediator manipulation, not proof of unique route.','Dose1 is numerical no-op, not independent rescue.']))


def admit():
    A.admit();p=B.read(B.OUT/'execution-freeze.json')
    assert p['selection_sha256']==B.file_sha(B.OUT/'selection.json')
    for name,sha in p['source_hashes'].items():assert B.file_sha(B.ROOT/'tests/glm5'/name)==sha,name
    return p


def capture_hooks(layer,cap):
    hooks=[layer.self_attn.q_proj.register_forward_hook(lambda m,a,y:cap.update(qraw=y)),
        layer.self_attn.k_proj.register_forward_hook(lambda m,a,y:cap.update(kraw=y)),
        layer.self_attn.v_proj.register_forward_hook(lambda m,a,y:cap.update(vraw=y)),
        layer.self_attn.q_norm.register_forward_hook(lambda m,a,y:cap.update(qnorm=y)),
        layer.self_attn.k_norm.register_forward_hook(lambda m,a,y:cap.update(knorm=y)),
        layer.self_attn.o_proj.register_forward_hook(lambda m,a,y:cap.update(attention=y)),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:cap.update(mlp_residual=a[0])),
        layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:cap.update(G=y)),
        layer.mlp.up_proj.register_forward_hook(lambda m,a,y:cap.update(V=y)),
        layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:cap.update(A=a[0])),
        layer.mlp.down_proj.register_forward_hook(lambda m,a,y:cap.update(mlp=y))]
    return hooks


def down_edit(module,args,output,positions,units,dose,residual=None,direction=None):
    import torch
    batch=torch.arange(len(positions),device=positions.device);value=args[0].clone()
    for i,pos in enumerate(positions):value[i,pos,units]*=dose
    changed=torch.nn.functional.linear(value,module.weight)
    if direction is None:return changed,None
    original=residual+output[batch,positions]
    target=(residual+changed[batch,positions]).float()-original.float();norm=target.norm(dim=-1)
    scale=norm.clone();best=None;error=torch.full_like(norm,float('inf'))
    for _ in range(8):
        trial=(output[batch,positions].float()+scale[:,None]*direction).to(output.dtype)
        actual=(residual+trial).float()-original.float();got=actual.norm(dim=-1)
        e=(got-norm).abs()/norm.clamp_min(1e-12);better=e<error
        best=trial if best is None else torch.where(better[:,None],trial,best);error=torch.minimum(error,e)
        scale*=norm/got.clamp_min(1e-12)
    changed=output.clone();changed[batch,positions]=best
    return changed,dict(target_norm=norm.cpu().tolist(),relative_error=error.cpu().tolist())


def run(split='confirm',pilot=False):
    B.configure(240*2**20);R=B.bind();C.admit()
    execution_source_hashes=source_hashes()
    if pilot:
        rows,meta=A.load('pilot');rows=rows[:8];folder=B.OUT/'tail-pilot'
        selection=dict(root_control=[(u+1)%B.WIDTH for u in B.roots()])
    else:
        admit();rows,meta=A.eligible(split);folder=B.OUT/f'tail-{split}';selection=B.read(B.OUT/'selection.json')
    folder.mkdir(exist_ok=True);assert not (folder/'summary.json').exists()
    roots=B.roots();controls=selection['root_control'];old=B.read(B.PREVIOUS/'frozen-selection.json')['graphs']['parameter']
    target_units={edge['child']:edge['children'] for edge in old['edges']};target_units[39]=sorted(set(roots+controls))
    all_original_rows,_=A.load('pilot' if pilot else split);byid={r['id']:i for i,r in enumerate(all_original_rows)}
    originals=A.state('pilot' if pilot else split,40);refs=originals[[byid[r['id']] for r in rows]]
    start=time.monotonic();allresults=[];allchecks=[];allprediction=[];normrecords=[]
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context();rng=np.random.default_rng(B.SEED+36)
        random_direction=torch.from_numpy(rng.normal(size=B.DIM).astype(np.float32)).cuda();random_direction/=random_direction.norm()
        for lo in range(0,len(rows),B.CHUNK):
          rr=rows[lo:lo+B.CHUNK];n=len(rr);dest=folder/f'chunk-{lo:05d}'
          if dest.with_suffix('.json').exists():
            record=B.read(dest.with_suffix('.json'));assert record['row_ids']==[r['id'] for r in rr]
            assert B.file_sha(dest.with_suffix('.npz'))==record['arrays_sha256']
            allresults+=record['effects'];allprediction+=record['predictions'];allchecks.append(record['check']);normrecords+=record['norm_controls'];continue
          B.log('tail_'+split,lo*40,len(rows)*40,start,chunk=lo,pilot=pilot)
          emb=R.C.weight('model.embed_tokens.weight').cuda();stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(rr)).cuda(),emb)
          del emb;torch.cuda.empty_cache();mask=R.mask_for(rr,causal);states=None;directions=None
          native={b:np.empty((len(ARMS),n,len(us),3),np.uint16) for b,us in target_units.items()}
          tangents={};rounding=[];calibration=[];nc=[];unit_margins=[];component_sum=torch.zeros((n,B.DIM,2,2),device='cuda',dtype=torch.float32)
          for block in range(40):
            layer=R.C.load_block(block,cfg)
            if block==36:states=stage[None].repeat(len(ARMS),1,1,1);del stage
            active=['prefix'] if block<36 else ARMS
            baseline_attention=torch.empty((n,B.DIM),device='cuda',dtype=torch.bfloat16) if block>=36 else None
            baseline_mlp=torch.empty_like(baseline_attention) if block>=36 else None
            baseline_residual=torch.empty_like(baseline_attention) if block>=36 else None
            baseline_a=torch.empty((n,B.WIDTH),device='cuda',dtype=torch.bfloat16) if block>=36 else None
            for ai,arm in enumerate(active):
             for i in range(0,n,B.BATCH):
              local=rr[i:i+B.BATCH];count=len(local);positions=torch.tensor([r['answer_pos'] for r in local],device='cuda');batch=torch.arange(count,device='cuda')
              current=stage if block<36 else states[ai];before=current[i:i+count].clone() if block>=37 and arm=='baseline' else None
              cap={};hooks=capture_hooks(layer,cap) if block>=36 else []
              if block==36 and arm!='baseline':
                units=B.matched() if arm.startswith('matched') else B.group();dose=1. if arm=='restore' else 2. if arm in ['double','matched_double'] else 0.
                direction=None
                if arm=='random_norm':direction=random_direction[None].expand(count,-1)
                if arm=='radial_norm':
                    radial=(baseline_residual[i:i+count]+baseline_mlp[i:i+count]).float();direction=-radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                def edit(m,a,y,units=units,dose=dose,direction=direction):
                    out,control=down_edit(m,a,y,positions,units,dose,baseline_residual[i:i+count],direction)
                    cap['effective_mlp']=out
                    if control is not None:
                        for j,r in enumerate(local):nc.append(dict(id=r['id'],arm=arm,target_norm=control['target_norm'][j],relative_error=control['relative_error'][j],passed=control['relative_error'][j]<=.02))
                    return out
                hooks.append(layer.mlp.down_proj.register_forward_hook(edit))
              if block>=37 and arm in ['zero_hold_attention','zero_hold_both']:
                def hold_attention(m,a,y):
                    out=y.clone();out[batch,positions]=baseline_attention[i:i+count];cap['effective_attention']=out;return out
                hooks.append(layer.self_attn.o_proj.register_forward_hook(hold_attention))
              if block>=37 and arm in ['zero_hold_mlp','zero_hold_both']:
                def hold_mlp(m,a,y):
                    out=y.clone();out[batch,positions]=baseline_mlp[i:i+count];cap['effective_mlp']=out;return out
                hooks.append(layer.mlp.down_proj.register_forward_hook(hold_mlp))
              if block==39 and arm in ['zero_restore_roots','zero_restore_random_roots']:
                units=roots if arm=='zero_restore_roots' else controls
                def restore_roots(m,a,y,units=units):
                    value=a[0].clone()
                    for j,p in enumerate(positions):value[j,p,units]=baseline_a[i+j,units]
                    out=torch.nn.functional.linear(value,m.weight);cap['effective_mlp']=out;return out
                hooks.append(layer.mlp.down_proj.register_forward_hook(restore_roots))
              current[i:i+count]=layer(current[i:i+count],attention_mask=mask[i:i+count],position_ids=pos,position_embeddings=pe,use_cache=False)
              for h in hooks:h.remove()
              if block>=36:
                if arm=='baseline':
                    baseline_attention[i:i+count]=cap['attention'][batch,positions];baseline_mlp[i:i+count]=cap['mlp'][batch,positions]
                    baseline_residual[i:i+count]=cap['mlp_residual'][batch,positions];baseline_a[i:i+count]=cap['A'][batch,positions]
                    if block>=37:
                        updated,root_tangent,numeric=local_step(layer,before,cap,directions[i:i+count],positions,pe,roots if block==39 else None)
                        directions[i:i+count]=updated;calibration.append(dict(block=block,offset=i,**numeric))
                        if block==39:
                            if 'roots' not in tangents:tangents['roots']=np.empty((n,len(roots),3,8),np.float32)
                            tangents['roots'][i:i+count]=root_tangent[:,:,:,:8].cpu().numpy()
                            if 'unit_roots' not in tangents:tangents['unit_roots']=np.empty((n,len(roots),3,len(B.group())),np.float32)
                            tangents['unit_roots'][i:i+count]=root_tangent[:,:,:,8:].cpu().numpy()
                if block in target_units:
                    native[block][ai,i:i+count]=R.bits(torch.stack([cap[f][batch,positions][:,target_units[block]] for f in 'GVA'],-1))
                if block>=37 and arm in ['zero','double']:
                    dose=0 if arm=='zero' else 1
                    component_sum[i:i+count,:,dose,0]+=cap.get('effective_attention',cap['attention'])[batch,positions].float()-baseline_attention[i:i+count].float()
                    component_sum[i:i+count,:,dose,1]+=cap.get('effective_mlp',cap['mlp'])[batch,positions].float()-baseline_mlp[i:i+count].float()
              del before,cap
            if block==36:
                positions=torch.tensor([r['answer_pos'] for r in rr],device='cuda');batch=torch.arange(n,device='cuda')
                h=states[:,batch,positions].float();seed=torch.stack([h[ARMS.index('zero')]-h[0],h[ARMS.index('double')]-h[0]],-1)
                unit_seed=-baseline_a[:,B.group()].float()[:,None,:]*layer.mlp.down_proj.weight[:,B.group()].float()[None]
                directions=torch.cat([torch.stack([seed[:,:,0]]*4+[seed[:,:,1]]*4,-1),unit_seed],-1)
                tangents['seed']=seed.cpu().numpy()
                tangents['unit_seed_rounding_relative']=((unit_seed.sum(-1)-seed[:,:,0]).norm(dim=-1)/seed[:,:,0].norm(dim=-1).clamp_min(1e-12)).cpu().numpy()
                del unit_seed
            del layer,baseline_attention,baseline_mlp,baseline_residual,baseline_a;gc.collect();torch.cuda.empty_cache()
            if block%5==4 or block>=36:B.log('tail_'+split,lo*40+n*(block+1),len(rows)*40,start,block=block,chunk=lo,pilot=pilot)
          # Full final-hidden and path/operator outputs are retained; no long-prefix copies.
          positions=torch.tensor([r['answer_pos'] for r in rr],device='cuda');batch=torch.arange(n,device='cuda')
          raw=states[:,batch,positions].clone();assert torch.equal(raw[0],raw[1]),'Native dose1 no-op failed'
          valid=torch.arange(B.LENGTH,device='cuda')[None,:]<positions[:,None]
          prefix_ok=bool(torch.all((states==states[:1])|~valid[None,:,:,None]));assert prefix_ok
          ref=torch.from_numpy(refs[lo:lo+n]).to('cuda',torch.bfloat16);relative=float((raw[0].float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
          assert relative<=.02
          actual=torch.stack([raw[ARMS.index('zero')].float()-raw[0].float(),raw[ARMS.index('double')].float()-raw[0].float()],-1)
          closure=actual-torch.from_numpy(tangents['seed']).cuda()-component_sum.sum(-1)
          check=dict(chunk=lo,restore_bitwise=True,prefix_bitwise=True,baseline_bitwise=bool(torch.equal(raw[0],ref)),baseline_relative_L2=relative,
                     smooth_attention_max_relative_L2=max(r['attention_fp32_reconstruction_relative_L2'] for r in calibration),
                     native_rounding_residual_relative=float(closure.norm()/actual.norm().clamp_min(1e-12)))
          assert check['smooth_attention_max_relative_L2']<=.02,'Smooth/native attention calibration failed'
          norm=Qwen3RMSNorm(B.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          final=norm(raw);head=R.C.weight('lm_head.weight').cuda();effects=[];predictions=[]
          for i in range(0,n,B.BATCH):
            count=min(B.BATCH,n-i);base=torch.nn.functional.linear(final[0,i:i+count],head).float();lp=base.log_softmax(-1);prob=lp.exp()
            normalized_tangents=rms_jvp(raw[0,i:i+count].float(),directions[i:i+count,:,8:],norm.weight,norm.variance_epsilon)
            for j,r in enumerate(rr[i:i+count]):
                competitor=max((k for k in range(3) if k!=r['gold']),key=lambda k:float(base[j,r['choice_ids'][k]]))
                readout=(head[r['choice_ids'][r['gold']]].float()-head[r['choice_ids'][competitor]].float())
                unit_margins.append(dict(id=r['id'],competitor=competitor,unit_ids=B.group(),
                    smooth_zero_margin_contributions=(readout@normalized_tangents[j]).cpu().tolist()))
            del normalized_tangents
            for ai,arm in enumerate(ARMS):
                logits=torch.nn.functional.linear(final[ai,i:i+count],head).float();ll=logits.log_softmax(-1);kl=(prob*(lp-ll)).sum(-1)
                for j,r in enumerate(rr[i:i+count]):
                    scores=logits[j,r['choice_ids']].cpu().tolist();bs=base[j,r['choice_ids']].cpu().tolist();g=r['gold']
                    delta=scores[g]-max(v for k,v in enumerate(scores) if k!=g)-bs[g]+max(v for k,v in enumerate(bs) if k!=g)
                    effects.append(dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],title=r['title'],format=r['format'],arm=arm,scores=scores,baseline_scores=bs,gold=g,delta_margin=delta,KL=float(kl[j])))
            for dose,arm in [(0,'zero'),(2,'double')]:
                teacher=torch.nn.functional.linear(final[ARMS.index(arm),i:i+count],head).float().log_softmax(-1);tp=teacher.exp();zero=(tp*(teacher-lp)).sum(-1)
                for col in ([0,1,2,3] if dose==0 else [4]):
                    predicted=(raw[0,i:i+count].float()+directions[i:i+count,:,col]).to(torch.bfloat16)
                    logits=torch.nn.functional.linear(norm(predicted),head).float();kl=(tp*(teacher-logits.log_softmax(-1))).sum(-1)
                    for j,r in enumerate(rr[i:i+count]):predictions.append(dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],title=r['title'],dose=dose,operator=OPERATORS[col],KL=float(kl[j]),zero_effect_KL=float(zero[j]),scores=logits[j,r['choice_ids']].cpu().tolist()))
          arrays=dict(final_raw=R.bits(raw),seed=tangents['seed'],root_jvp=tangents['roots'],root_unit_jvp=tangents['unit_roots'],unit_seed_rounding_relative=tangents['unit_seed_rounding_relative'],final_jvp=directions[:,:,[0,1,2,3,4]].cpu().numpy(),component_sum=component_sum.cpu().numpy(),
                      **{f'B{b}':value for b,value in native.items()})
          B.arrays(dest.with_suffix('.npz'),arrays)
          record=dict(row_ids=[r['id'] for r in rr],effects=effects,predictions=predictions,unit_margin_linearization=unit_margins,norm_controls=nc,check=check,calibration=calibration,arrays_sha256=B.file_sha(dest.with_suffix('.npz')),source_sha256=B.file_sha(__file__))
          B.save(dest.with_suffix('.json').relative_to(B.OUT),record)
          allresults+=effects;allprediction+=predictions;allchecks.append(check);normrecords+=nc
          del states,mask,raw,final,head,norm,directions,component_sum,seed,arrays,native;gc.collect();torch.cuda.empty_cache()
        assert execution_source_hashes==source_hashes(),'Execution source changed while running'
        B.save((folder/'summary.json').relative_to(B.OUT),dict(kind='real',effects=allresults,predictions=allprediction,norm_controls=normrecords,checks=allchecks,execution_source_hashes=execution_source_hashes,
            numeric_passed=True,row_ids=[r['id'] for r in rows],arms=ARMS,operators=OPERATORS,targets=target_units,roots=roots,root_controls=controls,
            arrays={p.name:B.file_sha(p) for p in sorted(folder.glob('*.npz'))},seconds=time.monotonic()-start,source_sha256=B.file_sha(__file__),
            scope='Only final answer token edited. Computed G/V/A recorded before any local root restoration. Baseline natural states condition the smooth local operator; no edited downstream state fed into it.'))
        B.log('tail_complete_'+split,len(rows),len(rows),start,pilot=pilot)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','freeze','run']);p.add_argument('--split',default='confirm',choices=['confirm','format']);a=p.parse_args()
    freeze() if a.stage=='freeze' else run('pilot',True) if a.stage=='pilot' else run(a.split)
