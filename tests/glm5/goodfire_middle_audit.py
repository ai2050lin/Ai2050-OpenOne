"""Read-only audits and query records for the all-coordinate middle atlas."""
import argparse
import gc
import time
import numpy as np
import goodfire_middle as M
C=M.C


def full_model_reference():
    """Verify the full HF forward; 14B uses explicitly streamed layer weights."""
    import torch
    from transformers import AutoModelForCausalLM
    tokens=np.load(M.OUT/'tokens.npy')[:4];checks=[];captured={};start=time.monotonic()
    M.write(M.OUT/'reference-check-binding.json',dict(source_sha256=C.file_sha(__file__),
        reason='Reference admission independent of legacy CUDA-initialized RoPE.',
        reference='Full HF forward; first four discovery documents, batch2, explicit causal mask; 14B native layers streamed to CUDA.',
        max_relative_L2=.02,heldout_used=False))
    with C.GpuLease(),torch.inference_mode():
        stream_hooks=[]
        if M.MODEL_KEY=='qwen4':
            model=AutoModelForCausalLM.from_pretrained(C.MODEL,local_files_only=True,dtype=torch.bfloat16,
                device_map='cuda',attn_implementation='sdpa').eval()
        else:
            from transformers import AutoConfig
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
            cfg=AutoConfig.from_pretrained(C.MODEL,local_files_only=True);cfg._attn_implementation='sdpa'
            with torch.device('meta'):
                model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
            model.model.embed_tokens.weight=torch.nn.Parameter(C.weight('model.embed_tokens.weight').cuda(),requires_grad=False)
            model.model.norm.weight=torch.nn.Parameter(C.weight('model.norm.weight').cuda(),requires_grad=False)
            model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda()
            for b,layer in enumerate(model.model.layers):
                def load(m,a,b=b):
                    resident=C.load_block(b,cfg)
                    m.load_state_dict(resident.state_dict(),assign=True,strict=True)
                def release(m,a,y):m.to_empty(device='meta')
                stream_hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        mask=torch.full((1024,1024),float('-inf'),device='cuda',dtype=torch.bfloat16).triu(1)[None,None]
        hooks=[]
        for b in M.BLOCKS:
            layer=model.model.layers[b]
            hooks.extend([layer.mlp.gate_proj.register_forward_hook(lambda m,a,y,b=b:captured.update({(b,'G'):y[:,C.POSITIONS].clone()})),
                layer.mlp.up_proj.register_forward_hook(lambda m,a,y,b=b:captured.update({(b,'V'):y[:,C.POSITIONS].clone()})),
                layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,b=b:captured.update({(b,'A'):a[0][:,C.POSITIONS].clone()}))])
        for lo in range(0,4,2):
            out=model.model(input_ids=torch.from_numpy(tokens[lo:lo+2].astype(np.int64)).cuda(),
                attention_mask=mask,position_ids=torch.arange(1024,device='cuda')[None],use_cache=False)
            for (b,f),value in captured.items():
                ref=np.load(M.OUT/f'B{b}-{f}.npy',mmap_mode='r')[lo:lo+2]
                got=value.contiguous().view(torch.uint16).cpu().numpy()
                rel=float(np.linalg.norm(C.unbits(got)-C.unbits(ref))/max(np.linalg.norm(C.unbits(ref)),1e-20))
                checks.append(dict(first_document=lo,block=b,field=f,relative_L2=rel,bitwise=bool(np.array_equal(got,ref))))
            got=out.last_hidden_state[:,C.POSITIONS].contiguous().view(torch.uint16).cpu().numpy()
            ref=np.load(M.OUT/'final-hidden.npy',mmap_mode='r')[lo:lo+2]
            rel=float(np.linalg.norm(C.unbits(got)-C.unbits(ref))/max(np.linalg.norm(C.unbits(ref)),1e-20))
            checks.append(dict(first_document=lo,boundary='final_norm',relative_L2=rel,bitwise=bool(np.array_equal(got,ref))))
            captured.clear();M.log('full_HF_reference',lo+2,4,start)
        for hook in hooks+stream_hooks:hook.remove()
        del model,out;gc.collect();torch.cuda.empty_cache()
    passed=all(r['relative_L2']<=.02 for r in checks)
    M.write(M.OUT/'full-model-reference.json',dict(passed=passed,checks=checks,seconds=time.monotonic()-start,
        scope='Same-node original full-model numerical reference on discovery only; not independent scientific replication.'))
    assert passed, 'Full HF reference exceeds .02 tolerance; measurements saved'
    return dict(available=True,kind='full_HF_discovery_reference',checks=len(checks),
        all_bitwise=all(r['bitwise'] for r in checks),max_relative_L2=max(r['relative_L2'] for r in checks))


def audit_native():
    start=time.monotonic();checks=[];nbytes=0
    for b in M.BLOCKS:
        for field,width in [('G',C.WIDTH),('V',C.WIDTH),('A',C.WIDTH),('input',C.DIM),('output',C.DIM)]:
            p=M.OUT/f'B{b}-{field}.npy';x=np.load(p,mmap_mode='r')
            assert x.shape==(1024,8,width) and x.dtype==np.uint16
            for lo in range(0,1024,64):assert np.isfinite(C.unbits(x[lo:lo+64])).all()
            checks.append(dict(block=b,field=field,shape=list(x.shape),finite=True));nbytes+=p.stat().st_size
        for field,shape in [('gate',(C.WIDTH,C.DIM)),('up',(C.WIDTH,C.DIM)),('down',(C.DIM,C.WIDTH))]:
            p=M.OUT/f'B{b}-W-{field}.npy';x=np.load(p,mmap_mode='r')
            assert x.shape==shape and x.dtype==np.uint16
            for lo in range(0,len(x),256):assert np.isfinite(C.unbits(x[lo:lo+256])).all()
            nbytes+=p.stat().st_size
        M.log('native_audit',b-17,11,start)
    x=np.load(M.OUT/'hidden-all.npy',mmap_mode='r')
    assert x.shape==(1024,M.SPEC['layers']+1,8,C.DIM)
    for lo in range(0,1024,32):assert np.isfinite(C.unbits(x[lo:lo+32])).all()
    # Re-read the same document families and positions from the historical B28.
    comparisons=[];docs=C.read(M.OUT/'documents.json')['docs']
    if M.MODEL_KEY=='qwen14':
        previous=M.HISTORY;doc_indices=np.arange(1024);slots=np.arange(8)
    else:
        previous=M.ROOT/'tests/glm5/result/mlp_tail_b28_b31_v1'
        plan=C.read(previous/'plan.json');slots=np.array([plan['positions'].index(p) for p in C.POSITIONS])
        source_docs=C.read(M.ROOT/'tests/glm5/result/mlp_tail_goodfire_v1/documents.json')['docs']
        index={d['doc_id']:i for i,d in enumerate(source_docs)}
        doc_indices=np.array([index[d['doc_id']] for d in docs])
        assert max(doc_indices)<2048
    unavailable=[str(previous/f'B28-{f}.npy') for f in ['G','V','A'] if not (previous/f'B28-{f}.npy').exists()]
    reference=full_model_reference()
    if unavailable:
        M.write(M.OUT/'missing-historical-reference.json',dict(paths=unavailable,
            affected_check='Historical B28 elementwise equality cannot be established.',new_fields_retained=True))
    for field in ([] if unavailable else ['G','V','A']):
        oldpath=previous/f'B28-{field}.npy';old=np.load(oldpath,mmap_mode='r')
        old=old.reshape(1024,8,C.WIDTH) if M.MODEL_KEY=='qwen14' else old.reshape(2048,16,C.WIDTH)
        new=np.load(M.OUT/f'B28-{field}.npy',mmap_mode='r')
        total_diff=total_ref=0.;equal=True;worst=0.
        for lo in range(0,1024,32):
            a=old[doc_indices[lo:lo+32]][:,slots];b=new[lo:lo+32]
            delta=C.unbits(a).astype(np.float64)-C.unbits(b);ref=C.unbits(a).astype(np.float64)
            diff=float(np.sum(delta*delta));den=float(np.sum(ref*ref))
            total_diff+=diff;total_ref+=den;worst=max(worst,(diff/max(den,1e-30))**.5)
            equal&=bool(np.array_equal(a,b))
        rel=(total_diff/max(total_ref,1e-30))**.5
        comparisons.append(dict(field=field,bitwise_equal=equal,relative_L2=rel,max_32doc_relative_L2=worst,
            reference_path=str(oldpath),reference_sha256=C.file_sha(oldpath)))
        # Legacy reference uses CUDA-initialized rotary frequencies. Its drift
        # remains a reported execution-version difference, not the v2 admission.
    assert all(x['product_bitwise'] and x['output_relative_L2']<=.02 for x in C.read(M.OUT/'collection.json')['replay'])
    M.write(M.OUT/'native-audit.json',dict(passed=True,checks=checks,historical_B28=comparisons,
        historical_reference_available=not unavailable,alternative_reference=reference,source_sha256=C.file_sha(__file__),
        native_GVA_weights_bytes=nbytes,hidden_boundary='embedding then every residual block, pre-final-norm',
        seconds=time.monotonic()-start))


def audit_analysis():
    checks=[]
    for b in M.BLOCKS:
        profile=C.read(M.OUT/f'B{b}-profile.json')
        assert profile['width']==C.WIDTH
        for f in ['G','V','A']:
            for name in ['mean','std','min','max']:
                a=np.asarray(profile['fields'][f][name]);assert len(a)==C.WIDTH and np.isfinite(a).all()
        with np.load(M.OUT/f'B{b}-regimes.npz') as z:
            for name in z.files:assert len(z[name])==C.WIDTH and np.isfinite(z[name]).all()
            for split in C.SPLITS:
                np.testing.assert_allclose(z[split+'_quadrant_fraction'].sum(1),1,atol=1e-7)
                for name in ['quadrant_abs_mass_fraction','quadrant_energy_fraction']:
                    mass=z[split+'_'+name].sum(1)
                    assert ((np.abs(mass-1)<1e-6)|(mass==0)).all()
        with np.load(M.OUT/f'B{b}-prediction.npz') as z:
            assert all(len(z[k])==C.WIDTH and np.isfinite(z[k]).all() for k in z.files)
        if b<28:
            with np.load(M.OUT/f'B{b}-B{b+1}-predictor.npz') as z:
                assert all(len(z[k])==C.WIDTH and np.isfinite(z[k]).all() for k in z.files)
                assert ((z['parent']>=0)&(z['parent']<C.WIDTH)).all()
        checks.append(dict(block=b,width=C.WIDTH,passed=True))
    M.write(M.OUT/'analysis-audit.json',dict(passed=True,checks=checks,
        scope='Shape/finite/value-conservation and index checks, not semantic validation.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS);p.add_argument('stage',choices=['native','analysis'])
    a=p.parse_args();M.configure(a.model);M.require_admission();globals()['audit_'+a.stage]()


if __name__=='__main__':main()
