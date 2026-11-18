"""Standard HF mask/position orchestration and cached short generation audit."""
import gc,time
import numpy as np
import b36_semantic14 as S

def run():
    S.configure(40*2**20);assert not (S.OUT/'hf-independent-check.json').exists();R=S.bind();start=time.monotonic()
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding,Qwen3RMSNorm
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    questions=['What is 1 + 1? Reply with only the digit.','What is 2 + 3? Reply with only the digit.',
        'Evaluate (((1 + 2) - 2) + 3). Output exactly one character: the result digit. No spaces, punctuation or explanation.',
        'Start with the number 1. Add 2 to it. Subtract 2 from it. Add 3 to it. Output exactly one character: the result digit. No spaces, punctuation or explanation.']
    rows=[]
    for i,q in enumerate(questions):
        prompt=tok.apply_chat_template([dict(role='user',content=q)],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(prompt,add_special_tokens=False)
        rows.append(dict(id=str(i),prompt=prompt,tokens=ids,answer_pos=len(ids)-1))
    with S.GpuLease():
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context();emb=R.C.weight('model.embed_tokens.weight').cuda();stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(rows)).cuda(),emb)
        del emb;mask=R.mask_for(rows,causal)
        for block in range(40):
            layer=R.C.load_block(block,cfg);stage=layer(stage,attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            del layer;gc.collect();torch.cuda.empty_cache()
            if block%10==9:S.log('independent_native',block+1,40,start)
        at=torch.tensor([r['answer_pos'] for r in rows],device='cuda');idx=torch.arange(4,device='cuda');native=stage[idx,at].clone();del stage,mask;torch.cuda.empty_cache()
        norm=Qwen3RMSNorm(S.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False);native=norm(native);del norm
        # Build standard HF model, but stream original layer tensors to fit VRAM.
        config=AutoConfig.from_pretrained(S.MODEL,local_files_only=True)
        with torch.device('meta'):model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        model.model.embed_tokens.weight=torch.nn.Parameter(R.C.weight('model.embed_tokens.weight').cuda(),requires_grad=False)
        model.model.norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        model.lm_head.weight=torch.nn.Parameter(R.C.weight('lm_head.weight').cuda(),requires_grad=False)
        model.model.rotary_emb=Qwen3RotaryEmbedding(config,device='cpu').cuda();hooks=[];calls=[0]
        for b,layer in enumerate(model.model.layers):
            def load(m,a,b=b):m.load_state_dict(R.C.load_block(b,config).state_dict(),assign=True,strict=True)
            def release(m,a,y,b=b):
                m.to_empty(device='meta');calls[0]+=1
                if b==39:S.log('independent_HF_forward',calls[0]//40,28,start)
            hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        ids=torch.from_numpy(R.padded(rows)).cuda();valid=torch.arange(S.LENGTH,device='cuda')[None,:]<torch.tensor([len(r['tokens']) for r in rows],device='cuda')[:,None]
        hf=model.model(input_ids=ids,attention_mask=valid.long(),use_cache=False).last_hidden_state[idx,at]
        error=float((hf.float()-native.float()).norm()/native.float().norm());bitwise=bool(torch.equal(hf,native))
        native_z=torch.nn.functional.linear(native,model.lm_head.weight).float();hf_z=torch.nn.functional.linear(hf,model.lm_head.weight).float()
        results=[dict(id=r['id'],prompt=r['prompt'],native_top1=int(native_z[i].argmax()),hf_top1=int(hf_z[i].argmax()),
            native_text=tok.decode([int(native_z[i].argmax())]),hf_text=tok.decode([int(hf_z[i].argmax())]),max_logit_difference=float((native_z[i]-hf_z[i]).abs().max())) for i,r in enumerate(rows)]
        # A tiny exact unpadded standard generation check, one fixed arithmetic item.
        query=rows[2];input_ids=torch.tensor([query['tokens']],device='cuda')
        output=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),max_new_tokens=24,do_sample=False,use_cache=True,
            pad_token_id=151643,eos_token_id=151645)
        generated=output[0,len(query['tokens']):].cpu().tolist()
        for h in hooks:h.remove()
        S.save('hf-independent-check.json',dict(kind='engineering_real_model_audit',native_vs_HF_hidden_relative_L2=error,native_vs_HF_bitwise=bitwise,rows=results,
            generated_item=query['id'],generated_tokens=generated,generated_text=tok.decode(generated),
            orchestration='HF builds its own mask and position IDs from a2D validity mask; standard HF generate uses its native cache. Only original layer weights are streamed, unchanged.',
            source_sha256=S.file_sha(__file__),seconds=time.monotonic()-start))
        print('HF check',error,bitwise,results,'generated',repr(tok.decode(generated)),flush=True)

if __name__=='__main__':run()
