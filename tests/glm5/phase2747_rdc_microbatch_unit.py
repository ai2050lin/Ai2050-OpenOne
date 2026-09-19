"""Small random CPU-module arithmetic tests; not a local LLM experiment."""
from rdc_formation_common import *
from rdc_formation_microbatch import NativeWave
from rdc_formation_glm_wave import GlmNativeWave
from rdc_formation_history import HiddenObserver


def main():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM,Glm4Config,Glm4ForCausalLM,GlmConfig,GlmForCausalLM
    start=time.monotonic();checks=[];torch.manual_seed(2747);torch.set_num_threads(2)
    for config_class,model_class in [(Qwen3Config,Qwen3ForCausalLM),(Glm4Config,Glm4ForCausalLM),(GlmConfig,GlmForCausalLM)]:
        config=config_class(vocab_size=64,hidden_size=64,intermediate_size=96,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=16,partial_rotary_factor=.5,
            attention_dropout=0.,max_position_embeddings=128,pad_token_id=0)
        config._attn_implementation='eager'
        model=model_class(config).bfloat16().eval();observer=HiddenObserver(model)
        engine=(GlmNativeWave if config.model_type=='glm' else NativeWave)(model,'cpu')
        old=[];new=[]
        for b,n in [(1,7),(2,11),(3,9)]:
            ids=torch.randint(1,64,(b,n));mask=torch.ones_like(ids)
            if b>1:mask[0,:3]=0;ids[0,:3]=0
            item={'input_ids':ids,'attention_mask':mask,'position_ids':(mask.cumsum(-1)-1).clamp_min(0),
                'use_cache':True,'cache':None,'collect_hidden':True}
            old.append(dict(item));new.append(dict(item))
        try:
            with torch.inference_mode():
                for step in range(4):
                    expected=[]
                    for r in old:
                        observer.active=True
                        v=model.model(input_ids=r['input_ids'],attention_mask=r['attention_mask'],position_ids=r['position_ids'],
                            past_key_values=r['cache'],use_cache=True)
                        r['cache']=v.past_key_values
                        expected.append((v.last_hidden_state[:,-1].clone(),np.stack([observer.state[i] for i in range(4)],axis=1)))
                    observer.active=False
                    actual=engine.forward(new)
                    for i,(r,s,w,a) in enumerate(zip(old,new,expected,actual)):
                        assert torch.equal(w[0],a['postnorm'])
                        assert np.array_equal(w[1],a['hidden'])
                        assert cache_id(r['cache'])==cache_id(s['cache'])
                        chosen=model.lm_head(w[0]).float().argmax(-1)[:,None]
                        mask=torch.cat([r['attention_mask'],torch.ones_like(chosen)],-1)
                        if step>=1 and i>0:mask[0,-1]=0;chosen[0,0]=0
                        for request in (r,s):request.update(input_ids=chosen.clone(),attention_mask=mask.clone(),position_ids=(mask.sum(-1)-1).clamp_min(0)[:,None])
                        checks.append({'model_type':config.model_type,'step':step,'microbatch':i,
                            'postnorm_allH_and_complete_KV_bit_equal':True})
        finally:observer.close();engine.close()
    value={'timestamp':stamp(),'source':snapshot(__file__),'engine':snapshot(Path(__file__).with_name('rdc_formation_microbatch.py')),
        'glm_architecture_adapter':snapshot(Path(__file__).with_name('rdc_formation_glm_wave.py')),
        'all_passed':True,'checks':checks,'seconds':time.monotonic()-start,
        'scope':'Random three-layer64width synthetic CPU modules,36checks. Qwen3 and supplementary Glm4Model retain their old tests;12additional checks cover the actual GlmModel architecture. Native checkpoint CUDA, offload weight reuse, production caps and whole packets require separate actual qualification.'}
    path=OUT/'engineering/microbatch'/('unit_'+str(time.time_ns())+'.json');save(path,value)
    save(path.parent/'unit_current.json',{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    print('FORMATION_MICROBATCH_UNIT',len(checks),flush=True)


if __name__=='__main__':main()
