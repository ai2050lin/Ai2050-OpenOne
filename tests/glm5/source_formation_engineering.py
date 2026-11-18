"""CPU-only random micro-architecture calibration, excluded from scientific evidence."""
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D

def main():
    import torch
    from transformers import Qwen3Config,GlmConfig
    from transformers.models.qwen3 import modeling_qwen3 as Q
    from transformers.models.glm import modeling_glm as G
    from source_binding_native_v2 import Capture
    from source_formation_native import decode
    from source_formation_causal import Editor
    torch.set_num_threads(4);torch.manual_seed(81);records=[]
    with torch.inference_mode():
        for Config,mod,prefix in [(Qwen3Config,Q,'Qwen3'),(GlmConfig,G,'Glm')]:
            cfg=Config(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
            cfg._attn_implementation='sdpa';layer=getattr(mod,prefix+'DecoderLayer')(cfg,0).bfloat16().eval()
            rt=SimpleNamespace(t=torch,mod=mod,hd=16,heads=4,kv=2)
            x=torch.randn(2,14,64).bfloat16();pos=torch.arange(14)[None].expand(2,-1)
            pe=getattr(mod,prefix+'RotaryEmbedding')(cfg,device='cpu')(x,pos);mask=torch.ones(2,1,14,14,dtype=torch.bool).tril()
            rows=[{'tokens':list(range(14)),'source':0,'entity':0,'target_record':0,'records':{str(j):{'span':[2*j,2*j+1]} for j in range(6)}} for _ in range(2)]
            cap=Capture(rt,layer);y=layer(x,attention_mask=mask,position_embeddings=pe);data=decode(cap,rows,14,{'position_embeddings':pe})
            err=float(data['error'].max());cap.close()
            meta={'heads':[{'block':0,'head':1}],'mlps':[{'block':0,'selected':list(range(16)),'random':list(range(16,32))}]}
            ed=Editor(rt,meta);ed.rows=rows;mod.ALL_ATTENTION_FUNCTIONS.register('formation_engineering',ed.attention);cfg._attn_implementation='formation_engineering'
            handles=[layer.self_attn.register_forward_hook(ed.attention_hook(0)),layer.mlp.down_proj.register_forward_hook(ed.mlp_hook(meta['mlps'][0]))]
            assert torch.equal(y,layer(x,attention_mask=mask,position_embeddings=pe))
            norm_errors=[]
            for pair in [['head_correct','head_other'],['mlp_selected','mlp_random']]:
                ed.probe=True;ed.proposals={}
                for arm in pair:ed.arm=arm;layer(x,attention_mask=mask,position_embeddings=pe)
                target=.5*torch.minimum(ed.proposals[pair[0]],ed.proposals[pair[1]])
                ed.probe=False;ed.targets={a:target for a in pair}
                for arm in pair:
                    ed.arm=arm;ed.local=[];changed=layer(x,attention_mask=mask,position_embeddings=pe)
                    assert torch.equal(y[:,:-1],changed[:,:-1]);assert not torch.equal(y[:,-1],changed[:,-1])
                    for info in ed.local:
                        n=torch.tensor(info['actual_norm']);t=torch.tensor(info['target_norm'])
                        norm_errors.append(float(((n-t).abs()/t.clamp_min(1e-8)).max()))
            ed.arm='base';assert torch.equal(y,layer(x,attention_mask=mask,position_embeddings=pe))
            for handle in handles:handle.remove()
            cfg._attn_implementation='sdpa'
            assert torch.equal(y,layer(x,attention_mask=mask,position_embeddings=pe))
            records.append({'architecture':prefix,'QKV_replay_max_relative_error':err,'norm_match_max_relative_error':max(norm_errors),
                'no_op_restore_and_prefix_locality':True,'device':'CPU','random_weights_not_research_evidence':True})
    D.save('engineering-calibration.json',{'passed':True,'checks':records})
    print(records)

if __name__=='__main__':main()
