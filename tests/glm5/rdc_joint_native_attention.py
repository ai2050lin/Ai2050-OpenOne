"""Actual selected-layer Q/K/V factors, all heads/sources; no full model or future input."""
import gc
import hashlib
from rdc_joint_common import *


class NativeAttention:
    def __init__(self, layer=12, device='cuda'):
        import torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm, Qwen3RotaryEmbedding
        from phase2712_rdc_prefix_native_probability import checkpoint
        self.torch, self.device, self.layer = torch,device,layer
        self.config = Qwen3Config.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
        self.config._attn_implementation = 'eager'
        with torch.device('meta'):
            self.attention = Qwen3Attention(self.config, layer)
            self.norm = Qwen3RMSNorm(self.config.hidden_size,self.config.rms_norm_eps)
        prefix = f'model.layers.{layer}.'
        weights = {k:checkpoint(prefix+'self_attn.'+k).to(device) for k in self.attention.state_dict()}
        self.attention.load_state_dict(weights,assign=True)
        self.norm.load_state_dict({'weight':checkpoint(prefix+'input_layernorm.weight').to(device)},assign=True)
        self.rotary = Qwen3RotaryEmbedding(self.config,device=torch.device(device))
        self.attention.eval(); self.norm.eval()
        self.last_probability = None
        self.last_keys = None

    def context(self, full_h12, ids, *, permuted_values=False, return_probability=False):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        t = self.torch
        assert len(full_h12) == len(ids)
        with t.inference_mode():
            x = t.as_tensor(np.asarray(full_h12),dtype=t.bfloat16,device=self.device)[None]
            n = x.shape[1]; a = self.attention
            u = self.norm(x)
            shape = (1,n,-1,a.head_dim)
            q = a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
            k = a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
            v = a.v_proj(u).view(shape).transpose(1,2)
            cos,sin = self.rotary(x,t.arange(n,device=self.device)[None])
            q,k = apply_rotary_pos_emb(q,k,cos,sin)
            k,v = repeat_kv(k,a.num_key_value_groups),repeat_kv(v,a.num_key_value_groups)
            logits = (q[:,:,-1:]@k.transpose(-1,-2))*a.scaling
            prob = logits.softmax(-1,dtype=t.float32).to(x.dtype)
            if permuted_values:
                seed = int(hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()[:16],16)
                order = np.random.default_rng(seed).permutation(n)
                # Reassign values only. Keep Q/K, rotary positions and query fixed.
                v = v[:,:,t.as_tensor(order,device=self.device)]
            context = (prob@v).transpose(1,2).reshape(1,-1)
            answer = context[0].float().cpu().numpy()
            if return_probability:
                return answer,prob[0,:,0].float().cpu().numpy()
            return answer

    def check_native(self,h,ids):
        t = self.torch
        with t.inference_mode():
            x = t.as_tensor(np.asarray(h),dtype=t.bfloat16,device=self.device)[None]
            n = len(h); pos = t.arange(n,device=self.device)[None]
            mask = t.full((n,n),t.finfo(x.dtype).min,dtype=x.dtype,device=self.device).triu(1)[None,None]
            direct,p = self.attention(self.norm(x),position_embeddings=self.rotary(x,pos),attention_mask=mask)
            context,prob = self.context(h,ids,return_probability=True)
            out = self.attention.o_proj(t.as_tensor(context,dtype=t.bfloat16,device=self.device)[None])[0]
            # Context uses a single-query GEMM, full native block uses all queries; record floor.
            mse = float(((out.float()-direct[0,-1].float())**2).mean())
            relative = mse/max(float(direct[0,-1].float().square().mean()),1e-20)
            diff = float(np.max(np.abs(prob-p[0,:,-1].float().cpu().numpy())))
            assert relative < .002 and diff < .03, (relative,diff)
            return {'output_relative_MSE_different_GEMM_shape':relative,'attention_max_abs_difference':diff,
                    'native_parameter_all_heads_sources':True,'full_context_width':len(context)}

    def close(self):
        del self.attention,self.norm,self.rotary
        gc.collect()
        self.torch.cuda.empty_cache()
