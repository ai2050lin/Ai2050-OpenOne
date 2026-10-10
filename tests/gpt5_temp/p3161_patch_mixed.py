# -*- coding: utf-8 -*-
"""Patch phase3161 script: VRAM-adaptive mixed placement for NF4 models (v2, LF file)."""
import sys

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3161_g4p4_head_attribution.py'
raw = open(P, 'rb').read()
s = raw.decode('utf-8')

OLD = """    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()"""

NEW = """    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    free_v, tot_v = torch.cuda.mem_get_info()
    if free_v > 11.0 * (1 << 30):
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
        log('device_map: all-GPU NF4 (free_vram=%.1f GB)' % (free_v / (1 << 30)))
    else:
        # 14b segfault fix (2026-10-09): bnb CUDA kernel segfaults when WDDM sysmem
        # fallback pages quantized weights (14B untied resident need ~11GB > free).
        # Mixed placement: layers 0..GPU_L-1 NF4 on GPU, layers GPU_L..NL-1 bf16 on
        # CPU (llm_int8_enable_fp32_cpu_offload=True auto-excludes cpu-keyed modules
        # from quantization), embed_tokens/lm_head CPU bf16 (untied; lm_head unused
        # since forward uses BASEM). Injection layer L_mid-1=19 stays on GPU
        # (GPU_L=20); ablation layers 20/21/22 on CPU bf16 -> o_proj input
        # head-zeroing semantics unchanged. hidden_states reads are .float().cpu()
        # (device-agnostic). Measured CPU bf16 matmul 2.1 TFLOPS -> ~9 s/fwd for
        # 20 CPU layers, ~16 min/anchor.
        GPU_L = 20
        assert GPU_L > L_MID - 1, 'injection layer L_mid-1 must stay on GPU'
        assert free_v > 3.8 * (1 << 30), ('vram below floor for GPU_L=20', free_v / (1 << 30))
        bnb2 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True,
                                  llm_int8_enable_fp32_cpu_offload=True)
        dmap = {'model.embed_tokens': 'cpu', 'model.norm': 0, 'lm_head': 'cpu'}
        for _li in range(NL):
            dmap['model.layers.%d' % _li] = 0 if _li < GPU_L else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb2, device_map=dmap, dtype=torch.bfloat16,
            trust_remote_code=True).eval()
        model.config.use_cache = False
        log('device_map: MIXED GPU_L=%d (0..%d NF4 GPU, %d..%d bf16 CPU, embed/lm_head CPU; free_vram=%.1f GB)'
            % (GPU_L, GPU_L - 1, GPU_L, NL - 1, free_v / (1 << 30)))
        _devs = [str(model.model.layers[i].mlp.down_proj.weight.device) for i in (0, GPU_L - 1, GPU_L, NL - 1)]
        assert 'cuda' in _devs[0] and 'cuda' in _devs[1] and _devs[2] == 'cpu' and _devs[3] == 'cpu', \\
            ('placement check failed', _devs)
        log('placement check: L0=%s L%d=%s L%d=%s L%d=%s'
            % (_devs[0], GPU_L - 1, _devs[1], GPU_L, _devs[2], NL - 1, _devs[3]))
        _free_after, _ = torch.cuda.mem_get_info()
        assert _free_after > 0.25 * (1 << 30), ('vram headroom after load too small', _free_after / (1 << 30))
        log('vram headroom after load: %.2f GB' % (_free_after / (1 << 30)))"""

cnt = s.count(OLD)
assert cnt == 1, 'OLD block count=%d (expected 1)' % cnt
s2 = s.replace(OLD, NEW)

# sanity: no bare-CR introduced, still bare-LF file
assert '\r' not in s2, 'CR introduced'
open(P, 'wb').write(s2.encode('utf-8'))

# re-read verify
s3 = open(P, 'rb').read().decode('utf-8')
assert s3 == s2, 'write/read mismatch'
assert s3.count('GPU_L = 20') == 1, 'GPU_L marker count'
assert s3.count('llm_int8_enable_fp32_cpu_offload=True') == 1, 'offload flag count'
import py_compile
py_compile.compile(P, doraise=True)
print('PATCH OK: size %d -> %d, py_compile passed' % (len(raw), len(s3.encode('utf-8'))))
