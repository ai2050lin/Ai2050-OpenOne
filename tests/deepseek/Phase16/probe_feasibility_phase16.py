# -*- coding: utf-8 -*-
"""Phase 16 (N2h1-alpha-9) 可行性探针。

目的（在 seal 冻结之前）：
  (a) 三臂能否在 layer 1..5 上挂钩（结构上：layers[i] 存在、hidden_states[i+1] 存在）；
  (b) 浅位点的替换效应是否**有限且可测**（dDonor 有限、非零、可归一化为 rho = dDonor/FULL_SWAP_15）；
  (c) 每臂每前向耗时 -> 预估 23 位点 x 14 alpha x 24 对 的正式成本；
  (d) 复述已知的 T=2 布局与 F1b 逐臂 tokenizer 解析（新口径的前置件）。

用法：
  ARMS=A0 python probe_feasibility_phase16.py      # 只探一臂
  python probe_feasibility_phase16.py              # 三臂串行

注意：三臂在同一进程内连续加载 nf4 大模型会段错误（Phase 15 实测），
      故本探针默认由外层驱动逐臂单独进程调用。
"""
import os
import io
import json
import time
import hashlib
import gc

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
P16 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P16, exist_ok=True)
os.makedirs(P16T, exist_ok=True)

EX = json.load(io.open(os.path.join(P15T, 'execution_phase15.json'), encoding='utf-8'))
RES15 = json.load(io.open(os.path.join(P15T, 'result_phase15.json'), encoding='utf-8'))

ARMS_SEL = os.environ.get('ARMS', '')
PROBE_SITES = [1, 2, 3, 4, 5, 6, 7]
PROBE_PAIRS = 3          # 探针只用前 3 对（速度优先）

TMPL = EX['template']
SUPS = list(EX['classes'])
DISC = [tuple(x) for x in EX['discovery']]
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
QUANT = EX['quant']

LOGL = []


def w(s=''):
    LOGL.append(str(s))
    print(s)
    import sys
    sys.stdout.flush()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def probe_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    w('')
    w('=' * 78)
    w('PROBE ARM %s (%s)' % (arm_id, acfg['model']))
    t_arm = time.time()
    MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])
    csha = sha(os.path.join(MDIR, 'config.json'))
    assert csha == acfg['config_sha256'], 'F0 config 漂移 %s' % arm_id
    w('  F0 config sha8 = %s OK' % csha[:8])

    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    def ids_of(s):
        return tok.encode(s, add_special_tokens=False)

    # (d) F1b 逐臂解析
    SUP_ID = {}
    for _wd in SUPS:
        _t = list(tok.encode(_wd, add_special_tokens=False))
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, 'F1b %r -> %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    w('  F1b sup_id = %s' % json.dumps(SUP_ID, ensure_ascii=False))

    # (d) T=2
    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    w('  F1 T-len hist = %s' % {k: len(v) for k, v in sorted(tl.items())})
    assert sorted(tl.keys()) == [2], 'T=2 布局不成立'

    max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT['max_memory'].items()}
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type=QUANT['bnb_4bit_quant_type'],
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=bool(QUANT['bnb_4bit_use_double_quant']))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, trust_remote_code=True,
        attn_implementation=QUANT['attn_implementation'], low_cpu_mem_usage=True,
        device_map=QUANT['device_map'], max_memory=max_mem)
    model.eval()
    w('  loaded %.1fs' % (time.time() - t0))
    layers = model.model.layers
    L = len(layers)
    HID = model.config.hidden_size
    devs = {}
    for nm_, p in model.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + 1
    w('  L=%d HID=%d param_devices=%s' % (L, HID, devs))

    @torch.no_grad()
    def fwd_patch(text, site, vec):
        ii = torch.tensor([ids_of(text)], device='cuda')
        mod = layers[site]

        def hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            t[0, -1, :] = vec.to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

        h = mod.register_forward_hook(hook)
        try:
            o = model(input_ids=ii)
        finally:
            h.remove()
        return o.logits[0, -1].float().detach().cpu().numpy()

    @torch.no_grad()
    def capture(text):
        ii = torch.tensor([ids_of(text)], device='cuda')
        o = model(input_ids=ii, output_hidden_states=True)
        HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in o.hidden_states], 0)
        return HH, o.logits[0, -1].float().detach().cpu().numpy()

    def score_of(v, sup, sid):
        v = v.copy(); v[sid] = -1e9
        own = float(v[SUP_ID[sup]])
        others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
        return own - float(np.mean(others))

    # 结构检查：layers[1..5] 存在 + hidden_states[s+1] 存在
    ii0 = torch.tensor([ids_of(TMPL % '苹果')], device='cuda')
    with torch.no_grad():
        lg_a = model(input_ids=ii0).logits[0, -1].float().detach().cpu().numpy()
        lg_b = model(input_ids=ii0).logits[0, -1].float().detach().cpu().numpy()
        o0 = model(input_ids=ii0, output_hidden_states=True)
    det = float(np.max(np.abs(lg_a - lg_b)))
    nh = len(o0.hidden_states)
    w('  (a) 结构: n_layers=%d ; len(hidden_states)=%d ; determinism=%.3e' % (L, nh, det))
    assert nh >= max(PROBE_SITES) + 1, 'hidden_states 长度不足'

    # capture 前 PROBE_PAIRS 对
    DP = [p for p in PAIRS_ALL if p[0] in [x[0] for x in DISC]][:PROBE_PAIRS]
    CAP = {}
    for wd, sup in INST_ALL:
        CAP[wd] = capture(TMPL % wd)
    BASE = {}
    for (rw, rs, dw, ds, sw) in DP:
        lg = CAP[rw][1]
        BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]),
                        sd0=score_of(lg, ds, ids_of(dw)[0]))
    bad = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
    w('  F2 base n=%d bad=%s ; sr0 mean=%+.3f' %
      (len(BASE), bad if bad else 'NONE', float(np.mean([b['sr0'] for b in BASE.values()]))))

    FULL_SWAP_15 = float(RES15['E2_full_swap'][arm_id]['FULL_SWAP'])
    LSTAR_15 = int(RES15['E3_localize'][arm_id]['L_star_own'])
    w('  冻结锚: FULL_SWAP_15=%.6f ; L*_own(15)=%d' % (FULL_SWAP_15, LSTAR_15))

    # (b)(c) 逐位点单点替换 alpha=1
    rows = []
    t_list = []
    for s in PROBE_SITES:
        t0 = time.time()
        dd = 0.0
        per = []
        for (rw, rs, dw, ds, sw) in DP:
            hr = CAP[rw][0][s + 1].astype(np.float32)
            hd = CAP[dw][0][s + 1].astype(np.float32)
            dv = hd - hr
            q = float(np.linalg.norm(dv) / max(np.linalg.norm(hr), 1e-9))
            lg = fwd_patch(TMPL % rw, s, torch.tensor(hr + 1.0 * dv, device='cuda'))
            x1 = score_of(lg, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
            dd += x1
            per.append(round(float(x1), 4))
        dt = time.time() - t0
        t_list.append(dt)
        dd /= max(len(DP), 1)
        rows.append(dict(site=s, dDonor=dd, rho=dd / FULL_SWAP_15, q_ell=q,
                         per=per, sec=round(dt, 2)))
        w('    L%-3d dDonor=%+9.4f  rho=%+8.4f  q=%7.4f  %d fw / %.2fs' %
          (s, dd, dd / FULL_SWAP_15, q, len(DP), dt))

    per_fw = float(np.mean(t_list)) / max(len(DP), 1)
    est = per_fw * 23 * 14 * 24
    w('  (c) 每前向 %.4f s -> 正式 23 位点x14a x24对 = %d fw ≈ %.0f s/臂' %
      (per_fw, 23 * 14 * 24, est))
    w('  ARM %s probe done %.1fs' % (arm_id, time.time() - t_arm))

    rec = dict(arm=arm_id, model=acfg['model'], L=L, HID=HID, param_devices=devs,
               determinism_maxdiff=det, hidden_levels=nh, sup_id=SUP_ID,
               F2_base_bad=bad, FULL_SWAP_15=FULL_SWAP_15, L_star_15=LSTAR_15,
               rows=rows, per_forward_s=per_fw, est_formal_s_per_arm=est,
               elapsed_s=round(time.time() - t_arm, 1))
    del model, CAP
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main():
    w('Phase 16 feasibility probe  clock=%s  ARMS=%r' % (time.strftime('%Y-%m-%d %H:%M:%S'), ARMS_SEL))
    w('probe_sites=%s probe_pairs=%d' % (PROBE_SITES, PROBE_PAIRS))
    order = list(EX['arm_order'])
    if ARMS_SEL:
        sel = [x.strip() for x in ARMS_SEL.split(',') if x.strip()]
        order = [a for a in order if any(a == s or a.startswith(s + '_') or a.startswith(s) for s in sel)]
    assert order, 'ARMS=%r 没有匹配到任何臂；可用: %s' % (ARMS_SEL, EX['arm_order'])
    out = {}
    for a in order:
        tag = a.split('_')[0]
        try:
            out[a] = probe_arm(a, EX['arms'][a])
        except Exception:
            import traceback
            w('!! PROBE %s FAILED' % a)
            w(traceback.format_exc())
            out[a] = dict(arm=a, error=traceback.format_exc()[-2000:])
    pp = os.path.join(P16T, '_probe_feasibility_%s.json' % (ARMS_SEL or 'ALL'))
    io.open(pp, 'w', encoding='utf-8', newline='\n').write(json.dumps(out, ensure_ascii=False, indent=1))
    io.open(os.path.join(P16T, '_probe_feasibility_%s.txt' % (ARMS_SEL or 'ALL')), 'w',
            encoding='utf-8', newline='\n').write('\n'.join(LOGL) + '\n')
    w('WROTE %s' % pp)


if __name__ == '__main__':
    main()
