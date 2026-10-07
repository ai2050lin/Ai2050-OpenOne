# -*- coding: utf-8 -*-
"""
Phase 19 可行性探针：量化口径（nf4 vs bf16）对**写入向量谱 w_ell** 及其质心 com_V 的影响。

动机（承 Phase 17 seal 的 quant.why 原文）：
  "bf16 下 qwen3-14b (29.5GB) 需 CPU 侧 >15GB，本机 RAM 可用 ~17GB -> 加载期被硬杀 ... 为保持三臂
   同一数值口径统一改为 nf4。A0 臂专职量化对其结论的影响。"
  => P19 死线：在 **A0 同模型同尺度** 上以 bf16 复算 w_ell 谱与 com_V，排除「深端集中」是量化地板效应。

三问：
  Q-a 各臂 bf16 在本机能否加载（成功与否 / 显存与 RAM 峰值 / device 分布 / 加载秒）
  Q-b 同口径（同 INST/PAIRS/U_l/区间求和）下，nf4 与 bf16 的 w_ell 谱与 com_V 差多少
  Q-c 同位点残差量级 median(|w_nf4 - w_bf16| / w_bf16)（量化误差的直观尺度）

用法（逐配置进程隔离，防 OOM 互相污染）：
  PROBE_ARM=A0 PROBE_QUANT=nf4  NPAIR=4 $PY probe_feasibility_phase19.py
  PROBE_ARM=A0 PROBE_QUANT=bf16 NPAIR=4 $PY ...
  PROBE_ARM=A1 PROBE_QUANT=bf16 NPAIR=4 $PY ...
  PROBE_ARM=A2 PROBE_QUANT=bf16 NPAIR=4 $PY ...
产物：tests/deepseek_temp/Phase19/_probe19_<ARM>_<QUANT>.{json,txt}
"""
import os
import sys
import io
import json
import time
import gc
import hashlib

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
os.makedirs(P19T, exist_ok=True)

EX = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
DISC = [tuple(x) for x in EX['discovery']]
ARM_CFG = EX['arms']
ANCH = EX['anchor_values']
QUANT17 = EX['quant']
NBW = int(EX['neighbourhood_width'])
MAP = {'A0': 'A0_calib_qwen3-4b-nf4', 'A1': 'A1_glm4-9b-nf4', 'A2': 'A2_qwen3-14b-nf4'}

ARM = os.environ.get('PROBE_ARM', 'A0')
QUANT = os.environ.get('PROBE_QUANT', 'both')
NPAIR = int(os.environ.get('NPAIR', '0'))   # 0 = 全量（全 41 实例估 U_l，与 P17 同口径）
LOADONLY = os.environ.get('PROBE_MODE', '') == 'loadonly'
ARMID = MAP[ARM]
ACFG = ARM_CFG[ARMID]
MDIR = os.path.join(ROOT, 'models', 'hf', ACFG['dir'])
REACH = [int(x) for x in ANCH[ARMID]['reach']]

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def ram():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullTotalPhys / 1e9, 2), round(m.ullAvailPhys / 1e9, 2)


def gpu_mem():
    try:
        free, total = torch.cuda.mem_get_info()
        return round(free / 2 ** 30, 2), round(total / 2 ** 30, 2)
    except Exception:
        return None, None


# ------------------------------------------------------------------ 统计口径（与 P17 逐字一致）
def com_of_mass(mass_by_site, sites):
    """区间求和 + 中点质心（P17 §com_of_mass 逐字同口径）。"""
    s = np.asarray(sites, float)
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    return float((vals * mid).sum() / den), vals


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


# ------------------------------------------------------------------ 加载
def load_model(scheme):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    t0 = time.time()
    if scheme == 'nf4':
        from transformers import BitsAndBytesConfig
        max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT17['max_memory'].items()}
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type=QUANT17['bnb_4bit_quant_type'],
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=bool(QUANT17['bnb_4bit_use_double_quant']))
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb, trust_remote_code=True,
            attn_implementation=QUANT17['attn_implementation'], low_cpu_mem_usage=True,
            device_map=QUANT17['device_map'], max_memory=max_mem)
    else:
        # bf16：除量化外与 nf4 臂逐项一致（同 attn_implementation / device_map / max_memory）
        max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT17['max_memory'].items()}
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation=QUANT17['attn_implementation'], low_cpu_mem_usage=True,
            device_map=QUANT17['device_map'], max_memory=max_mem)
    model.eval()
    return tok, model, round(time.time() - t0, 1)


def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ('o_proj', 'dense', 'out_proj'):
        if hasattr(a, nm):
            return getattr(a, nm), nm
    raise RuntimeError('no attn out proj')


def mod_dev(m):
    for p in m.parameters():
        return p.device
    return torch.device('cuda')


# ------------------------------------------------------------------ 测量
def measure(tok, model, pairs):
    layers = model.model.layers
    L = len(layers)
    CFG = model.config
    HID = int(CFG.hidden_size)
    NH = int(CFG.num_attention_heads)
    HDP = int(getattr(CFG, 'head_dim', HID // max(NH, 1)))
    OPR = [attn_out_proj(layers[l])[0] for l in range(L)]
    MLPS = [layers[l].mlp for l in range(L)]
    OIN = int(OPR[0].in_features)
    INDEV = model.get_input_embeddings().weight.device

    def ids_of(s):
        return tok.encode(s, add_special_tokens=False)

    def ids_t(text):
        return torch.tensor([ids_of(text)], device=INDEV)

    # 实例集 = 参与配对的并集
    if NPAIR > 0:
        keep = []
        for p in pairs:
            for x in (p[0], p[2]):
                if x not in keep:
                    keep.append(x)
        inst = [t for t in INST_ALL if t[0] in keep]
        if len(inst) < 4:
            inst = INST_ALL[:8]
    else:
        inst = list(INST_ALL)

    @torch.no_grad()
    def capture(text):
        ii = ids_t(text)
        so, sm = {}, {}
        hs = []
        for l in range(L):
            def mk_o(l):
                def f(mod, args):
                    so[l] = args[0].detach()[0, -1, :].float().cpu().numpy().copy()
                return f

            def mk_m(l):
                def f(mod, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    sm[l] = t.detach()[0, -1, :].float().cpu().numpy().copy()
                return f
            hs.append(OPR[l].register_forward_pre_hook(mk_o(l)))
            hs.append(MLPS[l].register_forward_hook(mk_m(l)))
        out = model(input_ids=ii, output_hidden_states=True)
        for h in hs:
            h.remove()
        HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in out.hidden_states], 0)
        return HH, so, sm

    t0 = time.time()
    CAP = {}
    for wd, sup in inst:
        CAP[wd] = capture(TMPL % wd)
    cap_s = round(time.time() - t0, 1)

    # 类子空间 U_l（level=l+1，全实例按类平均；与 P17 E3 同口径）
    by = {}
    for wd, sup in INST_ALL:
        if wd in CAP:
            by.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by]
    if len(AVAIL) < 2:
        raise RuntimeError('AVAIL=%s <2：实例集不足以估计 U_l' % AVAIL)
    rk = max(len(AVAIL) - 1, 1)
    U = {}
    for l in range(L):
        mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
        D = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        U[l] = Vt[:rk].astype(np.float32)

    NH1 = NH + 1
    blocks = np.zeros((NH1, OIN), np.float32)

    @torch.no_grad()
    def head_blocks(opmod, v):
        blocks[:] = 0.0
        for h in range(NH):
            blocks[h, h * HDP:(h + 1) * HDP] = v[h * HDP:(h + 1) * HDP]
        blocks[NH] = v
        t = torch.tensor(blocks, device=mod_dev(opmod), dtype=torch.bfloat16)
        o = opmod(t).float().cpu().numpy()
        return o[:NH], o[NH]

    def proj(v, Ub):
        return (v @ Ub.T) @ Ub

    # 与 P17 mass_profile 逐字同口径
    A = np.zeros(L - 1)
    AT = np.zeros(L - 1)
    ML = np.zeros(L - 1)
    TP = np.zeros(L - 1)
    DN = np.zeros(L - 1)          # 额外：未投影 ‖d_inc,l‖（对齐 Phase 12 diff_norms 口径）
    n_used = 0
    for (rw, rs, dw, ds, sw) in pairs:
        if rw not in CAP or dw not in CAP:
            continue
        _, OR_, MR = CAP[rw]
        _, OD, MD = CAP[dw]
        n_used += 1
        for l in range(L - 1):
            Ub = U[l]
            hbD, fullD = head_blocks(OPR[l], OD[l])
            hbR, fullR = head_blocks(OPR[l], OR_[l])
            d_attn = fullD - fullR
            d_mlp = MD[l] - MR[l]
            A[l] += float(np.linalg.norm(proj(d_attn + d_mlp, Ub)))
            AT[l] += float(np.linalg.norm(proj(d_attn, Ub)))
            ML[l] += float(np.linalg.norm(proj(d_mlp, Ub)))
            DN[l] += float(np.linalg.norm(d_attn + d_mlp))
            dd = hbD - hbR
            per = np.zeros(NH)
            for h in range(NH):
                per[h] = float(np.linalg.norm(proj(dd[h], Ub)))
            TP[l] += float(per.max())
    n = max(n_used, 1)
    out = dict(n_pairs=n_used, n_inst=len(inst), L=L, HID=HID, NH=NH, HDP=HDP,
               rank=rk, n_classes=len(AVAIL), classes=AVAIL,
               w_all=(A / n).tolist(), w_attn=(AT / n).tolist(), w_mlp=(ML / n).tolist(),
               w_top1=(TP / n).tolist(), d_norm=(DN / n).tolist())
    # com_V（区间求和，REACH 域）
    mA = {int(l): float(x) for l, x in enumerate(A / n)}
    mML = {int(l): float(x) for l, x in enumerate(ML / n)}
    mAT = {int(l): float(x) for l, x in enumerate(AT / n)}
    RE_L = [s for s in REACH if 0 <= s < L - 1]
    com_all, _ = com_of_mass(mA, RE_L)
    com_mlp, _ = com_of_mass(mML, RE_L)
    com_attn, _ = com_of_mass(mAT, RE_L)
    out['com_V'] = com_all
    out['com_V_mlp'] = com_mlp
    out['com_V_attn'] = com_attn
    out['median_reach'] = float(np.median(RE_L)) if RE_L else None
    out['argmax_w_layer'] = int(np.argmax(A / n)) if np.isfinite(A).all() else None
    nb = [l for l in RE_L if abs(l - com_all) <= NBW] if com_all is not None else []
    sall = float(sum(mA[l] for l in nb))
    out['nb'] = nb
    out['share_mlp_nb'] = (float(sum(mML[l] for l in nb)) / sall) if sall > 1e-12 else None
    out['share_attn_nb'] = (float(sum(mAT[l] for l in nb)) / sall) if sall > 1e-12 else None
    out['cap_s'] = cap_s
    return out


def main():
    t_start = time.time()
    w('=' * 78)
    w('Phase 19 探针 | ARM=%s (%s) | QUANT=%s | NPAIR=%d' % (ARM, ARMID, QUANT, NPAIR))
    w('  MDIR=%s' % MDIR)
    rt, ra = ram()
    gf, gt = gpu_mem()
    w('  起始 RAM total/avail = %.1f/%.1f GB ; GPU free/total = %s/%s GB' % (rt, ra, gf, gt))
    DISC_W = set(x[0] for x in DISC)
    # 与 P17 逐字同口径：disc_pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[0]/p[2] in CAP]
    # （INST_ALL 覆盖全部实例，故此处只需 p[0] in DISC_W）。
    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W]
    if NPAIR > 0:
        pairs = pairs[:NPAIR]
    schemes = ['nf4', 'bf16'] if QUANT == 'both' else [QUANT]

    rep = dict(arm=ARM, arm_id=ARMID, n_pair=NPAIR, md5_dir=ACFG['dir'],
               cfg_sha8=ACFG.get('config_sha8'), reach=REACH,
               p17_anchor_com_V=ANCH[ARMID].get('com_V'), runs={})
    for scheme in schemes:
        w('-' * 70)
        w('[%s] 加载中 ...' % scheme)
        rec = dict(scheme=scheme)
        try:
            tok, model, load_s = load_model(scheme)
            rec['load_ok'] = True
            rec['load_s'] = load_s
            dev_hist = {}
            for _, p in model.named_parameters():
                dev_hist[str(p.device)] = dev_hist.get(str(p.device), 0) + 1
            rec['device_hist'] = dev_hist
            rt2, ra2 = ram()
            gf2, gt2 = gpu_mem()
            rec['ram_after'] = [rt2, ra2]
            rec['gpu_after'] = [gf2, gt2]
            w('  加载 %.1fs ; device_hist=%s' % (load_s, dev_hist))
            w('  加载后 RAM avail=%.1f GB ; GPU free=%.1f GB' % (ra2, gf2 if gf2 else -1))
            if LOADONLY:
                with torch.no_grad():
                    _ii = torch.tensor(
                        [tok.encode(TMPL % INST_ALL[0][0], add_special_tokens=False)],
                        device=model.get_input_embeddings().weight.device)
                    _lg = model(input_ids=_ii).logits[0, -1].float().cpu().numpy()
                rec['loadonly'] = True
                rec['fwd_ok'] = bool(np.isfinite(_lg).all())
                w('  [loadonly] 一次前向 OK=%s -> 不计算任何研究量（holdout 保护）' % rec['fwd_ok'])
                rep['runs'][scheme] = rec
                del model
                gc.collect()
                torch.cuda.empty_cache()
                continue
            meas = measure(tok, model, pairs)
            rec.update(meas)
            w('  测量: n_pairs=%d n_inst=%d L=%d rank=%d' % (meas['n_pairs'], meas['n_inst'],
                                                             meas['L'], meas['rank']))
            w('  com_V(all)=%.4f  mlp=%.4f  attn=%.4f  median(REACH)=%.1f  argmax_w@L%s'
              % (meas['com_V'], meas['com_V_mlp'], meas['com_V_attn'], meas['median_reach'],
                 meas['argmax_w_layer']))
            w('  nb=%s share_mlp_nb=%.4f share_attn_nb=%.4f'
              % (meas['nb'], meas['share_mlp_nb'] or -1, meas['share_attn_nb'] or -1))
            w('  w_all@L6=%.4f  w_all@L28=%.4f  d_norm@L6=%.3f  d_norm@L30=%.3f'
              % (meas['w_all'][6], meas['w_all'][28], meas['d_norm'][6], meas['d_norm'][30]))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback as _tb
            rec['load_ok'] = False
            rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
            rec['tb'] = _tb.format_exc()[-2000:]
            w('  [!] 失败: %s' % rec['error'])
            w(_tb.format_exc()[-1400:])
        rep['runs'][scheme] = rec

    # 对照：若两口径都在，算谱对比
    if 'nf4' in rep['runs'] and 'bf16' in rep['runs'] \
            and rep['runs']['nf4'].get('load_ok') and rep['runs']['bf16'].get('load_ok'):
        a = np.asarray(rep['runs']['nf4']['w_all'], float)
        b = np.asarray(rep['runs']['bf16']['w_all'], float)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        rep['compare'] = dict(
            com_V_nf4=rep['runs']['nf4']['com_V'], com_V_bf16=rep['runs']['bf16']['com_V'],
            com_V_delta=abs(rep['runs']['nf4']['com_V'] - rep['runs']['bf16']['com_V']),
            spearman_w=spearman(a, b), median_rel_resid=float(np.median(rel)),
            p90_rel_resid=float(np.percentile(rel, 90)),
            argmax_nf4=rep['runs']['nf4']['argmax_w_layer'],
            argmax_bf16=rep['runs']['bf16']['argmax_w_layer'],
            share_mlp_nb_nf4=rep['runs']['nf4']['share_mlp_nb'],
            share_mlp_nb_bf16=rep['runs']['bf16']['share_mlp_nb'])
        w('=' * 78)
        w('对照（小规模 n=%d）：com_V nf4=%.4f vs bf16=%.4f  delta=%.4f'
          % (NPAIR, rep['compare']['com_V_nf4'], rep['compare']['com_V_bf16'],
             rep['compare']['com_V_delta']))
        w('  spearman(w_nf4,w_bf16)=%s ; median rel resid=%.4f ; p90=%.4f'
          % (rep['compare']['spearman_w'], rep['compare']['median_rel_resid'],
             rep['compare']['p90_rel_resid']))
        w('  argmax nf4=L%s bf16=L%s ; share_mlp_nb nf4=%.4f bf16=%.4f'
          % (rep['compare']['argmax_nf4'], rep['compare']['argmax_bf16'],
             rep['compare']['share_mlp_nb_nf4'] or -1, rep['compare']['share_mlp_nb_bf16'] or -1))

    rep['elapsed_s'] = round(time.time() - t_start, 1)
    rep['log'] = _log
    tag = '%s_%s' % (ARM, QUANT)
    jp = os.path.join(P19T, '_probe19_%s.json' % tag)
    tp = os.path.join(P19T, '_probe19_%s.txt' % tag)
    with io.open(jp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with io.open(tp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_log) + '\n')
    w('[done] %.1fs -> %s' % (rep['elapsed_s'], os.path.basename(jp)))


if __name__ == '__main__':
    main()
