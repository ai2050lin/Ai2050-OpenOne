# -*- coding: utf-8 -*-
"""Phase 3164 轴(b)：G5-A2 图谱缺口② —— RoPE 位置平移族跨模型（3156 协议移植）。

预注册：AGI_GPT5_MEMO L15177（3163 closeout 冻结）。
  (b) 3156 协议：双臂（真实前缀 A / 位置重置 B）x k in {0,1,2,4,8,16,32,64,128}；
      门 = KL_B <= 0.01 + top1_B（3156 实测 KL_B<=0.0023、top1_B 9/9）。
4b 不重跑：引用 3156 已封存 result（res c8c66d9f / seal b54418f2）。
口径（execution.json 冻结）：
  - KL_B = B 臂 last-position 输出分布 vs A0（k=0 真实前缀）输出分布的 KL（逐语言逐 k，全 18 项）。
  - top1_B = B 臂 argmax == A0 argmax（逐语言逐 k，门=18/18）。
  - 辅助登记（非门，跨模型对应性读数）：rope_rel_max = B(k>0) vs A0 逐层 rel disp 最大值
    （3156 实测 1.49e-2，严格 1e-3 门在 4b 已判 violated_strict_tol；本轴以输出级为主门）、
    d0 锚（A0/B0 输入恒等 -> 输出逐位同）、emb 锚（A 臂 emb 层跨 k 位移 ~0）、
    A 臂上下文效应（KL128/top128，前缀真实有效性对照）。
  - 精度：14b = NF4 pre-quantized；glm4 = NF4 现场量化（同轴(a)，known deviation 登记在案）。
  - 单进程（36 序列 x 1 forward，量小；无 per-anchor 必要）。
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
PHASE = 3164
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5a2b_position_shift_cross_model'
MODEL = os.environ.get('P3164B_MODEL', 'qwen3-14b')
SMOKE = os.environ.get('P3164B_SMOKE', '0') == '1'
SUMMARY = (MODEL == 'summary')
assert MODEL in ('qwen3-14b', 'glm4', 'summary'), MODEL
BASE = os.path.join(RDIR, 'phase3164', NAME, MODEL)
if SMOKE:
    BASE = os.path.join(BASE, 'smoke')
os.makedirs(BASE, exist_ok=True)
LOGP = os.path.join(BASE, 'run_log.txt')

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

# ---------------- 冻结材料（逐字 3156） ----------------
TARGETS = {'zh': '我喜欢吃苹果，因为它又甜又多汁。',
           'en': 'I like apples because they are sweet and juicy.'}
PREFIX = {'zh': '今天天气很好，我们在讨论语言模型如何表示语言。',
          'en': 'The weather is fine today and we keep discussing how models represent language. '}
K_FULL = [0, 1, 2, 4, 8, 16, 32, 64, 128]
K_SMOKE = [0, 1, 2]
LANGS = ['zh', 'en']
ARMS = ['A', 'B']
KL_B_GATE = 0.01
MDIR_MAP = {'qwen3-14b': 'Qwen3-14B-bnb-nf4', 'glm4': 'glm4-9b-chat-hf'}
PREC = {'qwen3-14b': 'nf4-pre', 'glm4': 'nf4'}

KG = K_SMOKE if SMOKE else K_FULL
LGS = ['zh'] if SMOKE else LANGS
NSEQ = len(LGS) * len(ARMS) * len(KG)

# ---------------- design freeze ----------------
design = dict(phase=PHASE, name=NAME, model=MODEL, prec=PREC.get(MODEL, 'n/a'), smoke=SMOKE,
              protocol='3156 isomorphic transplant (dual arm A/B x k grid); 4b ref = 3156 sealed c8c66d9f',
              k_grid=KG, langs=LGS, arms=ARMS,
              targets=TARGETS, prefix=PREFIX,
              gates=dict(kl_b_max=KL_B_GATE, top1_b='18/18 (all langs x k>0)',
                         aux_rope_rel_ref='3156 measured 1.49e-2 (registered, not gated)'),
              known_deviation='NF4 precision for 14b/glm4 (Q06/3156 were bf16 4b); '
                              'on-model readout self-consistent',
              execution='single process (36 seqs x 1 forward each)',
              frozen_before='any observation')
eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
exe_sha = hashlib.sha256(eblob).hexdigest()
exe_p = os.path.join(BASE, 'execution.json')
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha'] == exe_sha, 'DESIGN DRIFT'
    log('execution.json match (sha %s)' % exe_sha[:8])
else:
    json.dump({'phase': PHASE, 'name': NAME, 'design_sha': exe_sha, 'design': design,
               'created': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('execution.json FROZEN (sha %s)' % exe_sha[:8])

# ---------------- summary 模式 ----------------
if SUMMARY:
    p3156 = json.load(open(os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family',
                                        'qwen3-4b', 'result.json'), encoding='utf-8'))
    p3156_add = os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b',
                             'result_addendum.json')
    ref = dict(ref='3156 sealed (c8c66d9f/b54418f2; addendum a1b773b1 for KL_B)',
               prec='bf16', kl_b_max=0.0023, top1_b='9/9', rope_rel_max=1.49e-2,
               verdict=p3156['verdict'])
    models = {'qwen3-4b': ref}
    cls_vals = []
    for m in ('qwen3-14b', 'glm4'):
        rp = os.path.join(RDIR, 'phase3164', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        models[m] = dict(ref='3164 axis(b) this phase', prec=r['prec'],
                         kl_b_max=r['kl_b_max'], top1_b='%d/%d' % (r['top1_b_ok'], r['top1_b_tot']),
                         rope_rel_max=r['rope_rel_max'], cls=r['cls'],
                         res_sha8=r['res_sha8'], seal_sha8=r['seal_sha8'])
        cls_vals.append(r['cls'])
    agree = (cls_vals[0] == cls_vals[1])
    summary = dict(phase=PHASE, name=NAME + '_summary', axis='(b) RoPE position shift cross-model',
                   models=models, class_agreement_14b_glm4=agree,
                   verdict='g5a2b_summary|agree_%s|cls14b_%s|clsglm4_%s' % (
                       agree, cls_vals[0], cls_vals[1]),
                   runtime_s=round(time.time() - T0, 1))
    seal_result(summary, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ---------------- torch 路径 ----------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: %s prec=%s' % (MDIR, PREC[MODEL]))
if PREC[MODEL] == 'nf4-pre':
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).eval()
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
cfgm = model.config
NL = int(cfgm.num_hidden_layers)
HID = int(cfgm.hidden_size)
KOUT = NL - 1
MID = NL // 2
log('model loaded NL=%d D=%d mem=%.1f GB' % (NL, HID, torch.cuda.memory_allocated() / 2**30))

TGT_IDS = {lg: tok(TARGETS[lg], add_special_tokens=False)['input_ids'] for lg in LGS}
PRE_IDS = {lg: tok(PREFIX[lg], add_special_tokens=False)['input_ids'] for lg in LGS}
NTGT = {lg: len(TGT_IDS[lg]) for lg in LGS}
NTGT_MAX = max(NTGT.values())
for lg in LGS:
    assert NTGT[lg] >= 8, ('target too short', lg, NTGT[lg])
    assert len(PRE_IDS[lg]) * 16 >= max(KG), ('prefix too short even cycled', lg, len(PRE_IDS[lg]))
log('tgt tokens %s; prefix lens %s' % (NTGT, {lg: len(PRE_IDS[lg]) for lg in LGS}))

SEQS = []
for lg in LGS:
    for arm in ARMS:
        for k in KG:
            SEQS.append((lg, arm, k))
assert len(SEQS) == NSEQ

def make_inputs(lg, arm, k):
    tids = TGT_IDS[lg]
    n = NTGT[lg]
    if k == 0:
        ids = list(tids); mask = [1] * n; pos = list(range(n))
    else:
        pre = (PRE_IDS[lg] * (k // len(PRE_IDS[lg]) + 1))[:k]
        ids = pre + list(tids)
        assert len(ids) == k + n
        if arm == 'A':
            mask = [1] * (k + n); pos = list(range(k + n))
        else:
            mask = [0] * k + [1] * n; pos = [0] * k + list(range(n))
    return ids, mask, pos

i0a, m0a, p0a = make_inputs(LGS[0], 'A', 0)
i0b, m0b, p0b = make_inputs(LGS[0], 'B', 0)
assert i0a == i0b and m0a == m0b and p0a == p0b, 'k=0 arm inputs must be identical'
log('k=0 arm identity assert OK')

H16 = np.zeros((NSEQ, NTGT_MAX, NL + 1, HID), np.float16)
LG16 = np.zeros((NSEQ, int(cfgm.vocab_size)), np.float16)
with torch.no_grad():
    for i, (lg, arm, k) in enumerate(SEQS):
        ids, mask, pos = make_inputs(lg, arm, k)
        n = NTGT[lg]
        o = model(input_ids=torch.tensor([ids], dtype=torch.int64, device='cuda'),
                  attention_mask=torch.tensor([mask], dtype=torch.int64, device='cuda'),
                  position_ids=torch.tensor([pos], dtype=torch.int64, device='cuda'),
                  output_hidden_states=True)
        hs = o.hidden_states
        seg = torch.stack([h[0, k:k + n, :] for h in hs], 0)
        H16[i, :n] = seg.permute(1, 0, 2).float().detach().cpu().numpy().astype(np.float16)
        LG16[i] = o.logits[0, k + n - 1, :].float().detach().cpu().numpy().astype(np.float16)
        del o, hs, seg
        if (i + 1) % 8 == 0:
            log('collect %d/%d' % (i + 1, NSEQ))

# 确定性锚: 重采 2 行
det_max = 0.0
with torch.no_grad():
    for i in [0, NSEQ - 1]:
        lg, arm, k = SEQS[i]
        ids, mask, pos = make_inputs(lg, arm, k)
        n = NTGT[lg]
        o = model(input_ids=torch.tensor([ids], dtype=torch.int64, device='cuda'),
                  attention_mask=torch.tensor([mask], dtype=torch.int64, device='cuda'),
                  position_ids=torch.tensor([pos], dtype=torch.int64, device='cuda'),
                  output_hidden_states=True)
        seg = torch.stack([h[0, k:k + n, :] for h in o.hidden_states], 0)
        hv = seg.permute(1, 0, 2).float().detach().cpu().numpy().astype(np.float16)
        det_max = max(det_max, float(np.abs(hv.astype(np.float32) - H16[i, :n].astype(np.float32)).max()))
        del o
assert det_max < 1e-3, ('determinism check fail', det_max)
log('determinism recheck max=%.3e' % det_max)

npz_p = os.path.join(BASE, 'collect.npz')
np.savez_compressed(npz_p, H=H16, LG=LG16,
                    lang=np.array([s[0] for s in SEQS]),
                    arm=np.array([s[1] for s in SEQS]),
                    k=np.array([s[2] for s in SEQS], np.int32),
                    n_tgt=np.array([NTGT[s[0]] for s in SEQS], np.int32))
npz_sha = hashlib.sha256(open(npz_p, 'rb').read()).hexdigest()[:8]
del model
torch.cuda.empty_cache()
log('model released; npz sha8=%s' % npz_sha)

# ---------------- 分析（CPU） ----------------
z = np.load(npz_p)
H16 = z['H'].astype(np.float32)
LG16 = z['LG'].astype(np.float32)
SEQ_LANG = [s.decode() if isinstance(s, bytes) else str(s) for s in z['lang']]
SEQ_ARM = [s.decode() if isinstance(s, bytes) else str(s) for s in z['arm']]
SEQ_K = [int(v) for v in z['k']]
NTGTV = [int(v) for v in z['n_tgt']]
IDX = {(SEQ_LANG[i], SEQ_ARM[i], SEQ_K[i]): i for i in range(NSEQ)}

def rel_disp(Ha, Hb, n):
    num = float(np.linalg.norm((Ha[:n] - Hb[:n]).ravel()))
    den = float(np.linalg.norm(Ha[:n].ravel())) + 1e-18
    return num / den

def logsoftmax(v):
    v = v - v.max()
    return v - np.log(np.exp(v).sum())

# d0 锚
d0_max = 0.0
for lg in LGS:
    ia, ib = IDX[(lg, 'A', 0)], IDX[(lg, 'B', 0)]
    n = NTGTV[ia]
    d0_max = max(d0_max, float(np.abs(H16[ia, :n] - H16[ib, :n]).max()))
log('d0 anchor max abs diff=%.3e' % d0_max)

# KL_B / top1_B（主门）：B(k) vs A0 输出
KL_B = {}; TOP1_B = {}
klb_max = 0.0; top1_ok = 0; top1_tot = 0
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    p0 = np.exp(logsoftmax(LG16[ia0]))
    for k in KG:
        if k == 0:
            continue
        ib = IDX[(lg, 'B', k)]
        pk = np.exp(logsoftmax(LG16[ib]))
        kl = float((p0 * (np.log(p0 + 1e-30) - np.log(pk + 1e-30))).sum())
        KL_B['%s_k%d' % (lg, k)] = kl
        TOP1_B['%s_k%d' % (lg, k)] = float(int(np.argmax(LG16[ib])) == int(np.argmax(LG16[ia0])))
        klb_max = max(klb_max, kl)
        top1_ok += int(TOP1_B['%s_k%d' % (lg, k)])
        top1_tot += 1
log('KL_B max=%.4e; top1_B %d/%d' % (klb_max, top1_ok, top1_tot))

# 辅助: rope rel disp（B(k>0) vs A0 隐层）
rope = {}; rope_max = 0.0
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    n = NTGTV[ia0]
    for k in KG:
        if k == 0:
            continue
        r = rel_disp(H16[ia0], H16[IDX[(lg, 'B', k)]], n)
        rope['%s_k%d' % (lg, k)] = r
        rope_max = max(rope_max, r)
log('rope rel disp max=%.3e (3156 ref 1.49e-2)' % rope_max)

# 辅助: emb 锚 + A 臂 readout 曲线 + A 上下文效应
curve_A = {}
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    n = NTGTV[ia0]
    for k in KG:
        ia = IDX[(lg, 'A', k)]
        curve_A['%s_k%d' % (lg, k)] = [
            float(np.linalg.norm((H16[ia, :n, l] - H16[ia0, :n, l]).ravel()) /
                  (np.linalg.norm(H16[ia0, :n, l].ravel()) + 1e-18)) for l in range(NL + 1)]
emb_zero = max(max(c[0] for kk, c in curve_A.items() if kk.startswith(lg + '_')) for lg in LGS)
log('emb-layer max disp=%.3e (expected ~0)' % emb_zero)

KL_A = {}; TOP1_A = {}
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    p0 = np.exp(logsoftmax(LG16[ia0]))
    for k in KG:
        ia = IDX[(lg, 'A', k)]
        pk = np.exp(logsoftmax(LG16[ia]))
        KL_A['%s_k%d' % (lg, k)] = float((p0 * (np.log(p0 + 1e-30) - np.log(pk + 1e-30))).sum())
        TOP1_A['%s_k%d' % (lg, k)] = float(int(np.argmax(LG16[ia])) == int(np.argmax(LG16[ia0])))
kmax = max(KG)
kl_a_max = float(np.mean([KL_A['%s_k%d' % (lg, kmax)] for lg in LGS]))
top_a_max = float(np.mean([TOP1_A['%s_k%d' % (lg, kmax)] for lg in LGS]))
log('A-arm context effect: KL_A(k=%d) mean=%.4f top1_A=%.2f' % (kmax, kl_a_max, top_a_max))

# 门（预注册）
g_klb = bool(klb_max <= KL_B_GATE)
g_topb = bool(top1_ok == top1_tot and top1_tot > 0)
cls = 'rope_relative_supported' if (g_klb and g_topb) else 'rope_relative_violated'

verdict = 'g5a2b_%s|klb_%.4f|top1b_%d%d|rope_%.2e|kla_%.3f' % (
    cls, klb_max, top1_ok, top1_tot, rope_max, kl_a_max)
if SMOKE:
    verdict = 'SMOKE_' + verdict

result = dict(phase=PHASE, name=NAME, model=MODEL, smoke=SMOKE, prec=PREC[MODEL],
              design_sha=exe_sha, nl=NL, hidden=HID, readout=KOUT, kmax=kmax,
              n_seq=NSEQ, n_tgt=NTGT, runtime_s=round(time.time() - T0, 1),
              determinism_max=det_max, npz_sha8=npz_sha,
              kl_b={kk: float(v) for kk, v in KL_B.items()},
              top1_b={kk: float(v) for kk, v in TOP1_B.items()},
              kl_b_max=klb_max, top1_b_ok=top1_ok, top1_b_tot=top1_tot,
              rope_rel={kk: float(v) for kk, v in rope.items()}, rope_rel_max=rope_max,
              curve_A={kk: [float(x) for x in v] for kk, v in curve_A.items()},
              kl_a={kk: float(v) for kk, v in KL_A.items()},
              top1_a={kk: float(v) for kk, v in TOP1_A.items()},
              kl_a_kmax_mean=kl_a_max, top1_a_kmax=top_a_max,
              anchors=dict(d0_max_abs=d0_max, emb_layer_max_disp=emb_zero),
              gates=dict(kl_b_max_le=KL_B_GATE, g_klb=g_klb, g_top1b=g_topb),
              cls=cls, verdict=verdict)
seal_result(result, 'smoke_result.json' if SMOKE else 'result.json')
log('DONE runtime=%.1fs' % (time.time() - T0))
