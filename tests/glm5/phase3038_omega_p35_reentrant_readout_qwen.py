# -*- coding: utf-8 -*-
# Phase 3038 - Omega-P35: re-entrant readout anatomy
# The two-step protocol (prefill -> step-2 re-feeds the last
# token at position L, attending to 0..L) has been the readout
# of every intervention phase since 3028. Phase 3037 registered
# the observation that this re-entrant readout differs from the
# direct prefill readout at position L-1 (tok_match 9/11,
# max dp 0.764). This phase quantifies the difference
# systematically: T1 redistribution (dp, token agreement),
# T2 mechanism anatomy (step-2 self-attention mass per layer),
# T3 sharpening (top-1 gain, entropy ratio, top-10 overlap).
# Machine: verbatim 3035/3036 chain (prefill use_cache=True ->
# step-2 single token use_cache=False -> logits[0,-1] double
# -> float64 softmax); pure observation, no intervention.
# PREREG frozen below BEFORE any observation (execution.json
# written pre-run).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3038
NAME = 'omega_p35_reentrant_readout_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
DEEP = tuple(range(21, 35))

GEN_PROMPTS = (
    'The weather was cold, so',
    'He studied every night because',
    'She wanted to buy the car, but',
    'The experiment failed, therefore',
    'You should take an umbrella if',
    'The meeting was long, and',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',
    'We left early because',)
nP = len(GEN_PROMPTS)

PREREG = {
    'mode': 'pure observation, no intervention; eager '
            'attention, bf16, seed 3009; per prompt THREE '
            'passes: (1) direct prefill readout '
            'out.logits[0,-1]; (2) re-entrant chain = '
            'prefill use_cache=True -> step-2 single token '
            'ids[-1] with past, use_cache=False, logits '
            '[0,-1] (verbatim 3035/3036 protocol); (3) '
            'duplicate of (2) with output_attentions=True '
            'for the T2 anatomy; float64 softmax '
            'l-max/exp/sum on double logits',
    'question': 'why and how does the re-entrant readout '
                '(step-2 position L, attends 0..L incl. '
                'itself) differ from the direct prefill '
                'readout (position L-1, attends 0..L-1)? '
                'Is the difference redistributive, '
                'sharpening, or flattening, and does the '
                'step-2 self-attention (own fresh KV at '
                'position L) carry the mechanism?',
    'prompts': 'the 12 GEN_PROMPTS of 3028-3037 verbatim; '
               '11 of them map to the 3036 npz rows '
               'prompt_idx [0..7,9,10,11] for the a64 '
               'cross-phase anchor',
    'T1': 'redistribution: per prompt A = argmax(p_dir); '
          'dp = p_re[A] - p_dir[A]; tok_match = argmax(p_re)'
          ' == A; med|dp|, max dp over 12 (registered '
          'magnitude, no null - deterministic comparison)',
    'T2': 'mechanism anatomy (descriptive): per layer '
          'self_mass = median over 32 heads of attention '
          'from the step-2 query to its own key at '
          'position L; prev_mass = same to position L-1; '
          'deep-band (L21-34) med self vs early (L0-20) '
          'med self',
    'T3': 'sharpening: gain = p_re[A]/p_dir[A]; entropy '
          'ratio H_re/H_dir via logZ form (logZ = max + '
          'log sum exp(l)); top-10 set Jaccard; medians '
          'over 12',
    'verdict_tree': 'if med_gain > 1.05 AND med_H_ratio < '
                    '0.95 -> reentrant_sharpening_qwen; '
                    'elif med_gain < 0.95 AND med_H_ratio '
                    '> 1.05 -> reentrant_flattening_qwen; '
                    'else -> reentrant_redistributive_qwen '
                    '(reachability pre-checked on the 3037 '
                    'dp preview: med|dp| 0.092, '
                    'negligible branch dropped as '
                    'unreachable)',
    'anchors': 'a62 duplicate prefill prompt0 logits '
               'bit-identical (0.0); a63 duplicate '
               're-entrant chain logits bit-identical '
               '(0.0); a64 re-entrant p vs 3036 npz '
               'p0_top[:11,0]: max|dp| <= 1e-4 (a38-'
               'reachable gate) AND argmax identity vs '
               'tok_top[:11,0] per row; a65 '
               'output_attentions=True vs False logits '
               'max|diff| <= 1e-6; a66 source seals '
               '3036/3037 sha8(result) match seal.json',
    'control': 'no intervention -> no sham; the a64 '
               'cross-phase identity is the external '
               'control; a65 guards the attn-flag pass',
    'corrections': 'none yet (run1); follow-up of the '
                   '3037 a60 lesson: readout protocol '
                   'dimensionality must be matched before '
                   'cross-phase comparison; this phase '
                   'quantifies both protocols directly; '
                   'all arrays pre-initialized (3020 '
                   'lesson)',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
# rerun discipline: clear old artifacts
for fn in ('execution.json', 'result.json', 'seal.json',
           NAME + '.npz'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s' % created)

torch.manual_seed(3009)
np.random.seed(3009)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.hidden_size) == HID
log('model loaded')

W_U = model.lm_head.weight.detach().float() \
    .cpu().numpy()
assert W_U.shape == (int(model.config.vocab_size),
                     HID), W_U.shape


def softmax64(lg):
    l = lg - lg.max()
    p = np.exp(l)
    return p / p.sum()


def entropy(lg, p):
    mx = float(lg.max())
    lsum = float(np.log(np.exp(lg - mx).sum())) + mx
    return float(lsum - float((p * lg).sum()))


def prefill_logits(ids):
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    return lg


def reentrant_logits(ids, want_attn=False):
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
        past = out.past_key_values
        out2 = model(
            input_ids=torch.tensor(
                [[int(ids[-1])]], device='cuda'),
            past_key_values=past,
            use_cache=False,
            output_attentions=bool(want_attn))
    lg = out2.logits[0, -1].detach() \
        .double().cpu().numpy()
    att = None
    if want_attn:
        att = [a[0, :, 0, :].detach().float()
               .cpu().numpy().copy()
               for a in out2.attentions]
    return lg, att


tok_ids = []
for pr in GEN_PROMPTS:
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    tok_ids.append(list(int(x) for x in ids))
lens = [len(x) for x in tok_ids]
log('prompt lens=%s' % lens)

# ---------- anchors a62/a63/a65 (prompt 0) ----------
lg_b0 = prefill_logits(tok_ids[0])
lg_d0 = prefill_logits(tok_ids[0])
a62_diff = float(np.max(np.abs(lg_b0 - lg_d0)))
lg_r0, _ = reentrant_logits(tok_ids[0])
lg_r0b, _ = reentrant_logits(tok_ids[0])
a63_diff = float(np.max(np.abs(lg_r0 - lg_r0b)))
lg_a0, att0 = reentrant_logits(tok_ids[0],
                               want_attn=True)
a65_diff = float(np.max(np.abs(lg_a0 - lg_r0)))
log('a62=%.3e a63=%.3e a65=%.3e'
    % (a62_diff, a63_diff, a65_diff))

# ---------- a64: cross-phase identity vs 3036 ----------
z36 = np.load(os.path.join(
    BASE, 'phase3036',
    'omega_p33_fingerprint_curvature_map_qwen',
    'omega_p33_fingerprint_curvature_map_qwen.npz'),
    allow_pickle=True)
pidx36 = [int(x) for x in z36['prompt_idx'][:11]]
tok36 = [int(x) for x in z36['tok_top'][:11, 0]]
p036 = z36['p0_top'][:11, 0]
a64_dp = np.zeros(11)
a64_tok = np.zeros(11, dtype=bool)
for row in range(11):
    pi = pidx36[row]
    lg_r, _ = reentrant_logits(tok_ids[pi])
    p_r = softmax64(lg_r)
    a64_dp[row] = abs(float(p_r[tok36[row]])
                      - float(p036[row]))
    a64_tok[row] = int(np.argmax(p_r)) == tok36[row]
a64_max = float(a64_dp.max())
a64_ok = bool(a64_max <= 1e-4
              and bool(a64_tok.all()))
log('a64 max_dp=%.3e tok=%d/11 ok=%s'
    % (a64_max, int(a64_tok.sum()), a64_ok))

# ---------- a66: source seals ----------
a66_detail = []
for ph, nm in ((3036,
                'omega_p33_fingerprint_curvature_map_'
                'qwen'),
               (3037,
                'omega_p34_kv_situational_'
                'specificity_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a66_detail.append(bool(s == sealj['result_sha256_8']))
a66_ok = bool(a66_detail) and all(a66_detail)

# ---------- main passes ----------
dp_arr = np.zeros(nP)
gain_arr = np.zeros(nP)
H_dir_arr = np.zeros(nP)
H_re_arr = np.zeros(nP)
H_ratio = np.zeros(nP)
jac10 = np.zeros(nP)
tok_match = np.zeros(nP, dtype=bool)
a_dir_arr = np.zeros(nP, dtype=np.int64)
a_re_arr = np.zeros(nP, dtype=np.int64)
p_dirA = np.zeros(nP)
p_reA = np.zeros(nP)
med_self = np.zeros((nP, NL))
med_prev = np.zeros((nP, NL))
for pi in range(nP):
    ids = tok_ids[pi]
    lg_d = prefill_logits(ids)
    p_d = softmax64(lg_d)
    lg_r, _ = reentrant_logits(ids)
    p_r = softmax64(lg_r)
    A = int(np.argmax(p_d))
    a_dir_arr[pi] = A
    a_re_arr[pi] = int(np.argmax(p_r))
    tok_match[pi] = a_re_arr[pi] == A
    dp_arr[pi] = float(p_r[A]) - float(p_d[A])
    p_dirA[pi] = float(p_d[A])
    p_reA[pi] = float(p_r[A])
    gain_arr[pi] = float(p_r[A]) / max(float(p_d[A]),
                                       1e-30)
    H_dir_arr[pi] = entropy(lg_d, p_d)
    H_re_arr[pi] = entropy(lg_r, p_r)
    H_ratio[pi] = H_re_arr[pi] / max(H_dir_arr[pi],
                                     1e-30)
    s_d = set(np.argsort(-p_d)[:10].tolist())
    s_r = set(np.argsort(-p_r)[:10].tolist())
    jac10[pi] = float(len(s_d & s_r)) \
        / max(len(s_d | s_r), 1)
    lg_a, att = reentrant_logits(ids, want_attn=True)
    for li in range(NL):
        med_self[pi, li] = float(np.median(
            att[li][:, -1]))
        med_prev[pi, li] = float(np.median(
            att[li][:, -2]))
    log('P%d len=%d A=%d(%s) dp=%+.4f gain=%.3f '
        'Hratio=%.3f J10=%.2f match=%s'
        % (pi, len(ids), A,
           tok.decode([A]).strip()[:12], dp_arr[pi],
           gain_arr[pi], H_ratio[pi], jac10[pi],
           bool(tok_match[pi])))

med_abs_dp = float(np.median(np.abs(dp_arr)))
max_abs_dp = float(np.max(np.abs(dp_arr)))
med_gain = float(np.median(gain_arr))
med_Hratio = float(np.median(H_ratio))
med_jac = float(np.median(jac10))
n_match = int(tok_match.sum())
self_deep = float(np.median(med_self[:, DEEP]))
self_early = float(np.median(med_self[:, 0:21]))
prev_deep = float(np.median(med_prev[:, DEEP]))

# ---------- verdict ----------
a_ok = bool(a62_diff == 0.0 and a63_diff == 0.0
            and a64_ok and a65_diff <= 1e-6
            and a66_ok)
if med_gain > 1.05 and med_Hratio < 0.95:
    verdict = 'reentrant_sharpening_qwen'
elif med_gain < 0.95 and med_Hratio > 1.05:
    verdict = 'reentrant_flattening_qwen'
else:
    verdict = 'reentrant_redistributive_qwen'

log('=== verdict ===')
log('a62=%r a63=%r a64_ok=%s a65=%.3e a66=%r'
    % (a62_diff, a63_diff, a64_ok, a65_diff, a66_ok))
log('med|dp|=%.4f max|dp|=%.4f tok_match=%d/%d'
    % (med_abs_dp, max_abs_dp, n_match, nP))
log('med_gain=%.4f med_Hratio=%.4f med_J10=%.3f'
    % (med_gain, med_Hratio, med_jac))
log('self-attn med: deep=%.4f early=%.4f '
    'prev_deep=%.4f' % (self_deep, self_early,
                        prev_deep))
log('VERDICT=%s anchor_all_ok=%s'
    % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    prompts=np.array(GEN_PROMPTS),
    prompt_lens=np.array(lens, dtype=np.int64),
    dp=dp_arr, gain=gain_arr,
    H_dir=H_dir_arr, H_re=H_re_arr,
    H_ratio=H_ratio, jac10=jac10,
    tok_match=tok_match,
    a_dir=a_dir_arr, a_re=a_re_arr,
    p_dirA=p_dirA, p_reA=p_reA,
    med_self=med_self, med_prev=med_prev,
    self_deep=np.float64(self_deep),
    self_early=np.float64(self_early),
    prev_deep=np.float64(prev_deep),
    med_abs_dp=np.float64(med_abs_dp),
    max_abs_dp=np.float64(max_abs_dp),
    med_gain=np.float64(med_gain),
    med_Hratio=np.float64(med_Hratio),
    med_jac10=np.float64(med_jac),
    n_match=np.int64(n_match),
    a64_pidx=np.array(pidx36, dtype=np.int64),
    a64_dp=a64_dp, a64_tok=a64_tok,
    a62_diff=np.float64(a62_diff),
    a63_diff=np.float64(a63_diff),
    a64_max=np.float64(a64_max),
    a65_diff=np.float64(a65_diff),
    a66_ok=np.bool_(a66_ok),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'anchor_all_ok': a_ok,
    'anchors': {
        'a62_dup_prefill_bit': a62_diff,
        'a63_dup_reentrant_bit': a63_diff,
        'a64_max_dp_3036': a64_max,
        'a64_gate': 1e-4,
        'a64_tok_identity': bool(a64_tok.all()),
        'a65_attn_flag_diff': a65_diff,
        'a65_gate': 1e-6,
        'a66_source_seals': a66_ok,
    },
    'T1_redistribution': {
        'med_abs_dp': med_abs_dp,
        'max_abs_dp': max_abs_dp,
        'n_tok_match': n_match,
        'dp_per_prompt': [float(v) for v in dp_arr],
        'a_dir': [int(v) for v in a_dir_arr],
        'a_re': [int(v) for v in a_re_arr],
        'p_dirA': [float(v) for v in p_dirA],
        'p_reA': [float(v) for v in p_reA],
    },
    'T2_self_attention': {
        'self_deep_med': self_deep,
        'self_early_med': self_early,
        'prev_deep_med': prev_deep,
        'med_self_per_layer': [float(v) for v in
                               med_self.mean(axis=0)],
    },
    'T3_sharpening': {
        'med_gain': med_gain,
        'med_H_ratio': med_Hratio,
        'med_jac10': med_jac,
        'gain_per_prompt': [float(v)
                            for v in gain_arr],
        'H_ratio_per_prompt': [float(v)
                               for v in H_ratio],
        'jac10_per_prompt': [float(v)
                             for v in jac10],
    },
    'prereg': PREREG,
    'elapsed_s': round(elapsed, 1),
}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()) \
            .hexdigest()[:8]


seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_all_ok': a_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s script8=%s '
    'elapsed=%.1fs'
    % (seal['npz_sha256_8'], seal['result_sha256_8'],
       seal['exec_sha256_8'], seal['script_sha256_8'],
       elapsed))
log('sealed')
