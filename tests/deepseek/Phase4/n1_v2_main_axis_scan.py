# -*- coding: utf-8 -*-
"""
N1 v2: 主轴层扫描（embed -> layer -> unembed）  预注册见 N1_design_seal.json
修正 v1 三处缺陷：(1) 末尾公共 token 使 L0 sep==0 非退化；(2) 每词 8 框架最小对 -> 大数据量；
                 (3) C 段扩样 + 目标词句尾（目标决定输出）regime。
"""
import os, json, time, math, sys
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TAG = MODEL.replace('/', '_')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n1v2_report_%s.txt' % TAG)
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
L = len(_core.layers)
w('=== N1 v2 主轴层扫描 ===')
w('time %s  model=%s  layers=%d  load=%.1fs' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, time.time() - t0))

def ids_of(s): return tok.encode(s, add_special_tokens=False)
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.dot(a, b) / (na * nb))

@torch.no_grad()
def run(text=None, embeds=None):
    if embeds is None:
        ii = torch.tensor([ids_of(text)], device='cuda')
        out = model(input_ids=ii, output_hidden_states=True)
    else:
        out = model(inputs_embeds=embeds, output_hidden_states=True)
    hs = [h[0].float().cpu().numpy() for h in out.hidden_states]
    return hs, out.logits[0].float().cpu().numpy()

norm = _core.norm
head = model.lm_head
def lens_p(hvec):
    with torch.no_grad():
        z = head(norm(torch.tensor(hvec, device='cuda').to(torch.bfloat16))).float()
        return torch.softmax(z, -1)
def jsd(pa, pb):
    m = 0.5 * (pa + pb)
    return float(0.5 * (torch.sum(pa * torch.log((pa + 1e-12) / (m + 1e-12))) +
                        torch.sum(pb * torch.log((pb + 1e-12) / (m + 1e-12)))))

FRAMES = [
    '他昨天在店里买的那个{w}，', '我朋友上周送我的那个{w}，', '新闻里今天提到的那个{w}，',
    '她在网上搜了很久的那个{w}，', '老师上课讲到的那个{w}，', '我们昨天聊起的那个{w}，',
    '他手机里保存的那个{w}，', '姑姑从老家带来的那个{w}，',
]
POLY = {
    '苹果': ('切开尝了一口，味道特别甜', '开机用了半天，性能特别强'),
    '小米': ('煮了一锅粥，闻起来特别香', '开机用了半天，性能特别强'),
    '病毒': ('让他发烧了三天，医生说要休息', '让电脑文件全丢了，工程师说要重装'),
    '杜鹃': ('开在山坡上，花瓣是粉色的', '停在枝头上，叫声很清脆'),
}
CTRL_SAME = {'苹果': '香蕉', '小米': '面条', '病毒': '感冒', '杜鹃': '麻雀'}
CTRL_NEU = ('仔细看了半天，做工很精细', '仔细看了半天，价格很便宜')
TAIL = '。'

# ---------------- A 段 ----------------
def curve_pair(sa, sb):
    ha, _ = run(sa); hb, _ = run(sb)
    n = min(len(ha[0]), len(hb[0]))
    sep = [1.0 - cos(ha[l][-1], hb[l][-1]) for l in range(L + 1)]
    js = [jsd(lens_p(ha[l][-1]), lens_p(hb[l][-1])) for l in range(L + 1)]
    return np.array(sep), np.array(js)

S_sense, J_sense, S_ctrlA, J_ctrlA, S_ctrlB, J_ctrlB = [], [], [], [], [], []
perword_peak = {}
for wd, (sf, sc) in POLY.items():
    acc = []
    for fr in FRAMES:
        a = fr.format(w=wd) + sf + TAIL
        b = fr.format(w=wd) + sc + TAIL
        s, j = curve_pair(a, b); S_sense.append(s); J_sense.append(j); acc.append(s)
    perword_peak[wd] = int(np.argmax(np.mean(acc, 0)[1:]) + 1)
    cw = CTRL_SAME[wd]
    if len(ids_of(cw)) == 1:
        for fr in FRAMES:
            s, j = curve_pair(fr.format(w=cw) + sf + TAIL, fr.format(w=cw) + sc + TAIL)
            S_ctrlA.append(s); J_ctrlA.append(j)
    for fr in FRAMES:
        s, j = curve_pair(fr.format(w=cw or '桌子') + CTRL_NEU[0] + TAIL,
                          fr.format(w=cw or '桌子') + CTRL_NEU[1] + TAIL)
        S_ctrlB.append(s); J_ctrlB.append(j)

UNREL = ['他昨天在店里买的那个杯子，仔细看了半天，做工很精细。',
         '新闻里今天提到的那个公司，仔细看了半天，价格很便宜。',
         '老师上课讲到的那个公式，仔细看了半天，做工很精细。',
         '她网上搜了很久的那个方法，仔细看了半天，价格很便宜。',
         '我们昨天聊起的那个计划，仔细看了半天，做工很精细。',
         '他手机里保存的那个照片，仔细看了半天，价格很便宜。',
         '姑姑从老家带来的那个特产，仔细看了半天，做工很精细。',
         '我朋友上周送我的那个礼物，仔细看了半天，价格很便宜。']
N_pairs = 8
S_null = []
for i in range(0, N_pairs, 2):
    s, j = curve_pair(UNREL[i], UNREL[i + 1]); S_null.append(s)

def agg(lst):
    M = np.mean(lst, 0); SD = np.std(lst, 0); n = len(lst)
    return M, SD, n
Ms, SDs, ns = agg(S_sense); Mc, SDc, nc = agg(S_ctrlA + S_ctrlB); Mn, SDn, nn = agg(S_null)
dS = Ms - Mc
sem = np.sqrt(SDs ** 2 / ns + SDc ** 2 / nc)
z = dS / np.maximum(sem, 1e-9)
Js = np.mean(J_sense, 0); Jc = np.mean(J_ctrlA + J_ctrlB, 0); Jn = np.mean(S_null and [np.zeros(L + 1)] or [np.zeros(L + 1)], 0)

w('')
w('--- A 段：意义分流（末公共 token 位置；n_sense=%d n_ctrl=%d n_null=%d）---' % (ns, nc, nn))
w('L   sep_sense  sep_ctrl   dS      z       JSD_sense JSD_ctrl sep_null')
for l in range(L + 1):
    w('%2d  %.4f    %.4f    %+.4f  %+6.2f   %.4f    %.4f    %.4f' %
      (l, Ms[l], Mc[l], dS[l], z[l], Js[l], Jc[l], Mn[l]))
sig = [l for l in range(L + 1) if abs(z[l]) >= 2.0]
w('  |z|>=2 层: %s' % sig)
w('  dS peak @L%d=%.4f ; JSD peak @L%d=%.4f ; 每词 dS 峰层: %s' %
  (int(np.argmax(dS)), dS.max(), int(np.argmax(Js)), Js.max(),
   ', '.join('%s@L%d' % (k, perword_peak[k]) for k in POLY)))

# ---------------- B 段 ----------------
w('')
w('--- B 段：层次 margin_l = cos(h_l(W),水果) - cos(h_l(W),食物)（"这是{W}。" W 位置）---')
INST = ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '樱桃', '柠檬', '橘子']
INST = [x for x in INST if len(ids_of(x)) == 1]
CTXS = ['这是%s。', '我想说的是%s。', '他提到的%s。']
def first_ok(cands):
    for x in cands:
        if len(ids_of(x)) == 1: return x
    return None
SUP1 = first_ok(['水果']); SUP2 = first_ok(['食物', '东西', '物体', '饮料'])
CTRLW = first_ok(['桌子', '椅子', '沙发', '石头', '杯子'])
H = {}
need = INST + [x for x in [SUP1, SUP2, CTRLW] if x]
for wd in need:
    acc = []
    for c in CTXS:
        hs, _ = run(c % wd)
        p = len(ids_of(c.split('%s')[0]))
        acc.append(np.stack([hs[l][p] for l in range(L + 1)], 0))
    H[wd] = np.mean(acc, 0)
INST = [x for x in INST if x in H]
w('  instances(n=%d)=%s | SUP1=%s SUP2=%s CTRL=%s' % (len(INST), ' '.join(INST), SUP1, SUP2, CTRLW))
marg = (np.array([np.mean([cos(H[x][l], H[SUP1][l]) - cos(H[x][l], H[SUP2][l]) for x in INST]) for l in range(L + 1)])
        if (SUP1 and SUP2) else np.zeros(L + 1))
marg_ban = (np.array([np.mean([cos(H[x][l], H[SUP1][l]) - cos(H[x][l], H[CTRLW][l]) for x in INST]) for l in range(L + 1)])
            if (SUP1 and CTRLW) else np.zeros(L + 1))
w('L   margin(%s-%s)  margin(%s-%s)' % (SUP1, SUP2, SUP1, CTRLW))
for l in range(L + 1):
    w('%2d   %+.4f       %+.4f' % (l, marg[l], marg_ban[l]))
if len(INST) >= 3:
    a0 = max(1, L // 4); a1 = max(2, L // 2)
    seg = marg[a0:a1 + 1]
    w('  margin L0=%.4f  peak @L%d=%.4f  ratio=%.2fx' % (marg[0], int(np.argmax(marg)), marg.max(), marg.max() / max(abs(marg[0]), 1e-9)))
    w('  margin 中层(L%d-L%d) 均值=%.4f -> 相对 L0 保留率=%.0f%%' % (a0, a1, seg.mean(), 100 * seg.mean() / marg[0] if abs(marg[0]) > 1e-9 else 0))

# ---------------- C 段 ----------------
w('')
w('--- C 段：输入端口替换（目标位置嵌入行 -> 同类/异类/随机）---')
emb = model.get_input_embeddings()
def erow(wd):
    return emb(torch.tensor([ids_of(wd)[0]], device='cuda')).detach()
rng = np.random.default_rng(11)
res = {}
for wd, (sf, sc) in POLY.items():
    for fr in FRAMES[:4]:
        for tag, S in (('f', sf), ('c', sc)):
            for regime in ('final', 'suffix'):
                P = fr.format(w=wd) + (S + TAIL if regime == 'suffix' else '')
                ii = ids_of(P); p = len(ids_of(fr.format(w=wd))) - 1
                if regime == 'suffix':
                    p = len(ids_of(fr.format(w=wd)))
                E0 = emb(torch.tensor([ii], device='cuda'))
                _, lg0 = run(embeds=E0); p0 = torch.softmax(torch.tensor(lg0[-1]), -1)
                subs = {'sameclass': CTRL_SAME[wd], 'otherclass': '桌子',
                        'random': None}
                for nm, sw in subs.items():
                    if nm == 'sameclass' and len(ids_of(sw)) != 1: continue
                    r = erow(sw) if sw else emb(torch.tensor([int(rng.integers(0, emb.num_embeddings))], device='cuda')).detach()
                    E1 = E0.clone(); E1[0, p] = r
                    _, lg1 = run(embeds=E1); p1 = torch.softmax(torch.tensor(lg1[-1]), -1)
                    kl = float(torch.sum(p0 * torch.log((p0 + 1e-12) / (p1 + 1e-12))))
                    res.setdefault((nm, regime), []).append((kl, int(torch.argmax(p0)) == int(torch.argmax(p1))))
for k in sorted(res):
    v = res[k]; kls = [x[0] for x in v]
    w('  %-12s %-7s n=%2d  KL mean=%.3f sd=%.3f median=%.3f  top1-preserved=%.2f' %
      (k[0], k[1], len(v), np.mean(kls), np.std(kls), np.median(kls), np.mean([x[1] for x in v])))

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
