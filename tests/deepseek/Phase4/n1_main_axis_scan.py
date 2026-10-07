# -*- coding: utf-8 -*-
"""
N1: 主轴层扫描（embed -> layer -> unembed）
A 段 意义分流：最小对（前缀逐 token 相同，仅后缀不同）；对照词分离"后缀效应"与"目标词路由效应"
B 段 层次锋利化：中性上下文下，track cos(h_l(苹果), h_l(水果)) 等关系在层间的演化（L0 == embedding，可校验 E3b）
C 段 输入端口替换：把目标位置嵌入行替换为同类词/异类词/随机词，测输出 KL 与分流变化
零 GPU 依赖（模型 bf16 上卡）。默认 qwen3-4b。
"""
import os, json, time, math, sys
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TAG = MODEL.replace('/', '_')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n1_report_%s.txt' % TAG)
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
L = len(model.model.layers); H = model.config.hidden_size
w('=== N1 主轴层扫描 ===')
w('time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('model %s  layers=%d hidden=%d  load=%.1fs' % (MODEL, L, H, time.time() - t0))
free, tot = torch.cuda.mem_get_info()
w('cuda free %.1fGB / %.1fGB' % (free / 1e9, tot / 1e9))

def ids_of(s):
    return tok.encode(s, add_special_tokens=False)

@torch.no_grad()
def run(text_or_ids, embeds=None):
    if embeds is None:
        ii = torch.tensor([ids_of(text_or_ids)], device='cuda')
        out = model(input_ids=ii, output_hidden_states=True)
    else:
        out = model(inputs_embeds=embeds, output_hidden_states=True)
    hs = [h[0].float().cpu().numpy() for h in out.hidden_states]  # [L+1, T, H]
    return hs, out.logits[0].float().cpu().numpy()

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.dot(a, b) / (na * nb))

# ---------------- A 段 ----------------
POLY = {
    '苹果': ('他昨天在店里买的那个苹果，', '切开尝了一口，味道特别甜', '开机用了半天，性能特别强'),
    '小米': ('他昨天带回来的那个小米，',   '煮了一锅粥，闻起来特别香', '开机用了半天，性能特别强'),
    '病毒': ('他昨天提到的那个病毒，',     '让他发烧了三天，医生说要多休息', '让电脑文件全丢了，工程师说要重装'),
    '杜鹃': ('他昨天拍的那张杜鹃，',       '开在山坡上，花瓣是粉色的', '停在枝头上，叫声很清脆'),
}
CTRL = {
    '香蕉': ('他昨天在店里买的那个香蕉，', '切开尝了一口，味道特别甜', '开机用了半天，性能特别强'),
    '桌子': ('他昨天在店里买的那个桌子，', '切开尝了一口，味道特别甜', '开机用了半天，性能特别强'),
}

def sep_curve(seq_a, seq_b):
    ha, _ = run(seq_a); hb, _ = run(seq_b)
    return np.array([1.0 - cos(ha[l][-1], hb[l][-1]) for l in range(L + 1)])

w('')
w('--- A 段：意义分流曲线 sep(l)=1-cos(h_l^A, h_l^B) @末位置 ---')
sense_curves = {}
for wed, (P, sf, sc) in POLY.items():
    sense_curves[wed] = sep_curve(P + sf, P + sc)
ctrl_curves = {}
for wed, (P, sf, sc) in CTRL.items():
    ctrl_curves[wed] = sep_curve(P + sf, P + sc)
# 词身份对照：同一后缀、不同词
lex_curves = {}
lex_curves['苹果vs香蕉(f)'] = sep_curve(POLY['苹果'][0] + POLY['苹果'][1], CTRL['香蕉'][0] + CTRL['香蕉'][1])
lex_curves['苹果vs桌子(f)'] = sep_curve(POLY['苹果'][0] + POLY['苹果'][1], CTRL['桌子'][0] + CTRL['桌子'][1])

S_sense = np.mean([sense_curves[k] for k in POLY], axis=0)
S_ctrl = np.mean([ctrl_curves[k] for k in CTRL], axis=0)
S_lex = np.mean([lex_curves[k] for k in lex_curves], axis=0)
dS = S_sense - S_ctrl
w('L    sep_sense  sep_ctrl  dS=diff   sep_lexic')
for l in range(L + 1):
    w('%2d   %.4f     %.4f    %+.4f    %.4f' % (l, S_sense[l], S_ctrl[l], dS[l], S_lex[l]))
w('  peak sep_sense @L%d=%.4f ; peak dS @L%d=%.4f ; argmax dS=%d' %
  (int(np.argmax(S_sense)), S_sense.max(), int(np.argmax(dS)), dS.max(), int(np.argmax(dS))))
w('  per-word peak layer: ' + ', '.join('%s@L%d' % (k, int(np.argmax(v))) for k, v in sense_curves.items()))

# logit lens: 输出分布分叉层
w('')
w('--- A 段：logit lens 输出分叉（末位置）---')
norm = model.model.norm; head = model.lm_head
def lens_div(seq_a, seq_b):
    ha, _ = run(seq_a); hb, _ = run(seq_b)
    ds = []
    for l in range(L + 1):
        with torch.no_grad():
            za = head(norm(torch.tensor(ha[l][-1], device='cuda').to(torch.bfloat16))).float()
            zb = head(norm(torch.tensor(hb[l][-1], device='cuda').to(torch.bfloat16))).float()
            pa = torch.softmax(za, -1); pb = torch.softmax(zb, -1)
            m = 0.5 * (pa + pb)
            js = 0.5 * (torch.sum(pa * torch.log((pa + 1e-12) / (m + 1e-12))) +
                        torch.sum(pb * torch.log((pb + 1e-12) / (m + 1e-12))))
        ds.append(float(js))
    return np.array(ds)
J = []
for wed, (P, sf, sc) in POLY.items():
    J.append(lens_div(P + sf, P + sc))
J = np.mean(J, axis=0)
w('L    '+''.join('%7d' % l for l in range(0, L + 1, 4)))
w('JSD  '+''.join('%7.3f' % J[l] for l in range(0, L + 1, 4)))
w('  JSD peak @L%d=%.4f ; JSD>=0.5*max from L%d' % (int(np.argmax(J)), J.max(),
   int(np.argmax(J >= 0.5 * J.max()))))

# ---------------- B 段 ----------------
w('')
w('--- B 段：层次锋利化（中性上下文 "这是{W}。" 在 W 位置取 h）---')
WORDS = ['苹果', '香蕉', '梨', '水果', '食物', '桌子', '汽车', '铁']
Hw = {}
for wd in WORDS:
    hs, _ = run('这是%s。' % wd)
    p = len(ids_of('这是'))  # W 的位置索引
    Hw[wd] = np.stack([hs[l][p] for l in range(L + 1)], 0)  # [L+1,H]
w('L0 校验（应等于 E3b 的 embedding cos）: cos(苹果,水果)=%.4f cos(苹果,食物)=%.4f cos(苹果,香蕉)=%.4f cos(苹果,梨)=%.4f' %
  (cos(Hw['苹果'][0], Hw['水果'][0]), cos(Hw['苹果'][0], Hw['食物'][0]),
   cos(Hw['苹果'][0], Hw['香蕉'][0]), cos(Hw['苹果'][0], Hw['梨'][0])))
w('L    c(苹,水果) c(苹,食物) c(苹,香蕉) c(苹,梨)  c(苹,桌子)  margin=c(苹,水)-c(苹,食)')
for l in range(L + 1):
    a = Hw['苹果'][l]
    m = (cos(a, Hw['水果'][l]) - cos(a, Hw['食物'][l]))
    w('%2d   %+.4f    %+.4f    %+.4f    %+.4f   %+.4f     %+.4f' %
      (l, cos(a, Hw['水果'][l]), cos(a, Hw['食物'][l]), cos(a, Hw['香蕉'][l]), cos(a, Hw['梨'][l]),
       cos(a, Hw['桌子'][l]), m))
marg = np.array([cos(Hw['苹果'][l], Hw['水果'][l]) - cos(Hw['苹果'][l], Hw['食物'][l]) for l in range(L + 1)])
w('  margin L0=%.4f -> peak @L%d=%.4f (放大 %.2fx)' % (marg[0], int(np.argmax(marg)), marg.max(), marg.max() / max(marg[0], 1e-9)))

# ---------------- C 段 ----------------
w('')
w('--- C 段：输入端口替换（目标位置嵌入行 -> 同类/异类/随机；测输出 KL）---')
emb = model.get_input_embeddings()
def row(wd):
    i = ids_of(wd)
    return emb(torch.tensor([i[0]], device='cuda')).detach()
def run_sub(P, S, wd, repl=None):
    ii = ids_of(P + S)
    p = len(ids_of(P))
    E = emb(torch.tensor([ii], device='cuda'))
    if repl is not None:
        E[0, p] = repl
    hs, lg = run(None, embeds=E)
    return hs, lg, p
import itertools
rng = np.random.default_rng(7)
res = {}
for wed, (P, sf, sc) in POLY.items():
    for nm, S in (('f', sf), ('c', sc)):
        hs0, lg0, p = run_sub(P, S, wed)
        p0 = torch.softmax(torch.tensor(lg0[-1]), -1)
        for sub_nm, sub in (('sametype', '香蕉' if wed == '苹果' else None),
                            ('othertype', '桌子'),
                            ('random', None)):
            if sub_nm == 'sametype' and sub is None:
                continue
            r = row(sub) if sub else emb(torch.tensor([int(rng.integers(0, emb.num_embeddings))], device='cuda')).detach()
            hs1, lg1, _ = run_sub(P, S, wed, repl=r)
            p1 = torch.softmax(torch.tensor(lg1[-1]), -1)
            kl = float(torch.sum(p0 * torch.log((p0 + 1e-12) / (p1 + 1e-12))))
            t1_0 = int(torch.argmax(p0)); t1_1 = int(torch.argmax(p1))
            res.setdefault(sub_nm, []).append((kl, t1_0 == t1_1))
for k, v in res.items():
    kls = [x[0] for x in v]; same = np.mean([x[1] for x in v])
    w('  %-10s n=%d  KL mean=%.3f median=%.3f  top1-preserved=%.2f' % (k, len(v), np.mean(kls), np.median(kls), same))

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
