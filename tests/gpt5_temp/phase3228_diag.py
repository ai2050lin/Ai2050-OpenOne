# -*- coding: utf-8 -*-
"""Phase 3228 SMOKE diagnosis: judgment readout degeneracy + C6 gate calibration."""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

SRC_DIR = p94.SRC_DIR
OUT = []
TOK_ISA, TOK_HASA, TOK_SHI = 101158, 18830, 20412
m = 'qwen3-4b'
D, NL = 2560, 36
NH = NL + 1
fin = NH - 1
eps = 1e-6

import glob
from tokenizers import Tokenizer
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
SNAP = glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0]
from safetensors import safe_open
tensors = {}
for fn, keys in [('model-00001-of-00003.safetensors', ['model.embed_tokens.weight']),
                 ('model-00003-of-00003.safetensors', ['model.norm.weight'])]:
    with safe_open(os.path.join(SNAP, fn), framework='pt') as f:
        for k in keys:
            tensors[k] = f.get_tensor(k).float().numpy()
W_U = tensors['model.embed_tokens.weight']
gamma = tensors['model.norm.weight'].reshape(-1).astype(np.float32)
WT = W_U[[TOK_ISA, TOK_HASA]].astype(np.float32)
WTc = W_U[[TOK_SHI, TOK_HASA]].astype(np.float32)
del tensors, W_U


def lnf(X):
    return (X / np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + eps)) * gamma


# ---------- A: smoke npz seen-panel margin analysis ----------
SM = os.path.join(SRC_DIR, 'phase3228', 'g5g26_e3_compute_seal', 'smoke_e3_flip_matrices.npz')
z = np.load(SM)
m0 = z['qwen3-4b_seen_m0']
J0 = z['qwen3-4b_seen_J0']
marg = z['qwen3-4b_seen_margins']  # (14, 3, 208, 2)
d0 = m0[:, 0] - m0[:, 1]
OUT.append('A: seen J0 isa-share=%.4f | d0=m_isa-m_hasa: mean=%.4f std=%.4f '
           'min=%.4f max=%.4f | hasa-side rows=%d'
           % (J0.mean(), d0.mean(), d0.std(), d0.min(), d0.max(), int((J0 == 0).sum())))
PERTS = ['u_isa', 'u_hasa'] + ['rand_%d' % s for s in range(20260390, 20260400)] \
    + ['orth_isa', 'orth_hasa']
for pi_, pn in enumerate(PERTS):
    row = []
    for ai_, a in enumerate((0.5, 1.0, 2.0)):
        d1 = marg[pi_, ai_, :, 0] - marg[pi_, ai_, :, 1]
        row.append('a%s: dD=%.5f flips=%d' % (a, float((d1 - d0).mean()),
                                              int(((d0 <= 0) & (d1 > 0)).sum()
                                                  + ((d0 > 0) & (d1 <= 0)).sum())))
    OUT.append('  %-14s %s' % (pn, ' | '.join(row)))
# char-pair margins
m0c = z['qwen3-4b_seen_margins_char']
dc0 = m0c[:, 0, :, 0] - 0  # placeholder
dc = m0c[0, 0, :, 0] - m0c[0, 0, :, 1]
OUT.append('  char-pair(是/有) baseline d: mean=%.4f std=%.4f'
           % (float(dc.mean()), float(dc.std())))

# ---------- B: 3157 rows — is the relation readout informative there? ----------
z57 = np.load(p94.SRC[p94.NPZ57_KEY[m]])
H57 = z57['H'].astype(np.float32)
RL = z57['rel'].astype(int)
Hf = H57[:, fin, :]
Hk = H57[:, NL - 1, :]
for tag, Hx in (('fin36', Hf), ('kout35', Hk)):
    B = lnf(Hx)
    mm = B @ WT.T
    d = mm[:, 0] - mm[:, 1]
    OUT.append('B 3157 %s: d(is一种-有) isa-rows mean=%.4f std=%.4f | hasa-rows '
               'mean=%.4f std=%.4f | sep=%.4f | acc(sign vs rel)=%.4f'
               % (tag, d[RL == 0].mean(), d[RL == 0].std(),
                  d[RL == 1].mean(), d[RL == 1].std(),
                  float(abs(d[RL == 0].mean() - d[RL == 1].mean())),
                  float((((d > 0) == (RL == 0)).mean()))))
    mmc = B @ WTc.T
    dc = mmc[:, 0] - mmc[:, 1]
    OUT.append('B 3157 %s char(是/有): isa %.4f/%.4f hasa %.4f/%.4f acc=%.4f'
               % (tag, dc[RL == 0].mean(), dc[RL == 0].std(),
                  dc[RL == 1].mean(), dc[RL == 1].std(),
                  float((((dc > 0) == (RL == 0)).mean()))))
# class-token echo on 3157 rows: isa rows contain class C; hasa rows contain part P
CLS_3157 = ['家具', '座椅', '建筑构件', '货币', '橡胶配件', '材料', '出版物', '工具',
            '餐具', '服饰', '日用品', '建筑', '计时工具', '容器', '雨具']
ENTS57 = [('桌子', '家具', '桌腿'), ('椅子', '座椅', '椅背'), ('窗户', '建筑构件', '窗框'),
          ('门', '建筑构件', '门把手'), ('硬币', '货币', '图案'), ('轮胎', '橡胶配件', '花纹'),
          ('纸', '材料', '纤维'), ('书', '出版物', '书页'), ('刀', '工具', '刀刃'),
          ('碗', '餐具', '碗底'), ('衣服', '服饰', '袖子'), ('鞋', '日用品', '鞋底'),
          ('桥', '建筑', '桥墩'), ('房子', '建筑', '屋顶'), ('汽车', '交通工具', '车轮'),
          ('钟表', '计时工具', '表盘')]
tk = Tokenizer.from_file(os.path.join(SNAP, 'tokenizer.json'))
B = lnf(Hf)
# per-row own-token echo: isa rows -> class token; hasa rows -> part token
# vectorized: build token id arrays per row
tid_own = np.zeros(128, dtype=int)
tid_other = np.zeros(128, dtype=int)
for ei, (e, c, p) in enumerate(ENTS57):
    tid_c = tk.encode(c, add_special_tokens=False).ids[0]
    tid_p = tk.encode(p, add_special_tokens=False).ids[0]
    for ri, rel in enumerate(('isa', 'hasa')):
        for pi_, pol in enumerate(('+', '-')):
            for cx in (0, 1):
                r = ei * 8 + ri * 4 + pi_ * 2 + cx
                tid_own[r] = tid_c if rel == 'isa' else tid_p
                tid_other[r] = tid_p if rel == 'isa' else tid_c
# need full W_U for arbitrary token rows: load union of tokens only (below)
uniq_ids = np.unique(np.concatenate([tid_own, tid_other,
                                     [tk.encode('有', add_special_tokens=False).ids[0],
                                      TOK_ISA]]))
with safe_open(os.path.join(SNAP, 'model-00001-of-00003.safetensors'), framework='pt') as f:
    Wfull = f.get_tensor('model.embed_tokens.weight').float().numpy()
sub = Wfull[uniq_ids]
lgs = B @ sub.T  # (128, len(uniq))
pos = {int(t): i for i, t in enumerate(uniq_ids)}
d_own_other = np.array([lgs[r, pos[int(tid_own[r])]] - lgs[r, pos[int(tid_other[r])]]
                        for r in range(128)])
rel_arr = np.array([0 if (r % 8) // 4 == 0 else 1 for r in range(128)])
OUT.append('B 3157 own-vs-other token echo (fin36): isa-rows mean=%.4f hasa-rows '
           'mean=%.4f acc=%.4f' % (d_own_other[rel_arr == 0].mean(),
                                   d_own_other[rel_arr == 1].mean(),
                                   float((((d_own_other > 0) == (rel_arr == 0)).mean()))))
del Wfull, sub, lgs

# ---------- C: C6 calibration — marg gap-stratified agreement ----------
z69 = np.load(p94.SRC['p3169_4b'])
MARG = z69['marg'].reshape(2190, 10)
H69f = z69['H'][:, :, fin, :].reshape(2190, D).astype(np.float32)
del z69
B = lnf(H69f)
with safe_open(os.path.join(SNAP, 'model-00001-of-00003.safetensors'), framework='pt') as f:
    Wfull = f.get_tensor('model.embed_tokens.weight').float().numpy()
from tokenizers import Tokenizer as TKz
CLS = ['水果', '动物', '交通工具', '家具', '金属', '颜色', '乐器', '天气', '运动', '电器']
cls_tok = [TKz.from_file(os.path.join(SNAP, 'tokenizer.json')).encode(c).ids[0] for c in CLS]
c0 = B @ Wfull[cls_tok].T
del Wfull, B, H69f
am_mine = c0.argmax(1)
am_cached = MARG.argmax(1)
gap = np.sort(MARG, axis=1)
top2gap = gap[:, 9] - gap[:, 8]
agree = (am_mine == am_cached)
OUT.append('C: overall agreement=%.4f | cached top2-gap: median=%.4f p25=%.4f p75=%.4f'
           % (agree.mean(), float(np.median(top2gap)), float(np.percentile(top2gap, 25)),
              float(np.percentile(top2gap, 75))))
for thr in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
    msk = top2gap >= thr
    OUT.append('  gap>=%0.2f: rows=%d agreement=%.4f'
               % (thr, int(msk.sum()), float(agree[msk].mean())))
# also: agreement of my fp32 readout with cached on MY OWN top2 gap
gapm = np.sort(c0, axis=1)
mygap = gapm[:, 9] - gapm[:, 8]
for thr in (0.0, 0.05, 0.1, 0.2, 0.5):
    msk = mygap >= thr
    OUT.append('  mygap>=%0.2f: rows=%d agreement=%.4f'
               % (thr, int(msk.sum()), float(agree[msk].mean())))

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3228_diag_out.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('diag done')
