# -*- coding: utf-8 -*-
"""Q9-A 风格/语域编码的组件定位（Qwen3-4B / GLM-4-9B 通用）

预注册：research/gpt5/docs/PAT_Q9A_PREREG_v1.md（冻结于 2026-11-01 18:50）

问题：同一命题内容在 6 种语域变体（zh/en × 口语/学术/代码注释）下，
  (1) 风格有多少在 hs[0]（token 身份）就存在？
  (2) 剩余的风格信息由哪个组件写进残差流——attn_out 还是 mlp_out？
  (3) 语域能否跨语言迁移？
这是"三块拼合"（embedding / MLP / attention）的第一个组件级切口，
并决定后续 SAE（Goodfire 思路）该挂在哪个组件上。

用法：
    SMOKE=1 .venv/Scripts/python.exe -X utf8 tests/gpt5/pat_q9a.py
    .venv/Scripts/python.exe -X utf8 tests/gpt5/pat_q9a.py
    MODEL=D:/develop/model/ZhipuAI/glm-4-9b-chat-hf Q9A_8BIT=1 .venv/Scripts/python.exe -X utf8 tests/gpt5/pat_q9a.py
"""
import os
import sys
import time
import gc
import json as _json
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import provenance                       # noqa: E402

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


SMOKE = os.environ.get('SMOKE') == '1'
SEED = int(os.environ.get('SEED', '20261101'))
MODEL_PATH = os.environ.get('MODEL', 'D:/develop/model/Qwen/Qwen3-4B')
IS_GLM = 'glm' in MODEL_PATH.lower()
USE_8BIT = os.environ.get('Q9A_8BIT') == '1'
N_FACTS = 8 if SMOKE else 60
N_PERM = 20 if SMOKE else 200
BATCH = 8 if (IS_GLM and USE_8BIT) else 16
OUT_DIR = ROOT / 'tests' / 'gpt5' / 'result'
OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------- 材料（预注册 §1.1）
# (zh_s, zh_p, en_s, en_p)：zh_p 必须能接在「很」后
FACTS = [
    ('苹果', '甜', 'apple', 'sweet'), ('柠檬', '酸', 'lemon', 'sour'),
    ('辣椒', '辣', 'chili pepper', 'spicy'), ('咖啡', '苦', 'coffee', 'bitter'),
    ('蜂蜜', '稠', 'honey', 'thick'), ('棉花', '软', 'cotton', 'soft'),
    ('丝绸', '光滑', 'silk', 'smooth'), ('石头', '粗糙', 'stone', 'rough'),
    ('玻璃', '脆', 'glass', 'fragile'), ('铁块', '硬', 'iron', 'hard'),
    ('铅块', '沉', 'lead', 'heavy'), ('羽毛', '轻', 'feather', 'light'),
    ('冰面', '滑', 'ice', 'slippery'), ('竹子', '挺直', 'bamboo', 'straight'),
    ('橡皮', '柔韧', 'rubber', 'resilient'), ('木炭', '易燃', 'charcoal', 'flammable'),
    ('火焰', '亮', 'flame', 'bright'), ('煤炭', '黑', 'coal', 'black'),
    ('雪山', '高', 'snowy peak', 'tall'), ('沙漠', '干燥', 'desert', 'dry'),
    ('沼泽', '泥泞', 'swamp', 'muddy'), ('大海', '深', 'ocean', 'deep'),
    ('溪水', '清', 'stream', 'clear'), ('冬天', '冷', 'winter', 'cold'),
    ('夏天', '热', 'summer', 'hot'), ('薄荷', '凉', 'mint', 'cool'),
    ('深夜', '静', 'late night', 'quiet'), ('雷声', '响', 'thunder', 'loud'),
    ('花香', '浓', 'flower scent', 'strong'), ('钻石', '稀有', 'diamond', 'rare'),
    ('黄金', '贵重', 'gold', 'precious'), ('钢缆', '结实', 'steel cable', 'sturdy'),
    ('纸张', '薄', 'paper', 'thin'), ('城墙', '厚', 'city wall', 'thick'),
    ('古树', '苍老', 'ancient tree', 'old'), ('泉水', '冰凉', 'spring water', 'icy'),
    ('海风', '咸', 'sea breeze', 'salty'), ('醋', '刺鼻', 'vinegar', 'pungent'),
    ('烈酒', '醉人', 'liquor', 'intoxicating'), ('月光', '朦胧', 'moonlight', 'hazy'),
    ('星空', '辽阔', 'starry sky', 'vast'), ('森林', '茂密', 'forest', 'dense'),
    ('草地', '鲜绿', 'grass', 'verdant'), ('稻田', '金黄', 'rice paddy', 'golden'),
    ('熔岩', '滚烫', 'lava', 'scorching'), ('冰川', '缓慢', 'glacier', 'slow'),
    ('瀑布', '湍急', 'waterfall', 'rapid'), ('湖面', '平静', 'lake', 'calm'),
    ('旋风', '猛烈', 'whirlwind', 'violent'), ('微风', '温和', 'breeze', 'gentle'),
    ('陈酒', '醇厚', 'aged wine', 'mellow'), ('新茶', '清香', 'fresh tea', 'fragrant'),
    ('臭豆腐', '臭', 'stinky tofu', 'stinky'), ('奶酪', '浓郁', 'cheese', 'rich'),
    ('面包', '松软', 'bread', 'fluffy'), ('巧克力', '甜腻', 'chocolate', 'cloying'),
    ('皮革', '耐用', 'leather', 'durable'), ('尼龙', '廉价', 'nylon', 'cheap'),
    ('陶瓷', '易碎', 'ceramic', 'brittle'), ('不锈钢', '防锈', 'stainless steel', 'rustproof'),
]
FACTS = FACTS[:N_FACTS]
REG = ['colloq', 'acad', 'code']


def render(fact, lang, reg):
    zh_s, zh_p, en_s, en_p = fact
    if lang == 'zh':
        s, p = zh_s, zh_p
        if reg == 'colloq':
            return f'你知道吧，{s}很{p}。'
        if reg == 'acad':
            return f'已有研究表明，{s}呈现{p}的特性。'
        return f'# {s}: {p}=True  # 已验证'
    s, p = en_s, en_p
    if reg == 'colloq':
        return f'You know, {s} is really {p}.'
    if reg == 'acad':
        return f'Prior research indicates that {s} is highly {p}.'
    return f'# {s}: {p}=True  # verified'


def build_corpus():
    """返回 texts, meta(list of dict), split(np bool: True=discovery)"""
    texts, meta = [], []
    for i, f in enumerate(FACTS):
        for lang in ('zh', 'en'):
            for reg in REG:
                texts.append(render(f, lang, reg))
                meta.append({'fact': i, 'lang': lang, 'reg': reg,
                             'split': 'disc' if i % 2 == 0 else 'conf'})
    return texts, meta


# --------------------------------------------------------------- 模型与采集

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if IS_GLM and USE_8BIT:
        from transformers import BitsAndBytesConfig
        kw.update(device_map='cuda:0',
                  quantization_config=BitsAndBytesConfig(
                      load_in_8bit=True,
                      llm_int8_skip_modules=['lm_head', 'embed_tokens']))
    else:
        kw.update(device_map='cuda:0')
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)
    m.eval()
    return tok, m


def capture_field(tok, m, texts):
    """采集位置 A（末 token）与 B（A-1）的 resid_in[L+1]/attn_out[L]/mlp_out[L]。"""
    import torch
    base = m.model if hasattr(m, 'model') else m.transformer
    layers = base.layers
    L = len(layers)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    N = len(texts)
    d = m.config.hidden_size
    # NOTE（本机 transformers 实测）：hidden_states 元组长度 L+1，但**最后一项是
    # final_norm 之后的输出**（cos(hss[L], norm(resid_out[L-1]))=0.9999967 实测），
    # 不是 layer L-1 的残差输出。因此 resid_in 只取 hss[0..L-1]（L 项），
    # 末层残差输出用 R[:,L-1]+A[:,L-1]+M[:,L-1] 算术重构（恒等式已在 l≤L-2 验证）。
    R = np.zeros((N, L, d), np.float16)
    A = np.zeros((N, L, d), np.float16)
    M = np.zeros((N, L, d), np.float16)
    RB = np.zeros((N, L, d), np.float16)
    AB = np.zeros((N, L, d), np.float16)
    MB = np.zeros((N, L, d), np.float16)
    posA = np.zeros(N, np.int64)

    caps = {'attn': [None] * L, 'mlp': [None] * L}
    hooks = []

    def hook_attn(i):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            caps['attn'][i] = t.detach()
        return fn

    def hook_mlp(i):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            caps['mlp'][i] = t.detach()
        return fn

    for i, ly in enumerate(layers):
        hooks.append(ly.self_attn.register_forward_hook(hook_attn(i)))
        hooks.append(ly.mlp.register_forward_hook(hook_mlp(i)))

    ids_all = [tok.encode(t) for t in texts]
    order = np.argsort([len(x) for x in ids_all])
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            batch_ids = [ids_all[i] for i in idx]
            maxlen = max(len(x) for x in batch_ids)
            inp = torch.full((len(idx), maxlen), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), maxlen), dtype=torch.long)
            for j, b in enumerate(batch_ids):
                inp[j, :len(b)] = torch.tensor(b)
                mask[j, :len(b)] = 1
            inp, mask = inp.cuda(), mask.cuda()
            out = m(input_ids=inp, attention_mask=mask, output_hidden_states=True)
            hs = torch.stack(out.hidden_states[:L], 1)      # (B, L, T, d)：只取 resid_in[0..L-1]
            pa = mask.sum(1) - 1                            # 位置 A
            pb = (pa - 1).clamp(min=0)
            bidx = torch.arange(len(idx), device=hs.device)
            ra = hs[bidx, :, pa].half().cpu().numpy()       # (B, L, d)
            rb = hs[bidx, :, pb].half().cpu().numpy()
            aa = torch.stack([caps['attn'][l][bidx, pa] for l in range(L)], 1).half().cpu().numpy()
            ma = torch.stack([caps['mlp'][l][bidx, pa] for l in range(L)], 1).half().cpu().numpy()
            ab = torch.stack([caps['attn'][l][bidx, pb] for l in range(L)], 1).half().cpu().numpy()
            mb = torch.stack([caps['mlp'][l][bidx, pb] for l in range(L)], 1).half().cpu().numpy()
            for j, gi in enumerate(idx):
                R[gi] = ra[j]; A[gi] = aa[j]; M[gi] = ma[j]
                RB[gi] = rb[j]; AB[gi] = ab[j]; MB[gi] = mb[j]
                posA[gi] = pa[j].item()
            del out, hs, ra, rb, aa, ma, ab, mb
    for h in hooks:
        h.remove()
    say(f'[field] N={N} L={L} d={d} 耗时 {time.time()-t0:.1f}s')
    return R, A, M, RB, AB, MB, posA


def probe_acc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(ytr)) < 2:
        return float('nan')
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def perm_null(Xtr, ytr, Xte, yte, n_perm):
    accs = []
    for _ in range(n_perm):
        yp = rng.permutation(ytr)
        accs.append(probe_acc(Xtr, yp, Xte, yte))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    texts, meta = build_corpus()
    y_reg = np.array([REG.index(m['reg']) for m in meta])
    y_lang = np.array([0 if m['lang'] == 'zh' else 1 for m in meta])
    is_disc = np.array([m['split'] == 'disc' for m in meta])

    tok, m = load_model()
    R, A, M, RB, AB, MB, posA = capture_field(tok, m, texts)
    L = A.shape[1]
    tr, te = is_disc, ~is_disc

    # 位置 B 对照（预注册 §1.2）：resid_in[l+1] = resid_in[l] + attn + mlp 同样断言，
    # 然后只报告 B 位的 3-way 关键层探针（标点聚合混淆检查）
    errB = (RB[:, 1:].astype(np.float32) -
            (RB[:, :-1] + AB[:, :-1] + MB[:, :-1]).astype(np.float32))
    denB = np.linalg.norm(RB[:, 1:].astype(np.float32), axis=-1) + 1e-6
    relB = np.linalg.norm(errB, axis=-1) / denB
    IDENT_MAX = 1e-2
    say(f'[identityB] 位置B 相对残差: max={relB.max():.4g}')
    if not relB.max() < IDENT_MAX:
        raise AssertionError('identity check failed at position B')
    RBx = np.zeros((RB.shape[0], L + 1, RB.shape[2]), np.float16)
    RBx[:, :L] = RB
    RBx[:, L] = (RB[:, L - 1].astype(np.float32) + AB[:, L - 1] + MB[:, L - 1]).astype(np.float16)
    accB = {}
    for l in (0, 1, L // 2, L):
        accB[l] = probe_acc(RBx[tr, l].astype(np.float32), y_reg[tr],
                            RBx[te, l].astype(np.float32), y_reg[te])
        say(f'[P1B] 位置B l={l}: 3-way confirm acc={accB[l]:.4f}')

    # 恒等式断言（l=0..L-2）：resid_in[l+1] = resid_in[l] + attn_out[l] + mlp_out[l]
    err = (R[:, 1:].astype(np.float32) -
           (R[:, :-1] + A[:, :-1] + M[:, :-1]).astype(np.float32))
    denom = np.linalg.norm(R[:, 1:].astype(np.float32), axis=-1) + 1e-6
    rel = np.linalg.norm(err, axis=-1) / denom
    say(f'[identity] 位置A 相对残差(l=0..L-2): max={rel.max():.4g} p99={np.quantile(rel, 0.99):.4g}')
    # fp16 存储舍入地板：每个加项相对误差 ~5e-4，三项累加 + 小分母层可达 ~5e-3；
    # 实测 float32 内存中 bf16 计算的恒等式相对残差 ~2e-3（tests/gpt5_temp/q9a_diag.py）。
    IDENT_MAX = 2e-2 if IS_GLM else 1e-2
    ident_ok = bool(rel.max() < IDENT_MAX)
    if not ident_ok and IS_GLM:
        say(f'[identity] GLM 8-bit 量化噪声超 2e-3，记录实值不中止（预注册 §5）')
    elif not ident_ok:
        say('[identity] FAIL：残差流恒等式破裂，中止')
        raise AssertionError('identity check failed')

    # Rext: [:, :L] = resid_in[0..L-1]；[:, L] = 末层残差输出（pre-final-norm，算术重构）
    Rext = np.zeros((R.shape[0], L + 1, R.shape[2]), np.float16)
    Rext[:, :L] = R
    Rext[:, L] = (R[:, L - 1].astype(np.float32) + A[:, L - 1] + M[:, L - 1]).astype(np.float16)

    res = {'model': MODEL_PATH, 'is_glm': IS_GLM, 'L': L, 'd': int(A.shape[2]),
           'n_texts': len(texts), 'identity_rel_max': float(rel.max()),
           'identity_ok': ident_ok, 'identity_rel_max_B': float(relB.max()),
           'acc_posB': {str(k): v for k, v in accB.items()},
           'smoke': SMOKE}

    def acc_profile(Xall):
        """Xall: (N, L+1, d) → 逐层 pooled 3-way confirm acc"""
        accs = []
        for l in range(Xall.shape[1]):
            accs.append(probe_acc(Xall[tr, l].astype(np.float32), y_reg[tr],
                                  Xall[te, l].astype(np.float32), y_reg[te]))
        return accs

    # P1/P2: resid_in 层剖面（含末层 pre-norm 输出）
    t0 = time.time()
    acc_resid = acc_profile(Rext)
    say('[P2] resid_in 逐层 pooled 3-way confirm acc:')
    for l, a in enumerate(acc_resid):
        say(f'  l={l:2d} {a:.4f}')
    res['acc_resid'] = acc_resid

    # 零分布（预注册 §2 P6：l∈{0, L//2, L-1}）
    nulls = {}
    for l in (0, L // 2, L - 1):
        mu, sd = perm_null(Rext[tr, l].astype(np.float32), y_reg[tr],
                           Rext[te, l].astype(np.float32), y_reg[te], N_PERM)
        nulls[l] = {'mean': mu, 'sd': sd}
        say(f'[P6] 零分布 l={l}: mean={mu:.4f} sd={sd:.4f}')
    res['perm_null'] = nulls

    # V1 判决
    acc0 = acc_resid[0]
    inc = max(a - acc0 for a in acc_resid)
    l_best = int(np.argmax(acc_resid))
    null_sd = float(np.mean([v['sd'] for v in nulls.values()]))
    if acc0 >= 0.90 and inc <= 0.03:
        v1 = 'STYLE_IS_TOKEN_IDENTITY'
    elif inc >= 0.05 and inc >= 3 * null_sd:
        v1 = 'STYLE_HAS_COMPUTED_INCREMENT'
    else:
        v1 = 'AMBIGUOUS'
    res['v1'] = {'acc0': acc0, 'max_increment': inc, 'l_best': l_best,
                 'null_sd': null_sd, 'verdict': v1,
                 'increment_ratio': inc / max(acc_resid[l_best], 1e-9)}
    say(f'[V1] {v1}  acc0={acc0:.4f} max_inc={inc:.4f}@l={l_best} null_sd={null_sd:.4f}')

    # P3: 组件归因（风格轴投影，confirm 样本）
    comp = {}
    for lang in ('zh', 'en'):
        ml = y_lang == (0 if lang == 'zh' else 1)
        disc_l, conf_l = tr & ml, te & ml
        col, acad, code = REG.index('colloq'), REG.index('acad'), REG.index('code')
        proj_a, proj_m, dr = [], [], []
        for l in range(L):
            X = Rext[:, l + 1].astype(np.float32)  # resid_out[l]（resid_in[l+1]，末层为重构）
            grand = X[disc_l].mean(0)
            mc = X[disc_l & (y_reg == col)].mean(0)
            ma_ = X[disc_l & (y_reg == acad)].mean(0)
            v = (mc - ma_)
            v /= (np.linalg.norm(v) + 1e-9)
            pa_ = A[conf_l, l].astype(np.float32) @ v
            pm_ = M[conf_l, l].astype(np.float32) @ v
            proj_a.append(pa_); proj_m.append(pm_)
            dr.append((X[conf_l] @ v) - (R[conf_l, l].astype(np.float32) @ v))
        proj_a = np.array(proj_a); proj_m = np.array(proj_m); dr = np.array(dr)
        # 逐层带中位 |proj| 与配对置换
        from scipy import stats as st
        bands = [(0, L // 4), (L // 4, L // 2), (L // 2, 3 * L // 4), (3 * L // 4, L)]
        band_stat = []
        for b0, b1 in bands:
            ia = np.median(np.abs(proj_a[b0:b1]))
            im = np.median(np.abs(proj_m[b0:b1]))
            # 配对置换：逐样本 |a| vs |m| 差的符号置换
            diff = (np.abs(proj_a[b0:b1]) - np.abs(proj_m[b0:b1])).mean(1)
            obs = diff.mean()
            cnt = 0
            for _ in range(N_PERM):
                sgn = rng.choice([-1, 1], size=diff.shape)
                if abs((sgn * diff).mean()) >= abs(obs):
                    cnt += 1
            p = (cnt + 1) / (N_PERM + 1)
            band_stat.append({'band': [b0, b1], 'med_attn': float(ia),
                              'med_mlp': float(im), 'ratio': float(ia / (im + 1e-12)),
                              'p_paired': float(p)})
        # code 轴
        proj_a_code, proj_m_code = [], []
        for l in range(L):
            X = Rext[:, l + 1].astype(np.float32)
            mcode = X[disc_l & (y_reg == code)].mean(0)
            mothr = X[disc_l & (y_reg != code)].mean(0)
            v = (mcode - mothr)
            v /= (np.linalg.norm(v) + 1e-9)
            proj_a_code.append(np.median(np.abs(A[conf_l, l].astype(np.float32) @ v)))
            proj_m_code.append(np.median(np.abs(M[conf_l, l].astype(np.float32) @ v)))
        comp[lang] = {'bands': band_stat,
                      'proj_attn_colloq_vs_acad': proj_a, 'proj_mlp_colloq_vs_acad': proj_m,
                      'delta_resid_colloq_vs_acad': dr,
                      'med_attn_code': proj_a_code, 'med_mlp_code': proj_m_code}
        say(f'[P3:{lang}] 层带 |proj_attn|:|proj_mlp| 中位比: ' +
            ' '.join(f"{b['band'][0]}-{b['band'][1]}:{b['ratio']:.2f}(p={b['p_paired']:.3f})"
                     for b in band_stat))
    res['p3_component'] = {k: {'bands': v['bands'],
                               'med_attn_code_last': float(v['med_attn_code'][-1]),
                               'med_mlp_code_last': float(v['med_mlp_code'][-1])}
                           for k, v in comp.items()}

    # V2 判决（合并语言，按中段层带）
    ratios = []
    for lang in ('zh', 'en'):
        for b in comp[lang]['bands']:
            ratios.append((b['ratio'], b['p_paired']))
    med_ratio = float(np.median([r for r, _ in ratios]))
    sig = [r for r, p in ratios if p < 0.01]
    if len(sig) >= 3 and med_ratio > 2:
        v2 = 'ATT_CARRIES' if med_ratio > 0 else 'MLP_CARRIES'
        v2 = 'ATT_CARRIES' if med_ratio > 2 else ('MLP_CARRIES' if med_ratio < 0.5 else 'MIXED')
    elif len(sig) >= 3 and med_ratio < 0.5:
        v2 = 'MLP_CARRIES'
    else:
        v2 = 'MIXED'
    res['v2'] = {'median_ratio': med_ratio, 'n_sig_bands': len(sig), 'verdict': v2}
    say(f'[V2] {v2}  med_ratio={med_ratio:.2f} n_sig={len(sig)}')

    # P4: 跨语言迁移（colloq vs acad 二分）
    def binary(y_full, mask_lang, lab_true, lab_false):
        col, acad = REG.index('colloq'), REG.index('acad')
        yt = np.where(y_full == col, 1, 0)
        sel = np.isin(y_full, [col, acad]) & mask_lang
        return sel, yt

    col, acad = REG.index('colloq'), REG.index('acad')
    l_best_layer = int(np.argmax(acc_resid))
    zh_m, en_m = y_lang == 0, y_lang == 1
    yb = np.where(y_reg == col, 1, 0)
    sel_zh = np.isin(y_reg, [col, acad]) & zh_m
    sel_en = np.isin(y_reg, [col, acad]) & en_m
    l = l_best_layer
    in_zh = probe_acc(Rext[tr & sel_zh, l].astype(np.float32), yb[tr & sel_zh],
                      Rext[te & sel_zh, l].astype(np.float32), yb[te & sel_zh])
    zh2en = probe_acc(Rext[tr & sel_zh, l].astype(np.float32), yb[tr & sel_zh],
                      Rext[te & sel_en, l].astype(np.float32), yb[te & sel_en])
    in_en = probe_acc(Rext[tr & sel_en, l].astype(np.float32), yb[tr & sel_en],
                      Rext[te & sel_en, l].astype(np.float32), yb[te & sel_en])
    en2zh = probe_acc(Rext[tr & sel_en, l].astype(np.float32), yb[tr & sel_en],
                      Rext[te & sel_zh, l].astype(np.float32), yb[te & sel_zh])
    gap = float(min(in_zh - zh2en, in_en - en2zh))
    v3 = 'STYLE_ABSTRACT' if gap <= 0.05 else ('STYLE_LANGUAGE_BOUND' if gap >= 0.15 else 'PARTIAL')
    res['v3'] = {'in_zh': in_zh, 'zh2en': zh2en, 'in_en': in_en, 'en2zh': en2zh,
                 'gap': gap, 'verdict': v3, 'layer': l}
    say(f'[V3] {v3}  zh内={in_zh:.3f} zh→en={zh2en:.3f} en内={in_en:.3f} en→zh={en2zh:.3f}')

    # P5: 四基线（最后一层 resid 与 l=0）
    from sklearn.feature_selection import f_classif
    base = {}
    for l in (0, L - 1):
        X = Rext[:, l].astype(np.float32)
        # 词袋（字符 2-gram）
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(analyzer='char', ngram_range=(2, 2), min_df=1)
        Bw = cv.fit_transform(texts).toarray().astype(np.float32)
        acc_bag = probe_acc(Bw[tr], y_reg[tr], Bw[te], y_reg[te])
        # 长度
        ln = np.array([len(t) for t in texts], np.float32).reshape(-1, 1)
        acc_len = probe_acc(ln[tr], y_reg[tr], ln[te], y_reg[te])
        # 主导单坐标
        F, _ = f_classif(X[tr], y_reg[tr])
        j = int(np.nanargmax(F))
        acc_coord = probe_acc(X[tr, j:j + 1], y_reg[tr], X[te, j:j + 1], y_reg[te])
        # L1 范数
        n1 = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
        acc_l1 = probe_acc(n1[tr], y_reg[tr], n1[te], y_reg[te])
        base[l] = {'bag2': acc_bag, 'length': acc_len,
                   'top_coord': acc_coord, 'top_coord_j': j, 'l1': acc_l1}
        say(f'[P5] l={l}: 词袋={acc_bag:.4f} 长度={acc_len:.4f} '
            f'主导单坐标={acc_coord:.4f}(j={j}) L1={acc_l1:.4f}')
    res['p5_baselines'] = base

    # 落盘
    tag = 'glm' if IS_GLM else 'qwen'
    npz_path = OUT_DIR / f'pat_q9a_{tag}{"_smoke" if SMOKE else ""}.npz'
    np.savez_compressed(npz_path, R=R, A=A, M=M, RB=RB, AB=AB, MB=MB,
                        posA=posA, y_reg=y_reg, y_lang=y_lang, is_disc=is_disc)
    texts_path = OUT_DIR / f'pat_q9a_{tag}{"_smoke" if SMOKE else ""}_texts.json'
    texts_path.write_text(_json.dumps(texts, ensure_ascii=False, indent=1),
                          encoding='utf-8', newline='')
    doc = provenance.write_result(f'pat_q9a_{tag}', res, model_path=MODEL_PATH,
                                  seed=SEED, tool='tests/gpt5/pat_q9a.py')
    # 副本到 tests/gpt5/result/（owner 指令：测试结果保存在此）
    import shutil
    shutil.copy(doc, OUT_DIR / Path(doc).name)
    say(f'[save] {npz_path}')
    say(f'[save] {doc}  sha8={_json.load(open(doc, encoding="utf-8"))["sha8"]}')
    say(f'[save] {texts_path}')
    say(f'[total] 耗时 {time.time()-t0:.1f}s（分析段）')
    return res


if __name__ == '__main__':
    t_all = time.time()
    main()
    say(f'[ALL DONE] 总耗时 {time.time()-t_all:.1f}s')
