# -*- coding: utf-8 -*-
"""
N2b: N2 的稳健性 / 特异性对照
==============================
N2 观察到：cloze_is 下对最终 is-a 读数影响最大的既非 L24-33，
而是 L6 attention（Δscore −6.01）与 L0 MLP（Δscore −5.11）。
本脚本做四组对照，判断该现象是真机制还是伪影：

  D 三对照：zero / mean / matched-norm noise(5 seeds) —— 排除"消融把分布推离流形"的解释
  E 位置特异性：同一部件在 pos=-1(读点位) / pos=0 / 全部位置 消融 —— 读点位特异性
  F 任务特异性：neutral("这是{W}") 与 cloze_is("{W}是一种") 对比 —— 是通用路由还是任务化重建
  G 窗口联合消融：L24..L35 全部 attn / 全部 mlp 一起消融 vs 逐层之和 —— 窗口真实权重
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
SEED = 20261001
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2b_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers
L = len(layers); norm = _core.norm; head = model.lm_head
def ids_of(s): return tok.encode(s, add_special_tokens=False)

def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm): return getattr(a, nm)
    raise RuntimeError('no attn out proj')
ATTN = [attn_out_proj(layers[i]) for i in range(L)]
MLP = [layers[i].mlp for i in range(L)]
n_heads = model.config.num_attention_heads
o_in = ATTN[0].in_features
HD = o_in // n_heads

w('=== N2b 稳健性 / 特异性对照 ===')
w('time %s model=%s L=%d heads=%d hd=%d tie=%s' %
  (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, n_heads, HD, model.config.tie_word_embeddings))
w('tokenize "苹果是一种" -> %r' % (tok.convert_ids_to_tokens(ids_of('苹果是一种')),))
w('tokenize "这是苹果"   -> %r' % (tok.convert_ids_to_tokens(ids_of('这是苹果')),))
sys.stdout.flush()

GROUPS = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
SUP_ID = {s: ids_of(s)[0] for s in GROUPS if len(ids_of(s)) == 1}
SUPS = list(SUP_ID.keys())
CASES = [(wd, sup) for sup, ms in GROUPS.items() if sup in SUP_ID
         for wd in ms if len(ids_of(wd)) == 1][:24]
w('supers=%s  cases n=%d' % (' '.join(SUPS), len(CASES)))
sys.stdout.flush()

def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

@torch.no_grad()
def fwd(text, specs=()):
    ii = torch.tensor([ids_of(text)], device='cuda')
    handles = []
    for spec in specs:
        kind, l, mode, payload = spec
        if kind == 'attn':
            def mk(mode, payload):
                def hook(mod, args):
                    y = args[0].clone()
                    if mode == 'zero':
                        if payload == 'last': y[:, -1, :] = 0
                        elif payload == 'first': y[:, 0, :] = 0
                        else: y[:] = 0
                    elif mode == 'mean':
                        y[:, -1, :] = payload.to(y.dtype)
                    elif mode == 'noise':
                        y[:, -1, :] = y[:, -1, :] + payload.to(y.dtype)
                    return (y,) + tuple(args[1:])
                return hook
            handles.append(ATTN[l].register_forward_pre_hook(mk(mode, payload)))
        else:
            def mk2(mode, payload):
                def hook(mod, args, out):
                    y = out.clone()
                    if mode == 'zero':
                        if payload == 'last': y[:, -1, :] = 0
                        elif payload == 'first': y[:, 0, :] = 0
                        else: y[:] = 0
                    elif mode == 'mean': y[:, -1, :] = payload.to(y.dtype)
                    elif mode == 'noise': y[:, -1, :] = y[:, -1, :] + payload.to(y.dtype)
                    return y
                return hook
            handles.append(MLP[l].register_forward_hook(mk2(mode, payload)))
    out = model(input_ids=ii)
    for h in handles: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def capture(text, kind, l):
    ii = torch.tensor([ids_of(text)], device='cuda')
    box = {}
    if kind == 'attn':
        # 前置钩子：取 o_proj 的输入（= concat(head_out)，维度 = n_heads*head_dim = 4096）
        h = ATTN[l].register_forward_pre_hook(
            lambda m, a: box.__setitem__('v', a[0].detach().clone()))
    else:
        # 后置钩子：取 MLP 块输出（= hidden 2560）
        h = MLP[l].register_forward_hook(lambda m, a, o: box.__setitem__('v', o.detach().clone()))
    model(input_ids=ii)
    h.remove()
    v = box['v'][0].float()       # [T, D]
    return v

rng = np.random.default_rng(SEED)

# ---------- D 三对照 ----------
PICK = [('attn', 6), ('mlp', 0), ('attn', 34), ('attn', 35), ('attn', 29), ('mlp', 9), ('mlp', 24), ('mlp', 29)]
w('')
w('--- D 三对照：zero / mean / noise(5 seeds) ---')
w('part      | dScore_zero | dScore_mean | dScore_noise(mean+-sd) | dRank_zero')
D = {}
for kind, l in PICK:
    z = []; m = []; nn = []; rz = []
    for wd, sup in CASES:
        p = '%s是一种' % wd; sid = ids_of(wd)[0]
        v0 = fwd(p); s0, r0 = score_rank(v0, sup, sid)
        vz = fwd(p, [(kind, l, 'zero', 'last')]); sz, rzz = score_rank(vz, sup, sid)
        z.append(sz - s0); rz.append(rzz - r0)
        cap = capture(p, kind, l)
        mean_vec = cap[:-1].mean(0)
        vm = fwd(p, [(kind, l, 'mean', mean_vec)]); sm, _ = score_rank(vm, sup, sid)
        m.append(sm - s0)
        comp_norm = float(cap[-1].norm())
        dim = cap.shape[-1]
        reps = []
        for k in range(5):
            nz = torch.tensor(rng.standard_normal(dim) * (comp_norm / np.sqrt(dim)),
                              device='cuda', dtype=torch.bfloat16)
            vn = fwd(p, [(kind, l, 'noise', nz)]); sn, _ = score_rank(vn, sup, sid)
            reps.append(sn - s0)
        nn.append(reps)
    D[(kind, l)] = (np.mean(z), np.mean(m), np.array(nn), np.mean(rz))
    w('%-4s L%-3d | %+10.3f | %+11.3f | %+8.3f +- %-6.3f       | %+9.1f' %
      (kind, l, np.mean(z), np.mean(m), np.array(nn).mean(), np.array(nn).std(), np.mean(rz)))
sys.stdout.flush()

# ---------- E 位置特异性 ----------
w('')
w('--- E 位置特异性（同一部件，消融位置不同）---')
w('part     | pos=last(-1) | pos=first(0) | pos=ALL ')
for kind, l in [('attn', 6), ('mlp', 0), ('mlp', 9)]:
    a = []; b = []; c = []
    for wd, sup in CASES:
        p = '%s是一种' % wd; sid = ids_of(wd)[0]
        v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
        s1, _ = score_rank(fwd(p, [(kind, l, 'zero', 'last')]), sup, sid); a.append(s1 - s0)
        s2, _ = score_rank(fwd(p, [(kind, l, 'zero', 'first')]), sup, sid); b.append(s2 - s0)
        s3, _ = score_rank(fwd(p, [(kind, l, 'zero', 'all')]), sup, sid); c.append(s3 - s0)
    w('%-4s L%-3d | %+12.3f | %+12.3f | %+8.3f' % (kind, l, np.mean(a), np.mean(b), np.mean(c)))
sys.stdout.flush()

# ---------- F 任务特异性 ----------
w('')
w('--- F 任务特异性：neutral("这是{W}") vs cloze("{W}是一种") ---')
w('part     | dScore_neutral | dScore_cloze')
for kind, l in [('attn', 6), ('mlp', 0), ('mlp', 9), ('mlp', 24), ('attn', 34)]:
    a = []; b = []
    for wd, sup in CASES:
        sid = ids_of(wd)[0]
        pn = '这是%s' % wd
        v0 = fwd(pn); s0, _ = score_rank(v0, sup, sid)
        s1, _ = score_rank(fwd(pn, [(kind, l, 'zero', 'last')]), sup, sid); a.append(s1 - s0)
        pc = '%s是一种' % wd
        v0c = fwd(pc); s0c, _ = score_rank(v0c, sup, sid)
        s1c, _ = score_rank(fwd(pc, [(kind, l, 'zero', 'last')]), sup, sid); b.append(s1c - s0c)
    w('%-4s L%-3d | %+14.3f | %+12.3f' % (kind, l, np.mean(a), np.mean(b)))
sys.stdout.flush()

# ---------- G 窗口联合消融 ----------
w('')
w('--- G 窗口联合消融 L24..L35（同时全零）vs 逐层之和 ---')
per_attn = {}; per_mlp = {}
for kind in ['attn', 'mlp']:
    for l in range(24, 36):
        acc = []
        for wd, sup in CASES:
            p = '%s是一种' % wd; sid = ids_of(wd)[0]
            v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
            s, _ = score_rank(fwd(p, [(kind, l, 'zero', 'last')]), sup, sid)
            acc.append(s - s0)
        (per_attn if kind == 'attn' else per_mlp)[l] = np.mean(acc)
for kind, tbl in [('attn', per_attn), ('mlp', per_mlp)]:
    joint = []
    for wd, sup in CASES:
        p = '%s是一种' % wd; sid = ids_of(wd)[0]
        v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
        specs = [(kind, l, 'zero', 'last') for l in range(24, 36)]
        s, _ = score_rank(fwd(p, specs), sup, sid)
        joint.append(s - s0)
    ssum = sum(tbl.values())
    w('%-4s L24-35: joint=%+.3f | sum_single=%+.3f | |joint|/|sum| = %.2f | sum|singles|=%+.3f' %
      (kind, np.mean(joint), ssum, abs(np.mean(joint)) / max(abs(ssum), 1e-9),
       sum(abs(v) for v in tbl.values())))
    w('     per-layer: %s' % ' '.join('L%d(%+.2f)' % (l, v) for l, v in tbl.items()))
sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
