# -*- coding: utf-8 -*-
"""Q8-D 频率方向的前向干预（Qwen3-4B / GLM-4-9B 通用）

预注册：research/gpt5/docs/PAT_Q8D_PREREG_v1.md（§1–§6 冻结，跑后修订入 §8）

问题：Q8-B 证到「频率是嵌入表里一个 4–16 维稠密方向、对数刻度、与范数/语义正交」，
      但那是搬表里的行、没跑前向。这一轮问：这个方向在前向里**用不用**。
操作：把上下文某位置的 token 表示沿 w 平移 δ，看下一个 token 分布是否往同方向偏。

三个必须分开的东西（tied 混淆，预注册 §1.4）：
  IN   —— 只改输入表示（读出头用原矩阵）    ← 主判
  OUT  —— 只改读出头里受影响 token 的行     ← 自证解码器
  BOTH —— 就地改表（tied 模型的默认后果）

效率：IN 与 BOTH 共用同一次前向；OUT 完全复用基线的隐藏态（输入没变），只需加标量行偏移。

用法：
    SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q8d.py
    .venv/Scripts/python.exe -X utf8 tools/pat_q8d.py
    MODEL=D:/develop/model/ZhipuAI/glm-4-9b-chat-hf .venv/Scripts/python.exe -X utf8 tools/pat_q8d.py
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import provenance                       # noqa: E402
import pat_q8b as B                     # noqa: E402

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


SMOKE = os.environ.get('SMOKE') == '1'
SEED = int(os.environ.get('SEED', '20261101'))
MODEL_PATH = os.environ.get('MODEL', 'D:/develop/model/Qwen/Qwen3-4B')
B.MODEL_PATH = MODEL_PATH
IS_GLM = 'glm' in MODEL_PATH.lower()

PHASE = os.environ.get('Q8D_PHASE', '1')
_DEF_D = '-8,-4,-2,-1,0,1,2,4,8' if PHASE == '1' else '-2,-1,-0.5,0,0.5,1,2'
DELTAS = [float(x) for x in os.environ.get('Q8D_DELTAS', _DEF_D).split(',')]
# D1 用 0.5/0.75（宽网格下的粗闸）；D2 预注册改为 0.30/0.60（见 PAT_Q8D2_PREREG_v1 §1.2）
_DH = '0.5' if PHASE == '1' else '0.30'
_DA = '0.75' if PHASE == '1' else '0.60'
IS_P3 = PHASE == '3'
MAIN_POS = 'POS0' if PHASE == '3' else 'LAST'
GATE_H = float(os.environ.get('Q8D_GATE_H', _DH))
GATE_AGREE = float(os.environ.get('Q8D_GATE_AGREE', _DA))
N_EMB = 0 if PHASE == '1' else (4 if IS_GLM else 8)
N_RAND = 2 if SMOKE else (4 if IS_GLM else 8)
BATCH = 16 if IS_GLM else 32
N_PROMPT = 200
MAX_LEN = 40
ALLPOS_RAND_DOSES = (4.0, 8.0) if PHASE == '1' else (1.0, 2.0)


# ---------------------------------------------------------------- 方向与材料

def build_z(tok, V):
    """词表逐行的 zipf；无定义记 NaN（§2：主指标只在 M 上重归一化）。

    V 必须取**嵌入矩阵的行数**，不是 len(tok)：Qwen3-4B 的 lm_head/embed 有 151 936 行，
    而 tokenizer 只有 151 669 个 token（尾部是 padding 槽）。用 len(tok) 会让 z 与
    logits 维度对不上（第一次 SMOKE 就撞在这里）。
    """
    from wordfreq import zipf_frequency
    z = np.full(V, np.nan, np.float64)
    for i in range(V):
        s = tok.decode([i])
        if (not s) or ('\ufffd' in s):
            continue
        zi = zipf_frequency(s, 'zh')
        if zi > 0:
            z[i] = zi
    return z


NOUNS = ('苹果 香蕉 橘子 西瓜 葡萄 草莓 桃子 梨 菠萝 芒果 桌子 椅子 窗户 台灯 '
         '沙发 汽车 火车 飞机 轮船 学校 医院 公园 银行 书店 面包 米饭 面条 鸡蛋 '
         '牛奶 咖啡 啤酒 糖果 铅笔 书包 手机 电脑 钥匙 雨伞 衣服 鞋子 帽子 花园 '
         '河流 山峰 树叶 花朵 石头 木头').split()

# 候选模板池：全部以 {N} 结尾，且**句法上不完整**（把字句/介词/量词/及物动词）。
#
# 为什么要"不完整"：诊断发现句尾一旦语法完整，模型就只预测标点
# （"她很喜欢吃苹果" -> '，' 占 0.849，p_M 0.033），主指标 E[zipf] 直接失效。
# 而"他把苹果" p_M=0.971。这不是挑好看的，是主指标能不能算的前提。
TEMPLATES_CAND = [
    '他把{N}', '她把{N}', '我昨天把{N}', '妈妈把{N}', '工人把{N}', '弟弟把{N}',
    '他把书放在{N}', '我从{N}', '他走到{N}', '她站在{N}', '信寄到{N}', '车停在{N}',
    '一棵{N}', '一片{N}', '一朵{N}', '一条{N}', '一把{N}', '一张{N}',
    '一块{N}', '一根{N}', '一座{N}', '一件{N}', '一瓶{N}', '一袋{N}',
    '这个比{N}', '这本书讲的是关于{N}', '他对{N}', '我们在讨论{N}',
    '她正在吃', '他在喝', '我准备买', '孩子想要', '妈妈正在切',
    '爷爷喜欢种', '她昨天丢了一个', '他从包里拿出',
]

# 模板筛选闸（材料口径，预注册 §1.7 允许在 SMOKE 阶段调；只看基线，不看干预）
MIN_PM = 0.80          # 主指标必须在够多的质量上算
MIN_H = 1.00           # 基线熵太低（单峰 0.999）时分布没有位移空间
N_TMPL_PROBE = 12      # 每个模板用多少条句子评估


def build_prompts(tok, z, tmpls):
    """确定性生成 + 过滤「末位不是有频率定义的中日韩字符」的候选（§1.5）。

    过滤是必要的口径而不是挑选：末位若不是内容词，平移它的嵌入会把「频率效应」
    和「语法破坏」混在一起。保留顺序固定（模板序 × 词表序），取前 N_PROMPT 条。
    """
    cand = []
    for t in tmpls:
        for w in NOUNS:
            cand.append(t.replace('{N}', w))
    keep, drop = [], 0
    for s_ in cand:
        ids = tok(s_, add_special_tokens=False)['input_ids']
        # 末位内容词检查永远用裸串（包装版的末位是模板尾 token）
        if not ids or not np.isfinite(z[ids[-1]]) or (not tok.decode([ids[-1]])) \
                or ('\ufffd' in tok.decode([ids[-1]])):
            drop += 1
            continue
        if IS_P3:
            wid = tok(wrap_str(tok, s_), add_special_tokens=False)['input_ids']
            if not wid or len(wid) > MAX_LEN * 4:
                drop += 1
                continue
        elif len(ids) > MAX_LEN:
            drop += 1
            continue
        sl = tok.decode([ids[-1]])
        if (not sl) or ('\ufffd' in sl) or not np.isfinite(z[ids[-1]]):
            drop += 1
            continue
        keep.append(s_)
        if len(keep) >= N_PROMPT:
            break
    return keep, len(cand), drop


C8_PROBES = [
    ('中国的首都', '是', '北京'), ('日本的首都', '是', '东京'), ('法国的首都', '是', '巴黎'),
    ('英国的首都', '是', '伦敦'), ('德国的首都', '是', '柏林'), ('意大利的首都', '是', '罗马'),
    ('西班牙的首都', '是', '马德里'), ('俄国的首都', '是', '莫斯科'), ('韩国的首都', '是', '首尔'),
    ('一加一', '等于', '二'), ('二加二', '等于', '四'), ('三加一', '等于', '四'),
    ('五加五', '等于', '十'), ('十减三', '等于', '七'),
    ('最大的海洋', '是', '太平洋'), ('最高的山峰', '是', '珠穆朗玛峰'),
    ('最长的河流', '是', '尼罗河'), ('太阳系最大的行星', '是', '木星'),
    ('红色和黄色混合', '是', '橙色'), ('蓝色和黄色混合', '是', '绿色'),
    ('大的反义词', '是', '小'), ('多的反义词', '是', '少'), ('高的反义词', '是', '低'),
    ('长的反义词', '是', '短'), ('快的反义词', '是', '慢'), ('热的反义词', '是', '冷'),
    ('黑的相反颜色', '是', '白'), ('上的反义词', '是', '下'), ('开的反义词', '是', '关'),
    ('早的反义词', '是', '晚'), ('新的反义词', '是', '旧'), ('硬的相反性质', '是', '软'),
    ('一天', '有', '二十四'), ('一年', '有', '十二'), ('一只手的手指', '有', '五'),
    ('一个星期的天数', '有', '七'), ('三角形的内角和', '是', '一百八十'),
    ('正方形', '有', '四'), ('冰融化的温度', '是', '零'), ('水沸腾的温度', '是', '一百'),
]


def wrap_str(tok, s):
    """PHASE=3 时把 prompt 包进 chat 模板（add_generation_prompt=True），
    使 pos0 成为格式 token（Qwen: <|im_start|>；GLM: [gMASK]）。其余相位原样返回。

    诊断（q8d3_diag）：Qwen3 默认模板在 assistant\n 后**强制 <think>**（top1=1.000，
    p_M=0，主指标退化）；enable_thinking=False 后末尾为 </think>\n\n，top1='你' 0.776，
    分布健康。GLM-4 无 thinking 模式，不支持该 kwarg（模板不含 <think>）。"""
    if not IS_P3:
        return s
    kw = {}
    if 'qwen' in MODEL_PATH.lower():
        kw['enable_thinking'] = False
    txt = tok.apply_chat_template([{'role': 'user', 'content': s}],
                                  tokenize=False, add_generation_prompt=True, **kw)
    if 'glm' in MODEL_PATH.lower():
        # 诊断（q8d3_gdiag）：HF 版 GLM-4 模板止于 <|assistant|>，模型在该处 1.000 预测 '\n'
        # （官方格式是 <|assistant|>\n{content}）⇒ p_M=0、主指标退化。补尾换行后
        # top1='这句话' 0.468，分布健康。这是模板补全（官方格式的缺省部分），非材料挑选。
        txt = txt + '\n'
    return txt


def encode(tok, prompts, pad_id):
    import torch
    prompts = [wrap_str(tok, s) for s in prompts]
    L = max(len(tok(s, add_special_tokens=False)['input_ids']) for s in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    msk = torch.zeros((len(prompts), L), dtype=torch.long)
    for i, s in enumerate(prompts):
        v = tok(s, add_special_tokens=False)['input_ids']
        ids[i, :len(v)] = torch.tensor(v, dtype=torch.long)
        msk[i, :len(v)] = 1
    return ids, msk


# ---------------------------------------------------------------- 模型

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if IS_GLM and os.environ.get('Q8D_8BIT') == '1':
        # 全精度的 18.8 GB 权重在 17.1 GB 卡上必须 offload，而 offload 路径在本机
        # 反复在加载阶段段错误（与外部内存压力相关）。8-bit 后 ~9.5 GB 可整卡放下，
        # 没有 CPU 侧，崩不了。**这是材料变更，必须写进结论**：
        # lm_head 与 embed_tokens 跳过量化（llm_int8_skip_modules），
        # 所以嵌入平移与读出都是 bf16 原值，量化只发生在中间的 attention/MLP。
        from transformers import BitsAndBytesConfig
        kw.update(device_map='cuda:0',
                  quantization_config=BitsAndBytesConfig(
                      load_in_8bit=True,
                      llm_int8_skip_modules=['lm_head', 'embed_tokens']))
    elif IS_GLM:
        kw.update(device_map='auto', max_memory={0: '15GiB', 'cpu': '20GiB'})
    else:
        kw.update(device_map='cuda:0')
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)
    m.eval()
    return tok, m


def base_mod(m):
    for a in ('model', 'transformer'):
        if hasattr(m, a):
            return getattr(m, a)
    raise RuntimeError('找不到 base transformer')


def real_head_weight(m):
    """取**真实的** lm_head 权重。

    踩过的坑：GLM-4-9B 在 max_memory 下 offload，`get_output_embeddings().weight` 停在
    meta device 上（`dev=meta`），值全是未初始化垃圾：手算 logits 与原生差 1.8e4、
    ‖o2‖=115761 vs ‖ref‖=764。calibrate_head 因此断言失败并中止 —— 这正是它该做的事。
    解法：从 safetensors 直读。
    """
    import torch
    W = m.get_output_embeddings().weight
    if W.device.type != 'meta':
        return W.detach()
    from safetensors import safe_open
    p = Path(MODEL_PATH)
    idx = p / 'model.safetensors.index.json'
    if idx.exists():
        mp = _json.loads(idx.read_text(encoding='utf-8'))['weight_map']
        key = next(k for k in mp if k.endswith('lm_head.weight'))
        fp = p / mp[key]
    else:
        fp = p / 'model.safetensors'
        with safe_open(str(fp), framework='pt') as f:
            key = next(k for k in f.keys() if k.endswith('lm_head.weight'))
    with safe_open(str(fp), framework='pt') as f:
        arr = f.get_tensor(key).to(torch.bfloat16)
    say('lm_head 原为 meta，已从 %s 直读（%s %s）' % (fp.name, key, tuple(arr.shape)))
    return arr.cuda()


def untie(m):
    """把 lm_head 复制成独立 Parameter。

    探针验过：Qwen3-4B tie=True、data_ptr 相同；解绑后 logits max|Δ|=0.000e+00。
    不做这一步就没有「纯输入」臂 —— 改输入必然连带改读出，无法归因。
    """
    import torch
    head = m.get_output_embeddings()
    W = real_head_weight(m).clone()
    new = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype, device=W.device)
    new.weight.data.copy_(W)
    m.lm_head = new
    return m


def calibrate_head(m, ids, msk):
    """**断言，不假设**：先确认「logits = W·h」里 h 到底是哪个张量。

    探针 3 发现 GLM 手算 head 与原生差 16.5 —— 说明不能想当然认为 h = hs[-1]。
    这里枚举候选，取 max|Δ| 最小者，并要求它确实小；否则直接报错，
    免得整轮结论建立在一个错的 h 上（这是本轮最贵的一类错误）。
    """
    import torch
    W = real_head_weight(m)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=msk, output_hidden_states=True)
        ref = out.logits[:, -1, :].float()
        cands = {}
        hs = out.hidden_states
        for name, idx in (('hs_last', -1), ('hs_last2', -2), ('hs_last3', -3)):
            cands[name] = hs[idx][:, -1, :]
        cands['base0'] = base_mod(m)(input_ids=ids, attention_mask=msk,
                                     use_cache=False)[0][:, -1, :]
        best, bestd = None, float('inf')
        for k, h in cands.items():
            d = float((ref - h.float() @ W.float().T).abs().max())
            say('    候选 %-9s max|Δ|=%.4f' % (k, d))
            if d < bestd:
                best, bestd = k, d
    say('  head 标定：选用 %s（max|Δ|=%.4f）%s'
        % (best, bestd, 'PASS' if bestd < 1.0 else 'FAIL —— 手算 head 不可信，中止'))
    if bestd >= 1.0:
        raise RuntimeError('calibrate_head 失败：没有候选能复现 logits')
    return best


def make_hidden_fn(m, which):
    """返回 h_last 取法：'base0' 走 base 模块（省显存），否则走 output_hidden_states。"""
    import torch
    if which == 'base0':
        def fn(ids=None, msk=None, embeds=None):
            if embeds is None:
                o = base_mod(m)(input_ids=ids, attention_mask=msk, use_cache=False)
            else:
                o = base_mod(m)(inputs_embeds=embeds, attention_mask=msk, use_cache=False)
            return o[0]
        return fn
    idx = {'hs_last': -1, 'hs_last2': -2, 'hs_last3': -3}[which]

    def fn(ids=None, msk=None, embeds=None):
        kw = dict(attention_mask=msk, use_cache=False, output_hidden_states=True)
        if embeds is None:
            o = m(input_ids=ids, **kw)
        else:
            o = m(inputs_embeds=embeds, **kw)
        return o.hidden_states[idx]
    return fn


# ---------------------------------------------------------------- 核心

def forward_last(hidden_fn, emb, ids, msk, u_t, delta, pos, batch, dtype):
    """返回末位隐藏态 [N, d]（float32）。delta=0 时走原生 input_ids 路径。"""
    import torch
    N, L = ids.shape
    last = msk.sum(1) - 1
    outs = []
    with torch.no_grad():
        for i0 in range(0, N, batch):
            sl = slice(i0, min(i0 + batch, N))
            b_ids, b_msk, b_last = ids[sl], msk[sl], last[sl]
            nb = b_ids.shape[0]
            if delta == 0.0:
                h = hidden_fn(ids=b_ids, msk=b_msk)
            else:
                E0 = emb(b_ids).float()
                if pos == 'LAST':
                    w = torch.zeros_like(b_msk, dtype=torch.float32)
                    w[torch.arange(nb, device=ids.device), b_last] = 1.0
                elif pos == 'POS0':
                    # 只平移第 0 位（格式 token）。right padding 下 0 位必是实 token。
                    w = torch.zeros_like(b_msk, dtype=torch.float32)
                    w[:, 0] = b_msk[:, 0].float()
                else:
                    w = b_msk.float()
                E0 = E0 + (delta * u_t).view(1, 1, -1) * w.unsqueeze(-1)
                h = hidden_fn(msk=b_msk, embeds=E0.to(dtype))
            outs.append(h[torch.arange(nb, device=ids.device), b_last].float())
    return torch.cat(outs, 0)


def stats_of(L, zt, zt_ok, ref_logp, ref_top1):
    """从 [N, V] logits 算 §2 的全部指标。"""
    import torch
    p = torch.softmax(L, dim=-1)
    pM = p[:, zt_ok].sum(1)
    Ez = (p[:, zt_ok] * zt[zt_ok]).sum(1) / pM
    H = -(p * torch.log(p.clamp_min(1e-30))).sum(1)
    top1p = p.max(1).values
    logp = torch.log(p.clamp_min(1e-30))
    kl = (torch.exp(ref_logp) * (ref_logp - logp)).sum(1)
    agree = (p.argmax(1) == ref_top1).float()
    # top10 只在有 zipf 定义的行里取（对无定义频率的行取 zipf 无意义）
    pm = torch.where(zt_ok, p, torch.full_like(p, -1.0))
    t10 = pm.topk(10, dim=-1).indices
    zt10 = zt[t10].mean(1)
    return dict(Ez=Ez, pM=pM, H=H, top1p=top1p, kl=kl, agree=agree,
                top10=zt10, maxlogit=L.abs().max(1).values)


def apply_out(L, h, ids, msk, u_t, delta, pos):
    """OUT 臂：给受影响 token 的行加 δ·(h·u)。就地改 L。"""
    import torch
    last = msk.sum(1) - 1
    dot = (h * u_t).sum(1)                     # [N]
    N = ids.shape[0]
    for r in range(N):
        if pos == 'LAST':
            tgt = [int(ids[r, last[r]])]
        else:
            tgt = torch.unique(ids[r][msk[r] > 0]).tolist()
        L[r, tgt] += delta * dot[r]
    return L


def boot_ci(x, n_boot=1000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (float('nan'), float('nan'))
    rs = np.random.default_rng(7)
    idx = rs.integers(0, len(x), (n_boot, len(x)))
    mm = x[idx].mean(1)
    return (float(np.quantile(mm, alpha / 2)), float(np.quantile(mm, 1 - alpha / 2)))


def mde_paired(d, alpha=0.05, power=0.8):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return float('nan')
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * d.std(ddof=1) / np.sqrt(len(d)))


def select_templates(m, tok, hidden_fn, emb, dtype, Whead, zt, zt_ok, pad_id):
    """模板筛选闸：只按**基线** p_M 与熵筛，不看任何干预结果。

    诊断发现两件事：
      1. 句法完整 -> 模型只预测标点（p_M 0.03），E[zipf] 无意义；
      2. 单峰 0.999 -> 分布没有位移空间，干预也看不出东西。
    所以闸门是 p_M ≥ MIN_PM 且 H ≥ MIN_H。这是材料口径，不是结果挑选。
    """
    import torch
    probes, owner = [], []
    for t in TEMPLATES_CAND:
        for wn in NOUNS[:N_TMPL_PROBE]:
            s_ = t.replace('{N}', wn)
            ids_ = tok(s_, add_special_tokens=False)['input_ids']
            wid_ = tok(wrap_str(tok, s_), add_special_tokens=False)['input_ids'] if IS_P3 else ids_
            if not ids_ or len(wid_) > MAX_LEN * 4:
                continue
            probes.append(s_)
            owner.append(t)
    ids, msk = encode(tok, probes, pad_id)
    ids, msk = ids.cuda(), msk.cuda()
    zero = torch.zeros(1, device='cuda')
    h = forward_last(hidden_fn, emb, ids, msk, zero, 0.0, 'LAST', BATCH, dtype)
    with torch.no_grad():
        L = h @ Whead.float().T
        pr = torch.softmax(L, dim=-1)
        pM = (pr[:, zt_ok].sum(1)).cpu().numpy()
        H = (-(pr * torch.log(pr.clamp_min(1e-30))).sum(1)).cpu().numpy()
    owner = np.array(owner)
    stats = []
    for t in TEMPLATES_CAND:
        sel = owner == t
        if sel.sum() == 0:
            continue
        stats.append(dict(tmpl=t, n=int(sel.sum()),
                          pM=float(np.median(pM[sel])), H=float(np.median(H[sel]))))
    keep = [d['tmpl'] for d in stats if d['pM'] >= MIN_PM and d['H'] >= MIN_H]
    say('模板筛选（闸：p_M≥%.2f 且 H≥%.2f）：%d/%d 通过'
        % (MIN_PM, MIN_H, len(keep), len(stats)))
    for d in stats:
        say('  %-22s pM=%.3f H=%.2f %s' % (d['tmpl'], d['pM'], d['H'],
                                           '✓' if d['tmpl'] in keep else '✗'))
    if len(keep) < 6:
        say('⚠ 通过数 <6，退化为按 p_M 排序取前 8（记入 §8）')
        keep = [d['tmpl'] for d in sorted(stats, key=lambda x: -x['pM'])[:8]]
    return keep, stats


def main():
    t0 = time.time()
    import torch
    say('=' * 78)
    say('Q8-D%s 频率方向前向干预  model=%s  SMOKE=%s  seed=%d'
        % ({'1': '（宽网格）', '2': '（窄网格注册）', '3': '3（pos0/格式位置）'}[PHASE],
           MODEL_PATH, SMOKE, SEED))
    say('=' * 78)

    # —— 静态方向 u（§1.1）
    tokE, E, tied = B.load_E()
    pools = B.build_pools(tokE, E)
    ic, zc = pools['char']
    iw, zw = pools['word']
    idx = np.concatenate([ic, iw])
    y = np.concatenate([zc, zw])
    mu = E.mean(0)
    X = (E[idx] - mu).astype(np.float32)
    yl = y - y.mean()
    G, bv = B.gram_xy(X, yl)
    w = B.solve_w(G, bv)
    u = (w / float(w @ w)).astype(np.float32)
    nu = float(np.linalg.norm(u))
    med_norm = float(np.median(np.linalg.norm(E[idx], axis=1)))
    t_pred = X @ w
    Eraw = E
    say('方向 ‖w‖=%.4f  ‖u‖=%.6f  median‖e‖=%.4f  预测 t 的 std=%.3f'
        % (float(np.linalg.norm(w)), nu, med_norm, float(t_pred.std())))
    del G, bv, X
    gc.collect()

    # —— 模型
    tok, m = load_model()
    say('模型加载 %.1f s  tied=%s  显存 %.2f GB'
        % (time.time() - t0, tied, torch.cuda.memory_allocated() / 1e9))
    if tied:
        m = untie(m)
        say('已解绑 lm_head（否则没有「纯输入」臂）')
    Whead = real_head_weight(m)
    assert Whead.device.type == 'cuda', 'lm_head 仍在 %s' % Whead.device
    V, d = Whead.shape
    say('词表 %d  d=%d  lm_head dtype=%s' % (V, d, Whead.dtype))

    z_np = build_z(tok, V)
    zt_raw = torch.tensor(z_np, dtype=torch.float32, device='cuda')
    zt_ok = torch.isfinite(zt_raw)
    zt = torch.where(zt_ok, zt_raw, torch.zeros_like(zt_raw))
    say('有 zipf 定义 %d / %d (%.1f%%)' % (int(zt_ok.sum()), V, 100.0 * zt_ok.sum() / V))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = 'right'
    emb = m.get_input_embeddings()
    dtype = Whead.dtype

    # —— head 标定（断言，不假设；先用两条临时序列）
    tmp_ids, tmp_msk = encode(tok, ['他把苹果', '她把香蕉'], pad_id)
    say('head 标定：')
    which = calibrate_head(m, tmp_ids.cuda(), tmp_msk.cuda())
    hidden_fn = make_hidden_fn(m, which)

    # —— 模板筛选闸（只看基线 p_M / H，不看任何干预）
    tmpls, tmpl_stats = select_templates(m, tok, hidden_fn, emb, dtype, Whead,
                                         zt, zt_ok, pad_id)
    prompts, n_cand, n_drop = build_prompts(tok, z_np, tmpls)
    say('prompt 候选 %d，末位不合规丢弃 %d，保留 %d'
        % (n_cand, n_drop, len(prompts)))
    if SMOKE:
        prompts = prompts[:8]
    ids, msk = encode(tok, prompts, pad_id)
    ids, msk = ids.cuda(), msk.cuda()

    u_t = torch.tensor(u, device='cuda')

    def to_np(s):
        return {k: v.detach().cpu().numpy() for k, v in s.items()}

    # —— 基线
    h0 = forward_last(hidden_fn, emb, ids, msk, u_t, 0.0, 'LAST', BATCH, dtype)
    with torch.no_grad():
        L0 = h0 @ Whead.float().T
        ref_logp = torch.log(torch.softmax(L0, dim=-1).clamp_min(1e-30))
        ref_top1 = L0.argmax(1)
        s0 = to_np(stats_of(L0, zt, zt_ok, ref_logp, ref_top1))
    say('基线：E[zipf]=%.4f  p_M 中位 %.4f  H 中位 %.3f  top1_p 中位 %.3f  max|logit| %.1f'
        % (s0['Ez'].mean(), np.median(s0['pM']), np.median(s0['H']),
           np.median(s0['top1p']), s0['maxlogit'].max()))
    if np.median(s0['pM']) < 0.5:
        say('⚠ p_M 中位 %.3f < 0.5 ⇒ §3 陷阱 3：主指标 E[zipf] 不可用，须退到 top1_zipf'
            % np.median(s0['pM']))

    # batch 划分不变性（预注册 §7）：同一条件不同 batch 只能有 1e-3 级差异
    h0b = forward_last(hidden_fn, emb, ids, msk, u_t, 0.0, 'LAST', BATCH * 2, dtype)
    with torch.no_grad():
        L0b = h0b @ Whead.float().T
        d_batch = float((L0b - L0).abs().max())
        sb = to_np(stats_of(L0b, zt, zt_ok, ref_logp, ref_top1))
        d_ez = float(np.abs(sb['Ez'] - s0['Ez']).max())
        d_ez_mean = float(abs(sb['Ez'].mean() - s0['Ez'].mean()))
    say('batch 不变性 %d vs %d：max|Δlogit|=%.3e  max|ΔE[zipf]|=%.4f  Δ均值=%.4f'
        % (BATCH, BATCH * 2, d_batch, d_ez, d_ez_mean))
    say('  口径：bf16 下不同 batch 会选到不同 kernel，logit 有位级差异是正常的；'
        '关键是主指标 ΔE[zipf] 是否远超它。%s'
        % ('PASS' if d_ez < 0.02 else 'NOISY —— 效应必须显著大于这个噪声'))
    del h0b, L0b, sb
    Ez0, H0 = s0['Ez'], s0['H']
    L0_ref, ref_logp_g, ref_top1_g = L0, ref_logp, ref_top1

    # δ=0 三臂一致性（预注册 §7 检查项）
    with torch.no_grad():
        L_out0 = apply_out(L0_ref.clone(), h0, ids, msk, u_t, 0.0, 'LAST')
        s_oo = to_np(stats_of(L_out0, zt, zt_ok, ref_logp_g, ref_top1_g))
    say('δ=0 两臂一致性：E[zipf] max|Δ|=%.3e  %s'
        % (np.abs(s_oo['Ez'] - s0['Ez']).max(),
           'PASS' if np.abs(s_oo['Ez'] - s0['Ez']).max() < 1e-6 else 'FAIL'))

    # —— 方向集
    rs = np.random.default_rng(SEED + 13)
    dirs = [('FREQ', u)]
    for k in range(N_RAND):
        r = rs.standard_normal(u.shape).astype(np.float32)
        dirs.append(('RAND%d' % k, (r / np.linalg.norm(r) * nu).astype(np.float32)))
    # 同流形对照：随机两个真实 token 嵌入之差。RAND 是各向同性向量，本来就"离开流形"，
    # 所以 FREQ 的 KL 远大于 RAND 只能说明"它在流形上"，不能说明它有功能角色。
    # 这个对照把"在流形上"这件事单独扣掉。
    pool_rows = np.concatenate([ic, iw])
    for k in range(N_EMB):
        a, b2 = rs.choice(pool_rows, 2, replace=False)
        r = (Eraw[a] - Eraw[b2]).astype(np.float32)
        r = r - r.mean()
        dirs.append(('EMB%d' % k, (r / np.linalg.norm(r) * nu).astype(np.float32)))
    say('方向：FREQ + %d 个 RAND + %d 个 EMB（位移长度已对齐 ‖u‖）' % (N_RAND, N_EMB))

    deltas = DELTAS if not SMOKE else [-4.0, 0.0, 2.0]
    recs = []

    def record(dirn, delta, pos, arm, s, nfwd_mark):
        dEz = s['Ez'] - Ez0
        ok = s['pM'] >= 0.5
        recs.append(dict(
            dir=dirn, delta=float(delta), pos=pos, arm=arm,
            dEz=dEz, Ez=s['Ez'], kl=s['kl'], H=s['H'], pM=s['pM'],
            agree=s['agree'], top10=s['top10'] - s0['top10'],
            dEz_mean=float(dEz[ok].mean()) if ok.sum() > 3 else float('nan'),
            dEz_ci=list(boot_ci(dEz[ok])) if ok.sum() > 3 else [None, None],
            mde=mde_paired(dEz[ok]) if ok.sum() > 3 else float('nan'),
            dEz_med=float(np.median(dEz[ok])) if ok.sum() > 3 else float('nan'),
            kl_mean=float(s['kl'][ok].mean()) if ok.sum() > 3 else float('nan'),
            dH_med=float(np.median(s['H'] - H0)),
            agree_mean=float(s['agree'].mean()),
            pM_med=float(np.median(s['pM'])), n_ok=int(ok.sum()), n=len(s['pM']),
            maxlogit=float(s['maxlogit'].max()),
            nfwd=nfwd_mark))

    nfwd = 0
    t1 = time.time()
    with torch.no_grad():
        POS_LIST = ('POS0', 'LAST') if IS_P3 else ('LAST', 'ALLPOS')
        POS_RAND_DOSES = {'ALLPOS': ALLPOS_RAND_DOSES, 'POS0': (2.0,), 'LAST': ()}
        for pos in POS_LIST:
            for dirn, uv in dirs:
                ut = torch.tensor(uv, device='cuda')
                for dl in deltas:
                    if dirn != 'FREQ' and dl == 0.0:
                        continue
                    if pos != 'LAST' and dirn != 'FREQ' and dl not in POS_RAND_DOSES[pos]:
                        continue
                    if pos == 'ALLPOS' and dirn == 'FREQ' and dl == 0.0:
                        continue
                    if IS_P3 and pos == 'LAST' and (dirn != 'FREQ' or dl not in (-2.0, 2.0)):
                        continue
                    h = forward_last(hidden_fn, emb, ids, msk, ut, dl, pos, BATCH, dtype)
                    nfwd += 1
                    L = h @ Whead.float().T
                    s_in = stats_of(L, zt, zt_ok, ref_logp_g, ref_top1_g)
                    record(dirn, dl, pos, 'IN', to_np(s_in), nfwd)
                    if dirn == 'FREQ':
                        Lb = apply_out(L.clone(), h, ids, msk, ut, dl, pos)
                        record(dirn, dl, pos, 'BOTH',
                               to_np(stats_of(Lb, zt, zt_ok, ref_logp_g, ref_top1_g)), nfwd)
                        del Lb
                    if pos != 'POS0':
                        # OUT 臂复用基线 h0（输入未变）。POS0 无 OUT 定义（预注册 §2 无此臂）。
                        Lo = apply_out(L0_ref.clone(), h0, ids, msk, ut, dl, pos)
                        record(dirn, dl, pos, 'OUT',
                               to_np(stats_of(Lo, zt, zt_ok, ref_logp_g, ref_top1_g)), 0)
                        del Lo
                    del L, h
                say('  %s × %s 完成（%d 次前向，%.0f s）' % (pos, dirn, nfwd, time.time() - t1))

    # —— C8 语义副作用
    c8 = run_c8(m, tok, hidden_fn, emb, u_t, dtype, Whead, z_np, BATCH)

    payload = aggregate(recs, deltas, s0, nu, med_norm, t_pred, nfwd,
                        len(prompts), n_cand, n_drop, tied, which, V, d, c8,
                        float(t_pred.std()), tmpl_stats)
    payload['runtime_s'] = round(time.time() - t0, 1)
    _rname = {'1': 'pat_q8d', '2': 'pat_q8d2', '3': 'pat_q8d3'}[PHASE]
    path = provenance.write_result(_rname, payload, model_path=MODEL_PATH,
                                   seed=SEED, tool='tools/pat_q8d.py')
    say('RESULT', path)
    txt = path.with_suffix('.log.txt')
    txt.write_text('\n'.join(LOG) + '\n', encoding='utf-8', newline='')
    say('LOG', txt)


def softmax_helper(L):
    import torch
    return torch.softmax(L, dim=-1)


def run_c8(m, tok, hidden_fn, emb, u, dtype, Whead, z_np, batch):
    """§C8：唯一答案探测句上的语义保持。干预位 = 前缀末位（内容词），读出位 = 句末。"""
    import torch
    pre = []
    for a, bch, ans in C8_PROBES:
        ia = tok(a, add_special_tokens=False)['input_ids']
        ib = tok(bch, add_special_tokens=False)['input_ids']
        av = tok(ans, add_special_tokens=False)['input_ids']
        if not ia or not av:
            continue
        pre.append((ia + ib, len(ia) - 1, av[0]))
    if len(pre) < 10:
        return dict(n=0, note='probe 不足')
    Lm = max(len(x[0]) for x in pre)
    ids = torch.full((len(pre), Lm), 0, dtype=torch.long)
    msk = torch.zeros((len(pre), Lm), dtype=torch.long)
    spos = torch.zeros(len(pre), dtype=torch.long)
    for i, (v, s, _) in enumerate(pre):
        ids[i, :len(v)] = torch.tensor(v)
        msk[i, :len(v)] = 1
        spos[i] = s
    ans = torch.tensor([x[2] for x in pre])
    ids, msk, spos, ans = ids.cuda(), msk.cuda(), spos.cuda(), ans.cuda()

    def acc(delta):
        N = ids.shape[0]
        hit = 0
        with torch.no_grad():
            for i0 in range(0, N, batch):
                sl = slice(i0, min(i0 + batch, N))
                b_ids, b_msk, b_s, b_a = ids[sl], msk[sl], spos[sl], ans[sl]
                nb = b_ids.shape[0]
                if delta == 0.0:
                    h = hidden_fn(ids=b_ids, msk=b_msk)
                else:
                    E0 = emb(b_ids).float()
                    w = torch.zeros_like(b_msk, dtype=torch.float32)
                    w[torch.arange(nb, device=ids.device), b_s] = 1.0
                    E0 = E0 + (delta * u.view(1, 1, -1)) * w.unsqueeze(-1)
                    h = hidden_fn(msk=b_msk, embeds=E0.to(dtype))
                last = b_msk.sum(1) - 1
                hl = h[torch.arange(nb, device=ids.device), last].float()
                lg = hl @ Whead.float().T
                hit += int((lg.argmax(1) == b_a).sum())
        return hit / N

    a0 = acc(0.0)
    out = dict(n=len(pre), acc_base=float(a0), acc={})
    for dl in ((2.0, 8.0) if not SMOKE else (2.0,)):
        out['acc'][str(dl)] = float(acc(dl))
    drops = [a0 - v for v in out['acc'].values()]
    worst = max(drops) if drops else float('nan')
    out['worst_drop'] = float(worst)
    out['verdict'] = ('SEMANTICS_INTACT' if worst <= 0.10 else 'SEMANTICS_HURT')
    say('C8 语义副作用：基线命中 %.3f，最差下降 %+.3f => %s'
        % (a0, -worst, out['verdict']))
    return out


# ---------------------------------------------------------------- 汇总

def _get(recs, dirn, delta, pos, arm):
    for r in recs:
        if r['dir'] == dirn and abs(r['delta'] - delta) < 1e-9 and r['pos'] == pos and r['arm'] == arm:
            return r
    return None


def aggregate(recs, deltas, s0, nu, med_norm, t_pred, nfwd, n_prompt, n_cand,
              n_drop, tied, which, V, d, c8, t_std, tmpl_stats=None):
    out = dict(config=dict(model=MODEL_PATH, smoke=SMOKE, seed=SEED, tied=bool(tied),
                           head_calib=which, V=int(V), d=int(d),
                           nu=nu, median_e_norm=med_norm, t_std=t_std,
                           n_prompt=int(n_prompt), n_cand=int(n_cand), n_dropped=int(n_drop),
                           n_forward=int(nfwd), deltas=DELTAS, n_rand=N_RAND),
               template_gate=dict(min_pm=MIN_PM, min_h=MIN_H, table=tmpl_stats or []),
               baseline=dict(Ez=float(s0['Ez'].mean()), pM_med=float(np.median(s0['pM'])),
                             H_med=float(np.median(s0['H'])),
                             top1p_med=float(np.median(s0['top1p'])),
                             maxlogit=float(s0['maxlogit'].max())),
               dose_table=[], C={})

    # 剂量表（FREQ，两 pos，三臂）
    POS_TABLE = ('POS0', 'LAST') if IS_P3 else ('LAST', 'ALLPOS')
    for pos in POS_TABLE:
        for arm in ('IN', 'OUT', 'BOTH'):
            row = dict(pos=pos, arm=arm, delta=[], dEz=[], ci=[], mde=[],
                       dH=[], agree=[], kl=[], rel_disp=[])
            for dl in deltas:
                r = _get(recs, 'FREQ', dl, pos, arm)
                if r is None:
                    continue
                row['delta'].append(dl)
                row['dEz'].append(r['dEz_mean'])
                row['ci'].append(r['dEz_ci'])
                row['mde'].append(r['mde'])
                row['dH'].append(r['dH_med'])
                row['agree'].append(r['agree_mean'])
                row['kl'].append(r['kl_mean'])
                row['rel_disp'].append(abs(dl) * nu / med_norm)
            out['dose_table'].append(row)

    # ---------- C1
    r_in = [(_get(recs, 'FREQ', dl, MAIN_POS, 'IN')) for dl in deltas]
    xs = np.array([r['delta'] for r in r_in])
    ys = np.array([r['dEz_mean'] for r in r_in])
    rho = float(stats.spearmanr(xs, ys).statistic) if np.isfinite(ys).sum() > 3 else float('nan')
    r2 = _get(recs, 'FREQ', 2.0, MAIN_POS, 'IN')
    d2, ci2 = (r2['dEz_mean'], r2['dEz_ci']) if r2 else (float('nan'), [None, None])
    ci_excl0 = (ci2[0] is not None) and (ci2[0] * ci2[1] > 0)
    mde2 = r2['mde'] if r2 else float('nan')
    if abs(rho) >= 0.8 and d2 >= 0.10 and ci_excl0:
        v1 = 'CAUSAL_KNOB'
    elif abs(rho) >= 0.8 and d2 <= -0.10 and ci_excl0:
        v1 = 'CAUSAL_KNOB_INVERSE'
    elif np.isfinite(mde2) and mde2 <= 0.02 and abs(d2) < mde2:
        # 定量界：不是"没测到"，是"测到了，界在 MDE 以内"
        v1 = 'NULL_BOUNDED'
    elif np.isfinite(mde2) and mde2 > 0.10:
        v1 = 'PENDING_POWER'
    elif abs(rho) >= 0.8:
        v1 = 'WEAK_COUPLING'
    else:
        v1 = 'KNOB_NOT_USED'
    out['C']['C1'] = dict(dose=[float(x) for x in xs], dEz=[float(v) for v in ys],
                          spearman=rho, dEz_at_2=float(d2), ci=list(ci2),
                          mde=float(mde2), verdict=v1,
                          note='主判口径：FREQ × IN × %s' % MAIN_POS)

    # ---------- C2
    nz = [r for r in r_in if r['delta'] != 0.0]
    xn = np.array([r['delta'] for r in nz])
    yn = np.array([r['dEz_mean'] for r in nz])
    fin = np.isfinite(yn)
    if fin.sum() >= 3:
        a, b0 = np.polyfit(xn[fin], yn[fin], 1)
        pred = a * xn[fin] + b0
        ss = ((yn[fin] - pred) ** 2).sum()
        st = ((yn[fin] - yn[fin].mean()) ** 2).sum()
        lin_r2 = float(1 - ss / st) if st > 0 else float('nan')
        small = abs(yn[fin][np.argmin(np.abs(xn[fin]))]) / max(abs(xn[fin][np.argmin(np.abs(xn[fin]))]), 1e-9)
        big = abs(yn[fin][np.argmax(np.abs(xn[fin]))]) / max(abs(xn[fin][np.argmax(np.abs(xn[fin]))]), 1e-9)
        flip = (yn[fin].max() > 0) and (yn[fin].min() < 0)
        if flip and abs(yn[fin].max() - yn[fin].min()) > 0.05:
            v2 = 'NONMONOTONE'
        elif lin_r2 >= 0.90:
            v2 = 'LINEAR'
        elif big < 0.5 * small:
            v2 = 'SATURATING'
        else:
            v2 = 'NONLINEAR_OTHER'
    else:
        a = b0 = lin_r2 = float('nan')
        v2 = 'PENDING_POWER'
    out['C']['C2'] = dict(slope=float(a), intercept=float(b0), r2=float(lin_r2), verdict=v2)

    # ---------- C3 方向特异性（δ=+2，LAST，IN）
    rf = _get(recs, 'FREQ', 2.0, MAIN_POS, 'IN')
    rr = [_get(recs, 'RAND%d' % k, 2.0, MAIN_POS, 'IN') for k in range(N_RAND)]
    rr = [r for r in rr if r is not None]
    if rf and len(rr) >= 3:
        af = abs(rf['dEz_mean'])
        ar = np.array([abs(r['dEz_mean']) for r in rr])
        p95 = float(np.percentile(ar, 95))
        medr = float(np.median(ar))
        klf = rf['kl_mean']
        klr = np.array([r['kl_mean'] for r in rr])
        kl_ratio = klf / max(float(np.median(klr)), 1e-12)
        df = af / np.sqrt(max(klf, 1e-12))
        dr = ar / np.sqrt(np.maximum([r['kl_mean'] for r in rr], 1e-12))
        emb_c = []
        for k in range(N_EMB):
            re = _get(recs, 'EMB%d' % k, 2.0, 'LAST', 'IN')
            if re is not None:
                emb_c.append(dict(abs=float(abs(re['dEz_mean'])), kl=float(re['kl_mean'])))
        ae = np.array([d['abs'] for d in emb_c]) if emb_c else np.array([])
        p95e = float(np.percentile(ae, 95)) if len(ae) >= 3 else float('nan')
        mede = float(np.median(ae)) if len(ae) else float('nan')
        if af <= p95:
            v3 = 'GENERIC_PERTURBATION'
        elif len(ae) >= 3 and af <= p95e:
            # 只强过各向同性随机，不强过同流形随机 => D1 的"KL 20x"只是"在流形上"
            v3 = 'ON_MANIFOLD_ONLY'
        elif (af / max(medr, 1e-12)) >= 2.0 and (not np.isfinite(mede) or (af / max(mede, 1e-12)) >= 2.0):
            v3 = 'DIRECTION_SPECIFIC'
        else:
            v3 = 'DIRECTION_SPECIFIC' if (af / max(medr, 1e-12)) >= 2.0 else 'GENERIC_PERTURBATION'
        out['C']['C3'] = dict(abs_freq=float(af), rand=[float(x) for x in ar],
                              rand_p95=p95, rand_med=medr, ratio=float(af / max(medr, 1e-12)),
                              kl_freq=float(klf), kl_rand_med=float(np.median(klr)),
                              kl_ratio=float(kl_ratio),
                              dir_per_sqrtkl_freq=float(df),
                              dir_per_sqrtkl_rand=[float(x) for x in dr],
                              emb=emb_c, emb_p95=p95e, emb_med=mede,
                              verdict=v3)
    else:
        out['C']['C3'] = dict(verdict='PENDING_POWER')

    # ---------- C4：PHASE<=2 tied 归因（δ=+2，LAST）；PHASE=3 改为 D3-2 位置 KL 比
    ro = _get(recs, 'FREQ', 2.0, 'LAST', 'OUT')
    if IS_P3:
        rl = _get(recs, 'FREQ', 2.0, 'LAST', 'IN')
        if rf and rl:
            kl0 = float(rf['kl_mean'])
            kll = float(rl['kl_mean'])
            r_kl = kl0 / max(kll, 1e-12)
            if r_kl >= 3.0:
                v4 = 'REGISTER_SENSITIVE'
            elif r_kl >= 0.5:
                v4 = 'POSITION_NEUTRAL'
            else:
                v4 = 'LAST_DOMINANT'
            out['C']['C4'] = dict(kl_pos0=kl0, kl_last=kll, ratio=float(r_kl), verdict=v4)
        else:
            out['C']['C4'] = dict(verdict='PENDING_POWER')
    elif rf and ro:
        a_in, a_out = abs(rf['dEz_mean']), abs(ro['dEz_mean'])
        ratio = a_in / max(a_out, 1e-12)
        if ratio < 0.30:
            v4 = 'READOUT_ARTIFACT'
        elif ratio > 3.0:
            v4 = 'INPUT_DRIVEN'
        else:
            v4 = 'MIXED_CONTRIBUTION'
        out['C']['C4'] = dict(abs_in=float(a_in), abs_out=float(a_out),
                              ratio=float(ratio), verdict=v4)
    else:
        out['C']['C4'] = dict(verdict='PENDING_POWER')

    # ---------- C5 能力门
    gate = []
    for dl in deltas:
        r = _get(recs, 'FREQ', dl, MAIN_POS, 'IN')
        if r is None:
            continue
        ok = (r['dH_med'] <= GATE_H) and (r['agree_mean'] >= GATE_AGREE)
        gate.append(dict(delta=dl, dH=r['dH_med'], agree=r['agree_mean'],
                         maxlogit=r['maxlogit'], ok=bool(ok)))
    n_bad = sum(1 for g in gate if not g['ok'])
    out['C']['C5'] = dict(table=gate, n_bad=int(n_bad), n=int(len(gate)),
                          gate_h=GATE_H, gate_agree=GATE_AGREE,
                          verdict=('DEGRADED' if n_bad >= max(2, len(gate) // 2) else 'GATE_OK'))

    # ---------- C6 位置特异性（PHASE=3 无 ALLPOS 臂，占位；D3-2 已在 C4）
    ra = _get(recs, 'FREQ', 2.0, 'ALLPOS', 'IN')
    if IS_P3:
        out['C']['C6'] = dict(verdict='N/A_PHASE3')
    elif rf and ra:
        a_l = abs(rf['dEz_mean'])
        a_a = abs(ra['dEz_mean'])
        ratio = a_a / max(a_l, 1e-12)
        if ratio > 3.0:
            v6 = 'GLOBAL_OFFSET'
        elif ratio < 0.5:
            v6 = 'POSITION_CONCENTRATED'
        else:
            v6 = 'LOCAL_TOKEN_KNOB'
        out['C']['C6'] = dict(abs_last=float(a_l), abs_allpos=float(a_a),
                              ratio=float(ratio), verdict=v6)
    else:
        out['C']['C6'] = dict(verdict='PENDING_POWER')

    # ---------- C8
    out['C']['C8'] = c8

    # ---------- 总判决
    v1 = out['C']['C1']['verdict']
    if v1.startswith('CAUSAL_KNOB') and out['C']['C4']['verdict'] == 'READOUT_ARTIFACT':
        v1 = 'CAUSAL_KNOB_READOUT_ONLY'
    if out['C']['C5']['verdict'] == 'DEGRADED':
        grade, claim = 'E0', '干预把模型打坏，不给因果结论'
    elif (v1 == 'CAUSAL_KNOB' and out['C']['C3'].get('verdict') == 'DIRECTION_SPECIFIC'
          and out['C']['C4'].get('verdict') != 'READOUT_ARTIFACT'
          and c8.get('verdict') == 'SEMANTICS_INTACT'):
        grade, claim = 'E2', '频率方向在前向被使用，且方向特异'
    elif v1 == 'KNOB_NOT_USED':
        grade, claim = 'E1', '频率方向是表里的排序轴，不参与计算'
    else:
        grade, claim = 'E1', '有耦合但未过方向特异性/归因/语义门'
    out['verdicts'] = dict(C1=v1, C2=out['C']['C2']['verdict'],
                           C3=out['C']['C3'].get('verdict'),
                           C4=out['C']['C4'].get('verdict'),
                           C5=out['C']['C5']['verdict'],
                           C6=out['C']['C6'].get('verdict'),
                           C8=c8.get('verdict'), grade=grade, claim=claim)
    say('-' * 78)
    for k, v in out['verdicts'].items():
        say('  %s = %s' % (k, v))
    return out


if __name__ == '__main__':
    main()
