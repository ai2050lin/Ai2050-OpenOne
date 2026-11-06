# -*- coding: utf-8 -*-
"""AI2050 模板 runner v3 —— 可合并充分统计量 + 行为门 + 金丝雀（dist_protocol.v3）

随模板包下发到 Worker 节点，自包含（只依赖 numpy；真实模式另需 torch/transformers）:
    python runner.py [--mode auto|smoke|real|synthetic] [--out outputs]

环境变量:
    AI2050_RUNNER_MODEL   HF 模型目录（auto 模式下存在即走真实模式）
    AI2050_SHARD          分片 "i/n"（缺省 0/1；金丝雀条目在每个分片都执行）
    AI2050_PRECISION      bf16 | fp16 | fp32（缺省 bf16）
    AI2050_RUNNER_MODE    --mode auto 时强制模式（synthetic 用于沙盒中心的流程验证）
    AI2050_DEVICE_MAP     缺省整模型放在 cuda；auto = GPU/CPU 分层摆放（权重保持原精度，显存放不下的大模型用）
    AI2050_MAX_MEMORY     配合 auto，如 "0=12GiB,cpu=20GiB"；实际摆放记入 summary.placement
    AI2050_OFFLOAD_DIR    配合 auto 的磁盘卸载目录（缺省在输出目录旁）

采集口径（corpus 声明）:
    layers     "all"（hidden_states 全部：hs[0] = 嵌入层输出，hs[i] = 第 i 个块之后的残差流，
               hs[-1] 由 HF 在最终归一化之后给出——真实模式会用 lm_head(hs[-1]) 与实际 logits 对账并记录）
               或 first/mid/last/emb/整数层号列表（缺省 first/mid/last，旧语料不变）。
    readouts   读出位置（缺省 ["target"]）：target = 目标槽位最后一个 token；cue = 句式线索片段最后一个 token；
               final = 整句最后一个 token；probe_answer = probe 提示最后一个 token（预测答案首 token 的位置）。
               只有 target 一个读出时层位标签与旧版相同（first/mid/last/L{i}）；多个读出时标签为 "读出@层"。

产出（Agent 上传 outputs/ 下全部文件）:
    summary.json      必有：状态、计数、逐格行为准确率、token 长度、本分片 η² 摘要、采集口径、model_snapshot
    suffstats.npz     真实/合成模式：每 (split, 条件格, 层位标签, 维) 的 n、Σx、Σx²（float64，可跨节点相加）。
                      suffstats.v2 = v1 + 每个标签的 读出/层号；体积超预算时只保留 2 的幂步长嵌套网格上的层（各分片相同、跨族可取交集）
    behavior.json     含 probe 的条目：correct（行为门判定）与 mode——有对照集时 forced_choice（完整答案对数概率
                      max log P(可接受) > max log P(对照)，记录最佳答案、最佳对照与 margin），否则 set_first_token
                      （下一 token 是否为任一可接受答案的首 token）；另保留旧口径 first_top1 / rank / logprob（主答案首 token）
    canary.json       金丝雀摘要（目标位置向量的范数、均值、前 16 维；按层位）
    items.npz         逐条向量（float16；全语料体积超预算时用随机投影降维。投影维数 k 只由语料研究样本总数、标签数与
                      模型维度决定，投影矩阵由语料指纹决定——所有分片相同，与分片大小无关），
                      附条目元数据；items.v2 另含每个标签的 读出/层号 与逐层 logit-lens（下一 token 与答案首 token
                      的对数概率、top-k）。corpus.capture_items=false 时不输出
    原生全场（不上传）  真实/合成模式在本地按层写 float16 原生向量（全部读出位置）：
                      $AI2050_NATIVE_DIR（缺省 ~/.ai2050/native_fields）/<family>/<tm_id>-v<version>/<语料指纹>/<模型键>/
                      shard-<i>of<n>/L<hs下标>.npz（数组 X: (读出 P, 条目 N, 维 d)）+ manifest.json（文件、形状、dtype、
                      sha256、条目 id 顺序、tag_readout/tag_layer）。summary.native_field 记录目录与 manifest 的 sha256。
                      AI2050_NATIVE_DIR=off 关闭。
    部件（可选，不上传）  corpus.components = {"readouts": [...], "layers": "all" | [块号]} 时，在所选读出位置采集每个
                      Transformer 块的注意力头输出（o_proj 的输入按头拆分，z: H×head_dim）与 MLP 输出（d）：
                      <原生全场目录>/components/B<块号>.npz（Z: (P, N, H·head_dim)，M: (P, N, d)，float16）+ components.json；
                      o_proj 权重按模型键保存一次：<原生根目录>/_weights/<模型键>/o_proj_B<块号>.npy（(d, H·head_dim) float16）。
                      第 h 个头写入残差流的向量 = W_O[:, h·hd:(h+1)·hd] · z_h；hs[b+1] − hs[b] = Σ_h W_O,h z_h + MLP_b（预归一化
                      架构、o_proj 无偏置时精确成立）——真实模式逐条核对这一闭合并记入 summary.components.closure。
                      summary.components 只上传摘要（形状、闭合误差、逐头平均范数）。未声明 components 的语料行为与旧版相同。
    MLP 中间单元（可选，不上传） corpus.mlp_units = {"readouts": [...], "layers": "all" | [块号]} 时，在所选读出位置
                      （可含 probe_answer）采集每个块 MLP 中间空间的 gate 预激活 G 与 value（up 投影）V：
                      <原生全场目录>/mlp/B<块号>.npz（G、V: (P, N, d_ff)，float16）+ mlp.json（激活函数、d_ff、条目顺序）。
                      MLP 单元的输出 A = act(G) ⊙ V（SwiGLU；送入 down_proj 的量），真实模式用 down_proj 的实际输入核对
                      前若干条并记入 summary.mlp_units.check。G、V 分开保存，便于区分符号区（+×+ 与 −×− 都给出正输出）。

模式:
    smoke      不加载模型，只校验语料与分片，status=smoke（不冒充实测）
    real       真实前向，status=real
    synthetic  无 GPU 的协议与分析方法验证：按「已知答案」生成向量——早层以词身份与句式为主、
               中层出现类别方向、末层类别与句式并存，另有 token 长度干扰和一个行为薄弱格；在因果上不可见的
               读出位置不植入对应因子；status=synthetic，中心节点只合并 kind=real 结果，合成结果永不进入研究统计。
"""
import argparse
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path

HEAD_DIMS = 16
LENS_TOPK = 5
ITEM_CORPUS_BUDGET_BYTES = 192 * 1024 * 1024   # 上传 items.npz 的全语料（全部分片合计）float16 体积预算；corpus.item_budget_bytes 可覆盖
ITEM_SHARD_CAP_BYTES = 56 * 1024 * 1024       # 每个分片 items.npz 的上限（中心单文件上传上限 64 MB，留出 logit-lens 等余量）
SUFFSTATS_BUDGET_BYTES = 160 * 1024 * 1024
SNAPSHOT_PATTERNS = ('*.safetensors', '*.bin', '*.pt', '*.gguf', 'config.json', 'generation_config.json',
                     'tokenizer*', 'vocab*', 'merges.txt', 'special_tokens_map.json', '*.model', '*.tiktoken')


def cond_key(cond):
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


def parse_shard(value):
    try:
        i, n = (int(x) for x in str(value or '0/1').split('/'))
        if n >= 1 and 0 <= i < n:
            return i, n
    except ValueError:
        pass
    raise SystemExit(f'invalid AI2050_SHARD={value!r}; expected "i/n"')


def layer_spec_of(corpus):
    spec = corpus.get('layers', ['first', 'mid', 'last'])
    return [spec] if isinstance(spec, str) else list(spec)


def readouts_of(corpus):
    r = corpus.get('readouts') or ['target']
    return [str(x) for x in r]


def resolve_layers(spec, n_hs):
    """层位声明 → [(层标签, hidden_states 下标)]。"""
    out = []
    for tag in spec:
        if tag == 'all':
            out.extend((f'L{i}', i) for i in range(n_hs))
        elif tag == 'emb':
            out.append(('emb', 0))
        elif tag == 'first':
            out.append(('first', 1))
        elif tag == 'mid':
            out.append(('mid', n_hs // 2))
        elif tag == 'last':
            out.append(('last', n_hs - 1))
        elif isinstance(tag, int) and 0 <= tag < n_hs:
            out.append((f'L{tag}', tag))
        else:
            raise ValueError(f'unknown layer tag: {tag}')
    return out


def full_tags(readouts, layers):
    """(读出, 层) → 存储标签。单一 target 读出时与旧版标签一致。"""
    if readouts == ['target']:
        return [t for t, _ in layers]
    return [f'{r}@{t}' for r in readouts for t, _ in layers]


def item_projection(n_total, n_tags, dim, budget=None):
    """items.npz 的向量空间：('full', d) 或 ('jl<k>', k)。只依赖语料（研究样本总数、标签数）与模型维度。"""
    budget = int(budget or ITEM_CORPUS_BUDGET_BYTES)
    if n_tags * n_total * dim * 2 <= budget:
        return 'full', dim
    k = int(max(128, min(dim, budget // max(n_tags * n_total * 2, 1))))
    return (f'jl{k}', k) if k < dim else ('full', dim)


def items_budget(corpus):
    """全语料 items 预算 = min(corpus.item_budget_bytes 或缺省 192 MB, 每分片上限 × 分片数)。
    只依赖语料（分片数是语料的一部分），各分片相同；单分片的试跑语料因此会投影到上传上限以内。"""
    budget = int(corpus.get('item_budget_bytes') or ITEM_CORPUS_BUDGET_BYTES)
    return min(budget, ITEM_SHARD_CAP_BYTES * max(int(corpus.get('shards') or 1), 1))


def jl_matrix(seed, dim, k):
    """固定种子的高斯随机投影 (d, k)，由语料指纹决定。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dim, k)).astype(np.float32) / np.sqrt(k)


def native_root():
    """原生全场根目录；AI2050_NATIVE_DIR=off/0/none 时返回 None。"""
    base = os.environ.get('AI2050_NATIVE_DIR', '').strip()
    if base.lower() in ('off', '0', 'none', 'false'):
        return None
    return Path(base) if base else Path.home() / '.ai2050' / 'native_fields'


def native_dir(corpus, shard, model_key):
    """原生全场目录（在 outputs 之外，不上传）；AI2050_NATIVE_DIR=off/0/none 时返回 None。"""
    root = native_root()
    if root is None:
        return None
    fp = hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]

    def safe(x):
        return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(x))
    return (root / safe(corpus.get('family_id') or 'family')
            / f"{safe(corpus.get('tm_id') or 'untracked')}-v{corpus.get('version', 0)}"
            / fp / safe(model_key) / f'shard-{shard[0]}of{shard[1]}')


def component_plan(corpus, n_blocks):
    """部件采集口径：{'readouts': [...], 'blocks': [块号]}；语料未声明 components 时返回 None。"""
    comp = corpus.get('components')
    if not isinstance(comp, dict):
        return None
    readouts = [r for r in comp.get('readouts') or [] if r in readouts_of(corpus) and r != 'probe_answer']
    layers = comp.get('layers', 'all')
    blocks = (list(range(n_blocks)) if layers == 'all'
              else sorted({int(b) for b in layers if isinstance(b, int) and 0 <= b < n_blocks}))
    return {'readouts': readouts, 'blocks': blocks} if readouts and blocks else None


class ComponentStore:
    """逐条部件输出（float16）：每条 z (P, B, H·hd) 与 m (P, B, d)；另累计闭合误差。"""

    def __init__(self, cplan, n_heads, head_dim, dim):
        self.readouts, self.blocks = list(cplan['readouts']), list(cplan['blocks'])
        self.n_heads, self.head_dim, self.dim = int(n_heads), int(head_dim), int(dim)
        self.ids, self.z, self.m = [], [], []
        self.closure = []          # 逐 (条目, 读出, 块) 的相对闭合误差

    def add(self, item_id, z, m):
        import numpy as np
        self.ids.append(item_id)
        self.z.append(np.asarray(z, dtype=np.float16))
        self.m.append(np.asarray(m, dtype=np.float16))

    def digest(self):
        import numpy as np
        Z = np.stack(self.z).astype(np.float32)                     # (N, P, B, H·hd)
        N, P, B, _ = Z.shape
        zn = np.linalg.norm(Z.reshape(N, P, B, self.n_heads, self.head_dim), axis=-1).mean(0)
        mn = np.linalg.norm(np.stack(self.m).astype(np.float32), axis=-1).mean(0)
        out = {'readouts': self.readouts, 'blocks': self.blocks, 'n_heads': self.n_heads, 'head_dim': self.head_dim,
               'dim': self.dim, 'n_items': N,
               'z_norm_mean': np.round(zn, 4).tolist(), 'mlp_norm_mean': np.round(mn, 4).tolist()}
        if self.closure:
            err = np.array(self.closure, dtype=np.float64)
            out['closure'] = {'n': int(len(err)), 'rel_err_mean': float(err.mean()), 'rel_err_max': float(err.max()),
                              'note': '‖(hs[b+1]−hs[b]) − (o_proj(z_b) + MLP_b)‖ / ‖hs[b+1]−hs[b]‖；'
                                      '末块的 hs[-1] 已过最终归一化时不核对'}
        return out

    def write(self, out_dir, weights_info):
        import numpy as np
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Z = np.stack(self.z)                                         # (N, P, B, H·hd)
        M = np.stack(self.m)                                         # (N, P, B, d)
        files, total = [], 0
        for k, b in enumerate(self.blocks):
            buf = io.BytesIO()
            np.savez(buf, Z=np.ascontiguousarray(Z[:, :, k].transpose(1, 0, 2)),
                     M=np.ascontiguousarray(M[:, :, k].transpose(1, 0, 2)))
            data = buf.getvalue()
            name = f'B{b}.npz'
            (out_dir / name).write_bytes(data)
            total += len(data)
            files.append({'path': name, 'block': b, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        manifest = {'schema': 'components.v1', 'axes': ['readout', 'item', 'dim'], 'readouts': self.readouts,
                    'ids': list(self.ids), 'blocks': self.blocks, 'n_heads': self.n_heads, 'head_dim': self.head_dim,
                    'dim': self.dim, 'files': files, 'o_proj': weights_info,
                    'note': '第 h 个头写入残差流 = W_O[:, h·hd:(h+1)·hd] @ z_h；MLP 输出已在残差空间'}
        mbytes = json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
        (out_dir / 'components.json').write_bytes(mbytes)
        return {'dir': str(out_dir), 'manifest_sha256': hashlib.sha256(mbytes).hexdigest(), 'bytes': total,
                'n_files': len(files), 'uploaded': False}


def mlp_plan(corpus, n_blocks):
    """MLP 中间单元采集口径：{'readouts': [...], 'blocks': [块号]}；语料未声明 mlp_units 时返回 None。"""
    spec = corpus.get('mlp_units')
    if not isinstance(spec, dict):
        return None
    readouts = [r for r in spec.get('readouts') or [] if r in readouts_of(corpus)]
    layers = spec.get('layers', 'all')
    blocks = (list(range(n_blocks)) if layers == 'all'
              else sorted({int(b) for b in layers if isinstance(b, int) and 0 <= b < n_blocks}))
    return {'readouts': readouts, 'blocks': blocks} if readouts and blocks else None


def silu_np(x):
    import numpy as np
    return x / (1.0 + np.exp(-x))


class MlpStore:
    """逐条 MLP 中间单元（float16）：每条 G、V 为 (P, B, d_ff)。"""

    def __init__(self, mplan, d_ff, act='silu'):
        self.readouts, self.blocks = list(mplan['readouts']), list(mplan['blocks'])
        self.d_ff, self.act = int(d_ff), act
        self.ids, self.G, self.V = [], [], []
        self.check = []            # 逐 (条目, 读出, 块) 的 ‖act(G)⊙V − down_proj 输入‖ / ‖down_proj 输入‖

    def add(self, item_id, G, V):
        import numpy as np
        self.ids.append(item_id)
        self.G.append(np.asarray(G, dtype=np.float16))
        self.V.append(np.asarray(V, dtype=np.float16))

    def _slice(self, arrs, p, k):
        """全部条目在 (读出 p, 块 k) 上的 (N, F) 数组；逐块取，避免把 (N, P, B, F) 整体堆叠进内存（大模型分层摆放时内存紧张）。"""
        import numpy as np
        return np.stack([a[p, k] for a in arrs])

    def digest(self):
        import numpy as np
        dn = np.zeros((len(self.readouts), len(self.blocks)))
        for p in range(len(self.readouts)):
            for k in range(len(self.blocks)):
                G = self._slice(self.G, p, k).astype(np.float32)
                V = self._slice(self.V, p, k).astype(np.float32)
                A = np.abs(silu_np(G) * V)
                strong = A >= np.quantile(A, 0.99, axis=0, keepdims=True)    # 每个单元自身的 99 分位
                dn[p, k] = ((G < 0) & (V < 0) & strong).sum() / max(int(strong.sum()), 1)
        out = {'readouts': self.readouts, 'blocks': self.blocks, 'd_ff': self.d_ff, 'act': self.act,
               'n_items': len(self.ids), 'double_negative_share_strong': np.round(dn, 4).tolist(),
               'note': 'double_negative_share_strong：|A| ≥ 该单元 99 分位的激活中 G<0 且 V<0（−×−）所占比例，按读出 × 块'}
        if self.check:
            err = np.array(self.check, dtype=np.float64)
            out['check'] = {'n': int(len(err)), 'rel_err_mean': float(err.mean()), 'rel_err_max': float(err.max()),
                            'note': '‖act(G)⊙V − down_proj 实际输入‖ / ‖down_proj 实际输入‖（bf16 前向，fp32 重算）'}
        return out

    def write(self, out_dir):
        import numpy as np
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files, total = [], 0
        P = len(self.readouts)
        for k, b in enumerate(self.blocks):
            buf = io.BytesIO()
            np.savez(buf, G=np.stack([self._slice(self.G, p, k) for p in range(P)]),
                     V=np.stack([self._slice(self.V, p, k) for p in range(P)]))
            data = buf.getvalue()
            name = f'B{b}.npz'
            (out_dir / name).write_bytes(data)
            total += len(data)
            files.append({'path': name, 'block': b, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        manifest = {'schema': 'mlp_units.v1', 'axes': ['readout', 'item', 'unit'], 'readouts': self.readouts,
                    'ids': list(self.ids), 'blocks': self.blocks, 'd_ff': self.d_ff, 'act': self.act, 'files': files,
                    'note': 'G = gate 预激活，V = up 投影；单元输出 A = act(G) ⊙ V（down_proj 的输入）'}
        mbytes = json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
        (out_dir / 'mlp.json').write_bytes(mbytes)
        return {'dir': str(out_dir), 'manifest_sha256': hashlib.sha256(mbytes).hexdigest(), 'bytes': total,
                'n_files': len(files), 'uploaded': False}


def save_o_proj(model_key, mats):
    """o_proj 权重（每块 (d, H·hd) float16）按模型键保存一次；返回 {'dir', 'files': [{block, path, sha256}]}。
    mats 的值可以是数组，也可以是返回数组的函数（逐块取用、写盘后即释放，避免大模型把全部块同时放进内存）。"""
    import numpy as np
    root = native_root()
    if root is None:
        return None
    wdir = root / '_weights' / ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(model_key))
    wdir.mkdir(parents=True, exist_ok=True)
    files = []
    for b, W in sorted(mats.items()):
        p = wdir / f'o_proj_B{b}.npy'
        if not p.exists():
            buf = io.BytesIO()
            np.save(buf, np.asarray(W() if callable(W) else W, dtype=np.float16))
            p.write_bytes(buf.getvalue())
            del buf
        files.append({'block': int(b), 'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return {'dir': str(wdir), 'files': files}


def suffstats_layer_subset(n_layers, n_readouts, n_cells, dim, budget=None):
    """体积超预算时的确定性层子集：步长取 2 的幂（1, 2, 4, …）的嵌套网格 {0, s, 2s, …} ∪ {末层}。
    取能放进预算的最小步长。不同模式族的条件格数不同，选出的步长可以不同，但子集互相嵌套
    （步长 2^j 的网格包含于 2^(j-1) 的网格），跨族取交集即为较粗的那个网格，谱图可以直接比较。
    只依赖语料与模型维度，所有分片相同。"""
    # corpus.suffstats_budget_bytes 可调低预算（中心单文件上传上限 64 MB）；显式预算下最少保留 {0, 末层} 两层
    limit = int(budget or SUFFSTATS_BUDGET_BYTES)
    floor = 2 if budget else 3
    per_layer = n_readouts * n_cells * dim * 16
    if per_layer * n_layers <= limit or n_layers <= floor:
        return list(range(n_layers))
    stride = 2
    while True:
        keep = sorted(set(range(0, n_layers, stride)) | {n_layers - 1})
        if len(keep) * per_layer <= limit or len(keep) <= floor:
            return keep
        stride *= 2


class ItemStore:
    """逐条向量（float16）。超过预算时投影到 k 维：投影矩阵由语料指纹确定，所有分片相同。"""

    def __init__(self, corpus_seed, readouts=None, tag_readout=None, tag_layer=None, tags=None,
                 n_total=None, budget=None):
        self.seed = corpus_seed
        self.n_total, self.budget = n_total, budget
        self.meta, self.vecs, self.lens = [], [], []
        self.tags = tags
        self.readouts = readouts or ['target']
        self.tag_readout, self.tag_layer = tag_readout, tag_layer

    def add(self, it, vectors, n_tok, correct, lens=None):
        self.meta.append({'id': it['id'], 'split': it['split'], 'cond': it['cond'],
                          'frame': it.get('frame_id', ''), 'filler': it.get('filler_id', ''),
                          'n_tok': int(n_tok), 'correct': -1 if correct is None else int(bool(correct))})
        import numpy as np
        self.vecs.append([np.asarray(v, dtype=np.float16) for v in vectors])
        self.lens.append(lens)

    def npz_bytes(self, layer_tags, lens_layers=None, vocab=None):
        import numpy as np
        L, N = len(layer_tags), len(self.meta)
        d = int(self.vecs[0][0].shape[0])
        space, k = item_projection(self.n_total or N, L, d, self.budget)
        R = jl_matrix(self.seed, d, k) if space != 'full' else None
        X = np.stack([np.stack([self.vecs[i][l].astype(np.float32) for i in range(N)]) @ R if R is not None
                      else np.stack([self.vecs[i][l] for i in range(N)]) for l in range(L)])
        arrays = dict(
            schema=np.array('items.v2'), layer_tags=np.array([str(t) for t in layer_tags]),
            space=np.array(space), dim_original=np.array(d), X=X.astype(np.float16),
            n_total=np.array(int(self.n_total or N)),
            ids=np.array([m['id'] for m in self.meta]), split=np.array([m['split'] for m in self.meta]),
            cond=np.array([json.dumps(m['cond'], sort_keys=True, ensure_ascii=False) for m in self.meta]),
            frame=np.array([m['frame'] for m in self.meta]), filler=np.array([m['filler'] for m in self.meta]),
            n_tok=np.array([m['n_tok'] for m in self.meta], dtype=np.int32),
            correct=np.array([m['correct'] for m in self.meta], dtype=np.int8),
            readouts=np.array(self.readouts),
            tag_readout=np.array(self.tag_readout or ['target'] * L),
            tag_layer=np.array(self.tag_layer if self.tag_layer is not None else [-1] * L, dtype=np.int32))
        if lens_layers is not None and self.lens and all(x is not None for x in self.lens):
            # lens_*: (读出 P, 层 Ll, 条目 N[, K])；-1 / NaN 表示该位置没有对应 token（如句末没有下一 token）
            arrays.update(
                lens_layers=np.array(lens_layers, dtype=np.int32),
                lens_next_id=np.stack([x['next_id'] for x in self.lens], axis=1).astype(np.int32),
                lens_next_logprob=np.stack([x['next_logprob'] for x in self.lens], axis=2).astype(np.float32),
                lens_answer_id=np.array([x['answer_id'] for x in self.lens], dtype=np.int32),
                lens_answer_logprob=np.stack([x['answer_logprob'] for x in self.lens], axis=2).astype(np.float32),
                lens_topk_ids=np.stack([x['topk_ids'] for x in self.lens], axis=2).astype(np.int32),
                lens_topk_logprob=np.stack([x['topk_logprob'] for x in self.lens], axis=2).astype(np.float16))
            if vocab:
                ids = sorted(vocab)
                arrays.update(lens_vocab_ids=np.array(ids, dtype=np.int32),
                              lens_vocab_text=np.array([vocab[i] for i in ids]))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue(), space


def write_native(store, out_dir, info):
    """按层写原生 float16 全场（全部读出位置）与 manifest.json；返回写入 summary 的摘要。"""
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    P = len(store.readouts)
    T = len(store.tags or store.tag_layer or [])
    Lc = T // max(P, 1)
    N = len(store.meta)
    files, total = [], 0
    for l in range(Lc):
        hs_index = int(store.tag_layer[l]) if store.tag_layer else l
        X = np.stack([np.stack([store.vecs[i][p * Lc + l] for i in range(N)]) for p in range(P)]).astype(np.float16)
        name = f'L{hs_index}.npz'
        buf = io.BytesIO()
        np.savez(buf, X=X)
        data = buf.getvalue()
        (out_dir / name).write_bytes(data)
        total += len(data)
        files.append({'path': name, 'hs_index': hs_index, 'layer_tag': (store.tags or [''] * T)[l].split('@', 1)[-1],
                      'array': 'X', 'shape': list(X.shape), 'dtype': 'float16', 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(), 'nonfinite': int((~np.isfinite(X)).sum())})
    manifest = {'schema': 'native_field.v1', 'axes': ['readout', 'item', 'dim'], 'readouts': list(store.readouts),
                'ids': [m['id'] for m in store.meta], 'tags': list(store.tags or []),
                'tag_readout': list(store.tag_readout or []), 'tag_layer': [int(x) for x in (store.tag_layer or [])],
                'files': files, **info}
    mbytes = json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    (out_dir / 'manifest.json').write_bytes(mbytes)
    return {'dir': str(out_dir), 'manifest': 'manifest.json', 'manifest_sha256': hashlib.sha256(mbytes).hexdigest(),
            'n_files': len(files), 'bytes': total, 'dtype': 'float16', 'uploaded': False,
            'nonfinite': sum(f['nonfinite'] for f in files),
            'note': '原生全场只存于节点本地（不上传）；按 manifest 的 sha256 校验后可按需拉取或用同一语料与模型重算'}


class Acc:
    """与 shared/dist_protocol/suffstats.py 同格式的累加器（runner 自包含，不 import 中心库）。"""

    def __init__(self, layer_tags, dim, factors, tag_readout=None, tag_layer=None):
        import numpy as np
        self.np = np
        self.layer_tags, self.dim, self.factors = list(layer_tags), dim, list(factors)
        self.tag_readout, self.tag_layer = tag_readout, tag_layer
        self.cells, self.index, self.rows = [], {}, []

    def add(self, split, cond, vectors):
        np = self.np
        key = (split, json.dumps(cond, sort_keys=True, ensure_ascii=False))
        if key not in self.index:
            self.index[key] = len(self.cells)
            self.cells.append(key)
            L = len(self.layer_tags)
            self.rows.append([0, np.zeros((L, self.dim)), np.zeros((L, self.dim))])
        row = self.rows[self.index[key]]
        row[0] += 1
        for l, v in enumerate(vectors):
            v = np.asarray(v, dtype=np.float64)
            row[1][l] += v
            row[2][l] += v * v

    def npz_bytes(self):
        np = self.np
        buf = io.BytesIO()
        L, C = len(self.layer_tags), len(self.cells)
        s = np.zeros((L, C, self.dim)); q = np.zeros((L, C, self.dim))
        for i, row in enumerate(self.rows):
            s[:, i] = row[1]; q[:, i] = row[2]
        np.savez_compressed(
            buf, schema=np.array('suffstats.v2'), layer_tags=np.array([str(t) for t in self.layer_tags]),
            factors=np.array(json.dumps(self.factors, ensure_ascii=False)),
            cell_split=np.array([c[0] for c in self.cells]), cell_cond=np.array([c[1] for c in self.cells]),
            n=np.array([r[0] for r in self.rows], dtype=np.int64), sum=s, sumsq=q,
            tag_readout=np.array(self.tag_readout or ['target'] * L),
            tag_layer=np.array(self.tag_layer if self.tag_layer is not None else [-1] * L, dtype=np.int32))
        return buf.getvalue()

    def eta2_digest(self, split='discovery'):
        """本分片 discovery 的逐因子平均 η²（仅供节点自检；正式统计在中心合并后计算）。"""
        np = self.np
        idx = [i for i, c in enumerate(self.cells) if c[0] == split]
        if len(idx) < 2:
            return {}
        out = {}
        for l, tag in enumerate(self.layer_tags):
            n = np.array([self.rows[i][0] for i in idx], dtype=float)
            s = np.stack([self.rows[i][1][l] for i in idx]); q = np.stack([self.rows[i][2][l] for i in idx])
            N = n.sum(); g = s.sum(0) / N
            sst = np.maximum(q.sum(0) - N * g * g, 1e-12)
            conds = [json.loads(self.cells[i][1]) for i in idx]
            per = {}
            for f in self.factors:
                lv = {}
                for j, c in enumerate(conds):
                    lv.setdefault(str(c[f]), []).append(j)
                ssf = np.zeros_like(g)
                for js in lv.values():
                    nl = n[js].sum(); ml = s[js].sum(0) / nl
                    ssf += nl * (ml - g) ** 2
                per[f] = float((ssf / sst).mean())
            out[tag] = per
        return out


class Plan:
    """一次运行的采集口径：读出 × 层 → 全部标签（items）与 suffstats 子集。"""

    def __init__(self, corpus, items, n_hs, dim):
        self.readouts = readouts_of(corpus)
        self.layers = resolve_layers(layer_spec_of(corpus), n_hs)
        self.tags = full_tags(self.readouts, self.layers)
        self.tag_readout = [r for r in self.readouts for _ in self.layers]
        self.tag_layer = [i for _ in self.readouts for _, i in self.layers]
        n_cells = len({(it['split'], cond_key(it['cond'])) for it in items}) or 1
        self.n_total = len(items)
        self.ss_budget = corpus.get('suffstats_budget_bytes')
        keep = suffstats_layer_subset(len(self.layers), len(self.readouts), n_cells, dim, self.ss_budget)
        self.ss_index = [p * len(self.layers) + l for p in range(len(self.readouts)) for l in keep]
        self.ss_layers = [self.layers[l][0] for l in keep]

    def ss_select(self, vecs):
        return [vecs[i] for i in self.ss_index]

    def acc(self, dim, factors):
        return Acc([self.tags[i] for i in self.ss_index], dim, factors,
                   [self.tag_readout[i] for i in self.ss_index], [self.tag_layer[i] for i in self.ss_index])

    def store(self, corpus):
        if not corpus.get('capture_items', True):
            return None
        return ItemStore(corpus_seed(corpus), self.readouts, self.tag_readout, self.tag_layer, self.tags,
                         n_total=self.n_total, budget=items_budget(corpus))

    def describe(self, n_hs):
        return {'readouts': self.readouts, 'n_hidden_states': n_hs, 'n_items_corpus': self.n_total,
                'layer_plan': [{'tag': t, 'hs_index': i} for t, i in self.layers],
                'suffstats_layers': self.ss_layers,
                'suffstats_layers_note': ('全部层' if len(self.ss_layers) == len(self.layers) else
                                          f'体积超预算（{int(self.ss_budget or SUFFSTATS_BUDGET_BYTES) >> 20} MB），suffstats 只保留 2 的幂步长嵌套网格上的层；'
                                          'items.npz 与 logit-lens 仍覆盖全部层')}


def summarize_vec(v):
    vals = [float(x) for x in v]
    return {'norm': math.sqrt(sum(x * x for x in vals)), 'mean': sum(vals) / max(len(vals), 1),
            'head': vals[:HEAD_DIMS], 'dim': len(vals)}


def model_snapshot(model_path):
    """模型权重与 tokenizer 文件的 sha256（按文件名排序后再合成一个总哈希），作为 model_snapshot。"""
    root = Path(model_path)
    files = {}
    if root.is_dir():
        seen = set()
        for pat in SNAPSHOT_PATTERNS:
            for p in sorted(root.glob(pat)):
                if p.is_file() and p.name not in seen:
                    seen.add(p.name)
                    h = hashlib.sha256()
                    with p.open('rb') as fh:
                        for chunk in iter(lambda: fh.read(1 << 22), b''):
                            h.update(chunk)
                    files[p.name] = {'sha256': h.hexdigest(), 'size': p.stat().st_size}
    total = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
    return {'sha256': total, 'files': files, 'n_files': len(files),
            'note': '权重（*.safetensors 等）、config 与 tokenizer 文件逐个 sha256；sha256 为按文件名排序后的合成哈希'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='auto', choices=['auto', 'smoke', 'real', 'synthetic'])
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    corpus = json.loads((here / 'corpus.json').read_text(encoding='utf-8'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = parse_shard(os.environ.get('AI2050_SHARD', '0/1'))
    precision = os.environ.get('AI2050_PRECISION', 'bf16').lower()

    all_items = corpus['items']
    canaries = [it for it in all_items if it.get('canary')]
    study = [it for it in all_items if not it.get('canary')]
    mine = [it for k, it in enumerate(study) if k % shard_n == shard_i]
    factors = list(corpus.get('factors', []))
    fp = hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]

    cells, splits = {}, {}
    for it in mine:
        cells[cond_key(it['cond'])] = cells.get(cond_key(it['cond']), 0) + 1
        splits[it['split']] = splits.get(it['split'], 0) + 1
    summary = {
        'protocol': 'dist_protocol.v3', 'analysis': 'suffstats', 'tm_id': corpus.get('tm_id'),
        'family_id': corpus.get('family_id'), 'corpus_fingerprint': fp,
        'shard': f'{shard_i}/{shard_n}', 'n_items': len(mine), 'n_items_total': len(study),
        'by_split': splits, 'conditions': dict(sorted(cells.items())), 'factors': factors,
        'layers': [str(t) for t in layer_spec_of(corpus)], 'readouts': readouts_of(corpus), 'precision': precision,
        'python': platform.python_version(), 'status': 'smoke',
    }

    mode = args.mode
    if mode == 'auto' and os.environ.get('AI2050_RUNNER_MODE') in ('smoke', 'real', 'synthetic'):
        mode = os.environ['AI2050_RUNNER_MODE']
    if mode == 'auto':
        mode = 'real' if os.environ.get('AI2050_RUNNER_MODEL') else 'smoke'

    try:
        if mode == 'real':
            run_real(corpus, study, mine, canaries, factors, precision, out, summary)
        elif mode == 'synthetic':
            run_synthetic(corpus, study, mine, canaries, factors, out, summary)
    except Exception as exc:  # 真实模式失败 → 诚实降级为 smoke，记录原因
        summary['status'] = 'smoke'
        summary['real_error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        summary['real_traceback'] = traceback.format_exc()[-4000:]
        for name in ('suffstats.npz', 'behavior.json', 'canary.json', 'items.npz'):
            p = out / name
            if p.exists():
                p.unlink()

    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                                      encoding='utf-8')
    print(f"[runner-v3] status={summary['status']} shard={summary['shard']} items={len(mine)} out={out}")
    return 0


def _target_pos(text, target, offsets, start=None):
    s = start if isinstance(start, int) and start >= 0 else text.find(target)
    if s < 0 or text[s:s + len(target)] != target:
        raise ValueError(f'target {target!r} not in text {text!r}')
    e = s + len(target)
    cands = [i for i, (cs, ce) in enumerate(offsets) if cs < e and ce > s]
    if not cands:
        raise ValueError(f'target {target!r} has no token span in {text!r}')
    return max(cands)


def _final_pos(offsets):
    """最后一个有字符跨度的 token（跳过末尾的特殊 token）。"""
    for i in range(len(offsets) - 1, -1, -1):
        if offsets[i][1] > offsets[i][0]:
            return i
    return len(offsets) - 1


def text_positions(it, offsets, readouts, straddle=None):
    """文本内读出位置（probe_answer 在 probe 提示上单独前向）。
    straddle：读出 token 越过片段末尾（分词把片段与后文合成一个 token）时按读出计数——此时读出状态已含后文。"""
    pos = {}
    spans = []
    if 'target' in readouts:
        pos['target'] = _target_pos(it['text'], it['target'], offsets, it.get('target_char'))
        spans.append(('target', it.get('target_char', it['text'].find(it['target'])) + len(it['target'])))
    if 'cue' in readouts:
        if not it.get('cue'):
            raise ValueError(f'item {it.get("id")} 缺少 cue（语料声明了 cue 读出）')
        pos['cue'] = _target_pos(it['text'], it['cue'], offsets, it.get('cue_char'))
        spans.append(('cue', it.get('cue_char', it['text'].find(it['cue'])) + len(it['cue'])))
    if 'final' in readouts:
        pos['final'] = _final_pos(offsets)
    if straddle is not None:
        for r, end in spans:
            if offsets[pos[r]][1] > end:
                straddle[r] = straddle.get(r, 0) + 1
    return pos


def parse_max_memory(spec):
    """"0=12GiB,cpu=20GiB" → {0: '12GiB', 'cpu': '20GiB'}；空串 → None。"""
    out = {}
    for part in str(spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition('=')
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f'AI2050_MAX_MEMORY 格式应为 "0=12GiB,cpu=20GiB"，收到 {spec!r}')
        out[int(k) if k.isdigit() else k] = v
    return out or None


def module_device(mod):
    """模块参数所在的执行设备（跳过 meta）；没有实体参数时返回 None。"""
    for p in mod.parameters():
        if p.device.type != 'meta':
            return p.device
    return None


def checkpoint_tensor(model_path, key):
    """从 safetensors 检查点按参数名读取原始张量（单文件或带 model.safetensors.index.json 的分片）。"""
    from safetensors import safe_open
    root = Path(model_path)
    idx = root / 'model.safetensors.index.json'
    if idx.is_file():
        fname = json.loads(idx.read_text(encoding='utf-8'))['weight_map'].get(key)
        if fname is None:
            raise KeyError(f'检查点中没有参数 {key}')
        path = root / fname
    else:
        path = root / 'model.safetensors'
    with safe_open(str(path), framework='pt') as f:
        return f.get_tensor(key)


def dense_weight(mod, ckpt=None):
    """线性层的稠密 float32 权重（CPU）。
    - bitsandbytes 4-bit（NF4/FP4）先反量化：直接读 .weight 得到的是打包后的 uint8，按它做逐头分解会静默出错；
    - 分层摆放（device_map=auto）时卸载到 CPU/磁盘的层，模块里只剩 meta 占位，真实权重由 accelerate 另行保管：
      给出 ckpt=(模型目录, 参数名) 时从检查点读取原始权重（与运行时使用的权重相同），否则明确报错。"""
    import torch
    w = mod.weight
    qs = getattr(w, 'quant_state', None)
    if qs is not None:
        import bitsandbytes.functional as bnbf
        return bnbf.dequantize_4bit(w.data, qs).float().cpu()
    if w.device.type == 'meta':
        # accelerate 的 AlignDevicesHook 持有卸载权重（weights_map 用相对键 'weight' 取到真实的 CPU / 磁盘张量）
        hook = getattr(mod, '_hf_hook', None)
        for h in (getattr(hook, 'hooks', None) or ([hook] if hook is not None else [])):
            wm = getattr(h, 'weights_map', None)
            if wm is not None:
                try:
                    t = wm['weight']
                except KeyError:
                    continue
                if tuple(t.shape) == tuple(w.shape):
                    return t.float().cpu()
        if ckpt is None:
            raise RuntimeError('权重是 meta 占位（已卸载），hook 中也没有可用权重，又没有给出检查点参数名，无法读取')
        t = checkpoint_tensor(*ckpt)
        if tuple(t.shape) != tuple(w.shape):
            raise RuntimeError(f'检查点参数 {ckpt[1]} 形状 {tuple(t.shape)} 与模块 {tuple(w.shape)} 不一致')
        return t.float().cpu()
    if not torch.is_floating_point(w):
        raise RuntimeError(f'不支持的权重类型 {w.dtype}（只支持浮点与 bitsandbytes 4-bit）')
    return w.detach().float().cpu()


def run_real(corpus, study, items, canaries, factors, precision, out, summary):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ['AI2050_RUNNER_MODEL']
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(precision, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 放置方式：缺省整模型放在 device；AI2050_DEVICE_MAP=auto 时按 AI2050_MAX_MEMORY 做 GPU/CPU 分层摆放（权重保持原精度，
    # 显存放不下的大模型用，如 Qwen3-14B bf16：max_memory 0=12GiB,cpu=20GiB）
    dm = os.environ.get('AI2050_DEVICE_MAP', '').strip().lower()
    max_memory = None
    loader = 'from_pretrained'
    if dm == 'auto':
        # 按分片逐个加载并摆放（accelerate.load_checkpoint_and_dispatch）：transformers 5 的 from_pretrained 会同时以
        # 写时复制方式映射全部分片，大模型（Qwen3-14B bf16 约 29.5 GB）在 Windows 上逼近提交内存上限时会访问冲突崩溃
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig
        max_memory = parse_max_memory(os.environ.get('AI2050_MAX_MEMORY', ''))
        offload_dir = os.environ.get('AI2050_OFFLOAD_DIR') or str(Path(out).resolve().parent / '_offload')
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path), torch_dtype=dtype)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map='auto', max_memory=max_memory, dtype=dtype, offload_folder=offload_dir,
            offload_buffers=True, no_split_module_classes=list(getattr(model, '_no_split_modules', None) or []))
        loader = 'accelerate.load_checkpoint_and_dispatch'
    elif dm:
        raise ValueError(f'AI2050_DEVICE_MAP 只支持 auto（收到 {dm!r}）')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
    model.eval()
    hf_map = getattr(model, 'hf_device_map', None) or {}
    by_dev = {}
    for v in hf_map.values():
        by_dev[str(v)] = by_dev.get(str(v), 0) + 1
    summary['placement'] = {'device_map': 'auto' if dm == 'auto' else device, 'loader': loader,
                            'max_memory': {str(k): v for k, v in (max_memory or {}).items()} or None,
                            'modules_by_device': by_dev or {device: 'all'},
                            'note': '分层摆放只改变执行位置，权重精度不变；disk 表示卸载到磁盘（很慢）'}
    summary['model_snapshot'] = model_snapshot(model_path)
    readouts = readouts_of(corpus)
    lm_head = model.get_output_embeddings()
    decoder = model.get_decoder() if hasattr(model, 'get_decoder') else getattr(model, 'model', None)
    final_norm = getattr(decoder, 'norm', None) or getattr(decoder, 'final_layernorm', None)
    state = {'plan': None, 'last_post_norm': None, 'vocab': {}}

    # 部件采集（corpus.components）：o_proj 的输入（按头拆分前的拼接 z）与 MLP 输出，只在研究句的前向中记录
    blocks_mod = list(getattr(decoder, 'layers', []) or [])
    cplan = component_plan(corpus, len(blocks_mod))
    cap = {'on': False, 'z': {}, 'm': {}}
    o_projs = {}
    if cplan:
        def _attn(layer):
            return getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
        for b in cplan['blocks']:
            layer = blocks_mod[b]
            attn, mlp = _attn(layer), getattr(layer, 'mlp', None)
            o_proj = getattr(attn, 'o_proj', None) if attn is not None else None
            if o_proj is None or mlp is None:
                raise RuntimeError(f'块 {b} 没有 self_attn.o_proj / mlp，无法采集部件')
            if getattr(o_proj, 'bias', None) is not None:
                summary.setdefault('components_warnings', []).append(f'块 {b} 的 o_proj 有偏置：逐头分解不含偏置项')
            o_projs[b] = o_proj

            def pre(mod, args, b=b):
                if cap['on']:
                    cap['z'][b] = args[0].detach()

            def post(mod, args, out, b=b):
                if cap['on']:
                    cap['m'][b] = (out[0] if isinstance(out, tuple) else out).detach()
            o_proj.register_forward_pre_hook(pre)
            mlp.register_forward_hook(post)
        cfg = model.config
        n_heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, 'head_dim', None) or cfg.hidden_size // n_heads)
    comp_store = None

    # MLP 中间单元（corpus.mlp_units）：gate 预激活 G、value V 与 down_proj 的实际输入（只用于核对）
    mplan = mlp_plan(corpus, len(blocks_mod))
    mcap = {'on': False, 'g': {}, 'v': {}, 'a': {}}
    mlp_store, mlp_act = None, None
    if mplan:
        for b in mplan['blocks']:
            mlp = getattr(blocks_mod[b], 'mlp', None)
            if mlp is None or getattr(mlp, 'down_proj', None) is None:
                raise RuntimeError(f'块 {b} 没有 mlp.down_proj，无法采集 MLP 中间单元')
            act = getattr(mlp, 'act_fn', None) or getattr(mlp, 'activation_fn', None)
            aname = type(act).__name__.lower()
            if 'silu' not in aname and 'swish' not in aname:
                raise RuntimeError(f'块 {b} 的 MLP 激活为 {type(act).__name__}，当前只支持 SiLU 门控（SwiGLU）')
            mlp_act = 'silu'
            if getattr(mlp, 'gate_proj', None) is not None and getattr(mlp, 'up_proj', None) is not None:
                def hg(mod, args, out, b=b):
                    if mcap['on']:
                        mcap['g'][b] = out.detach()

                def hv(mod, args, out, b=b):
                    if mcap['on']:
                        mcap['v'][b] = out.detach()
                mlp.gate_proj.register_forward_hook(hg)
                mlp.up_proj.register_forward_hook(hv)
            elif getattr(mlp, 'gate_up_proj', None) is not None:
                # 合并投影（如 GLM4）：输出按最后一维对半切，前半为 gate、后半为 up（与 HF Glm4MLP.forward 一致）
                def hgu(mod, args, out, b=b):
                    if mcap['on']:
                        g_, v_ = out.detach().chunk(2, dim=-1)
                        mcap['g'][b], mcap['v'][b] = g_, v_
                mlp.gate_up_proj.register_forward_hook(hgu)
            else:
                raise RuntimeError(f'块 {b} 的 MLP 没有 gate_proj/up_proj 或 gate_up_proj')

            def ha(mod, args, b=b):
                if mcap['on']:
                    mcap['a'][b] = args[0].detach()
            mlp.down_proj.register_forward_pre_hook(ha)

    def mlp_rows(p):
        """当前捕获在位置 p 的 (G, V)，各 (B, F)；前若干条逐块核对 act(G)⊙V 与 down_proj 的实际输入。"""
        G = torch.stack([mcap['g'][b][0, p].float().cpu() for b in mplan['blocks']])
        V = torch.stack([mcap['v'][b][0, p].float().cpu() for b in mplan['blocks']])
        if mlp_store is not None and len(mlp_store.check) < 64 * len(mplan['blocks']):
            A = torch.nn.functional.silu(G) * V
            for k, b in enumerate(mplan['blocks']):
                ref = mcap['a'][b][0, p].float().cpu()
                mlp_store.check.append(float((A[k] - ref).norm() / ref.norm().clamp_min(1e-6)))
        return G.numpy(), V.numpy()

    def run(text, capture=False, mlp=False):
        enc = tok(text, return_tensors='pt', return_offsets_mapping=True)
        off = enc.pop('offset_mapping')[0].tolist()
        ids = enc['input_ids'][0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        cap['on'] = bool(capture and cplan)
        mcap['on'] = bool(mlp and mplan)
        try:
            res = model(**enc, output_hidden_states=True)
        finally:
            cap['on'] = False
            mcap['on'] = False
        hs = res.hidden_states
        if state['last_post_norm'] is None:
            # 对账：HF 的 hidden_states[-1] 是否已经过最终归一化（lm_head(hs[-1]) 应等于实际 logits）
            direct = lm_head(to_dev(hs[-1][0, -1:], head_dev)).float().cpu()
            state['last_post_norm'] = bool(torch.allclose(direct, res.logits[0, -1:].float().cpu(), rtol=2e-2, atol=2e-2))
        return hs, res.logits, off, ids

    # 分层摆放时各层隐藏状态在不同设备上：送入最终归一化 / lm_head 前先搬到该模块所在设备
    norm_dev = module_device(final_norm) if final_norm is not None else None
    head_dev = module_device(lm_head)

    def to_dev(x, dev):
        return x.to(dev) if dev is not None else x

    def lens_at(hs, pos, layers):
        """逐层 logit-lens：中间层先过最终归一化再接 lm_head；hs[-1] 若已过归一化则直接接 lm_head。"""
        n_hs = len(hs)
        normed = []
        for _, i in layers:
            row = hs[i][0, pos]
            if i == n_hs - 1 and state['last_post_norm']:
                normed.append(to_dev(row, head_dev))
            else:
                r = final_norm(to_dev(row, norm_dev)[None])[0] if final_norm is not None else row
                normed.append(to_dev(r, head_dev))
        return torch.log_softmax(lm_head(torch.stack(normed)).float(), dim=-1)    # (L, V)

    def gather(logp, tid):
        if tid is None or tid < 0:
            return np.full((logp.shape[0],), np.nan, dtype=np.float32)
        return logp[:, tid].cpu().numpy().astype(np.float32)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def answer_logprobs(prompt_ids, cands):
        """完整答案对数概率 log P(答案 | 提示)：提示与答案分别分词后拼接（teacher forcing），右侧补齐后一次前向。"""
        seqs = [tok(c, add_special_tokens=False)['input_ids'] for c in cands]
        keep = [(c, s) for c, s in zip(cands, seqs) if s]
        if not keep:
            return {}
        n_p = len(prompt_ids)
        width = n_p + max(len(s) for _, s in keep)
        ids = torch.full((len(keep), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(keep), width), dtype=torch.long)
        for b, (_, s) in enumerate(keep):
            ids[b, :n_p + len(s)] = torch.tensor(list(prompt_ids) + s)
            mask[b, :n_p + len(s)] = 1
        lp = torch.log_softmax(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float(), dim=-1)
        out = {}
        for b, (c, s) in enumerate(keep):
            pos = torch.arange(n_p - 1, n_p - 1 + len(s), device=lp.device)
            out[c] = float(lp[b, pos, torch.tensor(s, device=lp.device)].sum().item())
        return out

    def score_probe(probe, prompt_ids, top1_id):
        """行为评分：有对照集 → 完整答案强制选择；否则 → 下一个 token 是否为任一可接受答案的首 token。"""
        answers = probe.get('answers') or [probe['answer']]
        contrast = probe.get('contrast') or []
        firsts = {s[0] for s in (tok(a, add_special_tokens=False)['input_ids'] for a in answers) if s}
        rec = {'set_top1': top1_id in firsts}
        if contrast:
            lp = answer_logprobs(prompt_ids, list(answers) + [c for c in contrast if c not in answers])
            a_best = max(((lp[a], a) for a in answers if a in lp), default=(None, None))
            c_best = max(((lp[c], c) for c in contrast if c in lp), default=(None, None))
            if a_best[0] is not None and c_best[0] is not None:
                rec.update({'mode': 'forced_choice', 'correct': a_best[0] > c_best[0],
                            'answer_logprob': a_best[0], 'best_answer': a_best[1],
                            'contrast_logprob': c_best[0], 'best_contrast': c_best[1],
                            'margin': a_best[0] - c_best[0], 'n_contrast': len(contrast)})
                return rec
        rec.update({'mode': 'set_first_token', 'correct': rec['set_top1']})
        return rec

    def topk(logp):
        v, i = torch.topk(logp, LENS_TOPK, dim=-1)
        for t in i.flatten().tolist():
            if t not in state['vocab']:
                state['vocab'][t] = tok.decode([t])
        return i.cpu().numpy(), v.cpu().numpy()

    acc = store = plan = None
    tok_len, behavior, straddle = {}, [], {}
    # 同一 probe（提示、答案集、对照集都相同）在多个句式里重复出现（如 ENT-XF 的肯定/否定与句式变体）：
    # 不需要 probe_answer 读出时缓存评分结果，结果与逐条重算相同
    probe_memo = {} if 'probe_answer' not in readouts else None
    with torch.inference_mode():
        for it in items:
            hs, _, off, ids = run(it['text'], capture=True, mlp=True)
            if plan is None:
                plan = Plan(corpus, study, len(hs), int(hs[0].shape[-1]))
                acc, store = plan.acc(int(hs[0].shape[-1]), factors), plan.store(corpus)
                summary.update(plan.describe(len(hs)))
                if cplan:
                    comp_store = ComponentStore(cplan, n_heads, head_dim, int(hs[0].shape[-1]))
                if mplan:
                    mlp_store = MlpStore(mplan, int(mcap['g'][mplan['blocks'][0]].shape[-1]), mlp_act)
            pos = text_positions(it, off, readouts, straddle)
            mrows = {}
            if mlp_store is not None:
                for r in mlp_store.readouts:
                    if r != 'probe_answer':
                        mrows[r] = mlp_rows(pos[r])
                mcap['g'].clear(); mcap['v'].clear(); mcap['a'].clear()
            if comp_store is not None:
                zs, ms = [], []
                for r in comp_store.readouts:
                    p = pos[r]
                    # 各块可能在不同设备上（分层摆放）：逐块搬到 CPU 再拼接
                    z = torch.stack([cap['z'][b][0, p].float().cpu() for b in comp_store.blocks])   # (B, H·hd)
                    m = torch.stack([cap['m'][b][0, p].float().cpu() for b in comp_store.blocks])   # (B, d)
                    for k, b in enumerate(comp_store.blocks):
                        if b + 1 < len(hs) - 1 or not state['last_post_norm']:
                            delta = hs[b + 1][0, p].float().cpu() - hs[b][0, p].float().cpu()
                            zk = cap['z'][b][0, p][None]
                            recon = o_projs[b](to_dev(zk, module_device(o_projs[b])))[0].float().cpu() + m[k]
                            comp_store.closure.append(float((delta - recon).norm() / delta.norm().clamp_min(1e-6)))
                    zs.append(z.numpy())
                    ms.append(m.numpy())
                comp_store.add(it['id'], np.stack(zs), np.stack(ms))
                cap['z'].clear(); cap['m'].clear()
            probe = it.get('probe')
            ans_id, correct, phs, pids = None, None, None, None
            if probe:
                ans_ids = tok(probe['answer'], add_special_tokens=False)['input_ids']
                key = json.dumps([probe['prompt'], probe['answer'], probe.get('answers'), probe.get('contrast')],
                                 ensure_ascii=False) if probe_memo is not None else None
                if key is not None and key in probe_memo:
                    core = probe_memo[key]
                elif ans_ids:
                    phs, plogits, _, pids = run(probe['prompt'], mlp=True)
                    ans_id = int(ans_ids[0])
                    logits = plogits[0, -1].float()
                    logp = torch.log_softmax(logits, dim=-1)
                    rank = int((logits > logits[ans_id]).sum().item()) + 1
                    core = {'rank': rank, 'logprob': float(logp[ans_id].item()), 'first_top1': rank == 1}
                    core.update(score_probe(probe, pids, int(torch.argmax(logits).item())))
                    if key is not None:
                        probe_memo[key] = core
                else:
                    core = None
                    phs, _, _, pids = run(probe['prompt'], mlp=True)
                if ans_ids:
                    ans_id = int(ans_ids[0])
                if core is not None:
                    rec = {'id': it['id'], 'split': it['split'], 'cell': cond_key(it['cond']), **core}
                    correct = rec['correct']
                    behavior.append(rec)
            if 'probe_answer' in readouts and phs is None:
                raise ValueError(f'item {it["id"]} 没有 probe（语料声明了 probe_answer 读出）')
            if mlp_store is not None:
                if 'probe_answer' in mlp_store.readouts:
                    mrows['probe_answer'] = mlp_rows(len(pids) - 1)
                mcap['g'].clear(); mcap['v'].clear(); mcap['a'].clear()
                mlp_store.add(it['id'], np.stack([mrows[r][0] for r in mlp_store.readouts]),
                              np.stack([mrows[r][1] for r in mlp_store.readouts]))
            vecs, lens = [], {'next_id': [], 'next_logprob': [], 'answer_logprob': [], 'topk_ids': [], 'topk_logprob': []}
            for r in readouts:
                src, p, seq = (phs, len(pids) - 1, pids) if r == 'probe_answer' else (hs, pos[r], ids)
                vecs.extend(src[i][0, p].float().cpu().numpy() for _, i in plan.layers)
                logp = lens_at(src, p, plan.layers)
                nid = int(seq[p + 1]) if p + 1 < len(seq) else (ans_id if r == 'probe_answer' else -1)
                nid = -1 if nid is None else nid
                ti, tv = topk(logp)
                lens['next_id'].append(nid)
                lens['next_logprob'].append(gather(logp, nid))
                lens['answer_logprob'].append(gather(logp, ans_id))
                lens['topk_ids'].append(ti)
                lens['topk_logprob'].append(tv)
            lens = {'next_id': np.array(lens['next_id']), 'next_logprob': np.stack(lens['next_logprob']),
                    'answer_id': -1 if ans_id is None else ans_id, 'answer_logprob': np.stack(lens['answer_logprob']),
                    'topk_ids': np.stack(lens['topk_ids']), 'topk_logprob': np.stack(lens['topk_logprob'])}
            acc.add(it['split'], it['cond'], plan.ss_select(vecs))
            tok_len.setdefault(cond_key(it['cond']), []).append(len(off))
            if store is not None:
                store.add(it, vecs, len(off), correct, lens)
        can = {}
        for c in canaries:
            hs, _, off, _ = run(c['text'])
            p = _target_pos(c['text'], c['target'], off)
            can[c['id']] = {t: summarize_vec(hs[i][0, p].float().cpu().numpy()) for t, i in plan.layers} if plan else {}
    if acc is None:
        raise RuntimeError('本分片没有样本')
    summary['readout_straddle'] = {'counts': straddle, 'n_items': len(items),
                                   'note': '读出 token 越过目标/cue 片段末尾的条目数（分词合并了后文）'}
    summary['final_layer_norm'] = 'post_final_norm' if state['last_post_norm'] else 'pre_final_norm（lm_head 对账不符，lens 对 hs[-1] 也施加最终归一化）'
    summary['hidden_state_boundary'] = ('hs[0] = 嵌入层输出（第 0 块输入）；hs[i] = 第 i 块之后的残差流；'
                                        'hs[-1] 见 final_layer_norm；位置为所选 token，采集为单条前向、无 padding、无 KV 缓存复用')
    model_key = f"{summary['model_snapshot']['sha256'][:16]}-{precision}"
    weights = None
    if comp_store is not None and comp_store.ids:
        names = {id(m): n for n, m in model.named_modules()}
        weights = lambda: save_o_proj(model_key, {  # noqa: E731
            b: (lambda b=b: dense_weight(o_projs[b], (model_path, names[id(o_projs[b])] + '.weight')).half().numpy())
            for b in comp_store.blocks})
    _write_outputs(out, acc, behavior, can, tok_len, summary, status='real',
                   extra={'model_path': model_path, 'device': device}, store=store,
                   lens_layers=[i for _, i in plan.layers], vocab=state['vocab'],
                   native={'corpus': corpus, 'shard': _shard_tuple(summary), 'model_key': model_key},
                   components=(comp_store, weights) if weights else None,
                   mlp=mlp_store if mlp_store is not None and mlp_store.ids else None)


def _synthetic_coef(tag, hs_index, n_hs):
    """合成系数 (词身份, 主因子, 句式因子, 交互)。first/mid/last 与旧版一致；L{i} 按相对深度插值，L0（嵌入）只有词身份。"""
    fixed = {'first': (2.0, 0.3, 1.5, 0.0), 'mid': (0.8, 1.6, 0.6, 0.0), 'last': (0.5, 1.1, 1.2, 0.4)}
    if tag in fixed:
        return fixed[tag]
    if hs_index == 0 or tag == 'emb':
        return (2.0, 0.0, 0.0, 0.0)
    d = hs_index / max(n_hs - 1, 1)
    d1 = 1 / max(n_hs - 1, 1)
    pts = [(d1, fixed['first']), (0.5, fixed['mid']), (1.0, fixed['last'])]
    if d <= pts[0][0]:
        return pts[0][1]
    for (xa, a), (xb, b) in zip(pts, pts[1:]):
        if d <= xb:
            t = (d - xa) / max(xb - xa, 1e-9)
            return tuple(u + t * (v - u) for u, v in zip(a, b))
    return fixed['last']


def run_synthetic(corpus, study, items, canaries, factors, out, summary):
    """方法验证用合成数据（已知答案）。对每个读出 r、层位 l：
        x = a_l·词身份(filler) + b_l·Σ类因子方向 + c_l·句式方向 + e_l·交互(格) + 0.04·字数·干扰方向 + 噪声（总范数约 0.8）
    其中「类因子」= factors[0]，「句式因子」= 其余因子；系数随层变化：
        first: 词身份 2.0、类 0.3、句式 1.5   mid: 0.8、1.6、0.6   last: 0.5、1.1、1.2（另有交互 0.4）
    layers="all" 时层数 = corpus.synthetic_layers（缺省 9），L0 视为嵌入层（只有词身份），其余按相对深度插值。
    非 target 读出另加一个读出偏移方向、词身份减半；corpus.visibility 标为不可见的 (读出, 因子) 不植入该因子。
    行为：factors[0] 的第 5 个水平（字典序）在第 2 个句式水平上正确率约 55%，其余约 95%。
    这些参数只用于检验分析方法能否把植入的规律找回来，不代表任何真实模型。"""
    import numpy as np
    dim = int(corpus.get('synthetic_dim', 96))
    n_hs = int(corpus.get('synthetic_layers', 9))
    plan = Plan(corpus, study, n_hs, dim)
    summary.update(plan.describe(n_hs))
    summary['final_layer_norm'] = 'synthetic'
    readouts = plan.readouts
    visibility = corpus.get('visibility') or {}
    salt = os.environ.get('AI2050_SYNTHETIC_SALT', '')

    def unit(name):
        rng = np.random.default_rng(int(hashlib.sha256((salt + name).encode()).hexdigest()[:8], 16))
        v = rng.standard_normal(dim)
        return v / np.linalg.norm(v)

    main_f = factors[0] if factors else None
    other_f = factors[1:]
    concepts = corpus.get('concepts') or {}

    def level_dir(f, lv):
        # 文法预注册了概念的水平（如 frame=interrog → 疑问）用「概念」作种子：同一合成模型里，不同模式族中
        # 同概念的水平共享方向。这是 atlas.v1 的阳性对照（植入共享结构），不代表真实模型。
        tag = concepts.get(f'{f}={lv}')
        return unit('concept=' + tag) if tag else unit(f'{f}={lv}')

    def visible(r, f):
        return bool((visibility.get(r) or {}).get(f, True))

    lv_main = sorted({str(it['cond'][main_f]) for it in study}) if main_f else []
    lv_other = sorted({str(it['cond'][other_f[0]]) for it in study}) if other_f else []
    weak = (lv_main[4] if len(lv_main) > 4 else None, lv_other[1] if len(lv_other) > 1 else None)
    nuis = unit('__length__')
    acc, store = plan.acc(dim, factors), plan.store(corpus)
    behavior, tok_len = [], {}
    n_layers = len(plan.layers)
    # 合成部件（只用于检验采集与闭合格式）：每块 H 个头（随机 W_O、随机 z）+ MLP = 残差增量 − Σ 头，闭合精确成立
    cplan = component_plan(corpus, n_hs - 1) if [i for _, i in plan.layers] == list(range(n_hs)) else None
    comp_store = None
    # 合成 MLP 中间单元（只用于检验采集格式与神经元选择性统计）：单元 0 为主因子首水平的 +×+ 检测器，
    # 单元 1 为同一水平的 −×− 区（双负）检测器，其余为噪声
    mplan = mlp_plan(corpus, n_hs - 1)
    mlp_store = MlpStore(mplan, int(corpus.get('synthetic_d_ff', 32))) if mplan else None
    first_level = None
    if mplan and main_f:
        first_level = sorted({str(it['cond'][main_f]) for it in items if main_f in it['cond']})[0]
    if cplan:
        n_heads, head_dim = int(corpus.get('synthetic_heads', 4)), int(corpus.get('synthetic_head_dim', 8))
        comp_store = ComponentStore(cplan, n_heads, head_dim, dim)
        W_syn = {b: np.random.default_rng(int(hashlib.sha256(f'{salt}W_O{b}'.encode()).hexdigest()[:8], 16))
                 .standard_normal((dim, n_heads * head_dim)) / np.sqrt(n_heads * head_dim) for b in cplan['blocks']}
    for it in items:
        rng = np.random.default_rng(int(hashlib.sha256((salt + it['id']).encode()).hexdigest()[:8], 16))
        cond = it['cond']
        vecs = []
        for r in readouts:
            for t, hi in plan.layers:
                a_, b_, c_, e_ = _synthetic_coef(t, hi, n_hs)
                if r == 'target':
                    v = a_ * unit('filler=' + str(it.get('filler_id', it['id'])))
                elif hi == 0 or t == 'emb':
                    v = a_ * unit(f'{r}-token=' + str(it.get('frame_id', '')))
                else:
                    v = 0.5 * a_ * unit('filler=' + str(it.get('filler_id', it['id']))) + 0.5 * unit('readout=' + r)
                if main_f and visible(r, main_f):
                    v = v + b_ * level_dir(main_f, cond[main_f])
                for f in other_f:
                    if visible(r, f):
                        v = v + c_ * level_dir(f, cond[f])
                if e_:
                    v = v + e_ * unit('cell=' + cond_key(cond))
                v = v + 0.04 * len(it['text']) * nuis + rng.standard_normal(dim) * (0.8 / np.sqrt(dim))
                vecs.append(v)
        acc.add(it['split'], cond, plan.ss_select(vecs))
        tok_len.setdefault(cond_key(cond), []).append(len(it['text']))
        if comp_store is not None:
            zs, ms = [], []
            for r in comp_store.readouts:
                base = readouts.index(r) * n_layers
                z = rng.standard_normal((len(comp_store.blocks), n_heads * head_dim)) * 0.3
                m = np.stack([vecs[base + b + 1] - vecs[base + b] - W_syn[b] @ z[k]
                              for k, b in enumerate(comp_store.blocks)])
                zs.append(z); ms.append(m)
            comp_store.add(it['id'], np.stack(zs), np.stack(ms))
        if mlp_store is not None:
            P_, B_, F_ = len(mlp_store.readouts), len(mlp_store.blocks), mlp_store.d_ff
            G = rng.standard_normal((P_, B_, F_)) * 0.5
            V = rng.standard_normal((P_, B_, F_)) * 0.5
            if first_level is not None and str(cond.get(main_f)) == first_level:
                G[:, :, 0] += 3.0; V[:, :, 0] += 3.0
                G[:, :, 1] -= 4.0; V[:, :, 1] -= 4.0
            mlp_store.add(it['id'], G, V)
        correct = None
        if it.get('probe'):
            p_ok = 0.55 if (main_f and other_f and str(cond[main_f]) == weak[0] and str(cond[other_f[0]]) == weak[1]) else 0.95
            correct = bool(rng.random() < p_ok)
            mode = 'forced_choice' if it['probe'].get('contrast') else 'set_first_token'
            behavior.append({'id': it['id'], 'split': it['split'], 'cell': cond_key(cond), 'mode': mode,
                             'correct': correct, 'rank': 1 if correct else 2, 'logprob': -0.1 if correct else -2.0,
                             'first_top1': correct, 'set_top1': correct,
                             **({'margin': 1.0 if correct else -1.0} if mode == 'forced_choice' else {})})
        depth = np.array([hi / max(n_hs - 1, 1) for _, hi in plan.layers], dtype=np.float32)
        P = len(readouts)
        lens = {'next_id': np.full((P,), 7, dtype=np.int32),
                'next_logprob': np.stack([-4.0 + 3.0 * depth + 0.1 * rng.standard_normal(n_layers) for _ in readouts]),
                'answer_id': 11 if it.get('probe') else -1,
                'answer_logprob': np.stack([(-6.0 + 5.5 * depth if it.get('probe') else np.full(n_layers, np.nan))
                                            for _ in readouts]),
                'topk_ids': np.tile(np.arange(LENS_TOPK, dtype=np.int32), (P, n_layers, 1)),
                'topk_logprob': np.tile(-np.arange(1, LENS_TOPK + 1, dtype=np.float32), (P, n_layers, 1))}
        if store is not None:
            store.add(it, vecs, len(it['text']), correct, lens)
    can = {c['id']: {t: summarize_vec(unit(c['id']) * (l + 1)) for l, (t, _) in enumerate(plan.layers)} for c in canaries}
    _write_outputs(out, acc, behavior, can, tok_len, summary, status='synthetic', extra={}, store=store,
                   lens_layers=[i for _, i in plan.layers], vocab={i: f'<syn{i}>' for i in range(LENS_TOPK)} | {7: '<next>', 11: '<ans>'},
                   native={'corpus': corpus, 'shard': _shard_tuple(summary),
                           'model_key': 'synthetic-' + hashlib.sha256(salt.encode('utf-8')).hexdigest()[:8]},
                   components=(comp_store, lambda: save_o_proj(
                       'synthetic-' + hashlib.sha256(salt.encode('utf-8')).hexdigest()[:8], W_syn))
                   if comp_store is not None and comp_store.ids else None,
                   mlp=mlp_store if mlp_store is not None and mlp_store.ids else None)


def _shard_tuple(summary):
    i, n = str(summary.get('shard', '0/1')).split('/')
    return int(i), int(n)


def corpus_seed(corpus):
    key = corpus.get('grammar_sha256') or json.dumps(corpus.get('items', [])[:3], sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:8], 16)


def _lens_digest(store, lens_layers):
    """逐读出、逐层的答案首 token 平均 logit-lens 对数概率（节点自检用）。"""
    import numpy as np
    rows = [x for x in store.lens if x is not None and x.get('answer_id', -1) >= 0]
    if not rows:
        return {}
    a = np.stack([x['answer_logprob'] for x in rows])          # (N, P, L)
    out = {}
    for p, r in enumerate(store.readouts):
        m = np.nanmean(a[:, p, :], axis=0)
        out[r] = [None if not np.isfinite(v) else round(float(v), 4) for v in m]
    return {'layers': list(lens_layers), 'answer_logprob_mean': out}


def _write_outputs(out, acc, behavior, can, tok_len, summary, status, extra, store=None, lens_layers=None, vocab=None,
                   native=None, components=None, mlp=None):
    (out / 'suffstats.npz').write_bytes(acc.npz_bytes())
    if store is not None and store.meta:
        data, space = store.npz_bytes(store.tags or acc.layer_tags, lens_layers, vocab)
        (out / 'items.npz').write_bytes(data)
        summary['item_space'] = space
        summary['item_space_rule'] = ('k 只由语料研究样本总数、标签数与模型维度决定（全语料 float16 预算 '
                                      f'{int(store.budget or ITEM_CORPUS_BUDGET_BYTES) >> 20} MB），投影矩阵由语料指纹决定，各分片相同')
    if store is not None and store.meta and native is not None:
        nd = native_dir(native['corpus'], native['shard'], native['model_key'])
        if nd is None:
            summary['native_field'] = {'disabled': True}
        else:
            try:
                summary['native_field'] = write_native(store, nd, {
                    'family_id': native['corpus'].get('family_id'), 'tm_id': native['corpus'].get('tm_id'),
                    'version': native['corpus'].get('version'), 'corpus_fingerprint': summary.get('corpus_fingerprint'),
                    'shard': summary.get('shard'), 'model_key': native['model_key'], 'status': status,
                    'precision': summary.get('precision'), 'dim': acc.dim,
                    'hidden_state_boundary': summary.get('hidden_state_boundary')})
            except OSError as exc:      # 本地磁盘问题不影响上传产物，如实记录
                summary['native_field'] = {'error': f'{type(exc).__name__}: {exc}'}
        if lens_layers is not None:
            summary['lens'] = {'topk': LENS_TOPK, **_lens_digest(store, lens_layers)}
    if components is not None:
        cstore, weights = components
        info = cstore.digest()
        nd = native_dir(native['corpus'], native['shard'], native['model_key']) if native is not None else None
        if nd is None:
            info['native'] = {'disabled': True, 'note': '原生目录关闭（AI2050_NATIVE_DIR=off），部件输出未保存'}
        else:
            try:
                info['native'] = cstore.write(nd / 'components', weights())
            except OSError as exc:
                info['native'] = {'error': f'{type(exc).__name__}: {exc}'}
        summary['components'] = info
    if mlp is not None:
        info = mlp.digest()
        nd = native_dir(native['corpus'], native['shard'], native['model_key']) if native is not None else None
        if nd is None:
            info['native'] = {'disabled': True, 'note': '原生目录关闭（AI2050_NATIVE_DIR=off），MLP 中间单元未保存'}
        else:
            try:
                info['native'] = mlp.write(nd / 'mlp')
            except OSError as exc:
                info['native'] = {'error': f'{type(exc).__name__}: {exc}'}
        summary['mlp_units'] = info
    (out / 'behavior.json').write_text(json.dumps({'schema': 'behavior.v1', 'items': behavior},
                                                  ensure_ascii=False), encoding='utf-8')
    (out / 'canary.json').write_text(json.dumps({'schema': 'canary.v1', 'entries': can},
                                                ensure_ascii=False), encoding='utf-8')
    by_cell = {}
    for b in behavior:
        rec = by_cell.setdefault(b['cell'], {'n': 0, 'correct': 0})
        rec['n'] += 1
        rec['correct'] += int(b['correct'])
    summary.update(extra)
    summary.update({
        'status': status,
        'dim': acc.dim,
        'layer_tags': acc.layer_tags,
        'behavior_by_cell': by_cell,
        'token_len_by_cell': {k: {'min': min(v), 'max': max(v), 'mean': round(sum(v) / len(v), 2)}
                              for k, v in sorted(tok_len.items())},
        'eta2_shard_digest': acc.eta2_digest(),
        'canary_ids': sorted(can),
    })


if __name__ == '__main__':
    raise SystemExit(main())
