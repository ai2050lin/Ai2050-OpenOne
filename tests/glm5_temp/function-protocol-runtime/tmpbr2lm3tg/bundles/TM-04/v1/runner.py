# -*- coding: utf-8 -*-
"""AI2050 模板通用 runner —— 多条件分解采集（M1/v2 版）

由中心节点随模板包下发，在 Worker 节点执行:
    python runner.py [--mode auto|smoke|real] [--out outputs]

真实模式: 环境变量 AI2050_RUNNER_MODEL 指向 HF 模型目录且 torch/transformers 可用时启用。
    采集残差流目标 token 向量（层位由 corpus.layers 指定，缺省 ['last']），
    逐条件求均值 → 逐维 η² → 条件方向 cos 矩阵 → outputs/means.npz。

两种分析模式（corpus.analysis）:
    oneway     单因子（TM-01..03）：按 cond_key 分组做单因素方差分解，行为与 v1 完全一致。
    factorial  双因子网格（TM-04/05）：balanced 无重复网格的 two-way 分解，
               逐维 η² 拆成 η²(A) + η²(B) + η²(interaction)（三者相加=1）；
               每个请求层位各分解一次（per_layer）；last 层另算因子水平方向 cos；
               corpus.fingerprint 存在时计算反词嵌入指纹（类内等角性 spread /
               类外分离度 sep_out）——特征 ID 依据指纹匹配而非激活峰值
               （2026-10-07 协议：句式=因子水平，不变量=指纹，干预=因果升格）。
    heads      head 级稳定性（TM-06）：采集全层注意力，对目标词位置统计
               每 (层,头) 的注意力份额 attn_target 与前缀吸引度 attn_pull；
               条件内排名 top-K head 集合 → 跨条件 Jaccard（head 集合指纹，
               尺度无关）；+ last 层残差流条件方向 cos / η²（语义读出稳定性）。

冒烟模式（默认/无 GPU）: 语料指纹 + 条件计数（status=smoke），仅用于管线联调，
    不冒充实测——诚实口径与项目 63 条装置纪律一致（SMOKE 必做必看、smoke 不算 real 结果）。

产出（Agent 会上传 outputs/ 下全部文件）:
    outputs/summary.json   必有
    outputs/means.npz      仅真实模式
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path


def cond_key(cond: dict) -> str:
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


def resolve_layers(spec, n_hs: int):
    """把层位别名解析成 hidden_states 绝对索引。first=1（第一层输出），mid=len//2，last=末层。"""
    out = []
    for tag in spec:
        if tag == 'first':
            out.append((tag, 1))
        elif tag == 'mid':
            out.append((tag, n_hs // 2))
        elif tag == 'last':
            out.append((tag, n_hs - 1))
        else:
            raise ValueError(f'unknown layer tag: {tag}')
    return out


def factorial_eta2(allv, cells, fa, fb, dim):
    """balanced 无重复双因子逐维分解。
    allv: (N, dim)；cells: 长度 N 的 (a,b) 元组列表。
    返回 eta2 (3, dim)（A 主效应 / B 主效应 / 交互=残余）与 grand (dim,)。
    """
    import numpy as np
    grand = allv.mean(axis=0)
    ss_total = ((allv - grand) ** 2).sum(axis=0)
    ss_total = np.maximum(ss_total, 1e-9)

    a_vals = sorted({c[0] for c in cells})
    b_vals = sorted({c[1] for c in cells})

    ss_a = np.zeros(dim)
    for a in a_vals:
        rows = [i for i, c in enumerate(cells) if c[0] == a]
        m = allv[rows].mean(axis=0)
        ss_a += len(rows) * (m - grand) ** 2
    ss_b = np.zeros(dim)
    for b in b_vals:
        rows = [i for i, c in enumerate(cells) if c[1] == b]
        m = allv[rows].mean(axis=0)
        ss_b += len(rows) * (m - grand) ** 2
    # 无重复网格：交互 = 饱和模型残余（SS_total − A − B），非负截断防数值噪声
    ss_int = np.maximum(ss_total - ss_a - ss_b, 0.0)
    eta2 = np.stack([ss_a / ss_total, ss_b / ss_total, ss_int / ss_total])
    return eta2, grand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='auto', choices=['auto', 'smoke', 'real'])
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    corpus = json.loads((here / 'corpus.json').read_text(encoding='utf-8'))
    items = corpus['items']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fp = hashlib.sha256(
        json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()[:16]

    groups: dict = {}
    for it in items:
        groups.setdefault(cond_key(it['cond']), []).append(it)

    analysis = corpus.get('analysis', 'oneway')
    layer_spec = corpus.get('layers', ['last'])
    summary = {
        'tm_id': corpus.get('tm_id'),
        'analysis': analysis,
        'layers': list(layer_spec),
        'corpus_fingerprint': fp,
        'n_items': len(items),
        'conditions': {k: len(v) for k, v in sorted(groups.items())},
        'python': platform.python_version(),
        'status': 'smoke',
    }
    if analysis == 'factorial':
        summary['factors'] = list(corpus.get('factors', []))

    want_real = args.mode == 'real' or (
        args.mode == 'auto' and os.environ.get('AI2050_RUNNER_MODEL')
    )

    if want_real:
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = os.environ['AI2050_RUNNER_MODEL']
            tok = AutoTokenizer.from_pretrained(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # heads 模式需要逐层注意力：SDPA 不返回 attentions，须强制 eager 实现
            want_attn = analysis == 'heads'
            kw = {'attn_implementation': 'eager'} if want_attn else {}
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, **kw,
            )
            model.eval()

            # ---- 采集：每 item 一次前向，按请求层位切片；heads 模式另采注意力 ----
            vecs: dict = {}       # layer_tag -> cond_key -> [vec]
            att_stats: dict = {}  # cond_key -> [(attn_target (L,H), attn_pull (L,H))]
            with torch.no_grad():
                for it in items:
                    text, target = it['text'], it['target']
                    enc = tok(text, return_tensors='pt', return_offsets_mapping=True)
                    off = enc.pop('offset_mapping')[0].tolist()
                    s = text.find(target)
                    if s < 0:
                        raise ValueError(f"target {target!r} not in text {text!r}")
                    e = s + len(target)
                    pos = max(i for i, (cs, ce) in enumerate(off) if cs < e and ce > s)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    outp = model(**enc, output_hidden_states=True, output_attentions=want_attn)
                    hs = outp.hidden_states
                    for tag, idx in resolve_layers(layer_spec, len(hs)):
                        h = hs[idx][0, pos, :].float().cpu().numpy()
                        vecs.setdefault(tag, {}).setdefault(cond_key(it['cond']), []).append(h)
                    if want_attn:
                        # attentions 是逐层 tuple：stack 成 (1,L,H,q,k) 再取 batch 0
                        att = torch.stack(outp.attentions, dim=1)[0].float()   # (L,H,q,k)
                        at = att[:, :, :, pos].mean(dim=-1)        # (L,H) 全句 → target 份额
                        pk = sum(1 for (cs, ce) in off if ce <= int(it.get('prefix_chars', 0)))
                        if pk > 0:
                            ap = att[:, :, pos, :pk].sum(dim=-1)   # (L,H) target → 前缀区段
                        else:
                            ap = torch.zeros_like(at)
                        att_stats.setdefault(cond_key(it['cond']), []).append(
                            (at.cpu().numpy(), ap.cpu().numpy()))

            last_tag = layer_spec[-1]
            keys = sorted(vecs[last_tag])

            if analysis == 'factorial':
                fa, fb = corpus['factors']
                cells = []
                for it in items:
                    cells.append((str(it['cond'][fa]), str(it['cond'][fb])))

                # ---- 逐层分解：η²(A)/η²(B)/η²(interaction) ----
                per_layer = {}
                eta2_last, grand = None, None
                for tag in layer_spec:
                    allv = np.stack([v for k in keys for v in vecs[tag][k]])
                    eta2, grand = factorial_eta2(allv, cells, fa, fb, allv.shape[1])
                    per_layer[tag] = {
                        fa: float(eta2[0].mean()), fb: float(eta2[1].mean()),
                        'interaction': float(eta2[2].mean()),
                        'dim_argmax_' + fa: int(eta2[0].argmax()),
                        'dim_argmax_' + fb: int(eta2[1].argmax()),
                    }
                    if tag == last_tag:
                        eta2_last = eta2

                # ---- last 层 cell 方向 cos 矩阵 ----
                means = np.stack([np.mean(np.stack(vecs[last_tag][k]), axis=0) for k in keys])
                C = means - grand
                Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
                cosm = Cn @ Cn.T
                offdiag = cosm - np.eye(len(keys))

                np.savez_compressed(
                    out / 'means.npz',
                    keys=np.array(keys), means=means,
                    eta2_factor_dim=eta2_last, cos=cosm,
                    layer_tags=np.array(layer_spec),
                )

                summary.update({
                    'status': 'real',
                    'model_path': model_path,
                    'device': device,
                    'dim': int(means.shape[1]),
                    'eta2_by_factor': {fa: per_layer[last_tag][fa],
                                       fb: per_layer[last_tag][fb],
                                       'interaction': per_layer[last_tag]['interaction']},
                    'per_layer': per_layer,
                    'cos_absmax_offdiag': float(np.abs(offdiag).max()),
                })

                # ---- 反词嵌入指纹（last 层，corpus.fingerprint 声明类内/类外参照词）----
                fp_cfg = corpus.get('fingerprint')
                if fp_cfg:
                    unembed = model.get_output_embeddings().weight.detach().float().cpu()
                    un_n = torch.nn.functional.normalize(unembed, dim=1)

                    def word_row(w):
                        ids = tok(w)['input_ids']
                        return un_n[ids[-1]]

                    fp_out = {}
                    for lvl_key, words in fp_cfg.items():
                        # 方向 = 该因子水平的均值方向（last 层，归一化）
                        lvl_rows = [i for i, c in enumerate(cells) if c[0] == lvl_key]
                        v = means[lvl_rows].mean(axis=0) - grand
                        vn = torch.tensor(v / max(float(np.linalg.norm(v)), 1e-9))
                        within = torch.stack([word_row(w) for w in words.get('within', [])])
                        outs = torch.stack([word_row(w) for w in words.get('out', [])]) \
                            if words.get('out') else None
                        cw = (within @ vn).abs()
                        entry = {
                            'spread_within': float((cw.max() - cw.min()).item()),
                            'n_within': int(cw.numel()),
                        }
                        if outs is not None:
                            co = (outs @ vn).abs()
                            entry['sep_out'] = float(co.max().item())
                        fp_out[lvl_key] = entry
                    summary['fingerprint'] = fp_out
            elif analysis == 'heads':
                # ---- head 级（TM-06）：残差流统计与 oneway 同口径，另加 head 集合指纹 ----
                means = np.stack([np.mean(np.stack(vecs[last_tag][k]), axis=0) for k in keys])
                allv = np.stack([v for k in keys for v in vecs[last_tag][k]])
                grand = allv.mean(axis=0)
                ss_total = ((allv - grand) ** 2).sum(axis=0)
                ss_total = np.maximum(ss_total, 1e-9)
                ss_between = np.zeros_like(ss_total)
                eta2_by_cond = {}
                for k in keys:
                    arr = np.stack(vecs[last_tag][k])
                    delta2 = (arr.mean(axis=0) - grand) ** 2
                    ss_between += arr.shape[0] * delta2
                    eta2_by_cond[k] = float((delta2 / ss_total).mean())
                eta2_dim = ss_between / ss_total
                C = means - grand
                Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
                cosm = Cn @ Cn.T
                offdiag = cosm - np.eye(len(keys))
                np.savez_compressed(
                    out / 'means.npz',
                    keys=np.array(keys), means=means, eta2_dim=eta2_dim, cos=cosm,
                )

                # head 集合指纹：条件内按 attn_target 排名取 top-K，跨条件 Jaccard（尺度无关）
                share, pull = {}, {}
                for k, lst in att_stats.items():
                    share[k] = np.stack([s[0] for s in lst]).mean(axis=0).reshape(-1)
                    pull[k] = np.stack([s[1] for s in lst]).mean(axis=0).reshape(-1)
                topk = int(corpus.get('topk', 32))

                def top_set(v):
                    return {int(i) for i in np.argsort(v)[::-1][:topk]}

                ks = sorted(share)
                jacc = {}
                for i in range(len(ks)):
                    for j in range(i + 1, len(ks)):
                        a, b = top_set(share[ks[i]]), top_set(share[ks[j]])
                        jacc[f'{ks[i]}~{ks[j]}'] = float(len(a & b) / max(len(a | b), 1))
                top_show = {k: [[int(i), round(float(share[k][i]), 5)]
                                for i in np.argsort(share[k])[::-1][:10]] for k in ks}
                lh = att_stats[ks[0]][0][0].shape  # (L, H)
                summary.update({
                    'status': 'real',
                    'model_path': model_path,
                    'device': device,
                    'dim': int(means.shape[1]),
                    'cond_factor': corpus.get('cond_factor', 'prefix'),
                    'n_layer_head': [int(lh[0]), int(lh[1])],
                    'topk': topk,
                    'topk_jaccard': jacc,
                    'head_share_top': top_show,
                    'prefix_pull_mean': {k: float(pull[k].mean()) for k in ks},
                    'prefix_pull_max': {k: [int(pull[k].argmax()), float(pull[k].max())] for k in ks},
                    'eta2_mean_overall': float(eta2_dim.mean()),
                    'eta2_by_cond': eta2_by_cond,
                    'cos_absmax_offdiag': float(np.abs(offdiag).max()),
                })
            else:
                # ---- oneway：与 v1 逐字段一致 ----
                means = np.stack([np.mean(np.stack(vecs[last_tag][k]), axis=0) for k in keys])
                allv = np.stack([v for k in keys for v in vecs[last_tag][k]])
                grand = allv.mean(axis=0)
                ss_total = ((allv - grand) ** 2).sum(axis=0)
                ss_total = np.maximum(ss_total, 1e-9)
                ss_between = np.zeros_like(ss_total)
                eta2_by_cond = {}
                for k in keys:
                    arr = np.stack(vecs[last_tag][k])
                    delta2 = (arr.mean(axis=0) - grand) ** 2
                    ss_between += arr.shape[0] * delta2
                    eta2_by_cond[k] = float((delta2 / ss_total).mean())
                eta2_dim = ss_between / ss_total

                C = means - grand
                Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
                cosm = Cn @ Cn.T
                offdiag = cosm - np.eye(len(keys))

                np.savez_compressed(
                    out / 'means.npz',
                    keys=np.array(keys), means=means, eta2_dim=eta2_dim, cos=cosm,
                )
                summary.update({
                    'status': 'real',
                    'model_path': model_path,
                    'device': device,
                    'dim': int(means.shape[1]),
                    'eta2_mean_overall': float(eta2_dim.mean()),
                    'eta2_by_cond': eta2_by_cond,
                    'cos_absmax_offdiag': float(np.abs(offdiag).max()),
                })
        except Exception as exc:  # 真实模式失败 → 诚实降级为 smoke，并记录原因
            summary['status'] = 'smoke'
            summary['real_error'] = f'{type(exc).__name__}: {exc}'

    (out / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(f"[runner] status={summary['status']} items={len(items)} out={out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
