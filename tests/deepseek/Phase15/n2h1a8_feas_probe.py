# -*- coding: utf-8 -*-
"""
Phase 15 (N2h1-alpha-8) 可行性探针
=================================
目标：在 qwen3-14b 与 glm4-9b 上复算「单点位点替换族 xhalf(ell)/J(ell) 双坐标剖面 + 置换零假设校准」。
本探针只做**装置可行性**，不做任何研究判决：
  A. 加载可行性（bf16 + device_map=auto，GPU 16GB 装不下 29.5/18.8GB）与分层放置统计
  B. 结构路径探测（layers / final norm / attn out_proj 模块名）
  C. T=2 token 布局核验（41 实例在 '%s是一种' 下是否都 tokenize 成 2 token）
  D. 前向计时（3 次）+ 双前向 bit 确定性
  E. 干预 hook 冒烟：在某层末位写入 0.1*随机向量，验证 (i) 不报错 (ii) 设备/dtype 一致 (iii) 与无干预输出可区分
  F. 依据 D 估算 Phase 15 正式运行的前向预算与墙钟
产物：tests/deepseek_temp/Phase15/feas_probe_phase15.txt
"""
import os
import sys
import io
import json
import time
import hashlib
import traceback

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
if not os.path.isdir(PT):
    os.makedirs(PT)
OUT = os.path.join(PT, 'feas_probe_phase15.txt')

_L = []


def w(s=''):
    _L.append(str(s))
    print(s)
    sys.stdout.flush()


def flush_report():
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_L) + '\n')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------------- 面板（与 Phase 12 同一 discovery 面板，用于 T=2 布局核验） ----------------
DISCOVERY = [['苹果', '水果'], ['香蕉', '水果'], ['梨', '水果'], ['西瓜', '水果'],
             ['狗', '动物'], ['猫', '动物'], ['老虎', '动物'], ['大象', '动物'],
             ['汽车', '交通工具'], ['火车', '交通工具'], ['飞机', '交通工具'], ['摩托车', '交通工具'],
             ['桌子', '家具'], ['椅子', '家具'], ['床', '家具'], ['沙发', '家具'],
             ['铁', '金属'], ['铜', '金属'], ['铝', '金属'], ['金', '金属'],
             ['红', '颜色'], ['蓝', '颜色'], ['绿', '颜色'], ['黄', '颜色']]
INSTANCES = sorted(set([p[0] for p in DISCOVERY] + [p[1] for p in DISCOVERY]))
TMPL = '%s是一种'

MODELS = [('Qwen3-14B', 'qwen3-14b'), ('glm4-9b-chat-hf', 'glm4-9b')]


def find_final_norm(core, model):
    for obj, pre in ((core, 'core'), (model.model, 'model'), (model, 'top')):
        if obj is None:
            continue
        for nm in ['norm', 'final_layernorm', 'ln_f', 'transformer.ln_f']:
            if '.' in nm:
                a, b = nm.split('.')
                if hasattr(obj, a) and hasattr(getattr(obj, a), b):
                    return getattr(getattr(obj, a), b), '%s.%s' % (pre, nm)
            elif hasattr(obj, nm):
                return getattr(obj, nm), '%s.%s' % (pre, nm)
    return None, None


def find_layers(model):
    for pre in ('model.language_model', 'model', 'transformer'):
        obj = model
        ok = True
        for part in pre.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, 'layers'):
            return obj.layers, pre + '.layers'
    return None, None


def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm):
            return getattr(a, nm), nm
    return None, None


def run_model(dirname, tag):
    MDIR = os.path.join(ROOT, 'models', 'hf', dirname)
    w('=' * 78)
    w('MODEL %s  (dir=%s)' % (tag, dirname))
    w('  config_sha256_8 = %s ; ckpt files = %d' %
      (sha8(os.path.join(MDIR, 'config.json')),
       len([f for f in os.listdir(MDIR) if f.endswith('.safetensors')])))
    rec = {'tag': tag, 'dir': dirname}

    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    w('  tokenizer loaded %.1fs' % (time.time() - t0))
    # ---- C. T=2 布局核验
    tl = {}
    for inst in INSTANCES:
        n = len(tok.encode(TMPL % inst, add_special_tokens=False))
        tl.setdefault(n, []).append(inst)
    w('  [C] token-length histogram of TMPL%%s under %r : %s' %
      (TMPL, {k: len(v) for k, v in sorted(tl.items())}))
    if list(sorted(tl.keys())) != [2]:
        w('      !! 非均匀 T=2 的实例：%s' %
          {k: v[:6] for k, v in sorted(tl.items()) if k != 2})
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['non_T2_instances'] = {str(k): v for k, v in sorted(tl.items()) if k != 2}

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='eager', device_map='auto', low_cpu_mem_usage=True)
    model.eval()
    w('  [A] model loaded %.1fs ; device_map spread = %d distinct devices' %
      (time.time() - t0, len(set(str(v) for v in model.hf_device_map.values()))))
    dev_count = {}
    for k, v in model.hf_device_map.items():
        dev_count[str(v)] = dev_count.get(str(v), 0) + 1
    w('      device_map summary: %s' % dev_count)
    # GPU / RAM 状态
    try:
        free, total = torch.cuda.mem_get_info(0)
        w('      GPU free/total MB = %.0f/%.0f' % (free / 1e6, total / 1e6))
    except Exception as e:
        w('      GPU mem query failed: %r' % e)
    rec['hf_device_map_summary'] = dev_count

    core = getattr(model.model, 'language_model', model.model)
    layers, lpath = find_layers(model)
    if layers is None:
        # fallback: 手动
        raise RuntimeError('cannot find .layers')
    L = len(layers)
    NORM, npath = find_final_norm(core, model)
    HID = model.config.hidden_size
    w('  [B] layers path = %s ; L = %d ; hidden = %d' % (lpath, L, HID))
    w('      final norm path = %s (%s)' % (npath, type(NORM).__name__ if NORM is not None else 'NONE'))
    oproj, onm = attn_out_proj(layers[0])
    oproj_mid, _ = attn_out_proj(layers[L // 2])
    NH = int(getattr(model.config, 'num_attention_heads'))
    KVH = int(getattr(model.config, 'num_key_value_heads', -1))
    HDP = int(getattr(model.config, 'head_dim', HID // max(NH, 1)))
    w('      attn out_proj name=%r in_features=%s (mid layer=%s)' %
      (onm, (oproj.in_features if oproj is not None else None),
       (oproj_mid.in_features if oproj_mid is not None else None)))
    w('      n_heads=%d kv_heads=%d head_dim=%d ; o_proj_in == n_heads*head_dim ? %s' %
      (NH, KVH, HDP,
       (oproj.in_features == NH * HDP) if oproj is not None else 'NA'))
    w('      tie_word_embeddings = %s ; vocab = %d ; rms_eps = %s ; rope_theta = %s' %
      (getattr(model.config, 'tie_word_embeddings', None), model.config.vocab_size,
       getattr(model.config, 'rms_norm_eps', None), getattr(model.config, 'rope_theta', None)))
    rec['L'] = L
    rec['hidden'] = HID
    rec['n_heads'] = NH
    rec['kv_heads'] = KVH
    rec['head_dim'] = HDP
    rec['o_proj_in'] = (oproj.in_features if oproj is not None else None)
    rec['tie'] = bool(getattr(model.config, 'tie_word_embeddings', False))
    rec['layers_path'] = lpath
    rec['norm_path'] = npath

    # 层放置
    lay_dev = {}
    for i, ly in enumerate(layers):
        d = str(next(ly.parameters()).device)
        lay_dev.setdefault(d, []).append(i)
    w('      layer placement: %s' %
      {k: ('L%d..L%d (n=%d)' % (v[0], v[-1], len(v))) for k, v in lay_dev.items()})
    rec['layer_placement'] = {k: [v[0], v[-1], len(v)] for k, v in lay_dev.items()}

    # ---- D. 前向计时 + 双前向确定性
    ii = torch.tensor([tok.encode(TMPL % '苹果', add_special_tokens=False)])
    dev_in = model.hf_device_map.get('model.embed_tokens', model.hf_device_map.get('transformer.embedding', 'cuda:0'))
    w('      input device per device_map = %s' % dev_in)
    with torch.no_grad():
        o = model(input_ids=ii)
        lg1 = o.logits[0, -1].float().detach().cpu().numpy()
        ts = []
        for _ in range(3):
            t1 = time.time()
            o = model(input_ids=ii)
            lg = o.logits[0, -1].float().detach().cpu().numpy()
            ts.append(time.time() - t1)
        lg2 = lg
    w('  [D] single-forward latencies (s) = %s ; median = %.3f' %
      (['%.3f' % x for x in ts], float(np.median(ts))))
    w('      determinism max|logits diff| between two forwards = %.3e' %
      float(np.max(np.abs(lg1 - lg2))))
    rec['fw_latency_s'] = ts
    rec['fw_median_s'] = float(np.median(ts))
    rec['determinism_maxdiff'] = float(np.max(np.abs(lg1 - lg2)))

    # ---- E. 干预 hook 冒烟
    ok_hook = False
    hook_err = None
    try:
        rng = np.random.default_rng(20261002)
        vec = rng.standard_normal(HID).astype(np.float32) * 0.1
        box = {}

        def hk(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.clone()
            h[0, -1, :] = torch.tensor(vec, dtype=h.dtype, device=h.device)
            box['dev'] = str(h.device)
            box['dtype'] = str(h.dtype)
            return h if not isinstance(out, tuple) else (h,) + tuple(out[1:])

        idx = min(L // 2, L - 1)
        h = layers[idx].register_forward_hook(hk)
        try:
            with torch.no_grad():
                o2 = model(input_ids=ii)
        finally:
            h.remove()
        lg3 = o2.logits[0, -1].float().detach().cpu().numpy()
        d = float(np.max(np.abs(lg3 - lg2)))
        w('  [E] intervene hook @L%d OK ; wrote device=%s dtype=%s ; max|logits diff| vs clean = %.3e' %
          (idx, box.get('dev'), box.get('dtype'), d))
        ok_hook = True
        rec['hook_dev'] = box.get('dev')
        rec['hook_dtype'] = box.get('dtype')
        rec['hook_effect_maxdiff'] = d
    except Exception as e:
        hook_err = traceback.format_exc()
        w('  [E] !! hook smoke FAILED: %r' % e)
        w(hook_err)
    rec['hook_ok'] = ok_hook
    rec['hook_error'] = hook_err

    # ---- F. 预算估算
    lat = float(np.median(ts))
    # 设计草案：12 sites x 9 alphas x 24 pairs = 2592 fw（正式臂）
    for ns, na, npair in [(12, 9, 24), (9, 9, 24), (18, 14, 24), (12, 5, 24)]:
        nfw = ns * na * npair
        w('  [F] budget: sites=%d alphas=%d pairs=%d -> %d fw -> %.1f min @%.2fs/fw' %
          (ns, na, npair, nfw, nfw * lat / 60.0, lat))
    rec['budget_rows'] = []
    for ns, na, npair in [(12, 9, 24), (9, 9, 24), (18, 14, 24), (12, 5, 24)]:
        rec['budget_rows'].append({'sites': ns, 'alphas': na, 'pairs': npair,
                                   'fw': ns * na * npair, 'min': ns * na * npair * lat / 60.0})

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    import gc
    gc.collect()
    w('  (model unloaded)')
    flush_report()
    return rec


def main():
    w('Phase 15 feas probe ; clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    w('torch %s ; cuda %s ; dev %s' %
      (torch.__version__, torch.cuda.is_available(),
       torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'))
    only = None
    if len(sys.argv) > 1:
        only = sys.argv[1]
    recs = []
    for dirname, tag in MODELS:
        if only and only not in dirname and only not in tag:
            continue
        try:
            recs.append(run_model(dirname, tag))
        except Exception:
            w('!! MODEL %s FAILED' % tag)
            w(traceback.format_exc())
            flush_report()
    flush_report()
    io.open(os.path.join(PT, 'feas_probe_phase15.json'), 'w', encoding='utf-8', newline='\n').write(
        json.dumps(recs, ensure_ascii=False, indent=1))
    w('REPORT -> %s' % OUT)


if __name__ == '__main__':
    main()
