# -*- coding: utf-8 -*-
import os, sys, time, json, hashlib
root = r'D:\AI2050\Ai2050-OpenOne'
out = []

out.append('now %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# ---- GPU ----
try:
    import torch
    out.append('torch %s cuda %s' % (torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out.append('gpu free %.1f MiB / total %.1f MiB' % (free/2**20, total/2**20))
except Exception as e:
    out.append('torch probe fail %r' % e)

# ---- processes ----
try:
    import psutil
    hits = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cl = ' '.join(p.info.get('cmdline') or [])
            if 'phase31' in cl or 'n1_' in cl or 'n2_' in cl or 'e3' in cl:
                hits.append('%d %s | %s' % (p.info['pid'], p.info['name'], cl[:160]))
        except Exception:
            pass
    out.append('busy_procs %d' % len(hits))
    for h in hits[:10]:
        out.append('  ' + h)
except Exception as e:
    out.append('psutil fail %r' % e)

# ---- target memo ----
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
out.append('memo exists %s' % os.path.exists(memo))
if os.path.exists(memo):
    b = open(memo, 'rb').read()
    out.append('memo bytes %d sha8 %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
    out.append('memo has_BOM %s' % b.startswith(b'\xef\xbb\xbf'))
    out.append('memo CRLF %d LF_total %d' % (b.count(b'\r\n'), b.count(b'\n')))
    T = b.decode('utf-8-sig', errors='replace')
    L = T.splitlines()
    out.append('memo lines %d' % len(L))
    out.append('memo last_line %r' % L[-1][:120])
    out.append('--- tail 12 ---')
    for l in L[-12:]:
        out.append('  ' + l[:150])

# ---- models ----
h = os.path.join(root, 'models', 'hf')
out.append('model dirs: ' + ', '.join(sorted([d for d in os.listdir(h) if os.path.isdir(os.path.join(h, d))])))

# ---- key configs ----
for m in ['qwen3-4b', 'qwen2.5-3b-instruct', 'glm4-9b-chat-hf', 'qwen3-1.7b']:
    p = os.path.join(h, m, 'config.json')
    if os.path.exists(p):
        J = json.load(open(p, encoding='utf-8'))
        tc = J.get('text_config', {})
        g = lambda k: J.get(k, tc.get(k))
        out.append('cfg %-24s L=%s H=%s heads=%s kv=%s hdim=%s tie=%s arch=%s' % (
            m, g('num_hidden_layers'), g('hidden_size'), g('num_attention_heads'),
            g('num_key_value_heads'), g('head_dim'), J.get('tie_word_embeddings'), J.get('architectures')))

# ---- n2 script present? ----
for f in ['n1c_ontology_cloze.py', 'n1b_ontology_readout.py', 'n1_v2_main_axis_scan.py']:
    p = os.path.join(root, 'tests', 'gpt5_temp', f)
    out.append('script %s %s' % (f, os.path.exists(p)))

dst = os.path.join(root, 'tests', 'gpt5_temp', 'probe_n2.txt')
open(dst, 'w', encoding='utf-8').write('\n'.join(out))
print('written', dst)
