# -*- coding: utf-8 -*-
"""Phase 15 诊断探针：SUP_ID 是 qwen 词表硬编码；检查三模型 tokenizer 的实际 token id。

若 glm4-9b 的 `水果` 等 token id != seal 的 sup_id，则 A1 臂的 score_of 用了**错误的类别 token**
⇒ base 分数近零、FULL_SWAP 触地、剖面平坦 —— 是**装置移植缺陷**，不是模型的机制差异。
只读词表，不加载权重。
"""
import os, io, json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf')
SUPS = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
DISC = ['苹果', '狗', '汽车', '桌子', '铁', '红']
SEAL = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15',
                                      'N2h1a8_design_seal.json'), encoding='utf-8'))
SID = SEAL['panel']['sup_id'] if 'sup_id' in SEAL.get('panel', {}) else None
EX = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15',
                                    'execution_phase15.json'), encoding='utf-8'))
SID = EX['sup_id']

o = []
def w(s=''):
    o.append(str(s)); print(s)

w('seal/exec 硬编码 sup_id = %s' % json.dumps(SID, ensure_ascii=False))
w('')

from transformers import AutoTokenizer
for name in ['qwen3-4b', 'glm4-9b-chat-hf', 'Qwen3-14B']:
    p = os.path.join(MDIR, name)
    w('=' * 78)
    w('MODEL %s  (%s)' % (name, p))
    try:
        tk = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    except Exception as e:
        w('  !! tokenizer load failed: %r' % (e,)); continue
    w('  tokenizer class = %s ; vocab_size = %s' % (type(tk).__name__, tk.vocab_size))
    for s in SUPS:
        ids = tk(s, add_special_tokens=False)['input_ids']
        w('    %-10s -> ids=%s  decode=%s  exec_sup_id=%s  MATCH=%s'
          % (s, ids, [tk.decode([i]) for i in ids], SID.get(s), ids == [SID.get(s)]))
    w('  --- 实例词（单 token 检查） ---')
    for s in DISC:
        ids = tk(s, add_special_tokens=False)['input_ids']
        w('    %-10s -> n=%d ids=%s' % (s, len(ids), ids))
    # 模板 T=2 检查
    for s in DISC:
        ids = tk('%s是一种' % s, add_special_tokens=False)['input_ids']
        w('    TMPL %-8s -> n=%d' % (s, len(ids)))
    w('')

OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15', '_probe_supid.txt')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('WROTE', OUT)
