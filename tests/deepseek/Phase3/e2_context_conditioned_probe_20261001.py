# -*- coding: utf-8 -*-
"""
E2 条件化比较探针（qwen3-4b）
目的：检验"词嵌入做概念比较"的可行性到底建立在哪个切片上。
设计：
  - 层扫描 l=0..NL：l=0 即 embedding 行（静态）
  - 关键量：同一 token 在不同语境下的层状态 cos（意义分离曲线）
  - 对照组：token 位于句首（因果遮罩 => 无前文）时的 cos 曲线，应恒 ~1.0
产出：tests/gpt5_temp/e2_report.txt
零依赖除 transformers/torch；不与任何在跑任务冲突（先查显存）。
"""
import os, sys, time, traceback
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'e2_report.txt')

L = []
def log(s=''):
    L.append(str(s))
    try:
        print(s, flush=True)
    except Exception:
        pass

def dump():
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

def find_all(hay, needle):
    n, m = len(hay), len(needle)
    if m == 0 or n < m:
        return []
    return [i for i in range(n - m + 1) if hay[i:i+m] == needle]

def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    na = a.norm(); nb = b.norm()
    if na == 0 or nb == 0:
        return float('nan')
    return float((a @ b) / (na * nb))

def main():
    t0 = time.time()
    log('=== E2 条件化比较探针（qwen3-4b, 层扫描）===')
    log('时间 %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log('模型 %s' % MODEL)

    if not torch.cuda.is_available():
        log('CUDA 不可用 —— 中止'); dump(); return
    free, total = torch.cuda.mem_get_info()
    log('GPU %s | free %.1f GB / total %.1f GB' % (torch.cuda.get_device_name(0), free / 2**30, total / 2**30))
    if free < 10 * 2**30:
        log('显存 < 10GB —— 中止（避免与在跑任务冲突，防 OOM）'); dump(); return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)

    model = None
    for kw in ({'dtype': torch.bfloat16}, {'torch_dtype': torch.bfloat16}, {}):
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
            log('loaded with kwargs %s' % (list(kw.keys()) or 'default'))
            break
        except Exception as e:
            log('load attempt %s failed: %r' % (list(kw.keys()), e))
    if model is None:
        log('模型加载失败 —— 中止'); dump(); return

    model = model.to('cuda').eval()
    NL = model.config.num_hidden_layers
    HID = model.config.hidden_size
    log('num_hidden_layers %d  hidden_size %d' % (NL, HID))

    def tid(s):
        return tok.encode(s, add_special_tokens=False)

    APPLE = tid('苹果'); BAN = tid('香蕉'); FRUIT = tid('水果'); COMP = tid('公司')
    log('token ids  苹果=%s(len%d)  香蕉=%s(len%d)  水果=%s(len%d)  公司=%s(len%d)'
        % (APPLE, len(APPLE), BAN, len(BAN), FRUIT, len(FRUIT), COMP, len(COMP)))

    # ---- 句子集 ----
    # 语境前置型（目标词在后，能被前文语境影响）
    SEN = {
        'F1': '这种水果非常甜，苹果就是一种水果。',   # 苹果=水果义
        'C1': '这家科技公司很有名，苹果就是一家公司。', # 苹果=公司义
        'F2': '这种水果非常甜，香蕉就是一种水果。',   # 同框架对照（香蕉）
    }
    # 目标词在句首（因果遮罩：任何层都看不到后文）
    SHORT = {
        'S1': '苹果是水果。',
        'S2': '苹果是公司。',
    }

    def capture(text):
        enc = tok(text, return_tensors='pt')
        ids = enc['input_ids'][0].tolist()
        with torch.no_grad():
            out = model(**{k: v.to('cuda') for k, v in enc.items()}, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len = NL+1
        st = [h[0].detach().to('cpu') for h in hs]  # each [T, HID]
        return ids, st

    C = {}
    for k, t in list(SEN.items()) + list(SHORT.items()):
        ids, st = capture(t)
        C[k] = {'ids': ids, 'st': st, 'text': t}
        log('  %s len=%d  ids=%s' % (k, len(ids), ids))

    def pos_last(key, needle):
        ids = C[key]['ids']
        occ = find_all(ids, needle)
        if not occ:
            return None
        return occ[-1] + len(needle) - 1  # 用该词最后一个 subtoken 的位置

    P = {}
    for k in C:
        P[(k, 'apple')] = pos_last(k, APPLE)
    for k in ['F1', 'C1', 'F2']:
        P[(k, 'fruit')] = pos_last(k, FRUIT)
        P[(k, 'comp')] = pos_last(k, COMP)
    for k in C:
        P[(k, 'ban')] = pos_last(k, BAN)

    log('')
    log('目标位置: ' + ' | '.join('%s@%s=%s' % (a, b, P[(a, b)]) for a, b in sorted(P.keys(), key=str)))

    # ---- 核心量：层扫描 ----
    idxF1, idxC1, idxF2 = P[('F1', 'apple')], P[('C1', 'apple')], P[('F2', 'ban')]
    idxF1_fr, idxC1_fr = P[('F1', 'fruit')], P[('C1', 'fruit')]
    idxC1_cp, idxF1_cp = P[('C1', 'comp')], P[('F1', 'comp')]
    idxS1, idxS2 = P[('S1', 'apple')], P[('S2', 'apple')]

    rows = []
    for l in range(NL + 1):
        def v(key, p):
            return C[key]['st'][l][p]

        sep = cos(v('F1', idxF1), v('C1', idxC1))            # 苹果 水果义 vs 公司义
        ctrl = cos(v('S1', idxS1), v('S2', idxS2))           # 句首对照（无前文）
        g_f = cos(v('F1', idxF1), v('F1', idxF1_fr)) - cos(v('C1', idxC1), v('F1', idxF1_fr))  # 果义更像"水果"吗
        g_c = cos(v('C1', idxC1), v('C1', idxC1_cp)) - cos(v('F1', idxF1), v('C1', idxC1_cp))  # 司义更像"公司"吗
        b_f = cos(v('F1', idxF1), v('F2', idxF2))            # 苹果(果义) vs 香蕉(果义)
        b_c = cos(v('C1', idxC1), v('F2', idxF2))            # 苹果(司义) vs 香蕉(果义)
        # 句内 null：苹果位置 vs 同句其余所有位置 |cos| 的均值
        T = C['F1']['st'][l].shape[0]
        aa = C['F1']['st'][l][idxF1]
        vals = [abs(cos(aa, C['F1']['st'][l][j])) for j in range(T) if j != idxF1]
        null = sum(vals) / max(1, len(vals))
        rows.append((l, sep, ctrl, g_f, g_c, b_f, b_c, null))

    log('')
    log('=== 层扫描表（l=0 为 embedding 行；cos）===')
    log('%-4s %9s %9s %9s %9s %9s %9s %9s' % ('L', 'sep(苹:果/司)', 'ctrl(句首)', 'g_fruit', 'g_comp', 'bana_f', 'bana_c', 'null|cos|'))
    for r in rows:
        log('%-4d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f' % r)

    def argmin(i):
        return min(rows, key=lambda r: r[i])
    def argmax(i):
        return max(rows, key=lambda r: r[i])

    log('')
    log('=== 判决性读数 ===')
    log('1) L0 静态行：sep=%.4f  (期望 1.0000 => 同一 token 在所有语境下嵌入行完全相同)' % rows[0][1])
    log('2) sep 最小层：L%d  sep=%.4f' % (argmin(1)[0], argmin(1)[1]))
    log('3) ctrl(句首对照) 全层范围 [%.4f, %.4f]，min@L%d' % (min(r[2] for r in rows), max(r[2] for r in rows), argmin(2)[0]))
    log('4) g_fruit 最大层：L%d  %.4f ；g_comp 最大层：L%d  %.4f' % (argmax(3)[0], argmax(3)[3], argmax(4)[0], argmax(4)[4]))
    log('5) bana_f 最大 L%d=%.4f ; bana_c 最低 L%d=%.4f ; 最大差 %.4f @L%d'
        % (argmax(5)[0], argmax(5)[5], argmin(6)[0], argmin(6)[6],
           max(r[5] - r[6] for r in rows), max(rows, key=lambda r: r[5] - r[6])[0]))
    log('6) null|cos| 全层范围 [%.4f, %.4f]' % (min(r[7] for r in rows), max(r[7] for r in rows)))

    # 关键层摘录
    KEY = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 22, 24, 26, 28, 29, 30, 32, 34, NL]
    log('')
    log('=== 关键层摘录 ===')
    log('%-4s %9s %9s %9s %9s' % ('L', 'sep', 'g_fruit', 'g_comp', 'bana_f-bana_c'))
    for r in rows:
        if r[0] in KEY:
            log('%-4d %9.4f %9.4f %9.4f %9.4f' % (r[0], r[1], r[3], r[4], r[5] - r[6]))

    log('')
    log('用时 %.1fs' % (time.time() - t0))
    dump()
    print('E2 DONE')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('EXCEPTION:')
        log(traceback.format_exc())
        dump()
        raise
