# -*- coding: utf-8 -*-
# p3152 patch: move embedding prefetch before model release; delete bad block
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3152_g1p2_tri_model_k1.py'
src = io.open(P, encoding='utf-8').read()

old1 = """    del model
    torch.cuda.empty_cache()
    log('model released (H16 in RAM)')"""
new1 = """    # ---- embedding rows prefetch (model still on GPU) ----
    need = sorted(set(E_TOK + CLS_TOK +
                      sorted({t for tk in TOKIDS for t in tk})))
    with torch.no_grad():
        _idx = torch.tensor(need, device='cuda')
        _emb_rows = model.get_input_embeddings().weight[_idx].float().cpu().numpy()
    EMB = {'need': need, 'rows': _emb_rows}
    log('embedding rows prefetched: %s' % (_emb_rows.shape,))
    del model
    torch.cuda.empty_cache()
    log('model released (H16 in RAM)')"""
assert src.count(old1) == 1, ('old1 count', src.count(old1))
src = src.replace(old1, new1)

i0 = src.index("    # ---- 特征预计算（B2 词袋 / M2 嵌入中介） ----")
i1 = src.index("    log('embedding rows prefetched: %s' % (emb_rows.shape,))")
i1 = src.index("\n", i1) + 1
removed = src[i0:i1]
assert 'safe_open' in removed and 'get_emb' in removed, 'bad block bounds'
src = src[:i0] + src[i1:]

# remove dead Dk_of helper (unused, model-dependent ref) - keep simple
old2 = """    def Dk_of(k, tr_rows):
        Y = Y_at(k)
        ref = Y[tr_rows].mean(0)
        return float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9

"""
assert src.count(old2) == 1, ('old2 count', src.count(old2))
src = src.replace(old2, "")

io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('patched ok; removed %d chars' % len(removed))
