"""验证修复版 _wt_blocks 在真实模型 28 层上连续工作（2026-10-19）

不启 HTTP、不加载模型权重，直接 import deploy/distributed_service.py 里的
_wt_blocks（已改为 safetensors header + seek 直读，不再 mmap），
对 qwen3-1.7b 全部 28 层 × 3 个承重矩阵跑一遍，检查：
  ① 无 OSError（原实现连续调用必挂）
  ② 数值与 safe_open 路径一致（抽样比对）
  ③ 单层耗时（决定前端「运行」的流畅度）
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'deploy'))

import distributed_service as DS  # noqa: E402

MODEL_DIR = 'models/hf/qwen3-1.7b'
PROBES = ['self_attn.q_proj.weight', 'mlp.gate_proj.weight', 'mlp.down_proj.weight']


def scan(path):
    """对单个 tensor 调 _wt_blocks 累计标量统计（与服务端 _wt_scalar_stats 同口径）。"""
    import math
    shard = None
    import glob
    for f in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        hdr, _ = DS._wt_header(f)
        if PRB in hdr:
            shard = f
            break
    if not shard:
        return None
    meta = hdr[PRB]
    tmeta = {'shape': meta['shape'], 'dtype': meta['dtype'], 'file': os.path.basename(shard)}
    n = 0
    s1a = s2 = 0.0
    sampled = False
    for _s, t, smp in DS._wt_blocks(shard, PRB, tmeta):
        sampled = sampled or smp
        n += int(t.numel())
        s1a += float(t.abs().sum())
        s2 += float((t * t).sum())
    return {'rms': round(math.sqrt(s2 / n), 6), 'mean_abs': round(s1a / n, 6), 'sampled': sampled}


def main():
    # 从 connect 载荷拿层→tensor 映射
    import json
    from pathlib import Path
    payload = DS._scan_live_model(Path(MODEL_DIR))
    tmap = {t['name']: t for t in payload['tensors']}
    import re
    RE = re.compile(r'(?:^|\.)(?:layers|h|blocks)\.(\d+)\.')
    by = {}
    for t in payload['tensors']:
        m = RE.search('.' + t['name'])
        if m:
            by.setdefault(int(m.group(1)), []).append(t)
    print('模型: %s · %d 层 · %d tensors' % (payload.get('arch'), len(by), len(payload['tensors'])))

    global PRB
    rows = []
    t0 = time.time()
    for layer in sorted(by):
        rms_sum, cnt = 0.0, 0
        for key in PROBES:
            hit = [t for t in by[layer] if key in t['name']]
            if not hit:
                continue
            PRB = hit[0]['name']
            st = scan(MODEL_DIR)
            if st:
                rms_sum += st['rms']
                cnt += 1
        rows.append((layer, rms_sum / cnt if cnt else 0, cnt))
    elapsed = time.time() - t0
    print('逐层扫描完成: %d 层 · 总 %.2fs · 平均 %.3fs/层（每层 %d 个 probe）'
          % (len(rows), elapsed, elapsed / max(1, len(rows)), len(PROBES)))

    vs = [r[1] for r in rows]
    print('rms 区间 %.6f – %.6f · 极差比 %.2fx' % (min(vs), max(vs), max(vs) / min(vs)))
    for l, v, c in rows[:5]:
        print('  L%-2d avg_rms=%.6f (%d)' % (l, v, c))
    print('  ...')
    for l, v, c in rows[-3:]:
        print('  L%-2d avg_rms=%.6f (%d)' % (l, v, c))

    # 等价性抽检：拿 L0 down_proj 与 safe_open 对比
    from safetensors import safe_open
    import math
    hit = [t for t in by[0] if 'down_proj' in t['name']][0]
    f0 = os.path.join(MODEL_DIR, hit['file'])
    with safe_open(f0, framework='pt', device='cpu') as fh:
        sl = fh.get_slice(hit['name'])
        rows_n = hit['shape'][0]
        blocks, brows = 32, 64
        if rows_n <= blocks * brows:
            rng = [(s, min(rows_n, s + brows)) for s in range(0, rows_n, brows)]
        else:
            step = (rows_n - blocks * brows) / (blocks - 1)
            rng = [(int(round(i * step)), min(rows_n, int(round(i * step)) + brows)) for i in range(blocks)]
        n = 0
        s2 = 0.0
        for r0, r1 in rng:
            tt = sl[r0:r1].float()
            n += int(tt.numel())
            s2 += float((tt * tt).sum())
    ref = round(math.sqrt(s2 / n), 6)
    PRB = hit['name']
    got = scan(MODEL_DIR)['rms']
    print('\n等价性: L0 down_proj direct=%.6f safeopen=%.6f → %s'
          % (got, ref, 'PASS' if abs(got - ref) < 1e-6 else 'FAIL'))
    ok = abs(got - ref) < 1e-6 and len(rows) == len(by)
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())