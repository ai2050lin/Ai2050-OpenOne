"""验证：safetensors 无 mmap 直读 与 safe_open 数值等价（2026-10-19）

背景：Windows 上 safetensors 的 framework='pt' 每次 safe_open 都 mmap 整个文件；
连续调用（逐层扫描 28 层）会累积未释放的 mapping，最终 OSError 1455（页面文件太小）。
即 /api/live/weights/scan 与 /api/live/weights/summary 在「连续多次」场景下必挂。

方案：自己解析 safetensors header（8B 长度 + JSON），按 data_offsets + seek 只读需要的行块，
完全不 mmap。内存峰值 = 一次行块字节数（几十 MB → 几 MB）。

本脚本做严格等价性检查：同一 tensor、同一行块，两条路径的 rms/mean_abs/max_abs 必须一致。
"""
import json
import math
import struct
import sys

import numpy as np
from safetensors import safe_open

MODEL = 'models/hf/qwen3-1.7b/model-00001-of-00002.safetensors'

# safetensors dtype → (numpy 存储 dtype, itemsize, 转 float32 的函数)
_DT = {
    'F64': (np.float64, 8, lambda a: a.astype(np.float32)),
    'F32': (np.float32, 4, lambda a: a.astype(np.float32)),
    'F16': (np.float16, 2, lambda a: a.astype(np.float32)),
    'BF16': (np.uint16, 2, lambda a: (a.astype(np.uint32) << 16).view(np.float32)),
    'I64': (np.int64, 8, lambda a: a.astype(np.float32)),
    'I32': (np.int32, 4, lambda a: a.astype(np.float32)),
    'I16': (np.int16, 2, lambda a: a.astype(np.float32)),
    'I8': (np.int8, 1, lambda a: a.astype(np.float32)),
    'U8': (np.uint8, 1, lambda a: a.astype(np.float32)),
}


def read_header(path):
    """返回 (header_dict, data_base_offset)。"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n).decode('utf-8'))
    return hdr, 8 + n


def stats_direct(path, name, hdr, base, blocks=32, block_rows=64):
    """无 mmap：按行块 seek 读取并累计标量统计。"""
    meta = hdr[name]
    dtype = meta['dtype']
    shape = meta['shape']
    if dtype not in _DT:
        raise ValueError('unsupported dtype ' + dtype)
    np_dt, item, to_f32 = _DT[dtype]
    off0, off1 = meta['data_offsets']

    if len(shape) <= 1:
        rows, cols = 1, int(shape[0]) if shape else 1
        starts, sampled = [(0, rows)], False
    else:
        rows, cols = int(shape[0]), int(shape[1])
        need = blocks * block_rows
        if rows <= need:
            starts = [(s, min(rows, s + block_rows)) for s in range(0, rows, block_rows)]
            sampled = False
        else:
            step = (rows - need) / (blocks - 1)
            starts = []
            for i in range(blocks):
                s = int(round(i * step))
                starts.append((s, min(rows, s + block_rows)))
            sampled = True

    row_bytes = cols * item
    n = 0
    s1 = 0.0
    s1a = 0.0
    s2 = 0.0
    mn = None
    mx = None
    with open(path, 'rb') as f:
        for r0, r1 in starts:
            if r1 <= r0:
                continue
            f.seek(base + off0 + r0 * row_bytes)
            raw = f.read((r1 - r0) * row_bytes)
            if len(raw) != (r1 - r0) * row_bytes:
                raise IOError('short read on %s rows %d-%d' % (name, r0, r1))
            arr = to_f32(np.frombuffer(raw, dtype=np_dt).reshape(r1 - r0, cols))
            n += arr.size
            s1 += float(arr.astype(np.float64).sum())
            s1a += float(np.abs(arr).astype(np.float64).sum())
            s2 += float((arr.astype(np.float64) ** 2).sum())
            lo, hi = float(arr.min()), float(arr.max())
            mn = lo if mn is None else min(mn, lo)
            mx = hi if mx is None else max(mx, hi)

    return {
        'name': name,
        'mean': round(s1 / n, 6),
        'mean_abs': round(s1a / n, 6),
        'rms': round(math.sqrt(s2 / n), 6),
        'min': round(mn, 6),
        'max': round(mx, 6),
        'max_abs': round(max(abs(mn), abs(mx)), 6),
        'sampled': sampled,
        'rows_examined': sum(r1 - r0 for r0, r1 in starts),
    }


def stats_safeopen(path, name, blocks=32, block_rows=64):
    """对照：原实现（safe_open mmap + torch 切片）。"""
    import torch
    with safe_open(path, framework='pt', device='cpu') as f:
        sl = f.get_slice(name)
        shape = list(sl.get_shape())
        if len(shape) <= 1:
            blocks_iter = [(0, sl[:].float(), False)]
        else:
            rows = shape[0]
            need = blocks * block_rows
            if rows <= need:
                rng = [(s, min(rows, s + block_rows)) for s in range(0, rows, block_rows)]
                sampled = False
            else:
                step = (rows - need) / (blocks - 1)
                rng = [(int(round(i * step)), min(rows, int(round(i * step)) + block_rows)) for i in range(blocks)]
                sampled = True
            blocks_iter = [(r0, sl[r0:r1].float(), sampled) for r0, r1 in rng]
    n = 0
    s1 = s1a = s2 = 0.0
    mn = mx = None
    for _r0, t, _s in blocks_iter:
        n += int(t.numel())
        s1 += float(t.sum())
        s1a += float(t.abs().sum())
        s2 += float((t * t).sum())
        lo, hi = float(t.min()), float(t.max())
        mn = lo if mn is None else min(mn, lo)
        mx = hi if mx is None else max(mx, hi)
    return {
        'mean': round(s1 / n, 6),
        'mean_abs': round(s1a / n, 6),
        'rms': round(math.sqrt(s2 / n), 6),
        'min': round(mn, 6),
        'max': round(mx, 6),
        'max_abs': round(max(abs(mn), abs(mx)), 6),
    }


def main():
    hdr, base = read_header(MODEL)
    targets = [
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.0.input_layernorm.weight',
    ]
    print('header tensors:', len(hdr) - 1, '· data base offset:', base)
    fail = 0
    for nm in targets:
        if nm not in hdr:
            print('SKIP (not in file):', nm)
            continue
        a = stats_direct(MODEL, nm, hdr, base)
        b = stats_safeopen(MODEL, nm)
        keys = ['mean', 'mean_abs', 'rms', 'min', 'max', 'max_abs']
        same = all(abs(a[k] - b[k]) < 1e-6 for k in keys)
        print('%-46s dtype=%-5s same=%s' % (nm.split('.', 2)[-1], hdr[nm]['dtype'], 'YES' if same else 'NO'))
        print('    direct   rms=%.6f mean|W|=%.6f max|W|=%.6f sampled=%s' %
              (a['rms'], a['mean_abs'], a['max_abs'], a['sampled']))
        print('    safeopen rms=%.6f mean|W|=%.6f max|W|=%.6f' %
              (b['rms'], b['mean_abs'], b['max_abs']))
        if not same:
            fail += 1
            for k in keys:
                if abs(a[k] - b[k]) >= 1e-6:
                    print('      DIFF %s: direct=%r safeopen=%r' % (k, a[k], b[k]))

    # 连续 30 次直读（不 mmap）——验证不会累积耗尽 commit
    print('\n连续直读 30 次（无 mmap）：')
    nm = targets[0]
    for i in range(30):
        stats_direct(MODEL, nm, hdr, base)
    print('  30/30 完成，无 OSError')

    print('\nRESULT:', 'PASS' if fail == 0 else 'FAIL(%d)' % fail)
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())