"""Finish a frozen native tail atlas with direct read-only access to the B32 seam."""
import time

import numpy as np

from mlp_tail_extension import BASE, OUT, RUN, BLOCKS, SPLITS, C, finalize


def associations():
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    records = []

    def raw(block, field):
        directory = BASE if block == 32 else OUT
        return np.load(directory / f'B{block}-{field}.npy', mmap_mode='r')

    def save(source, target, kind, values):
        assert torch.isfinite(values).all()
        name = f'B{source}-B{target}-{kind}.npy'
        np.save(OUT / name, values.cpu().numpy())
        records.append({'source': source, 'target': target, 'kind': kind, 'shape': [C.WIDTH, C.WIDTH]})
        C.log('full_associations', len(records), 20, started, source=source, target=target, kind=kind)

    slices = (slice(0, 16384), slice(16384, 24576), slice(24576, 32768))
    with C.GpuLease(), torch.inference_mode():
        for source in BLOCKS:
            target = source + 1
            for split, rows in zip(SPLITS, slices):
                x, y = (torch.from_numpy(C.unbits(raw(b, 'A')[rows])).cuda() for b in (source, target))
                x -= x.mean(0)
                y -= y.mean(0)
                denominator = x.square().sum(0).sqrt()[:, None] * y.square().sum(0).sqrt()[None, :]
                save(source, target, f'rho-{split}', (x.T @ y) / denominator.clamp_min(1e-20))
                del x, y, denominator
            down = torch.from_numpy(C.unbits(raw(source, 'W-down'))).cuda().T
            down = torch.nn.functional.normalize(down, eps=1e-20)
            for field in ('gate', 'up'):
                read = torch.from_numpy(C.unbits(raw(target, f'W-{field}'))).cuda()
                read = torch.nn.functional.normalize(read, eps=1e-20)
                save(source, target, f'weight-{field}', down @ read.T)
                del read
            del down
            torch.cuda.empty_cache()
    C.save_json(OUT / 'associations.json', {'schema': 'mlp_tail_associations.v1', 'kind': 'real', 'run_id': RUN,
                'matrices': records, 'scope': '共同响应和静态参数对齐。保留全部单元配对。'})


def main():
    plan = C.read(OUT / 'plan.json')
    assert C.file_sha(OUT / 'frozen-source.py') == plan['source_code_sha256']
    assert not any(OUT.glob('B3?-B3?-*.npy')) and not any(OUT.glob('B2?-B2?-*.npy'))
    for block in BLOCKS:
        assert all((OUT / f'B{block}-{field}.json').is_file() for field in ('profile', 'cues', 'readout'))
    C.save_json(OUT / 'association-implementation.json', {'kind': 'real', 'run_id': RUN,
                'source_sha256': C.file_sha(__file__),
                'original_frozen_source_sha256': plan['source_code_sha256'],
                'reason': 'Windows 沙盒拒绝创建 B32 硬链接。直接只读已验哈希的原始 B32 产物；保持全部原计算式、阈值和划分。'})
    associations()
    finalize(plan)


if __name__ == '__main__':
    main()
