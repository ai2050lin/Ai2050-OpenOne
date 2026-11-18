"""Describe Phase 426 distance support by frame, without inferring a causal effect."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import source_binding_transfer_cue as cue
from deploy.storage_policy import check_write


def main():
    root = cue.HISTORICAL
    report = {}
    for task in cue.TASKS:
        rows = json.loads((root/f'{task}-holdout-materials.json').read_bytes())
        by_frame = defaultdict(list)
        for row in rows:
            by_frame[row['frame']].append(row)
        stats = {}
        for frame, group in by_frame.items():
            stats[str(frame)] = {site: dict(median=median(dist), minimum=min(dist), maximum=max(dist))
                                 for site in ('qsrc','qent')
                                 if (dist:=[len(r['tokens'])-1-r['up'][site][-1] for r in group])}
        report[task] = stats
    out = check_write(cue.OUT/'historical-distance-support.json',1<<20)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
