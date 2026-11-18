"""Post hoc distance overlap and within-template support in the Answer: replay."""
from collections import defaultdict
import json

import source_binding_qkv_followup as exp
import source_binding_transfer_factors as io
from source_binding_design import sha


def main():
    io.OUT = exp.OUT
    manifest = io.read('holdout-manifest.json')
    assert manifest['passed']
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:[0,0,0])))
    for task in exp.TASKS:
        rows = {r['id']:r for r in io.read(f'holdout-{task}-materials.json')}
        for f in manifest['shards']:
            path=exp.OUT/f['file']
            if path.name.split('-')[1]!=task:
                continue
            assert sha(path)==f['sha256']
            for group in json.loads(path.read_bytes())['groups']:
                for i,id_ in enumerate(group['ids']):
                    r = rows[id_]
                    if r['cue']:
                        continue
                    base=group['items']['base'][i][0]
                    for cell,arm in group['items'].items():
                        if not cell.startswith('W|') or arm[i][0]==base:
                            continue
                        key=cell[2:]
                        site=key.split('@')[0]
                        sites=('qsrc','qent') if site=='names' else (site,)
                        dist=','.join(str(len(r['tokens'])-1-r['up'][s][-1]) for s in sites)
                        count=by[task][key][dist]
                        count[0]+=1
                        count[1]+=group['items']['H|'+key][i][0]==arm[i][0]
                        count[2]+=group['items']['Hpos|'+key][i][0]==arm[i][0]
    both={key:sorted(set(v) & set(by['zh'][key])) for key,v in by['colours'].items()}
    io.save('answer-distance-support.json',dict(counts=by,common_distance_by_cell=both,
            note='These are exact literal distance supports. Hpos changes only the rotary phase of edited names, not the rest of the text.'))


if __name__=='__main__':
    main()
