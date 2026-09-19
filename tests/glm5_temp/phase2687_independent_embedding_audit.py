"""Independent H0 operation-map reproduction from checkpoint embeddings only."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from collections import defaultdict
import numpy as np
import torch
from safetensors import safe_open
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2687_role_qkv_field'
MATERIAL=RESULT/'phase2686_independent_role_contract'
AXES=('family','language','unit','content_instance','form','roster_order','mention_order','target_index','output_function')
OPS=('roster_order','mention_order','target_index','form')


def main(partial):
    rows=read(MATERIAL/'material/initial.json');byindex={r['case_index']:r for r in rows}
    model=ROOT/'models/hf/qwen3-4b';key='model.embed_tokens.weight';path=model/read(model/'model.safetensors.index.json')['weight_map'][key]
    ids=sorted({r['prompt_ids'][r[p]] for r in rows for p in ('body_end_token','task_end_token')})
    with safe_open(str(path),framework='pt',device='cpu') as f:
        sl=f.get_slice(key);embedding={i:sl[i,:].float().numpy().astype(np.float64) for i in ids}
    cells=defaultdict(list)
    for r in rows:cells[r['family']+'_'+r['language']].append(r)
    checked=[]
    for cell,rr in cells.items():
        if not (OUT/f'analysis/cell_{cell}.json').exists():continue
        values={r['case_index']:np.stack([embedding[r['prompt_ids'][r[p]]] for p in ('body_end_token','task_end_token')]) for r in rr}
        with np.load(OUT/f'maps/operations_{cell}.npz') as z:
            for axis in OPS:
                edges=defaultdict(list)
                for r in rr:edges[tuple(r[k] for k in AXES if k!=axis)].append(r)
                pos=np.zeros((2,2560),np.uint16);neg=pos.copy();total=np.zeros((2,2560));absolute=total.copy();four=defaultdict(list)
                assert len(edges)==256
                for pair in edges.values():
                    a,b=sorted(pair,key=lambda r:r[axis]);d=values[b['case_index']]-values[a['case_index']]
                    pos+=d>0;neg+=d<0;total+=d;absolute+=np.abs(d)
                    four[tuple(a[k] for k in AXES if k not in (axis,'output_function'))].append(d)
                p4=np.zeros_like(pos);n4=np.zeros_like(pos);assert len(four)==64
                for dd in four.values():
                    assert len(dd)==4
                    ds=np.stack(dd);p4+=(ds>0).all(0);n4+=(ds<0).all(0)
                for suffix,value in [('positive',pos),('negative',neg),('sum',total),('sumabs',absolute),('all4_positive',p4),('all4_negative',n4)]:
                    assert np.array_equal(z[f'{axis}__h__{suffix}'][0],value),(cell,axis,suffix)
        checked.append(cell)
    if not partial:assert len(checked)==16
    report={'all_available_committed_cells_passed':True,'full16_passed':len(checked)==16,'checked_cells':checked,'unique_checkpoint_boundary_token_ids':len(ids),
        'native_H0_all2560coordinates_reproduced':True,'arrays_each_cell':24,'coordinate_values_compared':len(checked)*24*2*2560,
        'model_loaded':False,'cuda_initialized':torch.cuda.is_initialized(),'limits':'Independent checkpointE reconstruction validates indexing, edge orientation, counts and denominators atH0 only. It is not an independent deeper-layer semantic validation.'}
    assert not report['cuda_initialized'];save(OUT/'analysis/independent_embedding_map_audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args();main(args.allow_partial)
