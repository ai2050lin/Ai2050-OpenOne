"""Independent source-to-card checks using real retained fields (no new forward)."""
import time
import numpy as np
import qwen14_goodfire_extension as E
from qwen14_goodfire_query import card,full_relations
C=E.C


def main():
    E.configure();start=time.monotonic();checks=[]
    for b,u in [(27,0),(40,0),(28,-1),(39,C.WIDTH)]:
        try:card(b,u)
        except ValueError:checks.append({'check':'out_of_range_rejected','block':b,'unit':u,'passed':True})
        else:raise AssertionError('Invalid query accepted')
    for b,u in [(28,0),(39,C.WIDTH-1)]:
        result=card(b,u)
        assert result['block']==b and result['unit']==u and result['model_snapshot']==E.SNAP
        assert not result['semantic_confirmed']
        for f in ['G','V','A']:
            raw=E.data(b,f)
            assert np.isclose(result['statistics'][f]['mean'],C.unbits(raw[:,u]).mean(dtype=np.float64))
            for split,contexts in result['context_examples'][f].items():
                low,high=C.SPLITS[split]
                assert all(low<=x['document_index']<high and x['future_tokens_excluded'] and x['position']<=1023 for x in contexts)
        checks.append({'check':'boundary_card_matches_native_field','block':b,'unit':u,'passed':True})
        E.log('query_validation',len(checks),10,start)
    relations=full_relations(36,88)
    assert len(relations['incoming']['other_units'])==len(relations['outgoing']['other_units'])==C.WIDTH
    for direction in relations.values():
        for values in [*direction['static_parameter_cosines'].values(),*direction['observed_A_correlations'].values()]:
            assert len(values)==C.WIDTH and np.isfinite(values).all() and max(abs(x) for x in values)<=1.00001
    fitted=C.read(E.OUT/'B35-B36-prediction.json');parent=fitted['parents'][88]
    actual=relations['incoming']['observed_A_correlations']['discovery'][parent]
    error=abs(actual-fitted['discovery_correlation'][88]);assert error<1e-5,error
    checks.append({'check':'full_relation_row_matches_frozen_discovery_parent','absolute_error':error,'passed':True})
    # Read original checkpoint CPU tensors only; this does not execute a model.
    import torch
    for branch in ['gate','up','down']:
        name={'gate':'gate_proj','up':'up_proj','down':'down_proj'}[branch]
        tensor=C.weight(f'model.layers.36.mlp.{name}.weight')
        native=tensor[:,88] if branch=='down' else tensor[88]
        raw=np.load(E.OUT/f'B36-W-{branch}.npy',mmap_mode='r')
        stored=raw[:,88] if branch=='down' else raw[88]
        assert np.array_equal(native.contiguous().view(torch.uint16).numpy(),stored)
        checks.append({'check':'parameter_vector_matches_original_checkpoint','branch':branch,'passed':True})
        del tensor
    E.save(E.OUT/'query-validation.json',{'passed':True,'checks':checks,'seconds':time.monotonic()-start,
        'scope':'Ten source/card/index checks with actual retained data; CPU parameter reads only, no GPU model forward and no scientific mechanism promotion.'})
    print('passed',len(checks),flush=True)


if __name__=='__main__':main()
