"""CPU-only exact enumeration control, not pretrained semantic evidence."""
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
import phase2687_role_qkv_field as q
from phase2679_native_source_ledger import exact_bits
from phase2671_native_mlp_field import unbits


def main():
    original=q.SHAPES
    q.SHAPES={'h':(2,2,5),'probability':(2,2,3,256)}
    rows=[];data={}
    for f,r,o,v,fun in itertools.product(range(2),range(2),range(2),range(2),q.FUNCTIONS):
        i=len(rows);n=8+q.FUNCTIONS.index(fun)
        row=dict(case_index=i,form=f,roster_order=r,mention_order=o,target_index=v,output_function=fun,prompt_ids=[0]*n)
        rows.append(row)
        h=np.ones((2,2,5))*((r-o)*4+v*8+f*16)
        # Fourth coordinate reverses between functions, fifth stays zero.
        h[:,:,3]*=(-1)**q.FUNCTIONS.index(fun);h[:,:,4]=0
        p=np.zeros((2,2,3,256));p[...,:n]=(1+r+2*o+4*v+8*f)/64
        data[i]={'h':exact_bits(h),'probability':exact_bits(p)}
    maps=q.init_maps();q.operation_maps(rows,data,maps)
    checked=0
    for axis in q.OPERATIONS:
        pairs=[]
        for a in rows:
            if a[axis]:continue
            bb=[b for b in rows if b[axis]==1 and all(a[k]==b[k] for k in q.FACTORS if k!=axis)]
            assert len(bb)==1;pairs.append((a,bb[0]))
        for key,shape in q.SHAPES.items():
            deltas=[];valid=[]
            for a,b in pairs:
                d=unbits(data[b['case_index']][key]).astype(np.float64)-unbits(data[a['case_index']][key]).astype(np.float64)
                va=np.ones(shape,bool)
                if key=='probability':va[...,min(len(a['prompt_ids']),len(b['prompt_ids'])):]=False
                d[~va]=0;deltas.append(d);valid.append(va)
            d=np.stack(deltas)
            for suffix,expected in [('positive',(d>0).sum(0)),('negative',(d<0).sum(0)),('sum',d.sum(0)),('sumabs',np.abs(d).sum(0))]:
                assert np.array_equal(maps[f'{axis}__{key}__{suffix}'],expected);checked+=1
            if key=='probability':
                assert np.array_equal(maps[f'{axis}__{key}__valid_count'],np.stack(valid).sum(0));checked+=1
        hp=maps[f'{axis}__h__all4_positive'];hn=maps[f'{axis}__h__all4_negative']
        assert (hp[:,:,3:]==0).all() and (hn[:,:,3:]==0).all()
        assert ((hp+hn)[:,:,:3]==8).all();checked+=1
    q.SHAPES=original
    rows=q.read(q.CONTRACT/'material/initial.json');fresh=q.read(q.CONTRACT/'material/confirmation.json')
    b=q.budget_for(rows);c=q.budget_for(fresh);free=q.shutil.disk_usage(q.RESULT).free
    required=b['total']+c['total']+2*1024**3+q.FLOOR
    result={'all_checks_passed':True,'synthetic_exact_checks':checked,'real_model_tests':0,'initial_bytes':b,'confirmation_bytes':c,
            'free_bytes':free,'required_bytes':required,'storage_fits':free>required,'model_initialized':False}
    q.save(q.OUT/'analysis/cpu_algorithm_preflight.json',result);print(result,flush=True)
    assert result['storage_fits']


if __name__=='__main__':main()
