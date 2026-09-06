"""All-coordinate causal-prefix invariance in saved native BF16 fields, no model load."""
import json,sys,itertools,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
FIELD=RESULT/'phase2671_native_mlp_field';OUT=RESULT/'phase2676_native_mlp_delivery'
HS=((24,2355),(27,1217));AS=((23,6197),(26,3594),(27,3221),(28,5952),(28,8513))


def decode(x):return (x.astype(np.uint32)<<16).view(np.float32)


def main():
    rows=read(RESULT/'phase2670_native_mlp_contract/material/cases.json');groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('family','language','unit','content_instance','form','target_index','mention_order')),[]).append(r)
    maps={};summary={};n=0;unequal=0;t0=time.monotonic();candidates=[]
    for gi,(key,rr) in enumerate(groups.items()):
        assert len(rr)==8 and {(r['probe_index'],r['polarity'],r['mapping']) for r in rr}==set(itertools.product((0,1),repeat=3))
        ref=rr[0];prefix=ref['prompt_ids'][:ref['body_end_token']+1]
        assert all(r['body']==ref['body'] and r['prompt_ids'][:r['body_end_token']+1]==prefix for r in rr)
        with np.load(FIELD/f'field/case_{ref["case_index"]:04d}.npz') as z:base={k:z[k][:,0].copy() for k in ('h','a')}
        stat=summary.setdefault('/'.join(key[:2]),{'groups':0,'pairs':0,'pairs_with_any_bit_difference':0,'maximum_absolute_difference':0.})
        stat['groups']+=1
        for r in rr[1:]:
            diffany=False;maxd=0.;changed={}
            with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
                for metric,sites in (('h',HS),('a',AS)):
                    bits=z[metric][:,0];changed_bits=bits!=base[metric];d=np.abs(decode(bits).astype('float64')-decode(base[metric]));diffany|=bool(changed_bits.any());maxd=max(maxd,float(d.max()))
                    if metric+'__different_bits_count' not in maps:
                        maps[metric+'__different_bits_count']=np.zeros_like(bits,dtype='int32');maps[metric+'__maximum_abs_difference']=np.zeros_like(d);maps[metric+'__sum_abs_difference']=np.zeros_like(d)
                    maps[metric+'__different_bits_count']+=changed_bits;np.maximum(maps[metric+'__maximum_abs_difference'],d,out=maps[metric+'__maximum_abs_difference']);maps[metric+'__sum_abs_difference']+=d
                    changed[metric]=[{'layer':l,'coordinate':j,'absolute_difference':float(d[l,j]),'different_bits':bool(changed_bits[l,j])} for l,j in sites]
            n+=1;unequal+=diffany;stat['pairs']+=1;stat['pairs_with_any_bit_difference']+=diffany;stat['maximum_absolute_difference']=max(stat['maximum_absolute_difference'],maxd)
            if any(v['different_bits'] for vv in changed.values() for v in vv):candidates.append({'base_case':ref['case_index'],'other_case':r['case_index'],'prefix_tokens':len(prefix),'total_lengths':[len(ref['prompt_ids']),len(r['prompt_ids'])],'candidate_differences':changed})
        if (gi+1)%64==0:
            save(OUT/'analysis/source_prefix_progress.json',{'groups':gi+1,'total':1024,'pairs':n,'unequal_pairs':unequal,'elapsed_seconds':time.monotonic()-t0});print('PREFIX AUDIT',gi+1,'/1024',flush=True)
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'maps/source_prefix_all_coordinates.npz',**maps)
    checks={'1024exacttokenprefix_groups':len(groups)==1024,'7168fullcoordinate_pairs':n==7168,'all16family_languages':len(summary)==16,'allvalues_finite':all(np.isfinite(v).all() for v in maps.values())}
    result={'checks':checks,'all_checks_passed':all(checks.values()),'groups':1024,'pairs':n,'pairs_with_any_bit_difference':unequal,'by_family_language':summary,
        'changed_coordinates':{m:int((maps[m+'__different_bits_count']>0).sum()) for m in ('h','a')},'maximum_absolute_difference':{m:float(maps[m+'__maximum_abs_difference'].max()) for m in ('h','a')},
        'changed_frozen_candidate_pairs':candidates,'scope':'Identical actual token prefix through body_end; futurep/q/m changes. Every rawbodyH andMLPunit at everylayer, notTopK. Any nonzero difference would require floating implementation/measurement checks, not a backward-causation claim. This reuses8192measuredfields, not8192newforwards.'}
    save(OUT/'analysis/source_prefix_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('by_family_language','changed_frozen_candidate_pairs')}));assert result['all_checks_passed']


if __name__=='__main__':main()
