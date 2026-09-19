"""Whole-coordinate bilingual condition profiles and matched semantic-group tests."""
from collections import defaultdict
from rdc_update_common import *

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2);out=BASE/'language_analysis';start=time.monotonic()
    if (out/'result.json').exists():return
    rows=gzread(BASE/'language_material.json.gz');h=[];activation=[];means={};indices=defaultdict(list)
    for i,r in enumerate(rows):
        with np.load(native_path(r)) as z:
            hh=unbits(z['H']);aa=np.stack([unbits(z[f'L{b}_activation']) for b in (6,16,35)])
        h.append(hh);activation.append(aa);indices[(r['family'],r['language'],r['answer_style'],r['split'])].append(i)
    h=np.array(h);activation=np.array(activation)
    raw=[];rms=[];unit=[];labels=[]
    for group,ii in sorted(indices.items()):
        v=h[ii];raw.append(v.mean(0));rms.append((v/np.sqrt(np.mean(v*v,-1,keepdims=True)).clip(1e-8)).mean(0));unit.append(activation[ii].mean(0))
        labels.append({'family':group[0],'language':group[1],'style':group[2],'split':group[3],'rows':len(ii)})
    npz(out/'all_condition_profiles.npz',raw_H=np.array(raw),RMS_H=np.array(rms),raw_MLP_activation=np.array(unit))
    save(out/'condition_profile_labels.json',labels)
    comparisons=[];lookup={(r['source_group'],r['language'],r['answer_style']):i for i,r in enumerate(rows)}
    for layer in (0,6,12,17,24,36):
      for anchor in (0,1):
        v=h[:,layer,anchor].astype(float);v/=np.linalg.norm(v,axis=1,keepdims=True).clip(1e-12);cos=(v@v.T).astype(np.float32)
        npz(out/f'all_pair_cosine_H{layer}_anchor{anchor}.npz',cosine=cos)
        for lang,style in (('zh','direct'),('en','explain'),('zh','explain')):
          for family in sorted({r['family'] for r in rows}):
            same=[];diff=[];groups=[]
            for i,r in enumerate(rows):
                if (r['family'],r['language'],r['answer_style'],r['split'])!=(family,'en','direct','language_test'):continue
                j=lookup[r['source_group'],lang,style]
                others=[k for k,s in enumerate(rows) if s['source_group']!=r['source_group'] and (s['family'],s['language'],s['answer_style'],s['split'],s['truth'])==(family,lang,style,r['split'],r['truth'])]
                if not others:continue
                same.append(float(cos[i,j]));diff.append(float(cos[i,j]-cos[i,others].mean()));groups.append(r['source_group'])
            comparisons.append({'layer':layer,'anchor':anchor,'family':family,'target_language':lang,'target_style':style,
              'held_pairs':len(same),'mean_pair_cosine':float(np.mean(same)) if same else None,
              'same_semantic_group_over_same_family_truth':clustered(diff,groups) if diff else None})
    records=[read(p) for p in sorted((BASE/'language_capture/commits').glob('*.json'))]
    result={'timestamp':stamp(),'source':snapshot(__file__),'rows':640,'semantic_groups':160,'profiles':labels,'comparisons':comparisons,
      'behavior':read(BASE/'language_capture/result.json')['summary'],'all_H_scalars_analyzed':int(h.size),'all_activation_scalars_analyzed':int(activation.size),
      'precision':'Native BF16 fields decoded losslessly; unthresholded FP32 profile statistics and FP64 dot products. Unit and residual coordinates remain distinct.',
      'scope':'Five controlled constructions with entity/context variation; pair controls match family and Boolean truth and split. Same templates remain, so this is not a universal semantics claim. All generation errors are retained.',
      'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('full_coordinate_language_profiles',result['seconds']);print('LANGUAGE_ANALYSIS_DONE',result['seconds'],flush=True)

if __name__=='__main__':main()
