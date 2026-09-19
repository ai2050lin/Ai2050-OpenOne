"""Exploratory full-coordinate UD relation profiles with within-sentence signed-distance controls."""
from collections import defaultdict,Counter
from phase2714_rdc_full_source_history import *
from phase2711_rdc_prefix_atlas import span_token
TYPES=('nsubj','obj','iobj','obl','nmod','amod','advmod','advcl','acl','conj','compound','case','mark','det')


def main():
    out=OUT/'relations';start=time.monotonic()
    immutable(out/'protocol.json',{'timestamp':stamp(),'phase':2714,'source_sha':sha(Path(__file__)),
      'types':list(TYPES),'scope':'Exploratory retrospective observation after source-kernel results; reused512 main and64 fresh materials, NOT a new frozen confirmation of this new relation statistic.',
      'alignment':'UD dependent/head word spans to their final overlapping token; unaligned spans and self-token pairs excluded and counted. Individual token/word distinction preserved.',
      'control':'For every relation pair (a,b), choose deterministic other observed pair (c,c+b-a) in same sentence, exactly preserving signed token distance. No model intervention. Lexical identity/POS and semantic content not matched.',
      'statistic':'All2560 source_i*target_i entries, raw and coordinate z-score with frozen main all-token H12 mean/std. Compare against matched-control product, retain all signed differences and raw profiles.',
      'replication':'Training-reference full-coordinate profile cosine on main test and reused fresh;500 source-unit bootstrap replicates holding training profile fixed. Tiny groups explicitly counted.',
      'limits':['Diagonal paired-coordinate profile, NOT full2560x2560 covariance; original complete matrices remain available from2711.','Retrospective gold relation labels are not causal prefix-parser outputs or forecast inputs.','Matched positions can accidentally share a true relation and are not semantic counterfactuals.','No pure semantic gear, necessity, new theorem or autonomous closure follows.']})
    with np.load(CAMPAIGN/'qwen4/all_token_moments.npz') as z:
        n=z['counts'][:2].sum();mu=z['sums'][:2,12].sum(0)/n;std=np.sqrt(np.maximum(z['squares'][:2,12].sum(0)/n-mu*mu,1e-12))
    agg={};groups={};pairs=[];skipped=Counter()
    def add(key,raw,control,zprod,zcontrol,sid):
        if key not in agg:agg[key]={'n':0,**{k:np.zeros(2560,np.float64) for k in ('raw','control','zprod','zcontrol')}};groups[key]={}
        a=agg[key];a['n']+=1
        for name,v in [('raw',raw),('control',control),('zprod',zprod),('zcontrol',zcontrol)]:a[name]+=v
        if sid not in groups[key]:groups[key][sid]=[0,np.zeros(2560,np.float64)]
        groups[key][sid][0]+=1;groups[key][sid][1]+=zprod-zcontrol
    for fresh in (False,True):
        material=read(OUT/'fresh_material.json') if fresh else main_rows()
        for r in material:
            h=source_field(r,fresh).astype(np.float64);z=(h-mu)/std;words={w['id']:w for w in r['retrospective_ud']};split='fresh' if fresh else r['split']
            for w in words.values():
                rel=w['relation'].split(':')[0]
                if rel not in TYPES:continue
                if w['head'] not in words:skipped['missing_head']+=1;continue
                a=span_token(r,w['char_span']);b=span_token(r,words[w['head']]['char_span'])
                if a is None or b is None:skipped['unaligned_span']+=1;continue
                if a==b:skipped['same_token']+=1;continue
                d=b-a;options=[c for c in range(max(0,-d),min(len(h),len(h)-d)) if c!=a]
                if not options:skipped['no_same_distance_control']+=1;continue
                seed=int(hashlib.sha256(f'{r["sample_id"]}:{w["id"]}:{w["head"]}'.encode()).hexdigest()[:16],16)
                c=options[seed%len(options)];e=c+d
                add((rel,split),h[a]*h[b],h[c]*h[e],z[a]*z[b],z[c]*z[e],r['sample_id'])
                pairs.append({'sample_id':r['sample_id'],'scope':'fresh' if fresh else 'main','split':split,'language':r['language'],
                  'relation':rel,'word_ids':[w['id'],w['head']],'observed_token_pair':[a,b],'control_token_pair':[c,e],'signed_distance':d})
    arrays={};reports=[]
    for rel in TYPES:
        rec={'relation':rel,'splits':{},'repetition':[]}
        for split in ('train','validation','test','fresh'):
            key=(rel,split)
            if key not in agg:rec['splits'][split]={'pairs':0,'source_units':0};continue
            a=agg[key];rec['splits'][split]={'pairs':a['n'],'source_units':len(groups[key])}
            for name in ('raw','control','zprod','zcontrol'):arrays[f'{rel}_{split}_{name}']=(a[name]/a['n']).astype(np.float32)
            arrays[f'{rel}_{split}_raw_delta']=((a['raw']-a['control'])/a['n']).astype(np.float32)
            arrays[f'{rel}_{split}_z_delta']=((a['zprod']-a['zcontrol'])/a['n']).astype(np.float32)
            rec['splits'][split]['mean_z_product_minus_control']=float(arrays[f'{rel}_{split}_z_delta'].mean())
        for split in ('test','fresh'):
            if (rel,'train') not in agg or (rel,split) not in agg:continue
            train=arrays[f'{rel}_train_z_delta'].astype(float);test=arrays[f'{rel}_{split}_z_delta'].astype(float)
            rawtr=arrays[f'{rel}_train_zprod'].astype(float);rawte=arrays[f'{rel}_{split}_zprod'].astype(float)
            cosine=lambda a,b:float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
            entries=list(groups[(rel,split)].values());counts=np.array([v[0] for v in entries]);sums=np.stack([v[1] for v in entries]);g=len(entries)
            rng=np.random.default_rng(2714);weights=rng.multinomial(g,np.full(g,1/g),size=500);profiles=weights@sums/(weights@counts)[:,None]
            cs=profiles@train/np.maximum(np.linalg.norm(profiles,axis=1)*np.linalg.norm(train),1e-30)
            rec['repetition'].append({'split':split,'full_coordinate_cosine_before_control':cosine(rawtr,rawte),
              'full_coordinate_cosine_after_control':cosine(train,test),'fixed_train_source_bootstrap_CI95':np.quantile(cs,[.025,.975]).tolist(),
              'source_units':g,'pairs':agg[(rel,split)]['n'],'low_support_warning':g<10 or agg[(rel,split)]['n']<20})
        reports.append(rec)
    npz(out/'all_coordinate_profiles.npz',**arrays);save(out/'pairs.json',pairs)
    save(out/'result.json',{'timestamp':stamp(),'phase':2714,'reports':reports,'pairs':len(pairs),'skipped':dict(skipped),
      'counts_by_scope':dict(Counter(p['scope'] for p in pairs)),'full_coordinates':2560,'relations':len(TYPES),'evidence':'Exploratory retrospective relation-conditioned paired-coordinate profiles, not an independently frozen new relation law.',
      'source_fields':'All512 main full-source H12 fields (16 reused original panels) plus64 new fresh full-source fields.',
      'seconds':time.monotonic()-start,'new_theorem':False,'semantic_causal_gear_identified':False})
    print('SOURCE_RELATIONS_COMPLETE',len(pairs),dict(skipped),[(r['relation'],[(x['split'],round(x['full_coordinate_cosine_after_control'],4),x['source_units']) for x in r['repetition']]) for r in reports],flush=True)
    guard(12*1024**2)


if __name__=='__main__':main()
