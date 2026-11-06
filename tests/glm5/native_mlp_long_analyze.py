"""Frozen text-only context predictor, two heldout tests and strict controls."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from native_mlp_long_features import categories,prefix_features,FEATURES,mcnemar
from shared.dist_protocol.mechanism_trace import unbits,file_sha,digest
def fits(y,ids,pos,x,gram,perm):
    counts=np.bincount(ids,minlength=151936);mu=y.mean(0);means=np.zeros((151936,y.shape[1]),np.float32)
    np.add.at(means,ids,y);means/=np.maximum(1,counts)[:,None];means[counts==0]=mu
    r=y-means[ids];pm=np.stack([r[pos//64==b].mean(0) for b in range(16)])
    r-=pm[pos//64]
    coef=solve(gram,x.T@r,assume_a='pos');null=solve(gram,x.T@r[perm],assume_a='pos')
    return means,pm,coef,null
def main():
    from transformers import AutoTokenizer
    data=P.read(P.OUT/'documents.json');collection=P.read(P.OUT/'collection.json');pr=P.read(P.PUBLIC/'prereg.research.json');docs=data['docs'];units=data['units']
    freeze=P.read(P.OUT/'execution-freeze.json');assert all(file_sha(P.ROOT/n)==h for n,h in freeze['dependencies'].items())
    raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r');tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);cat=categories(tok)
    train=[r['doc_index'] for r in docs if r['split']=='discovery'];positions=np.arange(32,1024,8)
    x0=np.concatenate([prefix_features(raw[i,:1024],cat)[positions] for i in train]);mean=x0.mean(0);scale=np.maximum(x0.std(0),.01);x=(x0-mean)/scale
    ids=np.concatenate([raw[i,positions] for i in train]);pos=np.tile(positions,len(train));gram=x.T@x+100*np.eye(x.shape[1]);rng=np.random.default_rng(P.SEED)
    perm=np.arange(len(ids))
    for token in np.unique(ids):
        ix=np.flatnonzero(ids==token);perm[ix]=rng.permutation(ix)
    pair_x={};coins={}
    for split in ['confirm','lockbox']:
        pp=data['pairs'][split];pair_x[split]=[]
        for which in ['a','b']:
            pair_x[split].append(np.stack([prefix_features(raw[p[which],:1024],cat)[p['position']] for p in pp]))
        coins[split]={k:rng.integers(0,2,(len(pp),len(units))).astype(bool) for k in ['base','prediction_tie','truth_tie','null_tie']}
    rows=[];predictions={s:np.zeros((256,2,len(units)),np.float32) for s in pair_x};truths={s:np.zeros((256,2,len(units)),np.float32) for s in pair_x};nulls={s:np.zeros((256,2,len(units)),np.float32) for s in pair_x}
    # Fit discovery, evaluate confirm only; lockbox values are not accessed yet.
    for l in range(36):
        indexes=[i for i,r in enumerate(units) if r['layer']==l];layer_units=[units[i]['unit'] for i in indexes]
        a=np.load(P.OUT/f'L{l:02d}-A.npy',mmap_mode='r');y=unbits(np.asarray(a[train][:,positions])).reshape(-1,len(indexes))
        means,pm,coef,null=fits(y,ids,pos,x,gram,perm)
        np.savez(P.OUT/f'predictor-L{l:02d}.npz',means=means,position_mean=pm,coef=coef,null_coef=null,feature_mean=mean,feature_scale=scale)
        for split in ['confirm']:
            pp=data['pairs'][split]
            for side,key in enumerate(['a','b']):
                px=(pair_x[split][side]-mean)/scale;baseline=np.stack([means[p['token_id']]+pm[p['position']//64] for p in pp])
                predictions[split][:,side,indexes]=baseline+px@coef;nulls[split][:,side,indexes]=baseline+px@null
                truths[split][:,side,indexes]=np.stack([unbits(a[p[key],p['position']]) for p in pp])
        del a,y
    def metrics(split):
        pp=data['pairs'][split];v=truths[split];pred=predictions[split];null=nulls[split];result=[]
        for j,u in enumerate(units):
            truth=np.where(v[:,0,j]!=v[:,1,j],v[:,0,j]>v[:,1,j],coins[split]['truth_tie'][:,j]);base=coins[split]['base'][:,j]
            pick=np.where(pred[:,0,j]!=pred[:,1,j],pred[:,0,j]>pred[:,1,j],coins[split]['prediction_tie'][:,j])
            npick=np.where(null[:,0,j]!=null[:,1,j],null[:,0,j]>null[:,1,j],coins[split]['null_tie'][:,j])
            m=mcnemar(base==truth,pick==truth);neg=mcnemar(base==truth,npick==truth)
            m['nonzero_pair_fraction']=float(np.mean(v[:,0,j]!=v[:,1,j]));m['passed']=bool(m['delta']>=.15 and m['p']<.05/432 and m['nonzero_pair_fraction']>=.8)
            neg['passed']=bool(neg['delta']>=.15 and neg['p']<.05/432)
            result.append({**u,'metrics':m,'shuffle_control':neg})
        return result
    confirm=metrics('confirm');stats=np.load(P.OUT/'population-statistics.npz');write=np.load(P.OUT/'write-norm.npy');group=[];random=[];selected_layers=set();skipped=[]
    for row in sorted([r for r in confirm if r['selection']=='stratified_random'],key=lambda r:-r['metrics']['delta']):
        l,u=row['layer'],row['unit']
        if l in selected_layers:continue
        rms=np.sqrt(stats['squares'][0,l,2]/stats['count'][0]);den=max(rms[u],1e-9);wn=write[l];energy=rms*wn
        candidates=[v for v in range(9728) if v!=u and .8<=rms[v]/den<=1.25 and .8<=wn[v]/max(wn[u],1e-9)<=1.25 and .8<=energy[v]/max(energy[u],1e-9)<=1.25]
        if not candidates:skipped.append([l,u]);continue
        ru=int(rng.choice(candidates));group.append([l,u]);random.append([l,ru]);selected_layers.add(l)
        if len(group)==16:break
    assert len(group)==16
    selection={'kind':'real','run_id':P.RUN,'units':group,'random_units':random,'selected_only':'confirm ranking gains; oneunit per distinctlayer; lockbox tensors not read before this file frozen',
        'matching':'A_RMS,down_column_norm andtheirproduct ratios each in[.8,1.25],same layer','skipped_no_strict_match':skipped,'provenance':collection['provenance']}
    P.save(P.OUT/'selection.json',selection)
    for l in range(36):
        indexes=[i for i,r in enumerate(units) if r['layer']==l];a=np.load(P.OUT/f'L{l:02d}-A.npy',mmap_mode='r');p=np.load(P.OUT/f'predictor-L{l:02d}.npz');pp=data['pairs']['lockbox']
        for side,key in enumerate(['a','b']):
            px=(pair_x['lockbox'][side]-mean)/scale;baseline=np.stack([p['means'][p0['token_id']]+p['position_mean'][p0['position']//64] for p0 in pp])
            predictions['lockbox'][:,side,indexes]=baseline+px@p['coef'];nulls['lockbox'][:,side,indexes]=baseline+px@p['null_coef'];truths['lockbox'][:,side,indexes]=np.stack([unbits(a[p0[key],p0['position']]) for p0 in pp])
    lock=metrics('lockbox');cards=[]
    for j,(c,k) in enumerate(zip(confirm,lock)):
        l,u=c['layer'],c['unit'];pp=np.load(P.OUT/f'predictor-L{l:02d}.npz');sub=[r['unit'] for r in units if r['layer']==l].index(u);coef=pp['coef'][:,sub]
        top=np.argsort(-np.abs(coef))[:5];pole={}
        for fold in [0,1,2]:
            for key in ['Gpos','Gneg','Apos','Aneg']:
                z=np.load(P.OUT/f'pool-{fold}-L{l:02d}-{key}.npz');values=z['value'][:,sub];order=np.argsort(values if key.endswith('neg') else -values)
                h=unbits(z['h'][order[:32],sub]);mu=stats['hsum'][fold,l]/stats['count'][fold];center=h-mu
                norm=np.linalg.norm(center,axis=1);direction=(center/np.maximum(norm[:,None],1e-9)).mean(0)
                pole[f'{fold}:{key}']={'coherence':float(np.linalg.norm(direction)),'examples':[{'doc_index':int(z['doc'][t,sub]),'position':int(z['position'][t,sub]),'value':float(values[t])} for t in order[:8]]}
        params=np.load(P.OUT/f'parameters-L{l:02d}.npz');gp=np.load(P.OUT/f'pool-0-L{l:02d}-Gpos.npz');gn=np.load(P.OUT/f'pool-0-L{l:02d}-Gneg.npz')
        delta=unbits(gp['h'][:,sub]).mean(0)-unbits(gn['h'][:,sub]).mean(0)
        def cosine(x):return float(x@delta/max(np.linalg.norm(x)*np.linalg.norm(delta),1e-12))
        alignment={key:cosine(unbits(params[key][sub])) for key in ['gate','up','down']}
        errors=[]
        split='lockbox';v=truths[split];prv=predictions[split]
        for i,pair in enumerate(data['pairs'][split]):
            if v[i,0,j]!=v[i,1,j] and (prv[i,0,j]>prv[i,1,j])!=(v[i,0,j]>v[i,1,j]):
                errors.append({'pair':pair,'actual':v[i,:,j].tolist(),'prediction':prv[i,:,j].tolist()})
        cards.append({**c,'confirm':c['metrics'],'lockbox':k['metrics'],'replicated':bool(c['metrics']['passed'] and k['metrics']['passed']),
            'shuffle_confirm':c['shuffle_control'],'shuffle_lockbox':k['shuffle_control'],'description_status':'operational prefix-feature predictor; semantic function not independently labelled',
            'prefix_predictor_terms':[{'feature':FEATURES[t],'coefficient':float(coef[t])} for t in top], 'read_write_alignment':alignment,
            'alignment_note':'raw pre-normalization h pole direction, descriptive only; read/write mismatch is not a failed interpretation criterion',
            'poles':pole,'counterexamples':errors[:12],'all_error_count':len(errors)})
    np.savez(P.OUT/'paired-predictions.npz',**{f'{s}_{n}':d[s] for s in ['confirm','lockbox'] for n,d in [('prediction',predictions),('actual',truths),('shuffle',nulls)]})
    result={'kind':'real','run_id':P.RUN,'model_snapshot':collection['model_snapshot'],'cards':cards,'random_units':432,
        'random_replicated':sum(r['replicated'] for r in cards if r['selection']=='stratified_random'),
        'random_shuffle_double_pass':sum(r['shuffle_confirm']['passed'] and r['shuffle_lockbox']['passed'] for r in cards if r['selection']=='stratified_random'),
        'feature_names':FEATURES,'selection':selection,'provenance':collection['provenance']}
    P.save(P.OUT/'analysis.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['cards','feature_names','provenance','selection']},ensure_ascii=False))
if __name__=='__main__':main()
