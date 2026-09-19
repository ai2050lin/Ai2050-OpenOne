"""Shared prefix kernels and full-coordinate targets. All scaling is fit on training only."""
import hashlib
from scipy.linalg import eigh
from rdc_prefix_common import *

RIDGES=(1e-5,.001,.1,1.,10.)
KERNELS=('early_linear','embedding_history','full_linear','full_quadratic','graph_only','graph_additive','graph_interaction','hash_interaction')
TARGET_LAYERS=(23,24,36)


def descriptor(graph):
    """All typed event transitions and same-clause co-presence, not semantic parsing."""
    names=list(LEXICONS);m=len(names);order=np.zeros((m,m));co=np.zeros((m,m));last=np.zeros(m)
    events=graph['events'];clauses=[[]]
    for i,e in enumerate(events):
        k=names.index(e['type']);last[k]=(e['span'][1]+1)/(len(graph['observed_prefix'])+1)
        clauses[-1].append(k)
        if i:order[names.index(events[i-1]['type']),k]+=1
        if e['type']=='boundary':clauses.append([])
    for clause in clauses:
        count=np.bincount(clause,minlength=m).astype(float);co+=np.outer(count,count)
    x=np.concatenate([np.log1p(graph['features']),np.log1p(order.ravel()),np.log1p(co.ravel()),last])
    return x.astype(np.float32)


def build_features(run='qwen4',overwrite=False):
    out=CAMPAIGN/'shared_rules'/run
    if (out/'features.npz').exists() and not overwrite:return read(out/'rows.json')
    source=CAMPAIGN/run
    ids=sorted(p.stem for p in (source/'commits').glob('*.json'))
    rows=[];pack={k:[] for k in ('h0','h12','history12','h23','h24','h36','next_h0','next_h12','next_h36','graph','next_graph','hash_graph','native_attention')}
    for i,sid in enumerate(ids):
        r=read(source/f'rows/{sid}.json')
        with np.load(source/f'fields/{sid}.npz') as z:
            h=unbits(z['h']);history=z['H12_prefix_mean']
            for k in (0,3):
                graph=r['actual_anchor_graphs'][k];v=descriptor(graph);nextgraph=descriptor(r['actual_anchor_graphs'][k+1])
                seed=int(hashlib.sha256(graph['observed_prefix'].encode()).hexdigest()[:16],16)
                noise=np.random.default_rng(seed).standard_normal(len(v)).astype(np.float32)
                p=r['positions'][k]
                for name,x in [('h0',h[0,k]),('h12',h[12,k]),('history12',history[k]),('h23',h[23,k]),('h24',h[24,k]),('h36',h[36,k]),
                    ('next_h0',h[0,k+1]),('next_h12',h[12,k+1]),('next_h36',h[36,k+1]),('graph',v),('next_graph',nextgraph),('hash_graph',noise),
                    ('native_attention',unbits(z['L23_attention_out'][k]))]:pack[name].append(x.copy())
                rows.append({key:r[key] for key in ('sample_id','source_group','source_sentence_id','language','genre','split')}|{
                  'anchor':k//3,'anchor_array_index':k,'position':p,'next_position':r['positions'][k+1],
                  'token_id':r['prompt_ids'][p],'next_observed_token_id':r['prompt_ids'][p+1],
                  'char_endpoint':r['token_offsets'][p][1],'prefix':graph['observed_prefix'],
                  'new_token_available_for_temporal_prediction':r['prompt_ids'][p+1],
                  'field_path':str((source/f'fields/{sid}.npz').relative_to(CAMPAIGN)),
                  'graph_scope':'visible prefix cues, typed observed order and same-clause co-presence; no UD labels'})
        if i%64==63:print('FEATURES',run,i+1,len(ids),flush=True)
    npz(out/'features.npz',**{k:np.stack(v).astype(np.float32) for k,v in pack.items()})
    save(out/'rows.json',rows)
    return rows


def splits(rows):
    return [np.array([i for i,r in enumerate(rows) if r['split']==s],int) for s in ('train','validation','test')]


class KernelBank:
    def __init__(self,data,train,scales=None,temporal=False):
        self.temporal=temporal
        self.names=('h12','history12','h0','graph','hash_graph') if not temporal else ('h36','history12','next_h0','next_graph','hash_graph')
        self.raw={k:np.asarray(data[k],np.float64) for k in self.names}
        self.scales={};self.z={}
        for k,x in self.raw.items():
            if scales is None:
                if k in ('graph','next_graph'):
                    mean=x[train].mean(0);std=x[train].std(0);std=np.maximum(std,.1)
                    xx=(x-mean)/std;scale=max(float(np.sqrt(np.sum(xx[train]**2,1).mean())),1e-12)
                else:mean=np.zeros(x.shape[1]);std=np.ones(x.shape[1]);scale=max(float(np.sqrt(np.sum(x[train]**2,1).mean())),1e-12)
                self.scales[k]={'mean':mean,'std':std,'scale':scale}
            else:self.scales[k]={q:np.asarray(v) if q!='scale' else float(v) for q,v in scales[k].items()}
            s=self.scales[k];self.z[k]=(x-s['mean'])/s['std']/s['scale']

    def gram(self,name,left,right,other=None):
        other=self if other is None else other
        a,b,c,g,n=self.names
        dots={k:self.z[k][left]@other.z[k][right].T for k in self.names}
        base=(dots[a]+dots[b]+dots[c])/3
        if name=='early_linear':return 1+dots[a]
        if name=='embedding_history':return 1+(dots[b]+dots[c])/2
        if name=='full_linear':return 1+base
        if name=='full_quadratic':return (1+base)**2
        if name=='graph_only':return 1+dots[g]
        if name=='graph_additive':return 1+(3*base+dots[g])/4
        if name=='graph_interaction':return (1+base)*(1+dots[g])
        if name=='hash_interaction':return (1+base)*(1+dots[n])
        raise ValueError(name)

    def serial_scales(self):
        return {k:{p:float(v) if p=='scale' else v.tolist() for p,v in s.items()} for k,s in self.scales.items()}


def fit(gram,train,val,test,y,target_blocks,output,fixed_df=None):
    y=np.asarray(y,np.float64);means=y[train].mean(0);scales=np.empty(y.shape[1])
    for start,stop in target_blocks:
        scales[start:stop]=max(float(np.sqrt(np.mean((y[train,start:stop]-means[start:stop])**2))),1e-9)
    normalized=(y-means)/scales
    e,q=eigh((gram[np.ix_(train,train)]+gram[np.ix_(train,train)].T)/2,check_finite=False)
    assert e.min()>-1e-6*max(float(e.max()),1.);e=np.maximum(e,0)
    vq=gram[np.ix_(val,train)]@q;tq=gram[np.ix_(test,train)]@q
    qty=q.T@normalized[train];loss=[]
    for ridge in RIDGES:
        p=vq@(qty/(e[:,None]+ridge));loss.append(float(np.mean((p-normalized[val])**2)))
    ridge=RIDGES[min(range(len(RIDGES)),key=lambda i:(loss[i],-RIDGES[i]))]
    if fixed_df is not None:
        lo,hi=-24.,24.
        for _ in range(80):
            mid=(lo+hi)/2
            if np.sum(e/(e+np.exp(mid)))>fixed_df:lo=mid
            else:hi=mid
        ridge=float(np.exp((lo+hi)/2))
    spectral=qty/(e[:,None]+ridge);prediction=tq@spectral*scales+means
    validation=vq@spectral*scales+means
    npz(output,alpha=(q@spectral).astype(np.float32),means=means.astype(np.float32),target_scales=scales.astype(np.float32),
      train=train,ridge=np.array(ridge),eigenvalues=e.astype(np.float32))
    return prediction.astype(np.float32),validation.astype(np.float32),{'ridge':ridge,'fixed_effective_df':fixed_df,
      'effective_df':float(np.sum(e/(e+ridge))),'normalized_validation_mse':float(np.mean(((validation-y[val])/scales)**2)),
      'validation_ridge_grid':dict(zip(map(str,RIDGES),loss)), 'target_dimensions':y.shape[1]}


def errors(target,prediction,train_target,rows):
    target=np.asarray(target,np.float64);prediction=np.asarray(prediction,np.float64)
    err=np.square(prediction-target);mse=err.mean(1);energy=np.mean(np.asarray(train_target,np.float64)**2,0)
    order=np.argsort(energy,kind='stable');quartile=np.empty(len(order),int)
    for b,ii in enumerate(np.array_split(order,4)):quartile[ii]=b
    report={'n':len(target),'mse':float(err.mean()),'target_energy':float(np.mean(target**2)),
      'relative_mse':float(err.mean()/max(np.mean(target**2),1e-30)),
      'all_coordinate_energy_quartiles':[{'quartile':b,'coordinates':int((quartile==b).sum()),
        'mse':float(err[:,quartile==b].mean()),'target_energy':float(np.mean(target[:,quartile==b]**2))} for b in range(4)]}
    for key in ('language','genre','source_group'):
        report['by_'+key]={v:{'n':int(mask.sum()),'mse':float(mse[mask].mean())}
          for v in sorted({r[key] for r in rows}) if (mask:=np.array([r[key]==v for r in rows])).any()}
    return report,{'coordinate_mse':err.mean(0).astype(np.float32),'row_mse':mse.astype(np.float32),'train_energy_quartile':quartile}
