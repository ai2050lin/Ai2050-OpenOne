"""Full-coordinate kernel ridge; cache input Gram, tune only validation, never target-derived inputs."""
from rdc_conditional_common import *
RIDGES=(1e-5,.001,.1,1.,10.)


def group_ids(rows,mode):
    if mode=='family_language':return np.array([2*r['family_index']+(r['language']=='zh') for r in rows],int)
    if mode=='layout_query':return np.array([8*r['form']+4*r['style']+2*(r['language']=='zh')+int(r['negative_query']) for r in rows],int)
    if mode=='hash16':return np.array([int(hashlib.sha256(r['prompt'].encode()).hexdigest()[:8],16)%16 for r in rows],int)
    if mode=='language':return np.array([r['language']=='zh' for r in rows],int)
    if mode=='global':return np.zeros(len(rows),int)
    raise ValueError(mode)


class FullKernel:
    def __init__(self,x,train,val,test,kind='linear',groups=None,mix=1.):
        self.train,self.val,self.test=[np.asarray(i,int) for i in (train,val,test)]
        assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
        self.kind=kind;self.groups=groups;self.mix=mix
        self.scale=max(float(np.sqrt(np.mean(np.sum(np.asarray(x[train],np.float64)**2,axis=1)))),1e-12)
        self.z=np.asarray(x,np.float64)/self.scale
        k=self.kernel(train,train);self.e,self.q=np.linalg.eigh((k+k.T)*.5)
        assert self.e.min()>-1e-6*max(1,float(self.e.max()))
        self.e=np.maximum(self.e,0)
        self.v=self.kernel(val,train);self.t=self.kernel(test,train)
        self.vq=self.v@self.q;self.tq=self.t@self.q

    def kernel(self,left,right):
        k=1+self.z[left]@self.z[right].T
        if self.kind=='quadratic':k=k*k
        elif self.kind in ('conditional','independent'):
            same=(self.groups[left,None]==self.groups[None,right]).astype(float)
            k*=same if self.kind=='independent' else (1+self.mix*same)/(1+self.mix)
        elif self.kind!='linear':raise ValueError(self.kind)
        return k

    def fit(self,y,output=None,chunk=256):
        y=np.asarray(y);y=y[:,None] if y.ndim==1 else y
        losses=np.zeros(len(RIDGES));qt=self.q.T
        # Small output blocks bound RAM without deleting low-valued output dimensions.
        for start in range(0,y.shape[1],chunk):
            stop=min(start+chunk,y.shape[1]);qty=qt@y[self.train,start:stop]
            for j,ridge in enumerate(RIDGES):
                err=self.vq@(qty/(self.e[:,None]+ridge))-y[self.val,start:stop]
                losses[j]+=np.square(err).sum()
        losses/=len(self.val)*y.shape[1];best=min(range(len(RIDGES)),key=lambda i:(losses[i],-RIDGES[i]));ridge=RIDGES[best]
        pred=np.empty((len(self.test),y.shape[1]),np.float32);alpha=np.empty((len(self.train),y.shape[1]),np.float32) if output else None
        for start in range(0,y.shape[1],chunk):
            stop=min(start+chunk,y.shape[1]);spectral=(qt@y[self.train,start:stop])/(self.e[:,None]+ridge)
            pred[:,start:stop]=self.tq@spectral
            if alpha is not None:alpha[:,start:stop]=self.q@spectral
        if output:npz(output,alpha=alpha,train=self.train,test=self.test,scale=np.array(self.scale),ridge=np.array(ridge))
        return pred,{'ridge':ridge,'validation_mse':float(losses[best]),'validation_grid':dict(zip(map(str,RIDGES),losses.tolist())),
          'effective_degrees_of_freedom':float(np.sum(self.e/(self.e+ridge))),'input_coordinates':self.z.shape[1],
          'mse':float(np.mean((pred.astype(np.float64)-y[self.test])**2)),'n':len(self.test)}


def binary_report(y,p,indices,rows):
    y=np.asarray(y)[indices];p=np.asarray(p);correct=(p>.5)==y
    result={'n':len(indices),'mse':float(np.mean((p-y)**2)),'correct':correct.sum(0).tolist(),'accuracy':correct.mean(0).tolist()}
    for key in ('family','language','form','style'):
        result['by_'+key]={str(v):{'n':int(mask.sum()),'correct':correct[mask].sum(0).tolist(),'mse':float(np.mean((p[mask]-y[mask])**2))} for v in sorted({r[key] for r in rows},key=str) if (mask:=np.array([rows[i][key]==v for i in indices])).any()}
    return result
