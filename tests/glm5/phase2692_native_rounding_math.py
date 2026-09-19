"""Explicit eager BF16 round points in NumPy, not a native CUDA model."""
import numpy as np

def bf16(x):
    a=np.asarray(x,dtype=np.float32).copy(order='C');assert np.isfinite(a).all()
    u=a.view(np.uint32)
    rounded=(u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)
    return rounded.view(np.float32)

def rms_eager(x,gamma,epsilon):
    x=np.asarray(x,dtype=np.float32);gamma=np.asarray(gamma,dtype=np.float32)
    var=np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)
    reciprocal=np.float32(1.)/np.sqrt(var+np.float32(epsilon))
    return bf16(gamma*bf16(x*reciprocal))

def rotate_half(x):
    a,b=np.split(x,2,axis=-1);return np.concatenate((-b,a),axis=-1)

def rope_eager(x,cos,sin):
    return bf16(bf16(x*cos[:,None,:])+bf16(rotate_half(x)*sin[:,None,:]))

def probability_eager(q,k,mask,scale,positions):
    # BF16 input values accumulated by NumPy FP32 matmul. NOT a CUDA emulator.
    h=q.shape[1];fullk=np.repeat(k,h//k.shape[1],axis=1)
    dot=np.matmul(q[list(positions)].transpose(1,0,2),fullk.transpose(1,2,0)).transpose(1,0,2)
    score=bf16(bf16(dot)*np.float32(scale));m=np.asarray(mask,dtype=np.float32)
    assert ((m==0)|(m<-1e30)).all()
    score=score+m
    ex=np.exp(score-score.max(-1,keepdims=True));p=ex/ex.sum(-1,keepdims=True,dtype=np.float32)
    return bf16(p)

def av_eager(p,v):
    fullv=np.repeat(v,p.shape[1]//v.shape[1],axis=1)
    return bf16(np.matmul(p.transpose(1,0,2),fullv.transpose(1,0,2)).transpose(1,0,2))

def self_test():
    codes=np.concatenate((np.arange(0x7f80,dtype=np.uint32),np.arange(0x8000,0xff80,dtype=np.uint32)))
    values=(codes<<16).view(np.float32)
    assert np.array_equal(bf16(values).view(np.uint32),values.view(np.uint32))
    midpoint_count=0
    for offset in (0,0x8000):
        lower=np.arange(offset,offset+0x7f7f,dtype=np.uint32)
        a=(lower<<16).view(np.float32);b=((lower+1)<<16).view(np.float32)
        midpoint=((a.astype(np.float64)+b.astype(np.float64))/2).astype(np.float32)
        expected=((lower+(lower&1))<<16).view(np.float32)
        assert np.array_equal(bf16(midpoint),expected);midpoint_count+=len(midpoint)
    return {'finite_BF16_fixed_points':len(codes),'ties_to_even_midpoints':midpoint_count,'all_checks_passed':True,'pretrained_model_test':False}

if __name__=='__main__':print(self_test())

