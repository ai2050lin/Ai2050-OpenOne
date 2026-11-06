"""Input-only prefix features and exact matched-pair audit, no future states."""
import numpy as np
CAT=['CJK','Latin','digit','space','punct','newline','comma','period','bracket','quote','operator','byte_fragment']
FEATURES=[f'prior{w}_{c}' for w in [16,64,256,512] for c in CAT]+[f'previous{k}_{c}' for k in range(1,5) for c in CAT]
def categories(tok,size=151936):
    import unicodedata
    x=np.zeros((size,len(CAT)),np.float32)
    for i in range(size):
        t=tok.decode([i]);x[i]=[any('\u4e00'<=c<='\u9fff' for c in t),any(c.isascii() and c.isalpha() for c in t),any(c.isdecimal() for c in t),
            bool(t) and t.isspace(),any(unicodedata.category(c).startswith('P') for c in t),'\n' in t,any(c in ',，、' for c in t),
            any(c in '.。;；:：' for c in t),any(c in '()[]{}（）【】' for c in t),any(c in '\"\'“”‘’' for c in t),
            any(c in '+-*/=<>|&%^' for c in t),'�' in t]
    return x
def prefix_features(ids,cat):
    v=cat[np.asarray(ids)];n=len(v);p=np.arange(n);s=np.vstack([np.zeros((1,len(CAT)),np.float32),np.cumsum(v,axis=0)])
    columns=[(s[p]-s[np.maximum(0,p-w)])/np.maximum(1,np.minimum(p,w))[:,None] for w in [16,64,256,512]]
    columns += [v[np.maximum(0,p-k)]*(p>=k)[:,None] for k in range(1,5)]
    return np.concatenate(columns,axis=1).astype(np.float32)
def mcnemar(base,model):
    from scipy.stats import binomtest
    b=np.asarray(base,bool);m=np.asarray(model,bool);gain=int((~b&m).sum());loss=int((b&~m).sum())
    return {'n':len(b),'baseline_accuracy':float(b.mean()),'accuracy':float(m.mean()),'delta':float(m.mean()-b.mean()),
        'gains':gain,'losses':loss,'p':float(binomtest(gain,gain+loss,.5,alternative='greater').pvalue) if gain+loss else 1.}
