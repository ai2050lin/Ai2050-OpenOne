"""Memory-bounded pole statistics, preserving the frozen estimator and native indices."""
import gc
import time
import numpy as np
import mlp_tail_goodfire as C
from shared.dist_protocol.mechanism_trace import unbits


def main():
    from transformers import AutoTokenizer
    started=time.monotonic();tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    docs=C.read(C.BANK/'source-documents.json')['docs'];positions=C.read(C.OUT/'plan.json')['positions']
    tokens=np.load(C.TOKENS,mmap_mode='r');ids=np.asarray(tokens[:,positions]).reshape(-1)
    classes=[]
    decoded={int(i):tok.decode([int(i)]) for i in np.unique(ids)}
    for id in ids:
        text=decoded[int(id)]
        classes.append([any('\u4e00'<=c<='\u9fff' for c in text),any(c.isdigit() for c in text),
                        any(c in '，。！？；：,.!?;:' for c in text),'\n' in text,any(c in '{}[]();=<>`' for c in text)])
    classes=np.asarray(classes,np.float32)
    np.save(C.OUT/'token_ids.npy',ids);np.save(C.OUT/'token_classes.npy',classes)
    C.save_json(C.OUT/'documents.json',{'docs':[{**doc,'tokens':tokens[i,:1024].tolist()} for i,doc in enumerate(docs)],
                'positions':positions,'token_strings':decoded,'scope':'原始1024-token文档；示例只用当前位置及以前解释响应，后文单独标记。'})
    C.save_json(C.OUT/'profile-implementation.json',{'schema':'mlp_tail_implementation.v1','kind':'real','run_id':C.RUN,
                'original_frozen_source_sha256':C.read(C.OUT/'plan.json')['source_code_sha256'],
                'source_sha256':C.file_sha(__file__),'reason':'Available RAM约7.7GiB，按256单元分块计算，统计式/划分/阈值不变；开始前记录。'})
    splits={'discovery':(0,16384),'confirm':(16384,24576),'holdout':(24576,32768)}
    for bi,block in enumerate(C.BLOCKS):
        profiles={f:{'mean':[],'std':[],'min':[],'max':[],**{s:{'positive_indices':[],'negative_indices':[]} for s in splits}} for f in ['G','V','A']}
        cues={s:[] for s in splits};regimes=[]
        maps={f:np.load(C.OUT/f'B{block}-{f}.npy',mmap_mode='r') for f in profiles}
        for lo in range(0,C.WIDTH,256):
            hi=min(C.WIDTH,lo+256);g=unbits(maps['G'][:,lo:hi]);v=unbits(maps['V'][:,lo:hi])
            regimes.extend(np.stack([((g>=0)&(v>=0)).mean(0),((g>=0)&(v<0)).mean(0),((g<0)&(v>=0)).mean(0),((g<0)&(v<0)).mean(0)],1).tolist())
            for split,(a,b) in splits.items():
                cx=classes[a:b]-classes[a:b].mean(0);z=g[a:b]-g[a:b].mean(0)
                den=np.sqrt(np.square(cx,dtype=np.float64).sum(0)[:,None]*np.square(z,dtype=np.float64).sum(0)[None,:])
                cues[split].extend(np.divide(cx.T@z,den,out=np.zeros((5,hi-lo)),where=den>1e-12).T.tolist())
            del v,g
            for field in ['G','V','A']:
                x=unbits(maps[field][:,lo:hi]);p=profiles[field]
                for key,val in [('mean',x.mean(0,dtype=np.float64)),('std',x.std(0,dtype=np.float64)),('min',x.min(0)),('max',x.max(0))]:p[key].extend(val.tolist())
                for split,(a,b) in splits.items():
                    z=x[a:b];col=np.arange(hi-lo)[None,:]
                    high=np.argpartition(z,-8,axis=0)[-8:];low=np.argpartition(z,8,axis=0)[:8]
                    high=np.take_along_axis(high,np.argsort(z[high,col],axis=0)[::-1],axis=0)
                    low=np.take_along_axis(low,np.argsort(z[low,col],axis=0),axis=0)
                    p[split]['positive_indices'].extend((high+a).T.tolist());p[split]['negative_indices'].extend((low+a).T.tolist())
                del x
            C.log('profiles',(bi*C.WIDTH)+hi,4*C.WIDTH,started,block=block)
        C.save_json(C.OUT/f'B{block}-profile.json',{'schema':'mlp_tail_layer.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
             'block':block,'width':C.WIDTH,'fields':profiles,'sign_regime_fractions':regimes,'cue_correlations':cues,
             'cue_names':['当前token汉字','数字字符','标点字符','换行','代码符号'],
             'label_status':'candidate_token_cues_only_not_semantic_explanations',
             'scope':'全部9728原生单元；归一化后的MLP输入。token相关不是上下文语义或单义性。原始索引不重排。'})
        del maps,profiles,cues,regimes;gc.collect()


if __name__=='__main__':main()
