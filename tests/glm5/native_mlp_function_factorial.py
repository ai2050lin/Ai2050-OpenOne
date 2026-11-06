"""Descriptive factorial G/V/A audit; never refits the frozen candidate signs."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from shared.dist_protocol.mechanism_trace import unbits,digest
def main():
    outputs=[]
    for folder in [P.OUT,P.OUT/'place-polarity']:
        c=P.read(folder/('polarity-corpus.json' if folder==P.OUT else 'execution-corpus.json'));result=P.read(folder/'polarity-result.json');selection=P.read(P.OUT/'coordinate-selection.json')['selected']
        if folder!=P.OUT:selection=selection+[{'family':'location','layer':24,'unit':8443,'sign':P.read(P.OUT/'selection.json')['signs']['location'][2]}]
        byid={r['id']:r for r in c['items']};values=np.zeros((128,2,2,2,len(selection),3),np.float32)
        for row in result['rows']:
            item=byid[row['id']];k=int(item['id'].split(':')[1]);root=folder/'polarity-native'/digest(item['id'])[:20]
            for j,u in enumerate(selection):
                with np.load(root/f'block_{u["layer"]:02d}.npz') as z:values[k,item['fact'],item['question_polarity'],:,j]=np.stack([unbits(z[key][:,u['unit']]) for key in ['gate','value','activation']],axis=-1)
        for j,u in enumerate(selection):
            for stage in [0,1]:
                x=values[:,:,:,stage,j];coeff={}
                for name,signs in [('fact',np.array([[-1,-1],[1,1]])),('question',np.array([[-1,1],[-1,1]])),('fact_x_question',np.array([[1,-1],[-1,1]]))]:
                    b=(x*signs[None,:,:,None]).mean((1,2));coeff[name]={'GVA_mean':b.mean(0).tolist(),'GVA_RMS':np.sqrt((b*b).mean(0)).tolist()}
                outputs.append({'diagnostic':'universal' if folder==P.OUT else 'location_name_balanced','layer':u['layer'],'unit':u['unit'],'stage':stage,'coefficients':coeff,
                    'formula':'A(s,q)=a+b_s*s+b_q*q+b_sq*s*q, s/q in{-1,+1}; empirical4-cell contrast, standard factorial decomposition, not a new law or causal proof'})
        np.save(folder/'factorial-GVA.npy',values)
    P.save(P.OUT/'factorial-results.json',{'kind':'real','run_id':P.RUN,'rows':outputs,'provenance':P.read(P.OUT/'observations.json')['provenance'],'status':'posthoc descriptive decomposition, primary verdicts unchanged'})
    print(json.dumps({'unit_stage_comparisons':len(outputs)}))
if __name__=='__main__':main()
