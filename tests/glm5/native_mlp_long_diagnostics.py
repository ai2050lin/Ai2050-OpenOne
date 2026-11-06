"""Post-hoc audit only: frozen predictor residuals and native read equations."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from shared.dist_protocol.mechanism_trace import unbits,file_sha
def main():
    import torch
    data=P.read(P.OUT/'documents.json');a=P.read(P.OUT/'analysis.json');pred=np.load(P.OUT/'paired-predictions.npz');rows=[];reconstruction=[]
    for l in range(36):
        ix=[i for i,u in enumerate(data['units']) if u['layer']==l];pp=np.load(P.OUT/f'predictor-L{l:02d}.npz');param=np.load(P.OUT/f'parameters-L{l:02d}.npz')
        fields={k:np.load(P.OUT/f'L{l:02d}-{k}.npy',mmap_mode='r') for k in ['G','V','A']};gamma=torch.from_numpy(unbits(param['gamma']))
        unit_errors={j:{'G':[],'V':[],'A':[]} for j in range(len(ix))}
        for fold in [0,1,2]:
            for pole in ['Gpos','Gneg','Apos','Aneg']:
                z=np.load(P.OUT/f'pool-{fold}-L{l:02d}-{pole}.npz');h=torch.from_numpy(unbits(z['h']));rms=(h.square().mean(-1,keepdim=True)+1e-6).sqrt()
                normalized=(h/rms).to(torch.bfloat16)*gamma.to(torch.bfloat16);normalized=normalized.float().numpy()
                g=np.einsum('kud,ud->ku',normalized,unbits(param['gate']));v=np.einsum('kud,ud->ku',normalized,unbits(param['up']))
                for j,ii in enumerate(ix):
                    di=z['doc'][:,j];pos=z['position'][:,j];actual={k:unbits(vv[di,pos,j]) for k,vv in fields.items()}
                    ag=torch.from_numpy(actual['G']).to(torch.bfloat16);av=torch.from_numpy(actual['V']).to(torch.bfloat16)
                    product=(torch.nn.functional.silu(ag)*av).float().numpy()
                    for k,prediction in [('G',g[:,j]),('V',v[:,j]),('A',product)]:
                        unit_errors[j][k].extend((prediction-actual[k]).tolist())
        for j,ii in enumerate(ix):
            errors=unit_errors[j];reconstruction.append({'layer':l,'unit':data['units'][ii]['unit'],'n':len(errors['G']),
                'G_RMSE':float(np.sqrt(np.mean(np.square(errors['G'])))),'V_RMSE':float(np.sqrt(np.mean(np.square(errors['V'])))),
                'A_max_abs':float(np.max(np.abs(errors['A']))),'A_exact_fraction':float(np.mean(np.asarray(errors['A'])==0))})
        for split in ['confirm','lockbox']:
            baseline=np.stack([pp['means'][p['token_id']]+pp['position_mean'][p['position']//64] for p in data['pairs'][split]])
            truth=pred[split+'_actual'][:,:,ix];prediction=pred[split+'_prediction'][:,:,ix];shuffle=pred[split+'_shuffle'][:,:,ix]
            mse0=((truth-baseline[:,None])**2).mean((0,1));mse=((truth-prediction)**2).mean((0,1));msen=((truth-shuffle)**2).mean((0,1))
            for j,ii in enumerate(ix):rows.append({'layer':l,'unit':data['units'][ii]['unit'],'selection':data['units'][ii]['selection'],'split':split,
                'baseline_MSE':float(mse0[j]),'context_MSE':float(mse[j]),'shuffle_MSE':float(msen[j]),'relative_MSE_reduction':float(1-mse[j]/max(1e-12,mse0[j]))})
    d={'kind':'real','run_id':P.RUN,'scope':'Post-hoc frozen-output diagnostics; no threshold or primary verdict changed; true h read-equation checks are not a graph prediction benchmark',
        'reader_formula':'RMSNorm(h)*gamma dotted with actual gate/up rows; BF16 casting reproduced, FP32 sum audited; rms_norm_eps=1e-6',
        'rows':rows,'reconstruction':reconstruction,'A_exact_fraction':float(np.mean([r['A_exact_fraction'] for r in reconstruction])),
        'A_max_abs':max(r['A_max_abs'] for r in reconstruction),'provenance':a['provenance'],'script_sha256':file_sha(__file__)}
    P.save(P.OUT/'diagnostics.json',d);print(json.dumps({k:d[k] for k in ['A_exact_fraction','A_max_abs','scope']}))
if __name__=='__main__':main()
