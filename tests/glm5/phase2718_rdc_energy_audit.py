"""Complete-coordinate low-energy coverage and context from primary literature abstracts."""
from rdc_relation_common import *


def main():
    out=BASE/'energy_audit';guard(200*1024);ys=[]
    for r in rows():
        if r['split']=='train':ys.append(unbits(load_field(r)['h23'][0]))
    energy=np.mean(np.stack(ys).astype(float)**2,axis=0);order=np.argsort(energy);groups=np.empty(2560,int)
    for i,idx in enumerate(np.array_split(order,4)):groups[idx]=i
    with np.load(BASE/'boundary_compilation/first_source_affine.npz') as z:a=z['a'];b=z['b']
    em=[];ei=[];target=[]
    for r in rows(True):
        z=load_field(r,True);x=unbits(z['h12'][0]);y=unbits(z['h23'][0]);em.append((x.astype(float)*a+b-y)**2);ei.append((x.astype(float)-y)**2);target.append(y.astype(float)**2)
    em=np.mean(em,0);ei=np.mean(ei,0);target=np.mean(target,0);first=[]
    for q in range(4):
        mask=groups==q;first.append({'training_energy_quartile':q,'coordinates':int(mask.sum()),'fresh_actual_energy':float(target[mask].mean()),'identity_MSE':float(ei[mask].mean()),'affine_MSE':float(em[mask].mean()),'affine_error_over_actual_energy':float(em[mask].sum()/target[mask].sum())})
    npz(out/'first_boundary_coordinate_errors.npz',training_energy=energy,training_energy_quartile=groups,identity_MSE=ei,affine_MSE=em,fresh_target_energy=target)
    with np.load(BASE/'rules/current/test_h36_errors.npz') as z:partition=z['train_energy_quartile']
    attribution={}
    for scope in ('current','temporal','self_generation'):
        with np.load(BASE/f'output_geometry/{scope}_coordinate_profiles.npz') as z:p={k:z[k] for k in z.files}
        attribution[scope]=[{'training_H36_energy_quartile':q,'coordinates':int((partition==q).sum()),'mean_absolute_path_attribution_fraction':float(p['absolute'][partition==q].sum()/p['absolute'].sum()),'mean_signed_path_attribution_sum':float(p['signed'][partition==q].sum())} for q in range(4)]
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'first_boundary_fresh_by_training_energy':first,'output_path_attribution_by_frozen_training_energy':attribution,
      'limits':'Every coordinate remains in fitting/scoring. Quartiles are reporting partitions fixed from training energy, not compression or selected neurons. Path attribution is not an identified semantic or causal contribution; raw-energy rank is not an importance theorem.'})
    save(BASE/'literature_context.json',{'timestamp':stamp(),'checked_scope':'Primary-paper metadata and abstracts retrieved in web search; full-paper methodological review not claimed. Full page opening timed out, so no unseen body claims are used.',
      'references':[{'title':'Massive Activations in Large Language Models','url':'https://arxiv.org/abs/2402.17762','use':'Prior evidence that unusually large activations and their locations have been studied; this local amplitude finding is not a first discovery of the phenomenon.'},
        {'title':'Efficient Streaming Language Models with Attention Sinks','url':'https://arxiv.org/abs/2309.17453','use':'Initial-token attention concentration is an existing topic. Our large hidden norms do not alone establish this attention mechanism.'},
        {'title':'A Refined Analysis of Massive Activations in LLMs','url':'https://arxiv.org/abs/2503.22329','use':'Model and intervention differences caution against universalizing an indispensability or harmfulness claim.'}],
      'local_evidence_boundary':'No attention-sink causal claim, origin-layer localization or invariance across Qwen14/GLM boundary states was tested here.'})
    print('ENERGY_COVERAGE_AUDIT',first,attribution,flush=True)


if __name__=='__main__':main()
