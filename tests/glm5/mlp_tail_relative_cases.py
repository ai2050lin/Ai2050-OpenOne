"""Retrospective stress test of the largest reproducible archived pair per boundary."""

import numpy as np
from scipy.stats import spearmanr

import mlp_tail_relative_encoding as study
from shared.dist_protocol.mechanism_trace import file_sha, unbits


def main():
    report_path = study.OUTPUT/'report.json'
    report = study.atlas.read(report_path)
    examples = {}
    for block in range(28,35):
        boundary = f'B{block}-B{block+1}'
        with np.load(study.OUTPUT/f'{boundary}.npz') as arrays:
            pairs = arrays['discovery_peak_full']
        same_sign = (pairs[:,2]*pairs[:,3]>0) & (pairs[:,2]*pairs[:,4]>0)
        pair = pairs[np.argmax(np.where(same_sign, abs(pairs[:,2:]).min(1),-1))]
        source, target = map(int,pair[:2])
        x, y = (unbits(np.load(study.directory(b)/f'B{b}-A.npy', mmap_mode='r')[:,u]).astype(float)
                for b,u in ((block,source),(block+1,target)))
        xd,yd = x[study.DISC],y[study.DISC]
        xh,yh = x[study.HOLD],y[study.HOLD]
        slope = ((xd-xd.mean())*(yd-yd.mean())).sum()/np.square(xd-xd.mean()).sum()
        prediction = yd.mean()+slope*(xh-xd.mean())
        clipped_x,clipped_y = (np.clip(v,*np.quantile(v,[.01,.99])) for v in (xh,yh))
        doc_energy = np.square(xh-xh.mean()).reshape(-1,16).sum(1)
        largest_docs = np.argsort(-doc_energy)[:5]
        keep = ~np.isin(np.arange(len(xh))//16,largest_docs)
        alignments = {}
        for field in ('gate','up'):
            path = study.directory(block)/f'{boundary}-weight-{field}.npy'
            alignments[field] = float(np.load(path,mmap_mode='r')[source,target])
        examples[boundary] = {'source_unit':source,'target_unit':target,'r_by_split':pair[2:].tolist(),
                              'holdout_spearman':float(spearmanr(xh,yh).statistic),
                              'holdout_winsorized_r':float(np.corrcoef(clipped_x,clipped_y)[0,1]),
                              'holdout_r_without_five_highest_source_energy_docs':float(np.corrcoef(xh[keep],yh[keep])[0,1]),
                              'fraction_source_centered_energy_in_five_docs':float(doc_energy[largest_docs].sum()/doc_energy.sum()),
                              'linear_slope_discovery':float(slope),
                              'linear_holdout_fvu':float(np.square(yh-prediction).sum()/np.square(yh-yd.mean()).sum()),
                              'static_weight_alignment':alignments}
    study.atlas.save_json(study.OUTPUT/'retrospective_cases.json',{
        'provenance':report['provenance'],'source_report_sha256':file_sha(report_path),
        'source_code_sha256':file_sha(__file__),
        'scope':'Pairs chosen by best minimum absolute r across all exposed splits, same sign required. Case diagnostics only, no unbiased predictive performance or causal claim.',
        'examples':examples})
    print(examples,flush=True)


if __name__ == '__main__':
    main()
