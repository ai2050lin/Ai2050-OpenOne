"""Post-primary exploratory positional diagnosis; no selection or promotion."""
from collections import Counter,defaultdict
import numpy as np
import recursive_branch14 as R


def run():
    R.configure();R.admission();primary=R.read(R.OUT/'response-report.json')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    allrows=R.materials();records=[];summaries=[]
    for split in R.SPLITS[1:]:
        meta=R.read(R.OUT/split/'collection.json');rows=[allrows[i] for i in meta['row_indices']]
        for h in R.CANDIDATES:
            data=np.load(R.OUT/split/f'{h}-all-token-GVA.npy',mmap_mode='r')
            for i,r in enumerate(rows):
                if r['hypothesis']!=h:continue
                lo,hi=r['content_start'],r['content_end']+1
                a=R.unbits(data[i,lo:hi]);gpos=int(np.argmax(a[:,0]));apos=int(np.argmax(np.abs(a[:,2])))
                records.append(dict(id=r['id'],family=r['family'],hypothesis=h,split=split,role=r['role'],lexical=r['lexical'],
                    content_end_G=float(a[-1,0]),body_min_G=float(a[:,0].min()),body_max_G=float(a[gpos,0]),
                    any_body_G_positive=bool((a[:,0]>0).any()),G_peak_position=lo+gpos,
                    G_peak_token_id=r['tokens'][lo+gpos],G_peak_token=tok.decode([r['tokens'][lo+gpos]]),
                    V_at_G_peak=float(a[gpos,1]),A_at_G_peak=float(a[gpos,2]),
                    body_max_abs_A=float(np.abs(a[apos,2])),A_peak_position=lo+apos,
                    scope='Body only; title, future question/options and padding excluded.'))
            del data
            for role in ['operative','quoted','inventory']:
                use=[x for x in records if x['hypothesis']==h and x['split']==split and x['role']==role]
                peaks=Counter(x['G_peak_token'] for x in use)
                summaries.append(dict(hypothesis=h,split=split,role=role,rows=len(use),
                    any_body_G_positive=sum(x['any_body_G_positive'] for x in use),
                    mean_content_end_G=float(np.mean([x['content_end_G'] for x in use])),
                    mean_body_max_G=float(np.mean([x['body_max_G'] for x in use])),
                    mean_body_max_abs_A=float(np.mean([x['body_max_abs_A'] for x in use])),
                    peak_token_counts=peaks.most_common(10)))
    R.save('position-diagnostic.json',dict(kind='post_primary_exploratory_diagnostic',rows=records,summary=summaries,
        primary_report_sha256=R.file_sha(R.OUT/'response-report.json'),source_sha256=R.file_sha(__file__),
        selection_reason='After frozen content-end hypotheses failed, inspect existing all-token candidate fields for transient response. No new forward pass, selector, threshold or intervention change.',
        limitations=['Maxima depend on body length and wording; not a matched semantic comparison','Peak token identity and surrounding lexical context remain alternatives','No new p-values, confirmatory status or training-time claim']))
    for s in summaries:print(s,flush=True)


if __name__=='__main__':run()
