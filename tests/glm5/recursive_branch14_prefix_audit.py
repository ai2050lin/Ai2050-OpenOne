"""Tokenizer continuation and future-token independence on saved pilots."""
import time
import numpy as np
import recursive_branch14 as R


def run():
    R.configure();R.admission();start=time.monotonic()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    rows=R.materials();n=0
    for r in rows:
        assert tok.encode(r['prompt'],add_special_tokens=False)==r['tokens']
        for letter,token in zip('ABC',r['choice_ids']):
            assert tok.encode(r['prompt']+' '+letter,add_special_tokens=False)==r['tokens']+[token],r['id']
            n+=1
    R.save('choice-boundary-audit.json',dict(passed=True,rows=len(rows),choice_continuations=n,
        material_sha256=R.file_sha(R.OUT/'materials.json'),source_sha256=R.file_sha(__file__),
        scope='All three full prompt continuations equal saved prompt tokens plus one choice ID.'))
    old=R.OUT.parent/'recursive_branch14_20261113_v5'
    prior=R.read(old/'materials.json')['rows'];pmeta=R.read(old/'pilot/collection.json');meta=R.read(R.OUT/'pilot/collection.json')
    oldrows=[prior[i] for i in pmeta['row_indices']];newrows=[rows[i] for i in meta['row_indices']]
    lookup={(r['family'],r['role'],r['lexical']):(i,r) for i,r in enumerate(oldrows)}
    pairs=[]
    for j,r in enumerate(newrows):
        i,prev=lookup[r['family'],r['role'],r['lexical']]
        if prev['tokens'][:prev['content_end']+1]==r['tokens'][:r['content_end']+1] and prev['tokens']!=r['tokens']:
            pairs.append((i,j))
    assert pairs,'No unchanged-prefix / changed-future cases to test'
    ix,jx=np.array(pairs).T;checks=[]
    for b in range(40):
        for field in ['G','V','A']:
            a=np.load(old/'pilot'/f'B{b}-{field}.npy',mmap_mode='r')[ix,0]
            v=np.load(R.OUT/'pilot'/f'B{b}-{field}.npy',mmap_mode='r')[jx,0]
            different=int(np.count_nonzero(a!=v));af=R.unbits(a);vf=R.unbits(v)
            checks.append(dict(block=b,field=field,unequal_values=different,
                               max_absolute_difference=float(np.max(np.abs(af-vf)))))
        if b%10==9:R.log('future_prefix_audit',b+1,40,start)
    R.save('future-prefix-audit.json',dict(passed=all(c['unequal_values']==0 for c in checks),
        matched_cases=len(pairs),matching='Identical complete content-visible token prefix; changed future question or options; same family/role/title.',
        matched_ids=[newrows[j]['id'] for i,j in pairs],checks=checks,
        previous_pilot_sha256=R.file_sha(old/'pilot/collection.json'),current_pilot_sha256=R.file_sha(R.OUT/'pilot/collection.json'),
        source_sha256=R.file_sha(__file__),scope='Saved discovery pilots only; numeric causal-prefix check, not additional semantic confirmation.'))
    print('prefix checks',len(pairs),'cases',all(c['unequal_values']==0 for c in checks),flush=True)


if __name__=='__main__':run()
