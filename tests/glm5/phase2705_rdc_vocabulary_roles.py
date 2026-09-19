"""Descriptive output-token partition of frozen full-vocabulary forecasts, not new fitting."""
from rdc_conditional_common import *
OUT=CAMPAIGN/'k_long'


def main():
    rows={r['sample_id']:r for r in read(OUT/'material.json')};config=read(ROOT/'models/hf/qwen3-4b/config.json');eos=config['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    reports=[]
    for path in sorted((OUT/'vocabulary').glob('*.json')):
        data=read(path)
        if 'states' not in data:continue
        groups={}
        for s in data['states']:
            r=rows[s['sample_id']];kind='EOS' if r['next_token_id'] in eos else 'lexical_or_number' if any(c.isalnum() for c in r['next_token']) else 'whitespace_or_punctuation'
            groups.setdefault(kind,[]).append(s)
        report={'model':data['summary']['model'],'by_native_emitted_token_kind':{k:{'n':len(v),'mean_KL':float(np.mean([s['kl_native_to_prediction'] for s in v])),
          'argmax_matches':sum(s['native_argmax_match'] for s in v),'prefixes':len({s['prefix_id'] for s in v})} for k,v in groups.items()}}
        reports.append(report)
    save(OUT/'vocabulary_token_kind_audit.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'status':'post-hoc descriptive partition, no fit/threshold changes',
      'partition':'Native emitted EOS ID; otherwise anyUnicodeletter/digit versus whitespace/punctuation. Partialwordpieces are possible; this is NOT semantic content versus format certainty.',
      'reports':reports,'limits':['Partition uses actualemittedtoken aftergeneration, never a predictor input.','Tokenkind is not a universal linguistic role.','Correlatedstates within32testprefixes, no independenttokenconfidence claim.']})
    for r in reports:
        if r['model'] in ('H12_quadratic','H12_previousH36_quadratic'):print(json.dumps(r,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
