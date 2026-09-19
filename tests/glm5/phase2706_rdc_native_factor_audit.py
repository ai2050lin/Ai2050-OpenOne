"""All-unit SwiGLU arithmetic floors and whitespace-only first-content scoring sensitivity."""
from rdc_conditional_common import *
from scipy.special import expit
OUT=CAMPAIGN/'l_aligned'


def main():
    records=[];text_scores=[]
    for key in ('qwen4','qwen14','glm4'):
        out=OUT/key;rows=read(out/'prefixes.json');assert len(list((out/'prefix_commits').glob('*.json')))==256
        text_correct=0;strict_correct=0;different=[]
        for r in rows:
            b=read(out/f'behavior/{r["sample_id"]}.json');strict_correct+=int(b['content_token_correct'])
            target=('Yes' if r['expected_yes'] else 'No') if r['language']=='en' else ('是' if r['expected_yes'] else '否')
            content=next((s['text'] for s in b['steps'] if s['lexical_content']),None)
            correct=content is not None and content.strip().casefold()==target.casefold();text_correct+=int(correct)
            if correct!=b['content_token_correct']:different.append({'sample_id':r['sample_id'],'first_lexical_piece':content,'target':target,'strict_id_correct':b['content_token_correct'],'trimmed_piece_correct':correct})
        text_scores.append({'model':key,'n':256,'strict_token_id_correct':strict_correct,'trimmed_casefold_first_piece_correct':text_correct,'changed_cases':different})
        with np.load(out/'features.npz') as z:
            layers=sorted({int(k.split('_')[1][1:]) for k in z.files if k.startswith('prefill_L')})
            for stage in ('prefill','content'):
             for l in layers:
                g,u,a=[unbits(z[f'{stage}_L{l}_{field}']).astype(np.float64) for field in ('gate','up','a')]
                ideal=g*expit(g)*u;err=np.square(ideal-a)
                record={'model':key,'stage':stage,'layer':l,'cases':len(g),'units':g.shape[1],'mse':float(err.mean()),
                  'max_abs':float(np.sqrt(err.max())),'relative_to_energy':float(err.sum()/max(np.square(a).sum(),1e-30)),
                  'sigmoid_evaluation':'Stable FP64 scipy.special.expit; no clipping of observed gate values'}
                records.append(record);npz(out/f'unit_errors/{stage}_L{l}_native_arithmetic_audit.npz',coordinate_mse=err.mean(0),energy=np.square(a).mean(0))
        print('NATIVE_FACTOR_AUDIT',key,strict_correct,text_correct,flush=True)
    save(OUT/'native_factor_arithmetic_audit.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'cases':records,'first_piece_text_sensitivity':text_scores,
      'scope':'Known native architecture arithmetic compared with original BF16 factors; not a forecast or proof of unique linguistic modules. Same exact states may recur atprefill/content, not independent samples.',
      'first_piece_limit':'Whitespace trimming/casefold changes only first lexicalwordpiece scoring, not primary captures/ID scores, alignment or full-answer/EOS scoring. A wordpiece may not be a complete word.'})


if __name__=='__main__':main()
