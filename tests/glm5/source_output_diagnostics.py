"""Post-calibration failure diagnostics; never replaces the frozen behavior gate."""
from collections import Counter
from transformers import AutoTokenizer
import source_output_core as S
from source_output_core import C

def main():
    C.checked();tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);output=[]
    for split in ['calibration','confirm']:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows']
        scores=sum([C.read(C.OUT/f'behavior-{split}-{i:04d}.json')['scores'] for i in range(len(rows)//24)],[])
        for key in C.read(C.OUT/f'behavior-{split}.json')['branches']:
            part=[(r,s) for r,s in zip(rows,scores) if S.branch(r)==key];counts=Counter();hist=Counter();cells={}
            for row,score in part:
                text=tok.decode([score['argmax']]);hist[(score['argmax'],text)]+=1
                labels=['1','2'] if row['alphabet']=='digits' else ['A','B']
                expected=labels[0 if row['output_sign']==1 else 1];source='1' if row['target_sign']==1 else '2'
                flags={'exact_top1':score['first_top1'],
                    'correct_after_trimming_whitespace':text.strip()==expected,
                    'correct_code_with_extra_whitespace':text.strip()==expected and not score['first_top1'],
                    'wrong_code_after_trimming':text.strip() in labels and text.strip()!=expected,
                    'record_number_in_letter_task':row['alphabet']=='letters' and text.strip() in ['1','2'],
                    'correct_record_number_in_letter_task':row['alphabet']=='letters' and text.strip()==source,
                    'bare_candidate_correct':score['strict_correct']}
                counts.update({k:int(v) for k,v in flags.items()})
                cell=cells.setdefault(f"position={row['target_sign']};map={row['code_sign']}",{'n':0,'correct_after_trimming_whitespace':0,'exact_top1':0})
                cell['n']+=1
                for k in ['correct_after_trimming_whitespace','exact_top1']:cell[k]+=int(flags[k])
            output.append({'split':split,'branch':key,'n':len(part),'counts':dict(counts),'cells':cells,
                'top1_token_histogram':[{'id':i,'text':t,'n':n} for (i,t),n in hist.most_common()]})
    C.save('failure-diagnostics.json',{'schema':'source_output_failure_diagnostics.v1','branches':output,
        'method':'Decode the already saved actual argmax token; classify whitespace-equivalent answer codes and unconverted record numbers. No new forward passes.',
        'timing':'Exploratory diagnostic motivated by calibration failures; not an independent confirmatory test.',
        'formal_gate_unchanged':True,'stopped_branches_remain_stopped':True,
        'limitation':'Whitespace trimming diagnoses first-token scoring only; it does not validate complete generation, repair mapping errors, or establish source-selection mechanisms.',
        'source_sha256':C.sha(__file__)})

if __name__=='__main__':main()
