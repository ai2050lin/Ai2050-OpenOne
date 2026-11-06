"""Third calibration: ask explicitly about an entry, natural Yes/No, before readout."""
import sys,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v2 as V2
import native_mlp_long_materials as OLD
from shared.dist_protocol.mechanism_trace import digest,file_sha
ROOT=V2.ROOT;OUT=ROOT/'tests/glm5/result/native_mlp_function_v3';PUBLIC=ROOT/'research/native_mlp_function/v3';RUN='NMLP-FUNCTION-20261104-v3';MODEL=V2.MODEL;SEED=V2.SEED
read=V2.read;save=V2.save;FAMILIES=V2.FAMILIES
QUESTIONS={'conditional':'Does the action described in this entry require a particular detected condition?',
 'universal':'Does this entry say that every member of its set has a lid, with no exception?',
 'location':'Does this entry define a geographical place?',
 'enumeration':'Does the colon inside this entry introduce an enumeration of items?'}
def main():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);base=read(V2.OUT/'corpus.json');raw=np.load(OLD.OUT/'input-tokens.npy',mmap_mode='r');prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False)
    end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False);rows=[]
    for item in base['items']:
        records=item['records'];common='\n\nUse only the following invented catalogue definitions. Each entry has a distinct referent. Do not combine entries.\nEntry K: '+records['K']+'\nEntry M: '+records['M']+'\n'
        tails={key:tok.encode(common+f'Question about Entry {key}: '+QUESTIONS[item['family']]+' Reply with only Yes or No.\nAnswer:',add_special_tokens=False) for key in ['K','M']}
        assert len(tails['K'])==len(tails['M']);diff=np.flatnonzero(np.asarray(tails['K'])!=tails['M']);assert len(diff)==1
        bg=1024-len(prefix)-len(tails['K']);assert bg>=512;ids=prefix+raw[item['background_doc_index'],:bg].tolist()+tails[item['selected_key']]
        early=len(prefix)+bg+len(tok.encode(common,add_special_tokens=False))-2;keypos=len(prefix)+bg+int(diff[0]);assert early<keypos<1023
        rows.append({**item,'token_ids':ids+end,'checkpoint_ids':ids,'prompt':tok.decode(ids+end),'text':tok.decode(ids),'expected':'Yes' if item['semantic'] else 'No','contrast':'No' if item['semantic'] else 'Yes',
            'readout_positions':[early,keypos,1023,1023+len(end)],'input_tokens':1024+len(end),'task_question':QUESTIONS[item['family']]})
    c={**base,'run_id':RUN,'items':rows,'revision_reason':'v1 shared referents andv2 active-entry questions failed discovery behavior; v3 explicit EntryK/M question, distinct referents, natural Yes/No; all readouts include causal negative prefix, key, fully visible question andassistant boundary',
        'answer_labels':['Yes','No'],'question_before_checkpoint':True}
    save(OUT/'corpus.json',c);save(OUT/'discovery.json',read(V2.OUT/'discovery.json'))
    pr={**read(V2.PUBLIC/'prereg.research.json'),'run_id':RUN,'corpus_sha256':digest(c),'revision_reason':c['revision_reason'],
        'primary':'4 preselected native units A ordering at readout index2 (question fully visible, common Answer colon), sign from discovery only; assistant boundary index3 exploratory',
        'previous_pilot_failures_preserved':[file_sha(ROOT/'tests/glm5/result/native_mlp_function_v1/pilot-result.json'),file_sha(V2.OUT/'pilot-result.json')]}
    save(PUBLIC/'prereg.research.json',pr);print({'inputs':len(rows),'readouts':4})
if __name__=='__main__':main()
