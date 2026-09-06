"""Tokenizer-only native chat/boundary preflight; not formalmodel research."""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from transformers import AutoConfig,AutoTokenizer
from phase2683_crossmodel_function_atlas import ROOT,OUT,MODELS,KEYS,MODEL_KEYS,material,save

def main():
    result={'tokenizer_only':True,'models':{},'research_forwards':0}
    for key in KEYS:
        path=ROOT/'models/hf'/MODELS[MODEL_KEYS[key]];tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
        cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True);rows,cal=material(key,tok)
        group={}
        for r in rows:group.setdefault((r['family'],r['language'],r['unit'],r['content_instance'],r['target_index']),[]).append(r)
        comparisons=0
        for rr in group.values():
            assert len(rr)==4;prefix=rr[0]['prompt_ids'][:rr[0]['body_end_token']+1]
            for r in rr[1:]:assert r['prompt_ids'][:r['body_end_token']+1]==prefix;comparisons+=1
        T=max(len(r['prompt_ids']) for r in rows);shape=max(160,32*math.ceil(T/32));L,D,K=cfg.num_hidden_layers,cfg.hidden_size,cfg.intermediate_size;N=(L+1)*D+L*K
        budget=16*5*2*N+2*2*N*8+16*((L+1)*D+3*L*K+2*L*D)+2*(2*2*N+2*T*(L+1)*D)+64*1024**2
        result['models'][key]={'cases':len(rows),'calibration':len(cal),'body_prefix_comparisons':comparisons,'actual_token_range':[min(len(r['prompt_ids']) for r in rows),T],
            'field_execution_length':shape,'layers':L,'hidden':D,'mlp_units':K,'storage_upper':budget,
            'open_think_prompts':sum(r['prompt'].rfind('<think>')>r['prompt'].rfind('</think>') for r in rows),
            'cloze_assistant_suffix_example':next(r['prompt'][-140:] for r in rows if r['output_function']=='cloze'),'eos':tok.eos_token_id}
    result['all_checks_passed']=all(r['cases']==512 and r['calibration']==64 and r['body_prefix_comparisons']==384 for r in result['models'].values())
    save(OUT/'analysis/material_preflight.json',result);print(result);assert result['all_checks_passed']

if __name__=='__main__':main()
