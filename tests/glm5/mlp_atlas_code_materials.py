"""Whole-native-MLP study: fact × question × answer-code, prospectively frozen."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared.dist_protocol.mechanism_trace import digest,file_sha
from tools.power_analysis import check_phase
OUT=ROOT/'tests/glm5/result/mlp_atlas_code_v1';PUBLIC=ROOT/'research/mlp_atlas/code_v1'
MODEL=Path('D:/develop/model/Qwen/Qwen3-4B');RUN='MLP-ATLAS-CODE-20261104-v1';SEED=41105026
OLD=ROOT/'tests/glm5/result/native_mlp_long_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d,immutable=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(d,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    if immutable and p.exists():assert p.read_text(encoding='utf-8')==raw,('immutable conflict',str(p))
    else:p.write_text(raw,encoding='utf-8')
def token_sha(x):return hashlib.sha256(np.asarray(x,np.int32).tobytes()).hexdigest()

def main():
    from transformers import AutoTokenizer
    from scipy.stats import binom
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);previous=read(OLD/'documents.json')['docs'];used={(r['source'],str(r['id'])) for r in previous}
    old_tokens=np.load(OLD/'input-tokens.npy',mmap_mode='r');hashes={token_sha(row[:512]) for row in old_tokens};source=read(OLD/'wikipedia-en-fresh.json');new=[]
    for r in source['docs']:
        if (r['source'],str(r['id'])) in used or token_sha(r['tokens'][:512]) in hashes:continue
        new.append(r)
    new=sorted(new,key=lambda r:digest({'seed':SEED,'id':r['id']}));assert len(new)>=416
    chosen=new[:416];prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False);end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    codes={'yes_no':['No','Yes'],'arbitrary':['B','A'],'chinese':['否','是']};rng=np.random.default_rng(SEED);items=[]
    assignments={}
    for start,stop in [(0,32),(32,224),(224,416)]:
        for frame in [0,1]:
            indices=[k for k in range(start,stop) if k%2==frame];levels=rng.permutation([(a,b) for _ in range(len(indices)//4) for a in [0,1] for b in [0,1]])
            assignments.update({k:(int(a),int(b)) for k,(a,b) in zip(indices,levels)})
    for k,doc in enumerate(chosen):
        split='discovery' if k<32 else 'confirm' if k<224 else 'lockbox';pk,ns=assignments[k];positive_key='K' if pk else 'M'
        # Separate nonce stems and phrasings in each split; suffix identity balanced independently.
        stem={'discovery':'valuno','confirm':'xerimo','lockbox':'qulavo'}[split]+str(k)
        pos=stem+('_cobalt' if ns else '_umber');neg=stem+('_umber' if ns else '_cobalt')
        descriptions={positive_key:f'{pos} is a geographic settlement, a small town beside a northern coast.',('M' if pk else 'K'):f'{neg} is a portable instrument, a hand tool kept beside a northern coast.'}
        common='\n\nUse only the invented definitions. These are distinct referents.\nEntry K: '+descriptions['K']+'\nEntry M: '+descriptions['M']+'\n'
        tails={};labels={};frame=k%2
        for code,m,t,q in itertools.product(codes,[0,1],[0,1],[0,1]):
            selected=positive_key if t else ('M' if pk else 'K');truth=t==q;answer=int(truth)^m;false_label,true_label=codes[code][m],codes[code][1^m]
            instruction=f'Answer coding for this question: {true_label} means the statement is true; {false_label} means the statement is false. Reply with one answer code only.\n'
            noun='geographic place' if q else 'portable object'
            if split=='discovery':question=f'Does Entry {selected} define a {noun}?' if not frame else f'Is the referent of Entry {selected} a {noun}?'
            elif split=='confirm':question=f'According to Entry {selected}, is its referent a {noun}?' if not frame else f'For Entry {selected}, does the definition describe a {noun}?'
            else:question=f'When Entry {selected} is selected, is the item a {noun}?' if not frame else f'Read Entry {selected}: is its named referent a {noun}?'
            key=(code,m,t,q);tails[key]=tok.encode(common+instruction+question+'\nAnswer:',add_special_tokens=False);labels[key]=(codes[code][answer],selected)
        max_tail=max(map(len,tails.values()));bg=doc['tokens'][:1024-len(prefix)-max_tail];assert len(bg)>=768
        for (code,m,t,q),tail in tails.items():
            padding=[198]*(max_tail-len(tail));ids=prefix+bg+padding+tail;expected,selected=labels[code,m,t,q];alt=next(s for s in codes[code] if s!=expected)
            # Locate selector by the single-token pair difference, never by decoded substring heuristics.
            peer=prefix+bg+padding+tails[code,m,1-t,q];assert len(peer)==len(ids)==1024;diff=np.flatnonzero(np.asarray(ids)!=peer);assert len(diff)==1
            positions=[int(diff[0])-1,int(diff[0]),1023,1023+len(end)]
            items.append({'id':f'{code}:{k}:{m}:{t}:{q}','background_index':k,'background_doc_id':doc['source']+':'+str(doc['id']),'background_source':{key:doc[key] for key in ['source','repo','revision','license','file','row_group','row','id']},
                'split':split,'code':code,'mapping':m,'fact':t,'question':q,'decision':int(t==q),'expected':expected,'contrast':alt,'selected_key':selected,'positive_key':positive_key,'name_assignment':ns,'frame':frame,
                'padding_tokens':len(padding),'background_tokens':len(bg),'token_ids':ids+end,'readout_positions':positions,'prompt':tok.decode(ids+end),'entity':stem,'scope':'事实×问题×答案代码','input_tokens':len(ids+end)})
    # Whole-array single-unit sign tests: each unit × two readouts shares a fixed family budget.
    alpha=.05/(36*9728*2);critical=int(binom.isf(alpha,192,.5))+1;power=float(binom.sf(critical-1,192,.85))
    prereg={'schema':'mlp_atlas_code_prereg.v1','kind':'real','run_id':RUN,'model_snapshot':read(ROOT/'tests/glm5/result/native_mlp_function_v4/observations.json')['model_snapshot'],
        'n_inputs':len(items),'independent_backgrounds':416,'pilot':'discovery backgrounds0..15, all24 factor cells; per frame-cell8, >=7 correct to continue; no heldout inference before pilot gate',
        'behavior':'confirm/lockbox each192 backgrounds; each code×mapping×fact×question cell >=.90 accuracy. Failing cells disqualify their corresponding functional interpretations.',
        'criteria':{'whole_array_units':350208,'readouts':['question_end','assistant_boundary'],'alpha_per_unit_readout':alpha,'min_order_accuracy':.8,'min_nonzero_fraction':.8,'exact_binomial_critical_successes_n192':critical,'expected_order_accuracy':.85,'exact_binomial_power':power,
            'cross_code_decision':'discovery-fitted direction/sign must reproduce across both mapping orientations and all3code families in both heldouts; no answering-code invariance claimed for a failed behavior cell',
            'lexical_answer':'mapping inversion reverses contrast while decision fixed; code-specific label response is distinct from abstract decision',
            'embedding_and_prekey':'all truth pairs have identical final embedding and identical before-selector prefix; negative controls required'},
        'power':{'paired_effect':.35,'n':192,'assumed_discordance':.5,'global_corrected':check_phase(.35,192,p_d=.5,alpha=alpha),'exact_sign_power':power},
        'storage':{'budget_GiB':65,'retain':'full native G/V/A and all37hidden atquestion/assistant, all native indices; selected deep traces also allHead/QKV/parameters. Compact bank, no redundant attention rows for every item.','no_unbounded_loop':True},
        'native_unit_vs_dictionary':'all native units first; SAE/transcoder remains a separate extraction engine andcannot replace native identities',
        'source_scope':'unused documents from cached pinned Wikipedia rowgroups4-6, excluded from previous native long-text inputs; rowgroups/pretraining are not claimed unseen',
        'source_file_sha256':file_sha(OLD/'wikipedia-en-fresh.json'),'materials_source_sha256':file_sha(__file__)}
    assert power>=.8 and prereg['power']['global_corrected']['pass']
    corpus={'schema':'mlp_atlas_code_inputs.v1','kind':'real','run_id':RUN,'items':items,'codes':codes,'source_scope':prereg['source_scope'],'model_snapshot':prereg['model_snapshot']}
    prereg['corpus_sha256']=digest(corpus);save(OUT/'corpus.json',corpus);save(PUBLIC/'prereg.research.json',prereg)
    print(json.dumps({'inputs':len(items),'pilot_inputs':384,'unused_documents_available':len(new),'sign_power':power,'min_background_tokens':min(r['background_tokens'] for r in items)}))
if __name__=='__main__':main()
