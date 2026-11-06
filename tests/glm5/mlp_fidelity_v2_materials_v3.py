"""Fresh language-balanced backgrounds and unambiguous coastal/inland definitions."""
import sys,itertools,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import digest,file_sha

def main():
    from transformers import AutoTokenizer
    source=P.read(P.OUT/'fresh-documents.json');tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False);end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False);codes={'latin_ab':['B','A'],'latin_cd':['D','C']};rng=np.random.default_rng(P.SEED);items=[]
    # Explicitly control the language×frame confound: frame changes every language pair.
    assignments={}
    for lo,hi in [(0,128),(128,320),(320,512)]:
        for lang in ['en','zh']:
            for frame in [0,1]:
                rows=[i for i in range(lo,hi) if source['controlled'][i]['language']==lang and (i//2)%2==frame];levels=rng.permutation([0,1]*(len(rows)//2));assignments.update(zip(rows,map(int,levels)))
    for b,doc in enumerate(source['controlled']):
        split='discovery' if b<128 else 'confirm' if b<320 else 'lockbox';frame=(b//2)%2;key='K' if assignments[b] else 'M';other='M' if key=='K' else 'K';stem={'discovery':'solveta','confirm':'neruvin','lockbox':'pacorul'}[split]+str(b)
        descriptions={key:f'{stem}_one is a settlement directly on the sea coast. It is coastal, not inland.',other:f'{stem}_two is a settlement deep in the interior, far from the sea. It is inland, not coastal.'}
        common='\n\nThe following invented records are authoritative for this question. Both referents are settlements, and the two locations are mutually exclusive.\nRecord K: '+descriptions['K']+'\nRecord M: '+descriptions['M']+'\n';tails={}
        for code,m,t,q in itertools.product(codes,[0,1],[0,1],[0,1]):
            selected=key if t else other;positive=codes[code][1^m];negative=codes[code][m]
            instruction=f'Use the answer code {positive} for a true statement and {negative} for a false statement. Output exactly one code, with no explanation.\n'
            predicate='coastal' if q else 'inland'
            if split=='discovery':question=f'Is the settlement described in Record {selected} in a {predicate} location?' if not frame else f'Does the settlement in Record {selected} have a {predicate} location?'
            elif split=='confirm':question=f'Is a {predicate} location specified for the settlement in Record {selected}?' if not frame else f'Does Record {selected} explicitly locate the settlement in a {predicate} area?'
            else:question=f'Check the settlement description in Record {selected}: does it specify a {predicate} location?' if not frame else f'For the settlement named by Record {selected}, is the stated location {predicate}?'
            tails[code,m,t,q]=tok.encode(common+instruction+question+'\nAnswer:',add_special_tokens=False)
        longest=max(map(len,tails.values()));background=doc['tokens'][:1024-len(prefix)-longest];assert len(background)>=800
        for (code,m,t,q),tail in tails.items():
            ids=prefix+background+[198]*(longest-len(tail))+tail;peer=prefix+background+[198]*(longest-len(tail))+tails[code,m,1-t,q];assert len(ids)==len(peer)==1024;diff=np.flatnonzero(np.asarray(ids)!=peer);assert len(diff)==1
            expected=codes[code][int(t==q)^m];contrast=next(v for v in codes[code] if v!=expected)
            items.append({'id':f'{code}:{b}:{m}:{t}:{q}','background_index':b,'background_source':{k:v for k,v in doc.items() if k!='tokens'},'split':split,'frame':frame,'background_language':doc['language'],'code':code,'mapping':m,'fact':t,'question':q,'decision':int(t==q),'expected':expected,'contrast':contrast,'positive_key':key,'selected_key':key if t else other,'token_ids':ids+end,'input_tokens':len(ids+end),'readout_positions':[int(diff[0])-1,int(diff[0]),1023,1023+len(end)],'background_tokens':len(background),'prompt':tok.decode(ids+end),'scope':'document-defined coastal/inland settlements; English taskinstructions withEnglish/Chinesebackgrounds'})
    corpus={'schema':'mlp_fidelity_controlled_inputs.v1','kind':'real','run_id':P.RUN,'model_snapshot':P.read(P.OUT/'plan.json')['model_snapshot'],'codes':codes,'items':items,'source_sha256':file_sha(P.OUT/'fresh-documents.json'),'materials_source_sha256':file_sha(__file__)}
    assert len(items)==8192;P.save(P.OUT/'controlled-v3-corpus.json',corpus)
    P.save(P.PUBLIC/'materials-v3-freeze.research.json',{'kind':'real','run_id':P.RUN,'model_snapshot':corpus['model_snapshot'],'provenance':P.read(P.OUT/'plan.json')['provenance'],'corpus_sha256':digest(corpus),'source_sha256':file_sha(__file__),'n_inputs':8192,'n_backgrounds':512,'background_min_tokens':min(r['background_tokens'] for r in items),'readout_roles':['question_end','assistant_boundary'],'causal_visibility':'T pairs differ inone selectedrecord token, allprecedingtext equal; common finalreadout tokencolon; noanswer token inprompt answer slot'})
    print({'controlled_inputs':8192,'natural_docs':512,'min_background':min(r['background_tokens'] for r in items)},flush=True)
if __name__=='__main__':main()
