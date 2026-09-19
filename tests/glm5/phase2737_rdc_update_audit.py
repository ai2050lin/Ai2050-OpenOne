"""Independent material logic and same-shape complete native-parameter derivative audit."""
import argparse,re,gc
from rdc_update_common import *

def language_logic():
    rows=gzread(BASE/'language_material.json.gz');records=[]
    for r in rows:
        if (r['language'],r['answer_style'])!=('en','direct'):continue
        body=r['body'];question=r['question'];family=r['family']
        if family=='attribute_binding':
            facts=dict(re.findall(r'(\w+) owns a basket of (red|green) apples',body));who=re.search(r"in (\w+)'s basket",question).group(1);answer=facts[who]=='red'
        elif family=='negation_scope':
            neg,pos=re.search(r'(\w+) did not pack the apples, but (\w+) did pack them',body).groups();who,verb=re.search(r'"(\w+) (packed|did not pack) the apples"',question).groups()
            assert who in (neg,pos);answer=(who==pos) if verb=='packed' else (who==neg)
        elif family=='long_distance_role':
            agent,recipient=re.search(r'(\w+) handed a sealed packet to (\w+)',body).groups();who=re.search(r'Was (\w+) the recipient',question).group(1)
            assert who in (agent,recipient);answer=who==recipient
        elif family=='word_sense':
            # Limited corpus convention, not a general English sense classifier.
            assert ('Apple' in body)!=('apple' in body);assert ('edible fruit' in question)!=('technology company' in question)
            answer=('apple' in body)==('edible fruit' in question)
        else:
            edges=re.findall(r'every (\w+) is a (\w+)',body);initial=re.search(r'instance of (\w+)',body).group(1);goal=re.search(r'X\d+ is (\w+)\?',question).group(1)
            reachable={initial}
            while True:
                new=reachable|{b for a,b in edges if a in reachable}
                if new==reachable:break
                reachable=new
            answer=goal in reachable
        same=[x for x in rows if x['source_group']==r['source_group']];assert len(same)==4
        assert all(bool(x['truth'])==answer for x in same),(r,answer)
        assert all(x['target']==(('Yes' if answer else 'No') if x['language']=='en' else ('是' if answer else '否')) for x in same)
        records.append({'source_group':r['source_group'],'family':family,'independent_English_text_logic':bool(answer),'all4_expression_labels_agree':True})
    result={'timestamp':stamp(),'source':snapshot(__file__),'semantic_cases':len(records),'expressions':len(rows),'checks':records,
      'scope':'English text parsed independently of savedtruth/typededges for4families; word-sense uses this corpus explicit Apple/apple capitalization convention. Checks mirror labels across languages/styles, not a general proof that arbitrary translations preserve meaning.'}
    assert len(records)==160;save(BASE/'language_analysis/logic_audit.json',result);print('LANGUAGE_LOGIC_AUDIT',len(records),flush=True)

def gradients():
    import torch
    from transformers import AutoTokenizer
    from rdc_law_native import Tail,dense_gradient
    from phase2737_rdc_update_directions import material_arrays
    out=BASE/'learning';start=time.monotonic()
    if (out/'autograd_audit.json').exists():return
    rows=gzread(BASE/'program_material.json.gz');xx,rr,tt,_=material_arrays(rows[:8]);tail=Tail()
    x=torch.tensor(xx,device='cuda');r=torch.tensor(rr,device='cuda');target=torch.tensor(tt,device='cuda')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)];local=torch.tensor([digits.index(int(i)) for i in tt],device='cuda');reports={}
    for p in tail.w.values():p.requires_grad_(True)
    for part in ('full','content','format'):
        z=tail.forward(x,r,target);lp=z['logits'].double().log_softmax(-1);clp=z['logits'][:,digits].double().log_softmax(-1)
        loss=(-lp[torch.arange(8,device='cuda'),target] if part=='full' else -clp[torch.arange(8,device='cuda'),local] if part=='content' else -torch.logsumexp(lp[:,digits],-1)).mean()
        actual=torch.autograd.grad(loss,list(tail.w.values()))
        with np.load(out/f'{part}_factors.npz') as f:expected=dense_gradient({k:torch.tensor(f[k][:8],device='cuda') for k in ('x','a','s','bg','bu')})
        reports[part]={k:float((a-expected[k]).abs().max()/a.abs().max().clamp_min(1e-12)) for k,a in zip(tail.w,actual)}
        assert max(reports[part].values())<5e-5,(part,reports)
        del actual,expected,z,loss,lp,clp
    result={'timestamp':stamp(),'source':snapshot(__file__),'samples':[r['sample_id'] for r in rows[:8]],'batch':8,'parameters':74711040,
      'relative_max_errors':reports,'all_passed':True,'seconds':time.monotonic()-start,
      'scope':'Same8-row execution shape as frozen factor capture; all3 complete native-valued FP32 MLP matrices, full vocabulary, FP64 direct conditional scoring. Autograd is not a derivative of discrete BF16 rounding.'}
    save(out/'autograd_audit.json',result);ledger('new_material_full_native_gradient_audit',result['seconds']);print('UPDATE_GRADIENT_AUDIT',reports,flush=True)
    del tail;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--language-only',action='store_true');a=p.parse_args();language_logic() if a.language_only else gradients()
