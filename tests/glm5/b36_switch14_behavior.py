"""Natural paired behavior, immutable identity, and explicit site/behavior admission."""
import sys,time,re
import b36_switch14_design as D
import b36_switch14_engine as E
M=D.M;S=D.S


def sites(row,result,tok):
    found={}
    for k in range(1,len(result['tokens'])):
        prefix=tok.decode(result['tokens'][:k],skip_special_tokens=False)
        for site in D.candidate_sites(prefix):
            if site not in found and D.eligible(site,tok.decode([result['tokens'][k]],skip_special_tokens=False)):
                found[site]=dict(generated_cut=k,prefix=prefix,expected_class=D.target_class(row,site),natural_token=result['tokens'][k])
    return found


def run():
    S.configure(D.PEAK);p=S.read(S.OUT/'plan.json');M.admit(p);start=time.monotonic()
    assert S.file_sha(S.OUT/'calibration.json')==p['calibration_sha256'] and S.read(S.OUT/'calibration.json')['passed']
    import torch
    with S.GpuLease(),torch.inference_mode():
        for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('behavior_identity',i+1,len(p['model_files']),start)
        engine=E.Stream(start,p['budget']['hard_minutes_per_run'])
        for arm in D.ARMS:
            assert not (S.OUT/(arm+'.json')).exists()
            rows=[r for r in p['rows'] if r['arm']==arm];inputs=engine.inputs(rows);t=time.monotonic()
            out=engine.generate('behavior_'+arm,inputs,p['generation']['max_new_tokens']);records=engine.records(rows,out,inputs['input_ids'].shape[1])
            for row,r in zip(rows,records):r['sites']=sites(row,r,engine.tok)
            S.save(arm+'.json',dict(results=records,seconds=time.monotonic()-t,plan_sha256=S.file_sha(S.OUT/'plan.json')))
            del out,inputs
        engine.close();M.admit(p)
    audit()


def audit():
    from transformers import AutoTokenizer
    from scipy.stats import binomtest
    S.configure();p=S.read(S.OUT/'plan.json');M.admit(p);tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    cells=[];all_records={};missing=[];truth_checks=0
    for arm in D.ARMS:
        data=S.read(S.OUT/(arm+'.json'));assert data['plan_sha256']==S.file_sha(S.OUT/'plan.json')
        rows=[r for r in p['rows'] if r['arm']==arm];records=data['results'];assert [r['id'] for r in rows]==[r['id'] for r in records]
        for row,r in zip(rows,records):
            assert tok.encode(row['prompt'],add_special_tokens=False)==row['tokens']
            lines=re.search(r'```python\n(.*?)\n```',row['prompt'],re.S)[1].splitlines();ns={};truth=[]
            for i,line in enumerate(lines):
                exec(compile(line,'<author-created-assignment>','exec'),{'__builtins__':{}},ns)
                if i:truth.append(ns['x'])
            assert truth==row['visible_truth'] and truth[:row['depth']]==row['truth'];truth_checks+=1
            assert tok.decode(r['tokens'],skip_special_tokens=False)==r['text']
            assert M.T.evaluate(r['text'],row,tok.eos_token_id in r['tokens'])==r['parsed']
            assert sites(row,r,tok)==r['sites']
            if set(r['sites'])!=set(D.SITES):missing.append(dict(id=r['id'],present=list(r['sites'])))
        cell=dict(arm=arm,n=len(records),**{key:sum(r['parsed']['metrics'][key] for r in records) for key in ['step_count_correct','steps_correct','value_correct','format_correct','eos','joint']})
        cell['passed']=cell['joint']>=p['min_joint'];cells.append(cell);all_records[arm]=records
    a=[r['parsed']['metrics']['joint'] for r in all_records['first2']];b=[r['parsed']['metrics']['joint'] for r in all_records['first3']]
    gain=sum(not x and y for x,y in zip(a,b));loss=sum(x and not y for x,y in zip(a,b))
    passed=all(c['passed'] for c in cells) and not missing
    S.save('behavior-analysis.json',dict(passed=passed,cells=cells,missing_sites=missing,truth_checks=truth_checks,
        source_hashes={arm:S.file_sha(S.OUT/(arm+'.json')) for arm in D.ARMS},
        paired_descriptive=dict(gained=gain,lost=loss,p_exact=binomtest(min(gain,loss),gain+loss,.5).pvalue if gain+loss else 1),
        failures=[dict(id=r['id'],text=r['text'],parsed=r['parsed']) for rr in all_records.values() for r in rr if not r['parsed']['metrics']['joint']],
        analysis_source_sha256=S.file_sha(__file__)))
    print('Behavior gate',passed,cells,'missing_sites',len(missing),flush=True)


if __name__=='__main__':run() if sys.argv[1]=='run' else audit()
