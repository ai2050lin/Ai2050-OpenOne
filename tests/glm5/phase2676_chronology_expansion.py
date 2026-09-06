"""Prospective expansion of localized amplitude exceptions, not a new universal success gate."""
import argparse,gc,itertools,shutil
import numpy as np
import torch
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import row,encoded,LAYERS,SITES
from phase2671_native_mlp_field import run,unbits
from phase2662_symmetric_mapping_contract import load_native
from phase2663_symmetric_mapping_calibration import behavior_groups
from phase2657_truth_answer_maps import filename

OUT=RESULT/'phase2676_native_mlp_delivery/expansion'
EN='Alaric Beatrix Dorian Evelina Florian Isolde Lucian Mireille Percival Seraphina Thaddeus Ulrika Valerian Xanthe Yvonne Zachary'.split()
ZH='谭榆 梅汐 施洛 连岑 艾青 邢霄 牟星 齐岸 胥言 祝溪 宁杉 臧禾 申茉 柳泉 费桐 钱舟'.split()
HS=((24,2355),(27,1217))

def prepare():
    if (OUT/'protocol/frozen.json').exists():return read(OUT/'material/cases.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);cases=[]
    for lang,e,c,f,o,p,q,m,v in itertools.product(('en','zh'),range(8),range(4),range(2),range(2),range(2),range(2),range(2),range(2)):
        r=row('chronology',lang,e%4,0,f,v,o,p,q,m);a,b=(EN if lang=='en' else ZH)[2*e:2*e+2];old=r['entity_a'],r['entity_b']
        verb=('arrived','departed','registered','finished')[c] if lang=='en' else ('到达','离开','登记','完成任务')[c]
        for key in ('body','text'):
            r[key]=r[key].replace(old[0],'{{A}}').replace(old[1],'{{B}}').replace('{{A}}',a).replace('{{B}}',b).replace('arrived' if lang=='en' else '到达',verb)
        r.update(entity_a=a,entity_b=b,unit=e,content_instance=c,field_set='initial' if e<4 else 'confirmation',fp_selected=False,
            published=e in (4,5) and (c,f,o,p,q,m,v)==(0,0,0,0,0,0,0),case_index=len(cases),case_id=f'expanded_chronology/{lang}/e{e}/verb{c}/f{f}/o{o}/p{p}/q{q}/m{m}/v{v}')
        cases.append(encoded(tok,r))
    assert len(cases)==len({r['prompt'] for r in cases})==4096 and sum(r['published'] for r in cases)==4
    plan={'cases':4096,'factors':'2languages*8newentitypairs*4eventverbs*2forms*2orders*2probes*2polarities*2mappings*2targets',
        'trigger':'Phase2673all7oldcandidate directions64/64but strictdominance58..63/64; five sites have only onefailedgroup, mainlyChinesechronology. Triggerselectedfamily, not prospectiveglobalcapabilitysample.',
        'candidate_h':HS,'candidate_a':SITES,'split':'4initial+4confirmation entitypairs perlanguage;4contentverbs crossed independently',
        'priority':'Do not discard signconsistency because an amplitude gatefails. Keep every q/m behavior, but exact oldsign/dominance comparisons onq0m0 separately. Do not relax the oldgate and rename failure success.',
        'storage':'Alltoken allcoordinate6field moments remainprimarybackground; tiny frozen-candidate boundary traces inall4096behaviorrecords; only4complete fulltoken rawpacks saved. No unshown rawpack toclean, noTopK projection.',
        'limits':'Posthocselectedchronology, foureventverbs, reusedtwoforms/instruction/demos. Confirmation tests conditional robustness, not all8families or universal semantic gears.',
        'material_sha256':None}
    save(OUT/'material/cases.json',cases);plan['material_sha256']=sha(OUT/'material/cases.json');save(OUT/'protocol/frozen.json',plan);return cases

def observe(r,pack):
    h=unbits(pack['h']);a=unbits(pack['a']);g=unbits(pack['gate']);u=unbits(pack['up'])
    return {'native_candidates':{'h':[h[l,:,j].tolist() for l,j in HS],'a':[a[l,:,j].tolist() for l,j in SITES],
        'gate':[float(g[l,j]) for l,j in SITES],'up':[float(u[l,j]) for l,j in SITES]}}

def execute():
    assert (RESULT/'phase2675_native_mlp_crossmodel/analysis/final.json').exists()
    assert not (OUT/'analysis/completion.json').exists();cases=prepare()
    budget=4*max(len(r['prompt_ids']) for r in cases)*2*(37*2560+4*(3*9728+2*2560))+4*16*(37*2560+3*36*9728+2*36*2560)+100*1024**2
    assert shutil.disk_usage(ROOT).free>budget+8*1024**3,('expansion_budget',budget)
    model,tok=load_native('qwen4');records=run(model,tok,cases,OUT,raw_all=False,observer=observe)
    from phase2676_prefix_replay import execute as prefix_replay
    prefix_replay(model,tok,'bf16')
    del model;gc.collect();torch.cuda.empty_cache();tables=[]
    for lang,fold,p,q,m in itertools.product(('en','zh'),('initial','confirmation'),(0,1),(0,1),(0,1)):
        rr=[r for r in records if (r['language'],r['field_set'],r['probe_index'],r['polarity'],r['mapping'])==(lang,fold,p,q,m)]
        assert len(rr)==128
        for metric,ss in [('h',HS),('a',SITES)]:
            values=np.asarray([r['native_candidates'][metric] for r in rr])[:,:,-1];result={}
            for label,axis in [('target','target_index'),('form','form'),('order','mention_order')]:
                groups={}
                for i,r in enumerate(rr):groups.setdefault(tuple(r[k] for k in ('unit','content_instance','form','mention_order','target_index') if k!=axis),{})[r[axis]]=i
                ij=np.asarray([(g[0],g[1]) for g in groups.values()]);d=values[ij[:,0]]-values[ij[:,1]];result[label]={'mean':d.mean(0),'rms':np.sqrt((d*d).mean(0))}
            with np.load(RESULT/'phase2657_truth_answer_maps/maps/initial'/filename('chronology',lang,p,0,0)) as z:old=z[f'target__{"mlp" if metric=="a" else "h"}__mean']
            for si,(l,j) in enumerate(ss):tables.append({'language':lang,'fold':fold,'p':p,'q':q,'m':m,'metric':metric,'layer':l,'coordinate':j,'pairs':64,
                'contrasts':{k:{s:float(v[si]) for s,v in vv.items()} for k,vv in result.items()},'old_direction':bool(np.sign(result['target']['mean'][si])==np.sign(old[l,j]) and old[l,j]!=0),
                'dominates_form_order':bool(result['target']['rms'][si]>max(result['form']['rms'][si],result['order']['rms'][si]))})
    baseline={}
    for metric,ss in [('h',HS),('a',SITES)]:
        for l,j in ss:
            rr=[r for r in tables if (r['metric'],r['layer'],r['coordinate'],r['q'],r['m'])==(metric,l,j,0,0)]
            baseline[f'{metric}{l}[{j}]']={'n':len(rr),'old_direction':sum(r['old_direction'] for r in rr),'dominates':sum(r['dominates_form_order'] for r in rr),'both':sum(r['old_direction'] and r['dominates_form_order'] for r in rr)}
    save(OUT/'analysis/candidate_scope.json',tables)
    checks={'4096_cases':len(records)==4096,'4fullfield_examples':sum(r['published'] for r in cases)==4,'224candidate_groups':len(tables)==224,'fourfullcoordinate_momentgroups':len(list((OUT/'maps').glob('alltoken_*.npz')))==4,'material_immutable':sha(OUT/'material/cases.json')==read(OUT/'protocol/frozen.json')['material_sha256']}
    assert all(checks.values());save(OUT/'analysis/completion.json',{'checks':checks,'all_checks_passed':True,'behavior':behavior_groups(records),'normal_protocol_candidates':baseline,'scope':'8normalgroups (2languages*2folds*2probes),64matchedpairs each, posthoc familytargeted expansion. Not64familygroups or oldgateuniversalconfirmation.'});print('CHRONOLOGY EXPANSION COMPLETE',json.dumps(baseline),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);a=p.parse_args();globals()[a.action]()
