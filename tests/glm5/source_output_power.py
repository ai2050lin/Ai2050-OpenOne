"""Prospective secondary binary damage sensitivity; primary outcome is a margin."""
import argparse,math
from datetime import datetime
import source_output_core as S
from source_output_core import C
from source_output_causal import selected
from source_output_runner import admitted
from tools.power_analysis import check_phase

def freeze():
    p=C.checked();assert not list(C.OUT.glob('behavior-calibration-[0-9]*.json'))
    assert not (C.OUT/'secondary-power-freeze.json').exists()
    check=check_phase(.20,96,p_d=.25,alpha=p['alpha']);assert check['pass']
    C.save('secondary-power-freeze.json',{'schema':'source_output_secondary_binary_power.v1','time':datetime.now().astimezone().isoformat(),
        'base_plan_sha256':C.sha(C.OUT/'plan.json'),'source_sha256':C.sha(__file__),'calculator_sha256':C.sha(C.ROOT/'tools/power_analysis.py'),
        'primary_unchanged':'0.2 nats target-margin effect; continuous paired-family power is in the original plan.',
        'secondary_target':'Detect a large 20 percentage-point accuracy loss, assuming 25% discordant pairs. This is a sensitivity target, not a forecast of observed damage.',
        'sampling':'One predeclared code_sign=+1 row per chosen family, candidate and branch; 96 families. The opposite mapping is excluded only from this secondary binary statistic, not from any original comparison.',
        'check_phase':check,'alpha':p['alpha'],'sensitivity':[check_phase(d,96,p_d=.25,alpha=p['alpha']) for d in [.03,.05,.10,.20]],
        'warning':'Insufficient for subtle binary changes; shared phrase frames further limit population generalization. No flips would not prove no effect.',
        'exposure':'No behavioral or native model observations yet; candidates, behavior thresholds and original outcomes unchanged.'})

def report():
    d=C.read(C.OUT/'secondary-power-freeze.json');assert d['source_sha256']==C.sha(__file__) and d['calculator_sha256']==C.sha(C.ROOT/'tools/power_analysis.py')
    rows=selected(C.read(C.OUT/'confirm-materials.json')['rows'],admitted('confirm'));groups={}
    for i,row in enumerate(rows):
        if row['code_sign']!=1:continue
        item=C.read(C.OUT/f'causal-{i:04d}.json');base=item['baseline']['strict_correct']
        for key,edit in [(e['candidate'],e['edited']) for e in item['edits']]+[('joint',item['joint'])]:
            g=groups.setdefault((S.branch(row),key),{'n':0,'correct_to_wrong':0,'wrong_to_correct':0});g['n']+=1
            g['correct_to_wrong']+=int(base and not edit['strict_correct']);g['wrong_to_correct']+=int(not base and edit['strict_correct'])
    results=[]
    for (branch,candidate),v in groups.items():
        assert v['n']==96;b=v['correct_to_wrong'];c=v['wrong_to_correct'];discordant=b+c
        prob=min(1.,2*sum(math.comb(discordant,k) for k in range(min(b,c)+1))/2**discordant) if discordant else 1.
        results.append({'branch':branch,'candidate':candidate,**v,'net_accuracy_loss':(b-c)/v['n'],'exact_McNemar_two_sided_p':prob,
            'large_damage_supported':bool((b-c)/v['n']>=.2 and prob<d['alpha'])})
    C.save('secondary-binary-results.json',{'schema':'source_output_secondary_binary_results.v1','results':results,'scope':d['sampling'],'limitation':d['warning'],'formal_promotion':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','report']);globals()[p.parse_args().stage]()
