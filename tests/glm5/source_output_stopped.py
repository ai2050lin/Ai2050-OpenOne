"""Register only whole-branch stops; never load a model or create native data."""
import argparse
import source_output_core as S
from source_output_core import C
from source_output_runner import admitted,check_analysis

def main(split):
    C.checked();check_analysis();active=admitted(split)
    assert not active,'This recovery entry cannot collect or bypass a passed branch'
    gate=C.read(C.OUT/f'behavior-{split}.json')
    nf=len(C.read(C.OUT/(split+'-materials.json'))['rows'])//24
    C.save('admission-'+split+'.json',{
        'active_branches':[],'stopped_branches':list(gate['branches']),
        'rule':'whole predeclared task x alphabet branches; never discard individual families or conditions',
        'family_count_per_active_branch':nf,
        'calibration_behavior_sha256':C.sha(C.OUT/'behavior-calibration.json'),
        'split_behavior_sha256':C.sha(C.OUT/f'behavior-{split}.json'),
        'stop_registration_source_sha256':C.sha(__file__),
        'recovery':'Frozen collector has a Path/string concatenation error before admission output. The frozen source is retained. This separate entry only records the required all-branches-stopped state; no collection or selection can execute.',
    })
    C.save('native-'+split+'-complete.json',{'n':0,'reason':'All branches failed their behavior gates','registration_only':True})
    print(f'{split}: all branches stopped; no native collection',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['calibration','confirm']);main(parser.parse_args().split)
