"""Real CPU-only backend HTTP checks; does not initialize an experiment model."""
import sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2669_symmetric_multitoken_delivery import OUT


def main():
    base='http://127.0.0.1:5001/api/research-assets/'
    def get(path):
        with urllib.request.urlopen(base+path,timeout=60) as response:return json.load(response)
    options=get('native-multitoken-cases');value=get('native-multitoken-parameter');old=get('native-output-parameter');previous=get('native-sequence-parameter');health=get('health')
    panels=get('native-atlas-panels');page=get('native-atlas-rows?panel=phase2669_q14_mlp&start=0&count=1')
    request=urllib.request.Request(base+'file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(request,timeout=30) as response:status=response.status;size=len(response.read())
    startup=(OUT/'analysis/backend_cpu_final_stdout.log').read_text(encoding='utf-8',errors='replace')
    checks={'real64_sequence_cases':len(options['cases'])==64,'real_default_case256':value['case_index']==256,
        'real_two_sequence_branches':len(value['branches'])==2 and all(len(b['tokens'])>0 for b in value['branches']),
        'real_parameter_difference':value['parameter_derivative']==value['branches'][0]['derivative']-value['branches'][1]['derivative'],
        'prior_output_api_preserved':old['module']=='v_proj','previous_sequence_preserved':len(previous['branches'])==2,'three_part_derivatives':set(value['part_parameter_derivatives'])=={'content','format','eos'},'asset_root':health['available'],'range206':status==206 and size==1024,
        'model_load_skipped':'starting API server without local model' in startup,'ten_heatmap_types':len(panels['panels'])==10,'all17408_native_columns':page['coordinate_count']==17408 and len(page['rows'][0]['values'])==17408}
    proof={'checks':checks,'all_checks_passed':all(checks.values()),'base':base,'frontend':'http://localhost:5173',
        'launch':'Original server/server.py, hidden, CPU-only CUDA_VISIBLE_DEVICES=-1, skip model load, offline, localhost only.',
        'visual_boundary':'This script verifies real HTTP only; browser_checks.json separately records actual UI observations.'}
    save(OUT/'analysis/live_api_checks.json',proof);assert proof['all_checks_passed'];print(json.dumps(proof,ensure_ascii=True))


if __name__=='__main__':main()
