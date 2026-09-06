"""Real CPU-only backend HTTP checks; does not initialize an experiment model."""
import sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2661_sequence_coordinate_delivery import OUT


def main():
    base='http://127.0.0.1:5001/api/research-assets/'
    def get(path):
        with urllib.request.urlopen(base+path,timeout=60) as response:return json.load(response)
    options=get('native-sequence-cases');value=get('native-sequence-parameter');old=get('native-output-parameter');health=get('health')
    request=urllib.request.Request(base+'file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(request,timeout=30) as response:status=response.status;size=len(response.read())
    startup=(OUT/'analysis/backend_cpu_stdout.log').read_text(encoding='utf-8',errors='replace')
    checks={'real64_sequence_cases':len(options['cases'])==64,'real_default_case256':value['case_index']==256,
        'real_two_sequence_branches':len(value['branches'])==2 and all(len(b['tokens'])>0 for b in value['branches']),
        'real_parameter_difference':value['parameter_derivative']==value['branches'][0]['derivative']-value['branches'][1]['derivative'],
        'prior_output_api_preserved':old['module']=='v_proj','asset_root':health['available'],'range206':status==206 and size==1024,
        'model_load_skipped':'starting API server without local model' in startup}
    proof={'checks':checks,'all_checks_passed':all(checks.values()),'base':base,'frontend':'http://localhost:5173',
        'launch':'Original server/server.py, hidden, CPU-only CUDA_VISIBLE_DEVICES=-1, skip model load, offline, localhost only.',
        'visual_boundary':'Real HTTP, not browser interaction or screenshot QA.'}
    save(OUT/'analysis/live_api_checks.json',proof);assert proof['all_checks_passed'];print(json.dumps(proof,ensure_ascii=True))


if __name__=='__main__':main()
