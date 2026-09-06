"""Check the real existing backend without loading a research model."""
import json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2654_output_function_delivery import OUT


def main():
    base='http://127.0.0.1:5001/api/research-assets/'
    def get(p):
        with urllib.request.urlopen(base+p,timeout=30) as r:return json.load(r)
    options=get('native-output-cases');value=get('native-output-parameter');health=get('health')
    req=urllib.request.Request(base+'file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(req,timeout=30) as r:status=r.status;size=len(r.read())
    startup=(OUT/'analysis/backend_cpu_stdout.log').read_text(encoding='utf-8',errors='replace')
    checks={'real64_case_options':len(options['cases'])==64,'real_parameter_query':value['module']=='v_proj' and len(value['tokens'])>0,
        'real_backend_asset_root':health['available'],'real_range206':status==206 and size==1024,
        'model_load_skipped':'starting API server without local model' in startup}
    proof={'checks':checks,'all_checks_passed':all(checks.values()),'base':base,'bind':'127.0.0.1:5001','frontend':'http://localhost:5173',
        'launch':'Original server/server.py, hidden background process,AI2050_SKIP_MODEL_LOAD=1,CUDA_VISIBLE_DEVICES=-1,HF/Transformers offline, CPU-only backend, no research model resident.',
        'visual_boundary':'Real HTTP checked; browser interaction and screenshot QA not performed.'}
    save(OUT/'analysis/live_api_checks.json',proof);assert proof['all_checks_passed']
    marker='**Phase2654 本地客户端联通补充**';text=MEMO.read_text(encoding='utf-8-sig')
    if marker not in text:
        with MEMO.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+' ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+']。前端5173已有进程，后端5001原未启动；以跳过模型加载、离线、仅127.0.0.1监听的方式启动项目原有server/server.py隐藏后台服务。真实HTTP验证64示例、单参数读取、资产根目录和Range206通过；不占用实验模型显存，不声称完成浏览器视觉验收。相关日志与live_api_checks.json均位于本Phase analysis。\n')
    cpu_marker='**Phase2654 后端显存隔离补充**'
    if cpu_marker not in text:
        with MEMO.open('a',encoding='utf-8') as f:f.write('\n\n'+cpu_marker+' ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+']。进一步检查发现默认演示组件也会初始化CUDA，因此仅停止本轮启动的后端进程，并以CUDA_VISIBLE_DEVICES=-1重新启动为CPU-only后端。实验模型未加载；真实HTTP再次通过。后续CUDA研究不应将该读图服务误判为正在运行的实验。\n')
    print(json.dumps(proof,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
