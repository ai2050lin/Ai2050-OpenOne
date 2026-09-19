"""Read-only BF16 meta-device allocation estimates; no model weights or edits."""
import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import torch,psutil,shutil
from transformers import AutoConfig,AutoModelForCausalLM
from accelerate import init_empty_weights,infer_auto_device_map
from phase2620_native_coordinate_contract import *

def main():
    folder=ROOT/'models/hf/Qwen3-14B';cfg=AutoConfig.from_pretrained(folder,local_files_only=True)
    with init_empty_weights():model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16)
    model.tie_weights();assert all(p.is_meta for p in model.parameters())
    index=read(folder/'model.safetensors.index.json');reports=[]
    for gpu,cpu in ((13,11),(14,11),(14,11.5),(14,12),(14,12.5)):
        limits={0:int(gpu*1024**3),'cpu':int(cpu*1024**3)}
        mapping=infer_auto_device_map(model,max_memory=limits,no_split_module_classes=model._no_split_modules,dtype=torch.bfloat16,offload_buffers=True)
        totals={};physical={}
        for name,p in model.named_parameters():
            matches=[k for k in mapping if name==k or name.startswith(k+'.') or k==''];assert matches
            device=str(mapping[max(matches,key=len)]);totals[device]=totals.get(device,0)+p.numel()*2;physical[name]=device
        reserve=read(RESULT/'phase2691_crossmodel_role_confirmation/protocol/frozen.json')['models']['qwen14']['uncompressed_upper']
        report={'max_memory_bytes':limits,'device_map':mapping,'parameter_bytes':totals,
            'disk_parameter_bytes':totals.get('disk',0),'disk_free':shutil.disk_usage(RESULT).free,
            'model_result_upper':reserve,'minimum_disk_floor':8*1024**3,
            'disk_margin_after_result_and_offload':shutil.disk_usage(RESULT).free-totals.get('disk',0)-reserve-8*1024**3,
            'host_available_in_meta_process':psutil.virtual_memory().available,'CPU_weight_bytes':totals.get('cpu',0),
            'all_parameters_meta':True,'weights_loaded':False}
        reports.append(report);print({k:v for k,v in report.items() if k!='device_map'},flush=True)
    save(RESULT/'phase2691_crossmodel_role_confirmation/analysis/meta_memory_plans.json',reports)

if __name__=='__main__':main()
