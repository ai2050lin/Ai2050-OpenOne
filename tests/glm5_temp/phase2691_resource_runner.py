"""Explicit resource amendment: native Q14 auto-map with checkpoint-backed disk.

Formal material/scoring/capture unchanged; original scientific source is retained.
The local loader maps unconverted safetensors directly, not duplicated weights.
"""
import os,sys,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import torch,psutil
from transformers import AutoTokenizer,AutoModelForCausalLM
import transformers.modeling_utils as loading
import phase2691_crossmodel_role_confirmation as run
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2691_crossmodel_role_confirmation'
EXPECTED='27ff66783866fc1585d124a697dc7faaf169af58dab136cb2e4bdfb9bc5323b8'

def load_q14(key):
    assert key=='qwen14' and sha(TESTS/'phase2691_crossmodel_role_confirmation.py')==EXPECTED
    torch.set_num_threads(4);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    folder=(ROOT/'models/hf/Qwen3-14B').resolve();offload=OUT/'runtime/q14_checkpoint_index'
    assert psutil.virtual_memory().available>13*1024**3,'Need13GiB host availability after runtime imports'
    assert shutil.disk_usage(OUT).free>9*1024**3,'No rawcleanup for resource workaround'
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=True,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    safe_before=loading.safe_open;disk_before=loading.accelerate_disk_offload;offload_report={}
    def pread(*args,**kwargs):kwargs['backend']='pread';return safe_before(*args,**kwargs)
    def checked_disk_index(*args,**kwargs):
        index=disk_before(*args,**kwargs)
        assert index
        for name,info in index.items():
            p=Path(info['safetensors_file']).resolve()
            assert p.parent==folder and p.suffix=='.safetensors' and p.is_file() and info['dtype']=='bfloat16'
        offload_report.update({'entries':index,'all_original_checkpoint_paths':True,'duplicated_weight_bytes':0})
        save(OUT/'qwen14/protocol/checkpoint_disk_index.json',offload_report)
        return index
    loading.safe_open=pread;loading.accelerate_disk_offload=checked_disk_index
    before={'host_available':psutil.virtual_memory().available,'disk_free':shutil.disk_usage(OUT).free,'started':datetime.now().astimezone().isoformat()}
    save(OUT/'analysis/resource_runtime_amendment.json',{'scientific_source_sha256':EXPECTED,'wrapper_sha256':sha(Path(__file__)),
        'reason':'Original22GiB freehost guard failed before weightload, no formalcases. KeepBF16/noquantization/auto; reduce CPUresidency and useoriginal safetensors-backed disk index.',
        'max_memory':{'GPU0':'12GiB','cpu':'10GiB'},'device_map':'auto','offload_state_dict':True,'offload_buffers':True,
        'checkpoint_backed_not_duplicate':True,'material_scoring_capture_unchanged':True,'before':before,'completion_claimed':False})
    try:
        model=AutoModelForCausalLM.from_pretrained(folder,dtype=torch.bfloat16,device_map='auto',max_memory={0:'12GiB','cpu':'10GiB'},
            offload_folder=str(offload),offload_state_dict=True,offload_buffers=True,local_files_only=True,
            trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=safe_before;loading.accelerate_disk_offload=disk_before
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    assert offload_report['all_original_checkpoint_paths'] and 'disk' in model.hf_device_map.values()
    assert sum(p.stat().st_size for p in offload.rglob('*') if p.is_file())<16*1024**2,'Unexpected duplicated offload weights'
    report=read(OUT/'analysis/resource_runtime_amendment.json');report.update(weights_loaded=True,actual_device_map=model.hf_device_map,
        after_load={'host_available':psutil.virtual_memory().available,'disk_free':shutil.disk_usage(OUT).free,'time':datetime.now().astimezone().isoformat()},
        no_model_quantization=True,all_offloaded_weights_reference_retained_checkpoints=True)
    save(OUT/'analysis/resource_runtime_amendment.json',report)
    return model,tok

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in (*run.KEYS,'prepare','finalize')
    assert sha(TESTS/'phase2691_crossmodel_role_confirmation.py')==EXPECTED
    old_candidates=run.CANDIDATES
    run.CANDIDATES=[list(r) for r in old_candidates]
    assert run.CANDIDATES==read(OUT/'protocol/frozen.json')['Q14_candidates']
    assert tuple(tuple(r) for r in run.CANDIDATES)==tuple(old_candidates)
    save(OUT/'analysis/candidate_container_audit.json',{'all_checks_passed':True,'scientific_source_sha256':EXPECTED,
        'wrapper_sha256':sha(Path(__file__)),'candidate_values':run.CANDIDATES,
        'correction':'In-memory tuple entries normalized to JSON lists for the original strict equality guard. Addresses, signs, ordering, material and formal source file unchanged.',
        'stage':'Before any model weights loaded or formal conditions observed'})
    action=sys.argv[1]
    if action=='prepare':
        first=run.prepare();second=run.prepare();assert first==second
        print('2691 CANONICAL MATERIAL DOUBLE PREPARE PASSED, no model load',flush=True)
    elif action=='finalize':run.finalize()
    else:
        if action=='qwen14':run.load_native=load_q14
        run.run_one(action)
