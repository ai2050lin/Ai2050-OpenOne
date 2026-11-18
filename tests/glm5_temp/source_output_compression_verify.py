"""Close the interrupted compression experiment by checking every touched file."""
import ctypes,json
from datetime import datetime
from pathlib import Path
import source_output_core as S
from source_output_core import C

api=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
api.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)];api.restype=ctypes.c_uint32
lines=[json.loads(s) for s in (C.OUT/'storage-compression.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
records=[]
for item in lines:
    if item['stage']!='before_compression':continue
    path=Path(item['path']);assert path.resolve().is_relative_to((C.ROOT/'tests/glm5/result/branch_style_joint_v1').resolve())
    digest=C.sha(path);assert digest==item['sha256'];high=ctypes.c_uint32(0);low=api(str(path),ctypes.byref(high));allocated=(high.value<<32)|low
    records.append({'path':str(path),'sha256':digest,'content_unchanged':True,'allocated_before':item['allocated_bytes'],'allocated_after':allocated,
        'released_bytes':item['allocated_bytes']-allocated})
C.save('storage-compression-stopped.json',{'schema':'source_output_storage_compression.v1','execution_only':True,'time':datetime.now().astimezone().isoformat(),
    'reason':'The first completed 3.88 GiB native array saved only 1.5%; second saved zero. Stop further low-yield IO during the other study.',
    'all_touched_files_verified':True,'records':records,'released_bytes':sum(r['released_bytes'] for r in records),'files_deleted':0})
print(json.dumps({'files_verified':len(records),'released_MiB':sum(r['released_bytes'] for r in records)/2**20}),flush=True)
