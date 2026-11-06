"""Bounded native float safetensors reads shared by desktop node and center. No inference."""
import hashlib
import json
import math
import struct
import threading
from pathlib import Path

# Same file selection and composition as dist_protocol.runner_v3.model_snapshot.
# Verification is explicit; catalogue/header reads never hash multi-GB weights.
SNAPSHOT_PATTERNS = ('*.safetensors', '*.bin', '*.pt', '*.gguf', 'config.json', 'generation_config.json',
                     'tokenizer*', 'vocab*', 'merges.txt', 'special_tokens_map.json', '*.model', '*.tiktoken')
_snapshot_cache = {}
_snapshot_lock = threading.Lock()


def model_snapshot(folder):
    folder=Path(folder).resolve()
    def selected_paths():
        return sorted({p for pattern in SNAPSHOT_PATTERNS for p in folder.glob(pattern) if p.is_file()},key=lambda p:p.name)
    paths=selected_paths()
    if not paths:raise ValueError('模型目录缺少可核验文件')
    def signature():
        return tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in paths)
    before=signature()
    with _snapshot_lock:
        cached=_snapshot_cache.get(str(folder))
        if cached and cached[0]==before:return cached[1]
        files={}
        for p in paths:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(4 << 20),b''):h.update(block)
            files[p.name]={'sha256':h.hexdigest(),'size':p.stat().st_size}
        after=signature()
        if before!=after or paths!=selected_paths():raise ValueError('核验期间模型文件发生变化，请重试')
        result={'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode('utf-8')).hexdigest(),
                'n_files':len(files),'files':files,'source':'full_file_sha256','inference':False}
        _snapshot_cache[str(folder)]=(after,result)
        return result

def catalog(roots):
    models=[];seen=set()
    for root in roots:
        if not root.is_dir():continue
        # Bounded depth includes publisher/model directories, not arbitrary recursive file trees.
        for pattern in ('config.json','*/config.json','*/*/config.json'):
            for path in root.glob(pattern):
                folder=path.parent.resolve()
                if folder in seen:continue
                seen.add(folder)
                try:
                    cfg=json.loads(path.read_text(encoding='utf-8'))
                    keys=('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','vocab_size')
                    if not isinstance(cfg,dict) or cfg.get('quantization_config') or not all(type(cfg.get(k)) is int and cfg[k]>0 for k in keys):continue
                    cfg.setdefault('num_key_value_heads',cfg['num_attention_heads'])
                    cfg.setdefault('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
                    if cfg['num_attention_heads']%cfg['num_key_value_heads']:continue
                    files=sorted(folder.glob('*.safetensors'))
                    if not files:continue
                    identity=hashlib.sha256(str(folder).encode()).hexdigest()[:24]
                    models.append({'id':identity,'name':folder.name,'config':cfg,'_path':folder})
                except (OSError,ValueError):continue
    priority={'qwen3-4b':0,'glm-4-9b-chat-hf':1,'qwen3-14b':2}
    return sorted(models,key=lambda m:(priority.get(m['name'].lower(),3),m['name'],m['id']))


def header(path):
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Invalid safetensors header')
        n=struct.unpack('<Q',raw)[0]
        if n>32*1024*1024 or n>path.stat().st_size-8:raise ValueError('Invalid header size')
        data=f.read(n)
    h=json.loads(data)
    if not isinstance(h,dict):raise ValueError('Invalid tensor directory')
    h.pop('__metadata__',None)
    return h,n+8,hashlib.sha256(data).hexdigest()


def read_slice(folder,name,row,col,rows,cols):
    if row<0 or col<0 or not 1<=rows<=64 or not 1<=cols<=64 or rows*cols>4096:
        raise ValueError('切片越界或超过4096标量')
    for path in sorted(Path(folder).glob('*.safetensors')):
        h,base,fingerprint=header(path)
        if name not in h:continue
        meta=h[name];shape=meta['shape'];rank=len(shape)
        if rank not in (1,2):raise ValueError('当前浏览器支持一维或二维参数')
        if not all(type(x) is int and x>0 for x in shape):raise ValueError('Invalid tensor shape')
        nr,nc=(shape[0],shape[1]) if rank==2 else (shape[0],1)
        if row>=nr or col>=nc:raise ValueError('参数索引超出矩阵范围')
        dt={'BF16':'H','F16':'e','F32':'f','F64':'d'}.get(meta['dtype'])
        if not dt:raise ValueError('仅支持原生浮点权重')
        size=struct.calcsize('<'+dt);r=min(rows,nr-row);c=min(cols,nc-col)
        start,end=meta['data_offsets']
        if type(start) is not int or type(end) is not int or start<0 or end<start or end-start!=nr*nc*size or base+end>path.stat().st_size:raise ValueError('权重文件长度与形状不符')
        result=[]
        with path.open('rb') as f:
            for i in range(r):
                f.seek(base+start+((row+i)*nc+col)*size)
                raw=f.read(c*size)
                if len(raw)!=c*size:raise ValueError('权重文件被截断')
                values=list(struct.unpack('<'+dt*c,raw))
                if meta['dtype']=='BF16':values=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in values]
                if not all(math.isfinite(v) for v in values):raise ValueError('参数包含非有限值')
                result.append(values)
        return {'name':name,'shape':shape,'dtype':meta['dtype'],'start':[row,col],'actual_shape':[r,c],
                'values':result,'loaded_numel':r*c,'source':'safetensors_exact_slice',
                'header_sha256':fingerprint,'file':path.name}
    raise FileNotFoundError(name)


