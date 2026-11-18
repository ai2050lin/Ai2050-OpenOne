"""Package the verified web build without touching runtime data or credentials."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
from datetime import datetime

root=Path(__file__).resolve().parents[2]
out=root/'tests/glm5/result/mlp_condition_v2'
source=out/'web-release'
files=sorted(p for p in source.rglob('*') if p.is_file())
assert (source/'index.html').is_file()
manifest={'schema':'ai2050_client_web_release.v1','created_at':datetime.now().astimezone().isoformat(),
    'feature':'MLP-CONDITIONAL-PATHS-v2','client_version':json.loads((root/'frontend/package.json').read_text(encoding='utf-8'))['version'],
    'center_compatibility':'1.0.29 public research artifact protocol',
    'files':[{'path':'frontend/dist/'+p.relative_to(source).as_posix(),
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]}
raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
archive=root/'release/ai2050-client-conditional-v2-web.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p,entry in zip(files,manifest['files']):tar.add(p,arcname=entry['path'])
    item=tarfile.TarInfo('CLIENT_CONDITIONAL_v2_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    tar.add(root/'deploy/README_MLP_CONDITIONAL.md',arcname='CLIENT_CONDITIONAL_v2_README.md')
with tarfile.open(archive) as tar:
    for entry in manifest['files']:
        assert hashlib.sha256(tar.extractfile(entry['path']).read()).hexdigest()==entry['sha256']
record={'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'bytes':archive.stat().st_size,'files':len(files),'manifest':manifest}
(out/'client-release.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print({k:v for k,v in record.items() if k!='manifest'})
