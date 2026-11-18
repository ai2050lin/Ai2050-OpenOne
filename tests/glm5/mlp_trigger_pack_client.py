"""Package a same-origin web release, retaining per-file verification hashes."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
from datetime import datetime

root=Path(__file__).resolve().parents[2]
out=root/'tests/glm5/result/mlp_trigger_v1'
source=out/'web-release'
files=sorted(p for p in source.rglob('*') if p.is_file())
assert (source/'index.html').is_file()
manifest={'schema':'ai2050_client_web_release.v1','created_at':datetime.now().astimezone().isoformat(),
 'feature':'MLP-TRIGGER-UPSTREAM-v1 + MLP-TRIGGER-READ100-v1',
 'client_version':json.loads((root/'frontend/package.json').read_text(encoding='utf-8'))['version'],
 'center_compatibility':'1.0.29 public research artifact protocol',
 'api_origin':'same-origin',
 'files':[{'path':'frontend/dist/'+p.relative_to(source).as_posix(),
   'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]}
raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
archive=out/'packages/ai2050-client-mlp-trigger-web.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p,entry in zip(files,manifest['files']):tar.add(p,arcname=entry['path'])
 item=tarfile.TarInfo('CLIENT_TRIGGER_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 tar.add(root/'deploy/README_MLP_TRIGGER.md',arcname='CLIENT_TRIGGER_README.md')
with tarfile.open(archive) as tar:
 for entry in manifest['files']:
  assert hashlib.sha256(tar.extractfile(entry['path']).read()).hexdigest()==entry['sha256']
record={'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'bytes':archive.stat().st_size,'files':len(files),'manifest':manifest}
(out/'client-release.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print({k:v for k,v in record.items() if k!='manifest'})
