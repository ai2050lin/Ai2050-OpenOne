"""Record the actual production build, scoped lint and Python compilation."""
import subprocess
import py_compile
from rdc_binding_common import *

def main():
    start=time.monotonic();node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe');assert node.is_file()
    files=sorted((ROOT/'tests/glm5').glob('*rdc_binding*.py'))+[ROOT/'server/rdc_binding_service.py',ROOT/'server/server.py']
    for p in files:py_compile.compile(str(p),doraise=True)
    reports=[]
    for label,command in [('scoped_eslint',[str(node),'node_modules/eslint/bin/eslint.js','src/components/app/RdcBindingAtlas.jsx','src/main.jsx']),
      ('production_vite',[str(node),'node_modules/vite/bin/vite.js','build'])]:
        tick=time.monotonic();r=subprocess.run(command,cwd=ROOT/'frontend',capture_output=True,timeout=120)
        record={'kind':label,'command':command,'returncode':r.returncode,'stdout':r.stdout.decode('utf-8','replace'),
          'stderr':r.stderr.decode('utf-8','replace'),'seconds':time.monotonic()-tick};reports.append(record)
        assert r.returncode==0,record
    sources=files+[ROOT/'frontend/src/components/app/RdcBindingAtlas.jsx',ROOT/'frontend/src/components/app/RdcBindingAtlas.css',ROOT/'frontend/src/main.jsx']
    save(BASE/'verification/build.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'reports':reports,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'seconds':time.monotonic()-start,
      'scope':'Preexisting main bundle exceeds500kB warning remains; no package upgrades performed. Bare npm unavailable, existing explicit Node executable used.'})
    ledger('binding_build_and_compile',time.monotonic()-start)
    print('BINDING_BUILD_PASS',len(files),flush=True)

if __name__=='__main__':main()
