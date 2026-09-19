"""Scoped source checks and isolated build artifacts for construction client."""
import subprocess,sys
from rdc_construction_common import *


def main():
    start=time.monotonic();out=BASE/'client/code_checks';records=[]
    scripts=sorted((ROOT/'tests/glm5').glob('phase2745_rdc_construction*.py'))+sorted((ROOT/'tests/glm5').glob('rdc_construction*.py'))
    scripts+=sorted((ROOT/'tests/glm5').glob('phase2746_rdc_*.py'))+[ROOT/'tests/glm5/rdc_native_attention_parameters.py',ROOT/'tests/glm5/rdc_runtime_observer.py']
    scripts += [ROOT/'tests/glm5/rdc_native_tail.py',ROOT/'tests/glm5/rdc_history_prediction.py']
    scripts += [ROOT/'server/rdc_construction_service.py',ROOT/'tests/glm5_temp/rdc_construction_api.py']
    node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe');assert node.is_file()
    build=out/('build_'+str(time.time_ns()))
    tasks=[('python_compile',[sys.executable,'-X','utf8','-m','py_compile',*map(str,scripts)],ROOT),
        ('scoped_eslint',[str(node),str(ROOT/'frontend/node_modules/eslint/bin/eslint.js'),'src/components/app/RdcConstructionAtlas.jsx','src/components/app/RdcRuntimeAtlas.jsx'],ROOT/'frontend'),
        ('production_build',[str(node),str(ROOT/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(build)],ROOT/'frontend')]
    for name,cmd,cwd in tasks:
        tick=time.monotonic();p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        rec={'name':name,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-tick,'stdout':p.stdout,'stderr':p.stderr};records.append(rec)
        save(out/(name+'_'+str(time.time_ns())+'.json'),rec)
        print('CONSTRUCTION_CODE_CHECK',name,p.returncode,flush=True);assert p.returncode==0,rec
    save(BASE/'client/code_checks.json',{'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'checks':records,
        'source_versions':[snapshot(p) for p in scripts+[ROOT/'frontend/src/components/app/RdcConstructionAtlas.jsx',ROOT/'frontend/src/components/app/RdcConstructionAtlas.css',ROOT/'frontend/src/components/app/RdcRuntimeAtlas.jsx']],
        'build_output':str(build.relative_to(BASE)),'seconds':time.monotonic()-start})


if __name__=='__main__':main()
