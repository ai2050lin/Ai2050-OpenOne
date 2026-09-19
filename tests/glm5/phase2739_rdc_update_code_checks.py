"""Scoped Python/JS checks and a fresh full-client production build."""
import subprocess,sys
from rdc_update_common import *


def main():
    out=BASE/'client/code_checks';start=time.monotonic();records=[]
    scripts=sorted({p for phase in range(2736,2740) for p in (ROOT/'tests/glm5').glob(f'phase{phase}_rdc_update*.py')})
    scripts+=sorted((ROOT/'tests/glm5').glob('rdc_update*.py'))+[ROOT/'server/rdc_update_service.py']
    node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe');assert node.is_file()
    build=out/('build_'+str(time.time_ns()))
    tasks=[('python_compile',[sys.executable,'-X','utf8','-m','py_compile']+[str(p) for p in scripts],ROOT),
      ('scoring_regression',[sys.executable,'-X','utf8',str(ROOT/'tests/glm5/rdc_update_terminal_audit.py')],ROOT),
      ('scoped_eslint',[str(node),str(ROOT/'frontend/node_modules/eslint/bin/eslint.js'),'src/components/app/RdcUpdateAtlas.jsx'],ROOT/'frontend'),
      ('production_build',[str(node),str(ROOT/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(build)],ROOT/'frontend')]
    for name,cmd,cwd in tasks:
        tick=time.monotonic();p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        rec={'name':name,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-tick,'stdout':p.stdout,'stderr':p.stderr}
        records.append(rec);save(out/f'{name}.json',rec);print('UPDATE_CODE_CHECK',name,p.returncode,flush=True)
        assert p.returncode==0,rec
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':records,'python_files':len(scripts),
      'build_output':str(build.relative_to(BASE)),'source_versions':[snapshot(p) for p in scripts+[ROOT/'frontend/src/components/app/RdcUpdateAtlas.jsx',ROOT/'frontend/src/components/app/RdcUpdateAtlas.css',ROOT/'frontend/src/main.jsx',ROOT/'server/server.py']],
      'seconds':time.monotonic()-start,'scope':'New research files and standalone route linted; production application bundled into a new result subdirectory. No user build/output directory deleted or replaced.'}
    save(BASE/'client/code_checks.json',result);print('UPDATE_CODE_CHECKS_DONE',flush=True)


if __name__=='__main__':main()
