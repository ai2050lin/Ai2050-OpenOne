"""Scoped code checks with production build outputs inside current result tree."""
import subprocess,sys
from rdc_query_common import *

def main():
    out=BASE/'client/code_checks';start=time.monotonic();records=[]
    scripts=sorted({p for phase in range(2740,2745) for p in (ROOT/'tests/glm5').glob(f'phase{phase}_rdc_query*.py')})
    scripts+=sorted((ROOT/'tests/glm5').glob('rdc_query*.py'))+[ROOT/'server/rdc_query_service.py',ROOT/'tests/glm5_temp/rdc_query_api.py']
    node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe');assert node.is_file();build=out/('build_'+str(time.time_ns()))
    tasks=[('python_compile',[sys.executable,'-X','utf8','-m','py_compile']+[str(p) for p in scripts],ROOT),
      ('scoring_regression',[sys.executable,'-X','utf8',str(ROOT/'tests/glm5/rdc_query_scoring.py')],ROOT),
      ('scoped_eslint',[str(node),str(ROOT/'frontend/node_modules/eslint/bin/eslint.js'),'src/components/app/RdcQueryAtlas.jsx'],ROOT/'frontend'),
      ('production_build',[str(node),str(ROOT/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(build)],ROOT/'frontend')]
    for name,cmd,cwd in tasks:
        prior=out/f'{name}.json'
        if prior.exists() and read(prior)['exit_code']!=0:save(out/'failures'/(name+'_'+sha(prior)[:16]+'.json'),read(prior))
        tick=time.monotonic();p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        rec={'name':name,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-tick,'stdout':p.stdout,'stderr':p.stderr};records.append(rec)
        save(out/f'{name}.json',rec);print('QUERY_CODE_CHECK',name,p.returncode,flush=True);assert p.returncode==0,rec
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':records,'python_files':len(scripts),'build_output':str(build.relative_to(BASE)),
      'source_versions':[snapshot(p) for p in scripts+[ROOT/'frontend/src/components/app/RdcQueryAtlas.jsx',ROOT/'frontend/src/components/app/RdcQueryAtlas.css',ROOT/'frontend/src/main.jsx']],
      'seconds':time.monotonic()-start,'scope':'No user output directory overwritten. Local registered Node binary used because npm was absent from current shell PATH.'}
    save(BASE/'client/code_checks.json',result);print('QUERY_CODE_CHECKS_DONE',flush=True)

if __name__=='__main__':main()
