"""Actual local API, production build and isolated browser verification with saved logs."""
import os,subprocess,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['AI2050_SKIP_MODEL_LOAD']='1'
from rdc_conditional_common import *
NODE=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')


def main():
    commands=[('api',[sys.executable,'tests/glm5/phase2706_rdc_client_api_checks.py'],ROOT),
      ('eslint',[str(NODE),'node_modules/eslint/bin/eslint.js','src/components/app/RdcFeatureAtlas.jsx'],ROOT/'frontend'),
      ('vite_build',[str(NODE),'node_modules/vite/bin/vite.js','build'],ROOT/'frontend'),
      ('browser',[str(NODE),'tests/glm5_temp/rdc_conditional_client_test.cjs','--full'],ROOT)]
    results=[]
    for name,cmd,cwd in commands:
        started=time.monotonic();p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        result={'name':name,'command':cmd,'cwd':str(cwd),'exit_code':p.returncode,'seconds':time.monotonic()-started,'stdout':p.stdout,'stderr':p.stderr};results.append(result)
        save(CAMPAIGN/f'verification/{name}.json',result);print('CLIENT_VERIFICATION',name,p.returncode,flush=True)
        assert p.returncode==0,(name,p.stdout[-1500:],p.stderr[-1500:])
    save(CAMPAIGN/'client_verification.json',{'timestamp':stamp(),'passed':True,'source_sha':sha(Path(__file__)),'checks':results,
      'scope':'Read-only app API, selected-component ESLint, real Vite production build and isolated headless development-browser interactions; no user browser/profile or model execution.'})


if __name__=='__main__':main()
