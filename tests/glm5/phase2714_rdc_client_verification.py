"""Capture actual client check commands/results in this campaign, preserving older deliveries."""
import subprocess
import sys
import requests
from rdc_prefix_common import *


def main():
    out=CAMPAIGN/'verification';out.mkdir(parents=True,exist_ok=True)
    node=r'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    for attempt in range(15):
        try:
            if requests.get('http://127.0.0.1:5001/api/rdc-prefix/history/overview',timeout=5).status_code==200:break
        except requests.RequestException:pass
        time.sleep(2)
    else:raise RuntimeError('Read-only API startup check did not become ready')
    jobs=[('API',[sys.executable,'-X','utf8','tests/glm5/phase2713_rdc_prefix_client_checks.py'],ROOT),
      ('ESLint',[node,'node_modules/eslint/bin/eslint.js','src/components/app/RdcPrefixAtlas.jsx','src/components/app/RdcFeatureAtlas.jsx','src/main.jsx'],ROOT/'frontend'),
      ('production_build',[node,'node_modules/vite/bin/vite.js','build'],ROOT/'frontend'),
      ('isolated_browser',[node,'tests/glm5_temp/rdc_prefix_client_test.cjs'],ROOT)]
    records=[]
    for name,cmd,cwd in jobs:
        start=time.monotonic()
        with (out/f'{name}.stdout.log').open('w',encoding='utf-8') as so,(out/f'{name}.stderr.log').open('w',encoding='utf-8') as se:
            result=subprocess.run(cmd,cwd=cwd,stdout=so,stderr=se,timeout=180)
        records.append({'name':name,'command':cmd,'cwd':str(cwd),'exit_code':result.returncode,'seconds':time.monotonic()-start})
        save(out/'client_verification.json',{'timestamp':stamp(),'passed':len(records)==4 and all(r['exit_code']==0 for r in records),'records':records})
        print('FINAL_CLIENT',name,result.returncode,flush=True)
        assert result.returncode==0,(name,'See this campaign verification logs')
    assert read(out/'api_checks.json')['passed'] and read(out/'browser_checks.json')['passed']
    guard();print('FINAL_CLIENT_COMPLETE',flush=True)


if __name__=='__main__':main()
