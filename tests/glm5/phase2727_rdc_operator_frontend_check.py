"""Reproducible frontend build/lint receipt with exact source and artifact hashes."""
import py_compile
import subprocess
from rdc_operator_common import *


def main():
    start=time.monotonic();node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe')
    assert node.is_file()
    scripts=sorted({*list((ROOT/'tests/glm5').glob('phase272[4-7]_rdc*.py')),
        *list((ROOT/'tests/glm5').glob('rdc_operator*.py')),ROOT/'tests/glm5/rdc_native_conditional_operator.py'})
    for path in scripts:py_compile.compile(str(path),doraise=True)
    front=ROOT/'frontend';commands=[
        [str(node),'node_modules/eslint/bin/eslint.js','src/components/app/RdcOperatorAtlas.jsx'],
        [str(node),'node_modules/vite/bin/vite.js','build']]
    receipts=[]
    for cmd in commands:
        result=subprocess.run(cmd,cwd=front,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        receipts.append({'command':cmd,'cwd':str(front),'returncode':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:
            save(BASE/'verification/frontend_build_failure.json',{'timestamp':stamp(),'passed':False,'receipts':receipts})
            raise AssertionError(receipts[-1])
    sources=[front/'src/components/app/RdcOperatorAtlas.jsx',front/'src/components/app/RdcOperatorAtlas.css',
        front/'src/components/app/RdcJointAtlas.jsx',front/'src/App.jsx',ROOT/'server/rdc_operator_service.py',ROOT/'server/server.py']
    result={'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),
        'compiled_python_files':{str(p.relative_to(ROOT)):sha(p) for p in scripts},
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'receipts':receipts,
        'build_artifact_sha256':{str(p.relative_to(front)):sha(p) for p in sorted((front/'dist').rglob('*')) if p.is_file()},
        'scope':'Actual Python compilation, ESLint and production Vite build. Existing large-bundle warning is retained; browser interaction and scientific verification are separate.'}
    save(BASE/'verification/frontend_build.json',result);ledger('frontend_lint_production_build',time.monotonic()-start)
    print('OPERATOR_FRONTEND_BUILD_PASS',len(scripts),flush=True)


if __name__=='__main__':main()
