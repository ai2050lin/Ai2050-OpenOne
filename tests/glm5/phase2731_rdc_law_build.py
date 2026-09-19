"""Scoped static checks and actual frontend production build with warning receipts."""
import py_compile
import subprocess
from rdc_law_common import *


def main():
    start=time.monotonic();out=BASE/'verification';node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe')
    assert node.is_file()
    if not (out/'frontend_build.json').exists() and (out/'production_build.json').exists() and not (out/'build_console_encoding_recovery.json').exists():
        old=[read(out/f'{name}.json') for name in ('scoped_eslint','production_build')]
        seconds=sum(r['seconds'] for r in old)+1
        save(out/'build_console_encoding_recovery.json',{'timestamp':stamp(),'completed_command_receipts':old,
            'failure':'Both commands returned0. Printing Unicode buildoutput to GBK console raised UnicodeEncodeError before combinedreceipt.',
            'booked_seconds':seconds,'timing':'Measured commandtimers plus1second allowance for wrapperoverhead. Fullchecks rerun below.'})
        ledger('build_console_logging_interrupted_attempt',seconds)
    commands=[('scoped_eslint',[str(node),'node_modules/eslint/bin/eslint.js','src/components/app/RdcLawAtlas.jsx','src/components/app/RdcOperatorAtlas.jsx']),
              ('production_build',[str(node),'node_modules/vite/bin/vite.js','build'])]
    reports=[]
    for kind,cmd in commands:
        t0=time.monotonic();r=subprocess.run(cmd,cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        save(out/f'{kind}.json',{'timestamp':stamp(),'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'seconds':time.monotonic()-t0})
        print(kind,r.returncode,'full_stdout_stderr_saved','stderr_nonempty',bool(r.stderr.strip()),flush=True)
        assert r.returncode==0,kind
        reports.append({'name':kind,'returncode':r.returncode,'receipt':f'verification/{kind}.json','seconds':time.monotonic()-t0,'stderr_nonempty':bool(r.stderr.strip())})
    scripts=sorted(set((ROOT/'tests/glm5').glob('*rdc_law*.py'))|{ROOT/'server/rdc_law_service.py'})
    for path in scripts:py_compile.compile(str(path),doraise=True)
    client=[ROOT/'frontend/src/components/app/RdcLawAtlas.jsx',ROOT/'frontend/src/components/app/RdcLawAtlas.css',
        ROOT/'frontend/src/components/app/RdcOperatorAtlas.jsx',ROOT/'frontend/src/main.jsx',ROOT/'server/server.py',ROOT/'server/rdc_law_service.py']
    r={'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,'checks':reports,'seconds':time.monotonic()-start,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in client},
        'compiled_python_files':{p.relative_to(ROOT).as_posix():sha(p) for p in scripts},
        'scope':'Actual existing frontend full productionbuild and scoped ESLint; warnings/stderr retained verbatim in JSONreceipts. No claim that allunrelated existingcode passedlint.'}
    save(out/'frontend_build.json',r);ledger('law_frontend_build_static_checks',r['seconds'])


if __name__=='__main__':main()
