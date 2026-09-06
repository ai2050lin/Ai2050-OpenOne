"""Continue checked source stages after the ONE existing GPU collector exits."""
import argparse,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,MEMO,read,save
OUT=RESULT/'phase2684_source_campaign_delivery'


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream-pid',type=int,required=True);args=p.parse_args()
    OUT.joinpath('runtime').mkdir(parents=True,exist_ok=True)
    process=psutil.Process(args.upstream_pid)
    assert any('phase2678_padded_source_field.py' in s for s in process.cmdline()),process.cmdline()
    created=process.create_time()
    save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'waiting_for_existing2678','upstream_pid':args.upstream_pid,'upstream_created':created})
    while process.is_running() and process.create_time()==created:time.sleep(10)
    upstream=RESULT/'phase2678_padded_source_field/analysis/final.json'
    if not upstream.exists() or not read(upstream)['all_checks_passed']:
        save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'upstream_failed_or_incomplete','upstream_pid':args.upstream_pid});raise SystemExit(1)
    assert '\n## Phase 2678:' in MEMO.read_text(encoding='utf-8-sig')
    stages=[(2679,'phase2679_native_source_ledger.py'),(2680,'phase2680_native_mlp_source_paths.py')]
    for phase,name in stages:
        completed=list(RESULT.glob(f'phase{phase}_*/analysis/final.json'))
        if completed:
            assert len(completed)==1 and read(completed[0])['all_checks_passed'];continue
        log=OUT/f'runtime/{name[:-3]}.log'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tests/glm5'/name)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'running','phase':phase,'child_pid':child.pid,'log':str(log)})
            code=child.wait()
        if code:
            save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'failed','phase':phase,'exit_code':code,'log':str(log)});raise SystemExit(code)
        print('SOURCE STAGE COMPLETED',phase,flush=True)
    save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'2679_2680_complete','remaining':'2681..2684 not completed. Continue actualwholeplan via existingheartbeat; do not write futurephase completion or performcleanup.'})


if __name__=='__main__':main()
