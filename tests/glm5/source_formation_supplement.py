"""CPU-only completion of frozen supplementary analyses as native artifacts arrive."""
import contextlib,gc,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
import source_formation_parameters as P
import source_formation_content_controls as C
import source_formation_error_split as E

def invoke(label,fn,*args):
    path=D.OUT/(label+'-supplement-console.log');assert not path.exists(),path
    print('START '+label,flush=True);started=time.monotonic()
    with path.open('w',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):fn(*args)
    print(f'COMPLETE {label} {time.monotonic()-started:.1f}s',flush=True);gc.collect()

def run():
    timer=time.monotonic();completed=[]
    while True:
        for model in D.MODELS:
            root=D.OUT/model
            if not (root/'formation-fit.json').exists():continue
            for artifact,fn in [('native-parameter-fit',lambda:P.execute(model,'discovery',True)),('content-control-fit',lambda:C.fit(model))]:
                if not (root/(artifact+'.json')).exists():
                    assert not (root/'confirm-capture-started.json').exists(),'Secondary fit must precede confirmation native capture'
                    invoke(model+'-'+artifact,fn);completed.append(model+'/'+artifact)
            for split in D.COUNTS:
                if not (root/f'{split}-formation.json').exists():continue
                for suffix,fn in [('native-parameters',lambda:P.execute(model,split)),('content-controls',lambda:C.evaluate(model,split)),('error-split',lambda:E.evaluate(model,split))]:
                    if suffix=='error-split' and split=='discovery':continue
                    if not (root/f'{split}-{suffix}.json').exists():
                        invoke(model+'-'+split+'-'+suffix,fn);completed.append(model+'/'+split+'/'+suffix)
        if (D.OUT/'campaign-complete.json').exists():break
        print(f'WAIT primary artifacts; supplemental stages completed={len(completed)} elapsed={time.monotonic()-timer:.1f}s',flush=True)
        time.sleep(30)
    D.save('supplement-complete.json',{'completed':True,'executed':completed,'code_sha256':D.sha(Path(__file__)),'GPU_used':False})

if __name__=='__main__':run()
