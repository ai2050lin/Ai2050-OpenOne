"""Frozen independent behavior confirmation for pilot-failed strata, no fields."""
import argparse,time
import b36_composition14 as B
import b36_composition14_calibrate as Engine

PARENT=B.OUT


def prepare():
    B.configure();scope=B.read(PARENT/'scope-plan.json');data=B.read(PARENT/'materials.json')['rows']
    excluded=set(sum((scope['excluded_row_ids'][s] for s in ['confirm','format','order']),[]))
    rows=[dict(r,variant=r['split']) for r in data if r['id'] in excluded];assert len(rows)==1220
    B.OUT=PARENT/'behavior-only';B.configure();assert not (B.OUT/'calibration-plan.json').exists()
    B.save('calibration-plan.json',dict(rows=rows,n=len(rows),source_sha256=B.file_sha(Engine.__file__),generator_sha256=B.file_sha(__file__),length=B.LENGTH,batch=B.BATCH,
        exposure='Previously unopened confirmation families. Only scores, no native fields for these failed pilot strata. No fitting or intervention permitted.',
        source_scope_sha256=B.file_sha(PARENT/'scope-plan.json')))


def run():
    from b36_composition14_tail import admit
    admit();B.OUT=PARENT/'behavior-only';B.configure();start=time.monotonic()
    while True:
        try:Engine.run();break
        except RuntimeError as e:
            if str(e)!='GPU is leased by another research/live process' or time.monotonic()-start>3600:raise
            B.log('waiting_behavior_only',0,1220,start);time.sleep(15)


def finish():
    B.configure()
    from b36_composition14_tail import admit
    from b36_composition14_collect import behavior
    admit();raw=B.read(PARENT/'behavior-only/calibration-results.json');lookup={r['id']:r for r in raw['rows']}
    allrows=B.read(PARENT/'materials.json')['rows'];reports={};all_scored=[]
    for split in ['confirm','format','order']:
        if (PARENT/split/'collection.json').exists():
            native=B.read(PARENT/split/'collection.json')
            lookup.update({id_:dict(scores=s) for id_,s in zip(native['row_ids'],native['scores'])})
        rows=[r for r in allrows if r['split']==split];scores=[lookup[r['id']]['scores'] for r in rows]
        reports[split]=behavior(rows,scores)
        all_scored += [dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],title=r['title'],sequence=r['sequence'],split=split,format=r['format'],scores=s,gold=r['gold']) for r,s in zip(rows,scores)]
    order=[]
    pairs=[r for r in all_scored if r['split']=='order' or (r['split']=='confirm' and r['mode']=='program' and r['title']==0 and r['depth']>1)]
    for key in sorted({(r['branch'],r['depth'],r['sequence']) for r in pairs}):
        rr=[r for r in pairs if (r['branch'],r['depth'],r['sequence'])==key]
        correct=sum(r['scores'][r['gold']]>max(v for k,v in enumerate(r['scores']) if k!=r['gold']) for r in rr)
        order.append(dict(zip(['branch','depth','sequence'],key),n=len(rr),correct=correct,accuracy=correct/len(rr),passed=correct/len(rr)>=.9))
    B.save('full-behavior-confirmation.json',dict(kind='real',reports=reports,paired_order=order,rows=all_scored,source_sha256=B.file_sha(__file__),
        scope='Full originally proposed behavior grid, including prior pilot failures. These outputs cannot reopen failed pilot strata for internal analysis or retroactively promote v2.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','finish']);a=p.parse_args();globals()[a.stage]()
