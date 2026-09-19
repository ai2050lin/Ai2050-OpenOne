"""Read-only compact audit of actual completed native-head results."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_question_common import *

parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',required=True)
args=parser.parse_args()
for key in args.models:
    source=OUT/'fit'/key/'readout/nonconfirmation/result.json'
    actual=read(source);assert actual['all_passed']
    analysis=read(OUT/'readout_analysis'/key/'nonconfirmation/result.json')
    counts=[]
    for r in actual['records']:
        ref=r['field'];assert sha(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path']) as z: values=z['statistics']
        row={k:r[k] for k in ['variant','kind','split']}
        for name in ['native_choice','predicted_choice']:
            ids,n=np.unique(values[:,r['columns'].index(name)].astype(np.int64),return_counts=True)
            row[name]={str(i):int(c) for i,c in zip(ids,n)}
        counts.append(row)
    selected=[{'variant':r['variant'],**r['summary']['equal_cohort']} for r in analysis['summaries']
              if r['split']=='diagnostic' and r['kind']=='selected']
    paired=[{'right':r['right'],'metric':r['metric'],**r['paired']['equal_cohort']} for r in analysis['paired_comparisons']
            if r['split']=='diagnostic' and r['left']=='native_source_read:selected']
    _,_,rows,_=material(key)
    teacher={}
    for split in ['validation','diagnostic']:
        ids,n=np.unique([r['tokens']['teacher_ids_including_EOS'][0] for r in rows if r['split']==split],return_counts=True)
        teacher[split]={str(i):int(c) for i,c in zip(ids,n)}
    print(json.dumps({'model':key,'source_sha256':sha(source),'teacher_first_ID_counts':teacher,'counts':counts,'selected':selected,'paired':paired},ensure_ascii=False))
