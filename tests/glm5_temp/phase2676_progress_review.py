"""Read-only interim numeric health check; no selection or threshold changes."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
FP=RESULT/'phase2674_native_mlp_scalar';OUT=RESULT/'phase2676_native_mlp_delivery'

def main():
    text=(FP/'analysis/records.jsonl').read_text(encoding='utf-8');lines=text.splitlines(keepends=True);rr=[json.loads(s) for s in lines if s.endswith('\n')];sites=read(FP/'protocol/frozen.json')['sites'];summary={}
    for kind in ('gate','up','down'):
        for control in ('ordinary','low'):
            ee=[e for r in rr for e in r['effects'] if e['kind']=='single' and (sites[e['indices'][0]]['kind'],sites[e['indices'][0]]['control'])==(kind,control)];den=sum(abs(e['effect']) for e in ee)
            summary[kind+'/'+control]={'n':len(ee),'mean_abs_effect':den/len(ee) if ee else None,'fulltoken_relative_L1':sum(abs(e['effect']-e['predicted']) for e in ee)/den if den else None,
                'promptlast_only_relative_L1':sum(abs(e['effect']-e['prompt_last_only']) for e in ee)/den if den else None}
    report={'completed_prefixes':len(rr),'planned':128,'summary':summary,'all_noops_under_1e5':all(r['noop_error']<1e-5 for r in rr),
        'scope':'Partial immutable healthcheck; no candidate selection or experiment changes based on this report. Scientificphase not complete.'}
    save(OUT/f'analysis/interim/scalar_{len(rr):03d}.json',report);print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
