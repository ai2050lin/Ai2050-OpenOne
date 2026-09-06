"""Read-only progress interpretation; never marks an unfinished phase complete."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
path=root/'tests/glm5/result/phase2682_resolved_scalar_paths/analysis/records.jsonl'
records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
result={'partial':True,'prefixes':len(records),'conditions':sum(len(r['conditions']) for r in records),'groups':{}}
for kind in ('gate','up','down'):
    rows=[c for r in records for c in r['conditions'] if c['kind']==kind]
    den=sum(c['local_actual_l1'] for c in rows);err=sum(c['local_error_l1'] for c in rows)
    result['groups'][kind]={'n':len(rows),'local_error_over_actual_l1':err/den if den else None,'max_branch_error':max(c['branch_max_abs_error'] for c in rows),
        'max_local_coordinate_error':max(c['local_max_abs_error'] for c in rows),'max_abs_output_logprob_change':max(abs(c['output_full_delta']) for c in rows)}
print(json.dumps(result,indent=2))
