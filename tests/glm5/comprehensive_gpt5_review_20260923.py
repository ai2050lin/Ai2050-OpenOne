"""Document audit and CPU rechecks; never loads a model or edits old results."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import json
import re
import numpy as np
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/gpt5_comprehensive_review_20260923'
SOURCE = ROOT / 'research/gpt5/docs/AGI_GPT5_MEMO.md'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    snapshot = OUT / 'source_snapshot.md'
    if snapshot.exists():
        assert snapshot.read_bytes() == raw, 'Source changed; preserve this audit and use a new run directory.'
    else:
        snapshot.write_bytes(raw)
    text = raw.decode('utf-8-sig')
    heads = list(re.finditer(r'^## Phase ([^:：\n]+)[:：].*$', text, re.M))
    phases = []
    for i, m in enumerate(heads):
        body = text[m.end():heads[i+1].start() if i+1 < len(heads) else len(text)]
        phases.append(dict(phase=m.group(1), title=m.group(0),
                           line=text.count('\n', 0, m.start()) + 1,
                           body_characters=len(body)))
    old = ROOT / 'tests/glm5/result/gpt5_memo_audit_20260923/memo_snapshot.md'
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    created_local=datetime.now().astimezone().isoformat(),
                    source=str(SOURCE), sha256=sha(raw), bytes=len(raw),
                    lines=len(text.splitlines()), phase_sections=len(phases),
                    earlier_snapshot_same_lines_ignoring_terminal_blank_lines=(
                        text.rstrip() == old.read_text(encoding='utf-8-sig').rstrip()),
                    scope='Current file only; earlier numbered archives not independently audited; no model forwards.',
                    phases=phases)
    (OUT / 'source_inventory.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('prior_audit_reused', ROOT / 'tests/glm5/audit_gpt5_memo_20260923.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT, mod.MEMO = OUT, snapshot
    with (OUT / 'cpu_recheck.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        mod.inventory()
        mod.continuum()
        mod.mathematical_checks()
        mod.coverage()
    # Establish provenance for earlier follow-up corrections, without calling them new experiments.
    refs = [
        ROOT / 'tests/glm5/audit_gpt5_memo_20260923.py',
        ROOT / 'tests/glm5/result/gpt5_memo_audit_20260923/review.md',
        ROOT / 'tests/glm5/result/rdc_framework_audit_20260923/phase_report.md',
        ROOT / 'tests/glm5/result/rdc_framework_audit_20260923/numerical_rechecks.json',
        ROOT / 'tests/glm5/result/rdc_framework_audit_20260923/claim_ledger.json',
    ]
    provenance = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in refs}
    (OUT / 'prior_evidence_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    # Synthetic identifiability counterexample, not model evidence.
    w = np.array([[1., 1.]])
    observed = np.array([1.])
    minimum_norm = np.linalg.pinv(w) @ observed
    alternative = np.array([1., 0.])
    assert np.allclose(w @ minimum_norm, observed)
    assert np.allclose(w @ alternative, observed)
    assert not np.allclose(minimum_norm, alternative)
    pinv_check = dict(kind='synthetic_counterexample_not_model_test', W=w.tolist(),
                      observed=observed.tolist(), minimum_norm=minimum_norm.tolist(),
                      alternative=alternative.tolist(),
                      conclusion='A pseudoinverse coefficient vector does not identify actual MLP activations; correlated dictionary coefficients do not give additive output energy shares.')
    (OUT / 'additional_mathematical_check.json').write_text(json.dumps(pinv_check, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k != 'phases'}, ensure_ascii=False, indent=2))
    checks = json.loads((OUT / 'artifact_checks.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(selected_phases=len(checks['selected_phases']), result_files=len(checks['result_files']),
                         scripts=len(checks['script_sha256']),
                         seal_matches=sum(r.get('seal_result_matches') is True for r in checks['result_files']),
                         seal_mismatches=sum(r.get('seal_result_matches') is False for r in checks['result_files'])), ensure_ascii=False))

if __name__ == '__main__':
    main()
