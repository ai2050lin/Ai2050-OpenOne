"""Enforce strict winning margin, retaining the frozen argmax bookkeeping."""
import time
import branch_reuse14 as B


def check(split):
    data=B.read(B.OUT/split/'collection.json');allrows=B.read(B.OUT/'materials.json')['rows'];rows=[allrows[i] for i in data['row_indices']]
    margin=[s[r['gold']]-max(v for j,v in enumerate(s) if j!=r['gold']) for r,s in zip(rows,data['scores'])]
    cells=[]
    for branch,role,title in sorted({(r['branch'],r['role'],r['title']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['branch'],r['role'],r['title'])==(branch,role,title)]
        correct=sum(margin[i]>0 for i in ix);threshold=.85 if split=='pilot' else .9
        cells.append(dict(branch=branch,role=role,title=title,n=len(ix),correct=correct,accuracy=correct/len(ix),passed=correct/len(ix)>=threshold))
    accuracy=sum(m>0 for m in margin)/len(rows)
    return dict(split=split,cells=cells,overall_accuracy=accuracy,passed=all(r['passed'] for r in cells) and accuracy>=.9,
                ties=[r['id'] for r,m in zip(rows,margin) if m==0],argmax_behavior=data['behavior'],collection_sha256=B.file_sha(B.OUT/split/'collection.json'))


def run():
    B.configure()
    original=B.OUT/'execution-freeze.json';pending=B.OUT/'execution-freeze.pending-strict.json'
    assert original.exists() and not pending.exists()
    original_sha=B.file_sha(original)
    original.rename(pending)
    start=time.monotonic()
    B.save('strict-behavior-correction.json',dict(created_local=time.strftime('%Y-%m-%d %H:%M'),
        reason='Frozen collector uses argmax, which counts a first-listed gold tie as correct. Protocol requires strictly greater gold score. Preserve all scores and frozen files; add stricter independent gate before analysis. No threshold, material, selected feature or fit changes.',
        preserved_execution_sha256=original_sha,preserved_execution_path=str(pending.relative_to(B.OUT)),source_sha256=B.file_sha(__file__),
        orchestration='Temporarily withhold downstream execution admission receipt. Native collector has its own unchanged trial admission and continues; parent pipeline stops on missing execution receipt before analysis. Restore exact receipt after that stop and strict check.',
        pilot=check('pilot'),discovery=check('discovery'),confirmation_scores_inspected=False))
    assert check('pilot')['passed'] and check('discovery')['passed']
    last=0
    while not (B.OUT/'confirm/collection.json').exists():
        if time.monotonic()-last>45:
            B.log('strict_behavior_guard_wait',0,1,start);last=time.monotonic()
        time.sleep(1)
    report=check('confirm');B.save('strict-behavior-confirm.json',report)
    deadline=time.monotonic()+120
    while 'FileNotFoundError' not in (B.OUT/'pipeline.log').read_text(encoding='utf8'):
        assert time.monotonic()<deadline,'Expected controlled pipeline stop not observed; receipt remains withheld'
        time.sleep(1)
    assert B.file_sha(pending)==original_sha;pending.rename(original)
    B.save('strict-behavior-stop.json',dict(reason='Controlled pipeline stop completed before internal analysis',
        strict_passed=report['passed'],execution_receipt_restored_bitwise=True,source_sha256=B.file_sha(__file__)))
    print('Strict behavior passed' if report['passed'] else 'Strict behavior failed',flush=True)


if __name__=='__main__':run()
