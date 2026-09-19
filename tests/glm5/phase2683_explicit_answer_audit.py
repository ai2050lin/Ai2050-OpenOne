"""Post-hoc explicit-final-line audit, never a replacement semantic accuracy.

Original exact/normalized whole-string scores and every raw output stay intact.
"""
import argparse, re
from phase2620_native_coordinate_contract import *

OUT = RESULT/'phase2683_crossmodel_function_atlas'
KEYS = ('qwen14','glm4','ds7','ds7_answer')
LABEL = re.compile(r'^(?:answer|output(?:\s+code)?|code|答案|回答|输出(?:代码)?)\s*[:：]\s*', re.I)


def parse_line(line, options):
    text = line.strip().replace('**','').replace('`','').strip()
    text, labels = LABEL.subn('', text)
    text = text.strip().rstrip(' .。').strip()
    boxed = re.fullmatch(r'\\boxed\{\s*(.*?)\s*\}', text)
    if boxed:
        text = boxed.group(1).strip()
        inner = re.fullmatch(r'\\text\{\s*(.*?)\s*\}', text)
        if inner: text = inner.group(1).strip()
    normalized = text.casefold()
    if set(options) == {'是','否'}:
        normalized = {'是的':'是','不是':'否','不是的':'否'}.get(normalized, normalized)
    if normalized not in options: return None
    return {'choice':normalized, 'format':'boxed' if boxed else 'explicit_label' if labels else 'bare_line'}


def extract(text, options, boundary=True):
    options = tuple(str(option).casefold() for option in options)
    assert len(options) == 2 and len(set(options)) == 2
    if not boundary: return {'status':'no_final_boundary'}
    if not text.strip(): return {'status':'empty_final'}
    # Ignore standalone Markdown code-fence delimiters, not their content.
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not re.fullmatch(r'```[\w+-]*', line.strip())]
    if not lines: return {'status':'empty_final'}
    last = parse_line(lines[-1], options)
    if last is None: return {'status':'unparsed', 'last_line':lines[-1]}
    previous = [parse_line(line, options) for line in lines[:-1]]
    conflicts = [r for r in previous if r is not None and r['choice'] != last['choice']]
    if conflicts: return {'status':'ambiguous', 'last_line':lines[-1], 'conflicting_explicit_choices':sorted({r['choice'] for r in conflicts} | {last['choice']})}
    return {'status':'parsed', 'last_line':lines[-1], **last}


def self_test():
    cases = [
        ('Aster','parsed','aster'),('Birch.','parsed','birch'),(' **Aster**. ','parsed','aster'),
        ('**Answer:** **Aster**','parsed','aster'),('Output code: Birch','parsed','birch'),
        ('答案：Aster。','parsed','aster'),('Explanation mentions Birch.\nOutput: Aster','parsed','aster'),
        (r'\boxed{Aster}','parsed','aster'),(r'\boxed{\text{Birch}}','parsed','birch'),
        ('Explanation.\n\n'+r'\boxed{Aster}'+'.','parsed','aster'),('```text\nAster\n```','parsed','aster'),
        ('not Aster','unparsed',None),('Aster or Birch','unparsed',None),('Answer: Aster or Birch','unparsed',None),
        ('The person is Aster.','unparsed',None),('Aster\nI am not sure.','unparsed',None),
        ('Answer: Aster\nAnswer: Birch','ambiguous',None),('Aster\n'+r'\boxed{Birch}','ambiguous',None),
        ('','empty_final',None),('   \n','empty_final',None),('```\n```','empty_final',None),
        ('Answer: Asterina','unparsed',None),('Answer: not Birch','unparsed',None),
    ]
    for text, status, choice in cases:
        result = extract(text, ('Aster','Birch'))
        assert result['status'] == status and result.get('choice') == choice, (text, result)
        assert result == extract(text, ('Birch','Aster')), 'Extraction must not favor the gold option/order'
    assert extract('Aster', ('Aster','Birch'), False)['status'] == 'no_final_boundary'
    assert extract('不是。', ('是','否'))['choice'] == '否'
    assert extract('是的', ('甲','乙'))['status'] == 'unparsed'
    assert extract('否认甲', ('甲','乙'))['status'] == 'unparsed'
    return len(cases)*2+4


def audit_records(material, records):
    assert len(material) == len(records)
    out = []; groups = {}; mismatches = []
    for row, record in zip(material, records):
        assert row['case_index'] == record['case_index']
        text = record['generated']; whole = text.strip(); clean = whole.strip(' .。').casefold()
        if row['output_function'] in ('truth','mapped_truth') and row['language'] == 'zh':
            clean = {'是的':'是','不是':'否','不是的':'否'}.get(clean,clean)
        if record['content_correct'] != (clean == row['target'].casefold()) or record['strict_correct'] != (whole.casefold() == row['target'].casefold()):
            mismatches.append(row['case_index'])
        result = extract(text, (row['target'],row['alternate']), record['final_answer_available'])
        matched = result['status'] == 'parsed' and result['choice'] == row['target'].casefold()
        label = row['language']+'/'+row['output_function']
        group = groups.setdefault(label, {'n':0,'legacy_strict_match':0,'legacy_normalized_whole_match':0,
            'final_channel_identified':0,'nonempty_final_text':0,'parsed_correct':0,'parsed_wrong':0,
            'unparsed':0,'ambiguous':0,'empty_final':0,'no_final_boundary':0,'legacy_false_but_explicit_match':0})
        group['n'] += 1; group['legacy_strict_match'] += int(record['strict_correct'])
        group['legacy_normalized_whole_match'] += int(record['content_correct'])
        group['final_channel_identified'] += int(record['final_answer_available'])
        group['nonempty_final_text'] += int(record['final_answer_available'] and bool(text.strip()))
        if result['status'] == 'parsed': group['parsed_correct' if matched else 'parsed_wrong'] += 1
        else: group[result['status']] += 1
        group['legacy_false_but_explicit_match'] += int(matched and not record['content_correct'])
        out.append({'case_index':row['case_index'],'case_id':row['case_id'],'target':row['target'],'alternate':row['alternate'],
                    'legacy_normalized_whole_match':record['content_correct'],'explicit_match':matched if result['status']=='parsed' else None, **result})
    assert not mismatches, mismatches
    for group in groups.values():
        assert sum(group[k] for k in ('parsed_correct','parsed_wrong','unparsed','ambiguous','empty_final','no_final_boundary')) == group['n']
    return {'groups':groups,'records':out,'legacy_fields_reproduced':True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('self_test','all_completed',*KEYS))
    args = parser.parse_args(); tests = self_test()
    if args.action == 'self_test': print({'self_tests':tests,'passed':True}); return
    contract_path = OUT/'protocol/explicit_answer_audit.json'; code_hash = sha(Path(__file__))
    if contract_path.exists(): assert read(contract_path)['source_sha256'] == code_hash, 'Do not silently alter post-hoc parser contract'
    else:
        save(contract_path, {'source_sha256':code_hash,'self_tests':tests,'created':datetime.now().astimezone().isoformat(),
            'status':'Post-hoc measurement audit, NOT preregistered independent confirmation. Designed after inspecting nativeDS cases3,7,11 and noticing whole-string score limitation; do not claim blind evaluation.',
            'rule':'Only last nonempty non-fence line; exact candidate alone, explicit Answer/Output/Code/答案/回答/输出 label, or boxed candidate/text wrapper. Multiple contradictory candidate-only lines yield ambiguous. No substring success, free-form semantic judgment, gold-guided choice, first-line or best-of-line selection.',
            'original_field_meaning':'content_correct is limited-punctuation/Chinese-alias normalized WHOLE STRING equality, not semantic correctness. final_answer_available identifies the final channel/boundary and can still have empty final text.',
            'limits':'Matching explicit terminal answer is not verification of explanatory consistency or semantic reasoning. Unparsed text may be right or wrong. Longer explanations violate original output-only format even if the final explicit choice matches. Preserve all original frozen scores/outputs; report disjoint categories over alln, not just parsed subset.'})
    keys = [key for key in KEYS if (OUT/key/'analysis/completion.json').exists()] if args.action == 'all_completed' else [args.action]
    for key in keys:
        folder = OUT/key; assert read(folder/'analysis/completion.json')['cases'] == 512
        material_path = folder/'material/cases.json'; records_path = folder/'analysis/records.json'
        before = {'material':sha(material_path),'records':sha(records_path)}
        report = audit_records(read(material_path), read(records_path))
        report.update(protocol=key,all_checks_passed=True,post_hoc=True,semantic_accuracy_claim=False,
                      parser_contract_sha256=sha(contract_path),source_hashes=before,original_artifacts_unchanged=before == {'material':sha(material_path),'records':sha(records_path)})
        assert report['original_artifacts_unchanged']
        calibration = folder/'analysis/calibration.jsonl'
        if calibration.exists():
            rr = [json.loads(s) for s in calibration.read_text(encoding='utf-8').splitlines()]
            report['calibration'] = audit_records(read(folder/'material/calibration.json'),rr)
        save(folder/'analysis/explicit_answer_audit.json',report)
        print({'protocol':key,'groups':report['groups'],'post_hoc':True,'semantic_accuracy_claim':False},flush=True)


if __name__ == '__main__': main()
