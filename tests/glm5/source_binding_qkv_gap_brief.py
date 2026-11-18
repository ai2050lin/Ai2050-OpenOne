"""Export a compact audited index for the QKV and Answer: follow-up."""
from hashlib import sha256
import json
from pathlib import Path

import source_binding_qkv_followup as exp
import source_binding_transfer_factors as io


def main():
    io.OUT = exp.OUT
    main_report = io.read('holdout-report.json')
    readout = io.read('readout/readout-report.json')
    original = io.read('phase2921-gap-review.json')
    summaries = {}
    for task,x in main_report['tasks'].items():
        summaries[task] = dict(families=x['families'],edits=x['wide_edits'],same_target=x['common_n'],
            conditions=x['changed'],Q_only_answer_changes=x['Q_only_answer_change'],
            Q_swap={str(c):dict(**x['common']['QH'][str(c)],block23=x['common_attention_change_from_H'][f'QH/{c}'][0]) for c in (0,1)},
            downstream_fixed_delta=x['cue_effects']['downstream_fixed_packet'],
            answer_only_controls=x['answer_recovery'])
    paths = ['holdout-report.json','readout/readout-report.json','phase2921-gap-review.json','answer-distance-support.json']
    suffix = exp.OUT/'suffix/report.json'
    if suffix.exists():
        paths.append('suffix/report.json')
    if (exp.OUT/'suffix/region-comparison.json').exists():
        paths.append('suffix/region-comparison.json')
    file_ids = {name:sha256((exp.OUT/name).read_bytes()).hexdigest() for name in paths}
    io.save('result-index.json',dict(run=exp.RUN,snapshot=io.read('design.json')['snapshot'],
        files=file_ids,counts=summaries,gap=main_report['Answer_gap'],historical_gap=original['gap'],
        fixed_answer_state_readout=readout['tasks'],
        suffix_status='completed' if suffix.exists() else 'partial; see suffix campaign.log',formal_evidence_promotion=False))


if __name__=='__main__':
    main()
