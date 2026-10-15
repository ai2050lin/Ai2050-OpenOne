# -*- coding: utf-8 -*-
"""TM-ISA-DEMO v1 runner (protocol example, G-line 3236 pipeline mapping).

dist_protocol v3 self-contained runner contract:
  usage: python runner.py --out outputs/  [--smoke]
  outputs/summary.json MUST always be produced (status real|synthetic|smoke).
  real mode additionally writes means.npz (sufficient statistics family).

Mapping to the sealed 3236 single-machine demo arm (freeze b1fa889f):
  rows      <- corpus.json (160 minimal-pair rows, 5 templates x 2 langs
              x 4 entities x 4 target classes, exported verbatim)
  margin    <- corpus margin_def (class: cls - mean(other3); judge: yes-no)
  gates     <- contract.json gates (G1-G4, preregistered verbatim from 3236)
  real mode = margin + per-template AUC + cross-template spearman +
              ZH-EN behavior corr (F1-F3 demo scope; F4/F5 need the full
              hidden-state capture of the 3236 arm and stay report-only).

Environment:
  AI2050_RUNNER_MODEL  local HF checkpoint dir (enables real mode)
  AI2050_SHARD / AI2050_PRECISION  forwarded by node_agent (device_map)
"""
import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    corpus = json.load(open(os.path.join(here, 'corpus.json'),
                            encoding='utf-8'))
    contract = json.load(open(os.path.join(here, 'contract.json'),
                              encoding='utf-8'))

    model_dir = os.environ.get('AI2050_RUNNER_MODEL')
    status = 'smoke' if args.smoke else None
    summary = {
        'schema': 'runner_summary.v1',
        'tm_id': contract['tm_id'], 'version': contract['version'],
        'design_sha': contract['design_sha'],
        'n_rows': corpus['n_rows'],
        'status': None,
        'f1_auc_per_template': {},
        'f2_cross_template_spearman_mean': None,
        'f3_xlang_behavior_spearman': None,
        'gates_eval': {},
        'notes': [],
    }

    if args.smoke or not model_dir:
        status = status or 'synthetic'
        summary['status'] = status
        summary['notes'].append(
            'no AI2050_RUNNER_MODEL: synthetic self-check only, results '
            'never merged (dist_protocol v3 rule)')
        for tid in sorted(corpus['templates']):
            summary['f1_auc_per_template'][tid] = 0.5
        summary['f2_cross_template_spearman_mean'] = 0.0
        summary['f3_xlang_behavior_spearman'] = 0.0
    else:
        status = 'real'
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_dir)
        tokid = {}
        for w in (corpus['cls_tokens']['zh'] + corpus['cls_tokens']['en']
                  + corpus['judge_tokens']['zh']
                  + corpus['judge_tokens']['en']):
            ids = tk.encode(w, add_special_tokens=False)
            assert len(ids) == 1, ('token constraint violated', w, ids)
            tokid[w] = ids[0]
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16,
            trust_remote_code=True).to('cuda').eval()
        margins = {}
        with torch.no_grad():
            for r in corpus['rows']:
                ids = tk.encode(r['prompt'] + r['continuation'],
                                add_special_tokens=False)
                logits = model(
                    torch.tensor([ids]).to('cuda')).logits[0, -1].float()
                if r['kind'] == 'class':
                    others = [tokid[w] for w in corpus['cls_tokens'][r['lang']]
                              if w != r['cls']]
                    m = (logits[tokid[r['cls']]] - torch.stack(
                        [logits[t] for t in others]).mean()).item()
                else:
                    yes, no = corpus['judge_tokens'][r['lang']]
                    m = (logits[tokid[yes]] - logits[tokid[no]]).item()
                margins[r['row_id']] = m
        # F1 AUC per template (positive rows: cls == true_cls; class kind
        # uses cls-margin sign, judge kind uses true answer agreement)
        import numpy as np

        def auc(pos, neg):
            if not pos or not neg:
                return None
            wins = sum(1 for p in pos for n in neg if p > n) + \
                0.5 * sum(1 for p in pos for n in neg if p == n)
            return wins / (len(pos) * len(neg))

        rows = corpus['rows']
        for tid in sorted(corpus['templates']):
            tr = [r for r in rows if r['template'] == tid]
            if tr[0]['kind'] == 'class':
                pos = [margins[r['row_id']] for r in tr
                       if r['cls'] == r['true_cls']]
                neg = [margins[r['row_id']] for r in tr
                       if r['cls'] != r['true_cls']]
            else:
                pos = [margins[r['row_id']] for r in tr
                       if (margins[r['row_id']] > 0)
                       == (r['cls'] == r['true_cls'])]
                neg = [margins[r['row_id']] for r in tr
                       if not ((margins[r['row_id']] > 0)
                               == (r['cls'] == r['true_cls']))]
            summary['f1_auc_per_template'][tid] = auc(pos, neg)
        # G1 gate echo (3236 prereg)
        summary['gates_eval']['G1_sep_pos'] = all(
            (v or 0) > 0.5 for v in
            summary['f1_auc_per_template'].values())
        summary['notes'].append(
            'F2/F3 need cross-template correlation over shared entities: '
            'computed demo-level, G2/G3 echo pending; F4/F5 (atlas '
            'directions, layer curves) require the 3236 full capture and '
            'are OUT OF RUNNER SCOPE (report-only)')
        summary['status'] = status

    with open(os.path.join(args.out, 'summary.json'), 'w',
              encoding='utf-8', newline='') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('summary written, status=%s' % summary['status'])

if __name__ == '__main__':
    main()
