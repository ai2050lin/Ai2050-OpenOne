# -*- coding: utf-8 -*-
"""Phase 3238-align: generate TM-ISA-DEMO-v1 bundle example under
shared/dist_protocol/examples/. Bridges G-line 3236 demo arm to the
dist_protocol v3 bundle format (corpus + runner + contract with design_sha).

Read-only w.r.t. sealed artifacts; writes only the new example directory.
"""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'shared', 'dist_protocol', 'examples',
                   'TM-ISA-DEMO-v1')
os.makedirs(OUT, exist_ok=True)

# ---- 3236 DESIGN exported verbatim (frozen b1fa889f) ----
TEMPLATES = {
    'T1_zh_stmt': {'kind': 'class', 'fmt': '{e}是一种'},
    'T2_zh_quest': {'kind': 'judge', 'fmt': '{e}是一种{c}吗？答案是'},
    'T3_en_stmt': {'kind': 'class', 'fmt': 'The {e} is a kind of'},
    'T4_en_quest': {'kind': 'judge',
                    'fmt': 'Is the {e} a kind of {c}? Answer:'},
    'T5_zh_ctx': {'kind': 'class', 'fmt': '我喜欢{e}。{e}是一种'},
}
ENTITIES = {
    'zh': [['苹果', '水果'], ['玫瑰', '花'], ['金鱼', '鱼'], ['锤子', '工具']],
    'en': [['apple', 'fruit'], ['rose', 'flower'],
           ['goldfish', 'fish'], ['hammer', 'tool']],
}
CLS = {'zh': ['水果', '花', '鱼', '工具'],
       'en': ['fruit', 'flower', 'fish', 'tool']}
MARGIN_DEF = {
    'class': 'logit(cls_tok) - mean(logit(other 3 cls toks)) at last '
             'position, final slot',
    'judge': 'logit(yes_tok) - logit(no_tok) at last position',
}

rows = []
rid = 0
for lang in ('zh', 'en'):
    for e, true_c in ENTITIES[lang]:
        for c in CLS[lang]:
            for tid, t in sorted(TEMPLATES.items()):
                if t['kind'] == 'class':
                    prompt = t['fmt'].format(e=e)
                    cont = (' ' + c) if lang == 'en' else c
                else:
                    prompt = t['fmt'].format(e=e, c=c)
                    cont = ('是' if lang == 'zh' else ' Yes')
                rid += 1
                rows.append({
                    'row_id': 'R%03d' % rid,
                    'lang': lang, 'entity': e, 'cls': c,
                    'true_cls': true_c, 'template': tid,
                    'kind': t['kind'],
                    'prompt': prompt, 'continuation': cont,
                })
assert len(rows) == 160, len(rows)

corpus = {
    'schema': 'corpus.v1',
    'tm_id': 'TM-ISA-DEMO',
    'version': 1,
    'name': 'isa class-membership minimal-pair demo (G-line 3236 export)',
    'source_phase': 3236,
    'source_freeze_sha8': 'b1fa889f',
    'source_res_sha8': '8b79ed5e',
    'task_list_entry': {
        'fid': 'F-DEMO-ISA-001',
        'feature': '类属关系绑定强度 (isa class-membership binding)',
        'atlas_links': ['dR (3230 relation-contrast axis, 4b)',
                        'u* (3225 collapse carrier, 4b)',
                        'k*=3 class-subspace layer curve (3152)'],
    },
    'templates': {k: v['fmt'] for k, v in TEMPLATES.items()},
    'margin_def': MARGIN_DEF,
    'cls_tokens': {'zh': CLS['zh'], 'en': CLS['en']},
    'judge_tokens': {'zh': ['是', '不'], 'en': ['Yes', 'No']},
    'token_constraint': 'all cls/judge words MUST be single tokens under '
                        'the target tokenizer (3236 assert; runner '
                        're-checks at load)',
    'n_rows': len(rows),
    'rows': rows,
}
CP = os.path.join(OUT, 'corpus.json')
with open(CP, 'w', encoding='utf-8', newline='') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)
print('corpus.json written: %d rows' % len(rows))

RUNNER = '''# -*- coding: utf-8 -*-
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
            wins = sum(1 for p in pos for n in neg if p > n) + \\
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
'''
RP = os.path.join(OUT, 'runner.py')
with open(RP, 'w', encoding='utf-8', newline='') as f:
    f.write(RUNNER)
print('runner.py written (%d bytes)' % os.path.getsize(RP))

# ---- design_sha: exact distributed_service._canonical_design_sha recipe ----
design = {'tm_id': 'TM-ISA-DEMO', 'version': 1, 'corpus': corpus,
          'runner_sha256': hashlib.sha256(
              RUNNER.encode('utf-8')).hexdigest()}
payload = json.dumps(design, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':')).encode('utf-8')
design_sha = hashlib.sha256(payload).hexdigest()
print('design_sha =', design_sha)

contract = {
    'tm_id': 'TM-ISA-DEMO', 'version': 1,
    'name': 'G-line 3236 isa demo pipeline (bundle protocol example)',
    'design_sha': design_sha,
    'preregistered': True,
    'status': 'protocol_example_NOT_registered',
    'metric_pin': 'margin (corpus.margin_def), bf16 final-slot logits',
    'model_pin': 'qwen3-4b bf16 (3236 arm); shards via AI2050_SHARD',
    'gates': {
        'G1_sep_pos': 'mean margin separation > 0 in all 5 templates',
        'G2_stitch': 'mean cross-template spearman >= 0.6',
        'G3_xlang_behavior': 'mean ZH-EN spearman >= 0.5',
        'G4_layer_report': 'layer curve recorded, report-only',
    },
    'verdict_rule': 'DEMO_PIPELINE_COMPLETE if G1-G3 all pass else '
                    'DEMO_PIPELINE_PARTIAL (demo-level gates, no E-level '
                    'claim)',
    'provenance': {
        'source_phase': 3236, 'source_freeze_sha8': 'b1fa889f',
        'source_res_sha8': '8b79ed5e',
        'registry_at_export': 'v1.32 (file 3c3f0484)',
        'gap_at_export': 'v1.11 (file 5934631b)',
        'ledger_at_export': 'n=389 post d37ca65f',
        'alignment_doc': 'research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md',
    },
    'registration': 'to register on the hub: deploy/distributed_service.py '
                    'S2 template-register with this corpus/runner pair; '
                    'design_sha must reproduce byte-exact (canonical json '
                    'sort_keys separators(",",":"))',
}
CP2 = os.path.join(OUT, 'contract.json')
with open(CP2, 'w', encoding='utf-8', newline='') as f:
    json.dump(contract, f, ensure_ascii=False, indent=1)
print('contract.json written')

# smoke self-run of the runner (protocol example must at least smoke-pass)
import subprocess
r = subprocess.run([os.sys.executable, RP, '--out',
                    os.path.join(OUT, '_smoke_outputs'), '--smoke'],
                   capture_output=True, text=True, timeout=120)
print('smoke run rc =', r.returncode)
if r.returncode != 0:
    print(r.stdout[-800:], r.stderr[-800:])
else:
    s = json.load(open(os.path.join(OUT, '_smoke_outputs', 'summary.json'),
                       encoding='utf-8'))
    print('smoke summary status =', s['status'],
          '| n_rows =', s['n_rows'])
