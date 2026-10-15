# -*- coding: utf-8 -*-
"""Phase 3238-align self-check: client/server alignment verification.
Sealed artifacts must be untouched; all derived/new files verified."""
import hashlib
import json
import os
import sys

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
P = N = 0
FAILS = []


def chk(name, cond):
    global P, N
    N += 1
    print(('PASS ' if cond else 'FAIL ') + name)
    if cond:
        P += 1
    else:
        FAILS.append(name)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# C1: derive tree data plane refreshed (registry v32, ledger n=389)
t1 = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                 'atlas_tree_v1.json'), encoding='utf-8'))
chk('C1a tree v1 registry_version=32 + ledger n=389',
    t1.get('derived_from', {}).get('registry_version') == '32'
    and t1.get('derived_from', {}).get('ledger_measurements') == 389)
chk('C1b tree v1 inner tree_sha256_8=0a62ca19 (derive stdout anchor)',
    t1.get('tree_sha256_8') == '0a62ca19')

# C2: ai_rnd_service discovery (import via package)
sys.path.insert(0, ROOT)
import warnings
warnings.filterwarnings('ignore')
try:
    from server import ai_rnd_service as S
    cands = S._discover_atlas_versions()
    chk('C2a registry chain n=32 latest v1_32',
        len(cands) == 32 and cands[-1][1].name == 'atlas_registry_v1_32.json')
    chk('C2b default path = latest', S._atlas_registry_path()
        == cands[-1][1])
    gcands = S._discover_gap_versions()
    chk('C2c gap chain latest v1_11',
        gcands[-1][1].name == 'gap_ledger_v1_11.json')
    src = open(os.path.join(ROOT, 'server', 'ai_rnd_service.py'),
               encoding='utf-8').read()
    chk('C2d /atlas/gaps endpoint present', '"/atlas/gaps"' in src
        or "'/atlas/gaps'" in src)
except Exception as e:
    chk('C2 server import', False)
    print('  ERR:', type(e).__name__, e)

# C3: bundle example
EX = os.path.join(ROOT, 'shared', 'dist_protocol', 'examples',
                  'TM-ISA-DEMO-v1')
corpus = json.load(open(os.path.join(EX, 'corpus.json'), encoding='utf-8'))
contract = json.load(open(os.path.join(EX, 'contract.json'),
                          encoding='utf-8'))
runner = open(os.path.join(EX, 'runner.py'), encoding='utf-8').read()
chk('C3a corpus 160 rows + schema', corpus['n_rows'] == 160
    and len(corpus['rows']) == 160
    and corpus['schema'] == 'corpus.v1')
chk('C3b corpus source anchors 3236', corpus['source_freeze_sha8']
    == 'b1fa889f' and corpus['source_res_sha8'] == '8b79ed5e')
design = {'tm_id': 'TM-ISA-DEMO', 'version': 1, 'corpus': corpus,
          'runner_sha256': hashlib.sha256(
              runner.encode('utf-8')).hexdigest()}
import hashlib as _h
ds = _h.sha256(json.dumps(design, sort_keys=True, ensure_ascii=False,
                          separators=(',', ':')).encode('utf-8')).hexdigest()
chk('C3c contract design_sha reproduces',
    contract['design_sha'] == ds)
sm = json.load(open(os.path.join(EX, '_smoke_outputs', 'summary.json'),
                    encoding='utf-8'))
chk('C3d runner smoke summary contract fields',
    sm['status'] == 'smoke' and sm['n_rows'] == 160
    and 'design_sha' in sm and 'f1_auc_per_template' in sm)
chk('C3e contract gates preregistered G1-G4',
    set(contract['gates']) == {'G1_sep_pos', 'G2_stitch',
                               'G3_xlang_behavior', 'G4_layer_report'})

# C4: sealed artifacts untouched
reg32 = os.path.join(BASE, 'phase3237', 'g5g31_registry_landing',
                     'atlas_registry_v1_32.json')
gap11 = os.path.join(BASE, 'phase3237', 'g5g31_registry_landing',
                     'gap_ledger_v1_11.json')
r37 = json.load(open(os.path.join(BASE, 'phase3237',
                                  'g5g31_registry_landing', 'result.json'),
                     encoding='utf-8'))
chk('C4a registry v1.32 untouched (file 3c3f0484)',
    fsha(reg32) == r37['wrote']['registry_v132']['file_sha8'])
chk('C4b gap v1.11 untouched (file 5934631b)',
    fsha(gap11) == r37['wrote']['gap_v111']['file_sha8'])
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
chk('C4c ledger n=389 post d37ca65f',
    len(led['measurements']) == 389
    and led['ledger_sha256_8'] == 'd37ca65f')

# C5: alignment doc exists with anchors
doc = open(os.path.join(ROOT, 'research', 'gpt5', 'docs',
                        'CLIENT_SERVER_ALIGNMENT_v1.md'),
           encoding='utf-8').read()
chk('C5 alignment doc anchors', '0a62ca19' in doc and '2ecf0e9e' in doc
    and 'TM-ISA-DEMO-v1' in doc)

print('ALIGN CHECK %d/%d PASS' % (P, N))
if FAILS:
    print('fails:', FAILS)
    sys.exit(1)
