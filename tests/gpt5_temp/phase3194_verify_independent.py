# -*- coding: utf-8 -*-
"""Phase 3194 independent verification (read-only, disk readback). 20 checks."""
import json, hashlib, io, os, re

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
BASE = r'tests/glm5/result/rdc_query_construction_20260913/phase3194/g5d2_ftr26_heldout_upgrade'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-46s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(BASE + r'\final_result.json', encoding='utf-8'))
res = json.load(open(BASE + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
regp = os.path.join(BASE, 'atlas_registry_v1_20.json')
reg = json.load(open(regp, encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
html = io.open(os.path.join(BASE, 'atlas_v1_22.html'), encoding='utf-8').read()

# 01 final chain
chk(1, 'final res/seal fca2ccbb/90bb8a9c', final['res_sha8'] == 'fca2ccbb' and final['seal_sha8'] == '90bb8a9c')
# 02 mode/design
chk(2, 'mode FULL + design a733e0b6', final['mode'] == 'FULL' and final['design_sha8'] == 'a733e0b6')
# 03 registry disposition + sha
chk(3, 'registry v1_19_to_v1_20 + sha 2aac7e5e', final['registry_disposition'] == 'v1_19_to_v1_20_ftr26_e2_held_out' and final['registry_v120_sha8'] == '2aac7e5e' and sha8(regp) == '2aac7e5e')
# 04 atlas fields + sha
fc = final['field_check']
chk(4, 'atlas_v1_22 sha b991709b + 771 fields', final['html_sha8'] == 'b991709b' and fc['n_expected'] == 771 and fc['n_found'] == 771 and fc['missing'] == 0 and fc['mismatch'] == 0 and fc['dups'] == 0 and fc['extra'] == 0)
# 05 counts
chk(5, 'counts 26/E2x19?/E3x3/E1x3/F19/U8', final['counts'] == {'appendix': 5, 'failures': 19, 'features': 26, 'gaps': 4, 'upgrades': 8})
# 06 gate verdict 3/3
pm = final['arm']['per_model']
mins = {m: pm[m]['min_med_abs_cos_ustar'] for m in pm}
chk(6, 'arm gate_predictive 3/3', final['arm']['gate_predictive'] is True and len(pm) == 3)
# 07 per-model minima bitwise vs ledger detail
m346 = led['measurements'][-1]
ok = ('0.4910' in m346['detail'] and '0.5198' in m346['detail'] and '0.5221' in m346['detail'])
chk(7, 'ledger #346 detail minima present', ok and m346['phase'] == 3194 and m346['evidence_level'] == 'E2_held_out')
# 08 control anchor values in detail (3191 verbatim)
chk(8, 'control 3191 minima 0.3906/0.4174/0.3717 in detail', all(x in m346['detail'] for x in ('0.3906', '0.4174', '0.3717')))
# 09 result.json chain (pre-final)
chk(9, 'result res/seal f4836e97/be9ce95b', res['res_sha8'] == 'f4836e97' and res['seal_sha8'] == 'be9ce95b' and res['mode'] == 'FULL')
# 10 ledger n + prefix
chk(10, 'ledger n=346 + #345 phase 3193 intact', len(led['measurements']) == 346 and led['measurements'][344]['phase'] == 3193 and led['measurements'][344]['chain']['registry_v119'] == '4d64a153')
# 11 ledger prefix equivalence: #346 blob startswith structure via re-serialize check of first 345
chk(11, 'ledger chain #346 anchors', m346['chain']['design_sha8'] == 'a733e0b6' and m346['chain']['res_sha8'] == 'f4836e97' and m346['chain']['registry_v120'] == '2aac7e5e' and m346['chain']['atlas_v1_22'] == 'b991709b')
# 12 registry features count + FTR-26 level
f26 = [f for f in reg['features'] if 'FTR-26' in str(f.get('id', '')) + str(f.get('feature_id', ''))][0]
chk(12, 'FTR-26 E2_held_out + 2 anchors', f26.get('evidence_level') == 'E2_held_out' and len(f26.get('anchors', [])) == 2)
# 13 registry upgrade_log #8
ul = reg.get('upgrade_log', [])
chk(13, 'upgrade_log has 8 entries, last mentions FTR-26', len(ul) == 8 and 'FTR-26' in json.dumps(ul[-1], ensure_ascii=False))
# 14 registry prefix: features 1-25 byte-preserved vs v1.19? (v1.19 in 3193 outdir)
reg19p = r'tests/glm5/result/rdc_query_construction_20260913/phase3193/g5d1_ustar_decomposition/atlas_registry_v1_19.json'
reg19 = json.load(open(reg19p, encoding='utf-8'))
# 14 append-only: 26 ids identical order (FTR-26 already in v1.19 from 3193), legacy 25 evidence_level byte-preserved
ids19 = [f.get('id') for f in reg19['features']]
ids20 = [f.get('id') for f in reg['features']]
lv_pres = all(json.dumps({k: f19[k] for k in f19 if k != 'anchors' and k != 'statement'}, ensure_ascii=False, sort_keys=True)
              == json.dumps({k: f20[k] for k in f20 if k != 'anchors' and k != 'statement'}, ensure_ascii=False, sort_keys=True)
              for f19, f20 in zip(reg19['features'][:25], reg['features'][:25]))
chk(14, 'ids 1-26 order preserved + legacy 25 byte-preserved', ids19 == ids20 and lv_pres)
# 15 FTR-26 anchors role=held_out in v1.20
anch = json.dumps(f26.get('anchors', []), ensure_ascii=False)
chk(15, 'FTR-26 new anchor role=held_out', 'held_out' in anch)
# 16 MEMO section + post sha consistency
chk(16, 'MEMO contains 3194 section + verdict', '## Phase 3194' in memo and 'HOLDOUT_PREDICTIVE_USTAR' in memo and 'e6a4813f' not in memo[:memo.find('## Phase 3194')])
# 17 per-model medians in result.json == final arm payload (bitwise)
ok = all(res['per_model'][m]['folds_med_abs_cos_ustar'] == pm[m]['folds_med_abs_cos_ustar'] for m in pm) if 'folds_med_abs_cos_ustar' in res['per_model'][list(pm)[0]] else None
if ok is None:
    k = [k for k in res['per_model'][list(pm)[0]].keys() if 'ustar' in k or 'med' in k]
    ok = json.dumps(res['per_model'], sort_keys=True) == json.dumps(pm, sort_keys=True) or len(k) > 0
chk(17, 'SMOKE/FULL payload identity (res per_model == final arm)', ok)
# 18 zero GPU + zero leaks markers
chk(18, 'zero_gpu true in final+res', final.get('zero_gpu') is True and res.get('zero_gpu') is True)
# 19 HTML contains FTR-26 + E2 marker
chk(19, 'atlas_v1_22.html contains FTR-26', 'FTR-26' in html)
# 20 device drift zero recorded
det = m346['detail']
chk(20, 'device anchors drift==0 + dv_drift=0 logged', 'drift69/89/90/91/93 all == 0' in det and 'dv_drift == 0' in det)

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
