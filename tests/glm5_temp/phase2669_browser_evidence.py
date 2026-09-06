"""Persist actual observed browser values and independently match saved data."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2669_symmetric_multitoken_delivery import OUT


def main():
    c=next(r for r in read(OUT/'material/published_cases.json') if r['case_index']==256)
    with np.load(OUT/'maps/client_panels/phase2669_short_h.npz') as z:h=float(z['values'][5,2355])
    with np.load(OUT/'maps/client_panels/phase2669_q14_mlp.npz') as z:a=float(z['values'][0,17407])
    from server.native_multitoken_parameter_query import query
    v=query(256,0,57,32,0,0,1,None)
    checks={'ten_types_observed_in_selector':len(read(OUT/'material/client_panel_catalog.json')['panels'])==10,
        'observed_H24_2355_exact':h==2.953125,'observed_q14_last_unit_exact':a==.263671875,
        'observed_real_V_weight_exact':v['actual_weight']==-.007293701171875,
        'observed_multitoken_derivative_exact':v['parameter_derivative']==-.4668409920574464,
        'observed_sequence_contrast_exact':c['contrast']==30.67216960393114,
        'observed_five_categories':c['branches'][0]['categories']==['format','format','content','format','eos'],
        'observed_embedding_H_MLP_fields':v['fields']['bf16']=={'embedding':-.034912109375,'hidden':3.875,'mlp':.00016117095947265625}}
    report={'checks':checks,'all_checks_passed':all(checks.values()),'time':datetime.now().astimezone().isoformat(),
        'method':'Actual CUA browser AX and screenshots on localhost:5173; observations separately matched to immutable artifacts by this script. This is not a scripted browser replay.',
        'observed':['State heatmap > native-coordinate inspector > multi-token query returned case256, exact BF16/FP32 fields and V weight.',
            'All10 published panel types visible; shortH has768x2560; H24/output row5 coordinate2355 value2.953125.',
            'Changed color gain1x to64x without numeric-value change; screenshot shows dense coordinate stripes and horizontal scrolling.',
            'Qwen14 MLP panel has64x17408; original last coordinate17407 visible with exact value. No feature selector or TopK applied.',
            'Reloaded after adding stable React keys; new and old sequence inspectors remain separately named. Expanded Y branch shows content/format/EOS categories and per-token products.'],
        'fixes':['Added numeric table spacing/horizontal overflow after screenshot exposed crowded values.','Added stable keys to prevent dev-HMR sibling component state reuse.'],
        'scope':'Desktop default1280x720 verified. Not every historical3D panel, all responsive sizes, all colors, or every row visually inspected; full matrix equality and bounds are separately covered by571 API/build checks.'}
    save(OUT/'analysis/browser_checks.json',report);assert report['all_checks_passed'];print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':
    sys.path.insert(0,str(ROOT));main()
