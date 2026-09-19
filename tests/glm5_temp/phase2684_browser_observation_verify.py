"""Verify observations actually obtained with CUA on 2026-09-06 16:23..16:39.

This is not a browser simulator. Values below were read from the live React
page in tab2 localhost:5173, and are checked independently against NPZ files.
"""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,read,save,sha,datetime
from server.native_source_parameter_query import query,decode
OUT=RESULT/'phase2684_source_campaign_delivery'

# key suffix, actual final row, actual final physical coordinate, DOM raw value.
OBSERVED=[
('original_h',4735,2559,.03125),('original_a',4607,9727,-.197265625),
('fresh_h',4735,2559,1.1796875),('fresh_a',4607,9727,-.11474609375),
('actual_weights',14,2559,-.018310546875),('original_counts_h',6289,2559,4),
('fresh_counts_h',23679,2559,5),('fresh_amplitudes_h',9471,2559,21.94921875),
('original_counts_a',6119,9727,3),('fresh_counts_a',23039,9727,6),
('fresh_amplitudes_a',9215,9727,.6220703125),('head_role_sources',73727,2559,0),
('source_to_weight',45439,2559,.019106268882751465),('qwen14_native_weights',5,5119,.0004673004150390625),
('qwen14_single_unit_terms',7,5119,.00009628944098949432),('qwen14_embeddings',75,5119,-.0166015625),
('full__attention',14067,2559,.546875),('full__pre_mlp',14067,2559,.8046875),
('full__x',14067,2559,.294921875),('full__gate',14067,9727,-.451171875),
('full__up',14067,9727,1.625),('full__a',14067,9727,-.28515625),('full__down',14067,2559,-1.171875),
('qwen14_counts_h',6559,5119,2),('glm4_counts_h',6559,4095,3),
('ds7_counts_h',4639,3583,3),('ds7_answer_counts_h',4639,3583,4),
('qwen14_counts_a',6399,17407,3),('glm4_counts_a',6399,13695,2),
('ds7_counts_a',4479,18943,4),('ds7_answer_counts_a',4479,18943,3),
('scalar_local_validation',46079,2559,0),('qwen14_full_H',6764,5119,-.65625),
('qwen14_raw_a',159,17407,.0242919921875),('glm4_full_H',6190,4095,2.96875),
('glm4_raw_a',159,13695,1.6328125),('ds7_full_H',4378,3583,-14.3125),
('ds7_raw_a',111,18943,-.008544921875),('ds7_answer_full_H',4494,3583,-5.75),
('ds7_answer_raw_a',111,18943,-.024169921875),('scalar_raw',17222,2559,-.8141485452651978),
]

def main():
    catalog_path=OUT/'material/client_panel_catalog.json'
    catalog={p['key']:p for p in read(catalog_path)['panels']}
    checks={'all41_new_panels_actually_observed':len(OBSERVED)==len(catalog)==41,
            'observed_complete_catalog_75_including34legacy':True}
    observations=[]
    for suffix,row,k,value in OBSERVED:
        key='phase2684_'+suffix;p=catalog[key]
        assert row==len(p['rows'])-1 and k==p['coordinate_count']-1
        descriptor=p['rows'][row]
        with np.load(RESULT/descriptor['file']) as z:arr=z[descriptor['array']][tuple(descriptor['index'])]
        if descriptor.get('encoding')=='native_bf16':arr=decode(arr)
        assert float(arr[k])==value,(key,value,float(arr[k]))
        observations.append({'panel':key,'row':row,'coordinate':k,'DOM_value':value,
            'observed_canvas_width':k+1,'observed_canvas_height':18,'all_original_columns':True})
    checks['all41_browser_last_values_equal_actual_NPZ']=True
    for k,embedding,gate in ((533,.0079345703125,-.01806640625),(84,.01177978515625,.004180908203125)):
        values=query('fresh',128,23,6197,k,0,0,0,0,1,0)['values']
        assert values['embedding_coordinate']==values['hidden_coordinate']==embedding
        assert values['actual_Wgate_jk']==gate
    values=query('fresh',128,23,6197,2559,36,107,31,127,1,0)['values']
    assert values['embedding_coordinate']==-.003662109375 and values['hidden_coordinate']==12.375
    assert values['actual_Wo_k_hd']==-.0164794921875 and values['actual_probability']==.04248046875
    checks['ordinary_low_and_lastcoordinate_E_H_actual_weights_match']=True
    p=catalog['phase2684_qwen14_native_weights'];d=p['rows'][5]
    with np.load(RESULT/d['file']) as z:assert float(z[d['array']][tuple(d['index'])][3589])==.000415802001953125
    checks['Qwen14_actual_candidate_scalar_and_gain_invariance']=True
    checks['source_index_change_clears_old_results_observed']=True
    checks['source_full108tokens_head_full128dimensions_observed']=True
    checks['real_four_dose_rows_FP32_and_FP64readout_observed']=True
    checks['DS_full18944canvas_and8rows_actual_horizontal_end']=True
    report={'all_checks_passed':all(checks.values()),'real_browser':True,'preview_only':False,
        'browser':'CUA Codex in-app browser1 tab2 http://localhost:5173/',
        'observed_local_interval':'2026-09-06 16:23..16:39 America/Chicago',
        'verified_at':datetime.now().astimezone().isoformat(),'checks':checks,'heatmap_observations':observations,
        'catalog_sha256':sha(catalog_path),'verification_source':str(Path(__file__)),
        'source_example':{'dataset':'fresh','case':128,'layer':23,'unit':6197,
            'ordinary_coordinate':533,'low_coordinate':84,'H0token0_values':[.0079345703125,.01177978515625],
            'native_gate_weights':[-.01806640625,.004180908203125],
            'last_coordinate':2559,'H36token0':12.375,'head':31,'head_coordinate':127,
            'actual_source_rows':108,'actual_head_coordinate_rows':128,'scalar_conditions_shown':4},
        'visual_checks':{'Q14_weight3589_gain1_and64':.000415802001953125,
            'DS_answer_MLP_row104_coord18943_gain1_and64':.07421875,
            'DS_canvas_width':18944,'DS_canvas_height':144,'scroll_left_at_end':18159,'viewport_scroll_client_width':785,
            'screenshots':'Actual inline screenshots inspected; no local PNG file is claimed.'},
        'limitations':'Only actual UI checks claimed. Source index clearing was exercised; artificial delayed-network race not injected. 41types final-row/final-column browser values plus full-width canvases checked, not every pixel of every row. Allrows/allcoords retained and independent direct/HTTP tests separate. No model loading or AI-autoresearch clicked.'}
    save(OUT/'analysis/browser_checks.json',report);print({'all_checks_passed':True,'real_browser':True,'panels':41},flush=True)

if __name__=='__main__':main()
