"""Close the bounded four-phase delivery, never mark the scientific problem as solved."""
from rdc_continuity_common import *

def main():
    audit=read(CAMPAIGN/'delivery_audit.json');api=read(CAMPAIGN/'client_api_audit.json');browser=read(CAMPAIGN/'client_browser_audit.json')
    assert audit['passed'] and api['passed'] and browser['passed'] and browser['scope']=='E/F/G/Q14/GLM4'
    memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_text(encoding='utf8')
    assert '## Phase 2702:' in memo
    extension=read(CAMPAIGN/'e_confirmation/extension_result.json')
    extension['all_unit_diagnostic']=read(CAMPAIGN/'f_continuity/all_unit_audit.json')
    save(CAMPAIGN/'e_confirmation/extension_result.json',extension)
    paired=read(CAMPAIGN/'h_scale/paired_behavior.json')
    for model in ('qwen14','glm4'):
        save(CAMPAIGN/f'h_scale/{model}/extension_result.json',{'paired_behavior':{'scope':paired['scope'],'comparisons':[r for r in paired['comparisons'] if r['model_minus_qwen4']==model]}})
    for run,n in [('e_confirmation',1024),('f_continuity',1024),('g_generation',975),('h_scale_qwen14',256),('h_scale_glm4',256)]:
        announce(run,state='complete',completed=n,total=n,phase_delivery_complete=True,scientific_goal_solved=False)
    phases=[{'phase':2699,'state':'complete','result':'e_confirmation/result.json','deliverables':['fresh1024cases','24old frozen readers','104new grouped comparisons','full coordinate shape/prefix audit']},
      {'phase':2700,'state':'complete','result':'f_continuity/result.json','deliverables':['same ruler atall37checkpoints','allnativeMLP factorsandtrueweights','14earlier-state predictions/baselines']},
      {'phase':2701,'state':'complete','result':'g_generation/result.json','deliverables':['320naturalprefixes975states','20current-target comparisons','975full source/unit output ledgers']},
      {'phase':2702,'state':'complete','result':'h_scale/result.json','deliverables':['nonquantizedQ14andGLM4 serial256each','same256Qwen4comparison','all-unit error diagnostics','real-data client and provenance audit']}]
    next_steps=[
      {'workstream':'Mapping invariance and conditional transfer','question':'Which independently changed word, formulation, role, language or operation causes the old fixed mapping to fail?','design':'Freeze a crossed-material discovery/validation/confirmation design; compare fixed shared, condition-selected and low-complexity compositional maps. Keep base families together; retain allcoordinates and compare equal-capacity controls.','entry_gate':'Cost-controlled pilot and frozen data identities; do not merely replace nouns and refit.'},
      {'workstream':'Native multi-layer predictive composition','question':'Can earlier available states predict routing/gates, native writes and output without observed future factors or final normalizers?','design':'Compare all-coordinate current query with declared historical state/KV summaries; predict allunit factors then write through realweights. Include simple direct predictor, parameter-free persistence and equal-capacity baselines.','entry_gate':'Separate local arithmetic identities from prospective prediction, and report low-energy units and exceptions.'},
      {'workstream':'Long-form language operations','question':'Do transfer rules survive long sentence reorder, reference dependencies and more generation steps?','design':'Pilot content-preserving reorder with exact content/order/format/EOS scores separately, then extend only where native behavior and capture are reliable.','entry_gate':'Freeze length/token budgets after actual cost pilot; short Yes/No success is not evidence here.'},
      {'workstream':'Numerical first-divergence localization','question':'Which native operator first differs when only execution shape changes?','design':'Compare identical valid inputs at the first norm/projection/attention boundary; retain per-coordinate discrepancies and matched-shape controls.','entry_gate':'Do not assume RMSNorm sequence averaging or future-token softmax normalization; compare observed implementation.'}]
    save(CAMPAIGN/'handoff.json',{'timestamp':stamp(),'state':'bounded_plan_complete','phases':phases,'scientific_goal_solved':False,
      'review':'review.json','plan':'plan.json','audit':'delivery_audit.json','retained_raw_bytes':audit['total_raw_bytes'],
      'client':'http://localhost:5173/rdc','api':'http://localhost:5001/api/rdc/runs',
      'next_campaign_proposals_not_started':next_steps,'old_deferred_work':'Phase2691 four-protocol crossmodel campaign remains deferred, not replaced by this check.',
      'reproduce':['.venv/Scripts/python.exe tests/glm5/phase2699_rdc_confirmation_analysis.py','.venv/Scripts/python.exe tests/glm5/phase2700_rdc_fixed_ruler_native.py','.venv/Scripts/python.exe tests/glm5/phase2701_rdc_current_token_analysis.py','.venv/Scripts/python.exe tests/glm5/phase2701_rdc_current_output_ledger.py','.venv/Scripts/python.exe tests/glm5/phase2702_rdc_scale_analysis.py','.venv/Scripts/python.exe tests/glm5/phase2702_rdc_client_audit.py','.venv/Scripts/python.exe tests/glm5/phase2702_rdc_delivery_audit.py'],
      'background_scope':'No further model jobs or automatic research queue. Existing read-only local API and frontend remain available.'})
    save(CAMPAIGN/'terminal.json',{'timestamp':stamp(),'state':'delivery_complete','phase_ids':[2699,2700,2701,2702],'scientific_goal_solved':False,'pending_owned_model_jobs':0})
    print('BOUNDED_DELIVERY_COMPLETE',flush=True)

if __name__=='__main__':main()
