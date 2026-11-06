"""Fix whole-array analysis before opening any native bank values."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    freeze=P.read(P.OUT/'bank-execution-freeze.json');pr=P.read(P.PUBLIC/'prereg.research.json')
    plan={'schema':'mlp_atlas_code_analysis_plan.v1','kind':'real','run_id':P.RUN,'model_snapshot':freeze['model_snapshot'],'provenance':freeze['provenance'],
        'declared_before_bank_values_read_by_analysis':True,'bank_collection_contract_sha256':file_sha(P.OUT/'bank-execution-freeze.json'),'source_sha256':file_sha(__file__),
        'primary':{'object':'A, every36×9728native unit atbothreadouts','fit':'only discoverylatin_ab, average decision contrast acrossbothmapping orientations andallbackgrounds. signofmean; exactzero signremains0. No CD/甲乙 activation fitted.',
            'contrast':'For eachbackground, code andmapping, delta=.5*((A[T=0,Q=0]-A[T=1,Q=0])+(A[T=1,Q=1]-A[T=0,Q=1])). This cancels first-orderfact andquestion terms; equals2*b_TQ.',
            'heldout':'same fixed sign must pass accuracy>=.8, nonzero>=.8, exactone-sidedbinomial p<alpha_per_unit_readout in everycode×mapping ofbothconfirm/lockbox; eachhas192independentbackgrounds.',
            'FWER':'Bonferroni across350208units×2readouts. Allcode×mapping×split tests are anintersection-union rule: everycomponent mustpass, no selectionofthebestcomponent.',
            'behavior':'everyrelevantcode×mapping×fact×question×frame cellmustpass thefrozen.90 gate. Failedbehavior cannot become a functionalmechanism claim.',
            'alpha_per_unit_readout':pr['criteria']['alpha_per_unit_readout']},
        'secondary':{'lexical':'multiply decision contrast by(-1)^mapping; fit separate discovery signs percode. Descriptive response hypotheses only; no additional FWER-controlledsemantic discoveryclaim.',
            'factorial':'retain ordinary8cell coefficients1,T,Q,M,TQ,TM,QM,TQM for eachcode, split andreadout. G/V/A available asactualdata, no claimofnewalgebra.',
            'prediction':'discoverycellmean A predictor, evaluatedperunit onconfirm/lockbox againstdiscoveryglobalmean baseline. Usesoperationalinputfactor annotations, no targetactivation/expectedanswer asinput. Not anindependentmechanism orgraph-onlynext-token model.',
            'parameter_reconstruction':'A @ W_down withactualparameter slices versusactualMLP output. Algebraic source checkwithBF16rounding, notcausal necessity.'},
        'negative_controls':['truth-pairedall37before-selectorhidden hashesmustmatch','truth-pairedquestionembeddinghashesmustmatch','backgroundsplitdisjoint frompreviousnative inputs','discovery-shuffledfit signs fitonlyAB: recordheldoutresultswithoutcallingitindependentnewdata'],
        'no_automatic_closure':'Passing anarrayresponse test upgrades only a scoped response edge. Individual causal use, natural semantics, dictionary andgraph prediction remain separate unknowns.'}
    P.save(P.PUBLIC/'analysis-prereg.research.json',plan);print(json.dumps({'analysis_frozen':True,'alpha':plan['primary']['alpha_per_unit_readout'],'fit_code':'latin_ab'}))
if __name__=='__main__':main()
