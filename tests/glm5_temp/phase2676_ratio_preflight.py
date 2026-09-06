"""Plain-Python zero-denominator unit tests; no model import or phase append."""
import ast,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,save,sha


def main():
    path=ROOT/'tests/glm5/phase2676_numeric_resolution.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='effect_summary')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    summarize=namespace['effect_summary'];zero=summarize([{'effect64':0.,'predicted64':1e-20}]);tiny=summarize([{'effect64':1e-40,'predicted64':2e-40}]);ordinary=summarize([{'effect64':2.,'predicted64':3.},{'effect64':-2.,'predicted64':-1.}])
    checks={'zero_is_null':zero['relative_L1_error64'] is None and not zero['ratio_defined'],
            'zero_absolute_error_retained':zero['mean_abs_error64']==1e-20,
            'nonzero_tiny_not_floored':tiny['relative_L1_error64']==1. and tiny['ratio_defined'],
            'signed_effects_absolute_denominator':ordinary['relative_L1_error64']==.5,
            'json_null':json.loads(json.dumps(zero))['relative_L1_error64'] is None}
    assert all(checks.values())
    result={'all_checks_passed':True,'checks':checks,'examples':{'zero':zero,'tiny':tiny,'ordinary':ordinary},'source_sha256':sha(path),
            'scope':'Numerical summary engineering correction before2676resolution executes. Does not change any material, parameter dose, model computation, or science gate; not a newPhase.'}
    save(RESULT/'phase2676_native_mlp_delivery/analysis/ratio_preflight.json',result);print('ZERO-DENOMINATOR PREFLIGHT',checks)


if __name__=='__main__':main()
