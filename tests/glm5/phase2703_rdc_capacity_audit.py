"""Post-main-result capacity sensitivity: fixed training-input effective DOF, not new confirmation."""
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel,group_ids,binary_report
OUT=CAMPAIGN/'i_factorial'


def main():
    rows=read(OUT/'material.json');tr,va,te=splits(rows)
    immutable(OUT/'capacity_audit_protocol.json',{'source_sha':sha(Path(__file__)),'main_result_sha':sha(OUT/'result.json'),
      'status':'Exploratory post-result sensitivity, not a fresh testset confirmation. Motivated by different effectiveDOF under nominally equalgroupcount.',
      'rule':'Match training-input tr(K(K+lambdaI)^-1) to ntrain/4 by scalar bisection; no labels or testvalues used in lambda choice. Compare same six H24C kernels onboth frozen splits.',
      'limits':['EqualeffectiveDOF is one operational capacity measure, not identical function class.','Main testset is reused for audit; do not count as independent replication.','Unseen layoutgroups14/15 are intentionally absent from jointsplit training, so categorical lookup failure is not native LLM failure.']})
    with np.load(OUT/'features/state.npz') as z:x=unbits(z['h_c'][:,24])
    y=np.array([[r['fact_truth'],r['expected_yes']] for r in rows],np.float32)
    configs=[('shared','linear',None),('quadratic','quadratic',None),('family_shared','conditional','family_language'),('family_independent','independent','family_language'),('hash_shared','conditional','hash16'),('layout_shared','conditional','layout_query')]
    joint=np.array([r['form']==1 and r['style']==1 and r['language']=='zh' for r in rows]);results=[]
    for scope,train,val,test in [('entity_heldout',tr,va,te),('joint_condition_entity',tr[~joint[tr]],va[~joint[va]],te[joint[te]])]:
      for label,kind,group in configs:
        groups=group_ids(rows,group) if group else None;f=FullKernel(x,train,val,test,kind,groups)
        wanted=len(train)/4;lo,hi=-30.,30.
        for _ in range(120):
            mid=(lo+hi)/2;dof=np.sum(f.e/(f.e+np.exp(mid)))
            if dof>wanted:lo=mid
            else:hi=mid
        ridge=float(np.exp((lo+hi)/2));p=f.tq@((f.q.T@y[train])/(f.e[:,None]+ridge))
        result={'scope':scope,'model':label,'fixed_target_dof':wanted,'achieved_dof':float(np.sum(f.e/(f.e+ridge))),'ridge':ridge,**binary_report(y,p,test,rows)}
        if groups is not None:result['test_cases_in_unseen_groups']=int(sum(g not in set(groups[train]) for g in groups[test]))
        results.append(result);npz(OUT/f'predictions/fixed_dof_{scope}_{label}.npz',prediction=p.astype(np.float32),test=test)
        print('FIXED_DOF',scope,label,result['mse'],result['correct'],flush=True)
    save(OUT/'capacity_audit.json',{'timestamp':stamp(),'results':results,'limits':read(OUT/'capacity_audit_protocol.json')['limits']})


if __name__=='__main__':main()
