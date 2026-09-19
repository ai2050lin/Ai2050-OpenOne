"""Append the completed identification bundle to the cumulative scoped evidence."""
from copy import deepcopy
from phase2744_rdc_query_identifiability import *


def main():
    assert read(OUT/'queue/status.json')['all_passed'];a=read(OUT/'analysis/result.json');old=read(BASE/'theory_snapshot.json')
    if any(str(p['phase'])=='2744' for p in old['puzzles']):
        supplement=OUT/'analysis/pair_change_control.json'
        if supplement.exists() and old.get('identifiability_pair_change_control_sha256')!=sha(supplement):
            control=read(supplement);assert control['all_passed']
            save(BASE/'theory_history'/f'phase2744_before_pair_control_{sha(BASE/"theory_snapshot.json")}.json',old)
            old.update(timestamp=stamp(),source=snapshot(__file__),identifiability_pair_change_control=control,
              identifiability_pair_change_control_sha256=sha(supplement))
            old['puzzles'][-1]['boundary']+=' 配对变化补充核对：有序规则的整体状态/概率优势未转化为配对响应变化预测优势；未见查询上连零变化基线也未胜过。'
            old['puzzles'][-1]['artifacts'].append('identifiability/analysis/pair_change_control.json')
            old['first_principles_insight']+=' 共同响应拟合与条件差异拟合必须分开评估：一个预测器可改善整体向量误差，却仍未提取相同词袋下的关系变化。配对误差只是统计目标，并未搬运激活；结果要求寻找能预测条件化查询构造的算子，而非把总体拟合收益命名为关系机制。'
            save(BASE/'theory_snapshot.json',old)
        return
    assert len(old['puzzles'])==42 and len(old['formulas'])==40
    save(BASE/'theory_history'/f'phase2743_{sha(BASE/"theory_snapshot.json")}.json',old)
    result=deepcopy(old);result.update(timestamp=stamp(),source=snapshot(__file__),inherited_2743_snapshot_sha256=sha(BASE/'theory_snapshot.json'))
    result['puzzles'].append({'phase':'2744','retained_puzzle':'五类关系160个严格token多重集配对、冻结规则跨域预测、实际训练增量的关系行为，以及独立validation选定的概率校准替代解释。',
      'boundary':'相同token多重集只排除bag-only解释，不排除位置或浅层序列规则；已有语义/词汇材料成分不宣称未见。标量概率校准只作评分对照，原生生成未使用校准；部署旧训练增量不算新训练，也未匹配真实/置乱的累计参数位移。',
      'evidence_status':'prospective_identity_control_and_actual_parameter_transfer_with_calibration_alternatives',
      'artifacts':['identifiability/protocol.json','identifiability/analysis/result.json','identifiability/relations/result.json','identifiability/calibration/result.json','identifiability/behavior/result.json'],
      **{k:a[k] for k in ['common_phenomenon','candidate_rules','native_parameter_structure','unseen_composition_prediction','training_formation']}})
    result['formulas'] += [
      {'id':'token_matched_answer_aligned_separation','kind':'measurement_definition_not_unique_semantic_mechanism',
       'expression':'S_pair=(2*y_A-1)*(P(Yes|A,Yes/No)-P(Yes|B,Yes/No)); multiset(tokens(A))=multiset(tokens(B)), question(A)=question(B), y_B=1-y_A.',
       'variables':'A/B are frozen paired prompts with opposite recipe-derived binary answers. All actual token identities/counts match. Conditional probabilities are within the two declared answer tokens, separately from their full-vocabulary mass and natural generation.',
       'evidence':'identifiability/preflight.json; identifiability/analysis/result.json. Positive response separation excludes a strictly bag-only constant-on-pair explanation, not all ordered lexical/position heuristics.'},
      {'id':'held_validation_temperature_prior_mixture','kind':'fitted_scalar_probability_calibration_control',
       'expression':'p_(T,alpha)(v|x)=(1-alpha)*softmax(z(x)/T)_v+alpha*pi_train(v); pi_train(v)=(count_train(v)+1)/(N_train+V); (T,alpha) selected by document-equal validation NLL.',
       'variables':'Positive T from0.5/0.75/1/1.25/1.5/2/3; alpha from0/.01/.05/.1/.2/.5. Original576training-position unigram prior;192separate validation positions select,192prospective content positions evaluate. Each actually trained/native variant is calibrated separately.',
       'evidence':'identifiability/calibration/result.json; identifiability/analysis/result.json. This is a simple alternative predictive explanation for NLL changes, not a native training rule or a uniquely identified reason for learning effects. Calibrators are not used in the reported1600own-history trajectories.'}]
    result['identifiability']=a
    result['global_atlas_interface']['external']+=' Added320controlledexpressions/160exact-token-histogrampairs/80semanticgroups across5families, plus192prospectively evaluated natural contentpositions on96reserved documents.'
    result['global_atlas_interface']['internal']+=' Actual fourBF16trainedparameter deployments, matchedsixquery fields and all9728units atblocks16/35; nativecontrolled100query responses retainallcoordinates.'
    result['first_principles_insight']+=' 进一步区分词汇多重集、顺序/角色信息、输出概率校准与真实参数学习：相同bag下的响应差异尚不能唯一命名语义结构，训练NLL下降也要和温度/词频先验等简单解释竞争。'
    result['new_mathematical_increment']+=' 2744新增严格配对度量与留出校准比较，仍使用已有统计/概率方法，没有新增普遍语言定理。'
    save(BASE/'theory_snapshot.json',result);print('IDENTITY_THEORY',len(result['puzzles']),len(result['formulas']),flush=True)


if __name__=='__main__':main()
