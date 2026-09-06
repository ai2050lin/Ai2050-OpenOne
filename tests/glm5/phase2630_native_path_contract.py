"""Same-goal native path campaign: shared scalar weights act at all token positions."""
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2630_native_path_contract'

def main():
    previous=read(RESULT/'phase2629_expanded_native_confirmation/analysis/final_storage_delivery.json')
    contract={'phases':{2630:'前轮证据校准与生成轨迹原生参数合同',2631:'192新表面组合的自然全前缀greedy轨迹及双精度读出',2632:'全36层坐标与四层七矩阵的原生全token伴随导数',2633:'早中晚层真实单权重有限扰动及完整token求和对照',2634:'输出身份条件控制、低幅值全场与共享权重跨位置图谱',2635:'同目标扩大确认、原生跨位置参数查询与交付'},
        'model':'local qwen3-4b, CUDA BF16 nonquantized; no concurrent model','layers':[0,5,17,35],
        'modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        'material':'eight families x two languages x indices[12,16,20,24,28,32] x form1 x variants0,1 =192 new prompt combinations; items/templates separately appeared previously',
        'generation':'single-prompt, full-prefix no-cache greedy; up to160 long_reorder,32 others; selected native decisions at steps0,1,4 before EOS or at EOS explicitly labelled',
        'objective':'original native top1/top2 IDs frozen per decision; both BF16 head and diagnostic FP32 head readout; not ground-truth answer objectives',
        'gradient':'native autograd through BF16 operations; all hidden/MLP coordinates, no Top-K reduction; four layers seven matrices exact tokenwise factors',
        'shared_weight_rule':'d margin / d W[j,k] = sum_t adjoint_output[t,j] * input[t,k]; last-token-only approximation checked, not assumed',
        'storage':'full tensors at selected real decision prefixes; exact factors not low-rank approximation; stream full matrices for metrics; publish selected full tensors and scalar-query factors then clean unshown raw packs',
        'scalar_tests':'16 group-balanced step0 frames, all28 matrices; per-frame maximum-absolute full gradient is only numerical diagnostic, not semantic core; matched random index, +/-0.2 and +/-1.0 matrix RMS dosage, actual BF16 step recorded',
        'limits':['AD differentiates a smooth extension and ignores floating rounding discontinuities','whole parameter gradient may cancel across token positions','native output choice may be formatting or EOS','new combinations are not completely new words','no claim of minimal gears, semantic necessity or universal mechanism']}
    save(OUT/'protocol/frozen.json',contract)
    result={'provenance':str(Path(__file__)),'summary':{'planned_phases':list(contract['phases']),'new_prompt_combinations':192,'operator_sites':28,'no_donor':True},
        'checks':{'previous_campaign_delivered':previous['all_checks_passed'],'previous_goal_not_closed':not previous['mechanism_closed']}}
    finish(2630,'从末层单参数读取到真实生成跨位置共享权重的完整合同',OUT,result,
        '前轮最后MLP的单参数外积只因为当前输出对最后层其他token无依赖。提前到早层或K/V时，真实共享权重作用于所有token，必须用全token伴随梯度求和，不能沿用末位置外积。',
        r'Y_t=W X_t,\quad \frac{\partial m}{\partial W_{jk}}=\sum_t\frac{\partial m}{\partial Y_{t,j}}X_{t,k};\qquad G^{last}_{jk}=\frac{\partial m}{\partial Y_{T,j}}X_{T,k}.',
        '以2629已交付结果为起点，六个相关Phase一次完成；192新表面/上下文组合、真实生成多步；四层×七矩阵保留全物理坐标。',
        '重要问题不是再构造一个差分方向，而是辨认实际参数在多位置复用时，哪些输入坐标与下游读取导数共同决定其作用。先建立可核验的跨位置链式算法，再谈语义规律。',
        '上轮小参数实验只测BF16输出margin为零，不能由此断言内部状态完全未变；本轮并测FP32诊断读出，区分末端舍入与内部传播。AD局部敏感度不等于有限干预或语义因果必要性。',
        '执行完整生成轨迹、全量原生伴随图谱、单参数前向验证、条件控制和扩大确认；不创建并发模型，不修改本地模型权重文件。')

if __name__=='__main__':main()
