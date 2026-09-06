"""Same numerical weights, prefixes, physical coordinates and doses; different arithmetic."""
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2636_precision_contract'
PROBE_SITES=((0,'v_proj'),(5,'v_proj'),(17,'v_proj'),(35,'v_proj'),(0,'down_proj'),(35,'down_proj'))
SCALES=(.02,.2,1.)

def main():
    contract={'phases':{2636:'前轮证据复审、同值双精度与设备一致合同',2637:'八族中英16前缀BF16全场/单参数剂量基线',2638:'同权重FP32全场/单参数对照与早层误差来源',2639:'冻结算法新32前缀双精度扩大确认',2640:'精度残差全坐标跨层图谱、客户端、清理和终审'},
        'model':'local qwen3-4b only, nonquantized; CPU embedding lookup and CPU head storage, all36 transformer blocks/norm/rotary on CUDA in BOTH precisions',
        'initial_frames':'phase2632 index32 variant0 step0, all16 family/language groups',
        'expanded_frames':'phase2632 index28 variants0,1 step0, all32 contexts; observed gradients previously but never finite-probed; not unseen lexical semantic lockbox',
        'readout':'same frozen original BF16 native top1/top2 IDs, same FP32 dot-product readout in both modes; not target-answer loss; separately report baseline FP32-head argmax',
        'weights':'same stored BF16 numerical values promoted exactly to float32; no retraining, no persistent model change',
        'derivatives':'all36 hidden/MLP boundary coordinates plus all-token exact factors of four layers x seven projections; six sites receive actual scalar probes',
        'probes':{'sites':PROBE_SITES,'selectors':['prior per-frame diagnostic maximum','prior fixed matched index'],'scales':SCALES,'signs':[-1,1],
                  'dose':'target_scalar=float32(bfloat16(original + sign*scale*matrixRMS)); SAME target numerical value in BF16 and FP32; report zero steps, actual steps and restoration',
                  'count':'6sites x2selectors x3scales x2signs +noop =73 conditions/frame; extra baseline forward perframe'},
        'precision_controls':['TF32 disabled','same GPU block placement','CPU embedding transfers only exact lookup values','readout always FP32; only core arithmetic precision differs','all28 inspected weight matrices checked against saved2632 FP32 containers'],
        'limits':['FP32 is numerical control, not a new pretrained model or bit-exact real arithmetic','precision can change baseline state and output, so compare own-mode derivative and finite effect','max-gradient test indices chosen from old BF16 data, not semantic core','all coordinate observations precede any new semantic mechanism claim'],
        'next_if_same_goal':'After numeric calibration, enlarge output-matched natural language families; native versus common-task readout identities must remain separate.'}
    save(OUT/'protocol/frozen.json',contract)
    result={'provenance':str(Path(__file__)),'summary':contract,'checks':{'prior2635_complete':read(RESULT/'phase2635_expanded_native_path_confirmation/analysis/terminal_audit.json')['all_checks_passed'],
        'no_new_model_download':True,'same_goal':True}}
    finish(2636,'共享单参数的同值BF16/FP32全层数值对照与完整合同',OUT,result,
        '前轮早层误差大不能自动否定路径；反传忽略中间舍入。冻结实际参数数值、prefix、坐标、剂量、CUDA层位置和FP32读出，仅改变Transformer计算精度。精度变化导致基线变化单列，禁止把新基线当原模型行为完全相同。',
        r'W^{32}=\operatorname{float32}(W^{16});\quad W^*_{jk}=\operatorname{float32}(\operatorname{BF16}(W_{jk}+s\alpha\operatorname{RMS}(W)));\quad \Delta m_p=m_p(W^*)-m_p(W),\ p\in\{16,32\}.',
        '八族中英16前缀初测、另一32前缀双精度扩大；每例73条件。所有36层隐藏/MLP坐标、28矩阵全token因子测绘，实际参数扰动针对四层V及早/末层down六矩阵。',
        '先回答原生导数什么时候可以预测单参数有限作用，再讨论语义功能。末层V阳性是局部数值规律，不等于全模型的条件化编码；早层失败可拆解为可测量数值问题。',
        '没有事先断言舍入是唯一原因；FP32仍有误差和非线性。重新加载同一BF16文件到FP32不会恢复预训练时更高精度的信息。本轮原词表复用，只是新的参数数值对照，不是语义泛化证明。',
        '顺序完成2637—2640所有测试、扩大确认、原坐标客户端与清理；目标相同再进入更广的严格同输出语言族测绘。')

if __name__=='__main__':main()
