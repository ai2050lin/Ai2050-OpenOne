"""Sequential precision phases; each completed Phase is appended exactly once."""
import argparse
from phase2636_precision_engine import *

def main(phase):
    if any(RESULT.glob(f'phase{phase}_*/analysis/final.json')):raise RuntimeError(f'Phase{phase} already complete; do not repeat')
    if phase in (2637,2638):
        precision='bf16' if phase==2637 else 'fp32';out=RESULT/f'phase{phase}_{precision}_native_numeric_control'
        frames=[f for f in read(SOURCE/'material/frames.json') if f['index']==32 and f['variant']==0 and f['step']==0]
        summary,checks=run_precision(precision,frames,out)
    elif phase==2639:
        out=RESULT/'phase2639_expanded_paired_precision_control'
        frames=[f for f in read(SOURCE/'material/frames.json') if f['index']==28 and f['step']==0]
        summary={};checks={}
        for precision in ('bf16','fp32'):
            ss,cc=run_precision(precision,frames,out/precision);summary[precision]=ss;checks.update({precision+'_'+k:v for k,v in cc.items()})
    else:raise ValueError(phase)
    result={'provenance':str(Path(__file__)),'summary':{'frames':len(frames),'results':summary},'checks':checks}
    assert all(checks.values())
    finish(phase,('同数值BF16原生全场与1168条件精度基线' if phase==2637 else '同权重FP32全层1168条件单参数精度裁决' if phase==2638 else '32新前缀4672条件双精度单参数扩大确认'),out,result,
        '两种模式使用同值权重、同前缀、同索引和BF16可表示的同目标参数值；全部Transformer层在CUDA，词嵌入查表和共享输出头留CPU。实际score使用相同FP32两行读出，避免混入末端BF16头舍入。全量保存所有层边界坐标/MLP及28矩阵tokenwise精确因子。',
        r'G^{p}_{jk}=\sum_t\bar Y^p_{t,j}X^p_{t,k};\quad E_p=\frac{\sum_i|\Delta m^p_i-G^p_i\Delta W_i|}{\sum_i|\Delta m^p_i|};\quad\Delta W^{16}_i=\Delta W^{32}_i.',
        '初测16前缀或扩大32前缀，八族中英均覆盖；四层V及早/末层down，原BF16全矩阵最大导数索引和匹配固定索引，两方向、0.02/0.2/1.0倍矩阵RMS。每前缀73条件，另有一次带伴随的基线前向；所有零步长和阴性值保留。',
        '比较同一参数、同一剂量在两种核心计算精度下，自己的导数能否预测自己的有限输出作用。BF16边界场与2632原始GPU基线逐坐标核验，可排除CPU查表搬移引入状态变化；FP32基线差异及共同FP32输出头排名另存，不偷换为相同自然行为。',
        '反传局部性、有限非线性和数值舍入必须分账；同输出分数不等于完整答案。旧最大导数坐标有选择偏差，是数值探针而非固定语义主干。初测/扩大均复用已观察词表和模板；仅一个模型、六个实际干预矩阵，不能外推全语言机制。',
        '完成双精度扩大确认后，按全坐标比较误差在哪些层累积、低幅值场是否改变，并发布精度边界明确的图谱；不以本地数值校验冒充语言机制闭合。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',type=int);main(p.parse_args().phase)
