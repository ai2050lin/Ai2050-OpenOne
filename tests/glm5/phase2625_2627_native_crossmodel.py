"""One nonquantized model per process; replication of coordinate arithmetic, not semantics."""
import argparse, numpy as np
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model,MODELS
from phase2621_native_language_material import build
from phase2622_native_field_capture import collect,summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('qwen14','glm4','ds7'),required=True);args=ap.parse_args()
    phase={'qwen14':2625,'glm4':2626,'ds7':2627}[args.model];out=RESULT/f'phase{phase}_{args.model}_native_parameters'
    model,tok=load_model(args.model)
    cases=[r for r in build(tok) if r['index']==6 and r['form']==0]
    save(out/'protocol/frozen.json',{'cases':[r['case_id'] for r in cases],'phase':phase,'algorithm':'same native final-MLP and RMSNorm/head equation, model-local coordinates','no_behavior_or_transfer_claim':True})
    records=collect(model,tok,cases,out,save_full=False);info=read(out/'protocol/model.json')
    result={'provenance':str(Path(__file__)),'summary':{'model':MODELS[args.model],'layers':info['layers'],'hidden':info['hidden_size'],'mlp_neurons':info['intermediate_size'],**summarize(records)},
        'checks':{'all32_cases':len(records)==32,'nonquantized':info['nonquantized'],'finite_native_fields':all(np.isfinite(np.load(p,mmap_mode='r')).all() for p in (out/'field').glob('*.npy')),
            'actual_cuda_participation':any(p.device.type=='cuda' for p in model.parameters())}}
    finish(phase,f'{MODELS[args.model]}非量化模型原生坐标与最终MLP逐参数顺序复验',out,result,
        '在独立进程一次只加载一个模型，BF16、device_map=auto、12GiB GPU/20GiB CPU预算；相同32个文本重新分词，用本模型真实MLP和输出矩阵计算全维原生公式。',
        r'\partial_{W^{down}_{jk}}m=(\partial_jm)a_k,\quad a_k=\operatorname{SiLU}(g_k)u_k;\quad \text{GLM: }[g,u]=\operatorname{chunk}(W_{gate\_up}x).',
        '八族×中英×两变体，固定item6与form0共32个自然prompt，不按该模型行为选择；保留所有层的两位置HiddenState与全部MLP神经元，以及最终层矩阵全参数。',
        '算法可以在不同维度及split/merged gate-up架构上直接寻址原生标量参数；定量报告BF16舍入与FP32记账的差异。跨模型相同下标不作同义解释。',
        '这是原生算法及数据口径的小规模复验，不是语义规律的跨模型普遍性证明；没有在本Phase执行生成翻转或单位干预，chat协议也不是每个模型的最优能力测量。仅两个位置的MLP值，不含所有token MLP。',
        '按模型局部基底积累可查询参数图谱；继续完成其余模型及汇总展示，不把方程一致称条件齿轮同构。')

if __name__=='__main__':main()
