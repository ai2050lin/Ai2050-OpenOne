"""Independent scalar derivative unit checks and calibrated evidence interpretation."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2623_native_parameter_algorithms import field,finite_rank_one,SOURCE

W=field('final_down_weights');gamma=field('final_norm_weights');errors={'gate':[],'up':[],'down':[]}
def silu(x):return x/(1+np.exp(-x))
for case in (0,97,194,291,388,485,582,679):
    h=field('hidden_anchor_boundary')[case,-1,-1].astype('float64');x=field('final_mlp_input')[case].astype('float64')
    w=field('output_weight_contrast')[case].astype('float64')*gamma
    for j,k in ((0,0),(853,3242),(1706,6485),(2559,9727)):
        g=float(field('final_gate')[case,k]);u=float(field('final_up')[case,k]);a=float(field('mlp_anchor_boundary')[case,-1,-1,k]);v=W[:,k].astype('float64');eta=1e-4
        dplus=(silu(g+eta*x[j])-silu(g))*u;dminus=(silu(g-eta*x[j])-silu(g))*u
        gate_fd=(finite_rank_one(h,w,v,dplus)-finite_rank_one(h,w,v,dminus))/(2*eta)
        up_fd=(finite_rank_one(h,w,v,silu(g)*eta*x[j])-finite_rank_one(h,w,v,-silu(g)*eta*x[j]))/(2*eta)
        e=np.zeros(len(h));e[j]=1
        down_fd=(finite_rank_one(h,w,e,eta*a)-finite_rank_one(h,w,e,-eta*a))/(2*eta)
        errors['gate'].append(abs(gate_fd-float(field('gradient_gate')[case,k])*x[j]))
        errors['up'].append(abs(up_fd-float(field('gradient_up')[case,k])*x[j]))
        errors['down'].append(abs(down_fd-float(field('gradient_h')[case,j])*a))
phase29=RESULT/'phase2629_expanded_native_confirmation';cases=read(phase29/'material/cases.json');records=read(phase29/'analysis/native_records.json')
collision=[r['case_id'] for r in records if not r['semantic_first_token_distinct']]
before=[json.loads(x) for x in (RESULT/'phase2621_native_language_behavior/behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()]
after=[json.loads(x) for x in (phase29/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()]
reorder={}
for language in ('en','zh'):
    a=[r for r in before if r['family']=='long_reorder' and r['language']==language and r['form']==0]
    b=[r for r in after if r['family']=='long_reorder' and r['language']==language]
    reorder[language]={'initial_same_form_n':len(a),'initial_same_form_order_correct':sum(r['strict_correct'] for r in a)/len(a),
        'fresh_n':len(b),'fresh_order_correct':sum(r['strict_correct'] for r in b)/len(b),'fresh_content_preserved':sum(r['content_preserved'] for r in b)/len(b)}
summary={'max_fd_derivative_error':{k:float(max(v)) for k,v in errors.items()},'original_task_objective_native_fallback_n':len(collision),
    'fallback_interpretation':'for collisions original task-gradient equals native-gradient by construction; cosine=1 is not an empirical semantic result',
    'same_form_reorder':reorder,'real_arithmetic_checks_passed':all(max(v)<1e-5 for v in errors.values())}
save(phase29/'analysis/scientific_unit_checks.json',summary)
assert summary['real_arithmetic_checks_passed']
with MEMO.open('a',encoding='utf-8') as f:
    f.write('\n\n**Phase2629逐参数公式与比较口径终审补记** ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+'] '+json.dumps(summary,ensure_ascii=False)+'。gate/up/down单权重导数均通过额外double有限差分单元检查，但gate/up尚未做真实BF16单权重改动，不能混称三个矩阵都完成了物理参数实验。新旧重排必须同form比较；扩大英语form0的0%不是从混合form初测25%发生能力崩解。48条首token碰撞用native目标回退，其梯度余弦1是定义导致，不作为语义复用证据。\n')
print(json.dumps(summary,ensure_ascii=True))
