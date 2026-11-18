"""Local research report from audited real measurements; no mechanism promotion."""
import html,json,math,zipfile
import b36_semantic14 as S

NAMES={'pilot':'任意规则图（原始试跑，提前停止）','interface':'规则图接口诊断','arithmetic-interface':'算术表达式与文字指令','fewshot':'增加固定示范','order-pilot':'传递比较','python':'Python 接口诊断','expanded':'48 新家族 Python 校准'}
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def run():
    S.configure(30*2**20);a=S.read(S.OUT/'audit.json');assert a['status']=='passed'
    expanded=next(s for s in a['stages'] if s['stage']=='expanded');passed=all(c['passed'] for c in expanded['cells']);assert not passed,'If behavior passes, continue authorized mechanism phase instead of issuing failure report.'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=[f"Depth {c['depth']} / {c['format']}" for c in expanded['cells']];vals=np.array([c['top1_correct']/c['n'] for c in expanded['cells']]);ci=np.array([c['top1_wilson95'] for c in expanded['cells']])
    fig,ax=plt.subplots(figsize=(11,5.8));y=np.arange(len(vals));ax.barh(y,vals,color=['#548472' if c['passed'] else '#c17a55' for c in expanded['cells']],height=.65)
    ax.errorbar(vals,y,xerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),fmt='none',ecolor='#313d46',capsize=3)
    for i,c in enumerate(expanded['cells']):ax.text(1.02,i,f"{c['top1_correct']}/{c['n']}",va='center',fontsize=10)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(.9,color='#304050',linestyle='--',label='Frozen 90% gate');ax.set_xlim(0,1.14);ax.set_xlabel('Unrestricted next-token accuracy');ax.set_title('Fresh Python calibration: 48 distinct program families');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left');fig.text(.25,.015,'Bars: paired format views. Intervals: descriptive Wilson 95%, not simultaneous.\nNo B36 intervention or heldout confirmation was run.',fontsize=9,color='#4b5964');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(S.OUT/'expanded-calibration.png',dpi=160);plt.close(fig)
    frame_rows=[]
    for stage in a['stages']:
        for f in stage['frames']:frame_rows.append([NAMES[stage['stage']],f['frame'],f"{f['n']}/{f['planned']}",f"{f['strict']}/{f['n']}",f"{f['top1']}/{f['n']}",'未通过' if not f['passed'] else '通过'])
    cells=[[c['depth'],c['format'],f"{c['strict_correct']}/{c['n']}",f"{c['top1_correct']}/{c['n']}",f"[{c['top1_wilson95'][0]:.3f}, {c['top1_wilson95'][1]:.3f}]",'通过' if c['passed'] else '未通过'] for c in expanded['cells']]
    body=f'''<header><p class="eyebrow">{S.RUN} · {S.now()}</p><h1>先检验会不会做，再追踪怎样做</h1><p class="lead">新材料的联合行为校准未通过。本轮没有启动 B36 固定 32 单元的机制正例实验。</p></header>
    <section class="metrics"><div><b>{a['completed_14b_measurements']:,}</b><span>14B 逐题测量，含重复接口</span></div><div><b>{a['unique_construction_families']}</b><span>不同构造家族，均为发现/诊断</span></div><div><b>0</b><span>已开封的确认响应 / 新路径干预</span></div></section>
    <h2>这一轮回答了什么</h2><p>固定 Qwen3-14B 原始 BF16 快照，在 CUDA 上顺序读取原权重。先给任意规则图、算术程序和人物比较材料；分别要求直接答案与轮换后的答案代码，明确每一步的真值。原始试跑失败后，所有接口调整只用发现材料。Python 两步算术在 3 个旧家族上表现很好，因此固定接口后扩展到 48 个新家族。</p>
    <p>扩大校准逐格门槛为至少 44/48：候选答案中正确项严格领先，同时全词表下一 token 等于规定答案。9 格必须全部通过。不以已答对的子集替代失败条件，不把不同代码视图当成独立样本。下游机制分析依此停止；没有声称完成复杂特征逐层形成的验证。</p>
    <h2>48 个新程序家族</h2><img class="plot" src="expanded-calibration.png" alt="48个新程序家族的逐深度、逐答案格式正确率，附描述性区间">{table(['步数','答案格式','候选严格选择','全词表下一token','描述性95%区间','逐格判定'],cells)}
    <p>每个家族含 1/2/3 步、直接数字与两种代码映射。两个代码条件保持 A/B/C/D 的正常排列，只移动其对应数字。48 个新家族的各步程序前缀，与旧 6 个诊断程序及本次其他家族均不重复。数字和基础运算仍会复用；这不是未见运算符的测试。代码仅轮换 2/4 种，尚未覆盖所有映射。</p>
    <h2>完整校准账本</h2>{table(['材料','接口ID','已测/计划','候选严格选择','全词表下一token','全部必需格'],frame_rows)}
    <p>汇总正确率用于定位问题，不能代替逐格准入。原始规则图计划 1,152 条，448 条提交后，错误数已使每个候选句式至少一格不可能达到冻结门槛，因此停止剩余前向。被中断且未提交的分片不计数。后续诊断格有的仅 3 个家族，不用其百分比估计总体能力。</p>
    <h2>评分与组合真值</h2><div class="formula">sₖ = Tₖ(sₖ₋₁); a = π(s_d)<br>m = z[a] − max(z[b], b∈候选且b≠a)<br>候选严格正确：m &gt; 0；直接下一token正确：argmax_v z[v] = token(a)</div>
    <p>s₀ 是题面初始状态，Tₖ 是第 k 个操作，sₖ 是可独立核对的中间真值，π 是答案代码表；直接数字作答时 π 为恒等。z 是实际作答位置的模型 logits。这些是任务与评分定义，不是对模型内部算法的假定。模型可能并行求值或构建全局排序；题目需要三步，不代表存在三个对应神经层。</p>
    <h2>实际反例</h2>'''
    for e in a['examples'][:3]:body+=f'<details><summary>{html.escape(e["id"])}：期望 {html.escape(e["expected"])}，首 token {html.escape(repr(e["actual"]))}</summary><pre>{html.escape(e["core"])}</pre><p>逐步真值：{html.escape(str(e["truth"]))}；候选严格正确：{e["forced_correct"]}</p></details>'
    body+='''<h2>运行器与模型复核</h2><p>4 条明确题目分别走原运行器和标准 HF 的掩码、位置构造，HiddenState 逐位相同、全词表最大 logit 差为 0。原始模型的 14 个权重与 tokenizer 文件已重算哈希，并与本地缓存的 HF 树一致；这不是新的远程认证。</p><p>独立复核中 1+1→2、2+3→5；但 ((1+2)−2)+3→2，标准 HF 自回归续写也为“2&lt;|im_end|&gt;”。因此部分失败确为错误答案，不只是格式不服从。另一份 Qwen3-4B 小样本参考运行对该算式答对，但模型、坐标和证据分开；思考模式触及 256 token 上限的输出标为截断，不计成功。没有据此证明大小模型的一般能力排序。</p>
    <h2>能保留的结论与硬伤</h2><ul><li>观察到具体接口下的答案与格式失败，且扩大后小样本的优秀表现未形成联合准入。不能推出模型普遍缺乏组合能力。</li><li>强制选择只比较预设候选；直接答案条件的完整语义评分和自由生成尚未完成。先输出解释可能随后给对答案。</li><li>本轮仍是探索/行为校准，无独立确认、跨节点复算或新 B36 因果结果。没有在本阶段训练探针、选择路径或采集机制正例原场。</li><li>原始规则图的状态词重复，比较题共用链拓扑；算术的表述变体不是不同语义领域。早期只按完整三步程序分族可能重叠短前缀，扩大校准已显式去重；未来确认也必须检查所有已暴露前缀。</li><li>实际存储与进度日志经过技术审计。技术审计通过不改变行为失败判定。</li></ul>
    <h2>三图谱与下一大阶段</h2><p>外部图谱新增有逐步真值、答案映射和失败范围的材料；内部图谱没有新增经验证的单元；关联图仅增加“未通过行为校准”的边界记录。RDC 没有新统一公式或机制闭合证据。</p><p>下一大阶段应联合校准自然自由生成、模型自行产生的中间状态和答案代码变换：在可承担的生成预算内区分“没算出”“算出但未结束”“最终答错”，先在参考模型建立评分流程，再在 14B 冻结新家族与全部代码映射。只有行为过门，才以共同材料比较层内状态与自回归历史，继而检验 B36 写入、后续 attention/MLP 和净输出中介。不得把人工给出的正确中间状态当成模型自产推理。</p><p>第一性原理上的问题仍是：外部关系复合在内部由什么可复用运算实现，以及它怎样与输出代码转换组合。先辨明沿深度计算与沿生成时间计算，才有资格讨论复杂特征的形成。</p>
    <h2>复核入口</h2><p><a href="audit.json">逐项审计</a> · <a href="expanded-plan.json">冻结的扩大校准计划</a> · <a href="expanded-score/summary.json">完整逐题结果</a> · <a href="hf-independent-check.json">HF 复核</a> · <a href="reference4b-check.json">4B 参考运行</a> · <a href="delivery-source.zip">源码包</a> · <a href="manifest.json">文件清单</a></p><footer>全部新增项目数据位于 D 盘。无服务器发布，无正式证据登记。</footer>'''
    css='''*{box-sizing:border-box}body{font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;color:#273742;background:#f7f9fa;margin:0}main{max-width:1100px;margin:auto;padding:40px 28px}header{border-bottom:2px solid #dae2e7;padding-bottom:24px}h1{font-size:36px;margin:8px 0}h2{margin-top:38px;font-size:23px}.eyebrow{font-size:12px;color:#627681}.lead{font-size:20px;max-width:900px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:25px 0}.metrics div{background:white;border:1px solid #dde5e8;padding:16px}.metrics b{display:block;font-size:29px}.metrics span{font-size:13px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;background:white}td,th{padding:9px 11px;border-bottom:1px solid #e0e6e9;text-align:left;white-space:nowrap}th{background:#e8eef1}a{color:#176581}.plot{width:100%;height:auto}.formula,pre{background:#edf1f3;padding:18px;overflow:auto;white-space:pre-wrap}details{margin:10px 0;background:white;padding:12px;border:1px solid #dde5e8}summary{cursor:pointer}footer{color:#667983;font-size:13px;border-top:1px solid #dbe4e8;margin-top:30px;padding-top:20px}@media(max-width:600px){main{padding:24px 16px}h1{font-size:28px}.metrics{grid-template-columns:1fr}.metrics b{display:inline;margin-right:12px}}'''
    (S.OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B36 新阶段行为校准</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf8')
    S.save('stage-decision.json',dict(status='behavior_gate_failed',expanded_passed_cells=sum(c['passed'] for c in expanded['cells']),expanded_required_cells=9,mechanism_status='not_run_due_to_failed_behavior_gate',fixed_group=S.group(),matched_group=S.matched(),selection_source='branch_reuse14_20261113_v3/frozen-selection.json',reason='The joint behavior gate was required before tracking B36 as a mechanism for2/3-step semantic composition. No individual-correct-trial filtering or threshold change.',completed_measurements=a['completed_14b_measurements'],new_mechanism_interventions=0,heldout_responses_opened=0))
    print('Report written',flush=True)

if __name__=='__main__':run()
