"""Standalone descriptive figure and readable finding summary."""
import html,json
import b36_transition14 as M
S=M.S

def run():
    S.configure();a=S.read(S.OUT/'analysis.json');cells=a['cells'];natural=['no_table','length_control','table','code','full','no_steps','counted0','counted2'];totals={arm:{k:sum(c[k] for c in cells if c['arm']==arm) for k in ['n','step_count_correct','steps_correct','value_correct','slot_correct','code_correct','joint']} for arm in natural+M.HISTORY}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,1,figsize=(13,9),layout='constrained');x=np.arange(len(natural));width=.36
    for i,(key,label,color) in enumerate([('step_count_correct','步骤数量正确','#568ca8'),('value_correct','最终数值正确','#d08b48')]):
        values=[100*totals[arm][key]/totals[arm]['n'] if not (arm=='no_steps' and key=='step_count_correct') else np.nan for arm in natural]
        bars=axes[0].bar(x+(i-.5)*width,values,width,label=label,color=color)
        for bar,value in zip(bars,values):
            if np.isfinite(value):axes[0].text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}',ha='center',fontsize=8)
    axes[0].set_xticks(x,['无表\n步骤＋数值','等长无关\n文字','有表\n步骤＋数值','增加代码','再加行号','去掉步骤','明确次数\n轮换0','明确次数\n轮换2']);axes[0].set_ylim(0,112);axes[0].set_ylabel('正确率（%）');axes[0].set_title('自然生成：配对控制选项表和输出字段，N=96家族');axes[0].legend(loc='lower left');axes[0].grid(axis='y',alpha=.15)
    x=np.arange(4);breakdown={r['arm']:r for r in a['breakdown']}
    for i,(label,color) in enumerate([('最终数值等于任务真值','#568ca8'),('最终数值等于扰动值','#b95c68')]):
        values=[100*(totals[arm]['value_correct'] if i==0 else breakdown[arm]['value_equals_foil'])/totals[arm]['n'] if totals[arm]['n'] else 0 for arm in M.HISTORY]
        bars=axes[1].bar(x+(i-.5)*width,values,width,label=label,color=color)
        for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}',ha='center',fontsize=9)
    history_label='通过历史准入' if a['history_audit']['admitted'] else '未达96/96准入，仅作描述'
    axes[1].set_xticks(x,['原样回放','自产历史＋\n强制Value边界','正确历史＋\n强制Value边界','扰动末值＋\n强制Value边界']);axes[1].set_ylim(0,112);axes[1].set_ylabel('比例（%）');axes[1].legend(loc='lower left');axes[1].set_title(f"历史条件续写：回放逐token一致{a['history_audit']['exact']}/{a['history_audit']['n']}，{history_label}\n人工条件不是自主计算");axes[1].grid(axis='y',alpha=.15)
    fig.savefig(S.OUT/'behavior-overview.png',dpi=150);plt.close(fig)
    findings=[]
    for arm in natural:findings.append(f"{arm}：步骤数量正确{totals[arm]['step_count_correct']}/{totals[arm]['n']}，数值正确{totals[arm]['value_correct']}/{totals[arm]['n']}，联合成功{totals[arm]['joint']}/{totals[arm]['n']}。" if arm!='no_steps' else f"no_steps：数值正确{totals[arm]['value_correct']}/96，联合成功{totals[arm]['joint']}/96；该条件不要求输出步骤。")
    for arm in M.HISTORY:findings.append(f"{arm}：任务真值{totals[arm]['value_correct']}/{totals[arm]['n']}，扰动值{breakdown[arm]['value_equals_foil']}/{totals[arm]['n']}。")
    S.save('readable-findings.json',dict(totals=totals,findings=findings,figure_sha256=S.file_sha(S.OUT/'behavior-overview.png'),scope='Descriptive plots; paired inference and admission remain in analysis.json.'))
    report=S.OUT/'report.html';text=report.read_text(encoding='utf8');marker='<h2>七项配对比较（含长度补充）</h2>'
    if '<h2>结果总览</h2>' in text:
        begin=text.index('<h2>结果总览</h2>');end=text.index(marker,begin);text=text[:begin]+text[end:]
    section='<h2>结果总览</h2><img src="behavior-overview.png" alt="配对自然生成与历史条件续写结果" style="max-width:100%;height:auto"><p>步骤数量正确与步骤内容全对分别记账。下图人工提供的历史不计入自主计算；回放是否准入见后文。材料的未见范围仅指本系列已登记材料，不涉及模型预训练。</p><details><summary>逐条件数字</summary><p>'+html.escape(' '.join(findings))+'</p></details>'
    assert marker in text;report.write_text(text.replace(marker,section+marker),encoding='utf8')
    print('Figure and readable findings saved',flush=True)

if __name__=='__main__':run()
