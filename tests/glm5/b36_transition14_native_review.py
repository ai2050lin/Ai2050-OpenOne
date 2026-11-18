"""Posthoc audit of answer stability and saturation; no new model execution."""
import html,json
import numpy as np
import b36_transition14 as M
S=M.S


def plot_background():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(S.OUT/'native-field-summary.npz') as z:
        mean=z['mean'];rms=z['rms']
    scaled=mean/np.sqrt((rms*rms).mean(1,keepdims=True)).clip(1e-12)
    shown=np.arcsinh(20*scaled);bound=float(np.abs(shown).max())
    fig,ax=plt.subplots(figsize=(16,5),layout='constrained')
    im=ax.imshow(shown,origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-bound,vmax=bound)
    ax.set_xlabel('Native residual coordinate: all 5120, unchanged order');ax.set_ylabel('Raw boundary (0 = embedding)')
    ax.set_title('Descriptive low-amplitude view: color = asinh(20 * native mean / layer RMS)\nAll coordinates retained; nonlinear display only, not a new feature selection or significance test')
    fig.colorbar(im,ax=ax,label='Transformed display value');fig.savefig(S.OUT/'native-background-view.png',dpi=180);plt.close(fig)


def run():
    S.configure(30*2**20)
    a=S.read(S.OUT/'native-analysis.json');p=S.read(S.OUT/'native-plan.json');ex=S.read(S.OUT/'native-execution.json')
    assert a['native_plan_sha256']==S.file_sha(S.OUT/'native-plan.json')
    effect=[];tail=[];margins=[]
    for entry in ex['records']:
        path=S.OUT/entry['path'];assert S.file_sha(path)==entry['sha256'];r=S.read(path)
        assert S.file_sha(path.parent/'native.npz')==r['arrays_sha256']
        effect+=r['effects']
        with np.load(path.parent/'native.npz') as z:
            x=S.unbits(z['baseline_logits']).astype(np.float64)
        best=x.argmax(1);maximum=x[np.arange(len(x)),best];offset=x-maximum[:,None]
        prob=np.exp(offset);prob[np.arange(len(x)),best]=0.
        remaining=prob.sum(1);tail.extend((remaining/(1+remaining)).tolist())
        second=np.partition(x,-2,axis=1)[:,-2];margins.extend((maximum-second).tolist())
    base={r['id']:r for r in effect if r['arm']=='mask0'}
    summary=[]
    for arm in p['arms']:
        rr=[r for r in effect if r['arm']==arm]
        summary.append(dict(arm=arm,n=len(rr),changed_top1=sum(r['top1']!=base[r['id']]['top1'] for r in rr),
            wrong_code=sum(r['top1'] in r['choice_ids'] and not r['correct'] for r in rr),
            non_code_top1=sum(r['top1'] not in r['choice_ids'] for r in rr)))
    assert len(base)==len(tail)==184
    shape=lambda v:dict(min=float(np.min(v)),median=float(np.median(v)),max=float(np.max(v)))
    text=(f"184个确认视图中，15种内部条件相对自然基线改变首选token的次数分别为{[r['changed_top1'] for r in summary]}。"
        f"自然基线首选token相对全词表第二名的logit间隔范围{min(margins):.3f}–{max(margins):.3f}，中位数{np.median(margins):.3f}；"
        f"从已保存BF16 logits用float64重算的非首选概率质量最大{max(tail):.6g}。因此Code读出高度饱和：这里的正确率指标对小评分变化不敏感。"
        "这不能证明总体无效应、功能冗余或不存在更早的作用，但本轮没有建立固定32单元对该条件Code正确性的必要性，也没有建立正确率层面的路径救援。"
        "范围内评分差确实存在：全位置移除固定组的真值相对固定竞争代码间隔平均变化约+0.791，匹配组约−0.021，范数匹配方向约−0.113；"
        "同时恢复后三层attention与MLP输出时约+0.103。这些是描述性响应差异，不增加确认检验，也不能据此计算唯一中介比例或定位语义首次形成。"
        "原生KL由float32概率计算，数值约1e-9或更小，受饱和与舍入影响，不把这种微小量作为输出分布完全不变的证明。"
        "下一阶段应先用同一次生成的KV状态完成无编辑分支复现，再在Steps/Value接续的非饱和决策位置冻结新材料及读出；不能从当前开封材料中挑选更容易翻转的样本当作独立确认。")
    plot_background()
    result=dict(kind='posthoc_descriptive_audit_not_new_confirmation',analysis_sha256=S.file_sha(S.OUT/'native-analysis.json'),
        source_sha256=S.file_sha(__file__),top1_stability=summary,baseline_top1_margin=shape(margins),
        baseline_non_top1_probability_mass_float64=shape(tail),interpretation=text,
        background_display=dict(figure_sha256=S.file_sha(S.OUT/'native-background-view.png'),input_summary_sha256=S.file_sha(S.OUT/'native-field-summary.npz'),transform='asinh(20 * native mean / per-layer population RMS)',selection='Posthoc visualization for low-amplitude visibility; all coordinates in original order, no clipping or sorting. Raw and ordinary RMS-scaled plots retained.'),
        scope='CPU recalculation of saved baseline logits and comparison of stored top1 IDs only; no new model runs, no changed primary tests or thresholds.')
    S.save('native-review.json',result)
    report=S.OUT/'report.html';body=report.read_text(encoding='utf8')
    marker='<h2>严格审视：高置信代码读出的限制</h2>'
    if marker in body:
        begin=body.index(marker);end=body.index('</p>',body.index('native-review.json',begin))+4;body=body[:begin]+body[end:]
    section=marker+'<p>'+html.escape(text)+'</p><img src="native-background-view.png" alt="保留全部原生坐标的低幅背景显示" style="max-width:100%;height:auto"><p>补充显示使用asinh(20×原值均值/逐层总体RMS)的对称完整色标，以显示低幅背景；全部坐标保持原顺序，没有裁剪或选择坐标。该事后色彩变换不参与特征选择、显著性或机制判定；原值图和普通RMS缩放图仍保留。</p><p><a href="native-review.json">描述性审查与复算</a></p>'
    report.write_text(body.replace('</html>',section+'</html>'),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
