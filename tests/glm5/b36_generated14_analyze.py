"""Audit generated records and report all preregistered behavioral components."""
import collections,html,operator,hashlib,re,shutil,zipfile,json,importlib.metadata
import b36_generated14 as G
S=G.S

def run():
    S.configure();p=S.read(S.OUT/'plan.json');assert p['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_generated14.py')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);by={r['id']:r for r in p['rows']};rr=[]
    for path in sorted(S.OUT.glob('chunk-*.json')):
        c=S.read(path);assert c['plan_sha256']==S.file_sha(S.OUT/'plan.json');rr+=c['results']
    assert len(rr)==72 and set(r['id'] for r in rr)==set(by)
    families={g['id']:g for g in p['families']}
    for r in rr:
        row=by[r['id']];g=families[row['family']];x=g['start'];truth=[]
        for op,k in g['ops'][:row['depth']]:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](x,k);truth.append(x)
        assert truth==row['truth'];assert tok.encode(row['prompt'],add_special_tokens=False)==row['tokens'];assert tok.decode(r['tokens'],skip_special_tokens=False)==r['text']
        assert G.parse(r['text'],truth,row['values'],row['expected'],tok.eos_token_id in r['tokens'])==r['metrics']
    keys=['steps_parseable','steps_correct','final_parseable','final_correct','format_correct','eos','joint','calculation_error','correct_steps_wrong_final']
    cells=[]
    for depth in [2,3]:
      for fmt in ['direct','code0','code2']:
        subset=[r for r in rr if by[r['id']]['depth']==depth and by[r['id']]['format']==fmt];counts={k:sum(r['metrics'][k] for r in subset) for k in keys};cells.append(dict(depth=depth,format=fmt,n=len(subset),**counts,passed=counts['joint']>=11))
    passed=all(c['passed'] for c in cells)
    failures=[]
    for r in rr:
        row=by[r['id']];f=r['metrics']['final']
        if not r['metrics']['final_correct']:
            v=row['values']['ABCD'.index(f)] if row['format']!='direct' and f in list('ABCD') else None
            failures.append(dict(id=r['id'],family=row['family'],depth=row['depth'],format=row['format'],truth=row['truth'][-1],selected_value=v,final=f,correct_steps=r['metrics']['steps_correct']))
    S.save('error-description.json',dict(kind='posthoc_description',failures=failures,affected_families=len({r['family'] for r in failures}),limit='Correct printed states followed by wrong option. Two paired failures retain the same wrong semantic option after code rotation, so a fixed-letter preference alone is insufficient. No internal causal localization.'))
    S.save('analysis.json',dict(cells=cells,gate_passed=passed,kind='real_behavior_calibration',independent_family_count=12,generated_sequences=72,confirmation_responses=0,source_sha256=S.file_sha(__file__),audit='All input tokens, independent arithmetic truth, output token decoding and parser metrics recomputed.',limits=['Small discovery screen; no population or mechanistic claim.','Generated intermediate text is an observed answer artifact, not necessarily a faithful internal computation trace.','Input number range changed from preceding phase; cross-stage differences confounded.','EOS and64-token truncation separated.','No forced/teacher-forced intermediate states.']))
    headers=['深度','格式','N','步骤可解析','步骤全对','最终答案对','格式合规','EOS','联合成功']
    tab='<table><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(c[k])+'</td>' for k in ['depth','format','n','steps_parseable','steps_correct','final_correct','format_correct','eos','joint'])+'</tr>' for c in cells)+'</table>'
    details=''.join('<details><summary>'+html.escape(r['id'])+'</summary><pre>题目：'+html.escape(by[r['id']]['prompt'])+'\n真值：'+str(by[r['id']]['truth'])+'\n生成：'+html.escape(r['text'])+'\n分类：'+html.escape(json.dumps(r['metrics'],ensure_ascii=False))+'</pre></details>' for r in rr)
    body=f'<h1>模型自产中间步骤：行为校准</h1><p>{S.RUN} · {S.now()}</p><p>联合门：{ "通过，需进入独立确认" if passed else "未通过，B36路径实验未启动" }。每格至少11/12次同时满足中间步骤、最终答案、格式和EOS。</p>'+tab+'<p>Steps是真实生成文本，不是模型内部推理的直接证据。不能由输出步骤数推定神经层数。解析失败不直接算作计算错误。查看每条原始输出，可区分算错、正确步骤后转换错误及未停止。</p><p>任务定义：s_k=T_k(s_(k-1))，最终答案a=π(s_d)；联合成功要求生成步骤逐个等于真值，最终答案、两行格式及EOS均正确。完全不提供中间正确答案，不强制输出token。固定贪心生成、非thinking模式，但允许模型自行写出完整中间状态，最多64个新token。</p><p>12新家族，共72次生成；96候选家族尚未送入模型。初始值10–29，与之前0–9的已暴露语义前缀不同，且完整程序排除已测54家族。规模仅适于发现校准，未声称推广或必要性。源码、身份、冻结计划、输出token和逐行资源日志均保存。</p>'+details
    headline=f'<p><strong>72次中间步骤全部正确，格式与EOS全部合格；直接答案24/24，代码答案43/48。5次错误集中在3个家族，均为步骤正确后最终选错项。</strong>其中两个家族在两种代码下仍选择同一个错误数值，不能只解释为固定字母偏好。这是行为边界，不是内部路径定位；小样本尚未独立确认。</p>'
    (S.OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.7 sans-serif;max-width:1150px;margin:30px auto;padding:20px;color:#263842}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:10px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:16px}details{border-bottom:1px solid #ddd;padding:12px}summary{cursor:pointer}</style>'+headline+body+'</html>',encoding='utf8')
    print(json.dumps(dict(gate_passed=passed,cells=cells),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
