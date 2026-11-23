'use strict';
(() => {
  const fixed = value => Number.isFinite(value) ? value.toFixed(4) : '未定义';
  const metric = (label,value) => '<div><small>'+label+'</small><strong>'+value+'</strong></div>';

  function geometry({field,values,layerA,layerB,labels,esc}) {
    const [,sites,width] = field.hiddenShape;
    const positions = field.positions.map((site,index)=>labels[index]+' · token '+site.index);
    const layerRows = layer=>positions.map((_,i)=>{
      const start=(layer*sites+i)*width;
      return values.subarray(start,start+width);
    });
    const result=window.AI2050RepresentationGeometry.compare(layerRows(layerA),layerRows(layerB));
    if(result.error)return '<p class="lv-footnote">'+esc(result.error)+'</p>';
    const cells = result.rdmA.flatMap((row,i)=>Array.from(row,(v,j)=> j>i
      ? '<tr><th scope="row">'+esc(labels[i])+' / '+esc(labels[j])+'</th><td>'+fixed(v)+'</td><td>'+fixed(result.rdmB[i][j])+'</td></tr>':'')).join('');
    return '<div class="lv-archived-metrics">'+metric('线性 CKA',fixed(result.cka))+metric('RSA · Spearman ρ',fixed(result.rsa))+'</div>'+
      '<p class="lv-footnote">同一道题的 '+sites+' 个对齐位置，每个位置 '+width.toLocaleString('en-US')+' 个原生坐标。分数描述这几个位置的层间结构，跨题复现仍待检验。</p>'+
      '<details class="lv-plugin-details"><summary>核对位置与成对距离</summary><p>距离采用 1 − 余弦。RSA 比较两层距离的排序，零范数或排序没有变化时显示未定义。</p><p>'+positions.map(esc).join('；')+'</p>'+
      '<table><thead><tr><th>归档位置</th><th>hs['+layerA+']</th><th>hs['+layerB+']</th></tr></thead><tbody>'+cells+'</tbody></table></details>';
  }

  function attention({head,topK,esc}) {
    const weights=head.rows.map(row=>row.mass);
    const other=1-weights.reduce((sum,v)=>sum+v,0);
    if(other < -1e-5)return '<p class="lv-footnote">归档的来源范围合计超过总质量。</p>';
    const names=head.rows.map((row,i)=>'第 '+(i+1)+' 行 · '+row.name);
    names.push('其余 token 合计');
    const stats=window.AI2050AttentionStats.summarize([...weights,Math.max(0,other)],{topK});
    if(!stats.ok)return '<p class="lv-footnote">'+esc(stats.reason)+'</p>';
    const rows=stats.ranked.map((r,i)=>'<tr><th scope="row">'+(i+1)+' · '+esc(names[r.index])+'</th><td>'+(100*r.probability).toFixed(3)+'%</td></tr>').join('');
    return '<div class="lv-archived-metrics">'+metric('分组熵 · bit',fixed(stats.entropyBits))+metric('有效来源组',fixed(stats.effectiveSources))+
      metric('前 '+stats.topK+' 组质量',(100*stats.topMass).toFixed(2)+'%')+'</div>'+
      '<p class="lv-footnote">最后输入 token 的注意力按五行正文和其余 token 合为六组。这里的熵与有效数量均按组计算。</p>'+
      '<details class="lv-plugin-details"><summary>查看六组的质量排名</summary><table><thead><tr><th>来源范围</th><th>注意力质量</th></tr></thead><tbody>'+rows+'</tbody></table></details>';
  }
  function dictionary({archive,layer,featureIndex,esc}){
    const feature=layer.features[featureIndex];
    if(!feature)return '<p class="lv-footnote">此层没有已归档的特征条目。</p>';
    const yes=v=>v?'通过':'未通过';
    const keys=layer.features.map((f,i)=>'<option value="'+i+'" '+(i===featureIndex?'selected':'')+'>F'+esc(f.id)+'</option>').join('');
    const counter=feature.signFailure?'最弱对包含方向相反的反例':'已检查的最弱对未见符号反例';
    const factor={location:'地点',industry:'行业',query:'查询条件'}[feature.nomination]||feature.nomination;
    const sourceNames=['训练报告','留出复核','特征验证'];
    const sourceLinks=layer.sources.map((s,i)=>'<a target="_blank" rel="noopener" href="../../../../'+esc(s.path)+'">'+esc(sourceNames[i])+'</a>').join(' · ');
    return '<label class="lv-control">归档 SAE 特征<select id="lv-dictionary-feature">'+keys+'</select></label>'+
      '<div class="lv-archived-metrics">'+metric('确认集 FVU',fixed(layer.confirmFVU))+metric('留出集 FVU',fixed(layer.lockboxFVU))+
      metric('问题前 / 回答边界复现',feature.replicated.map(yes).join(' / '))+'</div>'+
      '<p class="lv-footnote">F'+esc(feature.id)+' 的候选因子为「'+esc(factor)+'」。'+esc(archive.model)+' · B'+layer.block+' 的稀疏字典中共有 '+layer.m+' 个特征，归档了 '+layer.features.length+' 个候选。FVU 为重建中未解释的方差比例。</p>'+
      '<details class="lv-plugin-details"><summary>核对 F'+esc(feature.id)+' 的来源与证据</summary><p>本层字典重建门槛'+yes(layer.fidelityGate)+'，行为门槛'+yes(layer.behaviorGate)+'。</p><p>'+esc(counter)+'。后续因果检验资格：'+(feature.eligible?'可进入':'本次未获资格')+'。</p>'+
      '<p>运行 '+esc(archive.run)+'</p><p>模型快照 '+esc(archive.snapshot)+'</p><p>字典 SHA-256 '+esc(layer.dictionarySha)+'</p><p>'+sourceLinks+'</p></details>';
  }
  window.ExplorerPluginViews={geometry,attention,dictionary};
})();
