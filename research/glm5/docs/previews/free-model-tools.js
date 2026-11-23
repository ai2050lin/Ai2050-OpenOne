/* Offline instruments for the model exploration workspace. No model runs or network reads. */
(() => {
  'use strict';

  const catalog = () => window.FreeModelData;
  const titles = {
    summary:'向量统计', compare:'向量对照', qk:'Q/K 匹配分数', attention:'注意力分布',
    geometry:'层间表征几何', lens:'Logit Lens · 候选词', spectrum:'权重切片奇异值', field:'原生读数'
  };
  const nouns = {head:'Head 通道',ffn:'FFN 中间单元',hidden:'残差流坐标',parameter:'权重元素'};
  const html = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const number = value => Number.isFinite(value) ? Number(value.toPrecision(6)).toString() : '未定义';
  const raw = value => Number.isFinite(value) ? String(value) : '缺值';
  const fmtN = value => value.toLocaleString('en-US');
  const note = message => '<p class="fmt-note">'+html(message)+'</p>';
  const wait = message => '<p class="fmt-empty" role="status">'+html(message)+'</p>';
  const option = (v,label,sel) => '<option value="'+html(v)+'"'+(String(v)===String(sel)?' selected':'')+'>'+html(label)+'</option>';
  const select = (name,label,options) => '<label class="fmt-control"><span>'+html(label)+'</span><select data-fmt="'+name+'">'+options+'</select></label>';
  const metrics = pairs => '<dl class="fmt-metrics">'+pairs.map(([key,value])=>'<div><dt>'+html(key)+'</dt><dd>'+html(value)+'</dd></div>').join('')+'</dl>';
  const finiteVector = (v,size) => (Array.isArray(v)||ArrayBuffer.isView(v)) && v.length===size && size>0 && v.every(Number.isFinite);
  const vectorWidth = (model,kind) => ({head:model?.headDim,ffn:model?.ffn,hidden:model?.hidden})[kind];
  const fieldLabel = f => [f.label || f.caseId,'token '+f.position].filter(Boolean).join(' · ');
  let instance=0;
  function normBoundary(field) {
    if (field?.normalization==='post_final_norm' || field?.boundary==='最终归一化后') return 'post';
    if (field?.normalization==='pre_final_norm' || /^(词嵌入输出|B\d+ 残差输出(?:（归一化前）)?)$/.test(field?.boundary)) return 'pre';
    return null;
  }
  function provenance(field) {
    if (!field) return '';
    const rows = [['模型快照',field.snapshot],['运行',field.run],['题目',field.caseId],['采集边界',field.boundary],['数据来源',field.source],['SHA-256',field.sha256]];
    return '<details class="fmt-source"><summary>读数来源</summary><dl>'+rows.filter(([,v])=>v).map(([k,v])=>'<dt>'+html(k)+'</dt><dd>'+html(v)+'</dd>').join('')+'</dl></details>';
  }
  function statistics(values) {
    let sum=0,squared=0,positive=0,negative=0,low=Infinity,high=-Infinity;
    for (const x of values) {
      sum+=x;
      squared+=x*x;
      if (x>0) positive++;
      if (x<0) negative++;
      low=Math.min(low,x);
      high=Math.max(high,x);
    }
    return [['均值',number(sum/values.length)],['RMS',number(Math.sqrt(squared/values.length))],
      ['L2 范数',number(Math.sqrt(squared))],['最小值',number(low)],['最大值',number(high)],
      ['正 / 零 / 负',[positive,values.length-positive-negative,negative].map(fmtN).join(' / ')]];
  }
  function difference(a,b,i) {
    let aa=0,bb=0,ab=0,dd=0;
    for (let j=0;j<a.length;j++) {
      aa+=a[j]*a[j]; bb+=b[j]*b[j]; ab+=a[j]*b[j]; dd+=(b[j]-a[j])**2;
    }
    return [['余弦相似度',aa&&bb?number(Math.min(1,Math.max(-1,ab/(Math.sqrt(aa)*Math.sqrt(bb))))):'未定义（零向量）'],
      ['差向量 L2',number(Math.sqrt(dd))],['逐项差 RMS',number(Math.sqrt(dd/a.length))],['选中坐标 B − A',raw(b[i]-a[i])]];
  }

  // One-sided Jacobi on a small, complete tile. Scale first and rotate columns
  // of the tall orientation. Norms of the orthogonal columns are the singular values.
  function singularValues(matrix) {
    const m=matrix.length,n=matrix[0].length;
    const cols=Array.from({length:Math.min(m,n)},(_,j)=>m>=n?matrix.map(row=>row[j]):Array.from(matrix[j]));
    const largest=Math.max(...cols.flat().map(Math.abs));
    if (!largest) return cols.map(()=>0);
    for (const col of cols) for (let i=0;i<col.length;i++) col[i]/=largest;
    for (let sweep=0;sweep<80;sweep++) {
      let changed=false;
      for (let j=0;j<cols.length;j++) for (let k=j+1;k<cols.length;k++) {
        const a=cols[j],b=cols[k];
        let aa=0,bb=0,ab=0;
        for (let i=0;i<a.length;i++) {aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];}
        if (!ab || Math.abs(ab)<=1e-13*Math.sqrt(aa*bb)) continue;
        const tau=(bb-aa)/(2*ab);
        const t=(Math.sign(tau)||1)/(Math.abs(tau)+Math.hypot(1,tau));
        const c=1/Math.hypot(1,t),s=t*c;
        for (let i=0;i<a.length;i++) {
          const x=a[i],y=b[i];
          a[i]=c*x-s*y; b[i]=s*x+c*y;
        }
        changed=true;
      }
      if (!changed) return cols.map(col=>largest*Math.hypot(...col)).sort((a,b)=>b-a);
    }
    throw Error('当前切片的奇异值计算未收敛，可缩小范围后重试。');
  }

  function mount(host,{mode='tools',onPick}={}) {
    const listId='fmt-lens-tokens-'+(++instance);
    const root=document.createElement('section');
    root.className='fmt-panel';
    root.innerHTML='<div class="fmt-form" data-fmt-controls></div><div data-fmt-body></div>';
    host.replaceChildren(root);
    const controls=root.querySelector('[data-fmt-controls]'), body=root.querySelector('[data-fmt-body]');
    const preferences=new Map(),cache=new Map();
    let ctx=null,scope='',choice=mode==='readout'?'field':'summary',settings,model,fields=[],widget=null,activeWidgetKey='';
    const inputIdentities=new WeakMap();
    let ticket=0,closed=false,sourceGeneration=0,inputVersion='',lastContextKey='',lastTile=null;

    function disposeField() {widget?.destroy();widget=null;activeWidgetKey='';}
    function metadata() {
      const {kind,block,head,index}=ctx;
      if (ctx.importedFields?.length) return ctx.importedFields;
      return ['head','ffn','hidden'].includes(kind) ? catalog().fields(ctx.model,kind,block,kind==='head'?head:index)||[] : [];
    }
    const fieldA = () => fields.find(f=>String(f.id)===settings.a);
    const fieldB = () => fields.find(f=>String(f.id)===settings.b);
    const coordinate = () => ctx.index;
    const fieldSelector = (key,title,id,allowed=fields) => select(key,title,allowed.map(f=>option(f.id,fieldLabel(f),id)).join(''));
    const peers = () => fields.filter(f=>f.id!==fieldA()?.id && f.snapshot===fieldA()?.snapshot &&
      f.boundary===fieldA()?.boundary);
    const attentionFields = () => {
      const seen=new Set();
      return fields.filter(f=>{
        const id=JSON.stringify([f.snapshot,f.run,f.caseId,f.position]);
        if (seen.has(id) || !(f.rows || f.attention?.weights)) return false;
        seen.add(id);
        return true;
      });
    };
    const qkPair = () => {
      const q=fieldA();
      if (q?.measure!=='head_q' || !String(q.boundary).includes('RoPE 后') || !Number.isInteger(q.kvGroup)) return null;
      const k=fields.find(f=>f.measure==='head_k' && f.snapshot===q.snapshot && f.run===q.run &&
        f.caseId===q.caseId && f.position===q.position && f.tokenId===q.tokenId && f.kvGroup===q.kvGroup && String(f.boundary).includes('RoPE 后'));
      return k?[q,k]:null;
    };
    const action = label => '<button type="button" class="fmt-run" data-fmt-run>'+html(label)+'</button>';
    function noFields() {
      if (ctx.kind==='parameter') return '固定参数可用「权重切片奇异值」分析。向量统计需要选中一个已采集激活的 Head、FFN 或残差流。';
      const noun=nouns[ctx.kind]||'对象';
      if (!vectorWidth(model,ctx.kind)) return '请选择一个 Head、FFN、HiddenState 或命名权重张量。';
      return '此对象有 '+fmtN(vectorWidth(model,ctx.kind))+' 个'+noun+'。当前档案缺少这个模型和位置的完整激活向量。请选择相符的采集记录。';
    }
    function key() {
      return JSON.stringify([scope,choice,settings.a,settings.b,settings.layer,settings.side,settings.tokenA,settings.tokenB,ctx.index,ctx.row,ctx.col,ctx.revision,inputVersion]);
    }
    function show(htmlText) {disposeField();body.innerHTML=htmlText;}
    function inputsForTool() {
      let reason='';
      if (choice==='spectrum' || choice==='summary'&&ctx.kind==='parameter') {
        if (ctx.kind!=='parameter') reason='请在对象目录选中要分析的二维权重矩阵，再定位切片起始行和列。';
        else if (choice==='spectrum'&&catalog().tensor(ctx.model,ctx.tensor)?.shape.length!==2) reason='此参数只有一条坐标轴。奇异值分析需要二维权重矩阵。';
      } else if (choice==='attention' && (ctx.kind!=='head'||!(fieldA()?.rows||fieldA()?.attention?.weights))) {
        reason='需要当前头在查询 token 的完整注意力行，或覆盖全部来源的分组。可查看 Qwen3-4B 归档题或 Qwen3-14B · B26/H36 的五条正文记录。';
      } else if (choice==='qk' && (ctx.kind!=='head'||!qkPair())) {
        reason='需要所选头在相同 token 的 RoPE 后 Q 和该 KV 组的 RoPE 后 K。Qwen3-4B 随稿采集了各头的这两组通道。';
      } else if (choice==='geometry' && (ctx.kind!=='hidden'||fields.length<3)) {
        reason='层间几何需要两层中一一对应的至少三个位置，各有完整残差向量。请在 Qwen3-14B 的 HiddenState 中选择一层。';
      } else if (choice==='lens' && (ctx.kind!=='hidden'||!fieldA())) {
        reason='Logit Lens 需要有实测向量的 HiddenState、同快照输出权重和最终归一化边界。请先选中一个残差层。';
      } else if (choice==='lens' && (fieldA().snapshot!==model.snapshot||!normBoundary(fieldA()))) {
        reason='此读数未标明确切的最终归一化边界，或模型快照与输出权重不匹配。';
      } else if (choice!=='spectrum' && !fields.length) reason=noFields();
      else if (choice==='compare' && !peers().length) reason='这一采集边界只有一个可用条件。请补充同一快照和边界上的第二个条件或 token。Q、K、V 和 Z 分别测量。';
      return reason;
    }

    function renderControls() {
      const labelFor=(k,v)=>k==='summary'&&ctx.kind==='parameter'?'权重切片统计':v;
      let content=mode==='readout'?'':select('tool','数学工具',Object.entries(titles).map(([k,v])=>option(k,labelFor(k,v),choice)).join(''));
      if (choice==='field') {
        if (fields.length) content+=fieldSelector('a','采集位置',settings.a);
      } else {
        const candidates=choice==='qk'?fields.filter(f=>f.measure==='head_q'):choice==='attention'?attentionFields():fields;
        if (candidates.length && !['spectrum','geometry'].includes(choice)) content+=fieldSelector('a',choice==='compare'?'条件 A':'采集位置',settings.a,candidates);
        if (!inputsForTool()) {
          if (choice==='compare') content+=fieldSelector('b','条件 B',settings.b,peers());
          if (choice==='geometry') content+=select('layer','比较层',Array.from({length:model.layers+1},(_,i)=>option(i,i?'L'+i+' · B'+(i-1)+' 后':'L0 · 嵌入',settings.layer)).join(''));
          if (choice==='lens') {
            const tokens=model.lensTokens||[];
            const input=(key,title)=>'<label class="fmt-control"><span>'+title+'</span><input type="number" step="1" min="0" max="'+(model.vocab-1)+'" list="'+listId+'" data-fmt="'+key+'" value="'+html(settings[key])+'"></label>';
            content+=input('tokenA','候选 A · token ID')+input('tokenB','候选 B · token ID')+
              '<datalist id="'+listId+'">'+tokens.map(t=>'<option value="'+html(t.id)+'">'+html(t.label)+'</option>').join('')+'</datalist>';
          }
          if (ctx.kind==='parameter') content+=select('side','切片边长',[2,4,8,16].map(i=>option(i,i+' × '+i,settings.side)).join(''));
          content+=action(ctx.kind==='parameter'?'读取并计算':'计算');
        }
      }
      controls.innerHTML=content;
      controls.hidden=!content;
    }

    function hiddenField(f) {
      const i=coordinate(), values=f.values;
      const perPage=12,page=Math.floor(i/perPage);
      const rows=Array.from(values.slice(page*perPage,Math.min((page+1)*perPage,values.length)),(v,k)=>
        '<tr'+(page*perPage+k===i?' aria-current="true"':'')+'><td><button type="button" data-fmt-pick="'+(page*perPage+k)+'">d'+(page*perPage+k)+'</button></td><td>'+raw(v)+'</td></tr>').join('');
      show('<div class="fmt-coordinate"><div><span>'+html(f.boundary)+'</span><b>d'+i+' <output>'+raw(values[i])+'</output></b></div>'+
        '<form data-fmt-index><label>残差坐标 <input aria-label="残差坐标" type="number" required min="0" max="'+(values.length-1)+'" step="1" value="'+i+'"></label><button>定位</button></form></div>'+
        '<div class="fmt-heat"><canvas class="fmt-grid" tabindex="0" aria-label="全部残差坐标的热力图，左右方向键逐个定位" data-fmt-grid></canvas><i class="fmt-cursor" aria-hidden="true"></i></div>'+
        '<div class="fmt-legend" aria-label="对称原值色标"><span data-fmt-low></span><i aria-hidden="true"></i><span data-fmt-high></span></div>'+
        note(fmtN(values.length)+' 个坐标按原生编号从左到右、从上到下排列。蓝色为正，橙色为负，共用对称线性色标。')+
        '<details class="fmt-source"><summary>所选坐标附近的原值</summary><table class="fmt-table"><thead><tr><th>坐标</th><th>原值</th></tr></thead><tbody>'+rows+'</tbody></table></details>'+provenance(f));
      const canvas=body.querySelector('[data-fmt-grid]'),w=128,h=Math.ceil(values.length/w),g=canvas.getContext('2d');
      canvas.width=w; canvas.height=h;
      if (!g) return;
      let peak=0;
      for (const x of values) peak=Math.max(peak,Math.abs(x));
      body.querySelector('[data-fmt-low]').textContent=number(-peak);
      body.querySelector('[data-fmt-high]').textContent=number(peak);
      const marker=body.querySelector('.fmt-cursor');
      marker.style.left=100*((i%w)+0.5)/w+'%';
      marker.style.top=100*(Math.floor(i/w)+0.5)/h+'%';
      const im=g.createImageData(w,h);
      for (let j=0;j<values.length;j++) {
        const v=values[j],t=peak?Math.abs(v)/peak:0,base=v<0?[187,77,37]:[30,92,168];
        for (let k=0;k<3;k++) im.data[j*4+k]=Math.round(249*(1-t)+base[k]*t);
        im.data[j*4+3]=255;
      }
      g.putImageData(im,0,0);
      canvas.title='色标范围 '+number(-peak)+' 至 '+number(peak);
      canvas.onclick=e=>{
        const r=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-r.left)*w/r.width),row=Math.floor((e.clientY-r.top)*h/r.height);
        if (col>=0&&col<w&&row>=0&&row*w+col<values.length) onPick?.(row*w+col,ctx);
      };
      canvas.onkeydown=e=>{
        const next={ArrowRight:i+1,ArrowLeft:i-1,ArrowDown:i+w,ArrowUp:i-w,Home:0,End:values.length-1}[e.key];
        if (next!==undefined) {e.preventDefault();onPick?.(Math.min(values.length-1,Math.max(0,next)),ctx);body.querySelector('[data-fmt-grid]')?.focus();}
      };
    }

    function renderField() {
      if (ctx.kind==='parameter') {
        const tile=ctx.parameterTile||catalog().tile(ctx.model,ctx.tensor,ctx.row,ctx.col,1,1);
        const value=tile?.values?.[0]?.[0];
        show(metrics([['固定参数 ['+ctx.row+', '+ctx.col+']',raw(value)]])+note(Number.isFinite(value)?String(tile.source||'归档权重切片'):'使用参数窗口选取本地分片，可以查看此地址的原值。'));
        return;
      }
      const f=fieldA(),width=vectorWidth(model,ctx.kind);
      if (!width) {show(wait(noFields()));return;}
      if (ctx.kind==='hidden') {
        if (!finiteVector(f?.values,width)) {show(wait(noFields()));return;}
        hiddenField(f);return;
      }
      const widgetKey=[scope,settings.a].join('/');
      if (activeWidgetKey!==widgetKey) {
        show('<div data-fmt-native></div>'+provenance(f));
        widget=window.AI2050NativeField.create(body.querySelector('[data-fmt-native]'),{onPick:i=>onPick?.(i,ctx)});
        activeWidgetKey=widgetKey;
      }
      const i=coordinate();
      widget.update({kind:ctx.kind,identity:widgetKey,size:width,values:f?.values,selected:i,
        vectorLabel:ctx.kind==='head'?'B'+ctx.block+'/H'+ctx.head+' · 头内通道':'B'+ctx.block+' · FFN 中间单元',
        measure:f?.boundary,source:f?fieldLabel(f):'',missing:f?'':noFields(),
        comparisons:ctx.kind==='ffn'&&f?fields.map(x=>({value:x.values?.[i],label:fieldLabel(x),active:x===f})):null});
    }

    function render() {
      if (!ctx || closed) return;
      ticket++;
      renderControls();
      if (choice==='field') {renderField();return;}
      const reason=inputsForTool();
      if (reason) {show(wait(reason));return;}
      const saved=cache.get(key());
      if (saved) {show(saved);return;}
      const prompts={summary:ctx.kind==='parameter'?'从所选行列读取切片，统计其全部固定参数。':'计算完整原生向量的均值、范数与符号分布。',
        compare:'沿相同原生坐标对齐两个向量，计算余弦相似度和 B − A。',
        qk:'沿原生通道计算当前查询与同一 token 键的缩放内积。',
        attention:fieldA()?.attention?'根据所有可见源 token 的注意力计算熵、有效来源数和各位置的权重。':'按五条正文记录及其余 token 共六组统计注意力质量和熵。',
        geometry:'固定同一道题和相同 token 位置，比较两层残差向量之间的关系。',
        spectrum:'从所选行列开始读取正方形切片，边缘按实际大小截取。',
        lens:'沿最终输出权重分别读出两个候选 token 的 logit。自定义 ID 可使用匹配的本地权重分片。'};
      show(note(prompts[choice]));
    }

    async function calculate() {
      const stamp=++ticket,currentKey=key(),a=fieldA(),b=fieldB(),width=vectorWidth(model,ctx.kind);
      try {
        let output='';
        if (ctx.kind==='parameter' && ['spectrum','summary'].includes(choice)) {
          const name=ctx.tensor, r=ctx.row,c=ctx.col,side=settings.side;
          show(wait('正在读取当前切片…'));
          const tile=await catalog().readTile(ctx.model,name,r,c,side,side);
          if (stamp!==ticket||closed) return;
          const rows=tile?.values,cols=rows?.[0]?.length;
          if (!cols || !rows.every(row=>finiteVector(row,cols))) throw Error('权重切片含缺值。请在参数窗口关联对应模型分片、缩小切片，或定位到一块完整归档的区域。');
          if (choice==='spectrum') {
            const sigmas=singularValues(rows),energy=sigmas.reduce((s,x)=>s+x*x,0);
            output=metrics([['最大奇异值',number(sigmas[0])],['首项平方占比',energy?(100*sigmas[0]**2/energy).toFixed(1)+'%':'未定义（全零切片）']])+
            '<ol class="fmt-spectrum">'+sigmas.map((s,i)=>'<li><span>σ'+(i+1)+'</span><i style="width:'+(sigmas[0]?s/sigmas[0]*100:0)+'%"></i><output>'+number(s)+'</output></li>').join('')+'</ol>'+
            note('奇异值按降序排列。首项平方占比为 σ₁² / Σσᵢ²。');
          } else output=metrics(statistics(rows.flat()));
          output+=note(name+' · 行 '+r+'–'+(r+rows.length-1)+'，列 '+c+'–'+(c+cols-1)+' · 仅此切片。')+note(tile.source||'本地参数');
        } else if (choice==='lens') {
          const ids=[settings.tokenA,settings.tokenB];
          if (ids.some(id=>!Number.isSafeInteger(id)||id<0||id>=model.vocab)||ids[0]===ids[1]) throw Error('请填写两个不同且在词表范围内的 token ID。');
          if (!finiteVector(a.values,width)) throw Error('归档缺少此层、此位置的完整有限残差向量。');
          const norm=normBoundary(a), eps=model.rmsNormEps;
          if (norm==='pre' && !(Number.isFinite(eps)&&eps>0)) throw Error('模型配置缺少最终 RMSNorm 的 ε。');
          show(wait('正在读取候选词的输出权重…'));
          const [wA,wB,gamma]=await Promise.all([catalog().readWeightVector(ctx.model,'lm_head.weight',ids[0]),
            catalog().readWeightVector(ctx.model,'lm_head.weight',ids[1]),
            norm==='pre'?catalog().readWeightVector(ctx.model,'model.norm.weight'):null]);
          if (stamp!==ticket||closed) return;
          if (!finiteVector(wA?.values,width) || !finiteVector(wB?.values,width)) throw Error('至少一个候选缺少完整输出权重行。请关联 lm_head 对应分片，共享输出模型使用词嵌入分片。');
          if (norm==='pre'&&!finiteVector(gamma?.values,width)) throw Error('缺少同一快照的最终 RMSNorm 权重。请关联包含 model.norm.weight 的分片。');
          let inverseRms=1;
          if (norm==='pre') {
            const squared=a.values.reduce((s,x)=>s+x*x,0);
            inverseRms=1/Math.sqrt(squared/width+eps);
          }
          let scoreA=0,scoreB=0;
          for (let i=0;i<width;i++) {
            const x=norm==='pre'?a.values[i]*inverseRms*gamma.values[i]:a.values[i];
            scoreA+=x*wA.values[i]; scoreB+=x*wB.values[i];
          }
          const label=id=>(model.lensTokens||[]).find(t=>t.id===id)?.label||'token';
          output=metrics([[label(ids[0])+' · '+ids[0],number(scoreA)],[label(ids[1])+' · '+ids[1],number(scoreB)],['logit 差 A − B',number(scoreA-scoreB)]])+
            note(norm==='pre'?'使用同快照最终 RMSNorm，再与两个输出行分别作内积。':'该向量已通过最终归一化，直接与两个输出行分别作内积。')+
            note('按归档精度展开至数值读出，仅比较所选 token。词表排名及概率需要完整词表输出。')+
            note('输出权重 · '+[...new Set([wA.source,wB.source])].join('、')+(gamma?' · 最终归一化权重 '+gamma.source:''))+provenance(a);
        } else if (choice==='geometry') {
          const peer=catalog().fields(ctx.model,'hidden',settings.layer,ctx.index)||[];
          const identity=f=>JSON.stringify([f.snapshot,f.run,f.caseId,f.position,f.tokenId]);
          const second=new Map(peer.map(f=>[identity(f),f]));
          if (new Set(fields.map(identity)).size!==fields.length || second.size!==peer.length || fields.length!==peer.length || fields.some(f=>!second.has(identity(f)))) throw Error('两层的快照、运行、题号及 token 位置无法一一对齐。');
          const result=window.AI2050RepresentationGeometry.compare(fields.map(f=>f.values),fields.map(f=>second.get(identity(f)).values));
          if (result.error) throw Error(result.error);
          output=metrics([['线性 CKA',number(result.cka)],['RSA · Spearman ρ',number(result.rsa)]])+
            note('L'+ctx.block+' 与 L'+settings.layer+' · 同题 '+result.count+' 个位置。零方差或距离排序退化时，指标显示未定义。')+provenance(fields[0]);
        } else if (choice==='qk') {
          const [q,k]=qkPair()||[];
          if (!finiteVector(q?.values,width)||!finiteVector(k?.values,width)) throw Error('需要完整且同维的 Q 与 K 通道。');
          let dot=0;
          for (let i=0;i<width;i++) dot+=q.values[i]*k.values[i];
          output=metrics([['Q · K / √d',number(dot/Math.sqrt(width))],['通道数 d',fmtN(width)]])+
            note('B'+ctx.block+'/H'+ctx.head+' · 查询和键均位于 token '+q.position+'。得到这一项注意力分数，重建 softmax 权重还需要此前所有可见键。')+provenance(q)+provenance(k);
        } else if (choice==='attention'&&a.attention) {
          const {weights,tokens,tokenIds,queryPosition,source}=a.attention;
          if (queryPosition!==a.position || !Number.isSafeInteger(queryPosition) || weights?.length!==queryPosition+1 ||
              tokens?.length!==weights.length || tokenIds?.length!==weights.length) throw Error('查询位置和源 token 序列无法与该题的完整注意力行对齐。');
          const result=window.AI2050AttentionStats.summarize(weights);
          if (!result.ok) throw Error(result.reason);
          const tokenLabel=i=>tokens[i]===''?'（空字串）':String(tokens[i]).replaceAll('\n','↵').replaceAll('\t','⇥');
          const row=({index,weight})=>'<tr><td>'+index+'</td><td><code>'+html(tokenLabel(index))+'</code></td><td>'+html(tokenIds[index])+'</td><td>'+raw(weight)+'</td></tr>';
          const table=rows=>'<table class="fmt-table"><thead><tr><th>位置</th><th>源 token</th><th>词表 ID</th><th>注意力原值</th></tr></thead><tbody>'+rows.map(row).join('')+'</tbody></table>';
          output=metrics([['逐 token 熵',number(result.entropyBits)+' bit'],['有效来源数',number(result.effectiveSources)],
            ['前五项累计占比',number(100*result.topMass)+'%']])+
            table(result.ranked.slice(0,5))+
            note('查询位置 '+queryPosition+' · '+result.count+' 个源 token · 原始权重总和 '+number(result.total)+'。统计熵与占比时按该和归一化。')+
            '<details class="fmt-source"><summary>全部源 token · 按原生位置排序</summary><div class="fmt-table-scroll">'+
            table(Array.from(weights,(weight,index)=>({index,weight})))+'</div></details>'+note(source||a.source)+provenance(a);
        } else if (choice==='attention') {
          const rowMass=a.rows.map(r=>r.mass),other=1-rowMass.reduce((s,v)=>s+v,0);
          const result=window.AI2050AttentionStats.summarize([...rowMass,other],{topK:1});
          if (!result.ok) throw Error(result.reason);
          output=metrics([['六组熵',number(result.entropyBits)+' bit'],['有效来源组数',number(result.effectiveSources)]])+
            '<div class="fmt-masses">'+[...rowMass,other].map((x,i)=>'<div><span>'+html(i===rowMass.length?'其余 token':'正文第 '+(i+1)+' 行')+'</span><i><b style="width:'+x*100+'%"></b></i><output>'+(x*100).toFixed(2)+'%</output></div>').join('')+'</div>'+
            note('熵基于上述分组。每一行内各 token 的分配未在此表展开。')+provenance(a);
        } else {
          if (!finiteVector(a?.values,width)) throw Error('所选采集边界的向量缺失、含非有限值或维度与快照不符。');
          if (choice==='compare') {
            if (!finiteVector(b?.values,width)||a.snapshot!==b.snapshot||a.boundary!==b.boundary)
              throw Error('比较需要同一快照和采集边界的两个完整有限向量。');
            if (a.id===b.id) throw Error('A 和 B 指向同一读数。请为 B 选择另一项。');
            output=metrics(difference(a.values,b.values,coordinate()))+
              note(fmtN(width)+' 个'+nouns[ctx.kind]+' · A '+fieldLabel(a)+'，B '+fieldLabel(b))+provenance(a)+provenance(b);
          } else output=metrics(statistics(a.values))+note(fmtN(width)+' 个'+nouns[ctx.kind]+' · '+fieldLabel(a))+provenance(a);
        }
        if (stamp===ticket&&!closed) {
          cache.set(currentKey,output);
          if (cache.size>60) cache.delete(cache.keys().next().value);
          show(output);
        }
      } catch (err) {
        if (stamp===ticket&&!closed) show(wait(err?.message||'无法读取当前测量。'));
      }
    }

    root.addEventListener('change',event=>{
      const el=event.target,key=el.dataset.fmt;
      if (!key) return;
      if (key==='tool') setTool(el.value);
      else settings[key]=el.value!==''&&['layer','side','tokenA','tokenB'].includes(key)?Number(el.value):el.value;
      if (choice==='compare'&&!peers().some(f=>String(f.id)===settings.b)) settings.b=String(peers()[0]?.id||'');
      render();
    });
    root.addEventListener('click',event=>{
      if (event.target.closest('[data-fmt-run]')) calculate();
      const entry=event.target.closest('[data-fmt-pick]');
      if (entry) onPick?.(Number(entry.dataset.fmtPick),ctx);
    });
    root.addEventListener('submit',event=>{
      if (!event.target.matches('[data-fmt-index]')) return;
      event.preventDefault();
      const input=event.target.querySelector('input');
      if (input.reportValidity()) onPick?.(input.valueAsNumber,ctx);
    });
    function setTool(id) {
      choice=id;
      if (id==='qk'&&settings&&fieldA()?.measure!=='head_q') {
        const q=fields.find(f=>f.measure==='head_q');
        if (q) settings.a=String(q.id);
      }
      if (id==='attention'&&settings) {
        const allowed=attentionFields();
        if (allowed.length&&!allowed.includes(fieldA())) settings.a=String(allowed[0].id);
      }
    }

    return {
      update(next) {
        if (closed||!next) return;
        const previous=scope,priorInput=inputVersion,incoming=next.importedFields||[];
        inputVersion=JSON.stringify(incoming.map(f=>{
          if (!inputIdentities.has(f.values)) inputIdentities.set(f.values,++sourceGeneration);
          return [f.id,f.boundary,f.source,inputIdentities.get(f.values)];
        }));
        const changed=priorInput!==inputVersion,changedTile=next.parameterTile!==lastTile;
        lastTile=next.parameterTile;
        ctx={...next};
        model=catalog().model(ctx.model);
        scope=JSON.stringify([ctx.model,ctx.kind,ctx.block,ctx.kind==='head'?ctx.head:null,ctx.kind==='parameter'?ctx.tensor:null,ctx.fieldId??null]);
        const contextKey=JSON.stringify([scope,ctx.index,ctx.row,ctx.col,ctx.revision,inputVersion]);
        if (contextKey===lastContextKey && !(ctx.kind==='parameter'&&choice==='field'&&changedTile)) return;
        lastContextKey=contextKey;
        if (scope!==previous||changed) {
          fields=metadata();
          if (!preferences.has(scope)) preferences.set(scope,{a:String(ctx.fieldId||fields[0]?.id||''),b:String(fields[1]?.id||''),
            layer:Math.max(0,(ctx.block||1)-1),side:8,tokenA:model?.lensTokenIds?.[0]??32,tokenB:model?.lensTokenIds?.[1]??33});
          settings=preferences.get(scope);
          if (!fields.some(f=>String(f.id)===settings.a)) settings.a=String(fields[0]?.id||'');
          setTool(choice);
          if (!peers().some(f=>String(f.id)===settings.b)) settings.b=String(peers()[0]?.id||'');
        }
        render();
      },
      selectTool(id) {
        if (mode!=='readout'&&Object.hasOwn(titles,id)) {setTool(id);render();}
      },
      destroy() {closed=true;ticket++;disposeField();root.remove();}
    };
  }
  window.FreeModelTools=Object.freeze({mount});
})();
