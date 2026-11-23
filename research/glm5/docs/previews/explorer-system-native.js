'use strict';

// Full-screen display reads the same frozen input, snapshot and native block as the console.
(() => {
  const byId=id=>document.getElementById(id);
  const ffnMeasures=[['post_gate','门控后的激活 · SiLU(gate) × up'],['gate','gate 预激活'],['up','up 投影']];
  const headMeasures=[['z','Head 输出 z · O 投影前'],['q','Q 查询向量 · RoPE 后'],['k','K 键向量 · RoPE 后'],['v','V 值向量']];
  const sites=['输入末尾','最后一步前','Value 前','Code 前'];

  function choices(select,entries,value){
    const signature=JSON.stringify(entries);
    if(select.dataset.choices!==signature){
      select.replaceChildren(...entries.map(([key,label])=>new Option(label,key)));
      select.dataset.choices=signature;
    }
    select.value=value;
  }

  function create(api,{onUnit,onSample}){
    const host=byId('native-host'), measureSelect=byId('native-measure'), siteSelect=byId('native-site');
    let previousInput='',previousObject='',measureKind='',measure='',site=0,channel=0,recordUnit=null;
    let widget;
    function pick(index){
      if(api.state.kind==='Head'){channel=index;return;}
      if(api.state.mode==='archive' && api.state.archiveId)recordUnit=index;
      onUnit(index);
      render();
    }
    function render(){
      const s=api.state;
      const fromLive=s.mode==='live'?window.ExplorerLive.nativeInput(api):null;
      const r=fromLive?fromLive.record:api.records.find(r=>r.id===(s.mode==='archive'?s.archiveId:s.recordId));
      const matched=r && r.snapshot===s.snapshot && r.model===s.model?r:null;
      const recordMissingObject=s.mode==='archive' && s.archiveId && !api.linked(matched).length;
      host.hidden=Boolean(recordMissingObject);
      byId('native-unlinked').hidden=!recordMissingObject;
      measureSelect.disabled=Boolean(recordMissingObject);
      if(recordMissingObject){
        byId('native-title').textContent='整模型测试';
        choices(measureSelect,[['','无对象索引']],'');choices(siteSelect,[['','无对象索引']],'');siteSelect.disabled=true;
        return;
      }
      const isHead=s.kind==='Head',metrics=isHead?headMeasures:ffnMeasures;
      if(measureKind!==s.kind){measure=metrics[0][0];measureKind=s.kind;}
      const objectKey=[s.model,s.block,s.kind,isHead?s.index:''].join('/');
      const inputKey=[s.mode,s.snapshot,matched?.id||fromLive?.origin||'无归档'].join('/');
      if(inputKey!==previousInput){site=fromLive?.position||0;recordUnit=null;previousInput=inputKey;}
      if(objectKey!==previousObject){channel=0;recordUnit=null;previousObject=objectKey;}
      choices(measureSelect,metrics,measure);
      const head=window.AI2050HeadArchive;
      const headCase=matched && head?.snapshot===s.snapshot && head.run===matched.run?head.cases[matched.caseId]:null;
      const field=window.ExplorerLive.fieldFor(matched);
      const positions=field?field.positions.map((p,i)=>[String(i),sites[i]+' · token '+p.index]):
        headCase?[['0','最后输入 token · '+headCase.position]]:[];
      if(site>=positions.length)site=0;
      choices(siteSelect,positions.length?positions:[['0','当前输入未采集']],String(site));
      siteSelect.disabled=positions.length<2;
      const values=isHead
        ?(s.block===head?.block && s.index===head?.head && measure==='z'?headCase?.values:null)
        :(s.block===field?.block && measure==='post_gate'?window.ExplorerLive.ffnSlice(matched,site):null);
      const dims=api.records.find(row=>row.snapshot===s.snapshot).dims;
      const size=isHead?128:dims.width;
      const token=field?.positions[site]?.index ?? headCase?.position;
      const tokenId=field?.positions[site]?.tokenId ?? headCase?.tokenId;
      const input=matched?'Phase '+matched.phase+' · '+matched.run+' · '+(matched.caseId||matched.sourceKey):
        fromLive?.origin==='custom'?'自定义文本，等待本机采集':'当前没有选定历史题';
      const source=[s.model+' · 快照 '+s.snapshot.slice(0,12),input,
        token!=null?'token '+token+(tokenId!=null?'（ID '+tokenId+'）':''):''].filter(Boolean).join(' · ');
      const measureLabel=metrics.find(([key])=>key===measure)[1];
      const sharedKV=isHead && (measure==='k'||measure==='v')?' · 共享 KV 头 '+Math.floor(s.index/(dims.heads/8)):'';
      const missing=!matched?'当前输入尚未采集此场。选择已采集样例可以查看实测值。':
        '当前归档没有所选对象在该位置的完整'+measureLabel.split(' · ')[0]+'。';
      const selected=isHead?channel:(recordUnit??s.index);
      const comparisons=!isHead && values?field.positions.map((p,i)=>({
        label:sites[i]+' · token '+p.index,
        value:window.ExplorerLive.ffnSlice(matched,i)?.[selected],
        active:i===site
      })):null;
      byId('native-title').textContent=isHead?'Head 内部通道':'FFN 神经元';
      if(!widget)widget=window.AI2050NativeField.create(host,{onPick:pick});
      widget.update({
        identity:[inputKey,objectKey,measure,site].join('/'),kind:isHead?'head':'ffn',size,
        values:values||null,selected,comparisons,measure:measureLabel+sharedKV,source,
        vectorLabel:isHead?'B'+s.block+'/H'+s.index+' · 头内 128 维':'B'+s.block+' · FFN 中间层',
        missing:values?'':missing
      });
    }
    measureSelect.onchange=()=>{measure=measureSelect.value;render();};
    siteSelect.onchange=()=>{
      site=Number(siteSelect.value);
      if(api.state.mode==='live')window.ExplorerLive.useReadout(site);
      render();
    };
    for(const [id,record] of [['native-sample-head','p2983-confirm-000-02'],['native-sample-ffn','p2933-counted0-u2429']]){
      byId(id).onclick=()=>{
        byId(id).closest('details').open=false;
        measureKind='';previousInput='';recordUnit=null;
        onSample(record);
      };
    }
    return {render,selectedRecordUnit:()=>recordUnit,resetSelection(){recordUnit=null;},close(){previousInput='';recordUnit=null;}};
  }
  window.ExplorerNative={create};
})();
