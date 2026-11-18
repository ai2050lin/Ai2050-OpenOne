export const TASKS = [
  { id: 'binding', name: '实体属性绑定', question: '只改变物体颜色，模型的输出和内部响应怎样变化？' },
  { id: 'negation', name: '否定与纠正', question: '交换被否定与被确认的颜色，模型能否跟随确认信息？' },
  { id: 'order', name: '语序与位置', question: '事实不变、交换记录顺序，模型是否仍读取同一实体？' },
];
export function taskPair(kind, variant = 0) {
  const names = [['木盒','布袋'],['陶杯','纸筒'],['铁罐','竹篮']][variant % 3];
  const [a,b]=names;
  const prefix='你正在阅读一份虚构的仓库检查记录。请只依据这次记录回答，不使用物品通常具有的颜色，也不要猜测没有登记的信息。检查员按流程逐一核对标签、包装和货架位置。清点开始前，工作人员整理了通道，将空白登记表放在桌面，并确认每件物品都有独立名称。记录中的名称仅用于区分物品；记录顺序不表示重要程度。以下两条信息均来自本次检查，没有其他隐含的更正。\n';
  const suffix=`\n核对结束后，工作人员收起工具，清理工作台，并将记录交给下一班。交接时要求将不同物品的信息分开保存，不得把其中一件的属性套到另一件上。本段流程说明没有提供新的颜色信息，也不会改变前面的登记。请根据以上有效记录，回答${a}是什么颜色。只输出颜色名称，不要解释。`;
  const sentences=kind==='negation' ? [`${a}不是红色，而是蓝色。${b}不是蓝色，而是红色。`,`${a}不是蓝色，而是红色。${b}不是红色，而是蓝色。`] : kind==='order' ? [`${a}的颜色是红色。${b}的颜色是蓝色。`,`${b}的颜色是蓝色。${a}的颜色是红色。`] : [`${a}的颜色是红色。${b}的颜色是绿色。`,`${a}的颜色是蓝色。${b}的颜色是绿色。`];
  return sentences.map((s,i)=>({id:`${kind}-v1-${variant}-${i}`,condition:i?'B':'A',text:prefix+s+suffix,expected:kind==='negation'?(i?'红':'蓝'):kind==='order'?'红':i?'蓝':'红'}));
}
// Final input position only: all distinguishing information has appeared.
// Same snapshot/dimensions required; absolute positions can still differ.
export function compareTaskRuns(a,b) {
  if(!a||!b)return {reason:'完成同一组 A、B 后显示逐层比较。',layers:[]};
  if(!a.snapshot||a.snapshot!==b.snapshot)return {reason:'模型快照缺失或不同，不能比较。',layers:[]};
  if(!a.hidden?.length||a.hidden.length!==b.hidden?.length)return {reason:'层数据不完整，不能比较。',layers:[]};
  const layers=[];
  for(let l=0;l<a.hidden.length;l++){
    const x=a.hidden[l],y=b.hidden[l];
    if(!x?.length||x.length!==y?.length||!x.every(Number.isFinite)||!y.every(Number.isFinite))return {reason:'坐标尺寸不匹配或数值缺失，不能比较。',layers:[]};
    let xx=0,yy=0,xy=0,dd=0;for(let i=0;i<x.length;i++){xx+=x[i]*x[i];yy+=y[i]*y[i];xy+=x[i]*y[i];dd+=(y[i]-x[i])**2;}
    layers.push({layer:l,cosine:xx&&yy?Math.max(-1,Math.min(1,xy/Math.sqrt(xx*yy))):null,relativeL2:xx?Math.sqrt(dd/xx):null});
  }
  return {layers,reason:a.tokens.length===b.tokens.length?'输入 token 数相同；仍需区分语义、词汇身份和位置效应。':'输入 token 数不同：比较的是各自最终输入位置，包含长度/位置混杂。'};
}
