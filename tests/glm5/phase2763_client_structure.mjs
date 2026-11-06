import { Window } from '../../frontend/node_modules/happy-dom/lib/index.js';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import vm from 'node:vm';
const html=readFileSync(new URL('../../design/client-prototype/index.html',import.meta.url),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(script);
const w=new Window({url:'http://prototype.local/#space'});w.document.write(html.replace(/<script>[\s\S]*?<\/script>/,''));
w.HTMLCanvasElement.prototype.getContext=()=>new Proxy({}, {get:()=>()=>{}});
w.HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','')};
w.HTMLDialogElement.prototype.close=function(){this.removeAttribute('open');this.dispatchEvent(new w.Event('close'))};w.scrollTo=()=>{};
w.eval(script+'\nglobalThis.__sceneState=()=>({pointCount:points.length,towerCount:towerHits.length,tileCol,selected,head});');
const d=w.document,$=s=>d.querySelector(s),click=s=>$(s).click(),change=(s,v)=>{$(s).value=v;$(s).dispatchEvent(new w.Event('change'))};
const checks=[];function check(name,pass){checks.push({name,pass:!!pass});if(!pass)throw Error(name)}
check('four production primary navigation entries',d.querySelectorAll('.nav [data-page]').length===4);
check('four space subtabs',d.querySelectorAll('[data-sub]').length===4);
check('production slate background',html.includes('--bg:#f8fafc'));
check('production Tower colors',html.includes("'#2a78d6'")&&html.includes("'#eb6834'"));
check('no dark space fill',!script.includes("ctx.fillStyle='#15271f'"));
const configs=[['Qwen3-4B',32,9728,35,2560,8],['GLM4-9B',32,13696,39,4096,2],['Qwen3-14B',40,17408,39,5120,8]];
for(const [model,heads,units,max,dim,kv] of configs){
 change('#model',model);
 check(model+' complete head buttons',d.querySelectorAll('[data-head]').length===heads);
 check(model+' complete MLP points',w.__sceneState().pointCount===units);
 check(model+' complete model tower',w.__sceneState().towerCount===max+1);
 check(model+' layer slider',Number($('#layer').max)===max);
 $('#headIndex').value=heads-1;click('#goHead');
 check(model+' last head accessible',$('#nodeName').textContent.includes('H'+(heads-1)));
 change('#weightType','K');check(model+' shared KV address',$('#paramIndex').textContent.includes('k_proj.weight['+((kv-1)*128)+',0]'));
 change('#weightType','Q');$('#paramRow').value=127;$('#paramCol').value=dim-1;click('#goParam');
 check(model+' final Q element accessible',$('#paramIndex').textContent.includes('q_proj.weight['+(heads*128-1)+','+(dim-1)+']'));
 check(model+' final column tiled',$('#tileRange').textContent.includes(String(dim-1)));
 click('[data-tile="right"]');check(model+' tile upper bound',w.__sceneState().tileCol===Math.floor((dim-1)/8)*8);
 click('[data-tile="left"]');check(model+' tile previous block',w.__sceneState().tileCol===Math.floor((dim-1)/8)*8-8);
 if(model==='GLM4-9B'){change('#weightType','K_bias');$('#paramRow').value=127;$('#paramCol').value=0;click('#goParam');check('GLM shared K bias address',$('#paramIndex').textContent.includes('k_proj.bias[255]'))}
 else{change('#weightType','Q_norm');check(model+' layer shared q norm',$('#paramIndex').textContent.includes('q_norm.weight[0]'))}
 $('#unitIndex').value=units-1;click('#goUnit');check(model+' final MLP unit reachable',$('#nodeName').textContent.includes('单元 '+(units-1)));
 change('#weightType','up');check(model+' up parameter address',$('#paramIndex').textContent.includes(model==='GLM4-9B'?'gate_up_proj.weight['+(2*units-1)+',0]':'up_proj.weight['+(units-1)+',0]'));
 change('#weightType','down');$('#paramRow').value=dim-1;click('#goParam');check(model+' down final element',$('#paramIndex').textContent.includes('down_proj.weight['+(dim-1)+','+(units-1)+']'));
 click('#evidenceBtn');check(model+' demo evidence disclosure',$('#dialogBody').textContent.includes('示意'));click('.close');
}
// Changing back from the larger architecture must clamp unit/head indices.
change('#model','Qwen3-4B');check('unit clamps after model change',w.__sceneState().selected===9727);check('head clamps after model change',w.__sceneState().head===31);
click('.nav [data-page="config"]');check('configuration three subtabs',d.querySelectorAll('[data-sub]').length===3);click('[data-sub="1"]');check('contribution belongs to config',$('#contribute').classList.contains('active')&&$('.nav [data-page="config"]').classList.contains('active'));
click('[data-host="remote"]');click('#join');click('#confirmJoin');click('#control');check('managed pause still works',$('#nodeState').textContent.includes('暂停'));click('#revoke');
click('.nav [data-page="research"]');click('[data-sub="1"]');check('dataset belongs to research',$('#datasets').classList.contains('active'));change('#datasetFilter','语法体系');check('dataset filter retained',d.querySelectorAll('#datasetsBody tr').length===2);
click('.nav [data-page="progress"]');check('progress three subtabs',d.querySelectorAll('[data-sub]').length===3);click('[data-sub="2"]');check('rules on demand',!$('#secondaryNotice').classList.contains('hide')&&$('#secondaryNotice').textContent.includes('账本'));click('[data-sub="0"]');check('overview hides secondary explanation',$('#secondaryNotice').classList.contains('hide'));click('[data-area="0"]');check('progress detail retained',$('#dialog').hasAttribute('open'));click('.close');
const folder=new URL('./result/phase2763_client_structure/',import.meta.url);mkdirSync(folder,{recursive:true});writeFileSync(new URL('checks.json',folder),JSON.stringify({scope:'DOM simulation; no-op canvas, no browser pixel/layout verification',checks},null,2));
console.log(JSON.stringify({checks:checks.length,passed:checks.filter(x=>x.pass).length}));await w.happyDOM.close();
