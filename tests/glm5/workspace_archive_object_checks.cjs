const assert = require('node:assert/strict');
const path = require('node:path');
const b23 = require('../../frontend/src/components/v8/data/nativeObservationB23.json');
const b26 = require('../../frontend/src/components/v8/data/nativeObservationB26.json');
const b36 = require('../../frontend/src/components/v8/data/graphObservationExample.json');
const qwen4 = require('../../frontend/src/components/v8/data/nativeObservationQwen4.json');

function fromBf16(encoded,index) {
  const bytes = Buffer.from(encoded,'base64');
  const value = Buffer.alloc(4);
  value.writeUInt16LE(bytes.readUInt16LE(index*2),2);
  return value.readFloatLE();
}

async function verify(page,base,out) {
  const mutations = [];
  page.on('request',request=>{
    if(request.method()==='POST'&&new URL(request.url()).pathname.startsWith('/api/live/'))mutations.push(request.url());
  });
  await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl=Qwen3-14B&observeMode=live');
  const scene = page.getByRole('region',{name:'模型运行 3D 场景'});
  const objects = scene.getByRole('group',{name:'有归档的对象'});
  const readings = scene.getByRole('region',{name:'3D 原生读数'});
  const positions = scene.getByLabel('3D 归档读出位置');
  const open = name=>objects.getByRole('button',{name,exact:true}).click();
  const read = (label,value)=>page.waitForFunction(({label,expected})=>
    document.querySelector(`output[aria-label="${label}"]`)?.textContent===expected,{label,expected:String(value)});
  const unit = value=>read('MLP 单元原值',value);
  const channel = value=>read('Head 通道原值',value);
  const headValue = (row,head)=>fromBf16(row.blocks['26'].fields.head_z,128*head);
  await objects.getByRole('button',{name:'B26/H39',exact:true}).waitFor();
  const qwen14Objects = await objects.getByRole('button').evaluateAll(rows=>rows.map(row=>row.getAttribute('aria-label')));
  assert.deepEqual(qwen14Objects,['B26/H36','B23/U7708','B26/H0','B26/H39','B26/U17407','B36/U2429','B36/U3929']);

  await open('B23/U7708');
  await unit(fromBf16(b23.rows[0].bits,7708));
  assert.match(await readings.innerText(),/wikipedia-zh:18501 · token 945/);
  await positions.selectOption('negative');
  await unit(fromBf16(b23.rows[1].bits,7708));
  await open('B36/U3929');
  await unit(fromBf16(b36.mlp,3*17408+3929));
  await positions.selectOption('0');
  await open('B36/U2429');
  await unit(fromBf16(b36.mlp,2429));
  assert.equal(await positions.inputValue(),'0');

  await open('B26/H0');
  const b26Last = b26.rows.find(row=>row.position===190);
  const b26Middle = b26.rows.find(row=>row.position===95);
  await channel(headValue(b26Last,0));
  assert.equal(await scene.getByLabel('Head 着色量').inputValue(),'strength');
  assert.match(await readings.innerText(),/confirm-000-02 · token 190（ID 510）/);
  await positions.selectOption('95');
  await channel(headValue(b26Middle,0));
  await open('B26/H39');
  await channel(headValue(b26Middle,39));
  const lookup = scene.getByLabel('输入原生编号',{exact:true});
  await lookup.fill('B26/H36');
  await lookup.press('Enter');
  await channel(headValue(b26Middle,36));
  assert.equal(await page.getByRole('region',{name:'B26/H36 归档注意力'}).count(),0);
  await open('B26/U17407');
  const akt95=fromBf16(b26Middle.blocks['26'].fields.mlp_activation,17407);
  await unit(akt95);
  await page.reload();
  await unit(akt95);
  assert.equal(await positions.inputValue(),'95');
  assert.match(await page.getByRole('region',{name:'原生归档来源'}).innerText(),/B26 本机原题复跑/);
  await page.screenshot({path:path.join(out,'archive-shortcuts-qwen14.png')});
  await scene.getByRole('button',{name:'原生单元',exact:true}).click();
  const native = scene.getByRole('region',{name:'全屏原生单元'});
  await native.locator('[data-field-value]').filter({hasText:String(akt95)}).waitFor();
  assert.equal(await native.locator('[data-native-inspector]').getAttribute('data-size'),'17408');
  assert.equal(await native.getByLabel('全屏归档读出位置').inputValue(),'95');
  await scene.getByRole('button',{name:'3D 与控制台',exact:true}).click();
  await unit(akt95);

  await lookup.fill('B25/U17407');
  await lookup.press('Enter');
  await unit('未采集');
  await readings.getByRole('status').filter({hasText:'未随该归档发布'}).waitFor();
  await open('B26/H36');
  await channel(-1.8671875);
  assert.match(await readings.innerText(),/confirm-000-02 · token 190/);

  await scene.getByLabel('模型',{exact:true}).selectOption('Qwen3-4B');
  await objects.getByRole('button',{name:'B26/H14',exact:true}).waitFor();
  const qwen4Objects=await objects.getByRole('button').evaluateAll(rows=>rows.map(row=>row.getAttribute('aria-label')));
  assert.deepEqual(qwen4Objects,['B0/H0','B26/H14','B26/U0','B35/U9727']);
  await open('B26/H14');
  await channel(fromBf16(qwen4.blocks[26].fields.head_z,14*128));
  await open('B35/U9727');
  await unit(fromBf16(qwen4.blocks[35].fields.mlp_activation,9727));
  await page.screenshot({path:path.join(out,'archive-shortcuts-qwen4.png')});
  assert.deepEqual(mutations,[]);
  return {qwen14Objects,qwen4Objects,b26Rows:b26.rows.map(row=>row.position),noLiveRequests:true};
}

module.exports = {verify,handle:()=>false};
