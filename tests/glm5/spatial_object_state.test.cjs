const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const moduleUrl = pathToFileURL(path.resolve(__dirname, '../../frontend/src/components/v8/spatialObjectState.js'));

test('legacy links identify native blocks, heads, MLP units and embedding coordinates consistently', async () => {
  const { readSpatialObject: read } = await import(moduleUrl);
  for (const [query, expected] of [
    ['pkind=mlp&pblock=34&punit=2663', [35,'mlp',2663,null]],
    ['pkind=head&pblock=0&phead=0', [1,'attn',0,null]],
    ['tblock=18&ttab=mlp&tunit=9031', [19,'mlp',9031,null]],
    ['tblock=12&ttab=attention&thead=7&tunit=0', [13,'attn',7,null]],
    ['ptab=neurons&block=3&unit=19', [4,'mlp',19,null]],
    ['pkind=embedding&prow=100&pcol=7', [0,'embedding',100,7]],
    ['sblock=35&smodule=mlp&sindex=9&pblock=0&punit=1', [35,'mlp',9,null]],
    ['sblock=0&smodule=hs&sindex=0&scoord=0', [0,'hs',0,0]],
  ]) { const value=read(query); assert.deepEqual([value.layer,value.module,value.index,value.coordinate],expected,query); }
  assert.equal(read('sblock=2&sindex=-1').index,null);
  assert.equal(read('sblock=2&sindex=NaN').index,null);
});

test('canonical selection replaces stale object aliases and preserves provenance', async () => {
  const { writeSpatialObject, readSpatialObject } = await import(moduleUrl);
  const initial='http://localhost/workbench?view=response&snapshot=abc&run=R&trace=T&tpos=1023&pblock=34&punit=2663&rlayer=2';
  let changes=0;
  global.window={location:{href:initial},history:{state:{route:1},replaceState(state,_title,url){assert.deepEqual(state,{route:1});window.location.href=String(url);}},dispatchEvent(event){assert.equal(event.type,'spatial-object');changes++;}};
  try {
    const target={layer:19,module:'mlp',index:9031,coordinate:null,mode:''}; writeSpatialObject(target);
    const url=new URL(window.location.href); assert.deepEqual(readSpatialObject(url.search),target);
    for(const [key,value] of [['snapshot','abc'],['run','R'],['trace','T'],['tpos','1023'],['view','response']])assert.equal(url.searchParams.get(key),value);
    for(const key of ['pblock','punit','rlayer'])assert.equal(url.searchParams.has(key),false);
    writeSpatialObject(target);assert.equal(changes,1);
  } finally { delete global.window; }
});
