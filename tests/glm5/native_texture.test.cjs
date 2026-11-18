const { test } = require('node:test');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const helpers = () => import(pathToFileURL(path.resolve(__dirname, '../../frontend/src/components/v8/nativeTexture.js')));

test('missing and invalid dimensions do not allocate a canvas or invent a field', async () => {
  const { nativeTexture, nativeGrid } = await helpers();
  // No document exists in this test: a canvas allocation would fail.
  for (const value of [0,-1,NaN,Infinity,-Infinity,undefined,null,'32','unknown',0.5,2**32]) {
    assert.equal(nativeTexture(8,value,()=>assert.fail('must not fill')),null);
    assert.equal(nativeTexture(value,8,()=>assert.fail('must not fill')),null);
    assert.deepEqual(nativeGrid(value),[0,0]);
  }
});

test('valid native grids cover every index and preserve exact pixel values', async () => {
  const { nativeTexture, nativeGrid } = await helpers();
  for(const n of [1,32,2560,9728,13696,13824]) {
    const [w,h]=nativeGrid(n);assert.ok(w>0&&h>0&&w*h>=n&&w*(h-1)<n);
  }
  let copied;
  global.document={createElement(tag){assert.equal(tag,'canvas');return {getContext(){return {
    createImageData(w,h){assert.equal(w,8);assert.equal(h,4);return {data:new Uint8ClampedArray(w*h*4)};},
    putImageData(pixels,x,y){assert.equal(x,0);assert.equal(y,0);copied=pixels.data;}
  };}};}};
  try { const canvas=nativeTexture(8,4,pixels=>{pixels[0]=37;pixels[127]=255;});assert.equal(canvas.height,4);assert.equal(copied[0],37);assert.equal(copied[127],255); }
  finally{delete global.document;}
});
