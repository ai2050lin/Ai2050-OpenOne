const assert = require('node:assert/strict');
const {test} = require('node:test');
require('../../frontend/src/components/v8/representationGeometry.js');
const {compare} = globalThis.AI2050RepresentationGeometry;

function close(actual, expected) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual-expected)<1e-10, actual+' vs '+expected);
}

test('linear CKA uses centered observations, including different feature widths', () => {
  const x = [[1,4], [5,2], [7,8], [9,0]];
  const rotated = x.map(([a,b]) => [30-5*b, 19+5*a, 12]);
  close(compare(x, rotated).cka, 1);
  close(compare([[-1],[0],[1]], [[1],[-2],[1]]).cka, 0);
});

test('paired distance ranks use mean ranks for ties and exclude the diagonal', () => {
  const a = [[1,0],[1,0],[0,1]];
  const b = [[0,1],[1,0],[1,0]];
  const result = compare(a,b);
  assert.deepEqual(result.rdmA, [[0,0,1],[0,0,1],[1,1,0]]);
  close(result.cka, .25);
  close(result.rsa, -.5);
  close(compare(a,a).rsa, 1);
});

test('zero vectors, constant distances and incomplete data never generate an identity score', () => {
  const identical = [[0.1,.3],[0.1,.3],[0.1,.3]];
  const result = compare(identical, identical);
  assert.equal(result.cka, null);
  assert.equal(result.rsa, null);
  const zero = [[0,0],[1,0],[-1,0]];
  const measured = compare(zero,zero);
  close(measured.cka,1);
  assert.equal(measured.rsa,null);
  assert.equal(measured.rdmA[0][0],null);
  assert.equal(measured.rdmA[0][1],null);
  assert.equal(measured.rdmA[1][2],2);
  for (const bad of [[[2]], [null,[1],[2]], [[1],[2],[Infinity]], [[1],[3,4],[2]]]) {
    assert.ok(compare(bad,zero).error);
  }
});
