import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import test from 'node:test';
import { prepareResearchBundle, uploadResearchBundle } from '../../frontend/src/components/v8/researchBundleUpload.js';

const path = new URL('../../release/ai2050-research-qwen3-4b-b35.tar.gz', import.meta.url);

test('the client validates the released tar and resumes without rewriting registered objects', async () => {
  const bundle = await prepareResearchBundle(new Blob([await readFile(path)]));
  assert.equal(bundle.objects.size, 100);
  const [study] = bundle.studies;
  const ordered = [...bundle.objects.keys()];
  const registered = new Set();
  let interrupted = true;
  let writes = 0;
  async function request(path, options) {
    if (path.endsWith('/research-space')) return {
      schema: 'research_space_index.v1',
      artifacts: [...registered].map(sha => ({ sha })),
      studies: registered.size === 100 ? [study] : [],
    };
    if (path.endsWith('/ingest')) {
      assert.equal(options.method, 'POST');
      if (interrupted && writes === 3) throw new Error('interrupted');
      const sha = createHash('sha256').update(options.rawBody).digest('hex');
      assert.equal(sha, ordered[writes++]);
      registered.add(sha);
      return { sha256: sha };
    }
    assert.ok(path.endsWith(study.report_sha256));
    return { run_id: study.run_id, model_snapshot: study.model_snapshot };
  }
  await assert.rejects(uploadResearchBundle(bundle, request, () => {}), /interrupted/);
  assert.equal(registered.size, 3);
  interrupted = false;
  const progress = [];
  const resumed = await uploadResearchBundle(bundle, request, (n, total) => progress.push([n, total]));
  assert.equal(resumed.uploaded, 97);
  assert.deepEqual(progress.at(-1), [100, 100]);
  assert.equal((await uploadResearchBundle(bundle, request, () => {})).uploaded, 0);
  assert.equal(writes, 100);
});
