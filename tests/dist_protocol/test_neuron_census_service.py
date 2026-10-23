# -*- coding: utf-8 -*-
"""C0 神经元端点（server/neuron_census_service.py）：用合成普查与语料目录检查索引、单元卡（片段还原、直方图、
证据轴空态）、检索与越界处理。需要本机 qwen3-4b 的 tokenizer.json，缺失时跳过。"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TOK = ROOT / 'models' / 'hf' / 'qwen3-4b' / 'tokenizer.json'


@unittest.skipUnless(TOK.is_file(), '缺少 qwen3-4b tokenizer.json')
class NeuronCensusServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from shared.dist_protocol import census as C
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {k: os.environ.get(k) for k in ('AI2050_NATIVE_DIR', 'AI2050_RESEARCH_DIR')}
        os.environ['AI2050_NATIVE_DIR'] = str(cls.tmp / 'native')
        os.environ['AI2050_RESEARCH_DIR'] = str(cls.tmp / 'research')
        tok = Tokenizer.from_file(str(TOK))
        text = '我们来看一个问题。那么，什么是教学逻辑呢？这是一个常见的问题。' * 3
        ids = np.array(tok.encode(text, add_special_tokens=False).ids, dtype=np.int32)
        cls.hit = int(np.where(ids == tok.encode('呢', add_special_tokens=False).ids[0])[0][1])
        fp = 'synthcorpus00000'
        cdir = cls.tmp / 'native' / '_census' / 'corpus' / fp
        cdir.mkdir(parents=True)
        buf = io.BytesIO(); np.savez(buf, tokens=ids, doc_starts=np.array([0])); (cdir / 'tokens_s.npz').write_bytes(buf.getvalue())
        (cdir / 'manifest.json').write_text(json.dumps({'schema': 'census_corpus.v1', 'sources': [{'source': 'synth', 'file': 'tokens_s.npz'}]}), encoding='utf-8')
        import torch
        F, T = 6, len(ids)
        rng = np.random.default_rng(0)
        G = rng.standard_normal((T, F)).astype(np.float32) * 0.5
        V = rng.standard_normal((T, F)).astype(np.float32) * 0.5
        G[cls.hit, 2], V[cls.hit, 2] = 9.0, 8.0                    # 单元 2 在「呢」处强激活
        A = G / (1 + np.exp(-G)) * V
        acc = C.LayerAcc(F, k=4)
        acc.update(torch.tensor(A), torch.tensor(G), torch.tensor(V), torch.tensor(C.global_index(0, np.arange(T, dtype=np.int64))))
        d = acc.to_numpy()
        d['readout_top'] = np.tile(np.array(tok.encode('不确定', add_special_tokens=False).ids[:1] * 10, dtype=np.int32), (F, 1))
        d['readout_bot'] = np.zeros((F, 10), dtype=np.int32)
        rdir = cls.tmp / 'native' / '_census' / 'mk-synth' / 'run1'
        rdir.mkdir(parents=True)
        buf = io.BytesIO(); np.savez(buf, **d); (rdir / 'L0.npz').write_bytes(buf.getvalue())
        (rdir / 'manifest.json').write_text(json.dumps({
            'schema': 'mlp_census.v1', 'model_key': 'mk-synth', 'model_path': str(TOK.parent), 'corpus_fp': fp,
            'n_tokens_counted': T, 'n_layers': 1, 'd_ff': F, 'k': 4, 'skip_head': 0, 'seq_len': 512, 'sources': [],
            'model_snapshot': {'sha256': 'x'}}), encoding='utf-8')
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from server import neuron_census_service as S
        for fn in (S.census_manifest, S.census_layer, S.corpus, S.tokenizer, S.readout_index, S._selectivity_json):
            fn.cache_clear()
        app = FastAPI(); app.include_router(S.router)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_index_and_unit_card(self):
        idx = self.client.get('/api/neurons/index').json()
        self.assertEqual([r['model_key'] for r in idx['census']], ['mk-synth'])
        card = self.client.get('/api/neurons/unit', params={'model_key': 'mk-synth', 'block': 0, 'unit': 2}).json()
        self.assertEqual(card['schema'], 'unit_card.v1')
        self.assertEqual(card['kind'], 'mlp_neuron')
        top = card['census']['poles']['pos'][0]
        self.assertEqual(top['hit'], '呢')
        self.assertEqual(top['regime'], '++')
        self.assertTrue(top['left'].endswith('什么是教学逻辑'))
        self.assertEqual(sum(sum(v) for v in card['census']['hist']['by_regime'].values()), card['census']['n_tokens'])
        self.assertEqual(card['evidence_axes'], {'read': '未检验', 'control': '未检验', 'explain': '未检验'})
        self.assertEqual(card['template_selectivity'], [])
        self.assertEqual(card['interventions'], [])

    def test_search_and_errors(self):
        r = self.client.get('/api/neurons/search', params={'model_key': 'mk-synth', 'q': 'L0.U2'}).json()
        self.assertEqual(r['results'], [{'block': 0, 'unit': 2}])
        r = self.client.get('/api/neurons/search', params={'model_key': 'mk-synth', 'q': '不确定'}).json()
        self.assertEqual(r['mode'], 'readout')
        self.assertEqual(len(r['results']), 6)
        self.assertEqual(self.client.get('/api/neurons/unit', params={'model_key': 'mk-synth', 'block': 0, 'unit': 99}).status_code, 404)
        self.assertEqual(self.client.get('/api/neurons/unit', params={'model_key': 'nope', 'block': 0, 'unit': 0}).status_code, 404)
        self.assertEqual(self.client.get('/api/neurons/selectivity', params={'run': '../x.json'}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
