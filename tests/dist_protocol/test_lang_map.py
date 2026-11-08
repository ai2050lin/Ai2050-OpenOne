# -*- coding: utf-8 -*-
"""语言图谱结构（lang_map.v1）：已发布文件与重建结果一致且通过校验；语言目录 69 项全部归位、旧语言树全部可映射；
负对照：重复 ID、目录条目缺失、交叉归属与词的功能边悬空、词缺来源都被校验拦下；sha256 不一致时拒绝读取；
只读端点返回上三层与词层，未知语言项 404。"""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

from shared.dist_protocol import lang_map as LM  # noqa: E402

CATALOG = json.loads((ROOT / 'shared/dist_protocol/lang_catalog.json').read_text(encoding='utf-8'))
TREE = json.loads((ROOT / 'shared/dist_protocol/lang_tree.json').read_text(encoding='utf-8'))


class LangMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        LM.build(out_dir=cls.tmp, created='test')
        cls.built = LM.load(cls.tmp)
        cls.shipped = LM.load()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_shipped_files_match_rebuild(self):
        self.assertEqual(self.shipped['manifest']['files'], self.built['manifest']['files'], '改了结构或词表后要重建：python scripts/build_lang_map.py')

    def test_structure_is_valid_and_complete(self):
        m, w = self.shipped['map'], self.shipped['words']
        self.assertEqual(LM.validate(m, w, CATALOG), [])
        self.assertEqual([d['id'] for d in m['domains']], ['KN', 'RS', 'GR'])
        cat_ids = {i['id'] for d in CATALOG['domains'] for g in d['groups'] for i in g['items']}
        placed = [i['id'] for d in m['domains'] for s in d['subs'] for i in s['items'] if i['origin'] == 'catalog'] + [x['id'] for x in m['dims']]
        self.assertEqual(sorted(placed), sorted(cat_ids))            # 69 项每项恰好一处主归属
        for old in (i['id'] for i in TREE['items']):
            self.assertIn(old, m['tree_compat'])                       # 旧语言树的每一项都能映射
        self.assertEqual(m['stats']['words'], sum(len(v) for v in w['items'].values()))

    def test_flags_follow_data(self):
        for d in self.shipped['map']['domains']:
            for s in d['subs']:
                for it in s['items']:
                    n = len(self.shipped['words']['items'].get(it['id'], []))
                    self.assertEqual(it['flags'], [True, bool(it['example']), n > 0, bool(it['templates'])])
                    self.assertEqual(it['n_words'], n)

    def test_no_model_information(self):
        text = json.dumps(self.shipped['map'], ensure_ascii=False)
        for word in ('Qwen', 'GLM', 'logit', 'hidden', '注意力头', 'MLP'):
            self.assertNotIn(word, text)

    def test_negative_controls(self):
        m, w = self.shipped['map'], self.shipped['words']
        bad = copy.deepcopy(m)
        bad['domains'][0]['subs'][0]['items'].append(copy.deepcopy(bad['domains'][1]['subs'][0]['items'][0]))
        self.assertTrue(any('重复' in e for e in LM.validate(bad, w, CATALOG)))
        bad = copy.deepcopy(m)
        bad['dims'].pop()
        self.assertTrue(any('未归位' in e for e in LM.validate(bad, w, CATALOG)))
        bad = copy.deepcopy(m)
        bad['domains'][0]['subs'][0]['items'][0]['also'] = ['GR.nowhere']
        self.assertTrue(any('交叉归属' in e for e in LM.validate(bad, w, CATALOG)))
        badw = copy.deepcopy(w)
        next(iter(badw['items'].values()))[0]['fn'] = ['NO.SUCH']
        self.assertTrue(any('功能边' in e for e in LM.validate(m, badw, CATALOG)))
        badw = copy.deepcopy(w)
        next(iter(badw['items'].values()))[0]['src'] = ''
        self.assertTrue(any('来源' in e for e in LM.validate(m, badw, CATALOG)))

    def test_tampered_file_is_rejected(self):
        d = Path(tempfile.mkdtemp())
        try:
            for f in self.tmp.iterdir():
                shutil.copy(f, d / f.name)
            (d / 'words.json').write_text((d / 'words.json').read_text(encoding='utf-8').replace('香蕉', '香焦', 1), encoding='utf-8')
            with self.assertRaises(ValueError):
                LM.load(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_endpoints(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import lang_map_service
        app = FastAPI()
        app.include_router(lang_map_service.router)
        c = TestClient(app)
        r = c.get('/api/lang/map')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['stats']['items'], self.shipped['map']['stats']['items'])
        self.assertIn('files', r.json()['manifest'])
        r = c.get('/api/lang/map/words/SEM.CATEGORY')
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()['words']), 100)
        self.assertEqual(c.get('/api/lang/map/words/SYN.MOOD').json()['words'], [])
        self.assertEqual(c.get('/api/lang/map/words/NO.SUCH').status_code, 404)


if __name__ == '__main__':
    unittest.main()
