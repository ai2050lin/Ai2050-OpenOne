# -*- coding: utf-8 -*-
"""mech_bench.v1：切分、统计、植入找回、负控制，以及小模型上的仪器恒等式。"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import mech_bench as B


def enc(text):
    return [hash(w) % 9973 for w in text.split()]


def synth(strength, rng, n=120, budgets=B.BUDGETS, track='N', methods=('good', 'bad')):
    """good 的 f 随预算上升（strength 控制），bad 与随机相同。"""
    clusters = [f'c{i // 4}' for i in range(n)]
    full, empty = 3 + rng.normal(0, .2, n), rng.normal(0, .2, n)
    raw = {'clean': full, f'empty|{track}|cf': empty}
    base = lambda b: b ** 0.5 * 0.3
    for b in budgets:
        for d in range(3):
            raw[B.make_key(track, 'keep', 'cf', 'random', b, draw=d)] = empty + (full - empty) * base(b) + rng.normal(0, .2, n)
            raw[B.make_key(track, 'keep', 'cf', 'matched', b, draw=d)] = empty + (full - empty) * base(b) + rng.normal(0, .2, n)
        raw[B.make_key(track, 'keep', 'cf', 'bad', b)] = empty + (full - empty) * base(b) + rng.normal(0, .2, n)
        raw[B.make_key(track, 'keep', 'cf', 'good', b)] = empty + (full - empty) * min(1, base(b) + strength) + rng.normal(0, .2, n)
    for m in methods:
        raw[B.make_key(track, 'keep', 'cf', m, 1.0)] = full.copy()
    for m in ('random', 'matched'):
        for d in range(3):
            raw[B.make_key(track, 'keep', 'cf', m, 1.0, draw=d)] = full.copy()
    raw[f'empty_keep|{track}|cf'] = empty.copy()
    return raw, clusters


class Materials(unittest.TestCase):
    def test_splits_disjoint_and_deterministic(self):
        for words, salt in ((B.NAMES, 'ioi'), (list(B.CAPITALS), 'capital')):
            s = B.split_fillers(words, salt)
            self.assertEqual(s, B.split_fillers(words, salt))
            self.assertEqual(sum(map(len, s.values())), len(set(words)))
            self.assertFalse(set(s['discovery']) & set(s['confirm']) or set(s['confirm']) & set(s['lockbox']))

    def test_lockbox_frames_unseen(self):
        for task in B.TASKS:
            seen = {i for i, _ in B.frames_of(task, 'discovery')}
            self.assertEqual(seen, {i for i, _ in B.frames_of(task, 'confirm')})
            self.assertFalse(seen & {i for i, _ in B.frames_of(task, 'lockbox')})

    def test_items_same_tail_and_length(self):
        for task in B.TASKS:
            for split in ('discovery', 'lockbox'):
                out = B.build_items(task, split, 30, enc)
                self.assertGreater(len(out['items']), 0)
                for it in out['items']:
                    self.assertEqual(len(it['clean_ids']), len(it['corrupt_ids']))
                    self.assertEqual(it['clean_ids'][-1], it['corrupt_ids'][-1])
                    self.assertNotEqual(it['clean'], it['corrupt'])
                    self.assertNotEqual(it['a'], it['b'])

    def test_ioi_corruption_replaces_all_names(self):
        it = B.build_items('rs_ioi', 'discovery', 20, enc)['items'][0]
        self.assertIn(it['answer'], it['clean'])
        self.assertNotIn(it['answer'], it['corrupt'])
        self.assertNotIn(it['contrast'], it['corrupt'])

    def test_every_frame_ends_outside_filler(self):
        for spec in B.TASKS.values():
            for frame in spec['frames']:
                if '{X}' not in frame:
                    self.assertFalse(frame.endswith('}'))


class Stats(unittest.TestCase):
    def test_faithfulness_endpoints(self):
        full, empty = np.array([3., 4.]), np.array([0., 1.])
        self.assertEqual(B.faithfulness(full, full, empty), 1)
        self.assertEqual(B.faithfulness(empty, full, empty), 0)

    def test_budget_counts(self):
        self.assertEqual(B.budget_counts(156, [0.001, 1.0]), [1, 156])

    def test_implanted_method_found(self):
        raw, clusters = synth(0.5, np.random.default_rng(0))
        s = B.score_task(raw, clusters, methods=['good', 'bad'], n_boot=200)
        j = B.judge(s, {'pass': True})
        self.assertEqual(s['identity']['status'], 'pass')
        self.assertEqual(j['methods']['good']['verdict'], '支持')

    def test_random_like_method_rejected(self):
        raw, clusters = synth(0.5, np.random.default_rng(1))
        j = B.judge(B.score_task(raw, clusters, methods=['good', 'bad'], n_boot=200), {'pass': True})
        self.assertEqual(j['methods']['bad']['verdict'], '未支持')

    def test_missing_budget_stays_pending(self):
        raw, clusters = synth(0.5, np.random.default_rng(2))
        del raw[B.make_key('N', 'keep', 'cf', 'good', 0.05)]
        s = B.score_task(raw, clusters, methods=['good', 'bad'], n_boot=50)
        self.assertEqual(s['methods']['good']['status'], 'pending')
        self.assertEqual(B.judge(s, {'pass': True})['methods']['good']['verdict'], '待检')

    def test_missing_empty_is_pending(self):
        raw, clusters = synth(0.5, np.random.default_rng(3))
        del raw['empty|N|cf']
        self.assertEqual(B.score_task(raw, clusters, methods=['good'], n_boot=50)['status'], 'pending')

    def test_identity_failure_blocks_judgement(self):
        raw, clusters = synth(0.5, np.random.default_rng(4))
        raw['empty_keep|N|cf'] = raw['clean'].copy()
        s = B.score_task(raw, clusters, methods=['good', 'bad'], n_boot=50)
        self.assertEqual(B.judge(s, {'pass': True})['status'], 'instrument_failed')

    def test_behavior_failure_blocks_judgement(self):
        raw, clusters = synth(0.5, np.random.default_rng(5))
        s = B.score_task(raw, clusters, methods=['good', 'bad'], n_boot=50)
        self.assertEqual(B.judge(s, {'pass': False})['status'], 'behavior_failed')

    def test_behavior_gate(self):
        self.assertTrue(B.behavior_gate(np.ones(10) * 3, np.zeros(10))['pass'])
        self.assertFalse(B.behavior_gate(np.ones(10) * 0.5, np.zeros(10))['pass'])
        self.assertFalse(B.behavior_gate(np.array([-1.] * 5 + [9.] * 5), np.zeros(10))['pass'])

    def test_cluster_bootstrap_wider_than_item_bootstrap(self):
        rng = np.random.default_rng(6)
        shared = np.repeat(rng.normal(0, 1, 10), 20)
        full, empty = np.ones(200) * 3, np.zeros(200)
        a = {0.1: full * 0.5 + shared, 1.0: full}
        b = {0.1: full * 0.5, 1.0: full}
        grouped = B.paired_auc_diff(a, [b], full, empty, [str(i // 20) for i in range(200)], n_boot=300)
        single = B.paired_auc_diff(a, [b], full, empty, [str(i) for i in range(200)], n_boot=300)
        self.assertGreater(grouped['hi'] - grouped['lo'], (single['hi'] - single['lo']) * 1.5)

    def test_specificity(self):
        self.assertEqual(B.specificity(0.8, [0.1, 0.2])['status'], 'specific')
        self.assertEqual(B.specificity(0.8, [0.1, 0.6])['status'], 'not_specific')
        self.assertEqual(B.specificity(0.05, [0.0, 0.0])['status'], 'no_effect')
        self.assertEqual(B.specificity(0.8, [0.1, None])['status'], 'pending')

    def test_dict_compare_needs_sensitivity(self):
        n = np.ones(8)
        base = {'clean': n * 3, 'corrupt': n * 0, 'empty': n * 2.99}
        self.assertEqual(B.dict_compare(base, [1])['status'], 'no_power')
        raw = {'clean': n * 3, 'corrupt': n * 0, 'empty': n * 2, 'feature|1': n * 2.8, 'neuron|1': n * 2.2, 'feature|all': n * 2.9, 'neuron|all': n * 3}
        out = B.dict_compare(raw, [1])
        self.assertEqual((out['status'], out['feature_wins']), ('compared', 1))
        self.assertAlmostEqual(out['neuron_all'], 1)

    def test_known_head_recall(self):
        s = np.zeros((12, 12))
        for h in B.KNOWN_IOI_HEADS:
            l, i = map(int, h.split('.'))
            s[l, i] = 1
        self.assertEqual(B.known_head_recall(s)['hit'], 26)
        self.assertLess(B.known_head_recall(-s)['hit'], 26)

    def test_known_list_has_26_unique_heads(self):
        self.assertEqual(len(set(B.KNOWN_IOI_HEADS)), 26)


class Instrument(unittest.TestCase):
    """真实 hook 路径：全保留回到干净输出，全替换回到反事实输出。两种架构都测。"""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest('需要 torch')
        spec = importlib.util.spec_from_file_location('mech_bench_run', ROOT / 'scripts' / 'mech_bench_run.py')
        cls.R = importlib.util.module_from_spec(spec)
        sys.modules['mech_bench_run'] = cls.R
        spec.loader.exec_module(cls.R)

    def models(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        yield GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=32, n_embd=32, n_layer=3, n_head=4)).eval()
        yield Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)).eval()

    def test_identity_both_architectures(self):
        import torch
        for model in self.models():
            eng = self.R.Engine(model, 'cpu')
            clean = torch.randint(1, 50, (3, 7))
            corrupt = clean.clone()
            corrupt[:, 2] = (corrupt[:, 2] + 7) % 50
            a, b = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
            base = eng.ld(clean, a, b)
            bad = eng.ld(corrupt, a, b)
            cf = eng.capture(corrupt)
            nc = eng.n_nodes
            keep_all, keep_none = torch.ones(nc, dtype=torch.bool), torch.zeros(nc, dtype=torch.bool)
            eng.set_nodes(keep_all, cf)
            torch.testing.assert_close(eng.ld(clean, a, b), base, rtol=1e-4, atol=1e-4)
            eng.set_nodes(keep_none, cf)
            torch.testing.assert_close(eng.ld(clean, a, b), bad, rtol=1e-4, atol=1e-4)
            eng.clear()
            torch.testing.assert_close(eng.ld(clean, a, b), base, rtol=1e-5, atol=1e-5)

    def test_neuron_identity(self):
        import torch
        for model in self.models():
            eng = self.R.Engine(model, 'cpu')
            clean = torch.randint(1, 50, (2, 6))
            corrupt = clean.clone()
            corrupt[:, 1] = (corrupt[:, 1] + 3) % 50
            a, b = torch.tensor([1, 2]), torch.tensor([4, 5])
            base = eng.ld(clean, a, b)
            cf = eng.capture(corrupt, neurons=True)
            eng.set_units(torch.ones(eng.n_units, dtype=torch.bool), cf)
            torch.testing.assert_close(eng.ld(clean, a, b), base, rtol=1e-4, atol=1e-4)
            eng.clear()

    def test_gradient_attribution_matches_finite_effect_on_linear_step(self):
        """AtP 一阶近似：小扰动下，得分之和应接近实际输出变化。"""
        import torch
        for model in self.models():
            model = model.double()
            eng = self.R.Engine(model, 'cpu')
            clean = torch.randint(1, 50, (2, 6))
            emb = eng.embed(clean).detach()
            pert = emb + 1e-5 * torch.randn_like(emb)   # Qwen3 的 RMSNorm 内部用 float32，扰动不能再小
            a, b = torch.tensor([1, 2]), torch.tensor([4, 5])
            grad_score = eng.embed_grad_probe(emb, pert, a, b)
            actual = eng.ld_embeds(pert, a, b).sum() - eng.ld_embeds(emb, a, b).sum()
            self.assertAlmostEqual(float(grad_score), float(actual), delta=max(1e-10, abs(float(actual)) * 0.05))


if __name__ == '__main__':
    unittest.main()
