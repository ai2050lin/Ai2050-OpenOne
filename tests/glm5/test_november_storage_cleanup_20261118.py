"""Safety boundaries for the explicitly authorized A1/A2 raw-field retirement."""
import unittest
from pathlib import Path

import november_storage_cleanup_20261118 as C


class CleanupScopeTests(unittest.TestCase):
    def test_known_native_producers_are_retirable(self):
        cases = [
            ('recursive_branch14_20261113_v6', 'new_words/B39-A.npy'),
            ('recursive_branch14_20261113_v5', 'pilot/hidden-all.npy'),
            ('branch_reuse14_20261113_v3', 'confirm/chunk-00000/B19.npz'),
            ('branch_reuse14_20261113_v2', 'pilot/chunk-00064/states.npz'),
            ('b36_composition14_20261113_v3', 'format/chunk-00000/B36.npz'),
            ('source_formation_v1', 'glm9/confirm-native/hidden.npy'),
            ('source_binding_v1', 'qwen14/discovery-native/key.npy'),
            ('source_relay_v2', 'qwen14/confirm-native/trace_state.npy'),
            ('source_relay_v1', 'qwen14/discovery-native/vproj.npy'),
            ('common_bench_b1_repair_v2', 'lockbox-path-account/postattn.npy'),
        ]
        for run, path in cases:
            with self.subTest(run=run, path=path):
                self.assertTrue(C.role(run, Path(path))[0])

    def test_results_inputs_frozen_sources_and_minimal_evidence_are_protected(self):
        cases = [
            ('recursive_branch14_20261113_v6', 'frozen-source.zip'),
            ('recursive_branch14_20261113_v6', 'fit-procedure-combined.npz'),
            ('recursive_branch14_20261113_v6', 'full-field-procedure.npz'),
            ('recursive_branch14_20261113_v6', 'new_words/prefix-embedding.npy'),
            ('branch_reuse14_20261113_v3', 'fits/correlation-B36-r0-parent.npz'),
            ('branch_reuse14_20261113_v3', 'intervention-B36/chunk-00000.npz'),
            ('branch_reuse14_20261113_v3', 'confirm-predictions.npz'),
            ('b36_composition14_20261113_v3', 'tail-confirm/chunk-00000.npz'),
            ('b36_transition14_20261114_v1', 'native/confirm-000/native.npz'),
            ('own_state_explicit_20261115_v2', 'value-slot-001.npz'),
            ('goodfire_middle_20261113_v1', 'qwen4/proof-first4/B18-A.npy'),
            ('goodfire_middle_20261113_v1', 'qwen4/tokens.npy'),
            ('source_formation_v1', 'qwen14/confirm-formation-predictions.npz'),
            ('source_formation_v1', 'qwen14/confirm-native/units.npy'),
            ('source_binding_v1', 'qwen14/discovery-native/qkv_error.npy'),
            ('source_binding_v1', 'qwen14/discovery-native-v1-failure/completed-sample.npz'),
            ('source_relay_v1', 'padding-diagnostic.npz'),
            ('source_relay_v2', 'qwen14/confirm-parameter-ledger.npz'),
            ('common_bench_b1_repair_v2', 'lockbox-native/path_pred_final.npy'),
            ('common_bench_b1_repair_v2', 'lockbox-native/final_states.npy'),
            ('qwen14_mlp_b28_b39_v2', 'B36-A.npy'),
        ]
        for run, path in cases:
            with self.subTest(run=run, path=path):
                self.assertFalse(C.role(run, Path(path))[0])

    def test_unreviewed_storage_and_models_are_out_of_scope(self):
        for path in [C.ROOT/'models', C.ROOT/'tests/glm5_temp', C.BASE.parent]:
            with self.subTest(path=path):
                with self.assertRaises(RuntimeError):
                    C.validated(path)

    def test_target_list_is_exact_and_inside_reviewed_result_root(self):
        rows = list(C.originals())
        self.assertEqual(len(rows), 26)
        self.assertEqual({folder.name for _, folder in rows}, C.NAMES)
        self.assertTrue(all(folder.parent == C.BASE for _, folder in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
