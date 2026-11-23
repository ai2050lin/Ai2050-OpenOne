"""Role and deletion-boundary checks for continued storage retirement."""
import unittest
from pathlib import Path

import other_test_storage_cleanup_20261118 as C


class OtherCleanupTests(unittest.TestCase):
    def test_mixed_npz_preserves_results_and_question_identity(self):
        cases=[('cross_style','M.npy'),('cross_style','tok.npy'),('cross_style','item.npy'),
               ('cross_entity_mode','conf_zh_color.npy'),('cross_entity_mode','disc_en_DLA.npy'),
               ('mode','conf_zh_gate.npy'),('paradigm','ids.npy'),('paradigm','logits.npy'),
               ('paradigm','top.npy'),('paradigm','head_effect.npy'),('lookup','mass20.npy'),
               ('lookup','write30.npy'),('family','ids.npy')]
        for kind,name in cases:
            with self.subTest(kind=kind,name=name):self.assertFalse(C.raw_member(kind,name))

    def test_only_known_dense_members_are_retired(self):
        cases=[('cross_style','Hg.npy'),('cross_style','Hs.npy'),('cross_entity_mode','conf_en_codes.npy'),
               ('cross_entity_mode','disc_zh_B.npy'),('mode','disc_zh_H.npy'),('mode','conf_en_bare.npy'),
               ('cross_final','disc_zh_final.npy'),('paradigm','field.npy'),('paradigm','G.npy'),
               ('lookup','h00.npy'),('lookup','h40.npy'),('lookup','af20.npy'),('lookup','zf22.npy'),
               ('family','X.npy'),('family','Z.npy'),('family','V.npy')]
        for kind,name in cases:
            with self.subTest(kind=kind,name=name):self.assertTrue(C.raw_member(kind,name))

    def test_weights_clients_service_and_external_paths_are_protected(self):
        for path in [C.ROOT/'models',C.NATIVE/'_weights',C.NATIVE/'_agent_home',
                     C.NATIVE/'client-cache',C.NATIVE/'visualization-research',Path('C:/Windows')]:
            with self.subTest(path=path):
                with self.assertRaises(RuntimeError):C.safe(path)

    def test_junctions_are_rejected(self):
        junction=C.ROOT/'tests/glm5/result/qwen14_mlp_atlas_v1'
        if junction.exists():
            with self.assertRaises(RuntimeError):C.safe(junction)

    def test_refreshed_real_windows_identity_is_available(self):
        p=C.ROOT/'tests/glm5/test_other_test_storage_cleanup_20261118.py'
        record=C.metadata(p)
        self.assertGreater(record['inode'],0)
        self.assertGreater(record['nlink'],0)


if __name__=='__main__':unittest.main(verbosity=2)
