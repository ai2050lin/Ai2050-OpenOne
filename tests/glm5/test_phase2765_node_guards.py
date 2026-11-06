"""Selection failures should be actionable before querying devices or model directories."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'deploy'))
import node_service as S


class TaskSelection(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.svc=S.NodeService(Path(self.tmp.name))
        self.svc.creds=lambda:{'server':'http://unused.invalid','node_token':'test-fixture'}
        self.svc.probes['gpus']=Mock(side_effect=AssertionError('GPU should not be queried'))
    def tearDown(self):self.svc.shutdown();self.tmp.cleanup()
    def test_empty_selection_stops_before_platform_and_gpu(self):
        self.svc.cfg['task_kinds']=[];self.svc.platform_config=Mock(side_effect=AssertionError('No server lookup needed'))
        with self.assertRaisesRegex(S.Waiting,'尚未选择任务类型'):self.svc.round()
        self.svc.probes['gpus'].assert_not_called();self.svc.platform_config.assert_not_called()
    def test_unavailable_kind_is_distinct_from_missing_weights(self):
        self.svc.cfg['task_kinds']=['family_test'];self.svc.platform_config=lambda:{'task_kinds':['census_shard']}
        with self.assertRaisesRegex(S.Waiting,'暂未开放所选任务类型'):self.svc.round()
        self.svc.probes['gpus'].assert_not_called()


if __name__=='__main__':unittest.main()
