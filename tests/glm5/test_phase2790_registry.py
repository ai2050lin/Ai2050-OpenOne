"""Software fixtures only; no generated labels enter a real research campaign."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry


class RegistryTest(unittest.TestCase):
    def setUp(self):
        base=ROOT/'tests/glm5_temp';base.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.lib=Library(Path(self.tmp.name));self.reg=Registry(self.lib)
        self.ds={'schema':'dataset_version.v1','kind':'real','id':'fixture-data:v1','dataset_id':'software-fixture',
            'version':'1','source':'software-fixture only','license':'fixture','label_status':'software fixture',
            'role':'controlled_mechanism','content_sha256':'a'*64,'source_groups':['document-a'],'sample_count':2}
        self.reg.register(self.ds)
        self.proof=self.lib.ingest({'schema':'software_fixture','model_snapshot':'b'*64},'software-fixture')['sha256']
    def tearDown(self):
        assert Path(self.tmp.name).resolve().is_relative_to((ROOT/'tests/glm5_temp').resolve())
        self.tmp.cleanup()
    def claim(self,metric='candidate_log_odds'):
        return {'schema':'research_claim.v1','kind':'real','id':'fixture-claim:'+metric,'canonical_feature_id':'operation:fixture',
            'model_snapshot':'b'*64,'artifact_sha256':self.proof,'external_pattern_id':'language:style',
            'outcome_metric':metric,'scope':'software fixture scope','status':'inconclusive','dataset_versions':[self.ds['id']]}
    def campaign(self):
        return {'schema':'blind_review_campaign.v1','kind':'real','id':'fixture-review','rubric_version':'fixture1',
            'required_human_reviews':2,'purpose':'software fixture','items':[{'key':'private-base-acad','text':'软件测试文本。',
            'facts':{'item':'software-fixture'},'expected_style':'formal','condition':'base'}]}
    def label(self):return {'style':'mixed','meaning':'uncertain','naturalness':3,'human_attestation':True,'note':'software fixture'}
    def test_distinct_metrics_coexist_without_evidence_promotion(self):
        a=self.claim();a['status']='supported_in_scope';self.reg.register(a)
        self.reg.register(self.claim('free_generation'))
        cat=self.reg.catalog();self.assertEqual(len(cat['claims']),2)
        self.assertEqual({x['body']['status'] for x in cat['claims']},{'supported_in_scope','inconclusive'})
    def test_cannot_rebind_same_dataset_or_claim(self):
        bad={**self.ds,'content_sha256':'c'*64}
        with self.assertRaisesRegex(ValueError,'identity collision'):self.reg.register(bad)
        self.reg.register(self.claim());bad={**self.claim(),'outcome_metric':'another'}
        with self.assertRaisesRegex(ValueError,'identity collision'):self.reg.register(bad)
    def test_different_model_proof_rejected(self):
        with self.assertRaisesRegex(ValueError,'proof model'):self.reg.register({**self.claim(),'model_snapshot':'d'*64})
    def test_unregistered_data_and_duplicate_source_groups_rejected(self):
        with self.assertRaises(ValueError):self.reg.register({**self.claim(),'dataset_versions':['missing']})
        with self.assertRaises(ValueError):self.reg.register({**self.ds,'source_groups':['x','x']})
    def test_blind_assignment_hides_condition_keys_and_prior_labels(self):
        self.reg.create_review(self.campaign());item=self.reg.next_review('fixture-review','reviewer-one')
        self.assertNotIn('condition',str(item));self.assertNotIn('private-base-acad',str(item))
        self.assertNotIn('expected_style',str(item));self.assertEqual(item['facts'],{'item':'software-fixture'})
        self.reg.submit_review('fixture-review',item['item_id'],'reviewer-one',self.label())
        second=self.reg.next_review('fixture-review','reviewer-two')
        self.assertNotIn('mixed',str(second));self.assertNotIn('reviewer-one',str(second))
        with self.lib.db() as db:
            public=[self.lib.artifact(x['sha']) for x in db.execute('SELECT sha FROM artifacts')]
        self.assertFalse(any(x.get('schema')=='blind_human_label.v1' for x in public))
    def test_two_accounts_no_repeat_and_disagreement_remains_pending(self):
        self.reg.create_review(self.campaign())
        a=self.reg.next_review('fixture-review','one');b=self.reg.next_review('fixture-review','two')
        self.assertIsNone(self.reg.next_review('fixture-review','three'))
        self.reg.submit_review('fixture-review',a['item_id'],'one',self.label())
        self.assertIsNone(self.reg.next_review('fixture-review','one'))
        self.reg.submit_review('fixture-review',b['item_id'],'two',{**self.label(),'style':'formal'})
        summary=self.reg.review_summary()[0]
        self.assertEqual(summary['double_reviewed'],1);self.assertEqual(summary['disagreements'],1)
        self.assertEqual(summary['status'],'awaiting_adjudication');self.assertFalse(summary['evidence_promoted'])
    def test_cannot_submit_ai_as_human_or_overwrite_label(self):
        self.reg.create_review(self.campaign());a=self.reg.next_review('fixture-review','one')
        with self.assertRaises(ValueError):self.reg.submit_review('fixture-review',a['item_id'],'one',{**self.label(),'human_attestation':False})
        with self.assertRaises(ValueError):self.reg.submit_review('fixture-review',a['item_id'],'outsider',self.label())
        first=self.reg.submit_review('fixture-review',a['item_id'],'one',self.label())
        replay=self.reg.submit_review('fixture-review',a['item_id'],'one',self.label())
        self.assertEqual(first['sha256'],replay['sha256']);self.assertTrue(replay['replay'])
        with self.assertRaises(ValueError):self.reg.submit_review('fixture-review',a['item_id'],'one',{**self.label(),'meaning':'changed'})
    def test_campaign_immutable_and_excluded_designer(self):
        c=self.campaign();c['excluded_reviewer_accounts']=['designer'];self.reg.create_review(c)
        with self.assertRaises(ValueError):self.reg.next_review('fixture-review','designer')
        c['purpose']='changed'
        with self.assertRaises(ValueError):self.reg.create_review(c)

    def ai_campaign(self):
        blind=self.lib.ingest({'schema':'external_ai_blind_batch.v1','records':[{'id':'opaque','text':'软件测试。','facts':{}}]},'fixture-blind')['sha256']
        c={'schema':'external_ai_review_campaign.v1','kind':'real','id':'fixture-ai','rater_models':['model-one','model-two'],
            'input_sha256':'e'*64,'blind_artifact_sha256':blind,'n_items':1}
        self.reg.create_ai_review(c)
        return c

    def ai_label(self,rater='model-one'):
        return {'schema':'external_ai_ratings.v1','kind':'real','rater_model':rater,'input_sha256':'e'*64,
            'ratings':[{'id':'opaque','style':'formal','meaning':'preserved','naturalness':4,'reason':'软件测试标签。'}]}

    def test_external_ai_complete_distinct_raters_preserve_disagreement(self):
        self.ai_campaign();a=self.reg.submit_ai_review('fixture-ai',self.ai_label())
        b=self.ai_label('model-two');b['ratings'][0]['meaning']='changed';self.reg.submit_ai_review('fixture-ai',b)
        summary=self.reg.review_summary()[0]
        self.assertEqual(summary['completed_raters'],2);self.assertEqual(summary['disagreements'],1)
        self.assertFalse(summary['evidence_promoted']);self.assertEqual(len(summary['rating_artifacts']),2)
        self.assertFalse(a['human_labels'])

    def test_external_ai_hash_rater_ids_and_immutability(self):
        self.ai_campaign()
        for change in [{'rater_model':'unregistered'},{'input_sha256':'f'*64},{'ratings':[]},
            {'ratings':self.ai_label()['ratings']*2}]:
            with self.assertRaises(ValueError):self.reg.submit_ai_review('fixture-ai',{**self.ai_label(),**change})
        first=self.reg.submit_ai_review('fixture-ai',self.ai_label())
        self.assertEqual(first['sha256'],self.reg.submit_ai_review('fixture-ai',self.ai_label())['sha256'])
        b=self.ai_label();b['ratings'][0]['style']='spoken'
        with self.assertRaises(ValueError):self.reg.submit_ai_review('fixture-ai',b)


if __name__=='__main__':unittest.main()
