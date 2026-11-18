"""Engineering-only fixtures: bounded graph pages preserve latest evidence and filters."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.research_library import Library
from server import research_space_service as S

class GraphPageTests(unittest.TestCase):
    def test_latest_revision_filters_and_pages(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp') as tmp:
            lib=Library(tmp);snap='a'*64
            proof=lib.ingest({'kind':'engineering_fixture','model_snapshot':snap},'fixture')['sha256']
            def card(i,level='observation'):
                return {'id':f'unit{i:03}', 'layer':28,'boundary':'down.input','index':{'space':'mlp','unit':i},
                    'run_id':'fixture-run','method':'isolated engineering fixture','label':f'cue{i}',
                    'evidence':{'level':level,'artifact_sha256':proof},'counterexamples':[]}
            def put(cards):
                return lib.ingest({'schema':'research_graph.v1','kind':'real','model_snapshot':snap,'feature_cards':cards,'graph_edges':[]},'isolated-fixture')['sha256']
            put([card(i) for i in range(25)]);new=put([card(0,'negative')])
            hidden=put([card(25)])
            with lib.db() as db:db.execute('INSERT INTO mainline_publications VALUES(?,0)',(hidden,))
            with patch.object(S,'library',return_value=lib):
                page=S._research_space(12,12,snap,'fixture-run')
                self.assertEqual(page['counts']['objects'],25)
                self.assertEqual(page['objects_page']['total'],25)
                self.assertEqual([r['id'] for r in page['objects']],[f'unit{i:03}' for i in range(12,24)])
                selected=S._research_space(0,12,snap,'fixture-run','unit000','negative')
                self.assertEqual(len(selected['objects']),1)
                self.assertEqual(selected['objects'][0]['sha'],new)
                self.assertEqual(S._research_space(0,12,'b'*64)['objects'],[])
                self.assertEqual(len(S._research_space()['objects']),25)
                self.assertEqual(S._research_space(24,12)['objects'][0]['id'],'unit024')
                compact=S._research_space(24,12,objects_only=True)
                self.assertEqual(compact['objects_page']['total'],25)
                self.assertEqual(compact['objects'][0]['id'],'unit024')
                self.assertNotIn('artifacts',compact)

    def test_filter_does_not_revive_an_obsolete_revision(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp') as tmp:
            lib=Library(tmp);snap='a'*64
            proof=lib.ingest({'kind':'engineering_fixture','model_snapshot':snap},'fixture')['sha256']
            def put(run,label,level):
                return lib.ingest({'schema':'research_graph.v1','kind':'real','model_snapshot':snap,
                    'feature_cards':[{'id':'stable-id','layer':28,'boundary':'down.input','index':{'space':'mlp','unit':0},
                    'run_id':run,'method':'isolated engineering fixture','label':label,
                    'evidence':{'level':level,'artifact_sha256':proof},'counterexamples':[]}],'graph_edges':[]},'fixture')['sha256']
            old=put('old-run','old-label','observation');new=put('new-run','new-label','negative')
            with lib.db() as db:
                db.execute('UPDATE artifacts SET created=1 WHERE sha=?',(old,))
                db.execute('UPDATE artifacts SET created=2 WHERE sha=?',(new,))
            with patch.object(S,'library',return_value=lib):
                self.assertEqual(S._research_space(run_id='old-run',objects_only=True)['objects'],[])
                self.assertEqual(S._research_space(query='old-label',objects_only=True)['objects'],[])
                self.assertEqual(S._research_space(level='observation',objects_only=True)['objects'],[])
                self.assertEqual(S._research_space(run_id='new-run',objects_only=True)['objects'][0]['sha'],new)
                with lib.db() as db:db.execute('INSERT INTO mainline_publications VALUES(?,0)',(new,))
                self.assertEqual(S._research_space(run_id='old-run',objects_only=True)['objects'][0]['sha'],old)

if __name__=='__main__':unittest.main()
