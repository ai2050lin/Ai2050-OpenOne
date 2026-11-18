"""Synthetic engineering cases only; no fabricated scientific outputs."""
import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def mockrows():
    rows=[];items=[]
    for f in range(16):
        for s in [0,1]:
            for e in [0,1]:
                for o in [0,1]:
                    ident=f'{f}/{s}/{e}/{o}';rows.append({'id':ident,'source':s,'entity':e,'order':o,'frame':f%4})
                    items.append({'id':ident,'forced_correct':True,'top1_correct':True})
    return rows,items

class BindingTests(unittest.TestCase):
    def test_gate_complete(self):
        r,x=mockrows();self.assertTrue(D.behavior(r,x)['passed'])
    def test_missing_cell_rejected(self):
        r,x=mockrows();keep=[i for i,v in enumerate(r) if not(v['source']==1 and v['entity']==1 and v['order']==1)]
        self.assertFalse(D.behavior([r[i] for i in keep],[x[i] for i in keep])['passed'])
    def test_bad_frame_not_hidden_by_pool(self):
        r,x=mockrows()
        for i,v in enumerate(r):
            if v['frame']==0:x[i]['forced_correct']=False
        self.assertFalse(D.behavior(r,x)['passed'])
    def test_top1_gate(self):
        r,x=mockrows()
        for v in x:v['top1_correct']=False
        self.assertFalse(D.behavior(r,x)['passed'])
    def test_reordered_or_missing_result(self):
        r,x=mockrows()
        with self.assertRaises(AssertionError):D.behavior(r,x[::-1])
        with self.assertRaises(AssertionError):D.behavior(r,x[:-1])
    def test_no_baseline_no_internal_entry(self):
        import source_binding_run as R
        with tempfile.TemporaryDirectory(dir=D.ROOT/'tests/glm5_temp') as p,patch.object(D,'OUT',Path(p)),patch.object(D,'checked',return_value={}):
            with self.assertRaises(AssertionError):R.capture('qwen14','confirm')
            self.assertEqual(list(Path(p).iterdir()),[])
    def test_source_entity_order_cross(self):
        for s in [0,1]:
            for e in [0,1]:
                for o in [0,1]:
                    r=D.material(3,'test',s,e,o,0,['Baviko','Nenuti'],['cat','dog','fish','bird'])
                    self.assertEqual(r['gold'],r['values'][s*2+e]);self.assertIn(r['gold'],r['content'])
                    a,b=r['records'][s*2+e]['value_span'];self.assertEqual(r['content'][a:b],r['gold'])
                    a,b=r['query_name_span'];self.assertEqual(r['content'][a:b],r['names'][e])
    def test_full_seed_identity(self):
        a=D.rng('family-000000-same').normal(size=8);b=D.rng('family-000001-same').normal(size=8)
        self.assertFalse((a==b).all())

if __name__=='__main__':unittest.main()
