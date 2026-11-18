"""Engineering fixtures only: last-token exclusion and predictive controls."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_update import prefix_features,ridge,predict

class UpdateTests(unittest.TestCase):
    def test_ridge_predicts_planted_linear_rule(self):
        r=np.random.default_rng(3);x=r.normal(size=(512,32));w=r.normal(size=(32,10));test=r.normal(size=(128,32));truth=test@w+2
        pred=predict(test,ridge(x,x@w+2));self.assertLess(np.square(pred-truth).sum()/np.square(truth-truth.mean(0)).sum(),.001)
    def test_unrelated_targets_do_not_predict(self):
        r=np.random.default_rng(7);x=r.normal(size=(512,32));y=r.normal(size=(512,10));tx=r.normal(size=(256,32));ty=r.normal(size=(256,10));pred=predict(tx,ridge(x,y))
        self.assertLess(1-np.square(pred-ty).sum()/np.square(ty-y.mean(0)).sum(),.10)
    def test_prefix_excludes_last_query_and_padding(self):
        with tempfile.TemporaryDirectory(dir=D.ROOT/'tests/glm5_temp') as tmp,patch.object(D,'OUT',Path(tmp)):
            folder=Path(tmp)/'test/discovery-native';folder.mkdir(parents=True)
            (Path(tmp)/'test/discovery-native-manifest.json').write_text(json.dumps({'head_blocks':[9],'length':5}),encoding='utf-8')
            (Path(tmp)/'test-materials.json').write_text(json.dumps({'splits':{'discovery':[{'tokens':[1,2,3,4,5]},{'tokens':[1,2,3]}]}}),encoding='utf-8')
            # Fixed BF16 bit representations:1 and2, then perturb only future/self and padded cells.
            for name in ['key','vproj']:np.save(folder/f'{name}.npy',np.full((2,1,1,5,128),16256,np.uint16))
            first=prefix_features('test','discovery',9)
            for name in ['key','vproj']:
                a=np.load(folder/f'{name}.npy');a[:,:,:,-1]=16384;a[1,:,:,:2]=16384;np.save(folder/f'{name}.npy',a)
            self.assertTrue(np.array_equal(first,prefix_features('test','discovery',9)))
            a=np.load(folder/'key.npy');a[0,0,0,1]=16384;np.save(folder/'key.npy',a)
            self.assertFalse(np.array_equal(first,prefix_features('test','discovery',9)))

if __name__=='__main__':unittest.main()
