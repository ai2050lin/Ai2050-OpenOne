import unittest
import numpy as np
import hiddenstate_components as H

class Tests(unittest.TestCase):
    def setUp(self):
        self.x=H.design([dict(query=q,record=r,language=l) for q in [-1,1] for r in [-1,1] for l in [-1,1]])
    def test_planted_components_and_four_state(self):
        b=np.zeros((3,2,8,16));b[:,:,0,0]=3;b[:,:,6,2]=2;b[:,:,3,7]=4;b[:,:,7,11]=.7
        y=H.reconstruct(b,self.x);np.testing.assert_allclose(H.factorial(y,self.x),b,atol=1e-12)
        np.testing.assert_allclose(H.reconstruct(b,self.x,H.COLS4),y)
        self.assertGreater(np.mean((H.reconstruct(b,self.x,[0,1,2,3])-y)**2),0)
    def test_causally_invisible_future(self):
        b=np.zeros((2,2,8,10));b[:,:,0]=1;b[:,:,2,0]=2;y=H.reconstruct(b,self.x)
        for factor in [1,3,6,7]:self.assertEqual(float(np.max(np.abs(H.factorial(y,self.x)[:,:,factor]))),0)
    def test_probe_finds_state_and_random_labels_do_not(self):
        b=np.zeros((1,1,8,16));b[:,:,6,2]=2;y=H.reconstruct(b,self.x)
        scores=H.score(y,*H.probe(b,np.ones((1,1,16)),6))
        self.assertEqual(float(H.correct(scores,self.x[:,6])[0,0]),1)
        self.assertEqual(float(H.correct(scores,self.x[:,3])[0,0]),.5)
    def test_source_state_transfer_no_future_input(self):
        b=np.zeros((1,1,8,16));b[:,:,6,2]=2;b[:,:,3,4]=3;delta=b*.2;y=H.reconstruct(b,self.x)
        result=H.state_step(y,delta,self.x[:,6][None,None,:],self.x[:,3][None,None,:])
        np.testing.assert_allclose(result,y+H.reconstruct(delta,self.x,H.COLS4))
    def test_three_layer_onset_excludes_final_normalization(self):
        a=[.5]*37;a[34:]=[1,1,1];self.assertIsNone(H.stable_onset(a))
        a[12:15]=[.9,.95,.91];self.assertEqual(H.stable_onset(a),12)
    def test_bfloat_values_are_exact(self):
        np.testing.assert_array_equal(H.unbits(np.array([0x3f80,0xbf80,0x4000],dtype=np.uint16)),[1,-1,2])

if __name__=='__main__':unittest.main()
