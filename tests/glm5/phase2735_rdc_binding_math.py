"""Independent CPU float64 calculus checks for the automatic diagnostic."""
from rdc_binding_common import *

def main():
    import torch
    from rdc_law_native import tail_forward,dense_gradient,factor_gram
    from rdc_binding_gradients import error_factors,projection_coefficients
    from phase2735_rdc_binding_long_capture import score
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(2735);dtype=torch.float64
    n,d,m,v=4,5,7,11
    w={k:torch.randn(shape,dtype=dtype,requires_grad=True) for k,shape in [('g',(m,d)),('u',(m,d)),('d',(d,m))]}
    norm=torch.rand(d,dtype=dtype)+.5;head=torch.randn(v,d,dtype=dtype);x=torch.randn(n,d,dtype=dtype);r=torch.randn(n,d,dtype=dtype)
    target=torch.tensor([1,3,5,3]);digits=[1,3,5];eps=.017
    class Fixture:
        pass
    fixture=Fixture();fixture.w=w;fixture.norm=norm;fixture.head=head;fixture.eps=eps
    z=tail_forward(x,r,w,norm,head,eps,target);p=z['probabilities'];pc=torch.zeros_like(p);pc[:,digits]=z['logprobs'][:,digits].softmax(-1)
    ec=pc.clone();ec[torch.arange(n),target]-=1;ef=p-pc
    ff={part:error_factors(fixture,z,x,error) for part,error in [('content',ec),('format',ef),('full',ec+ef)]}
    logmass=torch.logsumexp(z['logprobs'][:,digits],1);loss={'full':z['loss'],'content':z['loss']+logmass,'format':-logmass};reports={}
    for part in loss:
        grads=torch.autograd.grad(loss[part].mean(),list(w.values()),retain_graph=True);expected=dense_gradient(ff[part])
        reports[part]={k:float((a-expected[k]).abs().max().detach()) for k,a in zip(w,grads)}
        assert max(reports[part].values())<1e-11,reports
    g={k:factor_gram(f)['total'] for k,f in ff.items()};cross=factor_gram(ff['content'],ff['format'])['total']
    gram_error=float((g['full']-g['content']-g['format']-cross-cross.T).abs().max().detach())
    assert gram_error<1e-9
    basis=torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]],dtype=dtype);target_vector=torch.tensor([2.,3.,4.],dtype=dtype)
    coef,rank=projection_coefficients(basis@basis.T,basis@target_vector);projection=coef@basis
    orth=float((basis@(target_vector-projection)).abs().max());assert orth<1e-11 and rank['numerical_rank']==2
    C=w['g'][:,:,None]*w['u'][:,None,:];bilinear=torch.einsum('ni,kij,nj->nk',x,C,x)
    a=(x@w['g'].T).sigmoid()*bilinear;native=torch.nn.functional.silu(x@w['g'].T)*(x@w['u'].T)
    bilateral_error=float((a-native).abs().max().detach());assert bilateral_error<1e-11
    parser_cases=[('4','4',True,'4'),('Answer: **4**','4',True,'4'),('The final answer is 3.','4',False,'3'),
      ('item_8_0 = 4\nLet us calculate','4',None,None),('\\boxed{4}','4',True,'4')]
    parser=[]
    for text,target_text,correct,parsed in parser_cases:
        record=score(text,target_text,[9],{10},8);assert record['conservative_final_digit']==parsed and record['conservative_final_correct']==correct
        parser.append({'text':text,'result':record})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,'dtype':'CPUfloat64',
      'gradient_absolute_max_errors':reports,'full_gradient_gram_identity_max_error':gram_error,
      'rank_deficient_span':rank,'orthogonality_max_error':orth,'bilinear_identity_max_error':bilateral_error,
      'conservative_parser_cases':parser,'seconds':time.monotonic()-start,
      'scope':'Independent small synthetic mathematical fixtures only; not additional native language samples or empirical semantic evidence.'}
    save(BASE/'verification/content_format_math.json',result);ledger('content_format_cpu_math',result['seconds']);print('CONTENT_FORMAT_CPU_MATH_PASS',reports,flush=True)

if __name__=='__main__':main()
