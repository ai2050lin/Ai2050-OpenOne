import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = open('D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/p3124_probe_out.txt',
           'w', encoding='utf-8')


def log(s):
    OUT.write(s + '\n')
    OUT.flush()


MDIR = 'D:/AI2050/Ai2050-OpenOne/models/hf/glm4-9b-chat-hf'
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NL = len(model.model.layers)
norm = model.model.norm
WU = model.lm_head.weight.detach()
log('arch=%s NL=%d' % (model.__class__.__name__, NL))

yes = int(tok(' yes', add_special_tokens=False)['input_ids'][0])
no = int(tok(' no', add_special_tokens=False)['input_ids'][0])
w_dn = (WU[yes] - WU[no]).float().cpu().numpy()

ids = tok('The apple is red. The banana is yellow. '
          'The sky is', return_tensors='pt').input_ids.cuda()
with torch.inference_mode():
    out = model(ids, output_hidden_states=True,
                use_cache=False)
    hs = out.hidden_states
    lg = out.logits[0].float().cpu().numpy()
    base = model.model(ids, use_cache=False)
    lhb = base.last_hidden_state[0].float().cpu().numpy()
    emb = model.model.embed_tokens(ids)[0].float().cpu().numpy()
    h_last = hs[-1][0].float().cpu().numpy()
    h_prev = hs[-2][0].float().cpu().numpy()
    h_emb = hs[0][0].float().cpu().numpy()
    h_renorm = norm(hs[-1][0]).float().cpu().numpy()

log('len(hs)=%d' % len(hs))
log('max|hs[-1]-lhb|=%.6g'
    % float(np.abs(h_last - lhb).max()))
log('max|hs[-1]-hs[-2]|=%.6g'
    % float(np.abs(h_last - h_prev).max()))
log('max|hs[0]-embed|=%.6g'
    % float(np.abs(h_emb - emb).max()))
log('max|norm(hs[-1])-hs[-1]|=%.6g'
    % float(np.abs(h_renorm - h_last).max()))

lm_m = lg[:, yes] - lg[:, no]
m_last = h_last @ w_dn
m_renorm = h_renorm @ w_dn
m_lhb = lhb @ w_dn


def rep(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = 0.0
    log('%s: r=%.6f maxdiff=%.6g'
        % (name, r, float(np.abs(a - b).max())))


rep('hs[-1]@w_dn (nonorm) vs lm', m_last, lm_m)
rep('renorm(hs[-1])@w_dn  vs lm', m_renorm, lm_m)
rep('lhb@w_dn             vs lm', m_lhb, lm_m)
log('DONE')
OUT.close()
