# -*- coding: utf-8 -*-
"""Phase 2886: hourglass validation - CKA curve + language probe.

Background: 2878 (qwen tied) and 2885 (DS7B untied) established the
model-general fact that NO unified cross-lingual axis exists at the
endpoint embedding/unembed rows.  The surviving hypothesis (attachment
"hourglass", now primary) says language identity is processed
mid-stream: shallow layers language-specific, mid layers
language-agnostic semantics, deep layers re-dress in target language.

Predictions (attachment weapon 1, made gateable here):
  P1 CKA inverted-U: cross-language representation similarity (linear
     CKA between en and fr sentence state sets) peaks mid-network.
  P2 language probe U-shape: a training-free language-identity probe
     (LOO nearest-centroid, en vs fr) is accurate at both ends and
     dips mid-network.

Protocol (frozen before any forward):
  S1  40 frozen en/fr parallel sentence pairs (script constant,
      simple declaratives sharing the 2878/2885 concept pools);
      qwen3-4b only (primary model).
  S2  states: last-token hidden state per sentence per layer
      (hidden_states tuple, li = 0 embed .. L); mean-pool variant
      computed descriptively, not gated.
  S3  CKA: linear CKA between the 40 x d en matrix and 40 x d fr
      matrix per layer (centered, Frobenius form).
  S4  probe: LOO nearest-centroid (cosine) language classification
      per layer on the 80 x d stacked last-token states.
  H1  hourglass CKA iff argmax_li CKA lies in mid third (12 <= li
      <= 24 for L=36) AND CKA_peak - max(mean CKA li 0-5, mean CKA
      li 31-36) >= 0.10.
  H2  probe dip iff mean(probe acc | mid third) <= mean(probe acc |
      first 6 or last 6) - 0.15.
  H3  sanity (descriptive): split-half within-en CKA per layer
      (ceiling reference); no gate.
  SEED=2886.  Output: phase2886/hourglass_cka/{execution.json,
  result.json, hourglass_cka.npz}
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2886', 'hourglass_cka')
SEED = 2886

PAIRS = [
    ('The cat is black.', 'Le chat est noir.'),
    ('The dog is big.', 'Le chien est grand.'),
    ('The house is small.', 'La maison est petite.'),
    ('The book is new.', 'Le livre est nouveau.'),
    ('The night is dark.', 'La nuit est sombre.'),
    ('The sun is bright.', 'Le soleil est brillant.'),
    ('The milk is cold.', 'Le lait est froid.'),
    ('The water is clear.', "L'eau est claire."),
    ('The tree is tall.', "L'arbre est grand."),
    ('The moon is white.', 'La lune est blanche.'),
    ('The sea is deep.', 'La mer est profonde.'),
    ('The sky is blue.', 'Le ciel est bleu.'),
    ('The flower is red.', 'La fleur est rouge.'),
    ('The city is loud.', 'La ville est bruyante.'),
    ('The war was long.', 'La guerre était longue.'),
    ('The king is old.', 'Le roi est vieux.'),
    ('The woman is kind.', 'La femme est gentille.'),
    ('The man is tall.', "L'homme est grand."),
    ('The mother is here.', 'La mère est ici.'),
    ('The father is strong.', 'Le père est fort.'),
    ('The bird sings.', "L'oiseau chante."),
    ('The fish swims.', 'Le poisson nage.'),
    ('The bread is warm.', 'Le pain est chaud.'),
    ('The wine is good.', 'Le vin est bon.'),
    ('The star shines.', "L'étoile brille."),
    ('The fire is hot.', 'Le feu est chaud.'),
    ('The mountain is high.', 'La montagne est haute.'),
    ('The snow is cold.', 'La neige est froide.'),
    ('The wolf howls.', 'Le loup hurle.'),
    ('The summer is short.', "L'été est court."),
    ('The table is round.', 'La table est ronde.'),
    ('The door is open.', 'La porte est ouverte.'),
    ('The horse runs.', 'Le cheval court.'),
    ('The cheese is French.', 'Le fromage est français.'),
    ('The egg is fresh.', "L'œuf est frais."),
    ('The bed is soft.', 'Le lit est doux.'),
    ('The light is warm.', 'La lumière est chaude.'),
    ('The earth is round.', 'La terre est ronde.'),
    ('The moon rises.', 'La lune se lève.'),
    ('The child sleeps.', "L'enfant dort."),
]

PREREG = {
    'S1': '40 frozen en/fr parallel pairs (script constant), qwen3-4b '
          'primary model only',
    'S2': 'last-token hidden state per sentence per layer; mean-pool '
          'descriptive only',
    'S3': 'linear CKA (centered Frobenius) en vs fr per layer',
    'S4': 'LOO nearest-centroid (cosine) language probe per layer',
    'H1': 'hourglass CKA iff argmax CKA li in [12,24] AND CKA_peak - '
          'max(mean CKA li0-5, mean CKA li31-36) >= 0.10',
    'H2': 'probe dip iff mean(probe|mid third) <= mean(probe|ends) - '
          '0.15 (ends = first 6 + last 6 li)',
    'H3': 'split-half within-en CKA per layer, descriptive ceiling',
    'verdict': 'hourglass_validated iff H1 and H2',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = float(np.linalg.norm(Yc.T @ Xc, 'fro')) ** 2
    den = (float(np.linalg.norm(Xc.T @ Xc, 'fro'))
           * float(np.linalg.norm(Yc.T @ Yc, 'fro')))
    return num / max(den, 1e-30)


def loo_nc_acc(X, lab):
    n = len(lab)
    ok = 0
    for i in range(n):
        m = np.arange(n) != i
        c0 = X[m & (lab == 0)].mean(0)
        c1 = X[m & (lab == 1)].mean(0)
        c0 = c0 / max(float(np.linalg.norm(c0)), 1e-30)
        c1 = c1 / max(float(np.linalg.norm(c1)), 1e-30)
        x = X[i] / max(float(np.linalg.norm(X[i])), 1e-30)
        ok += int(float(x @ c1) > float(x @ c0)) if lab[i] == 1 \
            else int(float(x @ c0) > float(x @ c1))
    return ok / n


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2886, 'name': 'hourglass_cka',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'prereg': PREREG, 'seed': SEED,
                   'n_pairs': len(PAIRS),
                   'model': 'qwen3-4b'},
                  f, indent=2, ensure_ascii=False)
    log = lambda m: print(m, flush=True)
    log('execution.json frozen (%d pairs)' % len(PAIRS))

    import torch
    from transformers import AutoTokenizer
    from phase2662_symmetric_mapping_contract import load_native

    MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True, use_fast=True)
    model, _ = load_native('qwen4')
    model.eval()
    NL = len(model.model.layers)

    states_last = []   # (2*n_sent, d) filled per layer later
    states_mean = []
    sents = []
    for en, fr in PAIRS:
        sents.append(en)
        sents.append(fr)
    lab = np.array([i % 2 for i in range(2 * len(PAIRS))])  # 0=en 1=fr

    with torch.no_grad():
        for si, s in enumerate(sents):
            ids = tok(s, return_tensors='pt')['input_ids'].to('cuda')
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states          # tuple len NL+1
            last = torch.stack([h[0, -1] for h in hs])
            mean = torch.stack([h[0].mean(0) for h in hs])
            states_last.append(last.float().cpu().numpy())
            states_mean.append(mean.float().cpu().numpy())
            if (si + 1) % 20 == 0:
                log('sentences [%d/%d]' % (si + 1, len(sents)))

    S_last = np.stack(states_last)     # (80, NL+1, d)
    S_mean = np.stack(states_mean)
    n_li = NL + 1

    cka = np.zeros(n_li)
    cka_mean_pool = np.zeros(n_li)
    cka_split = np.zeros(n_li)
    probe = np.zeros(n_li)
    rng = np.random.default_rng(SEED)
    en_idx = np.where(lab == 0)[0]
    half_a = en_idx[::2]
    half_b = en_idx[1::2]

    for li in range(n_li):
        Xe = S_last[lab == 0, li, :]
        Xf = S_last[lab == 1, li, :]
        cka[li] = linear_cka(Xe, Xf)
        Ye = S_mean[lab == 0, li, :]
        Yf = S_mean[lab == 1, li, :]
        cka_mean_pool[li] = linear_cka(Ye, Yf)
        cka_split[li] = linear_cka(S_last[half_a, li, :],
                                   S_last[half_b, li, :])
        probe[li] = loo_nc_acc(S_last[:, li, :], lab)

    mid = list(range(12, 25))
    ends = list(range(0, 6)) + list(range(31, 37))
    peak_li = int(np.argmax(cka))
    peak_val = float(cka[peak_li])
    end_max = max(float(cka[0:6].mean()), float(cka[31:37].mean()))
    h1 = bool(12 <= peak_li <= 24 and peak_val - end_max >= 0.10)
    probe_mid = float(probe[mid].mean())
    probe_ends = float(probe[ends].mean())
    h2 = bool(probe_mid <= probe_ends - 0.15)
    verdict = bool(h1 and h2)
    log('peak li=%d cka=%.4f end_max=%.4f H1=%s'
        % (peak_li, peak_val, end_max, h1))
    log('probe mid=%.4f ends=%.4f H2=%s' % (probe_mid, probe_ends, h2))

    res = {
        'phase': 2886,
        'model': 'qwen3-4b',
        'prereg': PREREG,
        'n_pairs': len(PAIRS),
        'H1': {'peak_li': peak_li, 'peak_cka': round(peak_val, 4),
               'end_max_cka': round(end_max, 4), 'verdict': h1},
        'H2': {'probe_mid': round(probe_mid, 4),
               'probe_ends': round(probe_ends, 4), 'verdict': h2},
        'verdict': verdict,
        'cka_per_layer': [round(float(x), 4) for x in cka],
        'cka_mean_pool': [round(float(x), 4) for x in cka_mean_pool],
        'cka_split_half_en': [round(float(x), 4) for x in cka_split],
        'probe_per_layer': [round(float(x), 4) for x in probe],
        'final_verdict': 'hourglass_validated=%s (H1=%s H2=%s; peak '
                         'li=%d cka=%.4f; probe mid %.3f vs ends %.3f)'
                         % (verdict, h1, h2, peak_li, peak_val,
                            probe_mid, probe_ends),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'hourglass_cka.npz'),
        S_last=S_last.astype(np.float32),
        S_mean=S_mean.astype(np.float32),
        labels=lab,
        sentences=np.array(sents, dtype=object),
        cka=cka, cka_mean_pool=cka_mean_pool, cka_split=cka_split,
        probe=probe)
    log('==== VERDICT: %s ====' % res['final_verdict'])
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
