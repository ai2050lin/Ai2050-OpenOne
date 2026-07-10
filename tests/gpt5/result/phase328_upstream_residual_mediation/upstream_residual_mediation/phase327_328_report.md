# Phase327-328 Natural Retrieval Physical Path Report

## Scope and corrected judgment

The Phase326 assessment was directionally correct: the frozen attention-head and MLP-product groups are useful late distributed carrier candidates, but they are not a complete retrieval path and do not establish single-neuron causality.

Phase327 tested the missing chain:

```text
natural object -> residual identity -> frozen Phase326 carrier set -> target phrase -> natural generation
```

Phase328 then tested one registered upstream intervention:

```text
category query residual -> downstream frozen carrier state -> full-vocabulary target rank
```

The result is a mixed calibration, not closure. Natural-state transplant effects replicate more broadly than natural identity, position necessity, or complete generation. No full chain and no causal path edge replicated across models.

## Frozen denominator

Phase327 used only color, category, and habitat, the three knowledge mechanisms with Phase326 cross-model set-level confirmation.

```text
54 new objects;
18 objects per mechanism;
2 new causal-order cloze templates;
108 prompts per model;
324 prompt-model cases;
5 natural object variants per prompt;
1620 natural-variant rows;
2916 position-intervention rows;
1944 natural-state-transplant rows;
648 generation rows.
```

Automatic validation found:

```text
target leakage = 0;
same-mechanism object overlap with Phase326 = 0;
duplicate case IDs = 0;
source-after-query order errors = 0.
```

The frozen Phase326 carrier members were never reselected on Phase327 or Phase328 data.

## Basic measures

Natural carrier identity specificity:

$$
C = \overline{\cos(z_{correct},z_{same\ target})}
  - \overline{\cos(z_{correct},z_{wrong\ target})}
$$

Residual identity specificity at layer and role:

$$
R_{l,r} = \overline{\operatorname{RMS}(h^{correct}_{l,r}-h^{wrong}_{l,r})}
 - \overline{\operatorname{RMS}(h^{correct}_{l,r}-h^{same}_{l,r})}
$$

Phrase necessity drop and matched-control specificity:

$$
D_c = \log P(y\mid x)_{base} - \log P(y\mid x)_{c}
$$

$$
S_{pos} = D_{joint} - \max(D_{random},D_{wrong\ layer})
$$

Natural donor gain and specificity:

$$
G_d = \log P(y\mid x,do(z\leftarrow z_d)) - \log P(y\mid x)
$$

$$
S_{donor} = \min(G_{correct},G_{same\ target})
 - \max(G_{wrong},G_{unrelated})
$$

These are operational measurements, not a unified language-mechanism formula.

## Phase327 cross-model results

| Mechanism | Natural identity | Position necessity | Natural transplant | Complete generation | Full chain |
|---|---|---|---|---|---|
| Color | none | Qwen3, GLM4 | Qwen3, GLM4 | GLM4 only | none |
| Category | Qwen3, GLM4 | Qwen3 only | all three | none | none |
| Habitat | Qwen3 only | DS7B only | all three | none | none |

Strict cross-model counts:

```text
natural identity: 1/3 mechanisms;
position necessity: 1/3 mechanisms;
natural-state transplant: 3/3 mechanisms;
complete generation: 0/3 mechanisms;
full natural chain: 0/3 mechanisms;
single-unit causality: 0.
```

The strongest physical distribution result is category identity. Within the registered first 70% of layers, the largest query-identity specificity occurred at:

```text
Qwen3: L24/35;
GLM4: L27/39;
DS7B: L19/27.
```

All three are at or near the search boundary. They are boundary maxima, not demonstrated local peaks. Category natural identity passed only Qwen3 and GLM4 because DS7B baseline candidate accuracy was 0.528, below the registered 0.60 gate.

The frozen-set position effects were dominated by the last position. Source and query effects were usually near zero. This supports a late readout-state interpretation and does not prove source-to-query propagation through the Phase326 members.

## Phase328 upstream mediation

Phase328 used only category. Layer selection used the first 12 Phase327 objects and query-role residual differences. Validation used the last 6 objects and both templates, for 12 independent validation prompts per model.

| Model | Frozen query layer | Correct phrase gain | Correct carrier-similarity gain | Global rank gain | Mediation pass | Top-1 unlock |
|---|---:|---:|---:|---:|---|---|
| Qwen3 | 24 | +0.213 | -0.0055 | +93.25 | no | no |
| GLM4 | 27 | +0.159 | +0.0293 | +425.17 | yes | no |
| DS7B | 19 | +0.351 | -0.0035 | +1687.83 | no | no |

Qwen3 failed because downstream carrier similarity did not move in the required direction and the wrong-layer control was stronger. DS7B failed because wrong and unrelated donors produced comparable phrase gains and carrier specificity was negative. GLM4 passed the registered residual-state mediation criteria, but the target remained outside top-1 in every validation case.

Therefore:

```text
single-model residual mediation candidate = GLM4 category;
cross-model residual mediation = no;
natural top-1 unlock = 0/3 models;
causal path edges = 0;
L5 promotion = 0.
```

## Hard limitations

1. Phase326 components were selected with direct unembedding attribution, which is intrinsically biased toward late readout layers.
2. The Phase327 set interventions are dominated by the last role; they do not demonstrate source-to-query propagation.
3. Pooled residual transplantation broadcasts one vector over a role span. It can change broad state geometry and is not a native tokenwise computation.
4. Large target-rank gains do not imply answer generation. All Phase328 target top-1 rates remained zero.
5. Carrier cosine is a coarse state measure. It can miss sign, subcomponent, and tokenwise mediation or respond to broad damage.
6. Token-length controls are not training-frequency controls. True corpus frequency was not available.
7. The three local models are small and architecturally different. Their internal mechanisms may differ materially from larger language models.
8. No single neuron was intervened on in Phase327-328. Component groups and pooled residual states must not be displayed as causal neurons.

## Atlas and client state

The physical atlas preserves the previous inventory:

```text
mapped families: 2/9;
unique physical candidates: 1121;
single-neuron candidates: 833;
component-set members: 288;
expanded-confirmed members: 72;
new synthetic neurons: 0;
single-unit causal nodes: 0.
```

It adds:

```text
9 noncausal natural-retrieval paths;
3 noncausal upstream residual-mediation edges;
0 strict natural chains;
0 cross-model causal path edges.
```

The 3D client displays these statuses on the existing model geometry. It does not reshape the DNN or promote mechanism-level effects to individual neurons.

## Progress and next registered stage

Measured coverage remains:

```text
physical candidate family coverage: 2/9 = 22.2%;
strict natural-chain coverage among the three tested knowledge mechanisms: 0/3;
cross-model causal path edge coverage: 0;
single-neuron causal closure: 0.
```

Phase329 should be a new denominator, not another Phase328 patch:

1. Freeze full-vocabulary blockers from top-50 logits and separate semantic competitors, continuation tokens, punctuation, and protocol tokens.
2. Replace pooled broadcast transplantation with tokenwise residual transplantation and norm-matched residual controls.
3. Require an upstream intervention to change component-by-component downstream carrier identity, target rank, and natural generation together.
4. Localize upstream modules only after residual-state mediation replicates across at least two models.
5. Start single-neuron CUDA intervention only after a cross-model path edge exists.

The current stage is complete. Phase329 changes the registered object of study from late carrier sets to full-vocabulary competition and tokenwise upstream computation.
