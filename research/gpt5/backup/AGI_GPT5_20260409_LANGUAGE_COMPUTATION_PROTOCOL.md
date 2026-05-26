# AGI GPT5 2026-04-09 Language Computation Protocol

## 1. Scope

This note finalizes two pending tasks:

1. Rebuild a minimal causal encoding structure for fruit concepts such as apple, banana, pear, and orange.
2. Extend the theory so that pronouns, prepositions, adverbs, and logical reasoning all live inside one unified state-update law.

The two formal protocol scripts are:

- [stage570_fruit_minimal_causal_encoding_protocol.py](/d:/develop/TransformerLens-main/tests/codex/stage570_fruit_minimal_causal_encoding_protocol.py)
- [stage571_unified_language_state_update_protocol.py](/d:/develop/TransformerLens-main/tests/codex/stage571_unified_language_state_update_protocol.py)

## 2. Fruit Minimal Causal Encoding

The working equation is:

```text
h_concept ~= B_global + B_family + E_concept + sum(A_attr) + sum(G_bind(concept, attr_i)) + C_context + eps
```

For apple:

```text
h_apple ~= B_global + B_fruit + E_apple + A_red + A_sweet + G_bind(apple, red) + G_bind(apple, sweet) + C_context + eps
```

The key move is to stop asking for an `apple neuron` and instead recover five causal parts:

- `B_fruit`: fruit family backbone
- `E_apple`: apple-specific concept offset
- `A_attr`: reusable attribute channels such as red or sweet
- `G_bind`: object-attribute binding bridge
- `C_context`: context-conditioned correction

If the decomposition is meaningful, the same structure should also describe banana, pear, and orange with the same family backbone and different concept offsets plus binding terms.

## 3. Unified Language State Update

The language theory should no longer be noun-centered only. The unified state is:

```text
S_t,l = (O_t,l, A_t,l, R_t,l, P_t,l, M_t,l, Q_t,l, G_t,l, C_t,l)
S_t,l+1 = F_l(S_t,l)
```

Where:

- `O_t`: object state
- `A_t`: attribute state
- `R_t`: relation frame
- `P_t`: reference state
- `M_t`: modifier state
- `Q_t`: reasoning state
- `G_t`: binding state
- `C_t`: context condition

This lets the theory absorb:

- pronouns as reference-state updates
- prepositions as relation-frame builders
- adverbs as modifier operators
- logic as reasoning-state propagation

## 4. Four Minimal Experiments

The unified protocol requires four minimal experiments:

1. Pronoun coreference:
   isolate whether pronouns mainly act as discourse pointers rather than ordinary nouns.
2. Preposition relation frame:
   isolate whether prepositions build object-object frames rather than local object attributes.
3. Adverb scope:
   isolate whether different adverbs target event trajectories, proposition confidence, or local properties.
4. Layerwise logic tracing:
   isolate whether intermediate conclusions appear as stable reasoning states before final readout.

## 5. Why These Two Tasks Belong Together

The fruit protocol solves the minimal causal structure problem for concepts.
The unified state-update protocol solves how other language functions join the same mathematics.

Together they push the project from:

- concept encoding guess

toward:

- language computation theory

## 6. Current Hard Limits

- The decomposition may not be unique.
- Most components are likely population-level, not single-neuron tags.
- `C_context` and `Q_reasoning` can become catch-all terms if experiments are weak.
- The theory only becomes strong if the protocols predict held-out cases and survive causal ablation.
