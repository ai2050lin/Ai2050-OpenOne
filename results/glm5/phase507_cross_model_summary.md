# Phase507 Orthogonal Semantic Field Cross-model Summary

## qwen3

L=36, d=2560, categories=7, objects/category=30

| metric | value |
|---|---:|
| final mean perp/para ratio | 34.4729 |
| final mean abs cos(phi,qc) | 0.009964 |
| final mean pca_n90 | 15.0000 |
| final mean phi_perp_norm | 69.6414 |
| last-probe category acc para | 0.8254 |
| last-probe category acc perp | 1.0000 |
| last-probe tc-mode acc para | 0.4921 |
| last-probe tc-mode acc perp | 0.8571 |
| mean rich category argmax | 0.0000 |

### Category Details

| category | final ratio | final cos | n90 | best rm_perp ΔD | best layer | strongest positive ΔD | pos layer | rich argmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fruit | 11.2600 | 0.008530 | 15 | -5.8001 | L18 | 0.5048 | L31 | 0.0000 |
| animal | 5.4600 | 0.020849 | 18 | -6.3225 | L9 | -1.7476 | L31 | 0.0000 |
| action | 73.7400 | 0.005486 | 15 | -2.9363 | L18 | 1.8749 | L31 | 0.0000 |
| emotion | 16.2000 | 0.006455 | 13 | -4.2313 | L9 | -0.0781 | L27 | 0.0000 |
| clothing | 33.1000 | 0.004032 | 16 | -3.7522 | L18 | -0.5309 | L31 | 0.0000 |
| color | 96.2100 | 0.002906 | 12 | -4.1846 | L33 | 3.1419 | L27 | 0.0000 |
| vehicle | 5.3400 | 0.021488 | 16 | -6.3681 | L18 | -0.4270 | L31 | 0.0000 |

## glm4

L=40, d=4096, categories=7, objects/category=30

| metric | value |
|---|---:|
| final mean perp/para ratio | 139.3600 |
| final mean abs cos(phi,qc) | 0.004482 |
| final mean pca_n90 | 16.4286 |
| final mean phi_perp_norm | 174.4214 |
| last-probe category acc para | 0.7302 |
| last-probe category acc perp | 1.0000 |
| last-probe tc-mode acc para | 0.5238 |
| last-probe tc-mode acc perp | 0.8413 |
| mean rich category argmax | 0.0000 |

### Category Details

| category | final ratio | final cos | n90 | best rm_perp ΔD | best layer | strongest positive ΔD | pos layer | rich argmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fruit | 125.1300 | 0.003799 | 17 | -4.5439 | L20 | -0.4273 | L37 | 0.0000 |
| animal | 141.9800 | 0.003147 | 18 | -3.4969 | L20 | -0.5363 | L37 | 0.0000 |
| action | 130.7200 | 0.002493 | 18 | -1.7717 | L10 | 0.6637 | L35 | 0.0000 |
| emotion | 72.0100 | 0.004319 | 15 | -2.4583 | L20 | -1.5802 | L10 | 0.0000 |
| clothing | 158.5100 | -0.002911 | 16 | -2.0605 | L10 | 0.1449 | L26 | 0.0000 |
| color | 328.5100 | 0.000933 | 14 | -2.9007 | L35 | -2.2927 | L37 | 0.0000 |
| vehicle | 18.6600 | 0.013772 | 17 | -6.0391 | L20 | -2.8842 | L37 | 0.0000 |

## deepseek7b

L=28, d=3584, categories=7, objects/category=30

| metric | value |
|---|---:|
| final mean perp/para ratio | 129.8600 |
| final mean abs cos(phi,qc) | 0.006044 |
| final mean pca_n90 | 8.4286 |
| final mean phi_perp_norm | 156.6486 |
| last-probe category acc para | 0.7460 |
| last-probe category acc perp | 1.0000 |
| last-probe tc-mode acc para | 0.5238 |
| last-probe tc-mode acc perp | 0.7619 |
| mean rich category argmax | 0.0000 |

### Category Details

| category | final ratio | final cos | n90 | best rm_perp ΔD | best layer | strongest positive ΔD | pos layer | rich argmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fruit | 9.6300 | 0.015234 | 6 | -6.6263 | L21 | -1.6877 | L23 | 0.0000 |
| animal | 62.5200 | -0.002725 | 12 | -1.5047 | L18 | 1.1996 | L21 | 0.0000 |
| action | 27.7700 | -0.008779 | 10 | 2.1736 | L25 | 4.3741 | L23 | 0.0000 |
| emotion | 312.3400 | 0.002245 | 9 | -1.1297 | L25 | 0.4028 | L14 | 0.0000 |
| clothing | 23.7100 | 0.004149 | 7 | -1.9097 | L7 | 0.8378 | L25 | 0.0000 |
| color | 23.4300 | -0.007878 | 9 | 0.8135 | L7 | 3.6215 | L23 | 0.0000 |
| vehicle | 449.6200 | 0.001297 | 6 | -1.8284 | L7 | -0.4016 | L23 | 0.0000 |

## Cross-model Takeaways

| model | mean ratio | mean abs cos | probe perp cat acc | probe perp tc acc | mean rich argmax |
|---|---:|---:|---:|---:|---:|
| qwen3 | 34.4729 | 0.009964 | 1.0000 | 0.8571 | 0.0000 |
| glm4 | 139.3600 | 0.004482 | 1.0000 | 0.8413 | 0.0000 |
| deepseek7b | 129.8600 | 0.006044 | 1.0000 | 0.7619 | 0.0000 |

