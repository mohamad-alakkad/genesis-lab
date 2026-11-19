# gen6-families

fresh held-out block: eval_seed_base + family offset + 50000..50099; best checkpoints were selected on block 0..99; 100 episodes per run per family; arms pool their seeds.

| family | arm | survival ticks (95% CI) | caught | seeds eaten | sheltered when threatened | dash when threatened |
|---|---|---|---|---|---|---|
| standard | gen6-food | 1064.0775 [1016.5311, 1111.6239] | 0.3875 | 21.5275 | 0.0322 | 0.182 |
| standard | gen6-life | 1235.3 [1193.9186, 1276.6814] | 0.13 | 18.435 | 0.0419 | 0.203 |
| relentless | gen6-food | 785.4475 [741.2934, 829.6016] | 0.755 | 15.9975 | 0.0331 | 0.2221 |
| relentless | gen6-life | 1093.2 [1049.706, 1136.694] | 0.36 | 15.9375 | 0.0372 | 0.2227 |
| night | gen6-food | 1122.2625 [1079.1844, 1165.3406] | 0.375 | 22.7425 | 0.0321 | 0.1888 |
| night | gen6-life | 1273.27 [1234.331, 1312.209] | 0.12 | 19.32 | 0.042 | 0.1866 |
| drought | gen6-food | 1019.7 [972.8011, 1066.5989] | 0.4375 | 19.585 | 0.0213 | 0.1712 |
| drought | gen6-life | 1176.0625 [1133.4264, 1218.6986] | 0.1725 | 16.7325 | 0.0187 | 0.1932 |
| fissure_heavy | gen6-food | 1016.1575 [967.6969, 1064.6181] | 0.4175 | 19.935 | 0.0377 | 0.1868 |
| fissure_heavy | gen6-life | 1156.335 [1111.5621, 1201.1079] | 0.1625 | 16.5675 | 0.0358 | 0.2124 |
| lure_heavy | gen6-food | 854.725 [810.9726, 898.4774] | 0.615 | 15.55 | 0.0392 | 0.1685 |
| lure_heavy | gen6-life | 1135.0275 [1093.2271, 1176.8279] | 0.2775 | 15.56 | 0.0367 | 0.1759 |
| one_shelter | gen6-food | 955.2575 [908.3406, 1002.1744] | 0.465 | 13.785 | 0.013 | 0.1957 |
| one_shelter | gen6-life | 1108.56 [1065.221, 1151.899] | 0.1725 | 12.5175 | 0.0138 | 0.2067 |
| unseen_layout | gen6-food | 964.1675 [916.4196, 1011.9154] | 0.4225 | 19.8675 | 0.0348 | 0.2041 |
| unseen_layout | gen6-life | 1098.745 [1052.6282, 1144.8618] | 0.2125 | 16.3475 | 0.0325 | 0.2218 |
| perturbed | gen6-food | 959.315 [915.5268, 1003.1032] | 0.56 | 17.935 | 0.0388 | 0.1642 |
| perturbed | gen6-life | 1184.435 [1144.0838, 1224.7862] | 0.2625 | 16.3225 | 0.0431 | 0.1799 |
| no_predator | gen6-food | 1445.66 [1423.2821, 1468.0379] | 0.0 | 36.475 | None | None |
| no_predator | gen6-life | 1443.8625 [1421.4521, 1466.2729] | 0.0 | 27.5175 | None | None |

## Arm comparison (survival, CI-separated?)
- standard: gen6-food 1064.0775, gen6-life 1235.3 — separated
- relentless: gen6-food 785.4475, gen6-life 1093.2 — separated
- night: gen6-food 1122.2625, gen6-life 1273.27 — separated
- drought: gen6-food 1019.7, gen6-life 1176.0625 — separated
- fissure_heavy: gen6-food 1016.1575, gen6-life 1156.335 — separated
- lure_heavy: gen6-food 854.725, gen6-life 1135.0275 — separated
- one_shelter: gen6-food 955.2575, gen6-life 1108.56 — separated
- unseen_layout: gen6-food 964.1675, gen6-life 1098.745 — separated
- perturbed: gen6-food 959.315, gen6-life 1184.435 — separated
- no_predator: gen6-food 1445.66, gen6-life 1443.8625 — overlapping
