# gen5-families

fresh held-out block: eval_seed_base + family offset + 50000..50099; best checkpoints were selected on block 0..99; 100 episodes per run per family; arms pool their seeds.

| family | arm | survival ticks (95% CI) | caught | seeds eaten | sheltered when threatened | dash when threatened |
|---|---|---|---|---|---|---|
| standard | gen5-broad | 1019.7 [971.9628, 1067.4372] | 0.3975 | 20.13 | 0.0359 | 0.1985 |
| standard | gen5-std | 1043.5475 [995.8573, 1091.2377] | 0.3975 | 20.7625 | 0.0368 | 0.1779 |
| relentless | gen5-broad | 783.845 [739.7465, 827.9435] | 0.755 | 16.1325 | 0.0258 | 0.2352 |
| relentless | gen5-std | 759.765 [714.1109, 805.4191] | 0.7425 | 15.9575 | 0.0289 | 0.2293 |
| night | gen5-broad | 1122.4575 [1079.0479, 1165.8671] | 0.375 | 22.6575 | 0.0294 | 0.202 |
| night | gen5-std | 1108.95 [1064.7983, 1153.1017] | 0.4075 | 22.895 | 0.0375 | 0.1814 |
| drought | gen5-broad | 1002.3325 [955.0911, 1049.5739] | 0.44 | 18.4925 | 0.0211 | 0.1764 |
| drought | gen5-std | 1028.7525 [982.6255, 1074.8795] | 0.4225 | 19.0925 | 0.0186 | 0.1659 |
| fissure_heavy | gen5-broad | 1003.3475 [955.296, 1051.399] | 0.4375 | 19.7175 | 0.0306 | 0.1877 |
| fissure_heavy | gen5-std | 1007.28 [958.8569, 1055.7031] | 0.425 | 19.8875 | 0.0382 | 0.188 |
| lure_heavy | gen5-broad | 849.3975 [805.0124, 893.7826] | 0.58 | 15.42 | 0.0326 | 0.1696 |
| lure_heavy | gen5-std | 823.2675 [777.9909, 868.5441] | 0.615 | 15.175 | 0.0332 | 0.1668 |
| one_shelter | gen5-broad | 961.0825 [914.2908, 1007.8742] | 0.4225 | 13.6725 | 0.0125 | 0.1898 |
| one_shelter | gen5-std | 973.235 [927.2021, 1019.2679] | 0.435 | 13.9375 | 0.0143 | 0.1782 |
| unseen_layout | gen5-broad | 972.38 [924.1015, 1020.6585] | 0.4325 | 20.1975 | 0.0263 | 0.1969 |
| unseen_layout | gen5-std | 944.955 [895.6837, 994.2263] | 0.39 | 19.2525 | 0.0253 | 0.1993 |
| perturbed | gen5-broad | 945.9475 [901.5429, 990.3521] | 0.575 | 17.55 | 0.0395 | 0.1726 |
| perturbed | gen5-std | 920.98 [876.4722, 965.4878] | 0.6225 | 17.185 | 0.0381 | 0.1669 |
| no_predator | gen5-broad | 1427.6875 [1402.1554, 1453.2196] | 0.0 | 36.2175 | None | None |
| no_predator | gen5-std | 1448.8225 [1428.1702, 1469.4748] | 0.0 | 36.5025 | None | None |

## Arm comparison (survival, CI-separated?)
- standard: gen5-broad 1019.7, gen5-std 1043.5475 — overlapping
- relentless: gen5-broad 783.845, gen5-std 759.765 — overlapping
- night: gen5-broad 1122.4575, gen5-std 1108.95 — overlapping
- drought: gen5-broad 1002.3325, gen5-std 1028.7525 — overlapping
- fissure_heavy: gen5-broad 1003.3475, gen5-std 1007.28 — overlapping
- lure_heavy: gen5-broad 849.3975, gen5-std 823.2675 — overlapping
- one_shelter: gen5-broad 961.0825, gen5-std 973.235 — overlapping
- unseen_layout: gen5-broad 972.38, gen5-std 944.955 — overlapping
- perturbed: gen5-broad 945.9475, gen5-std 920.98 — overlapping
- no_predator: gen5-broad 1427.6875, gen5-std 1448.8225 — overlapping
