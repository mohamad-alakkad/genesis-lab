# gen4 results

held-out family 'standard', 200 episodes per checkpoint, fresh seeds 3e9+50000..50199 (disjoint from the seeds used to select 'best'; same for all arms), deterministic argmax, run's own input mask

| arm | label | seeds | survival ticks (95% CI) | caught | seeds eaten | sheltered when threatened |
|---|---|---|---|---|---|---|
| gen4-mem-g99 | best | 2 | 739.3 [689.3654, 789.2346] | 0.6375 | 14.8175 | 0.0254 |
| gen4-mem-g99 | final | 2 | 754.9575 [703.5296, 806.3854] | 0.6875 | 15.0675 | 0.0164 |
| gen4-mem-g997 | best | 2 | 1059.2 [1010.9147, 1107.4853] | 0.3575 | 21.5775 | 0.0354 |
| gen4-mem-g997 | final | 2 | 980.63 [931.7014, 1029.5586] | 0.465 | 20.05 | 0.0323 |
| gen4-nomem-g99 | best | 2 | 634.6475 [588.9595, 680.3355] | 0.7125 | 12.61 | 0.0117 |
| gen4-nomem-g99 | final | 2 | 635.5525 [588.5004, 682.6046] | 0.7625 | 13.2575 | 0.0222 |
| gen4-nomem-g997 | best | 2 | 1074.1825 [1026.9183, 1121.4467] | 0.3775 | 22.1425 | 0.035 |
| gen4-nomem-g997 | final | 2 | 997.7925 [949.823, 1045.762] | 0.4825 | 20.825 | 0.0386 |

## Verdict (pre-registered rule: CI-separated survival)
- gamma 99: memory 739.3 vs masked 634.6475 ticks → memory better (CIs separated)
- gamma 997: memory 1059.2 vs masked 1074.1825 ticks → no CI-separated memory effect
- mem: gamma .997 1059.2 vs .99 739.3 ticks → longer horizon better (CIs separated)
- nomem: gamma .997 1074.1825 vs .99 634.6475 ticks → longer horizon better (CIs separated)

Reference: scripted survivor 724 ticks; sensory hider with 150-tick memory 933; gen3 best 689.
