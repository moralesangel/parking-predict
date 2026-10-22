# Results

Phase 5 report. Reproducible with:

```bash
uv run python scripts/train_evaluate.py
uv run python scripts/plot_results.py
```

## Summary

**The model beats persistence at all four horizons**, and the advantage grows
with the horizon: from a 12.8% improvement at 5 minutes to 15.0% at 30, against
calibrated persistence.

But the honest headline is more nuanced, and it is in §5: **almost all the gain
comes from smoothing the current state, not from learning patterns**. The
calendar and neighbouring-zone features, which were the plan's bet, contribute
less than 2% of the signal.

## Experimental setup

| | |
|---|---|
| Training | 2019-01 … 2019-09, subsampled 1 row in 4 -> **5,399,784 rows** |
| Validation | 2019-10 (LightGBM early stopping) -> 627,192 rows |
| **Test** | **2019-11 … 2019-12 in full, not subsampled -> 5,015,232 rows** |
| Base rate in test | 94.49% positive |

**Temporal** split, never random. The test set is the last two months of the
year, later than everything else.

Training is subsampled in time (one row in four, i.e. every 20 min) so it fits
in memory. The subsampling is **by position within each segment** rather than
random, so the temporal structure is not broken. The test set is not
subsampled: the metrics are measured over everything there is.

### The four models

| Model | What it uses |
|---|---|
| `persistence` | The current state, as-is. Answers 0 or 1. The plan's reference. |
| `calibrated_persistence` | The current state, expressed as the empirical probability that the state holds. **The fair reference.** |
| `historical_mean` | Mean per segment, weekday and hour bucket. Ignores the current state. |
| `lightgbm` | All 32 features. One model per horizon. |

Calibrated persistence exists because the raw version is at an artificial
disadvantage: by answering 0 or 1, log loss punishes it for the clipping rather
than for its content. The calibrated one has **exactly the same information** -
the current state, nothing more - and does not suffer that penalty. It is the
one that genuinely has to be beaten.

## 1. Comparison by horizon

![Brier by horizon](img/brier_by_horizon.png)

**Brier score** (lower is better):

| Horizon | LightGBM | Calibrated persistence | Persistence | Historical mean |
|---|---|---|---|---|
| 5 min | **0.01706** | 0.01957 | 0.02186 | 0.04743 |
| 10 min | **0.02202** | 0.02575 | 0.03010 | 0.04742 |
| 15 min | **0.02489** | 0.02927 | 0.03522 | 0.04741 |
| 30 min | **0.02972** | 0.03497 | 0.04445 | 0.04741 |

**Log loss** (lower is better):

| Horizon | LightGBM | Calibrated persistence | Persistence | Historical mean |
|---|---|---|---|---|
| 5 min | **0.0597** | 0.0871 | 0.3020 | 0.1910 |
| 10 min | **0.0754** | 0.1096 | 0.4158 | 0.1913 |
| 15 min | **0.0847** | 0.1221 | 0.4865 | 0.1905 |
| 30 min | **0.1010** | 0.1425 | 0.6141 | 0.1901 |

**LightGBM improvement** (Brier skill score):

| Horizon | vs calibrated persistence | vs raw persistence |
|---|---|---|
| 5 min | 12.8% | 22.0% |
| 10 min | 14.5% | 26.8% |
| 15 min | 14.9% | 29.3% |
| 30 min | **15.0%** | **33.1%** |

Three readings:

- **The advantage grows with the horizon.** That is expected and it is what
  justifies the project: the further away the arrival, the less the current
  state tells you. At 5 minutes persistence nearly suffices; at 30 it does not.
- **The historical mean is flat** (~0.0474 at all four horizons) because it
  ignores the current state, so the horizon is irrelevant to it. At 30 minutes
  it is still **worse than raw persistence** (0.0474 vs 0.0445): routine alone
  does not beat looking at how the street is right now, not even half an hour
  out.
- **Raw persistence does badly on log loss** (0.30 -> 0.61) compared with the
  calibrated one (0.087 -> 0.14). It is not that it predicts worse: it is that
  saying "0%" when there is a 32% chance is a serious error for a product that
  shows probabilities.

## 2. Calibration

![Calibration at 15 minutes](img/calibration.png)

This is the real deliverable: the screen shows a probability, so what matters
is not only ranking well but that the number means what it says.

Expected calibration error at 15 min:

| Model | ECE |
|---|---|
| LightGBM | **0.0015** |
| Calibrated persistence | 0.0014 |
| Raw persistence | 0.0352 |

LightGBM is well calibrated across the whole range (maximum deviation 0.017 in
the low bands) and it **covers the whole range**: it emits probabilities in all
ten deciles. The two persistence variants emit only two distinct values; that
is why the chart shows them as loose points rather than a curve, because
joining them would suggest intermediate values they never produce.

The orange point at `(0.0, 0.32)` is the entire argument against raw
persistence as a product: **it says "0% probability" in situations where 32% of
the time there is space.** It would be telling a driver not to go to a street
where one time in three they would find a spot.

## 3. Breakdown by occupancy level

![Brier by occupancy level](img/by_occupancy.png)

This is the breakdown the plan asks for, and the one that matters: the global
mean is dominated by empty segments, where everybody gets it right.

At 15 minutes:

| Free bays now | n | Base rate | Persistence | Calibrated | LightGBM | Improvement |
|---|---|---|---|---|---|---|
| **0 (full)** | 276,608 | **31.9%** | 0.3193 | 0.2174 | **0.1910** | 12.1% |
| 1 | 446,600 | 86.0% | 0.1404 | 0.1358 | **0.1077** | 20.7% |
| 2 | 520,227 | 96.5% | 0.0352 | 0.0343 | **0.0323** | 5.9% |
| 3-5 | 1,278,852 | 99.5% | 0.0055 | 0.0056 | **0.0053** | 4.5% |
| 6+ | 2,492,945 | 100.0% | 0.0001 | 0.0004 | 0.0001 | — |

What matters:

- **When the segment is full, only 31.9% of the time will there be space 15
  minutes later.** That is where the hard problem lives, and that is where
  LightGBM improves by 12.1% over the best persistence. It is a real and
  frequent case: 276,608 observations, 5.5% of the test set.
- **The best case is "1 free bay"**, at 20.7%. That makes sense: it is the most
  unstable situation - one arriving car flips the answer - and that is where the
  recent trend and turnover rate genuinely inform.
- **In "6+ free" the improvement means nothing.** The base rate is 99.99% and
  the absolute Brier is 0.0001; any relative difference there is noise over a
  problem that does not exist. It is mentioned because a 69% skill score on that
  row would look excellent in a table and would mean nothing.
- **The historical mean collapses in the full case** (Brier 0.507, worse than
  predicting nothing). Knowing that "this street usually has space on Tuesdays
  at 11" is useless when the street is full right now.

## 4. Breakdown by hour

![Brier by hour](img/by_hour.png)

The error follows the occupancy cycle: lowest overnight (everything empty,
everything easy) and highest at the two peaks, midday and evening. The model's
improvement is **positive across all 24 hours**, between 11.6% and 26.1%.

In absolute terms the gain is larger at peak hours, which is where there is
something to gain: at 12 UTC the Brier drops from 0.0523 to 0.0440.

## 5. What the model learned (and what it did not)

Gain-based importance at 15 minutes:

| Feature | Gain |
|---|---|
| `free_bays_mean15` | **46.1%** |
| `free_bays` | **44.2%** |
| `free_bays_mean30` | 3.2% |
| `free_bays_mean60` | 1.5% |
| `hour` | 1.2% |
| `occupancy_rate` | 0.7% |
| `neighbour_occupancy_rate` | 0.3% |
| `neighbour_free_bays` | 0.3% |
| rest (24 features) | < 2.5% |

**90% of the signal is two features: the current state and its mean over the
last 15 minutes.** This has to be said plainly because it partly contradicts
the plan's hypothesis:

- **Neighbouring zones contribute 0.6%.** The plan proposed them as a relevant
  feature and the data says they barely are. Possible explanation: adjacency
  derived from street names is crude, and a segment's own occupancy already
  summarises the pressure around it fairly well.
- **The calendar contributes ~1.6%** (`hour` + `day_of_week` + holidays). The
  daily cycle is very strong - §4 shows it - but the model does not need it: it
  is already implicit in the current state, which *is* the result of the cycle.
- **What genuinely adds value over persistence is the 15-minute rolling mean**,
  that is, smoothing the noise of the instantaneous measurement and capturing
  the trend. A fair reading of the result is that the model is, in essence, "a
  better-measured persistence".

That is a less impressive result than "ML discovers hidden patterns", but it is
the one the data gives.

## 6. Limitations

What these numbers do **not** demonstrate:

1. **These are 2019 results.** The dataset is historical and not suitable for
   production (reasons in [`id_crosswalk.md`](id_crosswalk.md) §5: temporal
   drift and sensor fleet renewal). LightGBM improving by 15% in 2019 does not
   guarantee it does so today. This evaluation will have to be repeated with the
   data the poller accumulates.
2. **No hyperparameter search.** The parameters are conservative and fixed. With
   such a strong baseline, what decides the outcome is whether the features add
   anything, and tuning the third decimal of `learning_rate` before knowing that
   would be optimising noise. The 15% improvement is therefore a reasonable
   **floor**, not a ceiling.
3. **Training is subsampled 1 row in 4.** With the complete data the model might
   improve somewhat; that has not been measured.
4. **`total_bays` is the observed inventory, not the real one.** If a sensor was
   dead for a whole month, its bay does not exist for the model, and the target
   "at least one is free" is evaluated over an incomplete denominator.
5. **The dates carry no timezone**, so the two 2019 clock-change days are
   corrupted. Over two months of test data that is noise.
6. **The alternative of predicting the count** and deriving the probability has
   not been evaluated, although the dataset supports it (`free_bays_h*`). The
   plan asked for it as an alternative to evaluate and it remains pending.

## 7. Conclusion for the product

**Yes, predicting is worth it**, with two caveats that should not be hidden:

- At **5 minutes**, calibrated persistence already does 87% of the work. A
  product that only showed the current state properly expressed as a probability
  would be nearly as good.
- At **30 minutes** - the real use case, the driver already on the way - the
  improvement is 15% over the best persistence and 33% over the raw one, and in
  full zones it reaches 12%. There the prediction does change the answer.

And the strongest argument is not the Brier score but the **calibration**: raw
persistence claims "0%" in situations with a real 32% probability. A product
built on it would send people searching further than necessary one time in
three.

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
