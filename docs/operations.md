# Operations

Phase 7 report: scheduled retraining and monitoring.

```bash
uv run python scripts/monitor.py    # monitoring report
uv run python scripts/retrain.py    # retraining job
```

Dashboard at `http://localhost:8000/monitoring.html`, which consumes
`GET /monitoring`.

## 1. Retraining has a gate, and that is the important part

Automated retraining carries a risk manual deployment does not: **nobody looks
at the result before it reaches production**. If a month turns out bad - a long
feed outage, roadworks emptying half the city - the retrained model can be worse
than the previous one, and worse than not predicting at all.

That is why a model is **never promoted merely for having been trained**. It has
to pass two tests on a temporal holdout:

1. **Beat calibrated persistence.** This follows directly from phase 5. A model
   that cannot beat it adds nothing that was not already in the current state,
   and deploying it would add an entire ML pipeline in exchange for nothing.
2. **Not regress against the production model** by more than 2%. Retraining must
   not be a step backwards. The 2% margin absorbs the noise of a finite holdout;
   without it the model would change every week over differences that mean
   nothing.

**A retraining run that does not promote is a correct outcome, not a failure.**
The job exits 0 and records the reason.

### The gate has already done its job

It is not a theoretical safeguard. On the first real run over the accumulated
data:

```
train 10976 rows | holdout 3808 rows
candidate 0.00776 | persistence 0.00672 | production none
NOT promoting: does not beat persistence (candidate 0.00776 >= persistence
0.00672); deploying it would add an ML pipeline in exchange for nothing
```

With 6 hours of data, LightGBM comes out **worse** than persistence. The gate
rejected it and production stayed as it was.

### Before that, the "is there enough data" gate

The job first checks there is something to train on, and **refuses** if not:

- **14 days minimum.** That covers two full weekly cycles; with less, the model
  has not seen a weekend often enough to tell it apart from noise.
- **50,000 labelled rows minimum**, so it does not train on a handful of
  segments during a feed outage.

These are judgement calls, not thresholds derived from the data, which is why
they are parameters (`--min-days`, `--min-rows`).

Right now the job refuses, and that is correct:

```
accumulated data: 0.24 days | 224 segments | 15456 rows
NOT retraining. insufficient data: only 0.24 days of data (minimum 14);
only 14784 labelled rows (minimum 50000)
```

### The registry records rejections too

`data/processed/registry/history.jsonl` keeps one line per decision, promoted
or not. Knowing the model has gone five weeks without promoting because it
cannot beat persistence is first-order operational information; keeping only
the wins would leave the registry lying by omission.

Promoted versions are archived under `versions/<timestamp>/`, so rolling back
is a directory copy.

## 2. The three monitoring signals

Each answers a different question, and a system can be fine on two and bad on
the third.

### Degradation: is the model still beating persistence *today*?

This is the signal that really matters, and it works because **the label
arrives on its own**: the prediction made at `t` for `t+h` can be checked as
soon as the poller captures `t+h`. Nothing has to be annotated by hand, only
`h` minutes waited.

Measured over the accumulated data, with the model trained on 2019:

| Horizon | n | Model Brier | Persistence Brier | Skill |
|---|---|---|---|---|
| 5 min | 10,936 | 0.01146 | 0.01161 | **+1.3%** |
| 10 min | 10,936 | 0.01484 | 0.01655 | +10.3% |
| 15 min | 10,713 | 0.01585 | 0.01895 | +16.3% |
| 30 min | 10,044 | 0.02144 | 0.02688 | **+20.3%** |

**The 2019-trained model still beats persistence on 2026 data.** That is better
than warned in phase 6, where I said it might not transfer.

But there is real degradation and it should not be hidden: **at 5 minutes the
skill falls from the 12.8% measured on 2019 to today's 1.3%**. At longer
horizons it holds up (15.0% -> 20.3%). The reading is that the part of the model
capturing short-term dynamics is what has aged worst.

The backtest **excludes interpolated rows**. Evaluating them would measure the
quality of the gap-filling, not the model: a row carried over from the previous
instant makes persistence look perfect.

### Drift: does today's occupancy look like the training distribution?

PSI (Population Stability Index) over the occupancy rate, with fixed cut points.
**Occupancy** is compared rather than the target because it is the signal that
dominates the model: 90% of the gain comes from the current state and its recent
mean (`docs/results.md` §5).

Thresholds: < 0.10 stable, 0.10-0.25 moderate, > 0.25 large. These are
**industry convention, not a hypothesis test**: they exist to trigger a review,
not to conclude anything on their own.

Current measurement: **PSI 0.0451 (stable)** against June 2019.

**This figure has to be read with great caution** and it is the report's most
serious limitation: the current window is ~6 **night-time** hours and the
reference is a full month. The PSI is comparing distributions with different
hour coverage, so it mixes "the city has changed" with "it is night-time now".
With a week of polling the comparison becomes hour against hour and the figure
will be interpretable; until then, the number is a placeholder that demonstrates
the pipeline works, not a measure of drift.

### Coverage: how much of the city do we still have data for?

The easiest to overlook: **a model can look perfect while half the sensors go
dark**, because the metrics are computed over whatever is left. Falling coverage
means the product covers less of the city, even if it is just as accurate.

Current state: **56.2%** (3554 of 6324 bays), 56 zones with no data at all, 1710
frozen sensors and 1060 offline. Consistent with phase 2.

## 3. What the scheduled deployment does

The compose `operations` service runs in a loop, every 24 h by default:

1. `scripts/monitor.py` -> regenerates `report.json`
2. `scripts/retrain.py --horizon 15` -> attempts a retrain and promotion

It is a `sleep` loop rather than cron: the container is single-process, and
adding a cron daemon just for this would complicate the logs and the shutdown
signal. If the job grows, move it to a real scheduler.

`GET /monitoring` serves the latest report **without recomputing it**: doing so
would walk the whole accumulated series inside an HTTP request. The response
includes `generated_at` and `is_stale`, because **a stale monitoring report is
itself a symptom**: it means the scheduled job is not running.

## 4. Limitations

1. **The drift is not interpretable yet**, for the reason above: windows with
   different hour coverage are being compared. It is the weakest piece of the
   report.
2. **Retraining has never promoted anything**, because there is not enough data.
   The full path is exercised (trains, evaluates, decides) but the promotion
   branch is only covered by tests, not by real data.
3. **Only the 15-minute horizon is retrained.** The other three still use the
   2019 model. Extending it is a loop, but it quadruples the job's runtime.
4. **There are no alerts, only a report.** Nobody is notified if `is_degraded`
   turns `true`; the dashboard has to be checked. Wiring it to a real channel is
   out of scope for this phase.
5. **The 24 h threshold for "stale report" is arbitrary**, chosen to fit a daily
   job.

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
