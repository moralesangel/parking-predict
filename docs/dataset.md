# The prediction problem and the dataset

Phase 4 report. Reproducible with:

```bash
uv run python scripts/build_dataset.py --months 2019-01:2019-12
```

## The problem

> Given a street segment's state at `t`, what is the probability that **at
> least one bay is free** at `t + h`, for `h` in {5, 10, 15, 30} minutes?

The benchmark to beat is **persistence**: answering with the current state. The
whole project exists to check whether that can be improved, because when you
drive 12 minutes to a destination what matters is not how the zone looks now
but how it will look when you arrive.

The dataset stores **two targets per horizon**:

- `target_h5`, `target_h10`, `target_h15`, `target_h30` - boolean, at least one
  bay is free. This is the classification the plan asks for.
- `free_bays_h5` … `free_bays_h30` - the count of free bays. This enables the
  alternative the plan requires evaluating: predicting the count and deriving
  the probability from it, rather than classifying directly.

## The spatial unit is the street segment

Decided: `segment_id`, not H3 cells. Three reasons, in order of weight:

1. **It is the only level at which the archive and the real-time feed
   reconcile.** The archive has no coordinates, so assigning H3 cells would
   require latitudes that do not exist and would leave the 42.7M events out of
   any spatial feature. Measured in [`id_crosswalk.md`](id_crosswalk.md): the
   segment join covers 88% of the events.
2. **It matches the driver's decision.** "Is there space on this block?" is the
   real question; a 100 m hexagon corresponds to nothing visible from the car.
3. **Sensor failures are zone-level, not bay-level** (see
   [`data_quality.md`](data_quality.md) §4), so a street-aggregated unit fits
   the way the data degrades.

A segment is `STREET|CROSS_A|CROSS_B` with the two bounding streets sorted
alphabetically, because the data uses both orders for the same segment.

**Both sides of the street are pooled.** A driver looking for space on a block
checks both. It is also the granularity of the geometry dataset's
`roadsegmentid`, which keeps the crosswalk intact.

### Adjacency without geometry

The "neighbouring zone" features need a graph, and the archive has no
coordinates. It is derived from the names themselves: a segment occupies two
**intersections** (`{STREET, CROSS_A}` and `{STREET, CROSS_B}`), and two
segments are neighbours if they share one.

This captures the adjacent block and the perpendicular corner, and - crucially -
does **not** join two Collins Street segments at opposite ends of the CBD, which
is the mistake a "same street" rule would make.

Result: median of 3 neighbours per segment, maximum 7, and 24 isolated segments
that receive nulls in those features.

## The mistake that nearly ruined the dataset: `VehiclePresent`

Worth documenting because it produces no failure, only plausible and wrong
numbers.

The archive **is not a list of stays**. It is a partition of each bay's time
into intervals, and `VehiclePresent` says what each one is:

- `true` -> the bay was **occupied** during that interval
- `false` -> the bay was **free** during that interval

The source description spells it out: an event records whether a vehicle was
present or not, *"which can determine stay time **and vacancy time**"*.

Measured over the week of 1-8 March 2019:

| Intervals | Rows | Share of bay-time |
|---|---|---|
| `VehiclePresent = true` | 378,792 | **43.7%** |
| `VehiclePresent = false` | 340,829 | 47.9% |
| Total | 719,621 | 91.6% |

Treating every row as a stay gives an occupancy of **91.6%**, which is not an
occupancy at all but the sum of occupied plus free. The symptom that gave it
away was the hourly profile: 91.8% at three in the morning and 91.5% at midday,
in other words no daily cycle. A CBD car park with no daily cycle does not
exist.

With the correct filter the profile is as expected, with a double peak:

```
  04h  21.7%  ##########
  07h  34.4%  #################
  09h  48.3%  ########################
  11h  64.0%  ###############################   <- midday peak
  17h  43.7%  #####################
  19h  59.4%  #############################     <- dinner peak
  23h  34.7%  #################
```

And the 43.7% agrees with what the real-time feed measures today (44-55%
depending on the hour), which is an independent check.

The remaining 8.4% of bay-time is covered by no interval at all: those are the
sensor's own observation gaps.

## Occupancy reconstruction

From intervals to a time series, without the cartesian product that does not
terminate over 42.7M rows:

1. Each occupied interval is split into two deltas: `+1` on arrival, `-1` on
   departure.
2. They are sorted by time within each segment.
3. The **cumulative sum** gives the occupied bays after each event.
4. A `join_asof` carries that value onto the 5-minute grid.

Cost: one sort instead of a join. The full year processes in ~8 min.

Two details that change the result:

- **Intervals overlapping the window are included**, not only those starting
  inside it. A car parked since before still occupies its bay, and filtering on
  `arrival >= start` would make the segment look emptier.
- **`total_bays` counts both interval types.** A bay free for the whole window
  produces no occupied interval but still exists; deriving the inventory from
  occupied intervals alone would lose it precisely in the emptiest segments.

A **5-minute** grid because that is the plan's shortest horizon: a coarser grid
could not express the 5-minute target.

## The dataset

| | |
|---|---|
| Period | 2019-01-01 to 2019-12-31 |
| Rows | **29,123,136** |
| Segments | 301 |
| Features | 32 |
| Targets | 4 booleans + 4 counts |
| Format | Parquet partitioned under `data/processed/dataset/month=YYYY-MM/` |

It is processed **month by month** to bound memory, and each month is computed
with 60 min of padding on the left (the longest lag) and 30 min on the right
(the longest horizon), trimmed afterwards. Without that padding the first rows
of each month would have null lags and the last ones null targets, and an
artificial gap would appear on the 1st of every month.

Segments with fewer than 3 observed bays are dropped: with 1 or 2, "at least
one is free" is effectively the state of a single bay rather than a zone-level
problem.

### Features

All computable at `t`. None looks into the future.

- **Current state**: `free_bays`, `occupied_bays`, `total_bays`,
  `occupancy_rate`
- **Recent history**: `free_bays_lag{5,10,15,30,60}` and their rate
  equivalents; `free_bays_trend{...}` (change against the lag);
  `free_bays_mean{15,30,60}` (rolling means)
- **Neighbours**: `neighbour_free_bays`, `neighbour_occupancy_rate`,
  `neighbour_count`
- **Calendar**: `hour`, `minute_of_day`, `minute_sin`, `minute_cos`,
  `day_of_week`, `is_weekend`, `is_holiday`

`minute_of_day` also goes in as sine and cosine: midnight is one minute away
from 23:59 and a flat integer claims they are 1439 apart.

The trends exist because two segments with the same occupancy can have opposite
futures depending on whether they are filling or emptying.

`feature_columns()` derives the list from the data rather than hard-coding it,
so adding a new horizon cannot smuggle its column in as a feature.

### Temporal split, never random

`temporal_split()` cuts on `timestamp`. A random split would put `t+5` in
training and `t` in test, and those are overwhelmingly correlated: the metrics
would be excellent and the model unable to predict the real future.

**Known boundary leak.** With horizon `h`, the label of the last training row
comes from `train_end + h`, which already falls in validation. With
`h <= 30 min` over months of data that is negligible; if the blocks get shorter,
the last `h` rows of each one must be dropped.

## The base rate: the most important number for phase 5

**The target is positive 94-96% of the time.**

| Month | `target_h5` positive |
|---|---|
| 2019-01 | 95.96% |
| 2019-06 | 93.90% |
| 2019-12 | 94.37% |

This shapes all of phase 5's evaluation:

- **"There is always space" is right 95% of the time.** Accuracy is useless as a
  metric. That is why the plan asks for Brier score, log loss and calibration
  curves.
- **The interesting case is the remaining 5%**, which is exactly what the plan
  anticipated when it asked for a breakdown by occupancy level. A model that
  improves the global average without distinguishing the "nearly full zone" case
  is worthless: nobody needs to be told there is space on an empty street.
- Persistence is going to be **very strong** at 5 minutes. If it is not beaten,
  that has to be documented as-is.

## Limitations

1. **A bay with no interval at all is invisible.** There is no way to tell
   "does not exist", "was occupied the whole period" and "the sensor was dead"
   apart. `total_bays` is the *observed* inventory, not the real one.
2. **58 rows out of 606,816 (0.01%) have `occupied_bays > total_bays`**, from
   overlapping intervals on the same bay. `free_bays` is clipped at 0; the
   source intervals are not corrected.
3. **Local time with no timezone.** The archive's dates are Melbourne local with
   daylight saving. They are processed as naive instants, so the two 2019
   clock changes have one duplicated hour and one non-existent one. At yearly
   scale that is noise, but it invalidates any fine analysis of those two days.
4. **The Victoria 2019 public holidays come from the public register and should
   be verified** against the official source, especially the movable ones (the
   Friday before the AFL Grand Final, and Melbourne Cup). A wrong holiday breaks
   nothing, it only adds noise to one feature.
5. **This dataset is from 2019 and is not production-grade.** The reasons are in
   [`id_crosswalk.md`](id_crosswalk.md) §5: temporal drift and sensor fleet
   renewal. It serves to build and validate the pipeline; the production model
   will be trained on whatever the poller accumulates.
6. **Real-time data does not go through this pipeline yet.** The functions are
   written against the archive's schema. Adapting them to the poller's snapshots
   is phase 7 work, once there are weeks of accumulated data.

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
