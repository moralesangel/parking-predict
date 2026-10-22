# Sensor data quality

Phase 2 report. Figures measured on the snapshot of
**2026-10-22 11:46:49 UTC** unless stated otherwise, and recomputable with:

```bash
uv run python scripts/quality_report.py
```

## Summary

**Of the 6324 bays the feed publishes, only 3557 (56.2%) give a usable
reading.** The other 2767 fail in two distinct ways, which have to be kept
apart because their causes and consequences differ.

| State | Bays | % | Apparent occupancy |
|---|---|---|---|
| `ok` - reports and changes state | 3557 | 56.2% | 44.5% |
| `stuck` - reports but frozen | 1707 | 27.0% | 39.3% |
| `offline` - absent from the feed batch | 1060 | 16.8% | 53.9% |

## 1. The two failure modes

### `offline`: the relay is not reporting

**Exact** detection, no threshold: `lastupdated < max(lastupdated)` within the
snapshot. Because `lastupdated` is a batch stamp rather than a per-sensor one
(`docs/data_schema.md` §1.2), a bay that is not in the current batch carries
the stamp of the last one it appeared in.

### `stuck`: the sensor is frozen

This is the **more numerous failure and the invisible one** if you only watch
`lastupdated`, which is what the source's own documentation recommends. The bay
*is* in the current batch - the relay reports it - but its `status_timestamp`
has not moved in an age: **median 150 days**, with cases up to **1432 days**
(nearly four years).

1707 bays, **32.4% of those online**. Ignoring it would mean treating as good
1707 readings that have been the same for months.

## 2. The 24 h threshold comes from the data, not a hunch

The age of `status_timestamp` among online sensors is clearly **bimodal**, and
the threshold sits in the valley:

```
  0-1h          1584  ###############################################################
  1-2h           718  ############################
  2-4h           633  #########################
  4-8h           421  ################
  8-12h          125  #####
  12-18h          64  ##
  18-24h          12            <-- valley: 12 bays out of 5264
  24-48h          59  ##
  48-72h          22
  72-168h         43  #
  168-336h       140  #####
  336-720h        60  ##
  720-2160h      274  ##########
  2160-8760h     602  ########################
  >8760h         507  ####################
```

The healthy mode tails off around 12 h, the 18-24 h band is almost empty, and
from 720 h (30 days) onwards a second mode appears. Any cut between 18 and 24
hours separates the two populations; 24 h is chosen because it is a parking
lot's natural turnover cycle.

The choice also **barely moves the result**, which is the sign that the
threshold falls in a valley rather than on a slope:

| Threshold | Reliable bays | Coverage | Occupancy |
|---|---|---|---|
| 6 h | 3227 | 51.0% | 44.4% |
| 12 h | 3482 | 55.1% | 44.6% |
| 18 h | 3545 | 56.1% | 44.5% |
| **24 h** | **3557** | **56.2%** | **44.5%** |
| 48 h | 3616 | 57.2% | 44.6% |
| 7 d | 3681 | 58.2% | 44.7% |
| 30 d | 3881 | 61.4% | 44.2% |

Configurable through `PARKING_STUCK_AFTER_HOURS`.

**Known false positive:** a bay with no time restriction and legitimately
occupied for more than 24 h would be marked stuck. Two things bound the risk:
the median of the flagged bays is 150 days, far beyond any plausible long stay;
and 60.7% of the stuck ones appear **free**, and a bay free for months in
Melbourne's CBD is not a long stay, it is a dead sensor. With the archive and
the per-bay restrictions (`on-street-car-park-bay-restrictions`, field
`duration1..6`) the threshold could be tuned per restriction type; that is
pending and does not block phase 3.

## 3. The bias of not filtering is not constant (and is worst at peak)

This corrects a preliminary estimate that compared snapshots from different
hours and produced a fixed figure of ~12 points. Measured **within each
snapshot**, the bias varies:

| Snapshot (UTC) | All bays | Reliable only | Bias |
|---|---|---|---|
| 10:04:49 | 54.7% | 62.3% | **+7.6 pts** |
| 10:22:49 | 53.6% | 60.5% | +6.9 pts |
| 11:42:49 | 45.0% | 45.0% | 0.0 pts |
| 11:44:49 | 44.7% | 44.5% | -0.1 pts |
| 11:46:49 | 44.7% | 44.5% | -0.1 pts |

The reason is mechanical, which makes the problem predictable: **stuck readings
carry a frozen occupancy of around 39.3% that does not follow the daily cycle.**
Averaging them in pulls the mean towards that 39.3%. The bias is therefore
proportional to the distance between true occupancy and 39.3%: almost nil when
the fleet sits near that value (as at 11:46) and largest when the CBD is full.

In other words, **the bias is worst exactly at peak hours, which is when the
prediction matters.** That is reason enough for `is_reliable` to be mandatory
in features and metrics rather than optional.

## 4. The failures are relay-level, not sensor-level

If sensors failed independently, a zone would show a mix of states. What is
observed is the opposite:

| | Zones |
|---|---|
| Zones with >= 5 bays | 368 |
| **Entirely offline** | **37** |
| **Entirely stuck** | **44** |

**81 zones (22%) fail as a block.** This matches what `docs/data_schema.md`
§1.2 showed: the 49 distinct `lastupdated` values are death cohorts, groups of
bays that disappeared in the same cycle.

Consequence for the product: the failure unit is the zone, so degradation is
not gradual but stepwise. A zone does not slowly lose reliability; it goes dark
all at once. Phase 3's map must be able to say **"no data in this zone"** as a
first-class state, rather than painting an empty zone as if it were free.

## 5. Coverage per zone

Across the 191 zones with at least 10 bays:

- **Median coverage: 72.2%**
- **56 zones (29%) are at 0%**: not a single reliable reading

Zones at 0% cannot be used to predict anything. In phase 4 they have to be
excluded from the training set - not as deleted rows, but flagged - and in
phase 6 the endpoint must answer "no data" rather than an invented probability
over zero observations.

## 6. Flapping sensors

9 bays alternate between reliable and unreliable within the observed window.
They are more damaging than their number suggests: a sensor that drops in and
out of the batch **moves its zone's denominator** without a single car having
moved, so aggregate occupancy changes through a measurement artefact. They are
candidates to exclude from the features even when they look healthy
individually.

`parking.quality.coverage.flapping_sensors` detects them over the accumulated
series. With more days of data this list will grow and it is worth reviewing
before phase 4.

## 7. Series continuity

`parking.quality.gaps` compares consecutive batch stamps against the feed's
cadence (120 s) with a tolerance of 3 cycles, so a normal delay is not flagged
as a gap.

Working as intended: during the test window it correctly detected the two gaps
left by the manual `--once` runs before the continuous poller was started
(18 min and 1 h 20 min). With the poller running, capture is one batch every 2
minutes with no losses.

The report does not attribute a cause to a gap, deliberately: "the poller did
not capture" and "the source did not publish" look identical in the stored
files, and telling them apart requires the poller's logs. Mixing them would
give a false sense of diagnosis.

## 8. Limitations of this report

It is worth being explicit about what **cannot yet be claimed**:

- **Coverage per hour is not characterised.** The observed window is ~2 hours
  (6 snapshots), so `coverage_by_hour` only returns two rows, both at 56.2%.
  Knowing whether coverage degrades at night needs at least a week of polling.
  The function is written and tested; the data is what is missing.
- **Flapping sensors are undercounted** for the same reason.
- **The split between `stuck` and `offline` is a snapshot.** The death cohorts
  suggest it changes in steps as a relay goes down or recovers, so the report
  is worth repeating periodically (phase 7).

## What phase 3 takes from this

1. **The usable fleet is ~3557 bays, not 6324.** Any metric over the 6324 is
   wrong.
2. **`is_reliable` is mandatory** in features and metrics, because the bias of
   omitting it peaks exactly at rush hour.
3. **"No data" has to be a visible state** on the map and in the API: 56 zones
   do not have a single reliable reading, and painting them as free would be a
   lie.

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
