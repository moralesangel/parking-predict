# Identifier crosswalk: historical <-> real time

Report on the **risk to verify** from phase 1 of the plan: *"the historical
identifiers may not match the real-time Kerbside ids"*.

Measured on **2026-10-22** against the real data.

## Summary

**The risk is confirmed: there is no bay-level join.** The two identifier
spaces are disjoint, and the archive carries no coordinates, so there is no
geometric fallback either.

**But there is a street-segment-level join**, keyed on the street name and the
two streets bounding the segment: it covers **88.0% of the historical events**.
That is the usable path, and it has a design consequence for phase 4 (see §5).

## 1. What identifiers exist

| Dataset | Identifiers | Coordinates |
|---|---|---|
| Historical 2019 (CSV) | `DeviceId`, `BayId`, `StreetMarker` | **no** |
| `on-street-parking-bay-sensors` (real time) | `kerbsideid`, `zone_number` | yes (lat/lon) |
| `on-street-parking-bays` (geometry) | `kerbsideid`, `roadsegmentid` | yes (lat/lon) |
| `on-street-car-park-bay-restrictions` | `bayid`, `deviceid` | no |

All **238 datasets** in the catalogue were swept for fields with bay or zone
identifier names. Result:

- `kerbsideid` appears **only** in the real-time and geometry datasets.
- `bayid` / `deviceid` appear **only** in the restrictions dataset.
- **No dataset in the catalogue contains both `kerbsideid` and
  `bayid`/`deviceid`/`StreetMarker`.** No published lookup table exists.

Reading: `DeviceId`/`BayId` are the sensor-era numbering (2011-2020) and
`kerbsideid` is the later one. The migration was never publicly documented.

## 2. Method

Historical sample: a 150 MB prefix of the ZIP, inflated in streaming ->
**9,413,822 events (22% of the 42.7M)** with **2409 distinct `DeviceId`** and
**2346 distinct `BayId`/`StreetMarker`**.

For each pair of datasets the overlap of identifier sets is measured as a
percentage of the smaller set. As a **control**, a join that must work is
included (`kerbsideid` between real time and geometry): it shows what a genuine
join looks like, and therefore how much overlap is numeric noise.

## 3. Results

```
=== A) historical -> real time, direct ===
  ~   hist.DeviceId          n realtime.kerbsideid        179 common =   7.4 %
  NO  hist.BayId             n realtime.kerbsideid         21 common =   0.9 %
  NO  hist.DeviceId          n bays.kerbsideid             66 common =   2.7 %
  NO  hist.BayId             n bays.kerbsideid              1 common =   0.0 %

=== B) historical -> restrictions (candidate bridge) ===
  OK  hist.DeviceId          n restr.deviceid            1459 common =  60.6 %
  OK  hist.BayId             n restr.bayid               1779 common =  75.8 %

=== C) restrictions -> real time (would close the bridge) ===
  ~   restr.deviceid         n realtime.kerbsideid        314 common =   7.4 %
  NO  restr.bayid            n realtime.kerbsideid         76 common =   1.8 %

=== D) control: geometry -> real time (must join) ===
  OK  realtime.kerbsideid    n bays.kerbsideid           4825 common =  95.5 %
```

Interpretation:

- **(D) is the reference: a real join scores 95.5%.**
- **(A) and (C) are not joins.** The 7.4% overlap of `DeviceId` with
  `kerbsideid` is arithmetic coincidence: `DeviceId` lives in [23,913, 27,596]
  and `kerbsideid` in [5691, 90,812], so the ranges overlap and some numbers
  land in both by chance. Joining on that would silently produce false matches
  - worse than not joining.
- **(B) is a join.** 60.6% and 75.8% confirm that the archive and the
  restrictions share a numbering. The restrictions dataset is in fact a
  leftover from the sensor era.
- **The historical -> restrictions -> real time bridge breaks at the second
  hop** (C). It is unusable.

### The geometric route: ruled out at the source

Joining by coordinates is impossible: the historical CSV has **no latitude or
longitude column at all** (see `docs/data_schema.md` §3). It is not that the
join performs badly - there is nothing to attempt it with at bay level.

One intermediate hypothesis was explored and discarded:
`on-street-parking-bays.kerbsideid` is a text field and contains two values with
a letter suffix (`7568N`, `7570N`) that look exactly like the archive's
`StreetMarker` values (`2374N`, `1581S`). It turned out to be a dead end: only
**2 of 29,053 rows** have that shape, and they are load anomalies rather than an
alternative identifier.

## 4. What does work: the street-segment join

The archive describes location in text: `StreetName`, `BetweenStreet1`,
`BetweenStreet2`, `SideOfStreet`. And
`on-street-parking-bays.roadsegmentdescription` uses the format
`"X Street between A Street and B Street"`.

Normalising both to the same key - uppercase, collapsed whitespace, and the
pair of bounding streets **unordered** (because "A and B" and "B and A" are the
same segment) - the join works:

| Metric | Result |
|---|---|
| Historical segments present in geometry | 225 / 297 = **75.8%** |
| **Historical events covered** | 8,288,708 / 9,413,823 = **88.0%** |
| Bays (`StreetMarker`) covered | 1903 / 2343 = **81.2%** |
| Ambiguous segments (-> more than one `roadsegmentid`) | **1** |

Event coverage (88.0%) is higher than segment coverage (75.8%): the segments
that fail to join are mostly peripheral and low-traffic. The six with the most
lost events are Cobden St, William St (between Victoria and Walsh), Spring St
(between Lonsdale and Victoria Pde), Dudley St, O'Connell St and Rosslyn St -
CBD-edge areas, probably withdrawn from the current inventory or renamed.

With that join, every historical bay inherits a `roadsegmentid` and therefore
coordinates: the archive becomes located in space and comparable with the
real-time feed **at segment level, never at individual bay level**.

## 5. Consequences for the plan

**Phase 4's spatial unit must be the street segment (`roadsegmentid`), not an
H3 cell.** The plan left the choice open ("H3 cells … or street segments.
Justify the choice"). The justification is this: the segment is the only level
at which the archive and the real-time feed reconcile. With H3 the cells would
have to be assigned from coordinates, and the archive has none - the 42.7M
events would be lost to any spatial feature. The segment also matches the
driver's actual decision ("is there space on this block?") better than an
arbitrary hexagon.

**The 2019 archive is not suitable for training the production model**, for two
reasons independent of the crosswalk:

1. **Temporal drift.** 2019 is pre-COVID; the published series stops in May 2020
   and there is no data from 2021 onwards. Melbourne CBD occupancy patterns from
   seven years ago do not describe today's.
2. **The sensor fleet changed.** The renumbering (`DeviceId` -> `kerbsideid`)
   accompanies a fleet renewal: there is no guarantee a 2019 bay exists today,
   or the reverse.

The alternative the plan itself anticipated is therefore adopted: **the archive
is used to build and validate the pipeline** (42.7M real events are enough to
construct features, test the event-to-time-series conversion and compare
baselines), **and the production model will be trained on whatever history the
poller accumulates.**

That carries an honest cost worth stating up front: until the poller has
accumulated weeks of data, phase 5's results are results *about 2019*.
`docs/results.md` has to distinguish explicitly between the two.

## 6. Reproducibility

The crosswalk is not frozen as a data file: it is recomputed from the raw
datasets. The figures in this report come from `scripts/crosswalk_report.py`,
which re-downloads the inputs from the API and reprints the tables in §3 and §4.

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
