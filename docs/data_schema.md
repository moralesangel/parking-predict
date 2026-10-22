# Data schema

Verified against the live
[City of Melbourne Open Data](https://data.melbourne.vic.gov.au/) API on
**2026-10-22**. Every field and endpoint in this document was checked by
running the queries; none is assumed (plan rule 2). Data under CC BY 4.0.

The API is an **Opendatasoft Explore v2.1** instance:

```
https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/{dataset_id}
```

## 1. Real time: `on-street-parking-bay-sensors`

In-ground sensors in CBD bays. **6324 records**, one per bay. The dataset is
**overwritten** on every update: it is a snapshot of the current state, not a
time series. Preserving the history is exactly the poller's job.

### Fields

| Field | API type | Verified semantics |
|---|---|---|
| `kerbsideid` | `int` | Bay identifier. **Unique key**: 6324 distinct values in 6324 rows, 0 duplicates, 0 nulls. Observed range [5691, 90812]. |
| `status_description` | `text` | Only two values: `Present` and `Unoccupied`. 0 nulls. See §1.1. |
| `status_timestamp` | `datetime` (UTC) | Instant of the **last state change** for that bay. 0 nulls. |
| `lastupdated` | `datetime` (UTC) | Instant of the **last feed batch** the bay appeared in. 0 nulls. See §1.2. |
| `zone_number` | `int` | Restriction zone (`parkingzone` in other datasets). 536 distinct zones, range [7001, 7996], **520 nulls (8.2%)**. |
| `location` | `geo_point_2d` | Bay coordinates. 0 nulls. The format depends on the endpoint, see §1.3. |

### 1.1 Naming trap: `Present` means occupied

`status_description` describes the **car**, not the bay:

- `Present` -> a car is there -> the bay is **OCCUPIED** (3581 rows, 56.7%)
- `Unoccupied` -> no car -> the bay is **FREE** (2743 rows, 43.3%)

That 56.7% is the raw split of the 6324 rows at one instant, not the fleet's
occupancy: it includes offline and stuck sensors. The usable figure is computed
over reliable readings only and moves with the time of day; see
[`data_quality.md`](data_quality.md).

This is an easy mistake and a silent one: it inverts the model's target without
anything failing. Ingestion normalises to an explicit `is_occupied` boolean and
the raw string never reaches the features.

### 1.2 The two timestamps are not the same thing

This is phase 1's main finding and it shapes all of phase 2.

**`lastupdated` is a batch stamp, not a per-sensor one.** Across the 6324
records there are only **49 distinct values**, and one of them covers 5264 bays
(`2026-10-22T09:26:50Z` at measurement time). Publication goes through network
relays: each cycle rewrites the dataset and every bay that reported receives the
**same** `lastupdated`. A bay that stops reporting stays **frozen** with the
stamp of the last batch it appeared in.

That turns offline-sensor detection into an exact rule rather than a threshold
heuristic:

```
sensor_live  <=>  lastupdated == max(lastupdated) in the snapshot
```

The 49 distinct values are "death cohorts": groups of sensors that dropped out
in the same cycle (83 bays on 2025-10-30, 79 on 2026-01-15, 70 on 2026-07-22,
and so on). Failing in blocks points at whole relays rather than individual
sensors - a useful distinction for the quality report.

Age of `lastupdated` measured on 2026-10-22:

| Age | Bays | % |
|---|---|---|
| <= 15 min | 5264 | 83.2% |
| <= 24 h | 5264 | 83.2% |
| <= 7 d | 5284 | 83.6% |
| <= 30 d | 5345 | 84.5% |
| **> 30 d** | **979** | **15.5%** |

**The usable fleet is ~5264 bays, not 6324.** The 979 older than 30 days are
decommissioned or permanently dead sensors the dataset never purges. Any
occupancy metric computed over all 6324 is biased.

**`status_timestamp` is per bay** and marks the last state change. Median 4.0 h;
only 8.8% changed in the last 15 min. Its age does **not** indicate a fault: a
bay legitimately occupied all night has an old `status_timestamp` and a fresh
`lastupdated`. Confusing the two is the mistake phase 2 has to avoid - and the
reason they are documented separately here.

| Age of `status_timestamp` | Bays | % |
|---|---|---|
| <= 15 min | 558 | 8.8% |
| <= 1 h | 1531 | 24.2% |
| <= 6 h | 3328 | 52.6% |
| <= 24 h | 3556 | 56.2% |
| > 30 d | 2427 | 38.4% |

The source itself endorses watching `lastupdated`; its description says:

> This dataset is updated through network relays and can be disrupted causing
> delays in data updates. Users of this data are encouraged to monitor the
> dataset's last updated timestamp (Lastupdated column) to check for delays in
> the data being updated.

### 1.3 Endpoint chosen for the poller

Three ways to read the dataset, with differences that matter:

| Endpoint | Requests per snapshot | `location` | Verdict |
|---|---|---|---|
| `/records?limit=100` | 64 (paginated, 100/req cap) | `{"lon":…,"lat":…}` | Rejected: 64 requests per snapshot |
| `/exports/parquet?limit=-1` | 1 (204 KB) | **`Binary` (WKB)** | Rejected: forces geometry decoding |
| **`/exports/json?limit=-1`** | **1 (1.5 MB)** | `{"lon":…,"lat":…}` | **Chosen** |

`/exports/json?limit=-1` returns all 6324 records in **a single request** with
coordinates already as lon/lat. At one snapshot every 5 min that is 288 requests
a day, comfortable for any reasonable limit.

The Parquet export delivering `location` as `Binary` (WKB) is the reason not to
use it, even though local storage is Parquet.

Sensor bounding box (measured): lon [144.93644, 144.98861], lat
[-37.84824, -37.79491] - roughly 4.6 x 5.9 km of central Melbourne.

## 2. Supporting datasets (API-queryable)

### `on-street-parking-bays` - bay geometry

29,053 rows. Fields: `roadsegmentid` (int), `kerbsideid` (**text**),
`roadsegmentdescription` (text), `latitude`, `longitude`, `lastupdated`,
`location`.

Three verified warnings:

- `kerbsideid` is **text here** and `int` in the sensor dataset. It has to be
  cast before joining.
- **23,981 of 29,053 rows (82.5%) have an empty `kerbsideid`.** Only 5070 carry
  one, consistent with the ~5264 bays with a live sensor: most of the bay
  inventory is not instrumented.
- Two rows (`7568N`, `7570N`) carry a letter suffix that breaks the numeric
  format. They are one-off anomalies, not a pattern.

`roadsegmentdescription` follows the format `"X Street between A Street and B
Street"`, which is the key to joining against the historical archive (see
[`id_crosswalk.md`](id_crosswalk.md)). 1035 of the 29,053 descriptions do not
match that pattern.

It provides `roadsegmentid`, the natural "street segment" unit for phase 4.

### `on-street-car-park-bay-restrictions` - per-bay restrictions

4263 rows. Keys **`bayid`** and **`deviceid`** (both int), followed by six
repeating restriction blocks: `description1..6`, `typedesc1..6`,
`duration1..6`, `starttime1..6`, `endtime1..6`, `fromday1..6`, `today1..6`,
`effectiveonph1..6`, `disabilityext1..6`, `exemption1..6`.

Real example:
`{bayid: 9003, deviceid: 29960, description1: "4P MTR M-F 10:00-18:30",
typedesc1: "4P Meter", duration1: 240}`.

**It does not contain `kerbsideid`**, and that is the heart of the crosswalk
problem.

### Others

- `sign-plates-located-in-each-parking-zone` (1821 rows): `parkingzone` plus
  `restriction_days`, `time_restrictions_start/finish`, `restriction_display`.
  Joins against `zone_number` from the real-time feed.
- `parking-zones-linked-to-street-segments`: `parkingzone` -> street segments.

## 3. Yearly sensor archive

### Not queryable through the API

All ten historical datasets (`on-street-car-parking-sensor-data-2011` …
`-2019` and `-2020-jan-may`) have **`has_records: false` and
`records_count: 0`**: the API does not serve their rows. They only exist as
archive ZIPs linked from the dataset's HTML description. For 2019:

```
https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip
```

717,085,379 bytes (717 MB), 42.7 million rows, a single CSV inside:
`On-street_Car_Parking_Sensor_Data_-_2019.csv`.

S3 answers with `Accept-Ranges: bytes`, so the schema was verified with a
`Range` request that inflated only the prefix of the deflate stream, without
downloading the whole file. The same mechanism gives the downloader resumption
for free.

Note: the series ends in May 2020. No archive has been published from 2021
onwards, even though the real-time feed is still live.

### Fields (20, verified against the real CSV)

| Field | Example | Notes |
|---|---|---|
| `DeviceId` | `23915` | Sensor identifier |
| `ArrivalTime` | `11/03/2019 03:00:56 AM` | **US format `MM/DD/YYYY hh:mm:ss AM/PM`**, Melbourne local time |
| `DepartureTime` | `11/03/2019 03:07:26 AM` | Same format |
| `DurationMinutes` | `7` | Rounded: 6 min 30 s is published as `7` |
| `StreetMarker` | `2374N` | Physical kerb marker: digits plus an orientation letter |
| `SignPlateID` | `287` | Often empty |
| `Sign` | `1P MTR M-SAT 7:30-18:30` | Restriction as free text; often empty |
| `AreaName` | `Windsor`, `Banks` | Area name |
| `StreetId` / `StreetName` | `123` / `BOURKE STREET` | |
| `BetweenStreet1ID` / `BetweenStreet1` | `1221` / `RUSSELL STREET` | Segment between two streets |
| `BetweenStreet2ID` / `BetweenStreet2` | `647` / `EXHIBITION STREET` | |
| `SideOfStreet` / `SideOfStreetCode` / `SideName` | `3` / `N` / `North` | |
| `BayId` | `1753` | Bay identifier |
| `InViolation` | `false` | Overstayed the restriction |
| `VehiclePresent` | `false` | |

Critical points when parsing:

- **US date format.** `11/03/2019` is 3 November, not 11 March. Parse with an
  explicit `%m/%d/%Y %I:%M:%S %p`, never with automatic inference: inference
  that succeeds on the first records will fail silently on days <= 12.
- **No timezone.** It is Melbourne local time (`Australia/Melbourne`) with
  daylight saving: the transitions contain ambiguous and non-existent instants.
  Converting to UTC requires an explicit decision about them.
- **It is an interval schema, not snapshots.** Each row is one interval
  (arrival -> departure). Reconstructing occupancy at a given instant means
  going from events to a time series: phase 4's job.
- **`VehiclePresent` decides what each interval is**, and misreading it ruins
  the dataset. See [`dataset.md`](dataset.md).
- **No coordinates.** No latitude or longitude column at all. This is the core
  of the crosswalk problem.
- The file is **sorted by `DeviceId`**, not by time. A prefix of the file is not
  a representative temporal sample but a subset of sensors: relevant when
  sampling for EDA.

## 4. The historical <-> real-time crosswalk

Result in its own report: [`docs/id_crosswalk.md`](id_crosswalk.md).

## Attribution

Data from [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
