# Weather construction and versioning

## Which weather does this release use?

The public dataset is **`beijing-2023-noaa-era5-v1`**. The V54 business mechanisms,
calendar, synthetic forecast model, and scoring are unchanged. Package version
`0.54.1` identifies this distribution; the weather input has its own identity.

The paper's experiments used a frozen Visual Crossing-derived input. That input
is not distributed here. The paper's results remain results of that original
configuration, not measurements on the replacement weather. Matching the seed
alone is insufficient to make results from different weather inputs comparable.

## Sources

All source snapshots are in [`data_sources/weather`](../data_sources/weather).
[`sources.json`](../data_sources/weather/sources.json) records URLs, request
parameters, checksums, and source terms.

- **NOAA ISD:** station `54511099999`, Beijing Capital Airport, hourly/synoptic
  observations for 2023. The 2022 file supplies observations before 00:00 UTC
  that belong to the first local day of 2023. Only local dates in 2023 are used.
- **ERA5 through Open-Meteo:** daily mean temperature, precipitation, snowfall,
  and cloud cover at requested coordinates 39.9042°N, 116.4074°E, using
  `models=era5` and `timezone=Asia/Shanghai`. The response identifies the actual
  grid coordinates. This is reanalysis, not a station observation.
- **ERA5-Land through Open-Meteo:** hourly snow depth at the same requested
  location. The ERA5 snow-depth field in the retrieved response is entirely
  missing, so it is not used or interpreted as zero. ERA5-Land supplies complete
  hourly snow depth for this snapshot.

Weather dates are from **2023**, replayed by month and day on the benchmark's
**2026 calendar**. The default 200 decision days cover April 1–October 17, with
settlement through October 31. The data file contains all 365 days; the impact
check focuses on this default 214-day window.

## Deterministic construction

1. **Observations.** Decode NOAA missing sentinels and quality flags. Accept
   passed-check and accepted/edited values (`0`, `1`, `4`, `5`, `9`, `A`, `I`,
   `M`, `P`, `R`, `U`; also `C` for temperature). Reject suspect, erroneous,
   malformed, and out-of-range values. Times are converted from UTC to local
   UTC+08 before aggregation.
2. **Temperature.** Average valid observations within each hour, then average
   the valid hourly means in the local day. This prevents differing report
   frequencies from weighting hours differently. ERA5 daily temperature is a
   fallback only when a day has no valid NOAA temperature observations. The
   provenance file records valid-hour coverage; partial coverage is not claimed
   to be a complete 24-hour station record.
3. **Rain.** Decode `AA1`–`AA4`. Require a valid numeric amount, an explicit
   accumulation period of 1–98 hours, and acceptable quality. Accumulation
   markers `3`/`4` with a valid amount are retained; inaccurate, deleted/missing
   period, and incomplete amounts are excluded. Within each report hour,
   select the shortest valid period. If equal-shortest reports disagree, omit
   that hour and record the conflict. Any positive selected amount makes its
   **local ending day** wet. If a day has no usable report, use ERA5 precipitation.
4. **Snow.** A positive ERA5 daily snowfall or a positive daily maximum of
   ERA5-Land snow depth gives the `snowy` category, taking priority over rain.
   Missing snow-depth values fail validation rather than becoming zeros.
5. **Sky category.** Otherwise a wet day is `rainy`; a dry day with ERA5 mean
   cloud cover at most 20% is `sunny`; the remaining days are `cloudy`.
6. **Runtime fields.** Retain the V54 calendar mapping and its synthetic daily
   temperature-span rule. The span is the monthly base
   `[10,11,12,12,11,10,8,8,10,11,11,10]` °C plus the category adjustment
   (`sunny: +2`, `cloudy: -1`, `rainy: -3`, `snowy: -2`); low/high are the mean
   minus/plus half the span. These are simulation inputs, not observed extremes.

Rain classification is a **report-ending-day approximation**, not a reconstruction
of exact precipitation times or daily precipitation totals. Accumulation
intervals may overlap or cross midnight; one valid report does not establish
full-day observation coverage. No weather values are patched to match the paper
input. Winter classification is supported by the open sources but is not
claimed to reproduce the original supplier's snow labels.

The builder never reads the original supplier data. Per-day construction details
are in [`daily_provenance.jsonl`](../data_sources/weather/daily_provenance.jsonl).

## Rebuild or verify

From the repository root:

```bash
# Offline check; does not rewrite the shipped data.
uv run python -m outfitter.public_weather --check

# Rebuild only weather and its provenance, preserving all other business inputs.
uv run python -m outfitter.public_weather
```

The committed source snapshots are the reproducibility reference. To retrieve
upstream inputs into a **new** directory:

```bash
uv run python scripts/download_weather_sources.py --output-dir /tmp/outfitter-weather-new
```

Fresh downloads can differ because upstream data are revised. Downloading new
inputs does not overwrite the release or authorize reusing its dataset version.
Before publishing changed inputs, give them a new weather dataset identity and
validate them. The downloader verifies an existing manifest instead of silently
refreshing it. Free Open-Meteo API access is subject to its service terms;
running and rebuilding from the committed snapshots requires no API calls.

## Input alignment with the paper weather

The implemented pipeline was compared locally with the frozen paper input:

| Measure | 200 decision days | 214 days including settlement |
|---|---:|---:|
| Identical weather categories | 190/200 (95.0%) | 202/214 (94.4%) |
| Mean absolute difference in daily mean temperature | 0.09375 °C | 0.09533 °C |
| Days with temperature difference at most 0.2 °C | 189/200 | 199/214 |
| Maximum absolute temperature difference | 2.70 °C | 2.70 °C |
| Public sunny / cloudy / rainy counts | 50 / 106 / 44 | 60 / 109 / 45 |

The decision window has five rain/non-rain and five sunny/cloudy differences.
These dates were also used in the earlier exploration of construction rules,
so this is an alignment description, not independent held-out validation.
The 214-day window includes the first 200 days. Similar inputs do not establish
unchanged model scores or rankings: weather also affects public forecasts,
requests, cancellations, equipment returns, and later resource allocations.

## Paired business-impact check

We ran the unchanged Calendar policy and the existing `decline_reservations`
policy on seeds 42 and 73, once with each weather input. The latter declines
advance bookings and focuses on walk-in business; it is a sensitivity check,
not an additional paper baseline. Within each policy/seed pair, the code,
configuration, business inputs, and random seed are identical except for
weather and its provenance. All eight runs completed 200 decision days and 14
settlement days, with no bankruptcy, no model calls, and reconciled cash flows.

| Policy | Seed | Original weather ΔV (CNY) | Public weather ΔV (CNY) | Change (CNY) |
|---|---:|---:|---:|---:|
| Calendar | 42 | 1,103.57 | 1,062.02 | −41.55 |
| Calendar | 73 | 1,135.45 | 1,083.45 | −52.00 |
| Decline reservations | 42 | −1,265.58 | −1,245.70 | +19.88 |
| Decline reservations | 73 | −1,342.21 | −1,392.53 | −50.32 |

Calendar's value creation decreased by about 3.8% and 4.6%. It remained
profitable on both seeds, while the walk-in-focused policy continued to lose
value. This broad pattern was preserved for these rules, but their trajectories
were not identical. For example, Calendar's compensation rose from CNY 173.46 to
228.96 on seed 42 and from CNY 353.24 to 364.80 on seed 73; pickup counts changed
by −1 and +1. Generated reservation requests changed by +5 and +13 across the
two seeds, respectively.

The differences are small enough to proceed with this separately versioned
release, not evidence that weather has no effect. This check does not establish
unchanged LLM scores or rankings. The aggregate results are available in
[`weather_impact_summary.json`](weather_impact_summary.json); no original weather
records or trajectories are included.

## Versions, historical runs, and licenses

New runs record `weather_dataset_id` in the controller manifest, freeze their
runtime input files, and retain the existing integrity checks. The default
configuration uses the public weather. Historical runs keep their original
data and implementation; they are not converted by replacing files or hashes.

- [NOAA ISD data and usage notice](https://registry.opendata.aws/noaa-isd/).
- [NOAA ISD field documentation](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf).
- [Weather data by Open-Meteo](https://open-meteo.com/), based on Copernicus ERA5
  and ERA5-Land; [historical API documentation](https://open-meteo.com/en/docs/historical-weather-api).
- Open-Meteo data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  [Service terms](https://open-meteo.com/en/terms) apply to API use separately.

This project transforms and combines the inputs. The benchmark weather is not
an unmodified NOAA product and does not imply provider endorsement. See
[DATA.md](DATA.md) for the distinction between code and data rights.
