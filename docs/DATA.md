# Data provenance and licensing

- **Products:** selected Taobao listing titles, specifications, and purchase-price
  information, mapped to benchmark service profiles and equipment capacities.
  Some lifecycle parameters use offline estimates, not measured product
  guarantees. Third-party listing material is not automatically covered by the
  repository's code license.
- **Weather:** `beijing-2023-noaa-era5-v1`, constructed from NOAA ISD
  observations at Beijing Capital Airport and Open-Meteo ERA5/ERA5-Land inputs.
  The source year is 2023; values are replayed by month and day on the benchmark
  calendar. Daily categories, synthetic temperature extremes, and public
  forecasts are transformations, not unaltered station observations.
- **Simulator parameters:** customer preferences, demand rules, operational
  costs, and initial conditions are benchmark parameters, not real customer
  transactions or personally identifying customer records.

Runtime inputs are frozen under `src/outfitter/runtime_data/`; metadata records
their provenance and integrity hashes. A run takes its own data snapshot.

## Weather terms and attribution

NOAA ISD inputs were accessed on November 12, 2026 via the
[NOAA Open Data Dissemination mirror](https://registry.opendata.aws/noaa-isd/).
NOAA makes these data publicly available for use and requests attribution.
Modified data must not be represented as unmodified NOAA products; no provider
endorsement is implied.

Weather data by [Open-Meteo](https://open-meteo.com/), based on Copernicus ERA5
and ERA5-Land, accessed November 12, 2026, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the source,
link the license, and identify modifications when sharing these data.
The combined benchmark weather dataset is distributed under CC BY 4.0 with
these source notices. API service conditions apply separately to fresh
downloads; normal runs and offline rebuilds use the included snapshots.

The paper used a different Visual Crossing-derived weather input. It is not
included in this distribution. Historical paper scores are not relabeled as
scores on the public replacement.

## Before public distribution

Apache-2.0 applies to the project's code, not automatically to third-party
listing material. The author must confirm the intended redistribution of
product-source material before changing repository visibility. Weather
replacement does not resolve unrelated third-party rights.

Keep private original inputs, old Git history, model credentials, and run
snapshots out of this repository. Different weather is a versioned data change;
historical results continue to refer to their original frozen inputs.

This research simulator is not a source of weather advice or commercial
forecasts. Its inputs and outputs are not guaranteed accurate, complete, or
suitable for operational decisions.
