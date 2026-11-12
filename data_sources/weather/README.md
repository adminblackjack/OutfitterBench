# Open weather inputs

These snapshots reproduce `beijing-2023-noaa-era5-v1` offline. The simulator uses
the built `src/outfitter/runtime_data/weather.jsonl`, not live weather services.

- `noaa-54511099999-2022.csv.gz` and `noaa-54511099999-2023.csv.gz`: complete
  NOAA ISD station CSVs, stored with lossless gzip compression. Local 2023 data
  are selected after timezone conversion; the preceding file covers the first
  local day's UTC boundary.
- `open-meteo-era5-beijing-2023.json`: the original daily ERA5 response. Its
  hourly snow-depth field is all-null and is not used.
- `open-meteo-era5-land-snow-beijing-2023.json`: the original hourly ERA5-Land
  snow-depth response, used for ground-snow classification.
- `sources.json`: source URLs, request parameters, file hashes, and terms.
- `daily_provenance.jsonl`: generated per-day coverage, selected precipitation
  reports, fallback choices, and classification inputs.

**Attribution:** NOAA Integrated Surface Database, accessed November 12, 2026
via the [NOAA Open Data Dissemination mirror](https://registry.opendata.aws/noaa-isd/).
Weather data by [Open-Meteo](https://open-meteo.com/), based on Copernicus ERA5
and ERA5-Land, accessed November 12, 2026, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The generated weather and provenance transform and combine these sources;
they are not unmodified NOAA observations or endorsed provider products.
The combined weather data are shared under CC BY 4.0 with these source notices.
The project's Apache-2.0 code license does not replace data licenses.

See [the construction method](../../docs/WEATHER.md). Source refreshes are not
silently substituted into this frozen dataset.
