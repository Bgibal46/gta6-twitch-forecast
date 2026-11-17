# GTA VI launch-day Twitch forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bgibal46/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bgibal46/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

An entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on make.fun: a program that reads a case's Twitch viewer counts up to the end of a release day's first hour and forecasts five quantiles of the concurrent viewers for each of hours 2 to 24.

## How it works

`main.py` (Python 3.13, standard library only) learns the shape of a launch day from the case's own comparables: past big launches' hourly curves, each expressed as log ratios to its own hour-1 print. It keeps undistorted launches, weights rolling-midnight unlocks and launches of a similar hour-1 size, fits a ridge-shrunk line of the log ratio on log hour-1 size (small first hours grow far more over the day than huge ones), and takes asymmetric spreads from the fit's residuals. Anything unexpected falls back to a persistence forecast, so a run always writes a valid file.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one `*.out` forecast per case: 23 rows, hours 2 to 24, five quantiles each. Deterministic: two runs write identical bytes.

## License

MIT. See [LICENSE](LICENSE).
