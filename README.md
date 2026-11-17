# GTA VI launch-day Twitch CCU forecaster

Entry for the make.fun bounty **"Forecast GTA VI's launch-day Twitch
viewers hour by hour"** (`gta6-twitch-ccu-forecast-01`).

`main.py` (Python 3.13, standard library only) reads one case per `*.in`
file (schema `makefun/twitch-ccu-case/v1`) from `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one `*.out` file per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). Each output is
`{"hours": [...]}` with 23 rows (hours 2–24 of the release day, UTC),
each `{"hour": h, "q": [q05, q25, q50, q75, q95]}`.

## Method

The case ships comparable past launches, each with its full hourly
Twitch viewer curve around its release day. For every comparable we
compute the log-ratio of each release-day hour to that launch's own
hour-1 count, `r_h = ln(viewers_h / viewers_1)`.

A launch whose hour-1 audience is close in size to this case's hour-1
audience gets a high weight (Gaussian kernel on log size, bandwidth
1.5); far-away launches count for less. For each hour we take the
size-weighted empirical quantiles of `r_h` across comparables and anchor
them on this case's hour-1 print `p`:

```
q_tau(h) = p * exp(quantile_tau(r_h))
```

Where comparables are thin for an hour, the empirical quantiles are
shrunk toward the house baseline (flat median, spread
`0.15 * sqrt(h - 1)`), so the forecast never collapses onto a handful
of launches. Log-ratios are clipped to ±3 and comparables with a
negligible hour-1 audience are ignored, so one late-unlocking launch
cannot stretch the quantiles by orders of magnitude.

The score is the pinball loss on the log scale, a proper scoring rule
for quantiles, so the honest empirical spread is exactly what it
rewards. On synthetic launch-day cases built from a shared shape
family the forecaster scores ~0.14 mean pinball loss vs ~0.27 for the
house baseline.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: no clock, no randomness, all iteration in sorted order.
Runs in well under a second per case.

## License

MIT — see `LICENSE`.
