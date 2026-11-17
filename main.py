#!/usr/bin/env python3
"""GTA VI launch-day Twitch CCU forecaster.

Entry for make.fun bounty "Forecast GTA VI's launch-day Twitch viewers
hour by hour" (gta6-twitch-ccu-forecast-01), harness program/v1.

Reads one case per *.in file (schema "makefun/twitch-ccu-case/v1") from
$MAKEFUN_INPUT_DIR (else /input/) and writes one *.out file per case to
$MAKEFUN_OUTPUT_DIR (else /output/). Each output is {"hours": [...]} with
23 rows (hours 2..24), each {"hour": h, "q": [q05, q25, q50, q75, q95]}.

Method
------
The case ships a set of comparable past launches ("comparables"), each with
its full hourly Twitch viewer curve around its release day. For every
comparable we compute the log-ratio of each release-day hour to that
launch's own hour-1 count:

    r_h = ln(viewers_h / viewers_1),  h = 2..24

A launch whose hour-1 audience is close in size to this case's hour-1
audience gets a high weight (Gaussian kernel on log size, bandwidth 1.0);
far-away launches count for less. For each hour we then take the
size-weighted empirical quantiles of r_h across comparables and anchor
them on this case's hour-1 print p:

    q_tau(h) = p * exp(quantile_tau(r_h))

Where comparables are thin for an hour we shrink the empirical quantiles
toward the house baseline (flat median, spread 0.15*sqrt(h-1)) so the
forecast never collapses onto a handful of launches. The pinball loss on
the log scale is a proper scoring rule for quantiles, so the honest
empirical spread is exactly what it rewards.

The program is pure standard library, deterministic (no clock, no
randomness, all iteration in sorted order), and runs in milliseconds.
"""

import json
import math
import os
import sys

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at LEVELS.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HOURS = list(range(2, 25))

# Gaussian kernel bandwidth on log hour-1 size (natural log units).
SIZE_BANDWIDTH = 1.5
# Effective comparable count at which the empirical quantiles get full
# weight; below it we blend toward the house baseline.
FULL_WEIGHT_COUNT = 12.0
# House baseline log-spread growth per hour.
BASELINE_SPREAD = 0.15
# Robustness guards: ignore comparables whose hour-1 audience is
# negligible (their ratios are noise), and clip log-ratios so one
# late-unlocking launch cannot stretch the quantiles by orders of
# magnitude.
MIN_ANCHOR_VIEWERS = 100.0
MAX_LOG_RATIO = 3.0

MAX_VIEWERS = 1_000_000_000.0


def parse_time(text):
    """Parse an ISO 8601 timestamp (with 'Z' or an offset) to epoch seconds."""
    from datetime import datetime, timezone
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of [(value, weight), ...] at `level`.

    Linear interpolation on the cumulative-weight scale. Deterministic:
    pairs are sorted by value first.
    """
    ordered = sorted(pairs, key=lambda vw: vw[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    target = level * total
    acc = 0.0
    prev_value = None
    prev_acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            if prev_value is None or acc == prev_acc:
                return value
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_value + frac * (value - prev_value)
        prev_value = value
        prev_acc = acc
    return ordered[-1][0]


def baseline_quantiles(p, hour):
    """House baseline: flat median, spread widening over the day."""
    spread = BASELINE_SPREAD * math.sqrt(hour - 1)
    return [p * math.exp(z * spread) for z in Z]


def release_day_series(comparable):
    """Hour -> viewers for a comparable's release day (hours 1..24).

    Hours are taken from the comparable's `hourly` points whose `from`
    falls inside [day_start, day_start + 24h). Entries with no usable
    viewer count are skipped.
    """
    try:
        start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    series = {}
    for point in comparable.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers <= 0:
            continue
        offset = (when - start) / 3600.0
        hour = int(math.floor(offset)) + 1
        if 1 <= hour <= 24 and hour not in series:
            series[hour] = viewers
    return series


def hour1_viewers(case):
    """This case's hour-1 print, with fallbacks if it is missing."""
    known = case.get("known") or {}
    try:
        value = float(known["viewers"])
        if value > 0:
            return value
    except (KeyError, TypeError, ValueError):
        pass
    hourly = case.get("history_hourly") or []
    for point in reversed(hourly):
        try:
            value = float(point["viewers"])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            continue
    return 1.0


def forecast_case(case):
    """Return the {"hours": [...]} forecast object for one case."""
    p = hour1_viewers(case)
    log_p = math.log(p)

    # Log-ratio curves of the comparables, weighted by hour-1 size.
    # curves[h] = [(r_h, weight), ...] for h = 2..24.
    curves = {h: [] for h in HOURS}
    for comparable in case.get("comparables") or []:
        series = release_day_series(comparable)
        anchor = series.get(1)
        if not anchor or anchor < MIN_ANCHOR_VIEWERS:
            continue
        log_anchor = math.log(anchor)
        weight = math.exp(-0.5 * ((log_anchor - log_p) / SIZE_BANDWIDTH) ** 2)
        if weight <= 0:
            continue
        for hour in HOURS:
            viewers = series.get(hour)
            if viewers and viewers > 0:
                ratio = math.log(viewers / anchor)
                ratio = max(-MAX_LOG_RATIO, min(MAX_LOG_RATIO, ratio))
                curves[hour].append((ratio, weight))

    rows = []
    for hour in HOURS:
        base = baseline_quantiles(p, hour)
        pairs = curves[hour]
        if not pairs:
            quantiles = base
        else:
            empirical = []
            for level in LEVELS:
                value = weighted_quantile(pairs, level)
                # value is a log-ratio; convert to an absolute quantile.
                empirical.append(
                    p * math.exp(value) if value is not None else p)
            # Blend toward the baseline where comparables are thin.
            # Effective count: sum of weights, capped at FULL_WEIGHT_COUNT.
            eff = min(sum(w for _, w in pairs), FULL_WEIGHT_COUNT)
            alpha = eff / FULL_WEIGHT_COUNT
            quantiles = []
            for q_emp, q_base in zip(empirical, base):
                blended = math.exp(
                    alpha * math.log(q_emp) + (1.0 - alpha) * math.log(q_base)
                )
                quantiles.append(blended)
        # Enforce the output contract: finite, > 0, <= 1e9, non-decreasing.
        clean = []
        prev = 0.0
        for value in quantiles:
            if not math.isfinite(value) or value <= 0:
                value = prev if prev > 0 else p
            value = min(value, MAX_VIEWERS)
            value = max(value, prev)
            value = round(value, 2)
            clean.append(value)
            prev = value
        rows.append({"hour": hour, "q": clean})
    return {"hours": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(in_dir) if name.endswith(".in")
    )
    for name in names:
        in_path = os.path.join(in_dir, name)
        stem = name[:-3]
        out_path = os.path.join(out_dir, stem + ".out")
        with open(in_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"))
        if len(text.encode("utf-8")) > 65536:
            # Contract cap: fall back to the house baseline at full
            # precision would already be tiny; this should not happen,
            # but never emit an oversize file.
            text = json.dumps(result, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
