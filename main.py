"""Launch-day Twitch viewership forecaster (gta6-twitch-ccu-forecast-01).

The program forecasts, for each of hours 2..24 of a game's release day (UTC),
five quantiles of the Twitch category's concurrent viewers, anchored on the
known hour-1 print. It learns the shape of a launch day from the case's own
comparables: past big launches' hourly curves, each expressed as log ratios to
its own hour-1 print.

Model, per forecast hour h:
  - Keep comparables whose release-day curve is usable (an hour-1 print of at
    least 1,000 viewers and at least 12 scored hours) and whose launch was not
    distorted by early access (the case notes mark those "Tier:
    early-access-contaminated" or "Tier: weak"; unmarked launches are kept).
  - Weight each comparable: x3 when its unlock rolled at midnight local time
    like this contest's launches, x2 when its hour-1 print is within a factor
    of e of the case's hour-1 print.
  - Fit a ridge-shrunk line of the log ratio on log hour-1 size: small hour-1
    launches grow far more over the day than huge ones, whose first hour
    already holds most regions. The slope is clamped, and the size adjustment
    is capped, so an unprecedented launch never extrapolates wildly.
  - Spread comes from the fit's residuals: asymmetric log sigmas from the
    weighted 25th/75th residual quantiles, so the band widens more upward than
    downward when launches surprise to the upside, as they do in the evening.
  - Medians and spreads are smoothed over neighbouring hours; quantiles are
    the hour-1 print times exp(median + z_tau * sigma), with z from the case's
    own levels. Anything unexpected (no usable comparable, a malformed case)
    falls back to a persistence forecast, so a run always writes a valid file.

Standard library only, deterministic: comparables are processed in name order
and every aggregate is a fixed-order computation, so two runs write identical
bytes. Runs in well under a second per case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves the repository off sys.path; nothing beside this file
# is imported, so no path fix is needed.

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# A comparable needs a meaningful hour-1 print: tiny categories are sampled
# sparsely, so their ratios are noise.
MIN_HOUR1_VIEWERS = 1000
MIN_SCORED_HOURS = 12
# Ridge prior on the size slope, and the caps that keep extrapolation honest.
RIDGE_PRIOR = 5.0
MAX_SLOPE = 0.6
MAX_SIZE_ADJUSTMENT = 0.7
# Spread scale and floor, in log units.
SPREAD_SCALE = 0.75
SPREAD_FLOOR = 0.12
# Persistence fallback spread (log units per square-root hour).
FALLBACK_SIGMA = 0.15

TIER_EXCLUDED = ("early-access-contaminated", "weak")


def normal_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def round2_half_up(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def comparable_curves(case):
    """Release-day log-ratio curves of the case's comparables, in name order."""
    curves = []
    for comp in sorted(case.get("comparables", []), key=lambda c: str(c.get("name", ""))):
        unlock = str(comp.get("unlock", ""))
        tier = re.search(r"Tier:\s*([a-z\-]+)", unlock)
        if tier and tier.group(1) in TIER_EXCLUDED:
            continue
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, ValueError):
            continue
        by_hour = {}
        for point in comp.get("hourly", []):
            try:
                at = parse_time(point["from"])
                viewers = point["viewers"]
            except (KeyError, ValueError, TypeError):
                continue
            if viewers is None or viewers <= 0:
                continue
            hour = int((at - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                by_hour[hour] = viewers
        hour1 = by_hour.get(1)
        if hour1 is None or hour1 < MIN_HOUR1_VIEWERS:
            continue
        ratios = {h: math.log(v / hour1) for h, v in by_hour.items() if h >= 2}
        if len(ratios) < MIN_SCORED_HOURS:
            continue
        curves.append({
            "log_v1": math.log(hour1),
            "ratios": ratios,
            "rolling": "midnight local" in unlock.lower(),
        })
    return curves


def weighted_quantile(pairs, level):
    """Weighted quantile of (value, weight) pairs; pairs need not be sorted."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][0]


def smooth(values):
    """Three-hour moving average over hours 2..24 (edges use what exists)."""
    out = {}
    for h in range(2, 25):
        near = [values[hh] for hh in (h - 1, h, h + 1) if hh in values]
        out[h] = sum(near) / len(near)
    return out


def forecast_case(case):
    levels = [float(x) for x in case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])]
    z_values = [normal_ppf(p) for p in levels]
    known = [k for k in case.get("known", []) if k.get("hour") == 1]
    if known:
        hour1 = max(1.0, float(known[0]["viewers"]))
    else:
        history = case.get("history_hourly") or case.get("history") or []
        hour1 = max(1.0, float(history[-1]["viewers"])) if history else 1.0
    hours = [int(r["hour"]) for r in case.get("forecast", [])] or list(range(2, 25))
    curves = comparable_curves(case)
    log_p1 = math.log(hour1)

    medians, sigma_lo, sigma_hi = {}, {}, {}
    for h in range(2, 25):
        usable = [c for c in curves if h in c["ratios"]]
        if not usable:
            medians[h] = 0.0
            sigma_lo[h] = sigma_hi[h] = FALLBACK_SIGMA * math.sqrt(h - 1)
            continue
        weights = []
        for comp in usable:
            w = 3.0 if comp["rolling"] else 1.0
            if abs(comp["log_v1"] - log_p1) < 1.0:
                w *= 2.0
            weights.append(w)
        xs = [c["log_v1"] for c in usable]
        ys = [c["ratios"][h] for c in usable]
        mean_x = sum(xs) / len(xs)
        weight_sum = sum(weights)
        mean_y = sum(w * y for w, y in zip(weights, ys)) / weight_sum
        sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs)) + RIDGE_PRIOR
        sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
        slope = max(-MAX_SLOPE, min(MAX_SLOPE, sxy / sxx))
        adjustment = max(-MAX_SIZE_ADJUSTMENT,
                         min(MAX_SIZE_ADJUSTMENT, slope * (log_p1 - mean_x)))
        medians[h] = mean_y + adjustment
        residuals = [(y - (mean_y + slope * (x - mean_x)), w)
                     for x, y, w in zip(xs, ys, weights)]
        q25 = weighted_quantile(residuals, 0.25)
        q75 = weighted_quantile(residuals, 0.75)
        sigma_lo[h] = max(SPREAD_FLOOR, -q25 / 0.674 * SPREAD_SCALE)
        sigma_hi[h] = max(SPREAD_FLOOR, q75 / 0.674 * SPREAD_SCALE)
    medians = smooth(medians)
    sigma_lo = smooth(sigma_lo)
    sigma_hi = smooth(sigma_hi)

    rows = []
    for h in hours:
        if h not in medians:
            medians[h], sigma_lo[h], sigma_hi[h] = 0.0, FALLBACK_SIGMA * math.sqrt(max(1, h - 1)), FALLBACK_SIGMA * math.sqrt(max(1, h - 1))
        quantiles = []
        for z in z_values:
            sigma = sigma_lo[h] if z < 0 else sigma_hi[h]
            quantiles.append(hour1 * math.exp(medians[h] + z * sigma))
        quantiles = [round2_half_up(min(1e9, max(0.01, q))) for q in quantiles]
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def fallback_forecast(case):
    """Persistence forecast: every hour stays at the hour-1 print."""
    levels = [float(x) for x in case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])]
    z_values = [normal_ppf(p) for p in levels]
    known = [k for k in case.get("known", []) if k.get("hour") == 1]
    hour1 = max(1.0, float(known[0]["viewers"])) if known else 1.0
    hours = [int(r["hour"]) for r in case.get("forecast", [])] or list(range(2, 25))
    rows = []
    for h in hours:
        quantiles = [round2_half_up(min(1e9, max(0.01, hour1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(max(1, h - 1)))))) for z in z_values]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = fallback_forecast(case)
            except Exception:
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
