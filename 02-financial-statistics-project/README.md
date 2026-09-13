# Financial Statistics Project

## Quick overview

An individual, three-part empirical finance project covering volatility modelling, option-pricing models with latent states, and multi-asset portfolio risk analysis. The work combines statistical estimation with out-of-sample evaluation in MATLAB.

## Main components

1. **Volatility modelling:** fits and compares GARCH/EGARCH-family models for Volvo returns, including heavy-tailed innovations and diagnostic checks.
2. **Option pricing:** calibrates and evaluates Black-Scholes and stochastic-volatility/jump-model variants, using filtering methods for time-varying latent states.
3. **Portfolio analysis:** models a 12-asset dataset, evaluates forecasts out of sample, and compares portfolio construction and risk measures.

## Files

- [`report.pdf`](report.pdf) — full 41-page report with methods, figures, results, and interpretation
- `Project_Part1.m`, `Project_Part2.m`, `Project_Part3.m` — MATLAB analysis for the three project parts
- `lnL.m` — likelihood helper
- `ASSETSA.csv`, `ASSETSB.csv` — portfolio datasets used by Part 3

## Skills demonstrated

Maximum-likelihood estimation, volatility modelling, nonlinear filtering, out-of-sample validation, risk measurement, portfolio analysis, data cleaning, and technical reporting.

## Reproducibility note

Part 3 runs from the included CSV files. Parts 1 and 2 also rely on course-supplied `.mat` datasets and MATLAB helper functions that are not redistributed here. A copyrighted course helper used for one heavy-tailed likelihood specification is intentionally excluded.
