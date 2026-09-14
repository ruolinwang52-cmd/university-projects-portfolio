# Time-Series Analysis Project

## Quick overview

A collaborative forecasting project using hourly air-temperature data from Hojbakkegaard, Denmark. ARMA, Box-Jenkins, and Kalman-filter models are developed and compared across several test periods.

## Approach

- Explores and transforms the raw hourly series
- Selects relatively stationary modelling windows
- Uses ACF/PACF and residual diagnostics for ARMA model selection
- Builds a Box-Jenkins model with net radiation as an input
- Uses Kalman filtering for parameter estimation and prediction refinement
- Compares forecasts against a naive baseline on multiple holdout periods

## Files

- [`report.pdf`](report.pdf) — complete 24-page project report
- `project.m` — main MATLAB workflow
- `projectData24.mat` — project dataset
- [`DEPENDENCIES.md`](DEPENDENCIES.md) — complete transitive list of required teaching-library functions and MATLAB toolboxes

## Skills demonstrated

Time-series diagnostics, ARMA/Box-Jenkins modelling, state-space methods, Kalman filtering, forecast evaluation, and technical reporting.

## Authorship note

Completed collaboratively with Olle Serrander; both contributors are credited in the report.
