# Are Turkish REITs Proxies for Direct Real Estate Investment?

## Abstract
[This study investigates whether Turkish Real Estate Investment Trusts (REITs) can serve as effective proxies for direct real estate investment. Using monthly data from January 2014 to December 2025, we apply ARFIMA de-smoothing to the CBRT Housing Price Index to correct for appraisal-based valuation smoothing, then estimate time-varying correlations via a DCC-GARCH model. At the monthly frequency, we find a negligible correlation between REITs and direct real estate (Pearson r = 0.006, p = 0.94), confirming they are distinct assets for short-horizon investors. The DCC model reveals a weak but consistently positive dynamic correlation (mean ρ = 0.17, range: 0.01–0.21). Over annual horizons, the point estimate rises substantially (r = 0.64, p = 0.03), but with only 12 overlapping years, the bootstrap 95% confidence interval [-0.32, 0.91] cannot rule out zero. Five-year rolling correlations range from -0.06 to +0.77, indicating significant regime-dependence. We conclude that Turkish REITs are not reliable short-term real estate proxies, and longer time series are needed to assess their viability as long-term substitutes. These findings are consistent with the "developer REIT" hypothesis, wherein Turkish REITs derive substantial value from development activities rather than stabilized property income.]

## Methodology
- ARFIMA de-smoothing (GPH estimator)
- DCC-GARCH for time-varying correlation
- Bootstrap confidence intervals (10,000 replications)

## Data Sources
- REIT prices: Yahoo Finance (yfinance)
- CBRT Housing Price Index: CBRT EVDS (TP.HPI.A01)
- Instructions for access in `data/README.md`

## Requirements
See `requirements.txt` in root directory

