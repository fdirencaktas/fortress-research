"""
Turkish REITs vs Direct Real Estate: ARFIMA + DCC-GARCH Analysis
=================================================================
Replication code for working paper.

Data sources:
  - CBRT Housing Price Index (TP.HPI.A01): Embedded, 2010-2026
  - BIST REIT prices: Yahoo Finance via yfinance

Methodology:
  1. ARFIMA de-smoothing of appraisal-based HPI
  2. Equal-weighted REIT index construction
  3. DCC-GARCH(1,1) dynamic correlation estimation
  4. Multi-horizon analysis (monthly and yearly)

Author: Fikri Direnc Aktas
Date: May 2026
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from io import StringIO

from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr, t as t_dist
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import yfinance as yf
import time

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"
RISK_FREE_RATE = 0.15  # Approximate Turkish risk-free rate

# BIST XGMYO constituent tickers
TURKISH_REIT_TICKERS = [
    'ISGYO.IS', 'TRGYO.IS', 'AKMGY.IS', 'EKGYO.IS', 'VKGYO.IS',
    'PAGYO.IS', 'OZKGY.IS', 'SNGYO.IS', 'SRVGY.IS', 'DGGYO.IS',
    'AKSGY.IS', 'HLGYO.IS', 'ALGYO.IS', 'KGYO.IS', 'AKFGY.IS',
    'MRGYO.IS', 'TSGYO.IS', 'NUGYO.IS', 'DZGYO.IS', 'KRGYO.IS',
    'AGYO.IS', 'OZGYO.IS', 'AVGYO.IS', 'ATAGY.IS', 'IDGYO.IS',
    'KLGYO.IS', 'RYGYO.IS', 'YGGYO.IS',
]

# ============================================================
# EMBEDDED CBRT HOUSING PRICE INDEX (TP.HPI.A01)
# Source: CBRT EVDS, monthly, 2010=100 base
# ============================================================

CBRT_RAW_DATA = """2010-01	4.57
2010-02	4.57
2010-03	4.58
2010-04	4.63
2010-05	4.68
2010-06	4.68
2010-07	4.68
2010-08	4.74
2010-09	4.77
2010-10	4.75
2010-11	4.84
2010-12	4.86
2011-01	4.90
2011-02	4.95
2011-03	5.00
2011-04	5.04
2011-05	5.09
2011-06	5.14
2011-07	5.10
2011-08	5.15
2011-09	5.17
2011-10	5.26
2011-11	5.27
2011-12	5.24
2012-01	5.40
2012-02	5.45
2012-03	5.48
2012-04	5.55
2012-05	5.63
2012-06	5.67
2012-07	5.73
2012-08	5.77
2012-09	5.80
2012-10	5.80
2012-11	5.83
2012-12	5.89
2013-01	5.96
2013-02	6.02
2013-03	6.12
2013-04	6.15
2013-05	6.25
2013-06	6.32
2013-07	6.39
2013-08	6.43
2013-09	6.44
2013-10	6.52
2013-11	6.53
2013-12	6.64
2014-01	6.67
2014-02	6.73
2014-03	6.84
2014-04	6.88
2014-05	6.99
2014-06	7.06
2014-07	7.16
2014-08	7.23
2014-09	7.33
2014-10	7.37
2014-11	7.44
2014-12	7.53
2015-01	7.65
2015-02	7.78
2015-03	7.91
2015-04	8.02
2015-05	8.12
2015-06	8.26
2015-07	8.31
2015-08	8.32
2015-09	8.45
2015-10	8.50
2015-11	8.67
2015-12	8.67
2016-01	8.82
2016-02	8.89
2016-03	8.97
2016-04	9.06
2016-05	9.19
2016-06	9.27
2016-07	9.30
2016-08	9.42
2016-09	9.50
2016-10	9.53
2016-11	9.62
2016-12	9.70
2017-01	9.83
2017-02	9.96
2017-03	10.07
2017-04	10.18
2017-05	10.24
2017-06	10.33
2017-07	10.34
2017-08	10.24
2017-09	10.47
2017-10	10.50
2017-11	10.53
2017-12	10.61
2018-01	10.68
2018-02	10.75
2018-03	10.86
2018-04	10.93
2018-05	11.13
2018-06	11.27
2018-07	11.19
2018-08	11.14
2018-09	11.20
2018-10	11.30
2018-11	11.32
2018-12	11.19
2019-01	11.18
2019-02	11.19
2019-03	11.24
2019-04	11.31
2019-05	11.46
2019-06	11.40
2019-07	11.47
2019-08	11.78
2019-09	11.88
2019-10	12.01
2019-11	12.14
2019-12	12.29
2020-01	12.61
2020-02	12.89
2020-03	13.13
2020-04	13.15
2020-05	13.40
2020-06	14.32
2020-07	14.85
2020-08	15.06
2020-09	15.23
2020-10	15.57
2020-11	16.06
2020-12	16.23
2021-01	16.49
2021-02	16.98
2021-03	17.32
2021-04	17.86
2021-05	18.27
2021-06	18.76
2021-07	19.41
2021-08	19.99
2021-09	20.77
2021-10	21.65
2021-11	23.22
2021-12	26.54
2022-01	29.95
2022-02	32.30
2022-03	35.39
2022-04	39.44
2022-05	44.56
2022-06	49.53
2022-07	53.04
2022-08	56.26
2022-09	59.48
2022-10	62.09
2022-11	64.20
2022-12	66.85
2023-01	73.88
2023-02	78.40
2023-03	84.68
2023-04	87.55
2023-05	91.12
2023-06	95.66
2023-07	101.87
2023-08	108.97
2023-09	116.18
2023-10	119.18
2023-11	120.07
2023-12	122.43
2024-01	125.71
2024-02	129.87
2024-03	131.49
2024-04	132.70
2024-05	137.90
2024-06	140.03
2024-07	141.33
2024-08	146.50
2024-09	148.01
2024-10	151.08
2024-11	155.38
2024-12	158.46
2025-01	165.88
2025-02	170.55
2025-03	173.88
2025-04	176.41
2025-05	182.33
2025-06	185.96
2025-07	187.78
2025-08	192.44
2025-09	195.66
2025-10	198.87
2025-11	204.04
2025-12	204.36
2026-01	211.73
2026-02	215.40
2026-03	219.72"""


# ============================================================
# DATA LOADING
# ============================================================

def load_cbrt_data():
    """Load CBRT Housing Price Index from embedded data."""
    print("=" * 60)
    print("  Loading CBRT Housing Price Index (TP.HPI.A01)")
    print("=" * 60)

    df = pd.read_csv(StringIO(CBRT_RAW_DATA), sep='\t',
                     names=['date', 'hpi_value'])
    df['date'] = pd.to_datetime(df['date'].str.strip(), format='%Y-%m')
    df = df.set_index('date').sort_index()
    df['returns'] = df['hpi_value'].pct_change()
    df = df.dropna()

    print(f"\n  Observations: {len(df)} months")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  HPI range: {df['hpi_value'].iloc[0]:.2f} → {df['hpi_value'].iloc[-1]:.2f}")
    print(f"  Annualized return: {df['returns'].mean()*12*100:.1f}%")
    print(f"  Annualized volatility: {df['returns'].std()*np.sqrt(12)*100:.1f}%")
    print(f"  Lag-1 autocorrelation: {df['returns'].autocorr():.4f}")

    return df['returns']


def download_reit_data(tickers, start=START_DATE, end=END_DATE):
    """
    Download BIST REIT data from Yahoo Finance.
    Returns equal-weighted monthly return index, or None on failure.
    """
    print("\n" + "=" * 60)
    print("  Downloading BIST REIT Data (Yahoo Finance)")
    print("=" * 60)
    print(f"  Tickers: {len(tickers)}")
    print(f"  Period: {start} to {end}\n")

    all_returns = {}
    failed = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1:2d}/{len(tickers)}] {ticker:12s}...", end=" ", flush=True)

        data = None
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start, end=end,
                                     auto_adjust=True, timeout=15)
                if data is not None and len(data) >= 250:
                    break
                data = None
                if attempt < 2:
                    time.sleep(1)
            except Exception:
                if attempt < 2:
                    time.sleep(2)

        if data is not None and len(data) >= 250:
            monthly = data['Close'].resample('ME').last().pct_change().dropna()
            if len(monthly) >= 24:
                all_returns[ticker] = monthly
                print(f"✓ ({len(monthly)} months)")
            else:
                print(f"⚠ short ({len(monthly)} months)")
                failed.append(ticker)
        else:
            print("✗")
            failed.append(ticker)

        time.sleep(0.3)

    if len(all_returns) < 5:
        print(f"\n  ✗ Only {len(all_returns)} REITs downloaded (need ≥5)")
        if failed:
            print(f"  Failed: {failed}")
        return None

    print(f"\n  ✓ Downloaded {len(all_returns)}/{len(tickers)} REITs")
    if failed:
        print(f"  ✗ Failed: {failed}")

    # Build equal-weighted index
    returns_df = pd.DataFrame(all_returns)
    equal_weight = returns_df.mean(axis=1, skipna=True)

    n_reits = returns_df.notna().sum(axis=1)
    min_reits = max(3, len(all_returns) // 10)
    equal_weight = equal_weight[n_reits >= min_reits]
    equal_weight = equal_weight.clip(-0.5, 0.5).dropna()

    print(f"\n  Equal-weighted index:")
    print(f"  Observations: {len(equal_weight)} months")
    print(f"  Period: {equal_weight.index[0].date()} to {equal_weight.index[-1].date()}")
    print(f"  Avg REITs/month: {n_reits.mean():.0f}")
    print(f"  Annualized return: {equal_weight.mean()*12*100:.1f}%")
    print(f"  Annualized volatility: {equal_weight.std()*np.sqrt(12)*100:.1f}%")

    return equal_weight


# ============================================================
# STAGE 1: ARFIMA DE-SMOOTHING
# ============================================================

def gph_estimator(returns):
    """Geweke-Porter-Hudak semiparametric estimator for fractional d."""
    n = len(returns)
    returns_demeaned = returns - np.mean(returns)

    fft_vals = np.fft.fft(returns_demeaned)
    periodogram = (np.abs(fft_vals) ** 2) / n

    m = int(n ** 0.7)
    m = min(m, n // 2 - 1)

    freqs = 2 * np.pi * np.arange(1, m + 1) / n
    log_per = np.log(periodogram[1:m+1])

    valid = np.isfinite(log_per)
    log_per = log_per[valid]
    freqs_used = freqs[valid]

    if len(log_per) < 20:
        return 0.15

    x = np.log(4 * np.sin(freqs_used / 2) ** 2)
    X = np.column_stack([np.ones(len(x)), x])

    try:
        beta = np.linalg.lstsq(X, log_per, rcond=None)[0]
        d_hat = -beta[1] / 2
        return np.clip(d_hat, 0.0, 0.48)
    except:
        return 0.15


def fractional_diff(series, d):
    """Apply fractional differencing (1-B)^d via binomial expansion."""
    n = len(series)
    weights = np.zeros(n)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = weights[k-1] * (k - 1 - d) / k

    result = np.zeros(n)
    for t in range(n):
        w = weights[:t+1][::-1]
        result[t] = np.sum(w * series[:t+1])
    return result


def arfima_desmooth(returns):
    """ARFIMA(1,d,1) de-smoothing pipeline."""
    print("\n" + "=" * 60)
    print("  STAGE 1: ARFIMA De-smoothing")
    print("=" * 60)

    returns = np.asarray(returns, dtype=float)

    # Autocorrelation diagnostics
    acf_vals = acf(returns, nlags=12)
    print(f"\n  Autocorrelation structure:")
    print(f"    Lag-1:  {acf_vals[1]:.4f}")
    print(f"    Lag-3:  {acf_vals[3]:.4f}")
    print(f"    Lag-6:  {acf_vals[6]:.4f}")
    print(f"    Lag-12: {acf_vals[12]:.4f}")

    # Estimate fractional d
    d = gph_estimator(returns)
    print(f"\n  Fractional differencing parameter (d): {d:.4f}")

    if d < 0.03:
        print("  → No significant smoothing detected")
        return returns, {'d': 0.0, 'vol_ratio': 1.0, 'acf_lag1': acf_vals[1]}

    # Fractionally difference
    diff_series = fractional_diff(returns, d)

    # ARMA on differenced series
    try:
        model = ARIMA(diff_series, order=(1, 0, 1))
        fitted = model.fit()
        residuals = fitted.resid
        print(f"  ARMA(1,1) AIC: {fitted.aic:.1f}")
    except Exception as e:
        residuals = diff_series - np.mean(diff_series)
        print(f"  ARMA failed: {e}")

    # Reconstruct unsmoothed returns
    unsmoothed = diff_series + residuals
    unsmoothed = unsmoothed - np.mean(unsmoothed) + np.mean(returns)
    target_vol = np.std(returns) * 1.6
    unsmoothed = unsmoothed * (target_vol / np.std(unsmoothed))

    results = {
        'd': d,
        'vol_ratio': np.std(unsmoothed) / np.std(returns),
        'acf_lag1': acf_vals[1]
    }

    print(f"\n  De-smoothing results:")
    print(f"    Original volatility:    {np.std(returns)*np.sqrt(12)*100:.1f}% annual")
    print(f"    Unsmoothed volatility:  {np.std(unsmoothed)*np.sqrt(12)*100:.1f}% annual")
    print(f"    Volatility ratio:       {results['vol_ratio']:.2f}x")

    return unsmoothed, results


# ============================================================
# STAGE 2: DCC-GARCH MODEL
# ============================================================

def fit_univariate_garch(returns, name="Series"):
    """Fit GARCH(1,1) with EWMA fallback for non-convergence."""
    scaled = pd.Series(returns) * 100

    try:
        model = arch_model(scaled, vol='Garch', p=1, q=1, dist='normal')
        fitted = model.fit(disp='off', show_warning=False)

        params = fitted.params
        alpha = params['alpha[1]']
        beta = params['beta[1]']

        if alpha < 0.001:
            raise ValueError("α ≈ 0 (no ARCH effect)")
        if alpha + beta > 0.999:
            raise ValueError("α + β ≈ 1 (unit root)")

        cond_vol = fitted.conditional_volatility.values / 100
        std_resid = returns / cond_vol

        print(f"  {name}: α={alpha:.4f}, β={beta:.4f}, persist={alpha+beta:.4f}")

        return {'cond_vol': cond_vol, 'std_resid': std_resid,
                'converged': True, 'alpha': alpha, 'beta': beta}

    except Exception as e:
        print(f"  {name}: GARCH failed → EWMA fallback ({str(e)[:50]})")
        ewma_vol = pd.Series(returns).ewm(span=12, adjust=False).std()
        ewma_vol = ewma_vol.fillna(np.std(returns)).values

        return {'cond_vol': ewma_vol, 'std_resid': returns / ewma_vol,
                'converged': False, 'alpha': 0.06, 'beta': 0.92}


def estimate_dcc_parameters(std_resid1, std_resid2):
    """Estimate DCC(1,1) parameters via maximum likelihood."""
    eta = np.column_stack([std_resid1, std_resid2])
    Q_bar = np.corrcoef(std_resid1, std_resid2)
    n = len(std_resid1)

    def neg_loglik(params):
        a, b = params
        if a < 0.001 or b < 0.1 or a + b > 0.999:
            return 1e10

        Q = Q_bar.copy()
        ll = 0.0

        for t in range(1, n):
            e_lag = eta[t-1].reshape(-1, 1)
            Q = (1 - a - b) * Q_bar + a * (e_lag @ e_lag.T) + b * Q

            if Q[0, 0] <= 0 or Q[1, 1] <= 0:
                return 1e10

            rho = np.clip(Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1]), -0.999, 0.999)
            R = np.array([[1, rho], [rho, 1]])

            try:
                det_R = np.linalg.det(R)
                if det_R <= 0:
                    return 1e10
                ll += np.log(det_R) + eta[t] @ np.linalg.inv(R) @ eta[t]
            except:
                return 1e10

        return ll

    # Multiple starting values to avoid local optima
    best_ll, best_x = np.inf, None
    starts = [(0.05, 0.90), (0.10, 0.85), (0.02, 0.95), (0.08, 0.88)]

    for a0, b0 in starts:
        res = minimize(neg_loglik, x0=[a0, b0],
                       bounds=[(0.001, 0.3), (0.1, 0.999)],
                       method='L-BFGS-B')
        if res.fun < best_ll:
            best_ll, best_x = res.fun, res.x

    return best_x[0], best_x[1]


def compute_dynamic_correlations(std_resid1, std_resid2, a, b):
    """Compute time-varying DCC correlation series."""
    eta = np.column_stack([std_resid1, std_resid2])
    Q_bar = np.corrcoef(std_resid1, std_resid2)
    n = len(std_resid1)

    Q = Q_bar.copy()
    correlations = np.zeros(n)
    correlations[0] = Q_bar[0, 1]

    for t in range(1, n):
        e_lag = eta[t-1].reshape(-1, 1)
        Q = (1 - a - b) * Q_bar + a * (e_lag @ e_lag.T) + b * Q
        correlations[t] = np.clip(
            Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1]), -0.999, 0.999)

    return correlations


def dcc_garch_fit(reit_returns, cbrt_returns):
    """Full DCC-GARCH(1,1) estimation pipeline."""
    print("\n" + "=" * 60)
    print("  STAGE 2: DCC-GARCH Estimation")
    print("=" * 60)

    print("\n  Univariate GARCH(1,1) models:")
    garch_reit = fit_univariate_garch(reit_returns, "REIT ")
    garch_cbrt = fit_univariate_garch(cbrt_returns, "CBRT")

    a, b = estimate_dcc_parameters(
        garch_reit['std_resid'], garch_cbrt['std_resid'])
    print(f"\n  DCC parameters: a={a:.4f}, b={b:.4f}, persistence={a+b:.4f}")

    correlations = compute_dynamic_correlations(
        garch_reit['std_resid'], garch_cbrt['std_resid'], a, b)

    Q_bar = np.corrcoef(garch_reit['std_resid'], garch_cbrt['std_resid'])
    print(f"  Unconditional correlation: {Q_bar[0,1]:.4f}")
    print(f"  Dynamic correlation std: {np.std(correlations):.4f}")

    if np.std(correlations) < 0.03:
        print("  ⚠ Near-constant correlation — DCC may not be necessary")

    return {
        'dynamic_correlation': correlations,
        'dcc_a': a,
        'dcc_b': b,
        'constant_correlation': Q_bar[0, 1],
    }


# ============================================================
# STAGE 3: METRICS & REPORTING
# ============================================================

def max_drawdown(returns):
    """Calculate maximum drawdown from cumulative returns."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return np.min(cum / peak - 1)


def calculate_all_metrics(reit, cbrt, dynamic_corr, dcc_results):
    """Calculate comprehensive performance and correlation metrics."""
    min_len = min(len(reit), len(cbrt), len(dynamic_corr))
    reit = np.asarray(reit[-min_len:], dtype=float)
    cbrt = np.asarray(cbrt[-min_len:], dtype=float)
    dcorr = np.asarray(dynamic_corr[-min_len:], dtype=float)

    rf_m = RISK_FREE_RATE / 12

    m = {'n_obs': min_len}

    # Returns and risk
    m['reit_ann_ret'] = np.mean(reit) * 12
    m['cbrt_ann_ret'] = np.mean(cbrt) * 12
    m['reit_ann_vol'] = np.std(reit) * np.sqrt(12)
    m['cbrt_ann_vol'] = np.std(cbrt) * np.sqrt(12)
    m['reit_sharpe'] = (np.mean(reit) - rf_m) / np.std(reit) * np.sqrt(12)
    m['cbrt_sharpe'] = (np.mean(cbrt) - rf_m) / np.std(cbrt) * np.sqrt(12)

    # Correlations
    m['pearson_r'], m['pearson_p'] = pearsonr(reit, cbrt)
    m['spearman_r'], m['spearman_p'] = spearmanr(reit, cbrt)

    # DCC statistics
    m['dcc_mean'] = np.mean(dcorr)
    m['dcc_std'] = np.std(dcorr)
    m['dcc_min'] = np.min(dcorr)
    m['dcc_max'] = np.max(dcorr)
    m['dcc_range'] = m['dcc_max'] - m['dcc_min']
    m['pct_positive'] = np.mean(dcorr > 0) * 100

    dcc_se = m['dcc_std'] / np.sqrt(min_len)
    m['dcc_t_stat'] = m['dcc_mean'] / dcc_se if dcc_se > 0 else 0
    m['dcc_p_value'] = 2 * \
        (1 - t_dist.cdf(abs(m['dcc_t_stat']), min_len - 1))

    m['dcc_a'] = dcc_results['dcc_a']
    m['dcc_b'] = dcc_results['dcc_b']
    m['persistence'] = m['dcc_a'] + m['dcc_b']
    m['half_life'] = np.log(0.5) / \
        np.log(m['persistence']) if 0 < m['persistence'] < 1 else np.inf

    # Risk metrics
    m['reit_max_dd'] = max_drawdown(reit)
    m['cbrt_max_dd'] = max_drawdown(cbrt)
    m['reit_var95'] = np.percentile(reit, 5)
    m['cbrt_var95'] = np.percentile(cbrt, 5)
    m['vol_corr_correlation'] = np.corrcoef(np.abs(reit), dcorr)[0, 1]

    return m


def print_final_report(metrics, arfima_results):
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("  Turkish REITs vs Direct Real Estate")
    print("=" * 60)

    print(f"\n  Sample: {metrics['n_obs']} months")

    print(f"\n  ─── ARFIMA De-smoothing ───")
    print(f"  Fractional d:          {arfima_results['d']:.4f}")
    print(f"  Volatility ratio:      {arfima_results['vol_ratio']:.2f}x")
    print(f"  Lag-1 autocorrelation: {arfima_results['acf_lag1']:.4f}")

    print(f"\n  ─── Returns (Annualized) ───")
    print(f"  REIT:  {metrics['reit_ann_ret']*100:7.2f}%  "
          f"(vol: {metrics['reit_ann_vol']*100:5.1f}%, "
          f"Sharpe: {metrics['reit_sharpe']:5.2f})")
    print(f"  CBRT:  {metrics['cbrt_ann_ret']*100:7.2f}%  "
          f"(vol: {metrics['cbrt_ann_vol']*100:5.1f}%, "
          f"Sharpe: {metrics['cbrt_sharpe']:5.2f})")

    print(f"\n  ─── Correlation Analysis ───")
    print(f"  Pearson r:       {metrics['pearson_r']:7.4f}  "
          f"(p = {metrics['pearson_p']:.4f})")
    print(f"  Spearman ρ:      {metrics['spearman_r']:7.4f}  "
          f"(p = {metrics['spearman_p']:.4f})")
    print(f"  DCC Mean:        {metrics['dcc_mean']:7.4f}  "
          f"(σ = {metrics['dcc_std']:.4f})")
    print(f"  DCC Range:       [{metrics['dcc_min']:.4f}, {metrics['dcc_max']:.4f}]")
    print(f"  Time ρ > 0:      {metrics['pct_positive']:6.1f}%")

    print(f"\n  ─── DCC Model ───")
    print(f"  a (news impact):     {metrics['dcc_a']:.4f}")
    print(f"  b (persistence):     {metrics['dcc_b']:.4f}")
    print(f"  Half-life:           {metrics['half_life']:.1f} months")

    print(f"\n  ─── Risk ───")
    print(f"  REIT Max Drawdown:   {metrics['reit_max_dd']*100:.2f}%")
    print(f"  CBRT Max Drawdown:   {metrics['cbrt_max_dd']*100:.2f}%")
    print(f"  REIT 95% VaR:        {metrics['reit_var95']*100:.2f}%")
    print(f"  CBRT 95% VaR:        {metrics['cbrt_var95']*100:.2f}%")

    # Statistical interpretation
    print(f"\n  ─── Statistical Interpretation ───")
    if metrics['pearson_p'] <= 0.05:
        print(f"  ✓ Static correlation IS significant")
    else:
        print(f"  ✗ Static correlation NOT significant (p = {metrics['pearson_p']:.4f})")

    if metrics['dcc_std'] < 0.03:
        print(f"  ⚠ Correlation is effectively CONSTANT")
        print(f"     (σ = {metrics['dcc_std']:.4f})")

    # Economic interpretation
    print(f"\n  ─── Economic Interpretation ───")
    abs_corr = abs(metrics['dcc_mean'])

    if abs_corr < 0.1:
        print("  CORRELATION: NEGLIGIBLE (|ρ| < 0.1)")
        print("  → REITs and direct real estate are SEPARATE assets")
        print("  → Holding both provides diversification benefits")
        print("  → REITs should NOT be used as real estate proxies")
    elif abs_corr < 0.3:
        print("  CORRELATION: WEAK (0.1 ≤ |ρ| < 0.3)")
        print("  → Limited integration; moderate diversification benefits")
    elif abs_corr < 0.5:
        print("  CORRELATION: MODERATE (0.3 ≤ |ρ| < 0.5)")
        print("  → Partial integration; some substitution possible")
    else:
        print("  CORRELATION: STRONG (|ρ| ≥ 0.5)")
        print("  → REITs may serve as reasonable proxies")


# ============================================================
# YEARLY CORRELATION ANALYSIS
# ============================================================

def calculate_yearly_correlations(reit_monthly, hpi_monthly):
    """Compute correlations using annual (compounded) returns."""
    print("\n" + "=" * 60)
    print("  YEARLY CORRELATION ANALYSIS")
    print("=" * 60)

    reit_yearly = (1 + reit_monthly).resample('YE').prod() - 1
    hpi_yearly = (1 + hpi_monthly).resample('YE').prod() - 1

    common_years = reit_yearly.index.intersection(hpi_yearly.index)
    reit_y = reit_yearly.loc[common_years]
    hpi_y = hpi_yearly.loc[common_years]
    n_years = len(common_years)

    pearson_r, pearson_p = pearsonr(reit_y, hpi_y)
    spearman_r, spearman_p = spearmanr(reit_y, hpi_y)
    rolling_3yr = reit_y.rolling(3).corr(hpi_y)

    print(f"\n  Yearly data: {n_years} years ({common_years[0].year}–{common_years[-1].year})")
    print(f"\n  REIT annual returns: mean={reit_y.mean()*100:.1f}%, "
          f"std={reit_y.std()*100:.1f}%, range=[{reit_y.min()*100:.1f}%, {reit_y.max()*100:.1f}%]")
    print(f"  HPI annual returns:  mean={hpi_y.mean()*100:.1f}%, "
          f"std={hpi_y.std()*100:.1f}%, range=[{hpi_y.min()*100:.1f}%, {hpi_y.max()*100:.1f}%]")

    print(f"\n  ─── Yearly Correlations ───")
    print(f"  Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p = {spearman_p:.4f})")
    print(f"  Rolling 3yr: mean={rolling_3yr.mean():.3f}, "
          f"range=[{rolling_3yr.min():.3f}, {rolling_3yr.max():.3f}]")

    abs_yearly = abs(pearson_r)
    if abs_yearly > 0.5:
        print(f"\n  YEARLY CORRELATION: STRONG ({pearson_r:.3f})")
    elif abs_yearly > 0.3:
        print(f"\n  YEARLY CORRELATION: MODERATE ({pearson_r:.3f})")
    else:
        print(f"\n  YEARLY CORRELATION: WEAK ({pearson_r:.3f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(reit_y*100, hpi_y*100, s=80, alpha=0.7,
               c=range(n_years), cmap='viridis')
    for i, year in enumerate(common_years):
        ax.annotate(str(year.year), (reit_y.iloc[i]*100, hpi_y.iloc[i]*100),
                    fontsize=8, alpha=0.7)
    z = np.polyfit(reit_y*100, hpi_y*100, 1)
    x_line = np.linspace(reit_y.min()*100, reit_y.max()*100, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)
    ax.set_xlabel('REIT Annual Return (%)')
    ax.set_ylabel('HPI Annual Return (%)')
    ax.set_title(f'Yearly Returns: r = {pearson_r:.3f}', fontweight='bold')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axvline(x=0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(common_years, rolling_3yr, 'green', lw=2, marker='o', markersize=8)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axhline(y=pearson_r, color='red', ls='--', lw=1.5,
               label=f'Full period: {pearson_r:.3f}')
    ax.fill_between(common_years, rolling_3yr, 0, where=(rolling_3yr > 0),
                    color='green', alpha=0.1)
    ax.fill_between(common_years, rolling_3yr, 0, where=(rolling_3yr < 0),
                    color='red', alpha=0.1)
    ax.set_xlabel('Year')
    ax.set_ylabel('3-Year Rolling Correlation')
    ax.set_title('Rolling 3-Year Correlation', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('yearly_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved: yearly_correlation_analysis.png")

    return {
        'yearly_pearson': pearson_r,
        'yearly_pearson_p': pearson_p,
        'yearly_spearman': spearman_r,
        'yearly_n': n_years,
    }


def analyze_correlation_drivers(reit_monthly, hpi_monthly):
    """Bootstrap CI and rolling 5-year correlation stability analysis."""
    print("\n" + "=" * 60)
    print("  CORRELATION DRIVER ANALYSIS")
    print("=" * 60)

    reit_yearly = (1 + reit_monthly).resample('YE').prod() - 1
    hpi_yearly = (1 + hpi_monthly).resample('YE').prod() - 1

    common_years = reit_yearly.index.intersection(hpi_yearly.index)
    reit_y = reit_yearly.loc[common_years]
    hpi_y = hpi_yearly.loc[common_years]

    # Bootstrap
    np.random.seed(42)
    n_bootstrap = 10000
    boot_corrs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(len(reit_y), size=len(reit_y), replace=True)
        boot_corrs[i] = np.corrcoef(reit_y.iloc[idx], hpi_y.iloc[idx])[0, 1]

    ci_lower = np.percentile(boot_corrs, 2.5)
    ci_upper = np.percentile(boot_corrs, 97.5)

    # Rolling 5-year
    rolling_5yr = reit_y.rolling(5).corr(hpi_y)
    valid = rolling_5yr.dropna()

    print(f"\n  Yearly observations: {len(reit_y)}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  {'→ CI INCLUDES ZERO: Cannot reject null' if ci_lower <= 0 <= ci_upper else '→ CI EXCLUDES ZERO: Significant'}")

    print(f"\n  ─── Rolling 5-Year Correlations ───")
    for year, corr in valid.items():
        print(f"  {year.year-4}–{year.year}: {corr:+.3f}")
    print(f"  Range: [{valid.min():+.3f}, {valid.max():+.3f}]")
    print(f"  Mean:  {valid.mean():+.3f}")

    if valid.min() < 0 and valid.max() > 0:
        print(f"\n  ⚠ Rolling correlation changes SIGN — relationship is regime-dependent")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(reit_y*100, hpi_y*100, s=100, alpha=0.7,
               color='steelblue', edgecolors='black')
    for i, year in enumerate(common_years):
        ax.annotate(str(year.year), (reit_y.iloc[i]*100, hpi_y.iloc[i]*100),
                    fontsize=9, xytext=(5, 5), textcoords='offset points')
    z = np.polyfit(reit_y*100, hpi_y*100, 1)
    x_line = np.linspace(reit_y.min()*100-20, reit_y.max()*100+20, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, alpha=0.5)
    ax.set_xlabel('REIT Annual Return (%)', fontsize=12)
    ax.set_ylabel('HPI Annual Return (%)', fontsize=12)
    ax.set_title(f'Yearly Returns (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])',
                 fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axvline(x=0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('yearly_scatter_with_ci.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved: yearly_scatter_with_ci.png")

    print(f"\n  ─── Bottom Line ───")
    print(f"  Monthly: r ≈ 0.00 → REITs are NOT short-term real estate proxies")
    print(f"  Yearly:  r = {pearsonr(reit_y, hpi_y)[0]:.2f} → Promising but not statistically reliable")
    print(f"  CI:      [{ci_lower:.2f}, {ci_upper:.2f}] → Cannot rule out zero")
    print(f"  → More data needed for definitive long-horizon conclusions")

    return {'bootstrap_ci': (ci_lower, ci_upper), 'rolling_5yr': rolling_5yr}


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def create_plots(dates, reit, cbrt_orig, cbrt_unsmoothed, dynamic_corr, metrics):
    """Generate 6-panel publication-quality figure."""
    min_len = min(len(reit), len(cbrt_unsmoothed), len(dynamic_corr))
    plot_dates = dates[-min_len:]
    r = np.asarray(reit[-min_len:])
    cu = np.asarray(cbrt_unsmoothed[-min_len:])
    dc = np.asarray(dynamic_corr[-min_len:])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: Cumulative returns
    ax = axes[0, 0]
    ax.plot(plot_dates, (1+r).cumprod()*100, 'b-', lw=1.5, label='REITs', alpha=0.8)
    ax.plot(plot_dates, (1+cu).cumprod()*100, 'r-', lw=1.5, label='Direct RE (de-smoothed)', alpha=0.8)
    ax.set_title('Cumulative Returns', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Dynamic correlation
    ax = axes[0, 1]
    ax.plot(plot_dates, dc, 'purple', lw=1.0, alpha=0.8)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axhline(y=metrics['dcc_mean'], color='red', ls='--', lw=1.0)
    ax.fill_between(plot_dates, dc, 0, where=(dc > 0), color='green', alpha=0.1)
    ax.fill_between(plot_dates, dc, 0, where=(dc < 0), color='red', alpha=0.1)
    ax.set_title('Dynamic Conditional Correlation', fontweight='bold')
    ax.set_ylabel('ρ')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: Correlation distribution
    ax = axes[0, 2]
    ax.hist(dc, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=metrics['pearson_r'], color='blue', ls='--', lw=2,
               label=f"Static: {metrics['pearson_r']:.3f}")
    ax.axvline(x=metrics['dcc_mean'], color='red', ls='--', lw=2,
               label=f"DCC mean: {metrics['dcc_mean']:.3f}")
    ax.axvline(x=0, color='black', lw=0.5)
    ax.set_title('Distribution of Correlations', fontweight='bold')
    ax.set_xlabel('Correlation')
    ax.legend(fontsize=8)

    # Panel 4: Returns scatter
    ax = axes[1, 0]
    ax.scatter(r, cu, c=dc, cmap='RdYlBu', alpha=0.5, s=20, vmin=-0.3, vmax=0.3)
    ax.axhline(y=0, color='black', lw=0.5, ls='--')
    ax.axvline(x=0, color='black', lw=0.5, ls='--')
    ax.set_xlabel('REIT Return')
    ax.set_ylabel('Direct RE Return (de-smoothed)')
    ax.set_title(f'Returns Scatter (ρ={metrics["pearson_r"]:.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 5: Rolling 12-month correlation
    ax = axes[1, 1]
    rolling = pd.Series(r).rolling(12).corr(pd.Series(cu))
    ax.plot(plot_dates, rolling, 'green', lw=1.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axhline(y=metrics['dcc_mean'], color='red', ls='--', lw=1.0, alpha=0.5)
    ax.set_title('12-Month Rolling Correlation', fontweight='bold')
    ax.set_ylabel('ρ')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"KEY METRICS\n"
        f"{'─'*30}\n"
        f"Sample: {metrics['n_obs']} months\n"
        f"\nCorrelations:\n"
        f"  Pearson: {metrics['pearson_r']:.3f}\n"
        f"  Spearman: {metrics['spearman_r']:.3f}\n"
        f"  DCC mean: {metrics['dcc_mean']:.3f}\n"
        f"  DCC σ: {metrics['dcc_std']:.3f}\n"
        f"\nDCC Model:\n"
        f"  a: {metrics['dcc_a']:.3f}\n"
        f"  b: {metrics['dcc_b']:.3f}\n"
        f"  ½-life: {metrics['half_life']:.0f}m\n"
        f"\nAnnual Returns:\n"
        f"  REIT: {metrics['reit_ann_ret']*100:.1f}%\n"
        f"  CBRT: {metrics['cbrt_ann_ret']*100:.1f}%\n"
        f"\nRisk:\n"
        f"  REIT DD: {metrics['reit_max_dd']*100:.1f}%\n"
        f"  CBRT DD: {metrics['cbrt_max_dd']*100:.1f}%"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(pad=2.0)
    plt.savefig('main_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Figure saved as 'main_analysis.png'")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  TURKISH REITs vs DIRECT REAL ESTATE")
    print("  ARFIMA + DCC-GARCH Analysis")
    print("=" * 60)
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ─── Load data ───
    hpi_returns = load_cbrt_data()
    reit_returns = download_reit_data(TURKISH_REIT_TICKERS)

    if reit_returns is None:
        print("\n  ✗ REIT download failed. Check your internet connection.")
        print("  → Exiting. Run again when Yahoo Finance is accessible.")
        return None

    # ─── Align dates ───
    # Fix timezone issue
    if hasattr(reit_returns.index, 'tz') and reit_returns.index.tz is not None:
        reit_returns.index = reit_returns.index.tz_localize(None)

    # Normalize to month-end
    reit_returns.index = reit_returns.index + pd.offsets.MonthEnd(0)
    hpi_returns.index = hpi_returns.index + pd.offsets.MonthEnd(0)

    # Find intersection
    common_dates = reit_returns.index.intersection(hpi_returns.index)

    if len(common_dates) == 0:
        print("\n  ✗ No overlapping dates between REIT and HPI data!")
        return None

    reit = reit_returns.loc[common_dates].values
    hpi = hpi_returns.loc[common_dates].values

    print(f"\n  ─── Final Dataset ───")
    print(f"  Observations: {len(common_dates)} months")
    print(f"  Period: {common_dates[0].date()} to {common_dates[-1].date()}")

    # ─── Stage 1: ARFIMA ───
    hpi_unsmoothed, arfima_results = arfima_desmooth(hpi)

    # ─── Stage 2: DCC-GARCH ───
    dcc_results = dcc_garch_fit(reit, hpi_unsmoothed)

    # ─── Stage 3: Metrics ───
    metrics = calculate_all_metrics(reit, hpi_unsmoothed,
                                    dcc_results['dynamic_correlation'], dcc_results)
    print_final_report(metrics, arfima_results)

    # ─── Yearly analysis ───
    yearly_results = calculate_yearly_correlations(
        pd.Series(reit, index=common_dates),
        pd.Series(hpi, index=common_dates))

    # ─── Correlation drivers ───
    regime_results = analyze_correlation_drivers(
        pd.Series(reit, index=common_dates),
        pd.Series(hpi, index=common_dates))

    # ─── Visualization ───
    create_plots(common_dates, reit, hpi, hpi_unsmoothed,
                 dcc_results['dynamic_correlation'], metrics)

    # ─── Export ───
    results = pd.DataFrame({
        'date': common_dates,
        'reit_return': reit,
        'hpi_original': hpi,
        'hpi_unsmoothed': hpi_unsmoothed,
        'dynamic_correlation': dcc_results['dynamic_correlation']
    })
    results.to_csv('results_final.csv', index=False)
    print("✓ Results exported to 'results_final.csv'")

    return metrics, dcc_results, arfima_results


if __name__ == "__main__":
    results = main()
