#!/usr/bin/env python3
"""
separate.py
Complete example to:
 - load the provided 'time series data.csv'
 - choose a product (P1..P5) to forecast (default: P4)
 - split train/test (chronological, default 70/30)
 - fit SES, Holt, Holt-Winters (if seasonality), ARIMA (grid search small)
 - evaluate using MAPE, MAE, RMSE
 - save plots and an offline HTML report (report.html)

Usage:
    python separate.py

Notes:
 - File 'time series data.csv' must be in same folder.
 - Columns expected: t, ProductP1, ProductP2, ProductP3, ProductP4, ProductP5, price, temperature
 - P1,P2,P5 are weekly; P3,P4 are daily according to dataset description.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from jinja2 import Template

# ----------------------------
#  USER CONFIG
# ----------------------------
CSV_FILE = "time series data.csv"   # <- ถ้าชื่อไฟล์ต่าง ให้แก้ตรงนี้
PRODUCT = "ProductP4"               # <- เลือก ProductP1..P5 ที่ต้องการทดสอบ (default: ProductP4)
USE_EXOG = True                     # <- ถ้า Product มี exogenous variables (price, temperature) ให้ True (P4 มี)
SPLIT_RATIO = 0.7                   # train/test split ratio
CREATE_FAKE_DATES = True            # ถ้า True จะสร้าง datetime index จาก 't' ตามความถี่ที่เลือกด้านล่าง
FREQ = "D"                          # 'D' = daily, 'W' = weekly  (ตั้งให้ตรงกับ product)
SEASONAL_PERIODS = 7                # ถ้าเป็นรายวัน อาจใช้ 7 (week) ; ถ้าเป็นรายเดือนอาจ 12
MAX_ARIMA_P = 2
MAX_ARIMA_D = 1
MAX_ARIMA_Q = 2

OUTDIR = "output_report"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
#  Utility metrics
# ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero by small epsilon
    denom = np.where(np.abs(y_true) < 1e-6, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------
#  Load data
# ----------------------------
print("Loading", CSV_FILE)
df_raw = pd.read_csv(CSV_FILE)
print("Columns found:", list(df_raw.columns))

# Basic checks
required_cols = ["t", "price", "temperature"]
if PRODUCT not in df_raw.columns:
    raise SystemExit(f"ERROR: {PRODUCT} not found in CSV columns.")

# choose the series and optional exog
series = df_raw[["t", PRODUCT]].copy()
exog = None
if USE_EXOG and {"price", "temperature"}.issubset(df_raw.columns):
    exog = df_raw[["price", "temperature"]].copy()

# ----------------------------
#  Create index / handle 't'
# ----------------------------
# The CSV uses 't' as time point (1,2,3...). We will optionally convert to datetime
if CREATE_FAKE_DATES:
    # Create a date index based on FREQ using the first available real date as anchor or use a start date
    # We will create a date range of length len(series)
    n = len(series)
    start_date = pd.Timestamp("2019-01-01")  # arbitrary anchor; you can change to real start if you have it
    idx = pd.date_range(start=start_date, periods=n, freq=FREQ)
    series.index = idx
    series.index.name = "date"
    if exog is not None:
        exog.index = idx
    print(f"Created fake datetime index starting {start_date.date()} with freq='{FREQ}'.")
else:
    series = series.set_index("t")
    if exog is not None:
        exog.index = series.index
    print("Using 't' as index (no real datetime).")

series = series.rename(columns={PRODUCT: "y"})
print("Series head:\n", series.head())

# ----------------------------
#  Train/test split (chronological)
# ----------------------------
n = len(series)
n_train = int(n * SPLIT_RATIO)
train = series.iloc[:n_train].y
test = series.iloc[n_train:].y

exog_train = exog.iloc[:n_train] if exog is not None else None
exog_test = exog.iloc[n_train:] if exog is not None else None

print(f"\nTotal points: {n}  -> train: {n_train}, test: {n - n_train}")
print("Train period:", series.index[:n_train][0], "to", series.index[:n_train][-1])
print("Test period:", series.index[n_train], "to", series.index[-1])

# ----------------------------
#  Fit models
# ----------------------------
results = {}

# 1) Simple Exponential Smoothing (SES)
try:
    ses_model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    ses_forecast = ses_model.forecast(len(test))
    results['SES'] = {'model': ses_model, 'forecast': ses_forecast}
    print("SES fitted.")
except Exception as e:
    print("SES error:", e)

# 2) Holt (trend)
try:
    holt_model = Holt(train, initialization_method="estimated").fit()
    holt_forecast = holt_model.forecast(len(test))
    results['Holt'] = {'model': holt_model, 'forecast': holt_forecast}
    print("Holt fitted.")
except Exception as e:
    print("Holt error:", e)

# 3) Holt-Winters (seasonal) - additive seasonality by default; user can change to multiplicative
try:
    hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIODS, initialization_method="estimated").fit()
    hw_forecast = hw_model.forecast(len(test))
    results['HoltWinters'] = {'model': hw_model, 'forecast': hw_forecast}
    print("Holt-Winters fitted (additive).")
except Exception as e:
    print("Holt-Winters error (maybe insufficient points for seasonal_periods):", e)

# 4) ARIMA / SARIMAX with small grid search for (p,d,q) based on AIC (exog if provided)
best_aic = np.inf
best_order = None
best_sarimax = None
best_fore = None
try:
    for p in range(0, MAX_ARIMA_P+1):
        for d in range(0, MAX_ARIMA_D+1):
            for q in range(0, MAX_ARIMA_Q+1):
                try:
                    model = SARIMAX(train, order=(p,d,q), exog=exog_train if USE_EXOG and exog is not None else None, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                        best_sarimax = res
                except Exception:
                    continue
    if best_sarimax is not None:
        # forecast
        fore = best_sarimax.get_forecast(steps=len(test), exog=exog_test if USE_EXOG and exog is not None else None)
        arima_forecast = fore.predicted_mean
        results['ARIMA'] = {'model': best_sarimax, 'order': best_order, 'forecast': arima_forecast}
        print(f"ARIMA fitted. Best order (p,d,q)={best_order}, AIC={best_aic:.1f}")
    else:
        print("ARIMA: no successful fit in grid search.")
except Exception as e:
    print("ARIMA grid search error:", e)

# ----------------------------
#  Evaluate
# ----------------------------
eval_table = []
for name, info in results.items():
    y_pred = info['forecast']
    # align indices (if needed)
    y_pred.index = test.index
    m_mae = mean_absolute_error(test, y_pred)
    m_rmse = rmse(test, y_pred)
    m_mape = mape(test, y_pred)
    eval_table.append({'model': name, 'MAE': m_mae, 'RMSE': m_rmse, 'MAPE%': m_mape})
    print(f"{name}: MAE={m_mae:.3f}, RMSE={m_rmse:.3f}, MAPE={m_mape:.2f}%")

eval_df = pd.DataFrame(eval_table).sort_values("RMSE")
eval_df.to_csv(os.path.join(OUTDIR, "evaluation_table.csv"), index=False)

# ----------------------------
#  Plot results
# ----------------------------
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12,6))
train.plot(ax=ax, label="Train", marker='o')
test.plot(ax=ax, label="Test (Actual)", marker='o')
for name, info in results.items():
    y_pred = info['forecast']
    y_pred.index = test.index
    y_pred.plot(ax=ax, label=f"Forecast: {name}", linestyle='--')
ax.set_title(f"Forecasting {PRODUCT} - Train/Test split")
ax.legend()
plt.tight_layout()
plotfile = os.path.join(OUTDIR, "forecast_comparison.png")
fig.savefig(plotfile, dpi=150)
plt.close(fig)
print("Saved plot:", plotfile)

# Also save individual model residual plots
for name, info in results.items():
    try:
        model = info['model']
        fig2, ax2 = plt.subplots(2,1, figsize=(10,8))
        # predictions on train for residuals (if available)
        try:
            if hasattr(model, 'fittedvalues'):
                fitted = model.fittedvalues
            else:
                # SARIMAX result
                fitted = model.fittedvalues
            fitted.plot(ax=ax2[0], title=f"Fitted values ({name})")
            train.plot(ax=ax2[0], label='Train actual', alpha=0.6)
            ax2[0].legend()
            residuals = train - fitted
            residuals.plot(ax=ax2[1], title=f"Residuals ({name})")
        except Exception:
            ax2[0].text(0.2, 0.5, 'No fitted values available', transform=ax2[0].transAxes)
            ax2[1].text(0.2, 0.5, 'No residuals available', transform=ax2[1].transAxes)
        plt.tight_layout()
        fname = os.path.join(OUTDIR, f"{name}_diagnostics.png")
        fig2.savefig(fname, dpi=140)
        plt.close(fig2)
    except Exception as e:
        print(f"Plotting diagnostics for {name} failed:", e)

# ----------------------------
#  Build a simple offline HTML report
# ----------------------------
html_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forecast Report - {{ product }}</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 28px; color:#222;}
    h1,h2 { color:#2b5f8f }
    table { border-collapse: collapse; width: 70%; margin-bottom: 20px;}
    th, td { border: 1px solid #ccc; padding: 8px; text-align:center;}
    th { background:#f0f4f8; }
    .plot { max-width: 100%; height: auto; margin-bottom: 20px; border:1px solid #ddd; padding:6px; background:white;}
    .section { margin-bottom: 36px; }
    .note { color:#666; font-size:0.95rem; }
  </style>
</head>
<body>
  <h1>Forecast Report - {{ product }}</h1>
  <p class="note">Auto-generated report. Train/Test split: {{ n_train }} / {{ n_test }}. Frequency used: {{ freq }}. Seasonal periods (if used): {{ seasonal_periods }}.</p>

  <div class="section">
    <h2>1) Data summary</h2>
    <p>Data points: {{ n_total }}. Train from {{ train_start }} to {{ train_end }}; Test from {{ test_start }} to {{ test_end }}.</p>
  </div>

  <div class="section">
    <h2>2) Evaluation table</h2>
    <table>
      <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE (%)</th></tr></thead>
      <tbody>
        {% for row in eval_rows %}
        <tr>
          <td>{{ row.model }}</td>
          <td>{{ "%.3f"|format(row.MAE) }}</td>
          <td>{{ "%.3f"|format(row.RMSE) }}</td>
          <td>{{ "%.2f"|format(row.MAPE) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>3) Forecast vs Actual</h2>
    <img src="{{ plotfile }}" class="plot" alt="forecast comparison">
  </div>

  <div class="section">
    <h2>4) Diagnostics</h2>
    {% for name in model_names %}
      <h3>{{ name }}</h3>
      <img src="{{ name }}_diagnostics.png" class="plot" alt="{{ name }} diagnostics">
    {% endfor %}
  </div>

  <div class="section">
    <h2>5) Short Interpretation / Suggested conclusion</h2>
    <p class="note">
      - ดูที่ค่า RMSE/MAPE: model ที่มีค่าน้อยที่สุดจะถือว่าดีที่สุดสำหรับ dataset นี้ (ในเชิง MSE/MAE ที่วัดได้).<br>
      - ถ้า ARIMA (SARIMAX) ใช้ exogenous variables ได้ดี (และมี RMSE ต่ำกว่า) ให้พิจารณานำตัวแปรพวก price/temperature เข้า model ในการปฏิบัติจริง.<br>
      - Holt-Winters เหมาะเมื่อมี seasonality ชัด (ในที่นี้ใช้ seasonal_periods={{ seasonal_periods }}).<br>
    </p>
  </div>

  <footer style="margin-top:40px;color:#777;font-size:0.9rem;">
    Generated by separate.py
  </footer>
</body>
</html>
"""

t = Template(html_template)
rendered = t.render(
    product=PRODUCT,
    n_train=n_train,
    n_test=len(test),
    freq=FREQ,
    seasonal_periods=SEASONAL_PERIODS,
    n_total=n,
    train_start=series.index[0].date() if hasattr(series.index[0], 'date') else series.index[0],
    train_end=series.index[n_train-1].date() if hasattr(series.index[n_train-1], 'date') else series.index[n_train-1],
    test_start=series.index[n_train].date() if hasattr(series.index[n_train], 'date') else series.index[n_train],
    test_end=series.index[-1].date() if hasattr(series.index[-1], 'date') else series.index[-1],
    eval_rows=[{'model': r['model'], 'MAE': r['MAE'], 'RMSE': r['RMSE'], 'MAPE': r['MAPE%']} for r in eval_table],
    plotfile=os.path.basename(plotfile),
    model_names=list(results.keys())
)

# copy image files into OUTDIR root and write HTML there
html_path = os.path.join(OUTDIR, "report.html")
# ensure relative image paths in same folder
# Move/Save the main plot into OUTDIR (already saved)
# Save also diagnostic images (they are saved in OUTDIR above)
with open(html_path, "w", encoding="utf-8") as f:
    f.write(rendered)

print("Report generated:", html_path)
print("All outputs are in folder:", OUTDIR)

# ----------------------------
#  End of script
# ----------------------------
