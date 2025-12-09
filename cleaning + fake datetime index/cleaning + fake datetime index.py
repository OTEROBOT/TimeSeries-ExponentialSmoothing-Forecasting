# prepare_data.py
import os
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
BASE_DIR = r"C:\xampp\htdocs\2567\ExponentialSmoothing\TimeSeries-ExponentialSmoothing-Forecasting"
CSV_IN = os.path.join(BASE_DIR, "time series data.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "time_series_cleaned.csv")
JSON_FOR_DASHBOARD = os.path.join(BASE_DIR, "dashboard", "dataset.json")  # optional
PRODUCT_COL = "ProductP4"   # <-- เลือก column ที่เราจะทำ forecasting (P1..P5)
START_DATE_IF_NO_DATE = "2019-01-01"  # <-- ถ้าต้องสร้าง fake dates
FREQ = "D"  # D=day, W=week, M=month, H=hour  (ปรับถ้าจำเป็น)
# -----------------------------

df = pd.read_csv(CSV_IN)
print("Loaded:", CSV_IN)
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# 1) ถ้ามีคอลัมน์ 'date' อยู่แล้ว -> แปลงเป็น datetime
if "date" in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')

else:
    # ถ้าไม่มี date แต่มี 't' เป็นลำดับเวลา (1,2,3,...)
    if 't' in df.columns:
        # ตรวจว่า t เป็น integer/float
        if not np.issubdtype(df['t'].dtype, np.number):
            # ถ้า t เก็บเป็น string ให้แปลง
            df['t'] = pd.to_numeric(df['t'], errors='coerce')

        # สร้าง datetime index จาก t โดย map t=1 -> START_DATE_IF_NO_DATE
        # สมมติ t เป็น 1-based index
        df = df.sort_values('t').reset_index(drop=True)
        # สร้าง date series
        start = pd.to_datetime(START_DATE_IF_NO_DATE)
        df['date'] = df['t'].astype(int).apply(lambda x: start + pd.to_timedelta(x-1, unit=FREQ))
        df = df.set_index('date')
    else:
        raise ValueError("ไม่มีคอลัมน์ 'date' หรือ 't' ในไฟล์ CSV — ต้องมีอย่างใดอย่างหนึ่ง")

# 2) เลือก column target (ProductP4) และแปลงเป็น numeric
if PRODUCT_COL not in df.columns:
    raise ValueError(f"ไม่พบ column {PRODUCT_COL} ในไฟล์. เลือกคอลัมน์ที่มีอยู่: {df.columns.tolist()}")

series = pd.to_numeric(df[PRODUCT_COL], errors='coerce')

# 3) ตรวจ missing / duplicates
print("Original data points:", len(series))
missing = series.isna().sum()
print("Missing values in target:", missing)

# ถ้ามี missing น้อย -> interpolate
if missing > 0 and missing < len(series)*0.2:
    # แนะนำ linear interpolation (ถ้าเป็น time-series)
    series = series.interpolate(method='time').ffill().bfill()
    print("Applied interpolation for missing values.")
else:
    # ถ้ามี missing เยอะ -> แจ้งผู้ใช้ (คุณอาจเลือกลบหรือใช้ model-based imputation)
    print("Warning: many missing values. Consider more advanced imputation or clean source data.")

# ตรวจ duplicates index
dups = series.index.duplicated().sum()
if dups:
    print(f"Found {dups} duplicated datetime index values. Aggregating by mean.")
    series = series.groupby(level=0).mean()

# 4) ตั้ง frequency และตรวจความสม่ำเสมอ
try:
    # set freq if possible (will raise if non-regular)
    inferred = pd.infer_freq(series.index)
    print("Inferred freq:", inferred)
    if inferred is None:
        # force freq by reindexing with a complete range
        full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq=FREQ)
        series = series.reindex(full_idx)
        # fill new missing slots (if any) with interpolation
        series = series.interpolate(method='time').ffill().bfill()
        print("Reindexed to regular freq and filled gaps.")
    else:
        series = series.asfreq(inferred)
except Exception as e:
    print("Warning: could not infer freq automatically:", e)
    # fallback: reindex with FREQ
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq=FREQ)
    series = series.reindex(full_idx).interpolate(method='time').ffill().bfill()
    print("Reindexed with fallback FREQ and filled gaps.")

# 5) Trim/clip negative values (ถ้าไม่สมเหตุผล)
if (series < 0).any():
    print("Warning: negative values found in series. Replacing with 0 (or investigate).")
    series[series < 0] = 0

# 6) Final checks and save
print("Final length:", len(series))
print(series.head())

# Save cleaned CSV (index as date)
out_df = series.to_frame(name=PRODUCT_COL)
out_df.to_csv(CLEANED_CSV, index=True)
print("Saved cleaned CSV:", CLEANED_CSV)

# Optional: write small JSON for dashboard (each product as array)
import json, pathlib
pathp = pathlib.Path(JSON_FOR_DASHBOARD)
pathp.parent.mkdir(parents=True, exist_ok=True)
dashboard_obj = {col: pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(0).tolist()
                 for col in ['ProductP1','ProductP2','ProductP3','ProductP4','ProductP5'] if col in df.columns}
with open(JSON_FOR_DASHBOARD, 'w', encoding='utf-8') as f:
    json.dump(dashboard_obj, f, ensure_ascii=False, indent=2)
print("Saved dashboard JSON:", JSON_FOR_DASHBOARD)
