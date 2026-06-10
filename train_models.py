"""
CarIQ — Module 1 & 3 Training Script
=====================================
Module 1: Price Prediction — Random Forest on real v3 dataset
Module 3: Depreciation — same RF model, but simulates +2 years of real
          wear-and-tear using average annual km from the dataset itself.

Post-prediction adjustments (Module 1 & 3):
  - City Tier  : Tier 1 / Tier 2 / Tier 3 multiplier applied after model output
  - Body Type  : SUV / Sedan / Hatchback / MUV / Luxury Sedan multiplier applied after model output
  Both are user-supplied at inference time; multiplier tables are saved as
  models/market_adjustment_config.pkl so app.py loads them without hardcoding.

Run: python train_models.py
Output: models/price_model.pkl, models/depreciation_model.pkl,
        models/brand_annual_km.pkl, models/feature_cols.pkl,
        models/market_adjustment_config.pkl          ← NEW
"""

import os, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "v3.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CURRENT_YEAR   = 2026
FORECAST_YEARS = 2

print("\n" + "="*60)
print("  CarIQ Model Training — Module 1 & 3")
print("="*60)

# ── STEP 1: LOAD & CLEAN ──────────────────────────────────────────────────────
print("\n[1/8] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"      Raw rows: {len(df):,}")

def extract_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r'([\d.]+)', expand=False),
        errors='coerce'
    )

df['mileage_num'] = extract_numeric(df['mileage'])
df['engine_cc']   = extract_numeric(df['engine'])
df['power_bhp']   = extract_numeric(df['max_power'])
df['brand']       = df['name'].str.split().str[0]
df['vehicle_age'] = CURRENT_YEAR - df['year']

owner_map = {
    'First Owner': 1, 'Second Owner': 2,
    'Third Owner': 3, 'Fourth & Above Owner': 4,
    'Test Drive Car': 1
}
df['owner_num'] = df['owner'].map(owner_map).fillna(2)

def mileage_band(row):
    expected = row['vehicle_age'] * 17500
    if expected == 0: return 'Normal'
    ratio = row['km_driven'] / expected
    if ratio < 0.6:  return 'Low'
    if ratio < 1.2:  return 'Normal'
    if ratio < 1.8:  return 'High'
    return 'VeryHigh'

df['mileage_band'] = df.apply(mileage_band, axis=1)

df = df[df['selling_price'] > 10000]
df = df[df['km_driven'] < 500000]
df = df[df['vehicle_age'] >= 0]
df = df[df['year'] >= 2000]
df = df[df['fuel'].isin(['Petrol','Diesel','CNG','LPG'])]
df = df.dropna(subset=['engine_cc','power_bhp','mileage_num'])

print(f"      After cleaning: {len(df):,} rows")
print(f"      Year range: {df['year'].min()}–{df['year'].max()}")
print(f"      Price range: ₹{df['selling_price'].min():,}–₹{df['selling_price'].max():,}")
print(f"      Brands: {df['brand'].nunique()}")

# ── STEP 2: COMPUTE AVERAGE ANNUAL KM PER BRAND+FUEL ─────────────────────────
print("\n[2/8] Computing real-world wear rates from dataset...")

df_age_ok = df[df['vehicle_age'] > 0].copy()
df_age_ok['annual_km'] = df_age_ok['km_driven'] / df_age_ok['vehicle_age']

brand_fuel_km = (df_age_ok
    .groupby(['brand','fuel'])['annual_km']
    .median()
    .reset_index()
    .rename(columns={'annual_km': 'avg_annual_km'}))

global_median_km = df_age_ok['annual_km'].median()
print(f"      Global median annual km: {global_median_km:,.0f} km/year")

joblib.dump(brand_fuel_km,      os.path.join(MODEL_DIR, "brand_annual_km.pkl"))
joblib.dump(global_median_km,   os.path.join(MODEL_DIR, "global_median_km.pkl"))

# ── STEP 3: FEATURE ENGINEERING ──────────────────────────────────────────────
print("\n[3/8] Feature engineering...")

NUMERICAL_FEATURES   = ['vehicle_age','km_driven','owner_num','mileage_num','engine_cc','power_bhp','seats']
CATEGORICAL_FEATURES = ['brand','fuel','transmission','seller_type','mileage_band']
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET               = 'selling_price'

df['seats'] = df['seats'].fillna(df['seats'].mode()[0])

X      = df[ALL_FEATURES].copy()
y_log  = np.log1p(df[TARGET])

print(f"      Features: {len(ALL_FEATURES)}")

joblib.dump(ALL_FEATURES,         os.path.join(MODEL_DIR, "feature_cols.pkl"))
joblib.dump(NUMERICAL_FEATURES,   os.path.join(MODEL_DIR, "numerical_cols.pkl"))
joblib.dump(CATEGORICAL_FEATURES, os.path.join(MODEL_DIR, "categorical_cols.pkl"))

X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"      Train: {len(X_train):,} · Val: {len(X_val):,} · Test: {len(X_test):,}")

# ── STEP 4: BUILD PIPELINE & TUNE ────────────────────────────────────────────
print("\n[4/8] Building pipeline & tuning Random Forest...")

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUMERICAL_FEATURES),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_FEATURES),
], remainder='drop')

param_grid = {
    'rf__n_estimators':     [200, 300],
    'rf__max_depth':        [20, None],
    'rf__min_samples_leaf': [1, 2],
    'rf__max_features':     ['sqrt'],
}

pipeline = Pipeline([
    ('pre', preprocessor),
    ('rf',  RandomForestRegressor(random_state=42, n_jobs=-1))
])

print("      Running GridSearchCV (this takes ~2 min)...")
gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

best_params  = gs.best_params_
price_model  = gs.best_estimator_

print(f"      Best params: {best_params}")
print(f"      Best CV R²: {gs.best_score_:.4f}")

# ── STEP 5: EVALUATE MODULE 1 ─────────────────────────────────────────────────
print("\n[5/8] Evaluating Module 1 (Price Prediction)...")

def evaluate(model, X, y_log_true, label):
    y_log_pred = model.predict(X)
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_log_true, y_log_pred)
    print(f"      [{label}]  R²={r2:.4f}  MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}")
    return r2, mae, rmse

r2_val, mae_val, _           = evaluate(price_model, X_val,  y_val,  "Validation")
r2_test, mae_test, rmse_test = evaluate(price_model, X_test, y_test, "Test (held-out)")

joblib.dump(price_model, os.path.join(MODEL_DIR, "price_model.pkl"))
print(f"\n      ✓ price_model.pkl saved")

# ── STEP 6: MODULE 3 — DEPRECIATION ──────────────────────────────────────────
print("\n[6/8] Building Module 3 (Depreciation via real wear simulation)...")

km_lookup = brand_fuel_km.set_index(['brand','fuel'])['avg_annual_km'].to_dict()

df_dep = df.copy()
X_full = df_dep[ALL_FEATURES].copy()
df_dep['current_pred'] = np.expm1(price_model.predict(X_full))

df_dep['future_age']      = df_dep['vehicle_age'] + FORECAST_YEARS
df_dep['annual_km_rate']  = df_dep.apply(
    lambda r: km_lookup.get((r['brand'], r['fuel']), global_median_km), axis=1)
df_dep['future_km']       = df_dep['km_driven'] + (df_dep['annual_km_rate'] * FORECAST_YEARS)

def future_mileage_band(row):
    expected = row['future_age'] * 17500
    if expected == 0: return 'Normal'
    ratio = row['future_km'] / expected
    if ratio < 0.6:  return 'Low'
    if ratio < 1.2:  return 'Normal'
    if ratio < 1.8:  return 'High'
    return 'VeryHigh'

df_dep['future_mileage_band'] = df_dep.apply(future_mileage_band, axis=1)

X_future = X_full.copy()
X_future['vehicle_age']  = df_dep['future_age']
X_future['km_driven']    = df_dep['future_km']
X_future['mileage_band'] = df_dep['future_mileage_band']
X_future['owner_num']    = (X_future['owner_num'] + 1).clip(upper=4)

df_dep['future_pred']          = np.expm1(price_model.predict(X_future))
df_dep['depreciation_2yr_abs'] = (df_dep['current_pred'] - df_dep['future_pred']).clip(lower=0)
df_dep['depreciation_2yr_pct'] = (
    df_dep['depreciation_2yr_abs'] / df_dep['current_pred'].clip(lower=1) * 100
).clip(0, 80)

print(f"      Simulated {len(df_dep):,} depreciation trajectories")
print(f"      Avg 2-yr depreciation: {df_dep['depreciation_2yr_pct'].mean():.1f}%")
print(f"      Median 2-yr depreciation: {df_dep['depreciation_2yr_pct'].median():.1f}%")

y_dep_log = np.log1p(df_dep['future_pred'])

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_full, y_dep_log, test_size=0.20, random_state=42)

dep_pipeline = Pipeline([
    ('pre', ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_FEATURES),
    ], remainder='drop')),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=best_params.get('rf__max_depth', 20),
        min_samples_leaf=best_params.get('rf__min_samples_leaf', 1),
        max_features=best_params.get('rf__max_features', 'sqrt'),
        random_state=42, n_jobs=-1
    ))
])

dep_pipeline.fit(X_d_train, y_d_train)

y_dep_pred = dep_pipeline.predict(X_d_test)
r2_dep  = r2_score(y_d_test, y_dep_pred)
mae_dep = mean_absolute_error(np.expm1(y_d_test), np.expm1(y_dep_pred))
print(f"      Depreciation model — R²={r2_dep:.4f}  MAE=₹{mae_dep:,.0f}")

joblib.dump(dep_pipeline, os.path.join(MODEL_DIR, "depreciation_model.pkl"))
joblib.dump({
    'forecast_years':        FORECAST_YEARS,
    'current_year':          CURRENT_YEAR,
    'annual_km_rate_lookup': km_lookup,
    'global_median_km':      global_median_km,
}, os.path.join(MODEL_DIR, "depreciation_config.pkl"))

print(f"      ✓ depreciation_model.pkl saved")

# ── STEP 7: MARKET ADJUSTMENT CONFIG ─────────────────────────────────────────
# Post-prediction multipliers for city tier and body type.
# These are NOT learned — they are calibrated constants saved as config
# so app.py loads them from one place instead of hardcoding them.
#
# How to tune: if real-world feedback shows Mumbai premiums are higher,
# just edit CITY_TIER_MULTIPLIERS here and re-run train_models.py.
# No retraining needed — only the config pkl is updated.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Saving market adjustment config (city tier + body type)...")

# City Tier Multipliers
# ----------------------
# Applied as: adjusted_price = model_price * city_multiplier
# Tier 1 cities command a premium due to higher demand, better maintenance
# history, and stronger resale infrastructure.
CITY_TIER_MULTIPLIERS = {
    "Tier 1": 1.10,   # Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata
    "Tier 2": 1.04,   # Pune, Ahmedabad, Jaipur, Lucknow, Surat, Chandigarh
    "Tier 3": 1.00,   # All other cities — baseline, no adjustment
}

# City → Tier mapping for convenience (app.py can show city names, not tiers)
CITY_TO_TIER = {
    # Tier 1
    "Mumbai": "Tier 1", "Delhi": "Tier 1", "Bangalore": "Tier 1",
    "Bengaluru": "Tier 1", "Hyderabad": "Tier 1", "Chennai": "Tier 1",
    "Kolkata": "Tier 1",
    # Tier 2
    "Pune": "Tier 2", "Ahmedabad": "Tier 2", "Jaipur": "Tier 2",
    "Lucknow": "Tier 2", "Surat": "Tier 2", "Chandigarh": "Tier 2",
    "Nagpur": "Tier 2", "Indore": "Tier 2", "Bhopal": "Tier 2",
    "Coimbatore": "Tier 2", "Kochi": "Tier 2", "Vizag": "Tier 2",
    "Visakhapatnam": "Tier 2",
    # Everything else defaults to Tier 3 at runtime
}

# Body Type Multipliers
# ----------------------
# Applied as: adjusted_price = model_price * body_type_multiplier
# The base RF model sees engine_cc, power_bhp, seats — so it partially
# captures size/segment differences. These multipliers correct for the
# residual market premium that specs alone don't explain.
# Example: Toyota Fortuner (SUV) vs Toyota Camry (Sedan), same year/km
# → model gives similar predictions → multipliers correctly separate them.
BODY_TYPE_MULTIPLIERS = {
    "SUV":          1.12,   # Highest demand, strong resale in India
    "MUV":          1.06,   # Innova-class, fleet-friendly
    "Luxury Sedan": 1.08,   # Camry, Accord — niche but premium
    "Sedan":        1.03,   # City, Verna, Dzire
    "Hatchback":    1.00,   # Baseline — Swift, Alto, i20
    "Convertible":  1.05,   # Rare; slight curiosity premium
    "Other":        1.00,   # Unknown / not selected
}

market_adjustment_config = {
    "city_tier_multipliers": CITY_TIER_MULTIPLIERS,
    "city_to_tier":          CITY_TO_TIER,
    "body_type_multipliers": BODY_TYPE_MULTIPLIERS,
}

joblib.dump(market_adjustment_config, os.path.join(MODEL_DIR, "market_adjustment_config.pkl"))
print(f"      ✓ market_adjustment_config.pkl saved")
print(f"      City tiers defined : {list(CITY_TIER_MULTIPLIERS.keys())}")
print(f"      Body types defined : {list(BODY_TYPE_MULTIPLIERS.keys())}")
print(f"      Cities mapped      : {len(CITY_TO_TIER)}")


# ── apply_market_adjustments() ────────────────────────────────────────────────
def apply_market_adjustments(
    base_price: float,
    city_tier: str,          # "Tier 1" | "Tier 2" | "Tier 3"
    body_type: str,          # "SUV" | "Sedan" | "Hatchback" | "MUV" | "Luxury Sedan" | "Other"
    config: dict = None,
) -> dict:
    """
    Apply post-prediction city-tier and body-type multipliers to a base price.

    Parameters
    ----------
    base_price  : float  — raw model output in ₹
    city_tier   : str    — one of "Tier 1", "Tier 2", "Tier 3"
    body_type   : str    — user-selected body type from the app dropdown
    config      : dict   — market_adjustment_config dict (loaded from pkl).
                           If None, uses module-level constants (training time only).

    Returns
    -------
    dict with keys:
        adjusted_price      — final price after both multipliers (₹)
        city_multiplier     — multiplier applied for city tier
        body_multiplier     — multiplier applied for body type
        city_adjustment_amt — ₹ added/removed by city tier
        body_adjustment_amt — ₹ added/removed by body type
        total_adjustment_pct— total % change from base price

    Usage in app.py
    ---------------
        config = joblib.load("models/market_adjustment_config.pkl")
        result = apply_market_adjustments(
            base_price = model_prediction,
            city_tier  = user_selected_tier,   # or derive from city name via config['city_to_tier']
            body_type  = user_selected_body,
            config     = config,
        )
        final_price = result['adjusted_price']
    """
    if config is None:
        # Fallback to module-level constants (only available at train time)
        ct_map = CITY_TIER_MULTIPLIERS
        bt_map = BODY_TYPE_MULTIPLIERS
    else:
        ct_map = config["city_tier_multipliers"]
        bt_map = config["body_type_multipliers"]

    city_mult = ct_map.get(city_tier, 1.00)
    body_mult = bt_map.get(body_type, 1.00)

    after_city  = base_price * city_mult
    after_body  = after_city * body_mult

    return {
        "adjusted_price":       round(after_body),
        "city_multiplier":      city_mult,
        "body_multiplier":      body_mult,
        "city_adjustment_amt":  round(after_city - base_price),
        "body_adjustment_amt":  round(after_body - after_city),
        "total_adjustment_pct": round((after_body / base_price - 1) * 100, 2),
    }


# Quick sanity-check: Toyota SUV vs Sedan, same base price
print("\n      Sanity check — Toyota SUV vs Sedan @ ₹15,00,000 base, Tier 1 city:")
for bt in ["SUV", "Sedan", "Hatchback"]:
    r = apply_market_adjustments(1_500_000, "Tier 1", bt, market_adjustment_config)
    print(f"        {bt:<15} → ₹{r['adjusted_price']:>10,}  "
          f"(+{r['total_adjustment_pct']}% | "
          f"city +₹{r['city_adjustment_amt']:,} · body +₹{r['body_adjustment_amt']:,})")

# ── STEP 8: SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TRAINING COMPLETE — Summary")
print("="*60)
print(f"""
  MODULE 1 — Price Prediction (Random Forest)
    Validation R²  : {r2_val:.4f}
    Test R²        : {r2_test:.4f}
    Test MAE       : ₹{mae_test:,.0f}
    Test RMSE      : ₹{rmse_test:,.0f}
    Saved          : models/price_model.pkl

  MODULE 3 — Depreciation (Real-Wear Simulation + RF)
    Strategy       : +{FORECAST_YEARS}yr age, real brand+fuel km/yr from data
    R²             : {r2_dep:.4f}
    MAE            : ₹{mae_dep:,.0f}
    Saved          : models/depreciation_model.pkl
                     models/depreciation_config.pkl
                     models/brand_annual_km.pkl

  MARKET ADJUSTMENTS — Post-Prediction Weights
    City Tiers     : Tier 1 (+10%) · Tier 2 (+4%) · Tier 3 (baseline)
    Body Types     : SUV (+12%) · MUV (+6%) · Luxury Sedan (+8%)
                     Sedan (+3%) · Hatchback (baseline)
    Saved          : models/market_adjustment_config.pkl
    Function       : apply_market_adjustments(price, city_tier, body_type, config)

  HOW TO USE IN app.py
    config = joblib.load("models/market_adjustment_config.pkl")
    result = apply_market_adjustments(
        base_price = model.predict(...),
        city_tier  = city_to_tier.get(user_city, "Tier 3"),
        body_type  = user_selected_body_type,
        config     = config,
    )
    final_price = result['adjusted_price']

  NEXT STEP: python app.py
""")