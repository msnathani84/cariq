"""
Module 2 — 2-Year Maintenance Cost Prediction
Complete Pipeline: v3.csv → Feature Engineering → Label Generation → Model Training
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — BRAND CLASS MAPPING
# Source: J.D. Power VDS + Consumer Reports + HighOnCars 2025
# ─────────────────────────────────────────────────────────────────────────────

BRAND_CLASS = {
    # Class 0 — Economy High Reliability (₹5k–₹9k/yr)
    'maruti': 0, 'suzuki': 0, 'hyundai': 0, 'tata': 0,
    'datsun': 0, 'chevrolet': 0, 'renault': 0, 'daewoo': 0,

    # Class 1 — Mid Reliability (₹7k–₹12k/yr)
    'honda': 1, 'toyota': 1, 'nissan': 1, 'ford': 1,
    'mitsubishi': 1, 'fiat': 1, 'isuzu': 1,

    # Class 2 — Premium (₹9k–₹14k/yr)
    'mahindra': 2, 'volkswagen': 2, 'skoda': 2, 'kia': 2,
    'jeep': 2, 'mg': 2,

    # Class 3 — Luxury (₹40k–₹70k+/yr)
    'bmw': 3, 'audi': 3, 'mercedes': 3, 'mercedes-benz': 3,
    'land': 3, 'volvo': 3, 'jaguar': 3, 'lexus': 3, 'porsche': 3,
}

def get_brand_class(name):
    brand = str(name).split()[0].lower().strip()
    return BRAND_CLASS.get(brand, 1)  # default Mid if unknown

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BASE ANNUAL SERVICE COST
# Source: RideNRepair 2026 + HighOnCars 2025
# ─────────────────────────────────────────────────────────────────────────────

BASE_ANNUAL_SERVICE_COST = {
    (0, 'hatchback'): 7500,
    (0, 'sedan'):     9500,
    (0, 'suv'):      10000,
    (1, 'hatchback'): 9500,
    (1, 'sedan'):    11000,
    (1, 'suv'):      12500,
    (2, 'hatchback'): 11000,
    (2, 'sedan'):    13000,
    (2, 'suv'):      14000,
    (3, 'hatchback'): 50000,
    (3, 'sedan'):    55000,
    (3, 'suv'):      60000,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FUEL TYPE MULTIPLIERS
# Source: Autocar India 60,000 km maintenance cost study
# ─────────────────────────────────────────────────────────────────────────────

FUEL_MULTIPLIER = {
    'Petrol':   1.00,
    'Diesel':   1.25,
    'CNG':      1.10,
    'LPG':      1.10,
    'Electric': 0.65,
}

ANNUAL_SERVICE_VISITS = {
    'Petrol':   2.0,
    'Diesel':   2.5,
    'CNG':      2.2,
    'LPG':      2.2,
    'Electric': 1.5,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AGE PENALTY
# Source: Autocar India labour escalation documentation
# ─────────────────────────────────────────────────────────────────────────────

def age_penalty(vehicle_age):
    if vehicle_age > 8:
        return ((vehicle_age - 5) ** 2) * 900
    elif vehicle_age > 5:
        return ((vehicle_age - 5) ** 2) * 600
    elif vehicle_age > 3:
        return (vehicle_age - 3) * 1500
    else:
        return 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — KILOMETRE WEAR COST
# Source: MyCarhelpline (₹0.40/km for Swift documented)
# ─────────────────────────────────────────────────────────────────────────────

COST_PER_KM = {0: 0.40, 1: 0.55, 2: 0.75, 3: 2.50}

def km_wear_cost(km_driven, brand_class):
    if km_driven > 50000:
        return (km_driven - 50000) * COST_PER_KM.get(brand_class, 0.55)
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — UNPLANNED REPAIR COMPONENT
# Source: PolicyBazaar Motor Insurance Claim Analysis 2024
# ─────────────────────────────────────────────────────────────────────────────

CLAIM_COST_PER_INCIDENT = {
    'hatchback': 21084,
    'sedan':     21084,
    'suv':       29032,
    'electric':  39021,
}

CITY_CLAIM_FREQUENCY_PER_YEAR = {
    'Metro':  0.15,
    'Tier-2': 0.12,
    'Tier-3': 0.10,
}

def unplanned_repair_cost(seg, city_tier, vehicle_age):
    claim_cost = CLAIM_COST_PER_INCIDENT.get(seg, 21084)
    freq       = CITY_CLAIM_FREQUENCY_PER_YEAR.get(city_tier, 0.12)
    age_mult   = 1.0 + (max(0, vehicle_age - 4) * 0.04)
    age_mult   = min(age_mult, 1.80)
    return claim_cost * freq * age_mult * 2

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ENGINE SIZE ADJUSTMENT
# Source: Autocar India
# ─────────────────────────────────────────────────────────────────────────────

def engine_adjustment(engine_cc):
    if engine_cc > 2000:
        return 1.12
    elif engine_cc > 1500:
        return 1.05
    else:
        return 1.00

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CITY TIER LABOUR ADJUSTMENT
# Source: RideNRepair 2026 city pricing table
# ─────────────────────────────────────────────────────────────────────────────

CITY_LABOUR_MULT = {
    'Metro':  1.22,
    'Tier-2': 1.00,
    'Tier-3': 0.88,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — LABEL GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_maintenance_label(row):
    bc     = int(row['brand_class'])
    age    = float(row['vehicle_age'])
    km     = float(row['km_driven'])
    fuel   = int(row['fuel_enc'])
    engine = float(row['engine_cc'])
    seats  = float(row['seats'])
    city   = row.get('city_tier', 'Metro')

    fuel_name = {0:'Petrol', 1:'Diesel', 2:'CNG', 3:'LPG', 4:'Electric'}.get(fuel, 'Petrol')

    # Segment
    if seats >= 7 or engine >= 1800:
        seg = 'suv'
    elif engine >= 1200:
        seg = 'sedan'
    else:
        seg = 'hatchback'

    # Step 1 & 2: Base 2-year service cost
    base_annual = BASE_ANNUAL_SERVICE_COST.get((bc, seg), 10000)
    base_2yr = base_annual * 2

    # Step 3: Fuel multiplier
    base_2yr *= FUEL_MULTIPLIER.get(fuel_name, 1.0)

    # Step 4: Service frequency
    freq = ANNUAL_SERVICE_VISITS.get(fuel_name, 2.0)
    base_2yr *= (freq / 2.0)

    # Step 5: Age penalty
    base_2yr += age_penalty(age)

    # Step 6: KM wear
    base_2yr += km_wear_cost(km, bc)

    # Step 7: Engine size
    base_2yr *= engine_adjustment(engine)

    # Step 8: Unplanned repair
    base_2yr += unplanned_repair_cost(seg, city, age)

    # Step 9: City labour — SKIPPED during training
    # All v3 rows default to Metro; applying x1.22 universally inflates
    # all labels identically and causes validation overprediction.
    # Apply city adjustment at inference time instead.
    # base_2yr *= CITY_LABOUR_MULT.get(city, 1.0)

    # Step 10: High power adjustment
    max_power = float(row.get('max_power_bhp', 85))
    if max_power > 150:
        base_2yr *= 1.08
    elif max_power > 100:
        base_2yr *= 1.03

    # Step 11: ±10% noise
    noise = np.clip(np.random.normal(1.0, 0.10), 0.82, 1.18)
    base_2yr *= noise

    # Step 12: Clamp
    return round(max(15000, min(base_2yr, 250000)), -2)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FROM v3.csv
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df_raw):
    df = df_raw.copy()

    # Parse numeric columns from strings
    df['engine_cc']      = df['engine'].str.extract(r'([\d.]+)').astype(float)
    df['max_power_bhp']  = df['max_power'].str.extract(r'([\d.]+)').astype(float)
    df['mileage_kmpl']   = df['mileage'].str.extract(r'([\d.]+)').astype(float)

    # Derived features
    df['vehicle_age']    = 2024 - df['year']
    df['brand_class']    = df['name'].apply(get_brand_class)

    # Encodings
    fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
    df['fuel_enc']       = df['fuel'].map(fuel_map).fillna(0).astype(int)

    trans_map = {'Manual': 0, 'Automatic': 1}
    df['trans_enc']      = df['transmission'].map(trans_map).fillna(0).astype(int)

    owner_map = {
        'First Owner': 1, 'Second Owner': 2,
        'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 2
    }
    df['owner_enc']      = df['owner'].map(owner_map).fillna(2).astype(int)

    seller_map = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
    df['seller_enc']     = df['seller_type'].map(seller_map).fillna(0).astype(int)

    # Mileage band (wear buckets)
    df['mileage_band']   = pd.cut(
        df['km_driven'],
        bins=[0, 30000, 60000, 100000, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # City tier — v3 has no city column, default Metro
    df['city_tier']      = 'Metro'

    # Fill remaining nulls with median
    for col in ['engine_cc', 'max_power_bhp', 'mileage_kmpl', 'seats']:
        df[col] = df[col].fillna(df[col].median())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("MODULE 2 — MAINTENANCE COST TRAINING PIPELINE")
print("=" * 60)

# 1. Load
print("\n[1/6] Loading v3.csv...")
df_raw = pd.read_csv('v3.csv')
print(f"     Loaded {len(df_raw):,} rows")

# 2. Engineer features
print("\n[2/6] Engineering features...")
df = engineer_features(df_raw)
print(f"     Features ready. Nulls remaining: {df[['engine_cc','max_power_bhp','mileage_kmpl','seats']].isnull().sum().sum()}")

# 3. Generate labels
print("\n[3/6] Generating maintenance_2yr labels...")
df['maintenance_2yr'] = df.apply(generate_maintenance_label, axis=1)
print(f"     Label stats:")
print(f"       Min:    ₹{df['maintenance_2yr'].min():,.0f}")
print(f"       Max:    ₹{df['maintenance_2yr'].max():,.0f}")
print(f"       Mean:   ₹{df['maintenance_2yr'].mean():,.0f}")
print(f"       Median: ₹{df['maintenance_2yr'].median():,.0f}")

# Show distribution by brand class
print("\n     Mean maintenance by brand class:")
for bc, name in [(0,'Economy'),(1,'Mid'),(2,'Premium'),(3,'Luxury')]:
    subset = df[df['brand_class'] == bc]['maintenance_2yr']
    if len(subset):
        print(f"       Class {bc} ({name}): ₹{subset.mean():,.0f}  (n={len(subset)})")

# 4. Prepare ML features
MAINTENANCE_FEATURES = [
    'brand_class', 'vehicle_age', 'engine_cc', 'fuel_enc',
    'km_driven', 'max_power_bhp', 'seats', 'mileage_kmpl',
    'owner_enc', 'mileage_band', 'trans_enc',
]

print(f"\n[4/6] Preparing train/test split...")
df_model = df[MAINTENANCE_FEATURES + ['maintenance_2yr']].dropna()
print(f"     Rows after dropping nulls: {len(df_model):,}")

X = df_model[MAINTENANCE_FEATURES]
y = df_model['maintenance_2yr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# 5. Train & compare models
print("\n[5/6] Training models...")

models = {
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.10,
        max_depth=5, min_samples_leaf=4, random_state=42
    ),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=5,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=15,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    results[name] = {'model': model, 'mae': mae, 'r2': r2, 'mape': mape}
    print(f"     {name:25s}  MAE=₹{mae:,.0f}  R²={r2:.4f}  MAPE={mape:.1f}%")

# Pick winner
winner_name = min(results, key=lambda k: results[k]['mae'])
winner_model = results[winner_name]['model']
print(f"\n     ✓ Winner: {winner_name}")

# Feature importance
if hasattr(winner_model, 'feature_importances_'):
    importances = winner_model.feature_importances_
else:
    importances = np.ones(len(MAINTENANCE_FEATURES)) / len(MAINTENANCE_FEATURES)
fi = pd.Series(importances, index=MAINTENANCE_FEATURES).sort_values(ascending=False)
print("\n     Feature importances (winner):")
for feat, imp in fi.items():
    bar = '█' * int(imp * 40)
    print(f"       {feat:20s} {imp:.4f}  {bar}")

# 6. Save model
print("\n[6/6] Saving model...")
output = {
    'model': winner_model,
    'model_name': winner_name,
    'features': MAINTENANCE_FEATURES,
    'metrics': {k: {m: v for m, v in v.items() if m != 'model'} for k, v in results.items()},
    'label_stats': {
        'min': df['maintenance_2yr'].min(),
        'max': df['maintenance_2yr'].max(),
        'mean': df['maintenance_2yr'].mean(),
        'median': df['maintenance_2yr'].median(),
    }
}
with open('models/maintenance_model.pkl', 'wb') as f:
    pickle.dump(output, f)
print("     Saved → maintenance_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION AGAINST KNOWN REAL-WORLD CASES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALIDATION AGAINST REAL-WORLD CASES")
print("=" * 60)

VALIDATION_CASES = [
    {
        'name':     'Maruti Swift 2020 Petrol Metro 45k km',
        'expected': '₹18,000–₹25,000',
        'features': {'brand_class':0,'vehicle_age':4,'km_driven':45000,
                     'fuel_enc':0,'engine_cc':1197,'max_power_bhp':82,
                     'seats':5,'mileage_kmpl':23,'owner_enc':1,
                     'mileage_band':0,'trans_enc':0}
    },
    {
        'name':     'Hyundai Creta 2018 Diesel Metro 80k km',
        'expected': '₹38,000–₹55,000',
        'features': {'brand_class':0,'vehicle_age':6,'km_driven':80000,
                     'fuel_enc':1,'engine_cc':1493,'max_power_bhp':115,
                     'seats':5,'mileage_kmpl':21,'owner_enc':1,
                     'mileage_band':1,'trans_enc':0}
    },
    {
        'name':     'Honda City 2016 Petrol Tier-2 65k km',
        'expected': '₹35,000–₹50,000',
        'features': {'brand_class':1,'vehicle_age':8,'km_driven':65000,
                     'fuel_enc':0,'engine_cc':1497,'max_power_bhp':119,
                     'seats':5,'mileage_kmpl':17,'owner_enc':2,
                     'mileage_band':1,'trans_enc':0}
    },
    {
        'name':     'Mahindra Scorpio 2014 Diesel Metro 1.1L km',
        'expected': '₹65,000–₹95,000',
        'features': {'brand_class':2,'vehicle_age':10,'km_driven':110000,
                     'fuel_enc':1,'engine_cc':2179,'max_power_bhp':120,
                     'seats':7,'mileage_kmpl':15,'owner_enc':2,
                     'mileage_band':2,'trans_enc':0}
    },
    {
        'name':     'BMW 3 Series 2015 Diesel Metro 90k km',
        'expected': '₹1,20,000–₹1,80,000',
        'features': {'brand_class':3,'vehicle_age':9,'km_driven':90000,
                     'fuel_enc':1,'engine_cc':1995,'max_power_bhp':190,
                     'seats':5,'mileage_kmpl':17,'owner_enc':2,
                     'mileage_band':2,'trans_enc':1}
    },
]

for case in VALIDATION_CASES:
    row_df = pd.DataFrame([case['features']])[MAINTENANCE_FEATURES]
    pred = winner_model.predict(row_df)[0]
    print(f"\n  {case['name']}")
    print(f"    Expected : {case['expected']}")
    print(f"    Predicted: ₹{pred:,.0f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)