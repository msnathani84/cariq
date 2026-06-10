from flask import Flask, jsonify, request, render_template
import os, io, datetime
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True) #(os.path.dirname(os.path.abspath(__file__)), ".env"))
import numpy as np

from supabase import create_client

# ── OPTIONAL: OpenCV ──────────────────────────────────────────────────────────
try:
    import cv2
    from PIL import Image
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[CarIQ] WARNING: opencv-python-headless / pillow not installed.")

# ── MODULE 1 & 3 — Lazy Load ML Models ───────────────────────────────────────
import joblib
import pandas as pd

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_price_model = None
_depr_model = None
_all_features = None
_num_cols = None
_cat_cols = None
_depr_config = None
_brand_annual_km = None
_km_lookup = {}
_global_median_km = 15000
_forecast_years = 2
_current_year = 2026
ML_MODELS_AVAILABLE = False


def load_ml_models():
    global _price_model, _depr_model, _all_features, _num_cols, _cat_cols
    global _depr_config, _brand_annual_km, _km_lookup
    global _global_median_km, _forecast_years, _current_year
    global ML_MODELS_AVAILABLE

    if ML_MODELS_AVAILABLE:
        return

    _price_model = joblib.load(os.path.join(_MODEL_DIR, "price_model.pkl"))
    _depr_model = joblib.load(os.path.join(_MODEL_DIR, "depreciation_model.pkl"))
    _all_features = joblib.load(os.path.join(_MODEL_DIR, "feature_cols.pkl"))
    _num_cols = joblib.load(os.path.join(_MODEL_DIR, "numerical_cols.pkl"))
    _cat_cols = joblib.load(os.path.join(_MODEL_DIR, "categorical_cols.pkl"))
    _depr_config = joblib.load(os.path.join(_MODEL_DIR, "depreciation_config.pkl"))
    _brand_annual_km = joblib.load(os.path.join(_MODEL_DIR, "brand_annual_km.pkl"))

    _km_lookup = _brand_annual_km.set_index(["brand", "fuel"])["avg_annual_km"].to_dict()
    _global_median_km = _depr_config["global_median_km"]
    _forecast_years = _depr_config["forecast_years"]
    _current_year = _depr_config["current_year"]

    ML_MODELS_AVAILABLE = True
    print("[CarIQ] ✓ ML models loaded on demand")

# ── MARKET ADJUSTMENT CONFIG ──────────────────────────────────────────────────
try:
    _adj_config = joblib.load(os.path.join(_MODEL_DIR, "market_adjustment_config.pkl"))
    MARKET_ADJ_AVAILABLE = True
    print("[CarIQ] ✓ Market adjustment config loaded (city tier + body type)")
except Exception as _adj_e:
    _adj_config = {
        "city_tier_multipliers": {"Tier 1": 1.10, "Tier 2": 1.04, "Tier 3": 1.00},
        "city_to_tier": {
            "Mumbai": "Tier 1", "Delhi": "Tier 1", "Bangalore": "Tier 1",
            "Bengaluru": "Tier 1", "Hyderabad": "Tier 1", "Chennai": "Tier 1",
            "Kolkata": "Tier 1", "Pune": "Tier 2", "Ahmedabad": "Tier 2",
            "Jaipur": "Tier 2", "Lucknow": "Tier 2", "Surat": "Tier 2",
            "Chandigarh": "Tier 2", "Nagpur": "Tier 2", "Indore": "Tier 2",
            "Bhopal": "Tier 2", "Coimbatore": "Tier 2", "Kochi": "Tier 2", "Vizag": "Tier 2",
        },
        "body_type_multipliers": {
            "SUV": 1.12, "MUV": 1.06, "Luxury Sedan": 1.08,
            "Sedan": 1.03, "Hatchback": 1.00, "Convertible": 1.05, "Other": 1.00,
        },
    }
    MARKET_ADJ_AVAILABLE = False
    print(f"[CarIQ] WARNING: market_adjustment_config.pkl not found — using hardcoded fallback")

# ── MODULE 2 — Load Maintenance Cost model ────────────────────────────────────
try:
    import pickle, pandas as pd
    _MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(_MODEL_DIR, "maintenance_model.pkl"), "rb") as _f:
        _maint_bundle = pickle.load(_f)
    _maint_model    = _maint_bundle["model"]
    _maint_features = _maint_bundle["features"]
    _BRAND_CLASS_M2 = {
        'maruti': 0, 'suzuki': 0, 'hyundai': 0, 'tata': 0,
        'datsun': 0, 'chevrolet': 0, 'renault': 0, 'daewoo': 0,
        'honda': 1, 'toyota': 1, 'nissan': 1, 'ford': 1,
        'mitsubishi': 1, 'fiat': 1, 'isuzu': 1,
        'mahindra': 2, 'volkswagen': 2, 'skoda': 2, 'kia': 2, 'jeep': 2, 'mg': 2,
        'bmw': 3, 'audi': 3, 'mercedes': 3, 'mercedes-benz': 3,
        'land': 3, 'volvo': 3, 'jaguar': 3, 'lexus': 3, 'porsche': 3,
    }
    _CITY_LABOUR_MULT = {'Metro': 1.22, 'Tier-2': 1.00, 'Tier-3': 0.88}
    _METRO_CITIES = {"Mumbai", "Delhi", "Bengaluru", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"}
    MODULE2_AVAILABLE = True
    print("[CarIQ] ✓ Module 2 (maintenance_model.pkl) loaded")
except Exception as _e2:
    MODULE2_AVAILABLE = False
    print(f"[CarIQ] WARNING: Module 2 not loaded ({_e2})")

# ── APP SETUP ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL')
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

app = Flask(__name__)

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[CarIQ] ✓ Supabase Storage connected")
    except Exception as e:
        print(f"[CarIQ] WARNING: Supabase Storage not connected ({e})")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CURRENT_YEAR = 2026

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
BRAND_CLASS = {
    "Maruti Suzuki": "economy_hr", "Hyundai": "economy_hr", "Tata": "economy_hr",
    "Honda": "mid", "Toyota": "mid", "Mahindra": "premium",
    "Ford": "mid", "Volkswagen": "mid",
    "BMW": "luxury", "Audi": "luxury", "Mercedes-Benz": "luxury",
}
BASE_VALUE  = {"economy_hr": 650000, "mid": 800000, "premium": 950000, "luxury": 2500000}
FUEL_FACTOR = {"Petrol": 1.00, "Diesel": 0.92, "CNG": 0.88, "Electric": 1.10}
OWNER_MAP = {
    "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
    "Fourth & Above Owner": 4, "Test Drive Car": 1, "buyer": 1, "seller": 1,
}

# ── MODULE 1 HELPERS ──────────────────────────────────────────────────────────
def apply_market_adjustments(base_price, city, body_type):
    ct_map = _adj_config["city_tier_multipliers"]
    bt_map = _adj_config["body_type_multipliers"]
    c2t    = _adj_config["city_to_tier"]
    city_tier = c2t.get(str(city).strip(), "Tier 3")
    body_key  = str(body_type).strip() if body_type else "Other"
    if body_key not in bt_map:
        body_key = "Other"
    city_mult  = ct_map.get(city_tier, 1.00)
    body_mult  = bt_map.get(body_key, 1.00)
    after_city = base_price * city_mult
    after_body = after_city * body_mult
    return {
        "adjusted_price":       max(50000, round(after_body)),
        "city_tier":            city_tier,
        "city_multiplier":      city_mult,
        "body_multiplier":      body_mult,
        "city_adjustment_amt":  round(after_city - base_price),
        "body_adjustment_amt":  round(after_body - after_city),
        "total_adjustment_pct": round((after_body / max(base_price, 1) - 1) * 100, 2),
    }

def _mileage_band(vehicle_age, km_driven):
    expected = vehicle_age * 17500
    if expected == 0: return "Normal"
    ratio = km_driven / expected
    if ratio < 0.6:  return "Low"
    if ratio < 1.2:  return "Normal"
    if ratio < 1.8:  return "High"
    return "VeryHigh"

def _build_feature_row(listing, vehicle_age=None, km_override=None,
                        owner_override=None, mileage_band_override=None):
    brand  = str(listing.get("brand", "")).split()[0]
    year   = int(listing.get("year", CURRENT_YEAR - 5))
    km     = int(km_override if km_override is not None else listing.get("km", 50000))
    fuel   = str(listing.get("fuel", "Petrol"))
    trans  = str(listing.get("transmission", "Manual"))
    seller = str(listing.get("seller_type", listing.get("seller_name", "Individual")))
    if seller not in ("Individual", "Dealer", "Trustmark Dealer"):
        seller = "Individual"
    age = vehicle_age if vehicle_age is not None else (CURRENT_YEAR - year)
    age = max(0, age)
    owner_raw = listing.get("owner", "buyer")
    owner_num = int(owner_override if owner_override is not None else OWNER_MAP.get(str(owner_raw), 2))
    mb = (mileage_band_override if mileage_band_override is not None else _mileage_band(age, km))
    row = {
        "vehicle_age": age, "km_driven": km, "owner_num": owner_num,
        "mileage_num": float(listing.get("mileage_num", 17.0)),
        "engine_cc":   float(listing.get("engine_cc", 1200.0)),
        "power_bhp":   float(listing.get("power_bhp", 75.0)),
        "seats":       float(listing.get("seats", 5)),
        "brand": brand, "fuel": fuel, "transmission": trans,
        "seller_type": seller, "mileage_band": mb,
    }
    return pd.DataFrame([row])[_all_features]

def _rule_based_price(listing):
    brand     = listing.get("brand", "")
    year      = int(listing.get("year", CURRENT_YEAR - 5))
    km        = int(listing.get("km", 50000))
    fuel      = listing.get("fuel", "Petrol")
    city      = listing.get("city", "")
    brand_cls = BRAND_CLASS.get(brand, "mid")
    base      = BASE_VALUE[brand_cls]
    age       = CURRENT_YEAR - year
    age_factor = max(0.30, 1.0 - (age * 0.10) - max(0, age - 5) * 0.02)
    km_penalty = max(0, (km - 50000) * 2)
    fuel_f     = FUEL_FACTOR.get(fuel, 1.0)
    metro_cities = {"Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata"}
    reg_risk = base * 0.10 if (fuel == "Diesel" and age >= 8 and city in metro_cities) else 0
    return max(50000, int(base * age_factor * fuel_f - km_penalty - reg_risk))

def get_fair_price_full(listing):
    if not ML_MODELS_AVAILABLE:
        try:
            load_ml_models()
        except Exception as e:
            print(f"[CarIQ] ML model load failed: {e}")
    city      = listing.get("city", "")
    body_type = listing.get("body_type", "Other")
    if ML_MODELS_AVAILABLE:
        try:
            X = _build_feature_row(listing)
            log_pred  = _price_model.predict(X)[0]
            raw_price = max(50000, int(np.expm1(log_pred)))
        except Exception as e:
            print(f"[CarIQ] Module 1 inference error: {e}")
            raw_price = _rule_based_price(listing)
    else:
        raw_price = _rule_based_price(listing)
    adj = apply_market_adjustments(raw_price, city, body_type)
    return adj["adjusted_price"], raw_price, adj

def get_fair_price(listing):
    adjusted, _, _ = get_fair_price_full(listing)
    return adjusted

# ── MODULE 2 ──────────────────────────────────────────────────────────────────
def get_maintenance_cost(listing):
    if not MODULE2_AVAILABLE: return None
    try:
        brand     = str(listing.get("brand", "")).split()[0].lower().strip()
        year      = int(listing.get("year", CURRENT_YEAR - 5))
        km        = int(listing.get("km", 50000))
        fuel      = str(listing.get("fuel", "Petrol"))
        trans     = str(listing.get("transmission", "Manual"))
        city      = str(listing.get("city", ""))
        fuel_enc  = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}.get(fuel, 0)
        trans_enc = {"Manual": 0, "Automatic": 1}.get(trans, 0)
        owner_raw = listing.get("owner", "buyer")
        owner_enc = OWNER_MAP.get(str(owner_raw), 2)
        brand_class   = _BRAND_CLASS_M2.get(brand, 1)
        vehicle_age   = max(0, CURRENT_YEAR - year)
        engine_cc     = float(listing.get("engine_cc", 1200.0))
        max_power_bhp = float(listing.get("power_bhp", listing.get("max_power_bhp", 75.0)))
        seats         = float(listing.get("seats", 5))
        mileage_kmpl  = float(listing.get("mileage_num", listing.get("mileage_kmpl", 17.0)))
        if km < 30000:    mileage_band = 0
        elif km < 60000:  mileage_band = 1
        elif km < 100000: mileage_band = 2
        else:             mileage_band = 3
        row = {
            "brand_class": brand_class, "vehicle_age": vehicle_age,
            "engine_cc": engine_cc, "fuel_enc": fuel_enc, "km_driven": km,
            "max_power_bhp": max_power_bhp, "seats": seats,
            "mileage_kmpl": mileage_kmpl, "owner_enc": owner_enc,
            "mileage_band": mileage_band, "trans_enc": trans_enc,
        }
        X = pd.DataFrame([row])[_maint_features]
        base_pred = float(_maint_model.predict(X)[0])
        city_tier = ("Metro" if city in _METRO_CITIES else "Tier-3" if city else "Tier-2")
        city_mult = _CITY_LABOUR_MULT.get(city_tier, 1.00)
        return max(15000, min(int(round(base_pred * city_mult, -2)), 250000))
    except Exception as e:
        print(f"[CarIQ] Module 2 inference error: {e}")
        return None

# ── MODULE 3 ──────────────────────────────────────────────────────────────────
def get_depreciation(listing, adjusted_fair_price):
    if not ML_MODELS_AVAILABLE:
        try:
            load_ml_models()
        except Exception as e:
            print(f"[CarIQ] ML model load failed: {e}")
            return None, None, None
    try:
        brand     = str(listing.get("brand", "")).split()[0]
        fuel      = str(listing.get("fuel", "Petrol"))
        year      = int(listing.get("year", CURRENT_YEAR - 5))
        km        = int(listing.get("km", 50000))
        owner_raw = listing.get("owner", "buyer")
        owner_num = OWNER_MAP.get(str(owner_raw), 2)
        current_age = CURRENT_YEAR - year
        annual_km   = _km_lookup.get((brand, fuel), _global_median_km)
        future_age  = current_age + _forecast_years
        future_km   = km + annual_km * _forecast_years
        future_owner = min(4, owner_num + 1)
        future_mb    = _mileage_band(future_age, future_km)
        X_future = _build_feature_row(listing, vehicle_age=future_age, km_override=future_km,
                                       owner_override=future_owner, mileage_band_override=future_mb)
        log_future = _depr_model.predict(X_future)[0]
        raw_future = max(10000, int(np.expm1(log_future)))
        city      = listing.get("city", "")
        body_type = listing.get("body_type", "Other")
        future_adj = apply_market_adjustments(raw_future, city, body_type)
        future_val = future_adj["adjusted_price"]
        depr_abs = max(0, adjusted_fair_price - future_val)
        depr_pct = round(min(80.0, depr_abs / max(adjusted_fair_price, 1) * 100), 1)
        return future_val, depr_abs, depr_pct
    except Exception as e:
        print(f"[CarIQ] Module 3 inference error: {e}")
        return None, None, None

# ── MODULE 4 ──────────────────────────────────────────────────────────────────
def run_condition_detection(image_bytes):
    if not OPENCV_AVAILABLE: return None, None
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame    = np.array(pil_img)
    bgr      = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized  = cv2.resize(bgr, (640, 480))
    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges    = cv2.Canny(blurred, 50, 150)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels  = int(np.sum(edges > 0))
    density      = edge_pixels / total_pixels
    score = max(0, int(100 - density * 500))
    if density < 0.05:    label = "GOOD"
    elif density <= 0.12: label = "AVERAGE"
    else:                 label = "POOR"
    return score, label

# ── MODULE 5 ──────────────────────────────────────────────────────────────────
def run_fraud_detection(listing, condition_score):
    listed_price = int(listing["price"])
    fair_price   = get_fair_price(listing)
    price_ratio  = listed_price / fair_price if fair_price > 0 else 1.0
    fraud_flag   = (price_ratio < 0.60) and (condition_score is not None) and (condition_score < 40)
    price_conf = max(0, min(100, int((0.60 - price_ratio) / 0.60 * 100))) if price_ratio < 0.60 else 0
    cond_conf  = max(0, min(100, int((40 - condition_score) / 40 * 100))) if (condition_score is not None and condition_score < 40) else 0
    confidence = int((price_conf + cond_conf) / 2) if fraud_flag else max(price_conf, cond_conf)
    return fraud_flag, confidence, fair_price, round(price_ratio, 3)

# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def rows_to_dict(cur, row):
    if row is None: return {}
    return dict(zip([col[0] for col in cur.description], row))

def build_response(listing, model):
    d = dict(listing)
    m = dict(model) if model else {}
    d["models"] = {
        "price_pred":       m.get("price_pred"),
        "fair_price_rule":  m.get("fair_price_rule"),
        "price_ratio":      m.get("price_ratio"),
        "adjusted_price":   m.get("adjusted_price"),
        "city_tier":        m.get("city_tier"),
        "body_type":        m.get("body_type") or d.get("body_type", "Other"),
        "city_multiplier":  m.get("city_multiplier"),
        "body_multiplier":  m.get("body_multiplier"),
        "city_adj_amt":     m.get("city_adj_amt"),
        "body_adj_amt":     m.get("body_adj_amt"),
        "total_adj_pct":    m.get("total_adj_pct"),
        "maintenance_2yr":  m.get("maintenance_2yr"),
        "depreciation_2yr": m.get("depreciation_2yr"),
        "depreciation_pct": m.get("depreciation_pct"),
        "future_value_2yr": m.get("future_value_2yr"),
        "condition_score":  m.get("condition_score"),
        "condition_label":  m.get("condition_label"),
        "fraud_flag":       None if m.get("fraud_flag") is None else bool(m["fraud_flag"]),
        "fraud_confidence": m.get("fraud_confidence"),
    }
    d["opencv_available"]     = OPENCV_AVAILABLE
    d["ml_models_available"]  = ML_MODELS_AVAILABLE
    d["module2_available"]    = MODULE2_AVAILABLE
    d["market_adj_available"] = MARKET_ADJ_AVAILABLE
    return d

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS listings (
            id           SERIAL PRIMARY KEY,
            brand        TEXT NOT NULL,
            model        TEXT NOT NULL,
            year         INTEGER NOT NULL,
            km           INTEGER NOT NULL,
            fuel         TEXT NOT NULL,
            city         TEXT NOT NULL,
            price        INTEGER NOT NULL,
            transmission TEXT NOT NULL,
            seller_name  TEXT NOT NULL DEFAULT 'Anonymous',
            owner        TEXT NOT NULL DEFAULT 'buyer',
            body_type    TEXT NOT NULL DEFAULT 'Other',
            image_path   TEXT,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_outputs (
            id               SERIAL PRIMARY KEY,
            listing_id       INTEGER NOT NULL UNIQUE REFERENCES listings(id) ON DELETE CASCADE,
            price_pred       INTEGER,
            fair_price_rule  INTEGER,
            price_ratio      REAL,
            adjusted_price   INTEGER,
            city_tier        TEXT,
            body_type        TEXT,
            city_multiplier  REAL,
            body_multiplier  REAL,
            city_adj_amt     INTEGER,
            body_adj_amt     INTEGER,
            total_adj_pct    REAL,
            maintenance_2yr  INTEGER,
            depreciation_2yr INTEGER,
            depreciation_pct REAL,
            future_value_2yr INTEGER,
            condition_score  INTEGER,
            condition_label  TEXT,
            fraud_flag       INTEGER,
            fraud_confidence INTEGER
        )
    """)
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM listings")
    count = cur.fetchone()[0]
    if count == 0 and False:
        seeds = [
            ("Maruti Suzuki","Swift VXi",      2019,42000,"Petrol","Mumbai",   520000,"Manual",   "Rahul M.", "buyer","Hatchback"),
            ("Hyundai",      "i20 Sportz",     2020,28000,"Petrol","Pune",     720000,"Manual",   "Priya S.", "buyer","Hatchback"),
            ("Tata",         "Nexon XZ+",      2021,18000,"Petrol","Bengaluru",890000,"Automatic","Ankit R.", "buyer","SUV"),
            ("Honda",        "City ZX",        2018,61000,"Petrol","Delhi",    680000,"Manual",   "Devika L.","buyer","Sedan"),
            ("Mahindra",     "Scorpio S7",     2017,88000,"Diesel","Jaipur",   750000,"Manual",   "Vikram B.","buyer","SUV"),
            ("Toyota",       "Innova Crysta G",2019,52000,"Diesel","Chennai", 1450000,"Manual",   "Rajesh K.","buyer","MUV"),
        ]
        for s in seeds:
            listing_dict = {"brand":s[0],"model":s[1],"year":s[2],"km":s[3],
                            "fuel":s[4],"city":s[5],"price":s[6],"transmission":s[7],"body_type":s[10]}
            cur.execute(
                "INSERT INTO listings (brand,model,year,km,fuel,city,price,transmission,seller_name,owner,body_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10])
            )
            lid = cur.fetchone()[0]
            adjusted_price, raw_price, adj = get_fair_price_full(listing_dict)
            price_ratio     = round(s[6] / adjusted_price, 3)
            maintenance_2yr = get_maintenance_cost(listing_dict)
            future_val, depr_abs, depr_pct = get_depreciation(listing_dict, adjusted_price)
            cur.execute(
                """INSERT INTO model_outputs
                   (listing_id,price_pred,fair_price_rule,price_ratio,
                    adjusted_price,city_tier,body_type,
                    city_multiplier,body_multiplier,city_adj_amt,body_adj_amt,total_adj_pct,
                    maintenance_2yr,depreciation_2yr,depreciation_pct,future_value_2yr)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (lid,raw_price,raw_price,price_ratio,
                 adj["adjusted_price"],adj["city_tier"],listing_dict["body_type"],
                 adj["city_multiplier"],adj["body_multiplier"],
                 adj["city_adjustment_amt"],adj["body_adjustment_amt"],adj["total_adjustment_pct"],
                 maintenance_2yr,depr_abs,depr_pct,future_val)
            )
    conn.commit()
    cur.close()
    conn.close()

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/listings", methods=["GET"])
def get_listings():
    fuel  = request.args.get("fuel", "all")
    query = request.args.get("q", "").strip()
    conn  = get_db()
    cur   = conn.cursor()
    sql    = "SELECT l.*, m.* FROM listings l LEFT JOIN model_outputs m ON m.listing_id = l.id WHERE 1=1"
    params = []
    if fuel and fuel != "all":
        sql += " AND l.fuel = %s"
        params.append(fuel)
    if query:
        sql += " AND (l.brand ILIKE %s OR l.model ILIKE %s)"
        params += [f"%{query}%", f"%{query}%"]
    sql += " ORDER BY l.created_at DESC"
    cur.execute(sql, params)
    cols = [col[0] for col in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()
    results = []
    for r in rows:
        d = dict(zip(cols, r))
        models = {
            "price_pred":       d.get("price_pred"),
            "fair_price_rule":  d.get("fair_price_rule"),
            "price_ratio":      d.get("price_ratio"),
            "adjusted_price":   d.get("adjusted_price"),
            "city_tier":        d.get("city_tier"),
            "body_type":        d.get("body_type") or "Other",
            "city_multiplier":  d.get("city_multiplier"),
            "body_multiplier":  d.get("body_multiplier"),
            "city_adj_amt":     d.get("city_adj_amt"),
            "body_adj_amt":     d.get("body_adj_amt"),
            "total_adj_pct":    d.get("total_adj_pct"),
            "maintenance_2yr":  d.get("maintenance_2yr"),
            "depreciation_2yr": d.get("depreciation_2yr"),
            "depreciation_pct": d.get("depreciation_pct"),
            "future_value_2yr": d.get("future_value_2yr"),
            "condition_score":  d.get("condition_score"),
            "condition_label":  d.get("condition_label"),
            "fraud_flag":       None if d.get("fraud_flag") is None else bool(d["fraud_flag"]),
            "fraud_confidence": d.get("fraud_confidence"),
        }
        listing_keys = ["id","brand","model","year","km","fuel","city","price",
                        "transmission","seller_name","owner","body_type","created_at"]
        listing = {k: str(d.get(k)) if isinstance(d.get(k), datetime.datetime) else d.get(k) for k in listing_keys}
        listing["models"] = models
        listing["opencv_available"]     = OPENCV_AVAILABLE
        listing["ml_models_available"]  = ML_MODELS_AVAILABLE
        listing["module2_available"]    = MODULE2_AVAILABLE
        listing["market_adj_available"] = MARKET_ADJ_AVAILABLE
        results.append(listing)
    return jsonify(results)

@app.route("/api/listings/<int:lid>", methods=["GET"])
def get_listing(lid):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM listings WHERE id=%s", (lid,))
    l_row  = cur.fetchone()
    l_desc = cur.description
    if not l_row:
        cur.close(); conn.close()
        return jsonify({"error": "Not found"}), 404
    cur.execute("SELECT * FROM model_outputs WHERE listing_id=%s", (lid,))
    m_row  = cur.fetchone()
    m_desc = cur.description
    cur.close(); conn.close()
    l = dict(zip([col[0] for col in l_desc], l_row))
    m = dict(zip([col[0] for col in m_desc], m_row)) if m_row else {}
    return jsonify(build_response(l, m))

@app.route("/api/listings", methods=["POST"])
def create_listing():
    if request.content_type and "multipart" in request.content_type:
        data       = request.form.to_dict()
        image_file = request.files.get("image")
    else:
        data       = request.get_json() or {}
        image_file = None
    required = ["brand","model","year","km","fuel","city","price","transmission"]
    if not all(data.get(k) for k in required):
        return jsonify({"error": "Missing required fields"}), 400
    body_type = data.get("body_type", "Other")
    conn = get_db()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO listings (brand,model,year,km,fuel,city,price,transmission,seller_name,owner,body_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (data["brand"],data["model"],int(data["year"]),int(data["km"]),
         data["fuel"],data["city"],int(data["price"]),data["transmission"],
         data.get("seller_name","You"),data.get("owner","seller"),body_type)
    )
    lid = cur.fetchone()[0]
    listing_dict = dict(data)
    listing_dict["body_type"] = body_type
    adjusted_price, raw_price, adj = get_fair_price_full(listing_dict)
    price_ratio     = round(int(data["price"]) / adjusted_price, 3)
    maintenance_2yr = get_maintenance_cost(listing_dict)
    future_val, depr_abs, depr_pct = get_depreciation(listing_dict, adjusted_price)
    condition_score = None
    condition_label = None
    if image_file:
        image_bytes = image_file.read()

        if OPENCV_AVAILABLE:
            condition_score, condition_label = run_condition_detection(image_bytes)

        fname = f"listing_{lid}_{image_file.filename}"

        image_url = None

        if supabase:
            try:
                supabase.storage.from_("car-images").upload(
                    fname,
                    image_bytes,
                    {"content-type": image_file.mimetype}
                )
                image_url = supabase.storage.from_("car-images").get_public_url(fname)
            except Exception as e:
                print(f"[CarIQ] Supabase upload failed: {e}")

        if image_url:
            cur.execute(
                "UPDATE listings SET image_path=%s WHERE id=%s",
                (image_url, lid)
            )
        else:
            image_path = os.path.join(UPLOAD_DIR, fname)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            cur.execute(
                "UPDATE listings SET image_path=%s WHERE id=%s",
                (fname, lid)
            )
    fraud_flag = None
    fraud_confidence = None
    if condition_score is not None:
        fraud_flag, fraud_confidence, _, _ = run_fraud_detection(listing_dict, condition_score)
        fraud_flag = int(fraud_flag)
    cur.execute(
        """INSERT INTO model_outputs
           (listing_id,price_pred,fair_price_rule,price_ratio,
            adjusted_price,city_tier,body_type,
            city_multiplier,body_multiplier,city_adj_amt,body_adj_amt,total_adj_pct,
            maintenance_2yr,depreciation_2yr,depreciation_pct,future_value_2yr,
            condition_score,condition_label,fraud_flag,fraud_confidence)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (lid,raw_price,raw_price,price_ratio,
         adj["adjusted_price"],adj["city_tier"],body_type,
         adj["city_multiplier"],adj["body_multiplier"],
         adj["city_adjustment_amt"],adj["body_adjustment_amt"],adj["total_adjustment_pct"],
         maintenance_2yr,depr_abs,depr_pct,future_val,
         condition_score,condition_label,fraud_flag,fraud_confidence)
    )
    conn.commit()
    cur.execute("SELECT * FROM listings WHERE id=%s", (lid,))
    l_row = cur.fetchone(); l_desc = cur.description
    cur.execute("SELECT * FROM model_outputs WHERE listing_id=%s", (lid,))
    m_row = cur.fetchone(); m_desc = cur.description
    cur.close(); conn.close()
    l = dict(zip([col[0] for col in l_desc], l_row))
    m = dict(zip([col[0] for col in m_desc], m_row)) if m_row else {}
    return jsonify(build_response(l, m)), 201

@app.route("/api/condition", methods=["POST"])
def check_condition():
    if not OPENCV_AVAILABLE:
        return jsonify({"error": "OpenCV not installed"}), 503
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image provided"}), 400
    score, label = run_condition_detection(image_file.read())
    return jsonify({"condition_score": score, "condition_label": label})

@app.route("/api/listings/<int:lid>", methods=["DELETE"])
def delete_listing(lid):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("DELETE FROM listings WHERE id=%s", (lid,))
    conn.commit()
    cur.close(); conn.close()
    return jsonify({"deleted": lid})

@app.route("/api/listings/<int:lid>/models", methods=["PATCH"])
def update_models(lid):
    data    = request.get_json()
    allowed = ["price_pred","adjusted_price","city_tier","body_type",
               "city_multiplier","body_multiplier","city_adj_amt","body_adj_amt","total_adj_pct",
               "maintenance_2yr","depreciation_2yr","depreciation_pct",
               "future_value_2yr","condition_score","condition_label",
               "fraud_flag","fraud_confidence"]
    fields  = {k: v for k, v in data.items() if k in allowed}
    if not fields:
        return jsonify({"error": "No valid fields"}), 400
    set_clause = ", ".join(f"{k}=%s" for k in fields)
    conn = get_db()
    cur  = conn.cursor()
    cur.execute(f"UPDATE model_outputs SET {set_clause} WHERE listing_id=%s", (*fields.values(), lid))
    conn.commit()
    cur.execute("SELECT * FROM model_outputs WHERE listing_id=%s", (lid,))
    desc = cur.description
    row  = cur.fetchone()
    cur.close(); conn.close()
    m = dict(zip([col[0] for col in desc], row)) if row else {}
    return jsonify(m)

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "module_1_price":        "LIVE — Random Forest pkl" if ML_MODELS_AVAILABLE else "FALLBACK — rule-based",
        "module_2_maintenance":  "LIVE — Hist Gradient Boosting pkl" if MODULE2_AVAILABLE else "PENDING",
        "module_3_depreciation": "LIVE — Real-wear simulation RF" if ML_MODELS_AVAILABLE else "PENDING",
        "module_4_condition":    "LIVE" if OPENCV_AVAILABLE else "DISABLED — install opencv-python-headless",
        "module_5_fraud":        ("LIVE" if OPENCV_AVAILABLE else "DISABLED — requires Module 4"),
        "market_adjustments":    "LIVE — pkl loaded" if MARKET_ADJ_AVAILABLE else "ACTIVE — hardcoded fallback",
        "opencv_available":      OPENCV_AVAILABLE,
        "ml_models_available":   ML_MODELS_AVAILABLE,
        "module2_available":     MODULE2_AVAILABLE,
        "market_adj_available":  MARKET_ADJ_AVAILABLE,
    })

if __name__ == "__main__":
    init_db()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))