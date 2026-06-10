# CarIQ — Flask + SQLite Prototype

## Setup

```bash
pip install flask
python app.py
# → http://localhost:5000
```

The SQLite DB (`cariq.db`) is created automatically on first run with 6 seeded listings.

---

## REST API

### Listings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/listings` | All listings. Query params: `?fuel=Diesel&q=swift` |
| GET | `/api/listings/<id>` | Single listing with model outputs |
| POST | `/api/listings` | Create listing |
| DELETE | `/api/listings/<id>` | Delete listing |

**POST body:**
```json
{
  "brand": "Hyundai",
  "model": "i20",
  "year": 2020,
  "km": 35000,
  "fuel": "Petrol",
  "city": "Mumbai",
  "price": 680000,
  "transmission": "Manual",
  "seller_name": "Rahul",
  "owner": "seller"
}
```

### Model Outputs

| Method | Endpoint | Description |
|--------|----------|-------------|
| PATCH | `/api/listings/<id>/models` | Update any model output field |

**PATCH body (any subset of fields):**
```json
{
  "price_pred": 510000,
  "maintenance_2yr": 74000,
  "depreciation_2yr": 390000,
  "condition_score": 78,
  "condition_label": "GOOD",
  "fraud_flag": false,
  "fraud_confidence": 94
}
```

Once you train the ML models, just call this endpoint per listing to populate the cards.

---

## DB Schema

```sql
listings (id, brand, model, year, km, fuel, city, price, transmission, seller_name, owner, created_at)
model_outputs (id, listing_id, price_pred, maintenance_2yr, depreciation_2yr, condition_score, condition_label, fraud_flag, fraud_confidence)
```
