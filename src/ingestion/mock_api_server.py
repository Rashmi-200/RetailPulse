from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="RetailPulse Simulated Payment Gateway API", version="1.0.0")

PAYMENTS_CSV = Path("data/raw/olist_order_payments_dataset.csv")
_cached_df: Optional[pd.DataFrame] = None


def get_payments_df() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        if not PAYMENTS_CSV.exists():
            raise FileNotFoundError(f"Payments source file not found: {PAYMENTS_CSV}")
        _cached_df = pd.read_csv(PAYMENTS_CSV)
    return _cached_df


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "payment_gateway_api"}


@app.get("/api/v1/payments")
def fetch_payments(
    offset: int = Query(0, ge=0),
    limit: int = Query(50000, ge=1, le=200000),
    simulate_timeout: bool = Query(False),
):
    if simulate_timeout:
        raise HTTPException(status_code=504, detail="Simulated upstream payment gateway timeout")

    df = get_payments_df()
    total_records = len(df)
    slice_df = df.iloc[offset : offset + limit].copy()
    # Replace NaN with None for valid JSON serialization
    records = slice_df.where(pd.notnull(slice_df), None).to_dict(orient="records")
    return {
        "total_records": total_records,
        "offset": offset,
        "returned_records": len(records),
        "data": records,
    }
