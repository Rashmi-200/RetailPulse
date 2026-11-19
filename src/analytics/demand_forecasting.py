from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sqlalchemy import text
from sqlalchemy.engine import Engine
from xgboost import XGBRegressor

from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "is_weekend",
    "month",
]


class RetailDemandForecaster:
    """Forecasts daily retail order volume & sales revenue from Gold Star Schema using chronological splits."""

    def __init__(self, engine: Engine, model_dir: str = "models"):
        self.engine = engine
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "demand_forecaster.joblib"

    def load_gold_daily_series(self) -> pd.DataFrame:
        query = text(
            """
            SELECT
                d.full_date::date AS order_date,
                d.day_of_week,
                CASE WHEN d.is_weekend THEN 1 ELSE 0 END AS is_weekend,
                d.month,
                COUNT(DISTINCT f.order_id) AS daily_orders,
                COALESCE(SUM(f.gross_order_value), 0.0) AS daily_revenue
            FROM gold.dimdate d
            LEFT JOIN gold.factorders f
                ON d.date_key = f.order_date_key
               AND f.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY d.full_date, d.day_of_week, d.is_weekend, d.month
            ORDER BY d.full_date ASC;
            """
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Restrict to operational marketplace window (2017-01-15 to 2018-08-20) where daily volume is active
        df["order_date"] = pd.to_datetime(df["order_date"])
        df = df[(df["order_date"] >= "2017-01-15") & (df["order_date"] <= "2018-08-20")].reset_index(drop=True)
        return df

    @staticmethod
    def engineer_features(df: pd.DataFrame, target_col: str = "daily_orders") -> pd.DataFrame:
        feat = df.copy()
        feat["lag_1"] = feat[target_col].shift(1)
        feat["lag_7"] = feat[target_col].shift(7)
        feat["lag_14"] = feat[target_col].shift(14)
        feat["rolling_mean_7"] = feat[target_col].shift(1).rolling(window=7).mean()
        feat["rolling_std_7"] = feat[target_col].shift(1).rolling(window=7).std().fillna(0.0)
        return feat.dropna().reset_index(drop=True)

    @staticmethod
    def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true > 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    def train_and_benchmark(self, target_col: str = "daily_orders") -> Dict[str, Any]:
        raw_series = self.load_gold_daily_series()
        df = self.engineer_features(raw_series, target_col=target_col)

        # Strict chronological split: first 85% train, final 15% (~85 days) test
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train, y_train = train_df[FEATURE_COLS], train_df[target_col].values
        X_test, y_test = test_df[FEATURE_COLS], test_df[target_col].values

        # 1. Seasonal Naive Baseline (same day last week = lag_7)
        naive_preds = test_df["lag_7"].values
        benchmark_rows = [
            {
                "model": "Seasonal_Naive_Baseline (lag_7)",
                "MAE": round(float(mean_absolute_error(y_test, naive_preds)), 2),
                "RMSE": round(float(root_mean_squared_error(y_test, naive_preds)), 2),
                "MAPE_pct": round(self.safe_mape(y_test, naive_preds), 2),
            }
        ]

        candidates = {
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
            "XGBoost": XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=20, verbose=-1, random_state=42
            ),
        }

        best_model = None
        best_mae = float("inf")
        best_name = ""

        for name, reg in candidates.items():
            reg.fit(X_train, y_train)
            preds = np.maximum(0, reg.predict(X_test))
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            mape = self.safe_mape(y_test, preds)
            benchmark_rows.append(
                {
                    "model": name,
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE_pct": round(mape, 2),
                }
            )
            if mae < best_mae:
                best_mae = mae
                best_model = reg
                best_name = name

        # Forecast next 7 days iteratively from the end of Gold history
        future_forecasts = self._forecast_next_7_days(best_model, raw_series, target_col)

        joblib.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "target_col": target_col,
                "features": FEATURE_COLS,
                "benchmark": benchmark_rows,
                "next_7_days": future_forecasts,
            },
            self.model_path,
        )

        bench_df = pd.DataFrame(benchmark_rows)
        logger.info(f"\n=== Gold Demand Forecasting Benchmark ({target_col}) ===\n" + bench_df.to_string(index=False))
        return {
            "best_model": best_name,
            "test_window_days": len(test_df),
            "benchmark": benchmark_rows,
            "next_7_days_forecast": future_forecasts,
        }

    def _forecast_next_7_days(self, model, raw_series: pd.DataFrame, target_col: str) -> list[Dict[str, Any]]:
        history = raw_series[["order_date", target_col]].copy()
        forecasts = []
        for step in range(1, 8):
            next_date = history["order_date"].iloc[-1] + pd.Timedelta(days=1)
            vals = history[target_col].values
            row_feat = pd.DataFrame(
                [
                    {
                        "lag_1": vals[-1],
                        "lag_7": vals[-7],
                        "lag_14": vals[-14],
                        "rolling_mean_7": float(np.mean(vals[-7:])),
                        "rolling_std_7": float(np.std(vals[-7:])),
                        "day_of_week": next_date.dayofweek,
                        "is_weekend": int(next_date.dayofweek in (5, 6)),
                        "month": next_date.month,
                    }
                ]
            )
            pred_val = max(0.0, float(model.predict(row_feat)[0]))
            forecasts.append({"date": next_date.strftime("%Y-%m-%d"), f"predicted_{target_col}": round(pred_val, 1)})
            history = pd.concat(
                [history, pd.DataFrame([{"order_date": next_date, target_col: pred_val}])], ignore_index=True
            )
        return forecasts
