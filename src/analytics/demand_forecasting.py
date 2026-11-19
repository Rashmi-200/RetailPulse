from pathlib import Path
from typing import Any, Dict, List
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
    """Forecasts daily retail demand from Gold Star Schema with chronological holdout & rolling-origin backtesting."""

    def __init__(self, engine: Engine, model_dir: str = "models", min_active_daily_orders: int = 20):
        self.engine = engine
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "demand_forecaster.joblib"
        self.min_active_daily_orders = min_active_daily_orders

    def load_gold_daily_series(self) -> pd.DataFrame:
        """Dynamically derive contiguous active marketplace window from validated Gold orders (no hardcoded date cutoffs)."""
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

        df["order_date"] = pd.to_datetime(df["order_date"])
        # Derive complete operational window dynamically (exclude sparse ramp-up and incomplete dataset-end cutoff tail)
        active_median = df.loc[df["daily_orders"] >= self.min_active_daily_orders, "daily_orders"].median()
        complete_threshold = max(float(self.min_active_daily_orders), 0.5 * float(active_median))
        rolling_7d_med = df["daily_orders"].rolling(window=7, center=True, min_periods=1).median()
        complete_dates = df.loc[rolling_7d_med >= complete_threshold, "order_date"]
        if complete_dates.empty:
            raise ValueError("No complete operational marketplace window found.")

        start_date = complete_dates.min()
        end_date = complete_dates.max()
        df = df[(df["order_date"] >= start_date) & (df["order_date"] <= end_date)].reset_index(drop=True)
        logger.info(
            f"[FORECAST SERIES] Derived active window [{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}] "
            f"({len(df)} contiguous days, threshold >= {self.min_active_daily_orders} orders/day)"
        )
        return df

    @staticmethod
    def engineer_features(df: pd.DataFrame, target_col: str = "daily_orders") -> pd.DataFrame:
        """Strictly backward-looking lag & rolling features (all rolling stats shifted by 1 to prevent target leakage)."""
        feat = df.copy()
        feat["lag_1"] = feat[target_col].shift(1)
        feat["lag_7"] = feat[target_col].shift(7)
        feat["lag_14"] = feat[target_col].shift(14)
        # Explicit shift(1) guarantees window at index t uses only [t-7 ... t-1]
        feat["rolling_mean_7"] = feat[target_col].shift(1).rolling(window=7).mean()
        feat["rolling_std_7"] = feat[target_col].shift(1).rolling(window=7).std().fillna(0.0)
        return feat.dropna().reset_index(drop=True)

    @staticmethod
    def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true > 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    @staticmethod
    def _build_candidates() -> Dict[str, Any]:
        return {
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
            "LightGBM": LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=20, verbose=-1, random_state=42
            ),
        }

    def rolling_origin_backtest(self, df: pd.DataFrame, target_col: str = "daily_orders", n_folds: int = 4, horizon: int = 7) -> List[Dict[str, Any]]:
        """Expanding-window rolling-origin backtesting across multiple 7-day horizons."""
        total_len = len(df)
        eval_span = n_folds * horizon
        initial_train_end = total_len - eval_span

        preds_by_model: Dict[str, List[float]] = {
            "Seasonal_Naive_Baseline (lag_7)": [],
            "RandomForest": [],
            "XGBoost": [],
            "LightGBM": [],
        }
        actuals: List[float] = []

        for fold in range(n_folds):
            train_end = initial_train_end + fold * horizon
            test_end = train_end + horizon
            train_slice = df.iloc[:train_end]
            test_slice = df.iloc[train_end:test_end]

            assert train_slice["order_date"].max() < test_slice["order_date"].min(), "Leakage detected in rolling fold!"

            X_tr, y_tr = train_slice[FEATURE_COLS], train_slice[target_col].values
            X_te, y_te = test_slice[FEATURE_COLS], test_slice[target_col].values

            actuals.extend(y_te.tolist())
            preds_by_model["Seasonal_Naive_Baseline (lag_7)"].extend(test_slice["lag_7"].values.tolist())

            for name, reg in self._build_candidates().items():
                reg.fit(X_tr, y_tr)
                p = np.maximum(0.0, reg.predict(X_te))
                preds_by_model[name].extend(p.tolist())

        y_true_arr = np.array(actuals)
        summary = []
        for model_name, pred_list in preds_by_model.items():
            y_pred_arr = np.array(pred_list)
            summary.append(
                {
                    "model": model_name,
                    "folds": n_folds,
                    "horizon_days": horizon,
                    "MAE": round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 2),
                    "RMSE": round(float(root_mean_squared_error(y_true_arr, y_pred_arr)), 2),
                    "MAPE_pct": round(self.safe_mape(y_true_arr, y_pred_arr), 2),
                }
            )
        return summary

    def train_and_benchmark(self, target_col: str = "daily_orders") -> Dict[str, Any]:
        raw_series = self.load_gold_daily_series()
        df = self.engineer_features(raw_series, target_col=target_col)

        # 1. Chronological 85/15 Holdout Split
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        assert train_df["order_date"].max() < test_df["order_date"].min(), "Chronological split violation!"

        X_train, y_train = train_df[FEATURE_COLS], train_df[target_col].values
        X_test, y_test = test_df[FEATURE_COLS], test_df[target_col].values

        naive_preds = test_df["lag_7"].values
        holdout_rows = [
            {
                "model": "Seasonal_Naive_Baseline (lag_7)",
                "MAE": round(float(mean_absolute_error(y_test, naive_preds)), 2),
                "RMSE": round(float(root_mean_squared_error(y_test, naive_preds)), 2),
                "MAPE_pct": round(self.safe_mape(y_test, naive_preds), 2),
            }
        ]

        best_model = None
        best_mae = float("inf")
        best_name = ""

        for name, reg in self._build_candidates().items():
            reg.fit(X_train, y_train)
            preds = np.maximum(0.0, reg.predict(X_test))
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            mape = self.safe_mape(y_test, preds)
            holdout_rows.append(
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

        # 2. Rolling-Origin Backtest (4 folds x 7-day horizon)
        rolling_rows = self.rolling_origin_backtest(df, target_col=target_col, n_folds=4, horizon=7)

        # 3. Retrain best model on full active series for next-7-day forecast & validate against historical distribution
        best_model.fit(df[FEATURE_COLS], df[target_col].values)
        future_forecasts = self._forecast_next_7_days(best_model, raw_series, target_col)

        recent_90d = raw_series[target_col].tail(90).values
        hist_stats = {
            "recent_90d_mean": round(float(np.mean(recent_90d)), 1),
            "recent_90d_median": round(float(np.median(recent_90d)), 1),
            "recent_90d_p05": round(float(np.percentile(recent_90d, 5)), 1),
            "recent_90d_p10": round(float(np.percentile(recent_90d, 10)), 1),
            "recent_90d_p90": round(float(np.percentile(recent_90d, 90)), 1),
            "recent_90d_p95": round(float(np.percentile(recent_90d, 95)), 1),
        }
        fc_vals = [f[f"predicted_{target_col}"] for f in future_forecasts]
        fc_mean = round(float(np.mean(fc_vals)), 1)
        active_min = round(float(raw_series[target_col].min()), 1)
        active_max = round(float(raw_series[target_col].max()), 1)
        within_active_bounds = all(active_min <= v <= active_max for v in fc_vals)
        mean_within_iqr = hist_stats["recent_90d_p10"] <= fc_mean <= hist_stats["recent_90d_p90"]
        distribution_check = {
            **hist_stats,
            "active_window_min": active_min,
            "active_window_max": active_max,
            "forecast_min": min(fc_vals),
            "forecast_mean": fc_mean,
            "forecast_max": max(fc_vals),
            "within_active_min_max": within_active_bounds,
            "mean_within_p10_p90": mean_within_iqr,
        }

        joblib.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "target_col": target_col,
                "features": FEATURE_COLS,
                "active_window": {
                    "start": raw_series["order_date"].min().strftime("%Y-%m-%d"),
                    "end": raw_series["order_date"].max().strftime("%Y-%m-%d"),
                    "days": len(raw_series),
                },
                "benchmark": holdout_rows,
                "rolling_origin_backtest": rolling_rows,
                "distribution_validation": distribution_check,
                "next_7_days": future_forecasts,
            },
            self.model_path,
        )

        logger.info(
            f"\n=== Gold Demand Forecasting Chronological Holdout ({test_df['order_date'].min().strftime('%Y-%m-%d')} -> {test_df['order_date'].max().strftime('%Y-%m-%d')}) ===\n"
            + pd.DataFrame(holdout_rows).to_string(index=False)
        )
        logger.info(
            "\n=== Rolling-Origin 7-Day Horizon Backtest (4 Folds x 7 Days) ===\n"
            + pd.DataFrame(rolling_rows).to_string(index=False)
        )
        logger.info(f"Forecast Distribution Check vs. Recent 90d Actuals: {distribution_check}")

        return {
            "best_model": best_name,
            "test_window_days": len(test_df),
            "benchmark": holdout_rows,
            "rolling_origin_backtest": rolling_rows,
            "distribution_validation": distribution_check,
            "next_7_days_forecast": future_forecasts,
        }

    def _forecast_next_7_days(self, model, raw_series: pd.DataFrame, target_col: str) -> list[Dict[str, Any]]:
        history = raw_series[["order_date", target_col]].copy()
        forecasts = []
        for _ in range(7):
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
