"""Model definitions for weather prediction."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import joblib

from config import PRECIP_THRESHOLD_MM


# ── Shared LightGBM base parameters ──────────────────
BASE_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


class WeatherPredictor:
    """Multi-output MOS corrector for temperature, humidity, wind."""

    def __init__(self):
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.quantile_models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
        self.targets = {
            "temperature": {"objective": "regression", "metric": "mae"},
            "humidity": {"objective": "regression", "metric": "mae"},
            "wind_speed": {"objective": "regression", "metric": "mae"},
            "wind_direction": {"objective": "regression", "metric": "mae"},
        }
        self.target_to_obs_col = {
            "temperature": "obs_temp",
            "humidity": "obs_humidity",
            "wind_speed": "obs_wind_speed",
            "wind_direction": "obs_wind_dir",
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> dict[str, float]:
        scores = {}
        for target, info in self.targets.items():
            obs_col = self.target_to_obs_col[target]
            if obs_col not in y_train.columns:
                continue

            yt = y_train[obs_col].dropna()
            common = yt.index.intersection(X_train.index)
            if len(common) < 100:
                print(f"  Skipping {target}: only {len(common)} samples")
                continue

            params = {**BASE_PARAMS, "objective": info["objective"], "metric": info["metric"]}
            model = lgb.LGBMRegressor(**params)

            yv = y_val[obs_col].dropna()
            val_common = yv.index.intersection(X_val.index)

            model.fit(
                X_train.loc[common],
                yt.loc[common],
                eval_set=[(X_val.loc[val_common], yv.loc[val_common])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
            )
            self.models[target] = model

            preds = model.predict(X_val.loc[val_common])
            mae = np.mean(np.abs(preds - yv.loc[val_common].values))
            scores[target] = mae
            print(f"  {target}: val MAE = {mae:.3f}")

            # Quantile models for prediction intervals
            self.quantile_models[target] = {}
            for alpha in [0.1, 0.9]:
                qparams = {**BASE_PARAMS, "objective": "quantile", "alpha": alpha,
                           "n_estimators": 500, "num_leaves": 31}
                qmodel = lgb.LGBMRegressor(**qparams)
                qmodel.fit(X_train.loc[common], yt.loc[common])
                self.quantile_models[target][alpha] = qmodel

        return scores

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        results = {}
        for target, model in self.models.items():
            results[target] = model.predict(X)
            if target in self.quantile_models:
                results[f"{target}_q10"] = self.quantile_models[target][0.1].predict(X)
                results[f"{target}_q90"] = self.quantile_models[target][0.9].predict(X)

        # Physical constraints
        if "humidity" in results:
            results["humidity"] = np.clip(results["humidity"], 0, 100)
        if "wind_speed" in results:
            results["wind_speed"] = np.maximum(results["wind_speed"], 0)
        return results

    def save(self, path) -> None:
        joblib.dump({"models": self.models, "quantile_models": self.quantile_models}, path)

    def load(self, path) -> None:
        data = joblib.load(path)
        self.models = data["models"]
        self.quantile_models = data.get("quantile_models", {})


class PrecipitationPredictor:
    """Two-stage precipitation model: classify then regress."""

    def __init__(self):
        self.classifier = lgb.LGBMClassifier(
            **{**BASE_PARAMS, "objective": "binary", "is_unbalance": True}
        )
        self.regressor = lgb.LGBMRegressor(
            **{**BASE_PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.5}
        )
        self.quantile_models: dict[float, lgb.LGBMRegressor] = {}
        self.calibrator: IsotonicRegression | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train_precip: pd.Series,
        X_val: pd.DataFrame,
        y_val_precip: pd.Series,
    ) -> dict[str, float]:
        y_binary_train = (y_train_precip >= PRECIP_THRESHOLD_MM).astype(int)
        y_binary_val = (y_val_precip >= PRECIP_THRESHOLD_MM).astype(int)

        # Stage 1: Will it rain?
        self.classifier.fit(
            X_train, y_binary_train,
            eval_set=[(X_val, y_binary_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )

        # Calibrate probabilities with isotonic regression
        raw_probs = self.classifier.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y_binary_val.values)

        # Stage 2: How much? (trained only on rainy hours)
        rain_mask_train = y_train_precip >= PRECIP_THRESHOLD_MM
        rain_mask_val = y_val_precip >= PRECIP_THRESHOLD_MM

        scores = {}
        if rain_mask_train.sum() > 100:
            self.regressor.fit(
                X_train[rain_mask_train],
                y_train_precip[rain_mask_train],
                eval_set=[(X_val[rain_mask_val], y_val_precip[rain_mask_val])]
                if rain_mask_val.sum() > 10
                else None,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
            )

            # Quantile models for uncertainty
            for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
                qparams = {**BASE_PARAMS, "objective": "quantile", "alpha": alpha,
                           "n_estimators": 500, "num_leaves": 31}
                qmodel = lgb.LGBMRegressor(**qparams)
                qmodel.fit(X_train[rain_mask_train], y_train_precip[rain_mask_train])
                self.quantile_models[alpha] = qmodel

        # Scores
        from sklearn.metrics import brier_score_loss, roc_auc_score

        cal_probs = self.calibrator.predict(raw_probs)
        scores["brier"] = brier_score_loss(y_binary_val, cal_probs)
        scores["auc"] = roc_auc_score(y_binary_val, cal_probs)
        print(f"  Precip classification: Brier={scores['brier']:.4f}, AUC={scores['auc']:.4f}")
        return scores

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        raw_probs = self.classifier.predict_proba(X)[:, 1]
        prob = self.calibrator.predict(raw_probs) if self.calibrator else raw_probs

        raw_amount = self.regressor.predict(X)
        raw_amount = np.maximum(raw_amount, 0)

        results = {
            "precip_probability": prob,
            "precip_expected_mm": prob * raw_amount,
            "precip_if_rain_mm": raw_amount,
        }

        for alpha, qmodel in self.quantile_models.items():
            q = np.maximum(qmodel.predict(X), 0)
            results[f"precip_q{int(alpha*100):02d}_mm"] = q * prob

        return results

    def save(self, path) -> None:
        joblib.dump(
            {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "calibrator": self.calibrator,
                "quantile_models": self.quantile_models,
            },
            path,
        )

    def load(self, path) -> None:
        data = joblib.load(path)
        self.classifier = data["classifier"]
        self.regressor = data["regressor"]
        self.calibrator = data["calibrator"]
        self.quantile_models = data.get("quantile_models", {})