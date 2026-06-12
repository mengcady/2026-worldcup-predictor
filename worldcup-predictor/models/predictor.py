"""
World Cup Score Predictor

Uses XGBoost + Random Forest ensemble to predict match scores
based on ELO ratings, team strength, and form.
Poisson distribution is used for final score probability distribution.
"""

import math
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class ScorePredictor:
    """
    Ensemble score predictor combining XGBoost and Random Forest.
    Predicts expected goals for each team and computes full score distribution.
    """

    def __init__(self, use_xgboost: bool = True):
        self.use_xgboost = use_xgboost
        self.xgb_model_home = None
        self.xgb_model_away = None
        self.rf_model_home = None
        self.rf_model_away = None
        self.is_trained = False
        self.feature_columns = [
            "home_elo", "away_elo", "home_strength", "away_strength",
            "home_rank", "away_rank", "elo_diff", "strength_diff", "rank_diff",
        ]

    def _get_xgboost(self):
        """Lazy import XGBoost to avoid import errors if not installed."""
        try:
            import xgboost as xgb
            return xgb
        except ImportError:
            return None

    def _prepare_data(self, features: List[dict], labels: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert feature dicts and labels to numpy arrays."""
        X = np.array([[f[col] for col in self.feature_columns] for f in features])
        y_home = np.array([l["home_goals"] for l in labels])
        y_away = np.array([l["away_goals"] for l in labels])
        return X, y_home, y_away

    def train(self, features: List[dict], labels: List[dict], verbose: bool = True) -> dict:
        """
        Train both XGBoost and Random Forest models on historical data.
        Returns training metrics.
        """
        X, y_home, y_away = self._prepare_data(features, labels)

        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X, y_home, y_away, test_size=0.2, random_state=42
        )

        results = {}

        # --- Random Forest ---
        if verbose:
            print("Training Random Forest models...")

        self.rf_model_home = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        )
        self.rf_model_home.fit(X_train, y_home_train)
        rf_home_pred = self.rf_model_home.predict(X_test)

        self.rf_model_away = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        )
        self.rf_model_away.fit(X_train, y_away_train)
        rf_away_pred = self.rf_model_away.predict(X_test)

        results["rf_home_mae"] = float(mean_absolute_error(y_home_test, rf_home_pred))
        results["rf_away_mae"] = float(mean_absolute_error(y_away_test, rf_away_pred))

        if verbose:
            print(f"  RF Home MAE: {results['rf_home_mae']:.3f}")
            print(f"  RF Away MAE: {results['rf_away_mae']:.3f}")

        # --- XGBoost ---
        xgb = self._get_xgboost()
        if xgb and self.use_xgboost:
            if verbose:
                print("Training XGBoost models...")

            self.xgb_model_home = xgb.XGBRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0
            )
            self.xgb_model_home.fit(X_train, y_home_train)
            xgb_home_pred = self.xgb_model_home.predict(X_test)

            self.xgb_model_away = xgb.XGBRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0
            )
            self.xgb_model_away.fit(X_train, y_away_train)
            xgb_away_pred = self.xgb_model_away.predict(X_test)

            results["xgb_home_mae"] = float(mean_absolute_error(y_home_test, xgb_home_pred))
            results["xgb_away_mae"] = float(mean_absolute_error(y_away_test, xgb_away_pred))

            if verbose:
                print(f"  XGB Home MAE: {results['xgb_home_mae']:.3f}")
                print(f"  XGB Away MAE: {results['xgb_away_mae']:.3f}")
        else:
            if verbose:
                print("XGBoost not available, using Random Forest only")

        self.is_trained = True
        return results

    def predict(self, home_team: dict, away_team: dict, return_proba: bool = True) -> dict:
        """
        Predict match outcome between two teams.
        Returns expected goals, win/draw probabilities, and score distribution.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Build feature vector
        feature = np.array([[
            home_team["elo"], away_team["elo"],
            home_team["strength"], away_team["strength"],
            home_team["rank"], away_team["rank"],
            home_team["elo"] - away_team["elo"],
            home_team["strength"] - away_team["strength"],
            away_team["rank"] - home_team["rank"],
        ]])

        # Get predictions from both models
        rf_home = float(self.rf_model_home.predict(feature)[0])
        rf_away = float(self.rf_model_away.predict(feature)[0])

        if self.xgb_model_home is not None and self.xgb_model_away is not None:
            xgb_home = float(self.xgb_model_home.predict(feature)[0])
            xgb_away = float(self.xgb_model_away.predict(feature)[0])
            # Ensemble: average of both models
            exp_home = (rf_home + xgb_home) / 2.0
            exp_away = (rf_away + xgb_away) / 2.0
        else:
            exp_home = rf_home
            exp_away = rf_away

        # Clamp to realistic ranges
        exp_home = max(0.1, min(exp_home, 6.0))
        exp_away = max(0.1, min(exp_away, 6.0))

        result = {
            "home_team": home_team["name"],
            "away_team": away_team["name"],
            "expected_home_goals": round(exp_home, 2),
            "expected_away_goals": round(exp_away, 2),
        }

        if return_proba:
            # Compute Poisson distribution for all scores 0-6
            score_probs = {}
            win_prob = 0.0
            draw_prob = 0.0
            lose_prob = 0.0
            most_likely_score = None
            max_prob = 0.0

            for hg in range(7):
                for ag in range(7):
                    p_h = self._poisson_pmf(hg, exp_home)
                    p_a = self._poisson_pmf(ag, exp_away)
                    prob = p_h * p_a
                    key = f"{hg}-{ag}"
                    score_probs[key] = round(prob * 100, 2)

                    if prob > max_prob:
                        max_prob = prob
                        most_likely_score = key

                    if hg > ag:
                        win_prob += prob
                    elif hg == ag:
                        draw_prob += prob
                    else:
                        lose_prob += prob

            result["score_probabilities"] = score_probs
            result["most_likely_score"] = most_likely_score
            result["win_probability"] = round(win_prob * 100, 1)
            result["draw_probability"] = round(draw_prob * 100, 1)
            result["lose_probability"] = round(lose_prob * 100, 1)

        return result

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function: P(X=k) = (e^-lam * lam^k) / k!"""
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    def save_model(self, path: str):
        """Save trained models to disk."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "rf_home": self.rf_model_home,
            "rf_away": self.rf_model_away,
            "xgb_home": self.xgb_model_home,
            "xgb_away": self.xgb_model_away,
            "feature_columns": self.feature_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path: str) -> bool:
        """Load trained models from disk."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.rf_model_home = data["rf_home"]
            self.rf_model_away = data["rf_away"]
            self.xgb_model_home = data.get("xgb_home")
            self.xgb_model_away = data.get("xgb_away")
            self.feature_columns = data.get("feature_columns", self.feature_columns)
            self.is_trained = True
            return True
        except Exception:
            return False


class ELOBasedPredictor:
    """
    Pure ELO-based predictor as a simpler alternative.
    Uses expected goals formula: E[goals] = strength_ratio * avg_goals
    """

    @staticmethod
    def predict(home_team: dict, away_team: dict) -> dict:
        """Simple ELO-based prediction."""
        avg_goals = 2.5
        total_strength = home_team["strength"] + away_team["strength"]

        exp_home = avg_goals * (home_team["strength"] / total_strength) + 0.3  # home advantage
        exp_away = avg_goals * (away_team["strength"] / total_strength)

        # Compute Poisson probabilities
        score_probs = {}
        win_prob = draw_prob = lose_prob = 0.0
        most_likely = None
        max_p = 0.0

        for hg in range(7):
            for ag in range(7):
                p = (math.exp(-exp_home) * (exp_home ** hg) / math.factorial(hg)) * \
                    (math.exp(-exp_away) * (exp_away ** ag) / math.factorial(ag))
                prob = p * 100
                key = f"{hg}-{ag}"
                score_probs[key] = round(prob, 2)
                if p > max_p:
                    max_p = p
                    most_likely = key
                if hg > ag:
                    win_prob += p
                elif hg == ag:
                    draw_prob += p
                else:
                    lose_prob += p

        return {
            "home_team": home_team["name"],
            "away_team": away_team["name"],
            "expected_home_goals": round(exp_home, 2),
            "expected_away_goals": round(exp_away, 2),
            "score_probabilities": score_probs,
            "most_likely_score": most_likely,
            "win_probability": round(win_prob * 100, 1),
            "draw_probability": round(draw_prob * 100, 1),
            "lose_probability": round(lose_prob * 100, 1),
        }


if __name__ == "__main__":
    from data_fetcher import get_training_data, get_team_rating

    # Test training
    features, labels = get_training_data()
    predictor = ScorePredictor()
    results = predictor.train(features, labels)
    print(f"\nTraining results: {results}")

    # Test prediction
    brazil = get_team_rating("Brazil")
    argentina = get_team_rating("Argentina")
    pred = predictor.predict(brazil, argentina)
    print(f"\nBrazil vs Argentina:")
    print(f"  Expected: {pred['expected_home_goals']} - {pred['expected_away_goals']}")
    print(f"  Win/Draw/Lose: {pred['win_probability']}% / {pred['draw_probability']}% / {pred['lose_probability']}%")
    print(f"  Most likely: {pred['most_likely_score']}")

    # Test ELO predictor
    elo_pred = ELOBasedPredictor.predict(brazil, argentina)
    print(f"\nELO Predictor: {elo_pred['most_likely_score']}")
