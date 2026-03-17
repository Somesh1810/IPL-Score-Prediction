"""
predictor.py
Reusable prediction utility — import this in any script or notebook.

Usage:
    from predictor import IPLPredictor
    p = IPLPredictor()
    result = p.predict(
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        overs=12, current_score=100,
        wickets=3, runs_last5=48, wickets_last5=1
    )
    print(result)
"""

import joblib
import json
import numpy as np
import pandas as pd
import os


class IPLPredictor:
    """Load trained artefacts and make predictions."""

    FEATURES = [
        "batting_team", "bowling_team",
        "overs", "current_score", "wickets",
        "runs_last5", "wickets_last5",
        "run_rate", "wickets_left", "balls_remaining",
        "rr_last5", "pressure_index",
    ]

    def __init__(self, model_dir: str = "."):
        model_path = os.path.join(model_dir, "model.pkl")
        bat_enc    = os.path.join(model_dir, "le_bat.pkl")
        bowl_enc   = os.path.join(model_dir, "le_bowl.pkl")
        meta_path  = os.path.join(model_dir, "model_meta.json")

        if not all(os.path.exists(p) for p in [model_path, bat_enc, bowl_enc, meta_path]):
            raise FileNotFoundError(
                "Model artefacts missing. Run:\n"
                "  python generate_data.py\n"
                "  python train_model.py"
            )

        self.model   = joblib.load(model_path)
        self.le_bat  = joblib.load(bat_enc)
        self.le_bowl = joblib.load(bowl_enc)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.teams = self.meta["teams"]

    # ──────────────────────────────────────────────────────────────────────────
    def predict(
        self,
        batting_team: str,
        bowling_team: str,
        overs: int,
        current_score: int,
        wickets: int,
        runs_last5: int,
        wickets_last5: int,
    ) -> dict:
        """
        Returns a dict with:
            predicted_score (int)
            lower_bound     (int)
            upper_bound     (int)
            run_rate        (float)
            required_rr     (float)
            balls_remaining (int)
            wickets_left    (int)
        """
        # Validate
        if batting_team not in self.teams:
            raise ValueError(f"Unknown batting team: {batting_team}")
        if bowling_team not in self.teams:
            raise ValueError(f"Unknown bowling team: {bowling_team}")
        if batting_team == bowling_team:
            raise ValueError("Batting and bowling teams must be different.")

        # Feature engineering
        run_rate        = current_score / max(overs, 1)
        wickets_left    = 10 - wickets
        balls_remaining = (20 - overs) * 6
        rr_last5        = runs_last5 / 5.0
        pressure_index  = wickets / max(overs, 1)

        bat_enc  = self.le_bat.transform([batting_team])[0]
        bowl_enc = self.le_bowl.transform([bowling_team])[0]

        input_df = pd.DataFrame([{
            "batting_team":    bat_enc,
            "bowling_team":    bowl_enc,
            "overs":           overs,
            "current_score":   current_score,
            "wickets":         wickets,
            "runs_last5":      runs_last5,
            "wickets_last5":   wickets_last5,
            "run_rate":        run_rate,
            "wickets_left":    wickets_left,
            "balls_remaining": balls_remaining,
            "rr_last5":        rr_last5,
            "pressure_index":  pressure_index,
        }])

        prediction      = int(self.model.predict(input_df)[0])
        remaining_runs  = max(prediction - current_score, 0)
        remaining_overs = 20 - overs

        return {
            "predicted_score": prediction,
            "lower_bound":     max(current_score + 5, prediction - 10),
            "upper_bound":     prediction + 10,
            "run_rate":        round(run_rate, 2),
            "required_rr":     round(remaining_runs / max(remaining_overs, 1), 2),
            "balls_remaining": balls_remaining,
            "wickets_left":    wickets_left,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def batch_predict(self, records: list[dict]) -> list[dict]:
        """Predict for a list of match-state dicts."""
        return [self.predict(**r) for r in records]


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = IPLPredictor()
    result = p.predict(
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        overs=12,
        current_score=100,
        wickets=3,
        runs_last5=48,
        wickets_last5=1,
    )
    print("\n📊 Prediction Result")
    print("─" * 35)
    for k, v in result.items():
        print(f"  {k:<20} : {v}")
