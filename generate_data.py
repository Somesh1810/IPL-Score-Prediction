"""
generate_data.py
Generates a realistic synthetic IPL dataset for model training.
Run this once to produce data.csv before training.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Delhi Capitals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]

N = 5000  # number of innings records


def simulate_innings():
    batting_team = np.random.choice(TEAMS)
    bowling_team = np.random.choice([t for t in TEAMS if t != batting_team])

    # Over is between 5 and 19 (prediction mid-innings)
    over = np.random.randint(5, 20)

    # Base run rate influenced by team strength (random team factor)
    team_factor = np.random.uniform(0.85, 1.15)

    # Current runs roughly: run_rate * overs * team_factor + noise
    base_rr = np.random.uniform(6.5, 9.5)
    current_score = int(base_rr * over * team_factor + np.random.normal(0, 8))
    current_score = max(10, current_score)

    # Wickets: more overs → more likely more wickets
    wickets = int(np.random.beta(1.5, 4) * 10 * (over / 20))
    wickets = min(wickets, 9)

    # Last 5 overs stats
    last5_rr = np.random.uniform(5.0, 14.0)
    runs_last5 = int(last5_rr * min(5, over) + np.random.normal(0, 5))
    runs_last5 = max(0, runs_last5)
    wickets_last5 = min(int(np.random.poisson(0.8)), wickets)

    # Final score: projected based on current momentum + randomness
    remaining_overs = 20 - over
    projected_rr = base_rr * team_factor * np.random.uniform(0.9, 1.2)
    # Wicket penalty
    wicket_penalty = wickets * 1.5
    final_score = int(current_score + projected_rr * remaining_overs - wicket_penalty + np.random.normal(0, 12))
    final_score = max(current_score + 5, final_score)
    final_score = min(final_score, 270)

    return {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "overs": over,
        "current_score": current_score,
        "wickets": wickets,
        "runs_last5": runs_last5,
        "wickets_last5": wickets_last5,
        "final_score": final_score,
    }


rows = [simulate_innings() for _ in range(N)]
df = pd.DataFrame(rows)
df.to_csv("data.csv", index=False)
print(f"Generated {len(df)} records → data.csv")
print(df.head())
