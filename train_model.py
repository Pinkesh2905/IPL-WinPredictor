"""
IPL Win Probability Predictor — Model Training
Generates synthetic ball-by-ball data, engineers features, trains and evaluates models,
then saves the best pipeline to model.pkl.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")
np.random.seed(42)

IPL_TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans",
]

NUMERIC_FEATURES = [
    "runs_left", "balls_left", "wickets_left",
    "current_run_rate", "required_run_rate",
    "pressure_index", "innings_progress",
]
CATEGORICAL_FEATURES = ["batting_team", "bowling_team"]
TARGET = "win"


def generate_match_snapshots(n_matches: int = 4000) -> pd.DataFrame:
    rows = []

    for _ in range(n_matches):
        team_pair = np.random.choice(len(IPL_TEAMS), size=2, replace=False)
        batting_team = IPL_TEAMS[team_pair[0]]
        bowling_team = IPL_TEAMS[team_pair[1]]

        target = int(np.random.normal(175, 25))
        target = np.clip(target, 130, 230)

        n_snapshots = np.random.randint(4, 19)
        balls_checkpoints = sorted(np.random.choice(range(6, 120), size=n_snapshots, replace=False))

        score = 0
        wickets = 0

        for balls_played in balls_checkpoints:
            overs_done = balls_played / 6
            balls_left = 120 - balls_played
            runs_left = target - score

            if wickets >= 10 or score >= target:
                break

            crr = score / overs_done if overs_done > 0 else 0
            rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else 99.0
            wickets_left = 10 - wickets

            rrr_diff = rrr - crr
            wkt_factor = wickets_left / 10
            score_factor = (score - (target * balls_played / 120)) / (target * 0.15 + 1)

            log_odds = (
                -0.8 * rrr_diff
                + 2.5 * score_factor
                + 1.2 * wkt_factor
                + np.random.normal(0, 0.3)
            )
            win_prob = 1 / (1 + np.exp(-log_odds))
            label = int(np.random.random() < win_prob)

            rows.append({
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "target": target,
                "score": score,
                "balls_played": balls_played,
                "wickets_fallen": wickets,
                "runs_left": max(runs_left, 0),
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "current_run_rate": round(crr, 3),
                "required_run_rate": round(min(rrr, 36.0), 3),
                "win": label,
            })

            runs_this_over = int(np.random.normal(8.5, 3))
            score += max(0, min(runs_this_over, 24))
            if np.random.random() < 0.18:
                wickets = min(wickets + 1, 10)

    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["runs_left"] = df["runs_left"].clip(lower=0)
    df["balls_left"] = df["balls_left"].clip(lower=0)
    df["wickets_left"] = df["wickets_left"].clip(0, 10)
    df["required_run_rate"] = df["required_run_rate"].clip(upper=36.0)
    df["current_run_rate"] = df["current_run_rate"].clip(lower=0)
    df["pressure_index"] = (df["required_run_rate"] - df["current_run_rate"]).clip(-10, 20)
    df["innings_progress"] = (120 - df["balls_left"]) / 120
    return df


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print(f"  CV  ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Bowling Wins", "Batting Wins"],
                                 zero_division=0))

    return {"name": name, "pipeline": pipeline, "auc": auc, "cv_auc": cv_auc.mean()}


def main():
    print("Generating synthetic IPL dataset …")
    df = generate_match_snapshots(n_matches=4000)
    df = engineer_features(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False)
    print(f"Dataset saved → data/dataset.csv  ({len(df):,} rows)")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    candidates = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    learning_rate=0.05, random_state=42)),
    ]

    results = []
    for name, model in candidates:
        pipeline = build_pipeline(model)
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

    best = max(results, key=lambda r: r["cv_auc"])
    print(f"\n✅  Best model: {best['name']}  (CV AUC = {best['cv_auc']:.4f})")

    joblib.dump(best["pipeline"], "model.pkl")
    print("Model saved → model.pkl")

    clf = best["pipeline"].named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        pre = best["pipeline"].named_steps["preprocessor"]
        cat_names = list(
            pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
        )
        feature_names = NUMERIC_FEATURES + cat_names
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv("data/feature_importance.csv", index=False)
        print("Feature importances saved → data/feature_importance.csv")


if __name__ == "__main__":
    main()