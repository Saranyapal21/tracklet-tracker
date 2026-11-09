import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from tracklet_tracker.feature_preparation import build_all_features


def get_model_probabilities(X: pd.DataFrame, model: xgb.XGBClassifier):
    return model.predict_proba(X)[:, 1]


def load_saved_model(save_path: str, pos_weight: float = 10.665689) -> xgb.XGBClassifier:

    save_path = Path(save_path)
    if not save_path:
        raise FileNotFoundError(f"Model artifact not found: {save_path}")

    model = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        max_depth=4,
        learning_rate=0.1,
        n_estimators=150,
        random_state=42,
    )

    model.load_model(save_path)
    return model




def build_complete_dataset(base_path: str, limit: int):
    all_scenarios_data = []

    for i in range(1, limit + 1):
        OBSERVATIONS_PATH = f"{base_path}/scenario_{i:03d}/observations.jsonl"
        LABELS_PATH = f"{base_path}/scenario_{i:03d}/labels.jsonl"
        SUPPORTING_PATH = f"{base_path}/scenario_{i:03d}/supporting.jsonl"

        print(f"Processing scenario {i:03d}...")

        scenario_df = build_all_features(
            OBSERVATIONS_PATH, LABELS_PATH, SUPPORTING_PATH,
        )
        all_scenarios_data.append(scenario_df)

    full_dataset = pd.concat(all_scenarios_data, ignore_index=True)

    X = full_dataset.drop(
        columns=["target", "A_tracklet_id", "B_tracklet_id"], errors="ignore"
    )
    X = X.select_dtypes(exclude=["object", "string"])
    y = full_dataset["target"]

    return full_dataset, X, y



def build_train_test_eval_set(train_path: str, test_path: str, val_path: str):

    full_train, X_train, y_train = build_complete_dataset(train_path, 80)
    full_test, X_test, y_test = build_complete_dataset(test_path, 20)
    full_val, X_val, y_val = build_complete_dataset(val_path, 20)

    return (
        full_train,
        full_test,
        full_val,
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
    )

