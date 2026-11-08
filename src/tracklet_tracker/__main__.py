import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from tracklet_tracker.feature_preparation import build_all_features
from tracklet_tracker.train import build_model, get_model_probabilities
from tracklet_tracker.evaluate import (
    compute_confusion_matrix,
    compute_evaluation_metrics,
    print_feature_importance,
)


def build_complete_dataset(base_path: str, limit: int):
    all_scenarios_data = []

    for i in range(1, limit + 1):
        OBSERVATIONS_PATH = f"{base_path}/scenario_{i:03d}/observations.jsonl"
        LABELS_PATH = f"{base_path}/scenario_{i:03d}/labels.jsonl"
        SUPPORTING_PATH = f"{base_path}/scenario_{i:03d}/supporting.jsonl"
        SNAPSHOT_PATH = f"{base_path}/scenario_{i:03d}/snapshot.json"

        print(f"Processing scenario {i:03d}...")

        scenario_df = build_all_features(
            OBSERVATIONS_PATH, LABELS_PATH, SUPPORTING_PATH, SNAPSHOT_PATH
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


def load_saved_model(save_path: str) -> xgb.XGBClassifier:
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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



def run_evaluation(
    X: pd.DataFrame, y: pd.Series, model: xgb.XGBClassifier, threshold=0.9
):
    y_proba = get_model_probabilities(X, model)

    compute_confusion_matrix(y, y_proba, threshold)
    compute_evaluation_metrics(y, y_proba, threshold)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    master_path = project_root / "data" / "candidate-packet" / "data"

    train_path = master_path / "train"
    test_path = master_path / "test"
    val_path = master_path / "validation"

    full_train, full_test, full_val, X_train, X_test, X_val, y_train, y_test, y_val = (
        build_train_test_eval_set(train_path, test_path, val_path)
    )

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)

    #   Train the model on training data (Optionally save the model)
    model = build_model(X_train, y_train, X_val, y_val, project_root / "models")

    #   Load the saved model (If you are skipping the training)
    # model = load_saved_model(project_root / "models" / "xgb_model.json")

    #   Evaluate on validation data
    print("Validation Data performance:")
    run_evaluation(X_val, y_val, model, 0.9)

    print("\n\n\n")

    #   Evaluate on test data
    print("Test Data performance:")
    run_evaluation(X_test, y_test, model, 0.9)

    #   Print Feature Importance
    print_feature_importance(model, X_train)
