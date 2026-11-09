import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb


from tracklet_tracker.train import build_model
from tracklet_tracker.evaluate import (
    compute_confusion_matrix,
    compute_evaluation_metrics,
    print_feature_importance,
)
from tracklet_tracker.helper_functions import (
    build_train_test_eval_set, 
    get_model_probabilities, 
    load_saved_model
)
from tracklet_tracker.rule_based_scorer import get_baseline_probabilities



def run_evaluation(
    X: pd.DataFrame, y: pd.Series, model: xgb.XGBClassifier, threshold=0.9, 
    baseline=False, y_proba_baseline: float = None
):

    if baseline:
        compute_confusion_matrix(y, y_proba_baseline, threshold)
        compute_evaluation_metrics(y, y_proba_baseline, threshold)

        return

    
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

    print("Printing the final data shapes of train, test and validation accordingly")
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

    print("\n\n\n")

    #   Evaluate baseline on test data
    print("Test data baseline performance:")
    y_baseline_proba = get_baseline_probabilities(X_test)
    run_evaluation(X_test, y_test, None, 0.9, True, y_baseline_proba)

    #   Print Feature Importance
    print_feature_importance(model, X_train)
