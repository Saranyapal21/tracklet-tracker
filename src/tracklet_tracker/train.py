from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb


def build_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_dir: Optional[Path],
):

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        max_depth=4,
        learning_rate=0.1,
        n_estimators=150,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model.save_model(save_path / "xgb_model.json")

    return model


def get_model_probabilities(X: pd.DataFrame, model: xgb.XGBClassifier):
    return model.predict_proba(X)[:, 1]
