import json
import numpy as np
import pandas as pd
from pathlib import Path

from tracklet_tracker.__main__ import (
    build_train_test_eval_set,
    build_model,
    get_model_probabilities,
    load_saved_model
)


def generate_inference_output(A_tracklet_id: str, candidate_pairs_df: pd.DataFrame, threshold: float = 0.90) -> dict:
    candidates = candidate_pairs_df[candidate_pairs_df['A_tracklet_id'] == A_tracklet_id].copy()
    candidates = candidates.sort_values(by='probability', ascending=False)
    
    ranked_list = []
    for _, row in candidates.iterrows():
        ranked_list.append({
            "successor_tracklet_id": row['B_tracklet_id'],
            "score": round(row['probability'], 4),
            "supporting_measurements": {
                "gap_distance_meters": round(row.get('gap_distance_meters', 0), 2),
                "gap_time_seconds": round(row.get('gap_time_seconds', 0), 2),
                "required_speed_mps": round(row.get('required_speed_mps', 0), 2)
            }
        })

    accepted_successor = None
    status = "unresolved"
    
    if not candidates.empty:
        top_candidate = candidates.iloc[0]
        if top_candidate['probability'] >= threshold:
            accepted_successor = top_candidate['B_tracklet_id']
            status = "associated"
            
    output = {
        "tracklet_id": A_tracklet_id,
        "status": status,
        "accepted_successor_id": accepted_successor,
        "ranked_candidates": ranked_list
    }
    
    return output


def save_test_predictions(X_test, y_test, y_proba, output_path, threshold=0.90):
    results_df = X_test.copy()
    
    if len(np.shape(y_proba)) > 1 and np.shape(y_proba)[1] == 2:
        results_df['probability'] = y_proba[:, 1]
    else:
        results_df['probability'] = y_proba
        
    if y_test is not None:
        results_df['true_label'] = np.array(y_test)

    final_outputs = []
    unique_tracklets = results_df['A_tracklet_id'].unique()
    
    print(f"Generating inference outputs for {len(unique_tracklets)} unique tracklets...")
    for tracklet_a in unique_tracklets:
        record = generate_inference_output(
            A_tracklet_id=tracklet_a,
            candidate_pairs_df=results_df,
            threshold=threshold
        )
        final_outputs.append(record)
        
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in final_outputs:
            f.write(json.dumps(record) + '\n')
            
    print(f"Successfully saved test predictions to {output_path}")
    return final_outputs


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    master_path = project_root / "data" / "candidate-packet" / "data"

    train_path = master_path / "train"
    test_path = master_path / "test"
    val_path = master_path / "validation"

    full_train, full_test, full_val, X_train, X_test, X_val, y_train, y_test, y_val = (
        build_train_test_eval_set(train_path, test_path, val_path)
    )

    #   [Optional] If you want to again train the model before inference
    # model = build_model(X_train, y_train, X_val, y_val, project_root / "models")

    model = load_saved_model(project_root / "models" / "xgb_model.json")
    y_proba = get_model_probabilities(X_test, model)
    
    output_file = project_root / "outputs" / "test_inference_predictions.jsonl"
    save_test_predictions(
        X_test=full_test, 
        y_test=y_test, 
        y_proba=y_proba, 
        output_path=output_file, 
        threshold=0.90
    )

