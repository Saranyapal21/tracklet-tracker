import numpy as np
import pandas as pd

def get_baseline_probabilities(X: pd.DataFrame) -> pd.Series:
    
    # Max speed constraint (Commercial aircraft rarely exceed 300 m/s)
    valid_speed_mask = X['required_speed_mps'] <= 300.0
    
    # How closely does the required gap speed match the aircraft's reported speed?
    # If observed speed is missing (NaN), assume a standard cruising speed of 200 m/s.
    speed_diff = X['speed_diff_A_observ'].fillna(abs(X['required_speed_mps'] - 200.0))
    
    # Convert speed difference into a score from 0.0 to 1.0 using exponential decay.
    base_score = np.exp(-speed_diff / 50.0)
    
    # If the transponder reference IDs match exactly, give a massive confidence boost (+0.4)
    if 'ref_match_exact' in X.columns:
        base_score += (X['ref_match_exact'] * 0.4)
        
    
    final_scores = np.clip(base_score, 0, 1.0) * valid_speed_mask.astype(int)
    return final_scores

