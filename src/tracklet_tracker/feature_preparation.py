import numpy as np
import pandas as pd

from tracklet_tracker.process_scenario import aggregation, read_scenario, table_join


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in meters."""

    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    lambda1, lambda2 = np.radians(lon1), np.radians(lon2)
    dlambda = lambda2 - lambda1

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)

    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def prepare_features_and_labels(
    tracklets_summary: pd.DataFrame, 
    labels_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Generates all candidate pairs (A -> B), computes physical relationship features,
    and attaches binary targets based on the labels.
    """

    df_a = tracklets_summary.add_prefix("A_")
    df_b = tracklets_summary.add_prefix("B_")

    pairs = pd.merge(df_a.assign(key=1), df_b.assign(key=1), on="key").drop(
        "key", axis=1
    )
    pairs = pairs[pairs["A_tracklet_id"] != pairs["B_tracklet_id"]]

    pairs = pairs[pairs["B_start_time"] > pairs["A_end_time"]].copy()
    pairs["gap_time_seconds"] = (
        pairs["B_start_time"] - pairs["A_end_time"]
    ).dt.total_seconds()

    pairs["gap_distance_meters"] = haversine_distance(
        pairs["A_end_lat"],
        pairs["A_end_lon"],
        pairs["B_start_lat"],
        pairs["B_start_lon"],
    )

    pairs["required_speed_mps"] = pairs["gap_distance_meters"] / (
        pairs["gap_time_seconds"] + 1e-6
    )
    pairs["gap_bearing"] = calculate_bearing(
        pairs["A_end_lat"],
        pairs["A_end_lon"],
        pairs["B_start_lat"],
        pairs["B_start_lon"],
    )

    if "A_observ_ref" in pairs.columns:
        pairs["ref_match_exact"] = (
            (pairs["A_observ_ref"] == pairs["B_observ_ref"])
            & (pairs["A_observ_ref"].notna())
        ).astype(int)

    if "A_source" in pairs.columns:
        pairs["is_same_source"] = (pairs["A_source"] == pairs["B_source"]).astype(int)

    if "A_observ_spd" in pairs.columns:
        pairs["speed_diff_A_observ"] = abs(
            pairs["required_speed_mps"] - pairs["A_observ_spd"]
        )
        pairs["speed_diff_B_observ"] = abs(
            pairs["required_speed_mps"] - pairs["B_observ_spd"]
        )

    if "A_supporting_spd" in pairs.columns:
        pairs["telemetry_speed_diff"] = abs(
            pairs["A_supporting_spd"] - pairs["B_supporting_spd"]
        )



    # Drop core timestamps features (we have already computed derived features before)
    cols_to_drop = [
        "A_start_time",
        "A_end_time",
        "B_start_time",
        "B_end_time",
    ]

    pairs = pairs.drop(columns=cols_to_drop, errors="ignore")

    # Labels are unnecessary during inference.
    if labels_df is None:
        return pairs

    # Attach ground-truth labels during training/evaluation.
    truth_df = labels_df[["tracklet_id", "successor_tracklet_id"]].copy()
    truth_df = truth_df.rename(
        columns={
            "tracklet_id": "A_tracklet_id",
            "successor_tracklet_id": "B_tracklet_id",
        }
    )
    
    truth_df["target"] = 1
    final_dataset = pd.merge(
        pairs,
        truth_df,
        on=["A_tracklet_id", "B_tracklet_id"],
        how="left",
    )

    final_dataset["target"] = (
        final_dataset["target"].fillna(0).astype(int)
    )
    return final_dataset


def build_all_features(
    observations_path: str, labels_path: str, supporting_path: str
):
    observations_df, labels_df, supporting_df = read_scenario(
        observations_path, labels_path, supporting_path
    )

    master_df = table_join(observations_df, supporting_df)
    aggregated_df = aggregation(master_df)
    ml_dataset = prepare_features_and_labels(aggregated_df, labels_df)

    return ml_dataset
