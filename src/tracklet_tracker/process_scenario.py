import duckdb
import numpy as np
import pandas as pd


def process_sql_query(query: str):
    return duckdb.query(query).df()


def deduplicating_df(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    df = df.drop_duplicates(subset=[subset], keep=False)

    return df.reset_index(drop=True)


def process_observations(observations_path: str) -> pd.DataFrame:
    observations_df = pd.read_json(observations_path, lines=True)
    observations_df = deduplicating_df(observations_df, "observation_id")

    #   Removing all rows with invalid latitude and longitude
    observations_df = observations_df[
        observations_df["latitude"].between(-90.0, 90.0)
        & observations_df["longitude"].between(-180.0, 180.0)
    ]

    #   Converting to DateTime
    observations_df["observed_at"] = pd.to_datetime(observations_df["observed_at"])
    observations_df["received_at"] = pd.to_datetime(observations_df["received_at"])

    attributes_df = pd.json_normalize(observations_df["attributes"]).add_prefix(
        "observ_"
    )
    observations_df = pd.concat(
        [observations_df.drop(columns=["attributes"]), attributes_df], axis=1
    )

    if "observ_spd" in observations_df.columns:
        is_feed_a = (observations_df["source"] == "feed-a") & (
            observations_df["observ_spd"].notna()
        )
        observations_df.loc[is_feed_a, "observ_spd"] = (
            observations_df.loc[is_feed_a, "observ_spd"] * 0.514444
        )

    return observations_df


def process_supporting(supporting_path: str) -> pd.DataFrame:
    supporting_df = pd.read_json(supporting_path, lines=True)
    supporting_df = deduplicating_df(supporting_df, "record_id")

    if "received_at" in supporting_df.columns:
        supporting_df["received_at"] = pd.to_datetime(supporting_df["received_at"])

    if "attributes" in supporting_df.columns:
        attributes_df_2 = pd.json_normalize(supporting_df["attributes"]).add_prefix(
            "supporting_"
        )
        supporting_df = pd.concat(
            [supporting_df.drop(columns=["attributes"]), attributes_df_2], axis=1
        )

    return supporting_df


def read_scenario(
    observations_path: str, labels_path: str, supporting_path: str, snapshot_path: str
):
    observations_df = process_observations(observations_path)
    labels_df = pd.read_json(labels_path, lines=True)
    supporting_df = process_supporting(supporting_path)
    snapshot_df = pd.read_json(snapshot_path)

    return observations_df, labels_df, supporting_df, snapshot_df


def table_join(observations_df, supporting_df):
    merged_df = pd.merge(
        observations_df,
        supporting_df,
        on="observation_id",
        how="left",
        suffixes=("", "_supp"),
    )

    #   Sorting by tracklet_id and time (Probably the most important step)
    merged_df = merged_df.sort_values(by=["tracklet_id", "observed_at"]).reset_index(
        drop=True
    )
    return merged_df


def aggregation(merged_df: pd.DataFrame) -> pd.DataFrame:
    #   The clarifying document mentions:
    #   "Each tracklet contains observations from exactly one aircraft.
    #   Its membership is fixed. You do not need to split tracklets, merge duplicate tracklets,
    #   or question whether observations within a tracklet belong together."
    agg_funcs = {
        "observed_at": ["first", "last"],
        "latitude": ["first", "last"],
        "longitude": ["first", "last"],
        "source": ["first"],
        "observ_ref": ["first"],
        "observ_spd": ["mean"],
        "observ_hdg": ["mean"],
        "supporting_spd": ["mean"],
    }

    tracklets_summary = merged_df.groupby("tracklet_id").agg(agg_funcs).reset_index()
    tracklets_summary.columns = [
        "tracklet_id",
        "start_time",
        "end_time",
        "start_lat",
        "end_lat",
        "start_lon",
        "end_lon",
        "source",
        "observ_ref",
        "observ_spd",
        "observ_hdg",
        "supporting_spd",
    ]

    return tracklets_summary
