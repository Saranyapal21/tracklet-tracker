import json
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "TRACKLET_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Tracklet Tracker",
    page_icon="✈️",
    layout="wide",
)


def records_to_jsonl(records: list[dict]) -> str:
    return "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records
    )


def display_associations(records: list[dict], score_label: str):

    # Tracklet-level summary
    display_df = pd.DataFrame(
        [
            {
                "Tracklet ID": record["tracklet_id"],
                "Status": record["status"],
                "Accepted Successor": record["accepted_successor_id"],
                "Number of Candidates": len(record["ranked_candidates"]),
            }
            for record in records
        ]
    )

    st.subheader("Tracklet Associations")

    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
    )

    # Candidate-level scores
    candidate_rows = []

    for record in records:

        for rank, candidate in enumerate(
            record["ranked_candidates"], start=1
        ):

            measurements = candidate["supporting_measurements"]

            candidate_rows.append(
                {
                    "Tracklet ID": record["tracklet_id"],
                    "Rank": rank,
                    "Candidate Successor": candidate["successor_tracklet_id"],
                    score_label: candidate["score"],
                    "Decision": (
                        "Accepted"
                        if candidate["successor_tracklet_id"]
                        == record["accepted_successor_id"]
                        else "Not selected"
                    ),
                    "Gap Distance (m)": measurements["gap_distance_meters"],
                    "Gap Time (s)": measurements["gap_time_seconds"],
                    "Required Speed (m/s)": measurements["required_speed_mps"],
                }
            )

    st.subheader("Ranked Candidate Scores")

    if candidate_rows:

        scores_df = pd.DataFrame(candidate_rows)

        st.dataframe(
            scores_df,
            width='stretch',
            hide_index=True,
            column_config={
                score_label: st.column_config.NumberColumn(
                    score_label,
                    format="%.4f",
                ),
                "Gap Distance (m)": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "Gap Time (s)": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "Required Speed (m/s)": st.column_config.NumberColumn(
                    format="%.2f",
                ),
            },
        )

    else:
        st.info("No eligible candidate pairs were generated.")


def display_metrics(metrics: dict):

    columns = st.columns(4)

    columns[0].metric(
        "ROC-AUC",
        f"{metrics['auc_roc']:.3f}",
    )

    columns[1].metric(
        "Gini",
        f"{metrics['gini']:.3f}",
    )

    columns[2].metric(
        "Precision",
        f"{metrics['precision']:.3f}",
    )

    columns[3].metric(
        "Recall",
        f"{metrics['recall']:.3f}",
    )


st.title("Tracklet Tracker")

st.markdown(
    """
    Upload aircraft observation and supporting data to generate
    tracklet associations using a rule-based baseline and a trained
    XGBoost classifier.
    """
)


# API configuration
with st.sidebar:

    st.header("API Configuration")

    api_url = st.text_input(
        "FastAPI URL",
        value=API_URL,
    ).rstrip("/")


# Input files
st.header("Input Data")

col1, col2 = st.columns(2)

with col1:

    observations_file = st.file_uploader(
        "Observations JSONL",
        type=["jsonl"],
        accept_multiple_files=False,
    )

with col2:

    supporting_file = st.file_uploader(
        "Supporting JSONL",
        type=["jsonl"],
        accept_multiple_files=False,
    )


labels_file = st.file_uploader(
    "Labels JSONL (Optional — evaluation only)",
    type=["jsonl"],
    accept_multiple_files=False,
)


# Threshold
st.header("Inference Configuration")

threshold = st.slider(
    "Association confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.90,
    step=0.01,
)


run_inference = st.button(
    "Run Tracklet Association",
    type="primary",
    disabled=(
        observations_file is None
        or supporting_file is None
    ),
)


if run_inference:

    files = {
        "observations_file": (
            observations_file.name,
            observations_file.getvalue(),
            "application/x-ndjson",
        ),
        "supporting_file": (
            supporting_file.name,
            supporting_file.getvalue(),
            "application/x-ndjson",
        ),
    }

    if labels_file is not None:

        files["labels_file"] = (
            labels_file.name,
            labels_file.getvalue(),
            "application/x-ndjson",
        )

    data = {
        "threshold": str(threshold),
    }

    try:

        with st.spinner("Running tracklet association..."):

            response = requests.post(
                f"{api_url}/tracklets/associate",
                files=files,
                data=data,
                timeout=(10, 600),
            )

        if response.status_code != 200:

            error_response = response.json()

            error = error_response.get(
                "error",
                error_response,
            )

            st.error(
                error.get(
                    "message",
                    "Unable to complete inference.",
                )
            )

            if error.get("details"):
                st.json(error["details"])

        else:

            st.session_state["association_result"] = response.json()

            st.success("Tracklet association completed successfully!")

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to FastAPI. "
            "Make sure the backend server is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The inference request timed out. "
            "Try a smaller dataset or increase the request timeout."
        )

    except requests.exceptions.RequestException as exc:

        st.error(f"Request failed: {exc}")

    except ValueError:

        st.error("The API returned an invalid response.")



if "association_result" in st.session_state:

    result = st.session_state["association_result"]

    st.divider()

    st.header("Association Results")

    st.caption(
        f"Confidence threshold: {result['threshold']:.2f}"
    )

    learned = result["learned"]
    baseline = result["baseline"]

    tab1, tab2 = st.tabs(
        ["Data-Driven Model", "Rule-Based Baseline"]
    )

    with tab1:

        st.subheader("XGBoost Predictions")

        associated = sum(
            record["status"] == "associated"
            for record in learned
        )

        unresolved = len(learned) - associated

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tracklets", len(learned))
        col2.metric("Associated", associated)
        col3.metric("Unresolved", unresolved)

        display_associations(learned, "Model Score")

        st.download_button(
            "Download Learned Predictions",
            data=records_to_jsonl(learned),
            file_name="learned_predictions.jsonl",
            mime="application/x-ndjson",
        )

    with tab2:

        st.subheader("Rule-Based Predictions")

        associated = sum(
            record["status"] == "associated"
            for record in baseline
        )

        unresolved = len(baseline) - associated

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tracklets", len(baseline))
        col2.metric("Associated", associated)
        col3.metric("Unresolved", unresolved)

        display_associations(baseline, "Baseline Score")

        st.download_button(
            "Download Baseline Predictions",
            data=records_to_jsonl(baseline),
            file_name="baseline_predictions.jsonl",
            mime="application/x-ndjson",
        )

    # Optional evaluation
    evaluation = result.get("evaluation")

    if evaluation is not None:

        st.divider()

        st.header("Model Evaluation")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Rule-Based Baseline")
            display_metrics(evaluation["baseline"])

        with col2:

            st.subheader("Data-Driven Model")
            display_metrics(evaluation["learned"])

            