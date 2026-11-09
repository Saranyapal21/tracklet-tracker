
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from tracklet_tracker.inference import run_association_inference
from tracklet_tracker.helper_functions import load_saved_model


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.json"

REQUIRED_FILES = {
    "observations.jsonl",
    "supporting.jsonl",
}

OPTIONAL_FILE = "labels.jsonl"


st.set_page_config(
    page_title="Tracklet Tracker",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Tracklet Tracker")

st.caption(
    "Aircraft tracklet association using rule-based "
    "and machine learning models."
)


# =========================================================
# Load saved model
# =========================================================

@st.cache_resource
def load_model(model_path: str, modified_time: float):

    return load_saved_model(model_path)


if not MODEL_PATH.is_file():

    st.error(
        f"Model artifact not found: {MODEL_PATH}"
    )

    st.stop()


try:

    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

except Exception as exc:

    st.error(f"Unable to load model: {exc}")
    st.stop()


# =========================================================
# File readers
# =========================================================

def read_jsonl(
    uploaded_file,
    allow_empty: bool = False,
) -> pd.DataFrame:

    content = uploaded_file.getvalue()

    if not content.strip():

        if allow_empty:
            return pd.DataFrame()

        raise ValueError(
            f"{uploaded_file.name} is empty."
        )

    return pd.read_json(
        BytesIO(content),
        lines=True,
    )


# =========================================================
# Upload interface
# =========================================================

st.subheader("Upload input data")

upload_mode = st.radio(
    "Choose input method",
    [
        "Upload individual files",
        "Upload folder",
    ],
    horizontal=True,
)

observations_file = None
supporting_file = None
labels_file = None

upload_error = None


# ---------------------------------------------------------
# Method 1: Individual files
# ---------------------------------------------------------

if upload_mode == "Upload individual files":

    col1, col2 = st.columns(2)

    with col1:

        observations_file = st.file_uploader(
            "Position reports (observations.jsonl)",
            type=["jsonl"],
            key="observations",
        )

    with col2:

        supporting_file = st.file_uploader(
            "Supporting records (supporting.jsonl)",
            type=["jsonl"],
            key="supporting",
        )

    labels_file = st.file_uploader(
        "Ground-truth labels (optional: labels.jsonl)",
        type=["jsonl"],
        key="labels",
    )


# ---------------------------------------------------------
# Method 2: Folder upload
# ---------------------------------------------------------

else:

    uploaded_files = st.file_uploader(
        "Select the folder containing your input files",
        accept_multiple_files="directory",
        key="folder",
    )

    if uploaded_files:

        files_by_name = {}

        for uploaded_file in uploaded_files:

            # Support relative paths returned by directory upload.
            filename = Path(
                uploaded_file.name.replace("\\", "/")
            ).name

            if filename in files_by_name:

                upload_error = (
                    f"Duplicate filename found: {filename}"
                )

                break

            files_by_name[filename] = uploaded_file

        if upload_error is None:

            actual_files = set(files_by_name.keys())

            allowed_file_sets = [
                REQUIRED_FILES,
                REQUIRED_FILES | {OPTIONAL_FILE},
            ]

            if actual_files not in allowed_file_sets:

                upload_error = (
                    "The selected folder must contain exactly "
                    "observations.jsonl and supporting.jsonl. "
                    "labels.jsonl is optional. "
                    f"Found: {sorted(actual_files)}"
                )

            else:

                observations_file = files_by_name[
                    "observations.jsonl"
                ]

                supporting_file = files_by_name[
                    "supporting.jsonl"
                ]

                labels_file = files_by_name.get(
                    "labels.jsonl"
                )


if upload_error:
    st.error(upload_error)


# =========================================================
# Inference configuration
# =========================================================

st.subheader("Inference configuration")

threshold = st.slider(
    "Association threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.90,
    step=0.01,
)

st.divider()


# =========================================================
# Run inference
# =========================================================

if st.button(
    "Run tracklet association",
    type="primary",
    use_container_width=True,
):

    if upload_error:

        st.error(upload_error)

    elif observations_file is None or supporting_file is None:

        st.error(
            "Please upload both observations.jsonl "
            "and supporting.jsonl."
        )

    else:

        try:

            with st.spinner(
                "Processing tracklets and generating predictions..."
            ):

                # Read observations.
                observations_df = read_jsonl(
                    observations_file
                )

                # Supporting records may be empty.
                supporting_df = read_jsonl(
                    supporting_file,
                    allow_empty=True,
                )

                # Labels are optional.
                labels_df = None

                if labels_file is not None:

                    labels_df = read_jsonl(
                        labels_file
                    )

                    required_label_columns = {
                        "tracklet_id",
                        "successor_tracklet_id",
                    }

                    missing = (
                        required_label_columns
                        - set(labels_df.columns)
                    )

                    if missing:
                        raise ValueError(
                            "Missing label columns: "
                            f"{sorted(missing)}"
                        )

                # Run the existing ML pipeline.
                results = run_association_inference(
                    observations_df=observations_df,
                    supporting_df=supporting_df,
                    model=model,
                    threshold=threshold,
                    labels_df=labels_df,
                )

            st.success(
                "Tracklet association completed successfully!"
            )

            # -------------------------------------------------
            # Summary
            # -------------------------------------------------

            st.subheader("Inference summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Tracklets",
                results["tracklet_count"],
            )

            col2.metric(
                "Candidate pairs",
                results["candidate_pair_count"],
            )

            col3.metric(
                "Threshold",
                f'{results["threshold"]:.2f}',
            )

            # -------------------------------------------------
            # Association counts
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            for column, result_key, title in [
                (col1, "baseline", "Rule-based baseline"),
                (col2, "learned", "XGBoost model"),
            ]:

                associated = sum(
                    item["status"] == "associated"
                    for item in results[result_key]
                )

                unresolved = (
                    len(results[result_key]) - associated
                )

                with column:

                    st.markdown(f"**{title}**")

                    st.metric(
                        "Associated tracklets",
                        associated,
                    )

                    st.metric(
                        "Unresolved tracklets",
                        unresolved,
                    )

            st.divider()

            # -------------------------------------------------
            # Optional evaluation
            # -------------------------------------------------

            if results["evaluation"] is not None:

                st.subheader("Model evaluation")

                evaluation = results["evaluation"]

                if evaluation.get("available") is False:

                    st.info(
                        evaluation["reason"]
                    )

                else:

                    evaluation_df = pd.DataFrame([
                        {
                            "Model": "Baseline",
                            **evaluation["baseline"],
                        },
                        {
                            "Model": "XGBoost",
                            **evaluation["learned"],
                        },
                    ])

                    st.dataframe(
                        evaluation_df,
                        hide_index=True,
                        use_container_width=True,
                    )

            else:

                st.info(
                    "No labels uploaded. Predictions are available, "
                    "but ground-truth evaluation was skipped."
                )

            # -------------------------------------------------
            # Detailed predictions
            # -------------------------------------------------

            st.subheader("Association predictions")

            baseline_tab, learned_tab = st.tabs([
                "Rule-based baseline",
                "XGBoost model",
            ])

            with baseline_tab:

                baseline_df = pd.DataFrame(
                    results["baseline"]
                )

                st.dataframe(
                    baseline_df[
                        [
                            "tracklet_id",
                            "status",
                            "accepted_successor_id",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

                with st.expander(
                    "View ranked baseline candidates"
                ):

                    st.json(results["baseline"])

            with learned_tab:

                learned_df = pd.DataFrame(
                    results["learned"]
                )

                st.dataframe(
                    learned_df[
                        [
                            "tracklet_id",
                            "status",
                            "accepted_successor_id",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

                with st.expander(
                    "View ranked XGBoost candidates"
                ):

                    st.json(results["learned"])

            # -------------------------------------------------
            # Download results
            # -------------------------------------------------

            output_json = json.dumps(
                results,
                indent=4,
            )

            st.download_button(
                label="Download inference results",
                data=output_json,
                file_name="tracklet_association_results.json",
                mime="application/json",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Inference failed: {exc}"
            )

            st.exception(exc)
