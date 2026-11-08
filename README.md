# Govia Assignment
This project, developed as part of the Govia assignment, implements a robust machine learning pipeline designed to resolve and link fragmented 
aircraft trajectories using noisy, anonymous telemetry data. The pipeline cleans raw observation inputs, handles data conflicts, and engineers rigorous 
physical kinematics such as Haversine distances, time gaps, required flight speeds etc. By formulating the problem as a pairwise binary 
classification task and using a tuned XGBoost classifier alongside a Argmax inference filter, the pipeline successfully predicts true aircraft 
successor connections while prioritizing high precision over recall to strictly minimize costly false-positive associations, ultimately 
outperforming the rule-based baseline.

## Set up the Project Locally:

**Prerequisite:** This project requires uv, a Python package and project manager, to install dependencies and run the application. All project dependencies are specified in pyproject.toml, so there is no need to install them manually. The project follows a package-based structure, making it straightforward to set up and run. Simply follow the steps below:

1. Clone the repository
   ```bash
   git clone https://github.com/Saranyapal21/tracklet-tracker.git
   ```

2. Navigate to the project root
   ```bash
   cd tracklet-tracker/
   ```

3. Create a uv virtual environment
   ```bash
   uv venv
   ```

4. Activate the venv created
    ```bash
    source .venv/bin/activate
    ```

5. Install the project and it's dependencies
    ```bash
    uv sync
    ```

6. Once the project is properly setup and installed, run the pipeline for tracklet-tracker as a module. This will process the data, prepare features
   from them, train the XGBoost based model and perform inference on test data. It will run the baseline rule-based pipeline as well. So, in the end,    both the perforrmance metrices will be printed and we will be able to easily compare them
    ```bash
    python -m tracklet_tracker
    ```

7. [Optional] If you want to save the test set predictions, and their status as well, run the below command:
   ```bash
    python -m tracklet_tracker.inference
    ```


## Approach:

1. **Data Cleaning**: Processed raw sensor files, discarded out-of-bounds geographic coordinates, and bypassed missing altitude records to ensure clean spatial inputs.
2. **Duplicate Resolution**: Managed duplicate telemetry streams by retaining single valid retransmissions while entirely dropping conflicting records sharing the same observation ID.
3. **Tracklet Boundary Aggregation**: Sorted observations strictly by timestamp to compress time-series data into single start and end points for each tracklet.
4. **Candidate Pair Generation**: Generated complete possible connections within each independent scenario using a filtered Cartesian product, enforcing a strict temporal rule where subsequent tracklets must start after preceding ones finish.
5. **Kinematic Feature Engineering**: Computed physics features for every candidate pair, including _Haversine gap distances_, _time deltas_, _required flight speeds_, _directional bearings_ and _transponder reference matches_.
6. **Binary Target Labeling**: Formulated ground truth labels by performing a left join against scenario edge-lists, assigning a positive target to true consecutive pairs and a negative target to all unresolved, skipped, or invalid pairs.
7. **Model Inference & Precision Enforcement**: Deployed a tuned XGBoost classifier paired with a strict confidence threshold and Argmax filter to select only the single best successor per tracklet, satisfying business constraints, outperforming the rule-based baseline.


## Test Data Performance Comparison:

Total no. of rows (test data points): 3,870

**Confusion Matrix (Threshold = 0.9):**

**NOTE**: Incorrectly joining two different aircrafts is heavily penalized and not correct, whereas leaving a trajectory case unresolved 
(predicting null) is completely acceptable. So, we need to reduce the number of False Positives. To satisfy this cause, we have set the threshold to a higher value of 0.9

| Metric | Baseline | Data-Driven |
|---|---:|---:|
| True Negative (TN) | 3,181 | 3,551 |
| False Positive (FP) | 422 | 52 |
| False Negative (FN) | 210 | 204 |
| True Positive (TP) | 129 | 135 | 

     





**Performance Metrics:**

| Metric | Baseline | Data-Driven |
|---|---:|---:|
| AUC-ROC | 0.668 | **0.942** |
| Gini | 0.336 | **0.884** |
| Precision | 0.234 | **0.722** |
| Recall | 0.381 | **0.398** |
| Accuracy | 0.840 | **0.935** |
| Total Events | 339 | 339 |
| Event Rate | 0.086 | 0.086 |
| Capture Rate @ 10% | 0.327 | **0.628** |
| Capture Rate @ 25% | 0.487 | **0.932** |
| Capture Rate @ 40% | 0.575 | **0.973** |
| Capture Rate @ 50% | 0.655 | **0.991** |

*Bold values indicate improvements in the data-driven model over the baseline.*
