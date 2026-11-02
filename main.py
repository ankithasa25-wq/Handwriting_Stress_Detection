import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd


# ============================================================
# HANDWRITING STRESS DETECTION SYSTEM
# ML-Based Handwriting Stress Estimation
# ============================================================

DATASET_FILE = "handwriting_stress_dataset.csv"
MODEL_FILE = "handwriting_stress_model.pkl"


# ============================================================
# 1. FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "pressure_mean",
    "pressure_std",
    "pressure_min",
    "pressure_max",
    "pressure_range",
    "total_distance",
    "duration",
    "average_speed",
    "pen_down_count",
    "pen_up_count"
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n==============================================")
print(" HANDWRITING STRESS DETECTION SYSTEM")
print("==============================================")

if not os.path.exists(DATASET_FILE):

    print("\nERROR: Dataset not found.")
    print("Make sure handwriting_stress_dataset.csv")
    print("is in the same folder as main.py.")
    exit()


data = pd.read_csv(DATASET_FILE)

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)


# ============================================================
# 3. FIND STRESS COLUMN
# ============================================================

possible_targets = [
    "stress_score",
    "stress",
    "dass_stress",
    "dass_stress_score"
]

target_column = None

for column in possible_targets:

    if column in data.columns:

        target_column = column
        break


if target_column is None:

    for column in data.columns:

        if "stress" in column:

            target_column = column
            break


if target_column is None:

    print("\nERROR: Stress column not found.")
    print(data.columns.tolist())
    exit()


print("Stress target column:", target_column)


# ============================================================
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in data.columns
]


if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print("-", feature)

    exit()


# ============================================================
# 5. CLEAN DATA
# ============================================================

for feature in FEATURE_COLUMNS:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


data[target_column] = pd.to_numeric(
    data[target_column],
    errors="coerce"
)


data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data = data.dropna(
    subset=FEATURE_COLUMNS + [target_column]
)


print("Rows after cleaning:", len(data))


# ============================================================
# 6. PREPARE DATA
# ============================================================

X = data[FEATURE_COLUMNS].values.astype(float)

y = data[target_column].values.astype(float)


# ============================================================
# 7. PARTICIPANT-LEVEL SPLIT
# ============================================================

if (
    "user_number" in data.columns
    and "collection" in data.columns
):

    groups = (
        data["collection"].astype(str)
        + "_"
        + data["user_number"].astype(str)
    ).values

elif "user_number" in data.columns:

    groups = data["user_number"].astype(str).values

else:

    groups = np.arange(len(data))


unique_groups = np.unique(groups)

np.random.seed(42)

np.random.shuffle(unique_groups)


# 70% training
# 10% validation
# 20% testing

train_end = int(
    len(unique_groups) * 0.70
)

validation_end = int(
    len(unique_groups) * 0.80
)


train_groups = unique_groups[
    :train_end
]

validation_groups = unique_groups[
    train_end:validation_end
]

test_groups = unique_groups[
    validation_end:
]


train_mask = np.isin(
    groups,
    train_groups
)

validation_mask = np.isin(
    groups,
    validation_groups
)

test_mask = np.isin(
    groups,
    test_groups
)


X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\n==============================================")
print(" DATA SPLIT")
print("==============================================")

print("Training samples  :", len(X_train))
print("Validation samples:", len(X_validation))
print("Testing samples   :", len(X_test))

print("Training users    :", len(train_groups))
print("Validation users  :", len(validation_groups))
print("Testing users     :", len(test_groups))


# ============================================================
# 8. STANDARDIZATION
# ============================================================

feature_mean = X_train.mean(axis=0)

feature_std = X_train.std(axis=0)

feature_std[
    feature_std == 0
] = 1


X_train_scaled = (
    X_train - feature_mean
) / feature_std


X_validation_scaled = (
    X_validation - feature_mean
) / feature_std


X_test_scaled = (
    X_test - feature_mean
) / feature_std


# ============================================================
# 9. RIDGE REGRESSION
# ============================================================

class RidgeRegression:

    def __init__(self, alpha=1.0):

        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        X_bias = np.column_stack(
            [
                np.ones(X.shape[0]),
                X
            ]
        )

        identity = np.eye(
            X_bias.shape[1]
        )

        identity[0, 0] = 0

        matrix = (
            X_bias.T @ X_bias
            + self.alpha * identity
        )

        vector = X_bias.T @ y

        parameters = np.linalg.solve(
            matrix,
            vector
        )

        self.bias = parameters[0]

        self.weights = parameters[1:]

    def predict(self, X):

        return (
            X @ self.weights
            + self.bias
        )


# ============================================================
# 10. KNN REGRESSION
# ============================================================

class KNNRegression:

    def __init__(self, k=15):

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for sample in X:

            distances = np.sqrt(
                np.sum(
                    (self.X_train - sample) ** 2,
                    axis=1
                )
            )

            nearest_indices = np.argsort(
                distances
            )[:self.k]

            nearest_values = (
                self.y_train[
                    nearest_indices
                ]
            )

            prediction = np.mean(
                nearest_values
            )

            predictions.append(
                prediction
            )

        return np.array(predictions)


# ============================================================
# 11. MODEL EVALUATION FUNCTION
# ============================================================

def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def calculate_r2(actual, predicted):

    total = np.sum(
        (
            actual
            - np.mean(actual)
        ) ** 2
    )

    residual = np.sum(
        (
            actual
            - predicted
        ) ** 2
    )

    if total == 0:
        return 0

    return 1 - (
        residual / total
    )


# ============================================================
# 12. TRAIN RIDGE REGRESSION
# ============================================================

print("\n==============================================")
print(" TRAINING MODEL 1: RIDGE REGRESSION")
print("==============================================")


ridge_model = RidgeRegression(
    alpha=1.0
)


ridge_model.fit(
    X_train_scaled,
    y_train
)


ridge_validation_prediction = (
    ridge_model.predict(
        X_validation_scaled
    )
)


ridge_mae = calculate_mae(
    y_validation,
    ridge_validation_prediction
)


ridge_r2 = calculate_r2(
    y_validation,
    ridge_validation_prediction
)


print(
    f"Validation MAE: {ridge_mae:.2f}"
)

print(
    f"Validation R² : {ridge_r2:.3f}"
)


# ============================================================
# 13. TRAIN KNN REGRESSION
# ============================================================

print("\n==============================================")
print(" TRAINING MODEL 2: KNN REGRESSION")
print("==============================================")


knn_model = KNNRegression(
    k=15
)


knn_model.fit(
    X_train_scaled,
    y_train
)


knn_validation_prediction = (
    knn_model.predict(
        X_validation_scaled
    )
)


knn_mae = calculate_mae(
    y_validation,
    knn_validation_prediction
)


knn_r2 = calculate_r2(
    y_validation,
    knn_validation_prediction
)


print(
    f"Validation MAE: {knn_mae:.2f}"
)

print(
    f"Validation R² : {knn_r2:.3f}"
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

if ridge_mae <= knn_mae:

    best_model_name = "Ridge Regression"

    best_model = ridge_model

    validation_mae = ridge_mae

else:

    best_model_name = "KNN Regression"

    best_model = knn_model

    validation_mae = knn_mae


print("\n==============================================")
print(" MODEL SELECTION")
print("==============================================")

print(
    "Selected model:",
    best_model_name
)

print(
    f"Validation MAE: {validation_mae:.2f}"
)


# ============================================================
# 15. TEST SELECTED MODEL
# ============================================================

test_prediction = (
    best_model.predict(
        X_test_scaled
    )
)


test_mae = calculate_mae(
    y_test,
    test_prediction
)


test_r2 = calculate_r2(
    y_test,
    test_prediction
)


print("\n==============================================")
print(" FINAL MODEL PERFORMANCE")
print("==============================================")


print(
    "Final algorithm:",
    best_model_name
)

print(
    f"Test MAE: {test_mae:.2f}"
)

print(
    f"Test R² : {test_r2:.3f}"
)


# ============================================================
# 16. STRESS LEVEL FUNCTION
# ============================================================

def get_stress_level(score):

    if score <= 14:

        return "Normal"

    elif score <= 18:

        return "Mild"

    elif score <= 25:

        return "Moderate"

    elif score <= 33:

        return "Severe"

    else:

        return "Extremely Severe"


# ============================================================
# 17. SHOW TEST PREDICTIONS
# ============================================================

print("\n==============================================")
print(" SAMPLE TEST PREDICTIONS")
print("==============================================")


number_of_samples = min(
    10,
    len(y_test)
)


for i in range(
    number_of_samples
):

    actual = y_test[i]

    predicted = max(
        0,
        test_prediction[i]
    )

    print(
        f"\nSample {i + 1}"
    )

    print(
        f"Actual Stress Score    : {actual:.2f}"
    )

    print(
        f"Predicted Stress Score : {predicted:.2f}"
    )

    print(
        "Stress Level           :",
        get_stress_level(predicted)
    )


# ============================================================
# 18. RAW .SVC FILE FEATURE EXTRACTION
# ============================================================

def extract_features_from_svc(filename):

    x_values = []
    y_values = []
    timestamps = []
    pen_status = []
    pressures = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # EMOTHAW format:
            # X Y timestamp pen_status azimuth altitude pressure

            if len(parts) < 7:
                continue

            try:

                x = float(parts[0])
                y_value = float(parts[1])
                timestamp = float(parts[2])
                status = float(parts[3])
                pressure = float(parts[6])

            except ValueError:

                continue

            x_values.append(x)
            y_values.append(y_value)
            timestamps.append(timestamp)
            pen_status.append(status)
            pressures.append(pressure)


    if len(x_values) < 2:

        raise ValueError(
            "Could not extract enough handwriting data from the .svc file."
        )


    x_values = np.array(x_values)

    y_values = np.array(y_values)

    timestamps = np.array(timestamps)

    pen_status = np.array(pen_status)

    pressures = np.array(pressures)


    # --------------------------------------------------------
    # Pressure features
    # --------------------------------------------------------

    pressure_mean = np.mean(
        pressures
    )

    pressure_std = np.std(
        pressures
    )

    pressure_min = np.min(
        pressures
    )

    pressure_max = np.max(
        pressures
    )

    pressure_range = (
        pressure_max
        - pressure_min
    )


    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    dx = np.diff(
        x_values
    )

    dy = np.diff(
        y_values
    )


    distances = np.sqrt(
        dx ** 2 + dy ** 2
    )


    total_distance = np.sum(
        distances
    )


    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = (
        timestamps[-1]
        - timestamps[0]
    )


    if duration <= 0:

        duration = 1


    # --------------------------------------------------------
    # Average speed
    # --------------------------------------------------------

    average_speed = (
        total_distance
        / duration
    )


    # --------------------------------------------------------
    # Pen events
    # --------------------------------------------------------

    pen_down_count = np.sum(
        pen_status == 1
    )

    pen_up_count = np.sum(
        pen_status != 1
    )


    features = [

        pressure_mean,

        pressure_std,

        pressure_min,

        pressure_max,

        pressure_range,

        total_distance,

        duration,

        average_speed,

        pen_down_count,

        pen_up_count
    ]


    return np.array(
        features,
        dtype=float
    )


# ============================================================
# 19. PREDICT NEW HANDWRITING FILE
# ============================================================

def predict_new_file(filename):

    features = extract_features_from_svc(
        filename
    )


    features_scaled = (
        features - feature_mean
    ) / feature_std


    prediction = best_model.predict(
        features_scaled.reshape(1, -1)
    )[0]


    prediction = max(
        0,
        prediction
    )


    level = get_stress_level(
        prediction
    )


    return prediction, level, features


# ============================================================
# 20. SAVE MODEL
# ============================================================

model_information = {

    "model_name":
        best_model_name,

    "model":
        best_model,

    "feature_columns":
        FEATURE_COLUMNS,

    "feature_mean":
        feature_mean,

    "feature_std":
        feature_std,

    "test_mae":
        test_mae,

    "test_r2":
        test_r2
}


with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model_information,
        file
    )


print("\n==============================================")
print(" MODEL SAVED")
print("==============================================")

print(
    "File:",
    MODEL_FILE
)


# ============================================================
# 21. GRAPHICAL USER INTERFACE
# ============================================================

def open_handwriting_file():

    filename = filedialog.askopenfilename(

        title="Select Handwriting File",

        filetypes=[
            (
                "EMOTHAW SVC files",
                "*.svc"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )


    if not filename:

        return


    try:

        prediction, level, features = (
            predict_new_file(filename)
        )


        filename_label.config(
            text=os.path.basename(filename)
        )


        result_label.config(
            text=(
                f"Predicted Stress Score: "
                f"{prediction:.2f}\n\n"
                f"Estimated Stress Level: "
                f"{level}"
            )
        )


        feature_text.delete(
            "1.0",
            tk.END
        )


        for i, feature in enumerate(
            FEATURE_COLUMNS
        ):

            feature_text.insert(
                tk.END,
                f"{feature}: "
                f"{features[i]:.2f}\n"
            )


    except Exception as error:

        messagebox.showerror(
            "Prediction Error",
            str(error)
        )


# ============================================================
# 22. CREATE GUI
# ============================================================

root = tk.Tk()

root.title(
    "Handwriting Stress Detection"
)

root.geometry(
    "650x650"
)


title_label = tk.Label(
    root,
    text="Handwriting Stress Detection System",
    font=("Arial", 20, "bold")
)

title_label.pack(
    pady=20
)


description_label = tk.Label(
    root,
    text=(
        "ML-based estimation of stress from "
        "handwriting dynamics"
    ),
    font=("Arial", 11)
)

description_label.pack(
    pady=5
)


algorithm_label = tk.Label(
    root,
    text=(
        f"Selected ML Algorithm: "
        f"{best_model_name}"
    ),
    font=("Arial", 11, "bold")
)

algorithm_label.pack(
    pady=10
)


select_button = tk.Button(
    root,
    text="Select Handwriting File (.svc)",
    font=("Arial", 12),
    command=open_handwriting_file,
    padx=20,
    pady=10
)

select_button.pack(
    pady=15
)


filename_label = tk.Label(
    root,
    text="No file selected",
    font=("Arial", 10)
)

filename_label.pack(
    pady=5
)


result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 15, "bold"),
    justify="center"
)

result_label.pack(
    pady=25
)


feature_heading = tk.Label(
    root,
    text="Extracted Handwriting Features",
    font=("Arial", 12, "bold")
)

feature_heading.pack(
    pady=5
)


feature_text = tk.Text(
    root,
    height=12,
    width=70,
    font=("Consolas", 10)
)

feature_text.pack(
    pady=10
)


info_label = tk.Label(
    root,
    text=(
        "This system provides an ML-based estimate "
        "and is not a medical diagnosis."
    ),
    font=("Arial", 9)
)

info_label.pack(
    pady=10
)


# ============================================================
# 23. START APPLICATION
# ============================================================

print("\n==============================================")
print(" SOFTWARE READY")
print("==============================================")

print(
    "\nA graphical interface will now open."
)

root.mainloop()