import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# MLFLOW SETUP
# ============================================================

# Store MLflow metadata in a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create/use this experiment
mlflow.set_experiment("RandomForest_Iris_Experiment")


# ============================================================
# LOAD DATASET
# ============================================================

data = load_iris()

X = data.data
y = data.target


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

n_estimators = 100
max_depth = 5


# ============================================================
# START MLFLOW RUN
# ============================================================

with mlflow.start_run() as run:

    print(f"Run ID: {run.info.run_id}")

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------------

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "test_size",
        0.30
    )

    mlflow.log_param(
        "random_state",
        42
    )


    # --------------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # LOG METRICS
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # --------------------------------------------------------
    # LOG + REGISTER MODEL
    # --------------------------------------------------------
    #
    # RandomForest uses sklearn.tree._tree.Tree internally.
    #
    # MLflow / skops now requires us to explicitly trust this
    # type before serializing the model.
    #
    # This is appropriate here because WE created and trained
    # the model ourselves.
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=model,

        # "name" replaces the deprecated "artifact_path"
        name="random_forest_model",

        # Register model in MLflow Model Registry
        registered_model_name="RandomForestClassifierModel",

        # Explicitly trust RandomForest's underlying Tree type
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # --------------------------------------------------------
    # OUTPUT RESULTS
    # --------------------------------------------------------

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print(f"Run ID:       {run.info.run_id}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Estimators:   {n_estimators}")
    print(f"Max Depth:    {max_depth}")
    print("========================================")
    print()
    print("Model logged and registered successfully.")