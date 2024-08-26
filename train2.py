import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json
import os

# Create experiment and set it as active
mlflow.set_experiment("Wine_Model_Experiments")

# Function to log detailed experiment information as artifacts
def log_experiment_details(run_id, dataset_name, user, source, version, model_name, description):
    details = {
        "Dataset": dataset_name,
        "User": user,
        "Source": source,
        "Version": version,
        "Model": model_name,
        "Description": description
    }
    # Log custom experiment details as an artifact (JSON file)
    details_path = f"mlruns/{run_id}/artifacts/experiment_details.json"
    os.makedirs(os.path.dirname(details_path), exist_ok=True)
    with open(details_path, "w") as f:
        json.dump(details, f, indent=4)

# Load the Wine dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Define and train the model
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 1. Log the accuracy as a metric
    mlflow.log_metric("accuracy", accuracy)

    # 2. Log parameters (e.g., hyperparameters)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("criterion", "gini")

    # 3. Log custom tags (e.g., user-defined metadata)
    mlflow.set_tag("user", "Sarah")
    mlflow.set_tag("model_version", "1.0.0")

    # 4. Log the model as an artifact
    mlflow.sklearn.log_model(model, "model_with_env")

    # 5. Log detailed experiment information as a JSON artifact
    dataset_name = "Wine"
    user = "Sarah"
    source = "train_wine.py"
    version = "1.0.0"
    model_name = "Decision Tree Classifier"
    description = "Model trained on the Wine dataset to classify wine types."
    log_experiment_details(run_id, dataset_name, user, source, version, model_name, description)

    # 6. Log hyperparameters as a JSON artifact (useful for complex parameter sets)
    hyperparameters = {
        "max_depth": 3,
        "criterion": "gini",
        "splitter": "best"
    }
    hyperparams_path = f"mlruns/{run_id}/artifacts/hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    mlflow.log_artifact(hyperparams_path)

    # 7. Track system information (automatically done by MLflow)

    # 8. Log an artifact that captures some data visualization or report
    # For example, saving a sample of the test set as a CSV file and logging it
    test_sample_path = f"mlruns/{run_id}/artifacts/test_sample.csv"
    import pandas as pd
    test_sample_df = pd.DataFrame(X_test, columns=wine.feature_names)
    test_sample_df['true_label'] = y_test
    test_sample_df['predicted_label'] = predictions
    test_sample_df.to_csv(test_sample_path, index=False)
    mlflow.log_artifact(test_sample_path)

    # 9. Log the training script itself as an artifact
    script_path = "train2.py"
    mlflow.log_artifact(script_path)

    # Print accuracy
    print(f"Model accuracy: {accuracy:.4f}")

    # Print the MLflow run ID for reference
    print(f"Run ID: {run_id}")
