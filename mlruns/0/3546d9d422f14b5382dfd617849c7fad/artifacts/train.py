import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Function to log experiment details
def log_experiment(name, description, version, code_link):
    mlflow.set_tag("model_name", name)
    mlflow.set_tag("dataset_name", "Iris")
    mlflow.set_tag("version", version)
    mlflow.set_tag("description", description)
    mlflow.set_tag("code_link", code_link)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run() as run:
    # Define and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log the accuracy as a metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log experiment details
    model_name = "Logistic Regression"
    description = "Model trained on the Iris dataset to classify species."
    version = "1.0.0"
    code_link = "https://github.com/your-repository/your-code"  # Replace with your actual code link
    log_experiment(model_name, description, version, code_link)

    # Print accuracy
    print(f"Model accuracy: {accuracy:.4f}")

    # Print the MLflow run ID for reference
    print(f"Run ID: {run.info.run_id}")

# Additional step to track the source code file (optional)
# Save this script or any relevant files as artifacts
if not os.path.exists("scripts"):
    os.makedirs("scripts")
with open("scripts/train.py", "w") as f:
    f.write(open(__file__).read())

mlflow.log_artifacts("scripts")
