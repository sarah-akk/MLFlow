Requirements

The dependencies are managed using requirements.txt. Ensure all the necessary packages are installed:

    mlflow
    scikit-learn
    pandas

Running the Experiment

To run the experiment and log the results to MLflow:

bash

python train.py

This script will:

    Load the Iris dataset.
    Split the data into training and testing sets.
    Train a Logistic Regression model.
    Log the accuracy, hyperparameters, and model artifacts to MLflow.
    Save detailed experiment information and sample test data as JSON and CSV files.

Experiment Details

During the experiment, the following information is logged:

    Metrics: Model accuracy
    Parameters: max_iter, solver
    Tags: User and model version
    Artifacts: Trained model, experiment details, hyperparameters, and test sample data.

Accessing the MLflow Dashboard

To view the experiment details in the MLflow UI:

    Start the MLflow server:

bash

mlflow ui

    Open the following URL in your browser:

arduino

http://localhost:5000

    Navigate to the "Iris_Model_Experiments" experiment to view all runs.

Results

The experiment achieves an accuracy of approximately X.XXXX on the test set. Detailed results and artifacts can be accessed through the MLflow dashboard.
