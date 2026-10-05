import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load wine dataset
wine=load_wine()
X=wine.data
y=wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define params for RF Model
max_depth = 10
n_estimators=5

#Setting tracking URI to local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("YT-MLOPs-Exp1")

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    ## Creating confusion matrix
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save confusion matrix plot
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix plot as artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    ## Tags
    mlflow.set_tags({"Author": "Anjali", "Project Name": "Wine Classification"})

    print(f"Accuracy: {accuracy}")