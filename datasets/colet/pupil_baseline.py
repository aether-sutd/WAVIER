import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import skew, kurtosis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Add the path to the eda_patch_extraction module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eda_patch_extraction")))

from file_operations import get_task_files
from config import DATA_DIR


# Utility Functions
def display_options(title: str, options: dict[str, str]) -> None:
    """Display options from a dictionary."""
    print(f"\n{title}")
    for key, description in options.items():
        print(f"{key}: {description}")


def get_user_input(
    prompt: str, valid_choices: Optional[list[str]] = None, allow_multiple: bool = True, default: Optional[str] = None
) -> list[str]:
    """Handle user input with validation and optional default values."""
    while True:
        user_input = input(prompt).strip()
        if not user_input and default:
            return [default] if isinstance(default, str) else default
        choices = [c.strip() for c in user_input.split(",") if c.strip().isdigit()]
        if valid_choices and not all(c in valid_choices for c in choices):
            print("❌ Invalid input! Please enter valid choices.")
        else:
            return choices if allow_multiple else [choices[0]]


# Feature Calculation Functions
def calculate_pupil_features(file_path: str) -> dict[str, float]:
    """Calculate statistical features from the diameter_3d column in the given file."""
    data = pd.read_csv(file_path)
    if 'diameter_3d' in data.columns:
        diameter_data = data['diameter_3d'].dropna()
        return {
            "mean": diameter_data.mean(),
            "variation": diameter_data.var(),
            "skewness": skew(diameter_data),
            "kurtosis": kurtosis(diameter_data)
        }
    else:
        return {"mean": None, "variation": None, "skewness": None, "kurtosis": None}


def calculate_task_features(task_files_dict: dict[str, dict[str, list[str]]], data_type: str) -> dict[str, dict[str, dict[str, float]]]:
    """Calculate features for each file grouped by task and participant."""
    results = {}
    for task, files in task_files_dict.items():
        print(f"\nProcessing Task {task}:")
        for file_path in files[data_type]:
            features = calculate_pupil_features(file_path) if data_type == 'pupil' else {}
            participant_id = os.path.basename(file_path).split('_')[1]  # Extract participant ID
            if task not in results:
                results[task] = {}
            results[task][participant_id] = features
    return results


# Classification Functions
def prepare_data_for_classification(results: dict[str, dict[str, dict[str, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix X and labels y for classification."""
    X, y = [], []
    for task, participants in results.items():
        for participant_id, features in participants.items():
            if None not in features.values():  # Ensure no missing values
                X.append(list(features.values()))
                y.append(task)  # Use task as the label
    return np.array(X), np.array(y)


def perform_cross_validation(X: np.ndarray, y: np.ndarray) -> None:
    """Perform five-fold cross-validation for multiple models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    print("\nCross-Validation Results:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")


# Main Workflow
def main() -> None:
    """Main function to execute the workflow."""
    # Step 1: Display task options and get user input
    tasks_options = {
        "1": "Activity 1: No time pressure, single-tasking.",
        "2": "Activity 2: Time pressure, single-tasking.",
        "3": "Activity 3: No time pressure, multi-tasking.",
        "4": "Activity 4: Time pressure, multi-tasking."
    }
    display_options("📌 Available Tasks in COLET Dataset:", tasks_options)
    task_list = get_user_input("\nEnter task numbers (e.g., 1,2,4): ", valid_choices=list(tasks_options.keys()))
    print(f"Selected tasks: {task_list}")

    # Step 2: Get task files
    task_files_dict = get_task_files(DATA_DIR, task_list, ["pupil"])
    total_files = sum(len(files['pupil']) for files in task_files_dict.values())
    print(f"Total files found: {total_files}")

    # Step 3: Calculate features
    results = calculate_task_features(task_files_dict, "pupil")

    # Step 4: Prepare data for classification
    X, y = prepare_data_for_classification(results)
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    # Step 5: Perform cross-validation
    perform_cross_validation(X, y)


if __name__ == "__main__":
    main()