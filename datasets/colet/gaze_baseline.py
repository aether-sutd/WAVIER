import os
import sys
import numpy as np
import scipy.stats
from scipy.signal import medfilt
from itertools import groupby
import pandas as pd
from typing import Optional
from scipy.stats import skew, kurtosis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def compute_eye_features(file_path, screen_deg_x=30, screen_deg_y=17, velocity_threshold=35, min_fixation_duration=0.055):
    """
    Calculates fixation and saccade-related features from normalized gaze data.

    Parameters:
    - norm_pos_x, norm_pos_y: Arrays of normalized gaze positions [0,1]
    - timestamps: Time array in seconds
    - screen_deg_x, screen_deg_y: Visual angles of the screen (default for 24" monitor @ 60cm)
    - velocity_threshold: Threshold for I-VT classification (deg/sec)
    - min_fixation_duration: Minimum fixation duration in seconds

    Returns:
    - Dictionary with computed features
    """
    data = pd.read_csv(file_path)
    # Drop duplicate rows based on the 'gaze_timestamp' column
    data = data.drop_duplicates(subset='gaze_timestamp')
    #print(data.head())
    norm_pos_x = data['norm_pos_x'].dropna()
    norm_pos_y = data['norm_pos_y'].dropna()
    timestamps = data['gaze_timestamp'].dropna()

    # Check if timestamps are empty or have fewer than two elements
    if timestamps.empty or len(timestamps) < 2:
        print(f"Warning: Insufficient timestamps in file {file_path}. Skipping...")
        return {
            'fixation_frequency': None,
            'fixation_duration_mean': None,
            'fixation_duration_CV': None,
            'fixation_duration_skewness': None,
            'fixation_duration_kurtosis': None,
            'saccade_frequency': None,
            'saccade_duration_mean': None,
            'saccade_duration_CV': None,
            'saccade_duration_skewness': None,
            'saccade_duration_kurtosis': None,
            'saccade_velocity_mean': None,
            'saccade_velocity_CV': None,
            'saccade_velocity_skewness': None,
            'saccade_velocity_kurtosis': None
        }

    # Convert to degrees of visual angle
    deg_x = np.array(norm_pos_x) * screen_deg_x
    deg_y = np.array(norm_pos_y) * screen_deg_y

    # Time differences
    dt = np.diff(timestamps)
    if np.any(dt <= 0):  # Check for invalid or zero time differences
        print(f"Warning: Invalid time differences in file {file_path}. Skipping...")
        return {
            'fixation_frequency': None,
            'fixation_duration_mean': None,
            'fixation_duration_CV': None,
            'fixation_duration_skewness': None,
            'fixation_duration_kurtosis': None,
            'saccade_frequency': None,
            'saccade_duration_mean': None,
            'saccade_duration_CV': None,
            'saccade_duration_skewness': None,
            'saccade_duration_kurtosis': None,
            'saccade_velocity_mean': None,
            'saccade_velocity_CV': None,
            'saccade_velocity_skewness': None,
            'saccade_velocity_kurtosis': None
        }

    dx = np.diff(deg_x)
    dy = np.diff(deg_y)
    distance = np.sqrt(dx**2 + dy**2)
    velocity = distance / dt

    # Median filter to smooth velocities
    velocity_filtered = medfilt(velocity, kernel_size=5)

    # I-VT classification
    fixation_mask = velocity_filtered < velocity_threshold

    # Feature lists
    fixation_durations = []
    saccade_durations = []
    saccade_velocities = []

    start_idx = 0
    for key, group in groupby(fixation_mask):
        group_len = len(list(group))
        end_idx = start_idx + group_len
        duration = np.sum(dt[start_idx:end_idx])

        if key:  # Fixation
            if duration >= min_fixation_duration:
                fixation_durations.append(duration)
        else:  # Saccade
            if duration > 0:
                saccade_durations.append(duration)
                saccade_velocities.append(np.mean(velocity[start_idx:end_idx]))

        start_idx = end_idx

    # Total duration
    total_time = timestamps.iloc[-1] - timestamps.iloc[0]

    # Compute statistics safely
    def safe_stats(data):
        if len(data) < 2:
            return (0, 0, 0, 0)
        return (
            np.mean(data),
            np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
            scipy.stats.skew(data),
            scipy.stats.kurtosis(data)
        )

    # Fixation stats
    fix_mean, fix_cv, fix_skew, fix_kurt = safe_stats(fixation_durations)

    # Saccade stats
    sac_dur_mean, sac_dur_cv, sac_dur_skew, sac_dur_kurt = safe_stats(saccade_durations)
    sac_vel_mean, sac_vel_cv, sac_vel_skew, sac_vel_kurt = safe_stats(saccade_velocities)

    return {
        'fixation_frequency': len(fixation_durations) / total_time,
        'fixation_duration_mean': fix_mean,
        'fixation_duration_CV': fix_cv,
        'fixation_duration_skewness': fix_skew,
        'fixation_duration_kurtosis': fix_kurt,
        'saccade_frequency': len(saccade_durations) / total_time,
        'saccade_duration_mean': sac_dur_mean,
        'saccade_duration_CV': sac_dur_cv,
        'saccade_duration_skewness': sac_dur_skew,
        'saccade_duration_kurtosis': sac_dur_kurt,
        'saccade_velocity_mean': sac_vel_mean,
        'saccade_velocity_CV': sac_vel_cv,
        'saccade_velocity_skewness': sac_vel_skew,
        'saccade_velocity_kurtosis': sac_vel_kurt
    }



def calculate_task_features(task_files_dict: dict[str, dict[str, list[str]]], data_type: str) -> dict[str, dict[str, dict[str, float]]]:
    """Calculate features for each file grouped by task and participant."""
    results = {}
    for task, files in task_files_dict.items():
        print(f"\nProcessing Task {task}:")
        for file_path in files[data_type]:
            features = compute_eye_features(file_path) if data_type == 'gaze' else {}
            #print(features)
            participant_id = os.path.basename(file_path).split('_')[1]  # Extract participant ID
            if task not in results:
                results[task] = {}
            results[task][participant_id] = features
    return results


# Classification Functions
def prepare_data_for_classification(results: dict[str, dict[str, dict[str, float]]]) -> (np.ndarray, np.ndarray):
    """Prepare feature matrix X and labels y for classification."""
    X, y = [], []
    for task, participants in results.items():
        for participant_id, features in participants.items():
            if None not in features.values():  # Ensure no missing values
                X.append(list(features.values()))
                y.append(task)  # Use task as the label
    return np.array(X), np.array(y)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Plot and display the confusion matrix for a given model.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - model_name: Name of the model.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()


def perform_cross_validation(X: np.ndarray, y: np.ndarray) -> None:
    """
    Perform five-fold cross-validation for multiple models and plot the confusion matrix for the best-performing model.
    """
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

    best_model_name = None
    best_mean_accuracy = 0
    best_model = None
    best_y_pred = None

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

        # Check if this model has the highest mean accuracy
        if scores.mean() > best_mean_accuracy:
            best_mean_accuracy = scores.mean()
            best_model_name = model_name
            best_model = model
            best_y_pred = cross_val_predict(model, X, y, cv=skf)

    # Plot confusion matrix for the best-performing model
    if best_model_name and best_y_pred is not None:
        print(f"\nBest model: {best_model_name} with Mean Accuracy: {best_mean_accuracy:.4f}")
        print(f"Generating confusion matrix for {best_model_name}...")
        plot_confusion_matrix(y, best_y_pred, best_model_name)


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
    task_files_dict = get_task_files(DATA_DIR, task_list, ["gaze"])
    total_files = sum(len(files['gaze']) for files in task_files_dict.values())
    print(f"Total files found: {total_files}")

    # Step 3: Calculate features
    results = calculate_task_features(task_files_dict, "gaze")

    # Step 4: Prepare data for classification
    X, y = prepare_data_for_classification(results)
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    # Step 5: Perform cross-validation
    perform_cross_validation(X, y)


if __name__ == "__main__":
    main()