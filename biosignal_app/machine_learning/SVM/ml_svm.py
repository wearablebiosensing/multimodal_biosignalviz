import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler


# ==========================================================
# 1. STRICT BINARY FILTER
# ==========================================================
def enforce_binary_labels(df):
    return df[df["type"].isin([0, 1])]


# ==========================================================
# 2. LEAVE ONE PARTICIPANT OUT GENERATOR
# ==========================================================
def leave_one_participant_out(dataframe):
    for pid in dataframe["participantId"].unique():
        train_df = dataframe[dataframe["participantId"] != pid]
        test_df  = dataframe[dataframe["participantId"] == pid]
        yield train_df, test_df, pid


# ==========================================================
# 3. APPLY UNDERSAMPLING
# ==========================================================
def apply_undersampling(X, y, method="Clus"):

    if method == "Clus":
        print("   ➤ Using ClusterCentroids undersampling")
        sampler = ClusterCentroids(random_state=0)
    elif method == "Rus":
        print("   ➤ Using RandomUnderSampler")
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print("   ➤ Resampled distribution:", Counter(y_resampled))
    return X_resampled, y_resampled


# ==========================================================
# 4. METRICS
# ==========================================================
def compute_fold_metrics(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (2, 2):
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]

        sensitivity = TP/(TP+FN) if (TP+FN)>0 else np.nan
        specificity = TN/(TN+FP) if (TN+FP)>0 else np.nan
    else:
        sensitivity = np.nan
        specificity = np.nan

    return (
        accuracy_score(y_test, y_pred),
        balanced_accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, zero_division=0),
        sensitivity,
        specificity
    )


# ==========================================================
# 5. LOPO SVM
# ==========================================================
def svm_leave_one_participant_out_weighted(
        dataframe,
        undersample_method="Clus",
        save_dir="SVM_LOPO_RESULTS"
):

    os.makedirs(save_dir, exist_ok=True)
    dataframe = enforce_binary_labels(dataframe)
    results = []

    pipeline = Pipeline([
        ('svm', SVC(class_weight="balanced", random_state=8))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear'],
        'svm__gamma': ['scale', 'auto']
    }

    print("\n========== STARTING LOPO SVM ==========")

    for train_df, test_df, pid in leave_one_participant_out(dataframe):

        print(f"\n--- Leaving Out Participant: {pid} ---")

        non_feature_cols = [
            'type','filename','participantId',
            'activityID','BehaviorCode','WindowNumber','FeatureName'
        ]

        existing_cols = [c for c in non_feature_cols if c in train_df.columns]

        X_train = train_df.drop(columns=existing_cols)
        X_test  = test_df.drop(columns=existing_cols)
        y_train = train_df["type"]
        y_test  = test_df["type"]

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        # Undersample
        X_train, y_train = apply_undersampling(X_train, y_train, undersample_method)

        # Scale
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        # Inner CV
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=8)

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring="balanced_accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        print("   ➤ Best Params:", grid.best_params_)

        y_pred = grid.best_estimator_.predict(X_test)

        acc, bal_acc, f1, sen, spec = compute_fold_metrics(y_test, y_pred)

        print("   ➤ Accuracy:", acc)
        print("   ➤ Balanced Accuracy:", bal_acc)
        print("   ➤ F1:", f1)
        print("   ➤ Sensitivity:", sen)
        print("   ➤ Specificity:", spec)

        results.append({
            "participant": pid,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "sensitivity": sen,
            "specificity": spec
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(save_dir, "fold_results.csv"), index=False)

    summary = results_df.mean(numeric_only=True).to_dict()

    with open(os.path.join(save_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n========== FINAL SUMMARY ==========")
    print(summary)

    return results_df, summary