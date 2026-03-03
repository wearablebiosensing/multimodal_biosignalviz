import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.impute import SimpleImputer

# ==========================================================
# 1. STRICT BINARY FILTER
# ==========================================================
def enforce_binary_labels(df):
    df = df[df["type"].isin([0, 1])]
    return df


# ==========================================================
# 2. LEAVE ONE PARTICIPANT OUT GENERATOR
# ==========================================================
def leave_one_participant_out(dataframe):
    unique_pids = dataframe["participantId"].unique()
    for pid in unique_pids:
        train_df = dataframe[dataframe["participantId"] != pid]
        test_df  = dataframe[dataframe["participantId"] == pid]
        yield train_df, test_df, pid


# ==========================================================
# 3. APPLY UNDERSAMPLING (TRAIN ONLY)
# ==========================================================
def apply_undersampling(X, y, method="Clus"):
    """
    method: 'Clus' for ClusterCentroids, 'Rus' for RandomUnderSampler, None for no undersampling
    """
    if method == "Clus":
        print("   ➤ Using ClusterCentroids undersampling")
        cc = ClusterCentroids(random_state=0)
        X_resampled, y_resampled = cc.fit_resample(X, y)
    elif method == "Rus":
        print("   ➤ Using RandomUnderSampler")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    print("   ➤ Resampled training distribution:", Counter(y_resampled))
    return X_resampled, y_resampled


# ==========================================================
# 4. METRIC COMPUTATION
# ==========================================================
def compute_fold_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (2, 2):
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        sensitivity = TP/(TP+FN) if (TP+FN) > 0 else np.nan
        specificity = TN/(TN+FP) if (TN+FP) > 0 else np.nan
    else:
        sensitivity = np.nan
        specificity = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return accuracy, balanced_acc, f1, sensitivity, specificity


# ==========================================================
# 5. DEBUG PRINT BLOCK
# ==========================================================
def print_fold_debug_info(train_df, test_df, pid):
    print("\n====================================================")
    print(f" LOPO FOLD → Leaving Out Participant: {pid}")
    print("====================================================")
    print("\n   ➤ Training Participants:", sorted(train_df["participantId"].unique()))
    print("\n   ➤ Training Class Distribution:", Counter(train_df["type"]))
    print("\n   ➤ Test Class Distribution:", Counter(test_df["type"]))
    if "activityID" in train_df.columns:
        print("\n   ➤ Training Activity Distribution:")
        print(train_df["activityID"].value_counts())
    if "activityID" in test_df.columns:
        print("\n   ➤ Test Activity Distribution:")
        print(test_df["activityID"].value_counts())
    print("\n   ➤ Training Shape:", train_df.shape)
    print("   ➤ Test Shape:", test_df.shape)
    print("----------------------------------------------------")


# ==========================================================
# 6. LOPO RANDOM FOREST WITH UNDERSAMPLING + CLASS WEIGHT
# ==========================================================
def rf_leave_one_participant_out_weighted(
        dataframe,
        undersample_method="Rus",
        save_dir=None
):

    # os.makedirs(save_dir, exist_ok=True)

    dataframe = enforce_binary_labels(dataframe)
    results = []

    # Use class_weight="balanced" to stabilize minority predictions
    pipeline = Pipeline([
        ('rf', RandomForestClassifier(random_state=8, class_weight="balanced"))
    ])

    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10]
    }

    print("\n========== STARTING LOPO RANDOM FOREST ==========")
    print("Total Participants:", dataframe["participantId"].nunique())

    for train_df, test_df, pid in leave_one_participant_out(dataframe):

        print_fold_debug_info(train_df, test_df, pid)

        # ---------------- FEATURE SELECTION ----------------
        non_feature_cols = [
            'type', 'filename', 'participantId',
            'activityID', 'BehaviorCode', 'WindowNumber', 'FeatureName'
        ]
        existing_cols = [c for c in non_feature_cols if c in train_df.columns]

        X_train = train_df.drop(columns=existing_cols)
        X_test = test_df.drop(columns=existing_cols)
        y_train = train_df["type"]
        y_test = test_df["type"]

        # ---------------- IMPUTATION ----------------
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # ---------------- UNDERSAMPLING ----------------
        X_train, y_train = apply_undersampling(X_train, y_train, undersample_method)

        # ---------------- SCALING ----------------
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------- INNER CV ----------------
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=8)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print("\n   ➤ Best Hyperparameters:", grid_search.best_params_)

        # ---------------- TESTING ----------------
        y_pred = best_model.predict(X_test)
        acc, bal_acc, f1, sen, spec = compute_fold_metrics(y_test, y_pred)

        print("\n   ➤ Fold Accuracy:", acc)
        print("   ➤ Fold Balanced Accuracy:", bal_acc)
        print("   ➤ Fold F1-score:", f1)
        print("   ➤ Fold Sensitivity:", sen)
        print("   ➤ Fold Specificity:", spec)
        print("====================================================\n")

        results.append({
            "participant": pid,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "sensitivity": sen,
            "specificity": spec
        })

    results_df = pd.DataFrame(results)

    # ======================================================
    # SUMMARY METRICS
    # ======================================================
    valid_df = results_df.dropna(subset=["sensitivity", "specificity"])
    summary = {
        "Total_Folds": len(results_df),
        "Valid_Folds": len(valid_df),
        "Micro_Accuracy_Mean": results_df["accuracy"].mean(),
        "Micro_Accuracy_STD": results_df["accuracy"].std(),
        "Macro_Balanced_Accuracy_Mean": valid_df["balanced_accuracy"].mean(),
        "Macro_Balanced_Accuracy_STD": valid_df["balanced_accuracy"].std(),
        "Macro_F1_Mean": valid_df["f1_score"].mean(),
        "Macro_F1_STD": valid_df["f1_score"].std(),
        "Macro_Sensitivity_Mean": valid_df["sensitivity"].mean(),
        "Macro_Sensitivity_STD": valid_df["sensitivity"].std(),
        "Macro_Specificity_Mean": valid_df["specificity"].mean(),
        "Macro_Specificity_STD": valid_df["specificity"].std()
    }

    # ======================================================
    # SAVE RESULTS
    # ======================================================
    # results_df.to_csv(os.path.join(save_dir, "fold_results.csv"), index=False)
    # with open(os.path.join(save_dir, "summary_metrics.json"), "w") as f:
    #     json.dump(summary, f, indent=4)

    # print("\n========== FINAL SUMMARY ==========")
    # print(json.dumps(summary, indent=4))

    return results_df, summary