import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler


def enforce_binary_labels(df):
    return df[df["type"].isin([0,1])]


def leave_one_participant_out(dataframe):
    for pid in dataframe["participantId"].unique():
        yield (
            dataframe[dataframe["participantId"] != pid],
            dataframe[dataframe["participantId"] == pid],
            pid
        )


def apply_undersampling(X,y,method="Clus"):
    if method=="Clus":
        sampler=ClusterCentroids(random_state=0)
    elif method=="Rus":
        sampler=RandomUnderSampler(random_state=42)
    else:
        return X,y

    X_res,y_res=sampler.fit_resample(X,y)
    print("   ➤ Resampled:",Counter(y_res))
    return X_res,y_res


def compute_fold_metrics(y_test,y_pred):

    cm=confusion_matrix(y_test,y_pred)

    if cm.shape==(2,2):
        TP=cm[1,1]; TN=cm[0,0]
        FP=cm[0,1]; FN=cm[1,0]
        sen=TP/(TP+FN) if (TP+FN)>0 else np.nan
        spec=TN/(TN+FP) if (TN+FP)>0 else np.nan
    else:
        sen=np.nan; spec=np.nan

    return (
        accuracy_score(y_test,y_pred),
        balanced_accuracy_score(y_test,y_pred),
        f1_score(y_test,y_pred,zero_division=0),
        sen,
        spec
    )


def gb_leave_one_participant_out(
        dataframe,
        undersample_method="Clus",
        save_dir="GB_LOPO_RESULTS"
):

    os.makedirs(save_dir,exist_ok=True)
    dataframe=enforce_binary_labels(dataframe)
    results=[]

    pipeline=Pipeline([
        ('gb',GradientBoostingClassifier(random_state=8))
    ])

    param_grid={
        'gb__n_estimators':[100,200],
        'gb__learning_rate':[0.01,0.1],
        'gb__max_depth':[3,5]
    }

    print("\n========== STARTING LOPO GRADIENT BOOST ==========")

    for train_df,test_df,pid in leave_one_participant_out(dataframe):

        drop_cols=['type','filename','participantId',
                   'activityID','BehaviorCode','WindowNumber','FeatureName']

        existing=[c for c in drop_cols if c in train_df.columns]

        X_train=train_df.drop(columns=existing)
        X_test=test_df.drop(columns=existing)
        y_train=train_df["type"]
        y_test=test_df["type"]

        imputer=SimpleImputer(strategy="median")
        X_train=imputer.fit_transform(X_train)
        X_test=imputer.transform(X_test)

        X_train,y_train=apply_undersampling(X_train,y_train,undersample_method)

        scaler=MinMaxScaler().fit(X_train)
        X_train=scaler.transform(X_train)
        X_test=scaler.transform(X_test)

        skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=8)

        grid=GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring="balanced_accuracy",
            n_jobs=-1
        )

        grid.fit(X_train,y_train)

        y_pred=grid.best_estimator_.predict(X_test)

        acc,bal_acc,f1,sen,spec=compute_fold_metrics(y_test,y_pred)

        results.append({
            "participant":pid,
            "accuracy":acc,
            "balanced_accuracy":bal_acc,
            "f1_score":f1,
            "sensitivity":sen,
            "specificity":spec
        })

    results_df=pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir,"fold_results.csv"),index=False)

    summary=results_df.mean(numeric_only=True).to_dict()

    with open(os.path.join(save_dir,"summary_metrics.json"),"w") as f:
        json.dump(summary,f,indent=4)

    return results_df,summary