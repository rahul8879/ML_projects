import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate_model(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("ROC AUC:", round(roc_auc_score(y_test, proba), 4))
    print("Avg precision:", round(average_precision_score(y_test, proba), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=4))


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model.")
    parser.add_argument(
        "--data-path",
        default="data/processed_credit_data.csv",
        help="Path to processed training data.",
    )
    parser.add_argument(
        "--model-path",
        default="models/credit_risk_model.pkl",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=50000,
        help="Number of training rows to use for hyperparameter tuning.",
    )
    parser.add_argument("--cv", type=int, default=3, help="CV folds for tuning.")
    parser.add_argument("--n-iter", type=int, default=15, help="Randomized search iterations.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    target = "is_fraud"
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 6, 12, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }

    tune_size = args.tune_sample_size
    if len(X_train) > tune_size:
        tune_idx = y_train.sample(n=tune_size, random_state=42).index
        X_tune = X_train.loc[tune_idx]
        y_tune = y_train.loc[tune_idx]
    else:
        X_tune = X_train
        y_tune = y_train

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_tune, y_tune)
    best_model = search.best_estimator_
    print("Best params:", search.best_params_)
    print("\nTest metrics:")
    evaluate_model(best_model, X_test, y_test)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
