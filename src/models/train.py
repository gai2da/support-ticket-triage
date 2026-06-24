import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = "models"


def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def train_sentiment_lr(train_df, val_df, max_features=5000, C=1.0):
    X_train = train_df["customer_text_clean"].fillna("").astype(str)
    X_val = val_df["customer_text_clean"].fillna("").astype(str)
    y_train = train_df["sentiment"]
    y_val = val_df["sentiment"]

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_param("class_weight", "balanced")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        f1_neg = f1_score(y_val, y_pred, pos_label="negative", average="binary")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("f1_negative", f1_neg)
        mlflow.sklearn.log_model(pipeline, "model")

    save_model(pipeline, "sentiment_model.pkl")
    return pipeline


def train_sentiment_sgd(train_df, val_df, max_features=5000):
    X_train = train_df["customer_text_clean"].fillna("").astype(str)
    X_val = val_df["customer_text_clean"].fillna("").astype(str)
    y_train = train_df["sentiment"]
    y_val = val_df["sentiment"]

    with mlflow.start_run(run_name="sgd_classifier"):
        mlflow.log_param("model", "SGDClassifier")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("loss", "hinge")
        mlflow.log_param("class_weight", "balanced")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", SGDClassifier(loss="hinge", class_weight="balanced", random_state=42, n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        f1_neg = f1_score(y_val, y_pred, pos_label="negative", average="binary")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("f1_negative", f1_neg)
        mlflow.sklearn.log_model(pipeline, "model")

    return pipeline


def train_response_time_ridge(train_df, val_df, max_features=5000, alpha=1.0):
    X_train = train_df["customer_text_clean"].fillna("").astype(str)
    X_val = val_df["customer_text_clean"].fillna("").astype(str)
    y_train = train_df["response_time_log"]
    y_val = val_df["response_time_log"]

    with mlflow.start_run(run_name="ridge_regression_baseline"):
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("target", "response_time_log")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("reg", Ridge(alpha=alpha))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mae_real = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred))

        mlflow.log_metric("mae_log", mae)
        mlflow.log_metric("rmse_log", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae_minutes", mae_real)
        mlflow.sklearn.log_model(pipeline, "model")

    return pipeline


def train_response_time_ridge_onehot(train_df, val_df, max_features=5000, alpha=1.0):
    y_train = train_df["response_time_log"]
    y_val = val_df["response_time_log"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    train_company = ohe.fit_transform(train_df[["company_name"]].fillna("unknown"))
    val_company = ohe.transform(val_df[["company_name"]].fillna("unknown"))

    tfidf = TfidfVectorizer(max_features=max_features)
    X_train_text = tfidf.fit_transform(train_df["customer_text_clean"].fillna("").astype(str))
    X_val_text = tfidf.transform(val_df["customer_text_clean"].fillna("").astype(str))

    train_extra = csr_matrix(train_df[["hour", "is_weekend"]].values)
    val_extra = csr_matrix(val_df[["hour", "is_weekend"]].values)

    X_train = hstack([X_train_text, train_company, train_extra])
    X_val = hstack([X_val_text, val_company, val_extra])

    with mlflow.start_run(run_name="ridge_onehot_company"):
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("features", "text + onehot_company + hour + is_weekend")
        mlflow.log_param("alpha", alpha)

        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mae_real = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred))

        mlflow.log_metric("mae_log", mae)
        mlflow.log_metric("rmse_log", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae_minutes", mae_real)

    save_model({"model": reg, "tfidf": tfidf, "ohe": ohe}, "regression_model.pkl")
    return reg, tfidf, ohe


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///notebooks/mlruns.db")
    mlflow.set_experiment("customer-support-intelligence")

    train_df, val_df = load_data("data/processed/train.csv", "data/processed/val.csv")

    train_sentiment_lr(train_df, val_df)
    train_sentiment_sgd(train_df, val_df)
    train_response_time_ridge(train_df, val_df)
    train_response_time_ridge_onehot(train_df, val_df)