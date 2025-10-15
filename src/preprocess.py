

from __future__ import annotations
from typing import Tuple, List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Config

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Coerce TotalCharges to numeric, some rows are blank when tenure == 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop ID column if present
    if cfg.id_col in df.columns:
        df = df.drop(columns=[cfg.id_col])

    # Standardize target to 0-1
    if cfg.target_col in df.columns:
        df[cfg.target_col] = df[cfg.target_col].map({"No": 0, "Yes": 1}).astype("Int64")

    # Handle missing values - simple impute for numeric later via pipeline
    return df

def split_features_target(df: pd.DataFrame, cfg: Config):
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])
    return X, y

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols

def train_test_split_strat(X, y, cfg: Config):
    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
