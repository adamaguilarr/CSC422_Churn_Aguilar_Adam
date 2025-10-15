from __future__ import annotations
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_models() -> Dict[str, object]:
    models = {
        "log_reg": LogisticRegression(max_iter=200, n_jobs=None),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "knn": KNeighborsClassifier(n_neighbors=12)
    }
    return models
