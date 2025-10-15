from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    target_col: str = "Churn"
    id_col: str = "customerID"
    output_dir: Path = Path("results")
