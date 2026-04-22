from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResult:
    model_name: str
    train_runtime: float
    train_acc: float
    valid_acc: float
    test_acc: float
    total_runtime: float
    parameters: dict[str, Any]
    seed: int
    job_id: str
    total_epochs: int