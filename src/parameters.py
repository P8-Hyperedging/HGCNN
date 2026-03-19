from dataclasses import dataclass, asdict
from enum import Enum

class InputType(str, Enum):
    RANGE = "range"
    INPUT = "input"
    TOGGLE = "toggle"
    Select = "select"

@dataclass
class BaseParameter:
    name: str
    type: InputType

@dataclass
class RangeParameter(BaseParameter):
    min: float
    max: float
    step: float
    default: float

@dataclass
class InputParameter(BaseParameter):
    min: float
    max: float
    default: float

@dataclass
class ToggleParameter(BaseParameter):
    default: bool

@dataclass
class SelectParameter(BaseParameter):
    options: list[str]

Parameter = RangeParameter | InputParameter| ToggleParameter | SelectParameter

def get_parameters() -> list[Parameter]:
    return [
        RangeParameter(name="param1", min=1, max=5, step=0.5, default=2, type=InputType.RANGE),
        RangeParameter(name="param2", min=1, max=5, step=0.5, default=3, type=InputType.RANGE),
        InputParameter(name="param3", min=1, max=500, default=200, type=InputType.INPUT),
        ToggleParameter(name="param4", default=True, type=InputType.TOGGLE),
    ]

def get_allset_parameters() -> list[Parameter]:
    return [
        InputParameter(name="Epoch", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="Learning rate", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="Hidden layers", min=0, max=256, default=64, type=InputType.INPUT),
        InputParameter(name="Train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="Validation proportion", min=0, max=1, default=0.25, type=InputType.INPUT),
        InputParameter(name="Dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="Weight decay", min=0, max=1, default=0, type=InputType.INPUT),
    ]

def get_moonlab_parameters() -> list[Parameter]:
    return [
        InputParameter(name="Epoch", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="Learning rate", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="Hidden layers size", min=1, max=4096, default=128, type=InputType.INPUT),
        InputParameter(name="Train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="Validation proportion", min=0, max=1, default=0.25, type=InputType.INPUT),
        InputParameter(name="Dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="Weight decay", min=0, max=1, default=0.0005, type=InputType.INPUT),
        InputParameter(name="Gamma", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="Learning milestone", min=0, max=200, default=50, type=InputType.INPUT),
    ]

def get_qhgnn_parameters() -> list[Parameter]:
    return [
        InputParameter(name="Epoch", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="Learning rate", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="Hidden layers size", min=1, max=4096, default=128, type=InputType.INPUT),
        InputParameter(name="Train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="Validation proportion", min=0, max=1, default=0.25, type=InputType.INPUT),
        InputParameter(name="Dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="Weight decay", min=0, max=1, default=0.0005, type=InputType.INPUT),
        InputParameter(name="Gamma", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="Learning milestone", min=0, max=200, default=50, type=InputType.INPUT),
    ]


# Serialize to JSON-ready dict
def serialize(params: list[Parameter]):
    return [asdict(p) for p in params]
