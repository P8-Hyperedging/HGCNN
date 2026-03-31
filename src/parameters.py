from dataclasses import dataclass, asdict
from enum import Enum

class InputType(str, Enum):
    RANGE = "range"
    INPUT = "input"
    TOGGLE = "toggle"
    Select = "select"
    Text = "text"

@dataclass
class BaseParameter:
    name: str
    type: InputType

@dataclass
class RangeParameter(BaseParameter):
    min: float | None
    max: float
    step: float
    default: float

@dataclass
class InputParameter(BaseParameter):
    min: float
    max: float
    default: float

@dataclass
class TextInputParameter(BaseParameter):
    placeholder: str

@dataclass
class ToggleParameter(BaseParameter):
    default: bool

@dataclass
class SelectParameter(BaseParameter):
    options: list[str]

Parameter = RangeParameter | InputParameter| ToggleParameter | SelectParameter | TextInputParameter

def get_allset_parameters() -> list[Parameter]:
    return [
        InputParameter(name="num epochs", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="lr", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="hidden layer size", min=0, max=256, default=64, type=InputType.INPUT),
        InputParameter(name="train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="valid proportion", min=0, max=1, default=0.25, type=InputType.INPUT),
        InputParameter(name="dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="weight decay", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="seed", min=0, max=2**32-1, default=None, type=InputType.INPUT),
    ]


def get_moonlab_parameters() -> list[Parameter]:
    return [
        InputParameter(name="num epochs", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="lr", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="hidden layer size", min=1, max=4096, default=128, type=InputType.INPUT),
        InputParameter(name="train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="weight decay", min=0, max=1, default=0.0005, type=InputType.INPUT),
        InputParameter(name="gamma", min=0, max=1, default=0.5, type=InputType.INPUT),
        TextInputParameter(name="milestones input", placeholder="50,100", type=InputType.Text),
        InputParameter(name="seed", min=0, max=2**32-1, default=None, type=InputType.INPUT),
    ]


def get_qhgnn_parameters() -> list[Parameter]:
    return [
        InputParameter(name="num epochs", min=0, max=10000, default=500, type=InputType.INPUT),
        InputParameter(name="lr", min=0, max=1, default=0.001, type=InputType.INPUT),
        InputParameter(name="hidden layer size", min=1, max=4096, default=128, type=InputType.INPUT),
        InputParameter(name="train proportion", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="dropout", min=0, max=1, default=0, type=InputType.INPUT),
        InputParameter(name="weight decay", min=0, max=1, default=0.0005, type=InputType.INPUT),
        InputParameter(name="gamma", min=0, max=1, default=0.5, type=InputType.INPUT),
        InputParameter(name="milestones input", min=0, max=200, default=50, type=InputType.INPUT),
        InputParameter(name="seed", min=0, max=2**32-1, default=None, type=InputType.INPUT),
    ]


# Serialize to JSON-ready dict
def serialize(params: list[Parameter]):
    return [asdict(p) for p in params]
