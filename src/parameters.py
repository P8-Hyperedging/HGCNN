from dataclasses import dataclass, asdict
from enum import Enum
from typing import Union, List

class InputType(str, Enum):
    RANGE = "range"
    INPUT = "input"
    TOGGLE = "toggle"

@dataclass
class BaseParameter:
    name: str
    type: InputType

@dataclass
class RangeParameter(BaseParameter):
    min: float
    max: float
    default: float

@dataclass
class InputParameter(BaseParameter):
    min: float
    max: float
    default: float

@dataclass
class ToggleParameter(BaseParameter):
    default: bool

Parameter = Union[RangeParameter, InputParameter, ToggleParameter]

def get_parameters() -> List[Parameter]:
    return [
        RangeParameter(name="param1", min=1, max=5, default=2, type=InputType.RANGE),
        RangeParameter(name="param2", min=1, max=5, default=3, type=InputType.RANGE),
        InputParameter(name="param3", min=1, max=500, default=200, type=InputType.INPUT),
        ToggleParameter(name="param4", default=True, type=InputType.TOGGLE),
    ]

# Serialize to JSON-ready dict
def serialize(params: List[Parameter]):
    return [asdict(p) for p in params]
