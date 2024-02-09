from pydantic import BaseModel, conlist, confloat, field_validator
from typing import Union, List


class RSSISignals(BaseModel):
    signal: conlist(confloat(ge=-100, le=0), min_length=10, max_length=10)

    @field_validator("signal")
    def check_rssi_signal(cls, v: List[Union[int, float]]):
        return [float(value) if isinstance(value, int) else value for value in v]


class ClusterPrediction(BaseModel):
    floor: int = 0
    cluster: int
    position: tuple[float, float]
