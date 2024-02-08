from pydantic import BaseModel, conlist, confloat


class RSSISignals(BaseModel):
    signal: conlist(confloat(ge=-100, le=0), min_length=10, max_length=10)
