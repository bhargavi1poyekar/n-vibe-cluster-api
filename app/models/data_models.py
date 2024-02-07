from pydantic import BaseModel, conlist, confloat


class RSSISignals(BaseModel):
    signals: conlist(confloat(ge=-100, le=0), min_items=10, max_items=10)
