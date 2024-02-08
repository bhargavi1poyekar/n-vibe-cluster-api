import random
from typing import Any, Dict, List, Tuple

from app.models.data_models import RSSISignals
from app.utils.data_processing import aggregate_rssi_signals
from app.utils.model import load_model
from flask import Blueprint, request
from loguru import logger
import numpy as np
from pydantic import BaseModel

bp = Blueprint("api", __name__, url_prefix="/api")


class ClusterPrediction(BaseModel):
    floor: int = 0
    cluster: int
    position: tuple[float, float]


def predict_hierarchical_coords(rss_data, models_dict):

    rss_data = np.reshape(rss_data, (1, -1))
    cluster_prediction = models_dict["cluster"].predict(rss_data)
    position_prediction = models_dict[cluster_prediction[0]]["coord"].predict(rss_data)

    position_prediction = [p * 10e-8 for p in position_prediction]

    return ClusterPrediction(cluster=cluster_prediction[0], position=position_prediction[0])


@bp.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url} - {request.remote_addr}")


@bp.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code} - {response.get_data(as_text=True)}")
    return response


@bp.route("/biped_hq", methods=["POST"])
def biped_hq():
    data = request.get_json()
    data = [RSSISignals(signal=t) for t in data["tab"]]
    input_signal = aggregate_rssi_signals(data)
    model = load_model("biped_hq")
    prediction = predict_hierarchical_coords(input_signal.signal, model)
    return prediction.model_dump_json()
