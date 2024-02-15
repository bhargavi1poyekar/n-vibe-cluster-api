from werkzeug.exceptions import BadRequest

from app.models.data_models import RSSISignals, ClusterPrediction
from app.utils.data_processing import aggregate_rssi_signals
from app.utils.model import load_model
from flask import Blueprint, request, jsonify
from loguru import logger
import numpy as np
from pydantic import ValidationError

bp = Blueprint("api", __name__, url_prefix="/api")

import logging

def change_order(data_to_send: dict):
    new_data = {'tab': []}
    for tab in data_to_send["tab"]:
        new_tab = [tab[9], tab[8], tab[7], tab[4], tab[3], tab[1], tab[0], tab[2], tab[6], tab[5]]
        new_data['tab'].append(new_tab)

    return new_data

def predict_hierarchical_coords(rss_data, models_dict):

    rss_data = np.reshape(rss_data, (1, -1))
    cluster_prediction = models_dict["cluster"].predict(rss_data)
    position_prediction = models_dict[cluster_prediction[0]]["coord"].predict(rss_data)

    position_prediction = [p * 10e-8 for p in position_prediction]

    return ClusterPrediction(cluster=cluster_prediction[0] - 1, position=position_prediction[0])


@bp.before_request
def log_request_info():
    logger.info(
        f"Request: {request.method} {request.url} - {request.remote_addr} - Input data: {request.get_json()}"
    )


@bp.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code} - {response.get_data(as_text=True)}")
    return response


@bp.route("/biped_hq", methods=["POST"])
def biped_hq():
    try:
        data = request.get_json()
        if not data or "tab" not in data:
            raise BadRequest("Missing or invalid 'tab' in JSON payload.")

        data = change_order(data)
        data = [RSSISignals(signal=t) for t in data["tab"]]

        input_signal = aggregate_rssi_signals(data)

        model = load_model("biped_hq")

        prediction = predict_hierarchical_coords(input_signal.signal, model)

        return prediction.model_dump_json()

    except ValidationError as e:
        errors = e.errors()
        return jsonify(errors=errors), 400

    except BadRequest as e:
        return jsonify(error=str(e)), 400

    except Exception as e:
        return jsonify(error=f"An unexpected error occurred. {e}"), 500
