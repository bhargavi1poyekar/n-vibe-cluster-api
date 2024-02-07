from flask import Blueprint, request
from app.models.data_models import RSSISignals
from app.models.model import load_model
from loguru import logger

bp = Blueprint("api", __name__, url_prefix="/api")


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
    signals = RSSISignals(**data)
    model = load_model("path_to_your_model")
    prediction = model.predict(signals.signals)
    return {"prediction": prediction}
