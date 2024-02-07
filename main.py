from flask import Flask, request, after_this_request
from app.routes.biped import bp as api_bp
from loguru import logger
import sys
import logging

# Create a Flask application instance
app = Flask(__name__)


# Function to intercept standard logging messages
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelno, record.getMessage())


# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add terminal sink with INFO level
logger.add("logs/api_{time}.log", rotation="50 MB", level="INFO")  # Add file sink with rotation

# Replace Flask's default logger with Loguru's InterceptHandler
app.logger.addHandler(InterceptHandler())


# Middleware to log requests and responses
@app.before_request
def before_request_logging():
    logger.info(f"Request: {request.method} {request.url} - {request.remote_addr}")


@app.after_request
def after_request_logging(response):
    @after_this_request
    def log_response(response):
        logger.info(f"Response: {response.status_code} - {response.get_data(as_text=True)}")
        return response

    return log_response(response)


def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
