FROM python:3.8-slim
WORKDIR /app
#RUN apt update && apt-get install -y netcat
RUN pip install gunicorn Flask joblib pandas numpy==1.23.3 scipy scikit-learn==0.22.1 DateTime loguru
#RUN pip --no-cache-dir install torch
#RUN pip install torchvision

ADD app.py /app
ADD /model /app/model

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
