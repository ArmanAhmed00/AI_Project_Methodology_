# Bonus – cloud deployment image (Render / Azure Container Apps / GCP Cloud Run / AWS App Runner)
# Build:  docker build -t churn-api .
# Run:    docker run -p 8080:8080 churn-api
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/serving_model/ model/
ENV PYTHONPATH=/app/src PORT=8080
EXPOSE 8080
CMD mlflow models serve -m /app/model --env-manager local -h 0.0.0.0 -p ${PORT}
