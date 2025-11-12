from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
import json
import sys

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class JsonFormatter(logging.Formatter):
    """Custom JSON log formatter compatible with Cloud Logging."""
    def format(self, record):
        log_record = {
            "severity": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
        }

        # Handle structured message (dict)
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        else:
            log_record["message"] = record.getMessage()

        return json.dumps(log_record)


logger = logging.getLogger("mlops-assignment-service")
logger.setLevel(logging.INFO)
info_handler = logging.StreamHandler(sys.stdout)
error_handler = logging.StreamHandler(sys.stderr)

info_handler.addFilter(lambda record: record.levelno < logging.ERROR)
error_handler.addFilter(lambda record: record.levelno >= logging.ERROR)

formatter = JsonFormatter()
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)


# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="IRIS Model API")

# Load trained model
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "Iris Prediction API is running!"}

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    import time
    time.sleep(2)  # simulate work, normally this would be model loading
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(features: IrisFeatures, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            # Convert input to numpy array for the model
            data = np.array([
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]).reshape(1, -1)

            # Run prediction using your trained model
            prediction = model.predict(data)[0]

            latency = round((time.time() - start_time) * 1000, 2)

            # Log structured info about the prediction
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": features.dict(),
                "prediction": prediction,
                "latency_ms": latency,
                "status": "success"
            }))

            return {
                "prediction": prediction,
                "trace_id": trace_id,
                "latency_ms": latency
            }

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail=f"Prediction failed. Trace ID: {trace_id}")

