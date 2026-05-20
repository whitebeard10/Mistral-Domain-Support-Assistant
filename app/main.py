import os
import time
import logging
import json
import uuid
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.config import settings

# Structured JSON Logging Setup
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "request_id": getattr(record, "request_id", "N/A")
        }
        return json.dumps(log_record)

logger = logging.getLogger("api_logger")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="Mistral Domain Support Assistant API")

# Global engine instance
engine = None

class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: int = settings.MAX_NEW_TOKENS
    temperature: float = settings.TEMPERATURE
    top_p: float = settings.TOP_P

class GenerateResponse(BaseModel):
    response: str
    latency_seconds: float
    tokens_generated: int
    model_name: str
    request_id: str
    metadata: dict

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.on_event("startup")
async def startup_event():
    global engine
    if os.getenv("TESTING"):
        logger.info("TESTING environment detected. Skipping real model loading.")
        return
        
    try:
        from src.inference.engine import InferenceEngine
        adapter_path = os.getenv("ADAPTER_PATH", None)
        logger.info(f"Initializing Inference Engine with adapter: {adapter_path}")
        engine = InferenceEngine(adapter_path=adapter_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, fast_request: Request):
    request_id = fast_request.state.request_id
    
    if engine is None:
        logger.warning("Generation attempted but model not loaded", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Processing generation request: {request.instruction[:50]}...", extra={"request_id": request_id})
    
    try:
        start_time = time.time()
        result = engine.generate(
            instruction=request.instruction,
            max_new_tokens=request.max_new_tokens
        )
        total_latency = time.time() - start_time
        
        logger.info("Generation successful", extra={
            "request_id": request_id,
            "latency": total_latency,
            "tokens": result['tokens_generated']
        })
        
        return GenerateResponse(
            response=result['response'],
            latency_seconds=round(total_latency, 3),
            tokens_generated=result['tokens_generated'],
            model_name=settings.BASE_MODEL_NAME,
            request_id=request_id,
            metadata={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": engine is not None,
        "environment": "testing" if os.getenv("TESTING") else "production"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
