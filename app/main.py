import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    latency: float
    tokens_generated: int
    model_name: str

@app.on_event("startup")
async def startup_event():
    global engine
    if os.getenv("TESTING"):
        logger.info("TESTING environment detected. Skipping real model loading.")
        return
        
    from src.inference.engine import InferenceEngine
    adapter_path = os.getenv("ADAPTER_PATH", None)
    logger.info(f"Initializing Inference Engine with adapter: {adapter_path}")
    engine = InferenceEngine(adapter_path=adapter_path)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = engine.generate(
            instruction=request.instruction,
            max_new_tokens=request.max_new_tokens
        )
        
        return GenerateResponse(
            response=result['response'],
            latency=result['latency'],
            tokens_generated=result['tokens_generated'],
            model_name=settings.BASE_MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
