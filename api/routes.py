from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, HttpUrl
from typing import Optional
import os

from api.config import (
    ENABLE_GROQ, GROQ_API_KEY, 
    ENABLE_GEMINI, GEMINI_API_KEY,
    CONFIDENCE_THRESHOLD, MC_DROPOUT_RUNS,
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
)
from api.scraper import scrape_url
from src.utils import clean_text
from src.llm_verifier import hybrid_predict

router = APIRouter()

class PredictRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

@router.get("/health")
async def health_check(request: Request):
    app_state = request.app.state
    return {
        "status": "ok" if app_state.model else "error",
        "model_loaded": app_state.model is not None,
        "groq_enabled": ENABLE_GROQ and bool(GROQ_API_KEY),
        "gemini_enabled": ENABLE_GEMINI and bool(GEMINI_API_KEY)
    }

@router.post("/predict")
async def predict_news(payload: PredictRequest, request: Request):
    app_state = request.app.state
    
    if not app_state.model or not app_state.tokenizer:
        raise HTTPException(status_code=503, detail="Model currently not loaded")
        
    text_to_analyze = ""
    
    # Text Extraction
    if payload.url:
        scrape_result = scrape_url(payload.url)
        if not scrape_result.get("success"):
            raise HTTPException(status_code=400, detail=scrape_result.get("error", "Failed to extract article from URL"))
        text_to_analyze = scrape_result.get("text", "")
    elif payload.text:
        text_to_analyze = payload.text
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'url'")

    cleaned_text = clean_text(text_to_analyze)
    
    if len(cleaned_text) < MIN_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text too short (min {MIN_TEXT_LENGTH} chars)")
        
    if len(cleaned_text) > MAX_TEXT_LENGTH:
        cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]
        
    try:
        from src.model import mc_dropout_predict
        
        result = hybrid_predict(
            model=app_state.model,
            tokenizer=app_state.tokenizer,
            text=cleaned_text,
            device=app_state.device,
            mc_dropout_fn=mc_dropout_predict,
            temp_scaler=app_state.temp_scaler,
            llm_verifier=app_state.llm_verifier,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            n_mc_runs=MC_DROPOUT_RUNS
        )
        
        # Ensure proper JSON types
        for key in ["probabilities"]:
            if key in result:
                result[key] = {k: float(v) for k, v in result[key].items()}
        result["confidence"] = float(result["confidence"])
        result["uncertainty"] = float(result["uncertainty"])
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
