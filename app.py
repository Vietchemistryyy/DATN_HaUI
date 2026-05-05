from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import torch
import os

from api.config import HOST, PORT, GROQ_API_KEY, GEMINI_API_KEY
from api.routes import router as api_router
from src.model import load_trained_model
from src.utils import get_device
from src.llm_verifier import LLMVerifier
from src.calibration import TemperatureScaling

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and initializing verification pipeline...")
    
    # 1. Device
    device = get_device()
    app.state.device = device
    
    # 2. Load fine-tuned RoBERTa
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models", "roberta")
    
    try:
        model, tokenizer = load_trained_model(model_dir)
        model = model.to(device)
        model.eval()
        app.state.model = model
        app.state.tokenizer = tokenizer
        
        # Load Temperature Scaler
        temp_scaler_path = os.path.join(model_dir, "temp_scaler.pt")
        if os.path.exists(temp_scaler_path):
            temp_scaler = TemperatureScaling()
            temp_scaler.load_state_dict(torch.load(temp_scaler_path, map_location=device))
            temp_scaler.to(device)
            temp_scaler.eval()
            app.state.temp_scaler = temp_scaler
            print(f"Temperature scaler loaded successfully: T={temp_scaler.temperature.item():.4f}")
        else:
            app.state.temp_scaler = None
            print("No temp_scaler found. Running without calibration.")
            
        print("RoBERTa model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        app.state.model = None
        app.state.tokenizer = None
        app.state.temp_scaler = None
        
    # 3. LLM Verifier
    app.state.llm_verifier = LLMVerifier(
        groq_api_key=GROQ_API_KEY,
        gemini_api_key=GEMINI_API_KEY
    )
    if app.state.llm_verifier.is_available:
        print("LLM Verifier enabled (Groq/Gemini).")
    else:
        print("No LLM keys found. Running pure RoBERTa mode.")
        
    yield
    print("Shutting down...")

app = FastAPI(title="Fake News Detection System", lifespan=lifespan)

# API Router
app.include_router(api_router, prefix="/api")

# Static files and frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
