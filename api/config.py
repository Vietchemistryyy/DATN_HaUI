import os
from dotenv import load_dotenv

load_dotenv()

# Server Config
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# LLM Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENABLE_GROQ = os.getenv("ENABLE_GROQ", "true").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"

# Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MC_DROPOUT_RUNS = int(os.getenv("MC_DROPOUT_RUNS", "10"))

# Validation
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000
