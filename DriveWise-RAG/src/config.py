import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically find project root and load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

# Centralized Model Configuration
# Primary models default to 'gemini-flash-latest' for production-grade stability
INGESTION_VISION_MODEL = os.environ.get("INGESTION_VISION_MODEL", "gemini-flash-latest")
QUERY_UNDERSTANDING_MODEL = os.environ.get("QUERY_UNDERSTANDING_MODEL", "gemini-flash-latest")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-flash-latest")
EVALUATION_MODEL = os.environ.get("EVALUATION_MODEL", "gemini-flash-latest")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
