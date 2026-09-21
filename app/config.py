import os
from dotenv import load_dotenv

# Load values from .env
load_dotenv()

# ---------------------------
# API KEY
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to the .env file."
    )

if not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError(
        "OPENAI_API_KEY does not look valid. "
        "Check the value inside .env."
    )

# ---------------------------
# MODEL CONFIGURATION
# ---------------------------

RETRIEVAL_MODEL = "gpt-5-mini"
REASONING_MODEL = "gpt-5.5"
GENERATION_MODEL = "gpt-5-mini"
ORCHESTRATOR_MODEL = "gpt-5.5"