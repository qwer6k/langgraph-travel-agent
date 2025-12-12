import os
import time
import hashlib
from dotenv import load_dotenv
from amadeus import Client
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# ENV & CLIENTS
# ---------------------------------------------------------------------------

# 支持包内 .env，也支持项目根目录 .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# Core keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

# Optional keys
HOTELBEDS_API_KEY = os.getenv("HOTELBEDS_API_KEY")
HOTELBEDS_API_SECRET = os.getenv("HOTELBEDS_API_SECRET")

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

if not all([DEEPSEEK_API_KEY, AMADEUS_API_KEY, AMADEUS_API_SECRET]):
    raise ValueError(
        "Required API keys missing: DEEPSEEK_API_KEY, AMADEUS_API_KEY, AMADEUS_API_SECRET"
    )

# LLM（DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
)

# Amadeus
amadeus = None
try:
    amadeus = Client(client_id=AMADEUS_API_KEY, client_secret=AMADEUS_API_SECRET)
    print("✓ Amadeus client initialized")
except Exception as e:
    print(f"⚠ Amadeus client initialization warning: {e}")


def hotelbeds_headers() -> dict | None:
    """Generate Hotelbeds authentication headers (or None if not configured)."""
    if not HOTELBEDS_API_KEY or not HOTELBEDS_API_SECRET:
        return None

    utc_timestamp = int(time.time())
    signature = hashlib.sha256(
        f"{HOTELBEDS_API_KEY}{HOTELBEDS_API_SECRET}{utc_timestamp}".encode()
    ).hexdigest()
    return {
        "Api-key": HOTELBEDS_API_KEY,
        "X-Signature": signature,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
