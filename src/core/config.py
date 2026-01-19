import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-56bb3c0a5e5c48eb969be81eec3d9564")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8045/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3-flash")
    
    # Maps Settings
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", "d39d5a76e0ec8e18c673955bf7baf317")

settings = Settings()
