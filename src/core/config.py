import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL")
    MODEL_NAME: str = os.getenv("MODEL_NAME")
    
    # Maps Settings
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY")

settings = Settings()
