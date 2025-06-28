import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    PROJECT_ROOT_PATH: str = os.getenv("PROJECT_ROOT_PATH")
    MODEL: str = os.getenv("MODEL")
    DATA_PATH: str = os.getenv("DATA_PATH")

config = Config