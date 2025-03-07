import os

class Config:
    """Configuration settings for Flask app"""

    DEBUG = os.getenv("DEBUG", False)  # Set True for local testing
    DATABASE_URL = os.getenv("DATABASE_URL")  # Set this in Railway variables
    SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret")

CurrentConfig = Config()
