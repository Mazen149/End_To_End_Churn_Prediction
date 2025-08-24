from dotenv import load_dotenv
import os
import joblib

load_dotenv(override=True)

# .env variables
APP_NAME = os.getenv('APP_NAME')
VERSION = os.getenv('VERSION')
SECRET_KEY_TOKEN = os.getenv('SECRET_KEY_TOKEN')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, "models")

# Models
preprocessor = joblib.load(os.path.join(MODELS_FOLDER_PATH, 'preprocessor.pkl'))
forest_model = joblib.load(os.path.join(MODELS_FOLDER_PATH, 'forest_tuned.pkl'))
xgboost_model = joblib.load(os.path.join(MODELS_FOLDER_PATH, 'xgb-tuned.pkl'))