from django.apps import AppConfig
from django.conf import settings
import joblib


class NurseryApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'nursery_api'
    MODEL_FILE = settings.MODELS/"Bagging_KNN_Model.joblib"
    model = joblib.load(MODEL_FILE)
