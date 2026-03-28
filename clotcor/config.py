from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

DATASET_PATH = DATA_DIR / "Estadisticas.csv"
UNIQUE_VALUES_PATH = DATA_DIR / "Unique_values_dict.json"
MODEL_ARTIFACT_PATH = MODEL_DIR / "crime_predictor.joblib"

RANDOM_STATE = 42
