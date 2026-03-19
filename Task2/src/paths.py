from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# -------- DATA --------
DATA_DIR = ROOT_DIR / "data"

IMG_DATA_DIR = DATA_DIR / "img"
NER_DATA_DIR = DATA_DIR / "ner"

TRAIN_DIR = IMG_DATA_DIR / "train"
TEST_DIR = IMG_DATA_DIR / "test"
VAL_DIR = IMG_DATA_DIR / "val"

TRAIN_NER = "data/data_ner/train_data_ner.json"

# -------- MODELS --------
MODEL_DIR = ROOT_DIR / "models"

IMG_MODEL_DIR = MODEL_DIR / "img"
NER_MODEL_DIR = MODEL_DIR / "ner"

IMG_MODEL_PATH = IMG_MODEL_DIR / "model.pth"
NER_MODEL_PATH = NER_MODEL_DIR

# create dirs
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------- UTILS --------
UTILS_DIR = ROOT_DIR / "utils"
CLASSES_PATH = UTILS_DIR / "class_names.json"