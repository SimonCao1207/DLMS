import os

# Formats
FORMAT = {
    "task_id": [],
    "learning_rate" : [],
    "accuracy": [],
    "test_loss" : [],
}

HOME_DIR = os.path.join(os.path.expanduser ('~'),'.wq')
APP_ENGINE_PATH = f"sqlite:///{HOME_DIR}/process_data.db"
BASE_LOG_DIR = os.path.join(HOME_DIR, "logs")
DEFAULT_LOG_DIR_OUT = f"{BASE_LOG_DIR}/stdout.txt"
DEFAULT_RESULT_DIR_OUT = f"{BASE_LOG_DIR}/result.json"