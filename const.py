import os

# Formats
FORMAT = {
    "task_id": [],
    "accuracy": [],
}

HOME_DIR = os.path.join(os.path.expanduser ('~'),'.wq')
APP_ENGINE_PATH = f"sqlite:///{HOME_DIR}/process_data.db"
BASE_LOG_DIR = os.path.join(HOME_DIR, "logs")
DEFAULT_LOG_DIR_OUT = f"{BASE_LOG_DIR}/stdout.txt"