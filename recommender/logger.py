import logging
import os
from datetime import datetime

def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("financial_recommender")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(f"Логгер инициализирован, лог пишется в {log_file}")
    return logger