import logging
import os

def get_logger(name='data_pipeline', log_level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional: Save to file as well
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/{name}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
