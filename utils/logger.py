# utils/logger.py

import logging
from colorlog import ColoredFormatter
import os
from functools import wraps

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console Handler (more detailed, includes model streaming)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)

    # File Handler (less detailed, only required info)
    file_handler = logging.FileHandler('logs/application.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create a global logger instance
logger = setup_logger(__name__)

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Finished function: {func.__name__}")
        return result
    return wrapper

class LoggedObject:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr):
            return log_function_call(attr)
        return attr

# Usage example:
# class MyClass(LoggedObject):
#     def my_method(self):
#         self.logger.info("Doing something")
#         # Method implementation