import logging
from logging.handlers import RotatingFileHandler
import os
from utils.config_loader import config

class Logger:
    def __init__(self):
        self.log_dir = config.log_dir
        self.log_level = getattr(logging, config.get('log_level', 'INFO').upper())
        self.max_log_size = config.get('max_log_size', 5 * 1024 * 1024)  # 5 MB by default
        self.backup_count = config.get('log_backup_count', 3)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('main')
        self.logger.setLevel(self.log_level)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'main.log'),
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to main logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Create component-specific loggers
        self.create_component_logger('pipeline', self.log_level)
        self.create_component_logger('document_processor', self.log_level)
        self.create_component_logger('embedding_manager', self.log_level)
        self.create_component_logger('vector_store', self.log_level)
        self.create_component_logger('classification_manager', self.log_level)
        self.create_component_logger('file_handler', self.log_level)

    def create_component_logger(self, name, level):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, f'{name}.log'),
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)

    def get_logger(self, name='main'):
        return logging.getLogger(name)

# Create a global logger instance
logger_instance = Logger()
get_logger = logger_instance.get_logger