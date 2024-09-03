from sentence_transformers import SentenceTransformer
from utils.config_loader import config
from utils.logger import logger

class EmbeddingManager:
    def __init__(self):
        self.model_name = config['embedding_model_name']
        self.model = None

    def load_model(self):
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def encode(self, texts):
        if self.model is None:
            self.load_model()
        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise