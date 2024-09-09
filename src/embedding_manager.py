from sentence_transformers import SentenceTransformer
from utils.config_loader import config
from utils.logger import logger
from tqdm import tqdm
import numpy as np
import warnings

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

class EmbeddingManager:
    def __init__(self):
        self.model_name = config['embedding_model_name']
        self.batch_size = config.get('embedding_batch_size', 32)
        self.model = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def encode(self, texts, show_progress=True):
        if not isinstance(texts, list):
            texts = [texts]
        
        try:
            if show_progress:
                embeddings = []
                for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding texts"):
                    batch = texts[i:i+self.batch_size]
                    batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                return np.vstack(embeddings)
            else:
                return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    embedding_manager = EmbeddingManager()
    
    sample_texts = [
        "This is the first sample text.",
        "Here's another example of text to encode.",
        "And a third one for good measure."
    ]
    
    embeddings = embedding_manager.encode(sample_texts)
    
    print(f"\nCreated {len(embeddings)} embeddings.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding type: {type(embeddings)}")
    print(f"First embedding: {embeddings[0][:5]}...")  # Show first 5 values
    
    single_text = "This is a single text to encode."
    single_embedding = embedding_manager.encode(single_text)
    
    print(f"\nSingle text embedding shape: {single_embedding.shape}")
    print(f"Single embedding type: {type(single_embedding)}")
    print(f"Single embedding: {single_embedding[0][:5]}...")  # Show first 5 values