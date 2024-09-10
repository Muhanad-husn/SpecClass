from sentence_transformers import SentenceTransformer
from utils.config_loader import Config
from tqdm import tqdm
import numpy as np
import warnings
from functools import lru_cache
import hashlib
from utils.logger import get_logger
logger = get_logger(__name__)

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

class EmbeddingManager:
    def __init__(self):
        self.config = Config()
        self.model_name = self.config.embedding_model_name
        self.batch_size = self.config.get('embedding_batch_size', 32)
        self.cache_size = self.config.get('embedding_cache_size', 10000)
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

    @lru_cache(maxsize=10000)
    def _cached_encode(self, text_hash):
        # This method will never be called directly, it's just for caching
        pass

    def _hash_text(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, texts, show_progress=True):
        if not isinstance(texts, list):
            texts = [texts]
        
        try:
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            # Check cache for existing embeddings
            for i, text in enumerate(texts):
                text_hash = self._hash_text(text)
                cached_embedding = self._cached_encode(text_hash)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Encode uncached texts
            if uncached_texts:
                if show_progress:
                    uncached_embeddings = []
                    for i in tqdm(range(0, len(uncached_texts), self.batch_size), desc="Encoding texts"):
                        batch = uncached_texts[i:i+self.batch_size]
                        batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                        uncached_embeddings.extend(batch_embeddings)
                else:
                    uncached_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)

                # Cache new embeddings
                for text, embedding in zip(uncached_texts, uncached_embeddings):
                    text_hash = self._hash_text(text)
                    self._cached_encode.cache_clear()  # Clear the least recently used item if cache is full
                    self._cached_encode(text_hash)  # Cache the new embedding
                
                # Insert uncached embeddings back into the correct positions
                for i, embedding in zip(uncached_indices, uncached_embeddings):
                    embeddings.insert(i, embedding)

            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

    def clear_cache(self):
        self._cached_encode.cache_clear()
        logger.info("Embedding cache cleared")

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
    
    # Test caching
    cached_embeddings = embedding_manager.encode(sample_texts)
    print("\nCached embeddings retrieved.")
    
    single_text = "This is a single text to encode."
    single_embedding = embedding_manager.encode(single_text)
    
    print(f"\nSingle text embedding shape: {single_embedding.shape}")
    print(f"Single embedding type: {type(single_embedding)}")
    print(f"Single embedding: {single_embedding[0][:5]}...")  # Show first 5 values

    embedding_manager.clear_cache()
    print("\nCache cleared.")