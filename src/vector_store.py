import os
import shutil
from typing import List, Dict
import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from chromadb.config import Settings
from utils.config_loader import config
from utils.logger import logger
from src.embedding_manager import EmbeddingManager

class CustomEmbeddingFunction(Embeddings):
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_manager.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_manager.encode([text])[0].tolist()

class VectorStore:
    def __init__(self, default_collection_name='default'):
        self.chroma_db_dir = config['chroma_db_dir']
        self.embedding_manager = EmbeddingManager()
        self.embedding_function = CustomEmbeddingFunction(self.embedding_manager)
        self.vector_store = None
        self.current_collection_name = default_collection_name
        self.initialize_vector_store(self.current_collection_name)

    def initialize_vector_store(self, collection_name):
        logger.info(f"Initializing vector store with collection: {collection_name}")
        try:
            self.vector_store = Chroma(
                persist_directory=self.chroma_db_dir,
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                client_settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized vector store at {self.chroma_db_dir} with collection {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    def switch_collection(self, collection_name):
        logger.info(f"Switching to collection: {collection_name}")
        self.current_collection_name = collection_name
        self.initialize_vector_store(collection_name)

    def reset_vector_store(self):
        logger.info("Resetting vector store")
        try:
            if os.path.exists(self.chroma_db_dir):
                shutil.rmtree(self.chroma_db_dir)
                logger.info(f"Removed existing vector store at {self.chroma_db_dir}")
            os.makedirs(self.chroma_db_dir, exist_ok=True)
            self.initialize_vector_store(self.current_collection_name)
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            raise RuntimeError(f"Failed to reset vector store: {str(e)}")

    def store_documents(self, processed_documents: List[Document], embeddings: np.ndarray = None):
        logger.info(f"Storing {len(processed_documents)} documents in ChromaDB collection '{self.current_collection_name}'")
        try:
            texts = [doc.page_content for doc in processed_documents]
            metadatas = [doc.metadata for doc in processed_documents]
            ids = [str(i) for i in range(len(processed_documents))]
            
            if embeddings is not None:
                embeddings = embeddings.tolist()
            
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"Stored {len(processed_documents)} documents in ChromaDB collection '{self.current_collection_name}'")
        except Exception as e:
            logger.error(f"Error storing documents in vector store: {str(e)}")
            raise RuntimeError(f"Failed to store documents: {str(e)}")

    def similarity_search(self, query: str, k: int = None):
        if k is None:
            k = config.get('similarity_search_k', 5)  # Default to 5 if not specified in config
        logger.info(f"Performing similarity search for query: {query}")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise RuntimeError(f"Failed to perform similarity search: {str(e)}")

    def get_document_count(self):
        try:
            return len(self.vector_store.get()['ids'])
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            raise RuntimeError(f"Failed to get document count: {str(e)}")

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore('test_collection')
    
    sample_documents = [
        Document(page_content="This is the first sample text.", metadata={"index": 0, "source": "test"}),
        Document(page_content="Here's another example of text to encode.", metadata={"index": 1, "source": "test"}),
        Document(page_content="And a third one for good measure.", metadata={"index": 2, "source": "test"})
    ]
    
    vector_store.store_documents(sample_documents)
    
    print(f"Stored {len(sample_documents)} documents in the vector store.")
    
    query_text = "TCO"
    results = vector_store.similarity_search(query_text, k=2)
    
    print("\nSearch Results:")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")
    
    print(f"\nTotal documents in store: {vector_store.get_document_count()}")
    
    vector_store.vector_store.delete_collection()
    print("\nTest completed and collection deleted.")