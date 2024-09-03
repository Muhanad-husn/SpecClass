import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
import os
from utils.config_loader import config
from utils.logger import logger
from src.embedding_manager import EmbeddingManager

class VectorStore:
    def __init__(self):
        self.collection_name = config['collection_name']
        self.chroma_db_dir = config['chroma_db_dir']
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        self.embedding_manager = EmbeddingManager()
        self.persistent_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        self.vector_store = None

    def store_documents(self, documents: List[Document]):
        logger.info(f"Storing {len(documents)} documents in ChromaDB")
        try:
            collection = self.persistent_client.get_or_create_collection(self.collection_name)
            
            ids = [str(i) for i in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            embeddings = self.embedding_manager.encode(texts)
            
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            self.vector_store = Chroma(
                client=self.persistent_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_manager.encode
            )
            
            logger.info(f"Stored {len(documents)} documents in ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error storing documents in vector store: {str(e)}")
            raise

    def similarity_search(self, query, k=3):
        if self.vector_store is None:
            logger.error("Vector store not initialized. Call store_documents first.")
            raise ValueError("Vector store not initialized")
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise