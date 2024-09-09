from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from utils.logger import logger
from utils.config_loader import config
from utils.file_handler import FileHandler
from tqdm import tqdm
import os

class Pipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(default_collection_name='specification_book_collection')
        self.file_handler = FileHandler()
        self.batch_size = config.get('pipeline_batch_size', 32)

    def reset_vector_store(self):
        logger.info("Resetting vector store")
        self.vector_store.reset_vector_store()

    def process_and_store_documents(self):
        document_path = config['specifications_dir']
        logger.info(f"Starting document processing from: {document_path}")
        try:
            processed_documents = self.doc_processor.process_documents()
            logger.info(f"Processed {len(processed_documents)} documents")

            total_batches = (len(processed_documents) + self.batch_size - 1) // self.batch_size
            for i in tqdm(range(0, len(processed_documents), self.batch_size), total=total_batches, desc="Processing batches"):
                batch = processed_documents[i:i+self.batch_size]
                texts = [doc.page_content for doc in batch]
                embeddings = self.embedding_manager.encode(texts, show_progress=False)
                self.vector_store.store_documents(batch, embeddings)

            logger.info("Document processing and storage completed successfully")
        except Exception as e:
            logger.error(f"Error in document processing and storage: {str(e)}", exc_info=True)
            raise

    def verify_storage(self, num_samples=5):
        logger.info("Verifying document storage")
        try:
            total_docs = self.vector_store.get_document_count()
            logger.info(f"Total documents in storage: {total_docs}")

            if total_docs == 0:
                logger.warning("No documents found in storage")
                return

            sample_query = "sample query for verification"
            results = self.vector_store.similarity_search(sample_query, k=num_samples)

            logger.info(f"Retrieved {len(results)} sample documents:")
            for i, item in enumerate(results, 1):
                doc = item[0] if isinstance(item, tuple) else item
                logger.info(f"Sample {i}:")
                logger.info(f"Content: {doc.page_content[:100]}...")
                logger.info(f"Metadata: {doc.metadata}")
                if isinstance(item, tuple):
                    logger.info(f"Similarity Score: {item[1]}")

            logger.info("Storage verification completed")
        except Exception as e:
            logger.error(f"Error in storage verification: {str(e)}", exc_info=True)
            raise

    def get_items_and_contexts(self, k=3):
        logger.info("Getting items and contexts")
        try:
            file_path = self.file_handler.get_input_file()
            items, chosen_column, _ = self.file_handler.read_input_file(file_path)
            
            contexts = []
            for item in tqdm(items, desc="Retrieving contexts"):
                similar_docs = self.vector_store.similarity_search(item, k=k)
                context = "\n\n".join([doc[0].page_content if isinstance(doc, tuple) else doc.page_content for doc in similar_docs])
                contexts.append(context)
            
            logger.info(f"Retrieved contexts for {len(items)} items")
            return items, contexts, chosen_column
        except Exception as e:
            logger.error(f"Error in getting items and contexts: {str(e)}")
            raise

    def process_items(self, k=3):
        logger.info("Processing items for classification")
        try:
            items, contexts, chosen_column = self.get_items_and_contexts(k)
            results = []

            for item, context in zip(items, contexts):
                embedding = self.embedding_manager.encode(item)
                similar_docs = self.vector_store.similarity_search(item, k=k)
                
                result = {
                    'item': item,
                    'embedding': embedding,
                    'context': context,
                    'similar_docs': [doc[0] if isinstance(doc, tuple) else doc for doc in similar_docs]
                }
                results.append(result)

            logger.info(f"Processed {len(results)} items")
            return results
        except Exception as e:
            logger.error(f"Error in processing items: {str(e)}")
            raise

    def run(self):
        try:
            logger.info("Starting pipeline execution")
            self.reset_vector_store()
            self.process_and_store_documents()
            self.verify_storage()
            
            results = self.process_items()
            
            # Example of how to use the results
            for i, result in enumerate(results[:5], 1):  # Print first 5 results
                logger.info(f"Result {i}:")
                logger.info(f"Item: {result['item'][:50]}...")
                logger.info(f"Embedding shape: {result['embedding'].shape}")
                logger.info(f"Context length: {len(result['context'])}")
                logger.info(f"Number of similar docs: {len(result['similar_docs'])}")
                logger.info("---")
            
            logger.info("Pipeline execution completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()