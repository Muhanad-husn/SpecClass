import argparse
from utils.logger import logger
from utils.config_loader import config
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.classification_manager import ClassificationManager
from utils.file_handler import FileHandler
from tqdm import tqdm

class Pipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(default_collection_name='specification_book_collection')
        self.file_handler = FileHandler()
        self.classification_manager = None
        self.logger = logger
        self.batch_size = config.get('pipeline_batch_size', 32)

    def reset_vector_store(self):
        self.logger.info("Resetting vector store")
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

    def process_and_classify_items(self):
        file_path = self.file_handler.get_input_file()
        items, chosen_column, _ = self.file_handler.read_input_file(file_path)
        
        classified_items = []
        
        for item in tqdm(items, desc="Processing and classifying items"):
            # Embed the item
            embedding = self.embedding_manager.encode(item)
            
            # Retrieve similar documents
            similar_docs = self.vector_store.similarity_search(item, k=3)
            
            # Prepare context for classification
            context = "\n".join([doc[0].page_content if isinstance(doc, tuple) else doc.page_content for doc in similar_docs])
            
            # Classify the item
            try:
                classification_result = self.classification_manager.invoke(context, item)
                classified_item = {
                    'item': item,
                    'primary_classification': classification_result['primary_classification'],
                    'classification': classification_result['classification'],
                    'reasoning': classification_result['reasoning'],
                    'confidence': classification_result['confidence']
                }
            except Exception as e:
                self.logger.error(f"Error classifying item: {str(e)}")
                classified_item = {
                    'item': item,
                    'primary_classification': 'Error',
                    'classification': 'Error',
                    'reasoning': f"Error in classification: {str(e)}",
                    'confidence': 0.0
                }
            
            classified_items.append(classified_item)
        
        # Write results to file
        self.file_handler.write_results([item['item'] for item in classified_items], classified_items, chosen_column)
        
        return classified_items

    def run(self, reset=False, model_type=None, model_name=None):
        try:
            self.logger.info("Starting pipeline execution")
            
            if reset:
                self.logger.info("Resetting vector store as requested.")
                self.reset_vector_store()
            
            self.process_and_store_documents()
            self.verify_storage()
            
            if model_type is None:
                model_type = self.prompt_for_model_type()
            
            self.classification_manager = ClassificationManager(model_type=model_type, model_name=model_name)
            classified_items = self.process_and_classify_items()
            
            self.logger.info(f"Successfully classified {len(classified_items)} items")
            self.logger.info("Pipeline execution completed successfully")
            
            # Print a summary of the results
            print(f"\nClassified {len(classified_items)} items.")
            print("\nSample results:")
            for item in classified_items[:5]:  # Print first 5 results
                print(f"\nItem: {item['item'][:50]}...")
                print(f"Primary Classification: {item['primary_classification']}")
                print(f"Overall Classification: {item['classification']}")
                print(f"Confidence: {item['confidence']}")

            print("\nClassification process completed. Results have been written to CSV.")
            
            return classified_items
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            raise

    @staticmethod
    def prompt_for_model_type():
        valid_types = ["ollama", "openai", "claude"]
        while True:
            model_type = input("Please enter the model type (ollama/openai/claude): ").lower()
            if model_type in valid_types:
                return model_type
            else:
                print(f"Invalid model type. Please choose from {', '.join(valid_types)}.")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Run the document classification pipeline.")
        parser.add_argument("--reset", action="store_true", help="Reset the vector store before processing")
        parser.add_argument("--model-type", choices=["ollama", "openai", "claude"], help="Specify the model type to use")
        parser.add_argument("--model-name", help="Specify the model name to use")
        args = parser.parse_args()

        pipeline = Pipeline()
        pipeline.run(reset=args.reset, model_type=args.model_type, model_name=args.model_name)

if __name__ == "__main__":
    Pipeline.main()