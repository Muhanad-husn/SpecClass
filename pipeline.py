import argparse
from utils.config_loader import Config
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.classification_manager import ClassificationManager
from utils.file_handler import FileHandler
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

class Pipeline:
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(default_collection_name=self.config.collection_name)
        self.file_handler = FileHandler()
        self.classification_manager = None
        self.logger = logger

    def reset_vector_store(self):
        self.logger.info("Resetting vector store")
        self.vector_store.reset_vector_store()

    def process_and_store_documents(self):
        try:
            processed_documents = self.doc_processor.process_documents()
            self.logger.info(f"Processed {len(processed_documents)} documents")

            pipeline_batch_size = self.config.get('pipeline_batch_size', 100)  # Default to 100 if not specified
            total_batches = (len(processed_documents) + pipeline_batch_size - 1) // pipeline_batch_size
            
            for i in tqdm(range(0, len(processed_documents), pipeline_batch_size), total=total_batches, desc="Processing batches"):
                batch = processed_documents[i:i+pipeline_batch_size]
                self._process_batch(batch)

            self.logger.info("Document processing and storage completed successfully")
        except Exception as e:
            self.logger.error(f"Error in document processing and storage: {str(e)}", exc_info=True)
            raise

    def _process_batch(self, batch):
        texts = [doc.page_content for doc in batch]
        embeddings = self.embedding_manager.encode(texts, show_progress=False)
        self.vector_store.store_documents(batch, embeddings)

    def verify_storage(self):
        try:
            total_docs = self.vector_store.get_document_count()
            self.logger.info(f"Total documents in storage: {total_docs}")

            if total_docs == 0:
                self.logger.warning("No documents found in storage")
                return

            self._sample_documents()
        except Exception as e:
            self.logger.error(f"Error in storage verification: {str(e)}", exc_info=True)
            raise

    def _sample_documents(self, num_samples=5):
        sample_query = "sample query for verification"
        results = self.vector_store.similarity_search(sample_query, k=num_samples)

        self.logger.info(f"Retrieved {len(results)} sample documents:")
        for i, (doc, score) in enumerate(results, 1):
            self.logger.info(f"Sample {i}:")
            self.logger.info(f"Content: {doc.page_content[:100]}...")
            self.logger.info(f"Metadata: {doc.metadata}")
            self.logger.info(f"Similarity Score: {score}")

    def process_and_classify_items(self):
        try:
            file_path = self.file_handler.get_input_file()
            items, chosen_column, _ = self.file_handler.read_input_file(file_path)
            
            similar_docs = []
            for item in tqdm(items, desc="Retrieving similar documents"):
                docs = self.vector_store.similarity_search(item, k=3)
                similar_docs.append(docs)
            
            classified_items = self.classification_manager.process_and_classify_items(items, similar_docs)
            
            self.file_handler.write_results(items, classified_items, chosen_column)
            return classified_items
        except Exception as e:
            self.logger.error(f"Error in processing and classifying items: {str(e)}", exc_info=True)
            raise

    def run(self, reset=False, model_type=None, model_name=None):
        try:
            self.logger.info("Starting pipeline execution")
            
            if reset:
                self.reset_vector_store()
            
            self.process_and_store_documents()
            self.verify_storage()
            
            model_type = model_type or self.prompt_for_model_type()
            self.classification_manager = ClassificationManager(model_type=model_type, model_name=model_name)
            
            classified_items = self.process_and_classify_items()
            
            self._print_summary(classified_items)
            
            self.logger.info("Pipeline execution completed successfully")
            return classified_items
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            raise
        finally:
            # Clear caches
            self.embedding_manager.clear_cache()
            self.vector_store.clear_cache()

    def _print_summary(self, classified_items):
        print(f"\nClassified {len(classified_items)} items.")
        print("\nSample results:")
        for item in classified_items[:5]:
            print(f"\nItem: {item['item'][:50]}...")
            print(f"Primary Classification: {item['primary_classification']}")
            print(f"Overall Classification: {item['classification']}")
            print(f"Confidence: {item['confidence']}")
        print("\nClassification process completed. Results have been written to CSV.")

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