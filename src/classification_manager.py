# classification_manager.py
from utils.file_handler import FileHandler
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from utils.logger import logger
from models.base_agent import BaseAgent
from models.prompts import CLASSIFICATION_PROMPT, GUIDED_JSON
from utils.config_loader import config
import json

class ClassificationManager(BaseAgent):
    def __init__(self, model_type=None, model_name=None):
        model_type = model_type or config['model_type']
        model_name = model_name or config.get(f'{model_type}_model_name')
        super().__init__(model_type=model_type, model_name=model_name)
        self.file_handler = FileHandler()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(default_collection_name=config['collection_name'])
        self.spec_book_description = None
        self.item_description = None
        self.weighted_spec = None

    def collect_user_input(self):
        self.spec_book_description = input("Please enter a description for the specification book: ")
        self.item_description = input("Please enter a description for the items to be classified: ")
        self.weighted_spec = input("Enter any weighted specification (or press Enter if none): ")

    def get_prompt(self, context: str, query: str) -> str:
        if self.spec_book_description is None:
            self.collect_user_input()
        
        return CLASSIFICATION_PROMPT.format(
            context=context,
            item=query,
            spec_book_description=self.spec_book_description,
            item_description=self.item_description,
            weighted_spec=self.weighted_spec if self.weighted_spec else "No specific specification has more weight."
        )

    def process_response(self, response: str) -> dict:
        try:
            result = json.loads(response)
            return {
                'primary_classification': result.get('primary_classification', 'Unknown'),
                'classification': result.get('classification', 'Unknown'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'confidence': float(result.get('confidence', 0.0))
            }
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON response: {response}")
            return {
                'primary_classification': 'Error',
                'classification': 'Error',
                'reasoning': 'Failed to process the model response',
                'confidence': 0.0
            }

    def process_new_items(self, pipeline_results):
        try:
            new_items = [result['item'] for result in pipeline_results]
            logger.info(f"Processing {len(new_items)} new items from pipeline results")
            
            results = []
            for result in pipeline_results:
                similar_docs = self.vector_store.similarity_search(result['item'], k=3)
                results.append({
                    'item': result['item'],
                    'similar_docs': similar_docs,
                    'embedding': result['embedding']
                })

            logger.info(f"Successfully processed {len(results)} items")
            return results
        except Exception as e:
            logger.error(f"Error in process_new_items: {str(e)}", exc_info=True)
            raise

    def classify_items(self, results):
        classified_items = []
        for result in results:
            item = result['item']
            context = "\n".join([doc[0].page_content if isinstance(doc, tuple) else doc.page_content for doc in result['similar_docs']])
            
            try:
                classification_result = self.invoke(context, item)
                classified_items.append({
                    'item': item,
                    'primary_classification': classification_result['primary_classification'],
                    'classification': classification_result['classification'],
                    'reasoning': classification_result['reasoning'],
                    'confidence': classification_result['confidence']
                })
            except Exception as e:
                logger.error(f"Error classifying item: {str(e)}")
                classified_items.append({
                    'item': item,
                    'primary_classification': 'Error',
                    'classification': 'Error',
                    'reasoning': f"Error in classification: {str(e)}",
                    'confidence': 0.0
                })

        logger.info(f"Successfully classified {len(classified_items)} items")
        return classified_items