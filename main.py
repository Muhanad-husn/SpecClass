import os
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.classifier import SpecificationClassifier
from src.report_generator import ReportGenerator
from utils.logger import logger
from utils.file_handler import file_handler

def main():
    try:
        # Initialize components
        doc_processor = DocumentProcessor()
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        classifier = SpecificationClassifier(vector_store, embedding_manager)
        report_generator = ReportGenerator()

        # Process all specification files
        documents = doc_processor.load_specification_files()

        # Store documents in vector store
        vector_store.store_documents(documents)

        # Handle input file
        input_file_path = file_handler.get_input_file()
        input_df, input_column, sheet_name = file_handler.read_input_file(input_file_path)

        results = []
        for item in input_df[input_column]:
            similar_chunks = classifier.classify_item(item)
            report = report_generator.generate_report(item, similar_chunks)
            
            # Extract primary classification and reasoning from the report
            # This is a simplified extraction and might need to be adjusted based on the actual report format
            primary_classification = similar_chunks[0].metadata.get('section_title', 'Unknown')
            reasoning = report.split('\n')[0]  # Assuming the first line contains the main reasoning
            
            results.append({
                'similarity_score': similar_chunks[0].metadata.get('score', 0),
                'primary_classification': primary_classification,
                'reasoning': reasoning
            })

        # Write output file
        file_handler.write_output_file(input_df, results, input_column, os.path.basename(input_file_path))

        print("Classification completed. Check the output file in the data/output directory.")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")

if __name__ == "__main__":
    main()