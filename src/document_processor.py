from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from typing import List
from utils.config_loader import config
from utils.logger import logger
import os
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self):
        self.llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes" # config['llmsherpa_api_url']
        self.specifications_dir = config['specifications_dir']
        self.chunk_strategy = config['chunk_strategy']
        self.supported_extensions = ('.pdf', '.docx', '.pptx', '.html', '.txt', '.xml')

    def process_documents(self) -> List[dict]:
        all_documents = []
        
        if not os.path.exists(self.specifications_dir):
            logger.error(f"Specifications directory does not exist: {self.specifications_dir}")
            return all_documents

        files = [f for f in os.listdir(self.specifications_dir) if f.lower().endswith(self.supported_extensions)]
        if not files:
            logger.warning(f"No supported files found in the specifications directory. Supported formats: {', '.join(self.supported_extensions)}")
            return all_documents

        total_files = len(files)
        with tqdm(total=total_files, desc="Processing documents") as pbar:
            for filename in files:
                file_path = os.path.join(self.specifications_dir, filename)
                logger.info(f"Processing file: {filename}")
                try:
                    chunks = self._process_file(file_path, pbar)
                    all_documents.extend(chunks)
                    logger.info(f"Successfully processed {filename}, extracted {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
                pbar.update(1)
        
        if not all_documents:
            logger.warning("No documents were successfully processed.")
        else:
            logger.info(f"Total processed chunks: {len(all_documents)}")
        
        return all_documents

    def _process_file(self, file_path: str, pbar: tqdm) -> List[dict]:
        loader = LLMSherpaFileLoader(
            file_path=file_path,
            new_indent_parser=True,
            apply_ocr=True,
            strategy=self.chunk_strategy,
            llmsherpa_api_url=self.llmsherpa_api_url
        )
        chunks = loader.load()
        pbar.set_postfix({"Chunks": len(chunks)})
        return chunks

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    processed_chunks = processor.process_documents()
    print(f"\nProcessed {len(processed_chunks)} total chunks from all documents.")
    if processed_chunks:
        print("\nSample chunk:")
        print(f"Content: {processed_chunks[0].page_content[:200]}...")
        print(f"Metadata: {processed_chunks[0].metadata}")