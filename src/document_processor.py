import os
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain_core.documents import Document
from typing import List
from utils.config_loader import config
from utils.logger import logger

class DocumentProcessor:
    def __init__(self):
        self.api_url = config['llmsherpa_api_url']
        self.specifications_dir = config['specifications_dir']
        self.chunk_strategy = config['chunk_strategy']

    def load_specification_files(self) -> List[Document]:
        all_documents = []
        for filename in os.listdir(self.specifications_dir):
            file_path = os.path.join(self.specifications_dir, filename)
            if os.path.isfile(file_path):
                logger.info(f"Loading specification file: {filename}")
                try:
                    documents = self._load_file(file_path)
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} {self.chunk_strategy} from {filename}")
                except Exception as e:
                    logger.error(f"Error loading file {filename}: {str(e)}")
        
        if not all_documents:
            logger.error("No specification files were successfully loaded.")
            raise ValueError("No specification files loaded")

        logger.info(f"Total loaded documents: {len(all_documents)}")
        return all_documents

    def _load_file(self, file_path: str) -> List[Document]:
        loader = LLMSherpaFileLoader(
            file_path,
            new_indent_parser=True,
            apply_ocr=True,
            strategy=self.chunk_strategy,
            llmsherpa_api_url=self.api_url
        )
        return list(loader.lazy_load())