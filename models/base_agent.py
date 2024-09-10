# base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from models.llms import OllamaModel, OpenAIModel, ClaudeModel
from utils.config_loader import config
from utils.logger import get_logger
logger = get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, model_type: str = 'openai', model_name: str = None):
        self.model_type = model_type
        self.model_name = model_name or config.get(f'{model_type}_model_name')
        self.llm = self._get_llm()

    def _get_llm(self):
        if self.model_type == 'ollama':
            return OllamaModel(model=self.model_name)
        elif self.model_type == 'openai':
            return OpenAIModel(model=self.model_name)
        elif self.model_type == 'claude':
            return ClaudeModel(model=self.model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    @abstractmethod
    def get_prompt(self, context: str, query: str) -> str:
        pass

    @abstractmethod
    def process_response(self, response: str) -> Dict[str, Any]:
        pass

    def invoke(self, context: str, query: str) -> Dict[str, Any]:
        prompt = self.get_prompt(context, query)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = self.llm.invoke(messages)
            return self.process_response(response)
        except Exception as e:
            logger.error(f"Error in agent invocation: {str(e)}")
            return {"error": str(e)}