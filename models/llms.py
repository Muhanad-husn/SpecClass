# llms.py
import requests
import json
import os
from typing import List, Dict
from utils.config_loader import config
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from utils.logger import get_logger
logger = get_logger(__name__)

class BaseModel:
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(requests.RequestException))
    def _make_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    

class OllamaModel(BaseModel):
    def __init__(self, model: str = None):
        super().__init__(
            temperature=config.get('ollama_temperature', 0),
            model=model or config['ollama_model_name'],
            json_response=config.get('ollama_json_response', False),
            max_retries=config.get('ollama_max_retries', 3),
            retry_delay=config.get('ollama_retry_delay', 1)
        )
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = config.get('ollama_model_endpoint', "http://localhost:11434/api/generate")

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "temperature": self.temperature,
        }

        if self.json_response:
            payload["format"] = "json"
        
        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if self.json_response:
                response = json.dumps(json.loads(request_response_json['response']))
            else:
                response = str(request_response_json['response'])

            return response
        except requests.RequestException as e:
            logger.error(f"Error in invoking Ollama model: {str(e)}")
            return json.dumps({"error": f"Error in invoking model: {str(e)}"})
        except json.JSONDecodeError as e:
            logger.error(f"Error processing Ollama response: {str(e)}")
            return json.dumps({"error": f"Error processing response: {str(e)}"})
        

class ClaudeModel(BaseModel):
    def __init__(self, model: str = None):
        super().__init__(
            temperature=config.get('claude_temperature', 0),
            model=model or config['claude_model_name'],
            json_response=config.get('claude_json_response', False),
            max_retries=config.get('claude_max_retries', 3),
            retry_delay=config.get('claude_retry_delay', 1)
        )
        self.api_key = config.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables")
        self.headers = {
            'Content-Type': 'application/json', 
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        self.model_endpoint = "https://api.anthropic.com/v1/messages"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        content = f"{system}\n\nHuman: {user}\n\nAssistant:"
        if self.json_response:
            content += " Your output must be JSON formatted. Return only the specified JSON format, without any additional text."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 4096,
            "temperature": self.temperature,
        }

        try:
            response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'content' not in response_json or not response_json['content']:
                raise ValueError("No content in response")

            response_content = response_json['content'][0]['text']
            
            if self.json_response:
                return json.dumps(json.loads(response_content))
            else:
                return response_content

        except requests.RequestException as e:
            logger.error(f"Error in invoking Claude model: {str(e)}")
            return json.dumps({"error": f"Error in invoking model: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error processing Claude response: {str(e)}")
            return json.dumps({"error": f"Error processing response: {str(e)}"})


class OpenAIModel(BaseModel):
    def __init__(self, model: str = None):
        super().__init__(
            temperature=config.get('openai_temperature', 0),
            model=model or config['openai_model_name'],
            json_response=config.get('openai_json_response', False),
            max_retries=config.get('openai_max_retries', 3),
            retry_delay=config.get('openai_retry_delay', 1)
        )
        self.model_endpoint = 'https://api.openai.com/v1/chat/completions'
        self.api_key = config.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables")
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response_json = self._make_request(self.model_endpoint, self.headers, payload)

            if self.json_response:
                response = json.dumps(json.loads(response_json['choices'][0]['message']['content']))
            else:
                response = response_json['choices'][0]['message']['content']

            return response
        except requests.RequestException as e:
            logger.error(f"Error in invoking OpenAI model: {str(e)}")
            return json.dumps({"error": f"Error in invoking model: {str(e)}"})
        except json.JSONDecodeError as e:
            logger.error(f"Error processing OpenAI response: {str(e)}")
            return json.dumps({"error": f"Error processing response: {str(e)}"})