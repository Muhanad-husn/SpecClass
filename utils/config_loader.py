import yaml
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Load YAML config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Add API keys from environment variables to config
        self._config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        self._config['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def get(self, key, default=None):
        return self._config.get(key, default)

    def __str__(self):
        return str({k: v for k, v in self._config.items() if not k.endswith('API_KEY')})

    def __repr__(self):
        return self.__str__()

# Create a global config instance
config = Config()