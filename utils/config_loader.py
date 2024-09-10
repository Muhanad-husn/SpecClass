import yaml
import os
from dotenv import load_dotenv

def load_config():
    # Load environment variables from .env file
    load_dotenv()

    # Load YAML config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add API keys from environment variables to config
    config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    config['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    
    return config

config = load_config()