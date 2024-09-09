import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for OpenAI API key in environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        config['openai_api_key'] = openai_api_key
    
    return config

config = load_config()