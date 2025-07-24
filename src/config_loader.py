import os
import yaml

def load_config():
    """
    Load configuration values from config.yaml
    """
    # Get absolute path to the root directory (ReviewGuard/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ config.yaml not found at {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    
    return config