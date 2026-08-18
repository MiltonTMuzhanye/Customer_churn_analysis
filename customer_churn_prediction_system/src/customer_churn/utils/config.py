import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration, loading if not already loaded."""
        if config_name not in self.configs:
            return self.load_config(config_name)
        return self.configs[config_name]
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if config_name not in self.configs:
            self.load_config(config_name)
        
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.configs[config_name], updates)
    
    def save_config(self, config_name: str, path: str = None):
        """Save configuration to file."""
        if config_name not in self.configs:
            raise ValueError(f"Configuration {config_name} not loaded")
        
        save_path = Path(path) if path else self.config_dir / f"{config_name}.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(self.configs[config_name], f, default_flow_style=False)

# Global config loader
config_loader = ConfigLoader()