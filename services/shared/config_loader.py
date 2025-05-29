import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Shared configuration loader for all AI services"""
    
    def __init__(self, service_name: str, config_dir: str = "/app/config"):
        self.service_name = service_name
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "model_config.json"
        self._config_cache = None
        self._last_modified = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with caching and auto-reload"""
        try:
            # Check if file exists
            if not self.config_file.exists():
                logger.warning(f"Config file not found: {self.config_file}")
                return self._get_default_config()
            
            # Check if file was modified (for hot reloading)
            current_modified = self.config_file.stat().st_mtime
            if (self._config_cache is None or 
                self._last_modified is None or 
                current_modified > self._last_modified):
                
                logger.info(f"Loading config from {self.config_file}")
                with open(self.config_file, 'r') as f:
                    self._config_cache = json.load(f)
                self._last_modified = current_modified
                
                # Validate config
                self._validate_config(self._config_cache)
                
            return self._config_cache
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        config = self.load_config()
        return config.get("models", {}).get(model_name)
    
    def get_feature_flags(self) -> Dict[str, Any]:
        """Get all feature flags"""
        config = self.load_config()
        return config.get("features", {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific feature is enabled"""
        features = self.get_feature_flags()
        feature = features.get(feature_name, {})
        return feature.get("enabled", False)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        config = self.load_config()
        return config.get("performance", {})
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service metadata"""
        config = self.load_config()
        return {
            "service": config.get("service", self.service_name),
            "version": config.get("version", "unknown"),
            "config_loaded": True,
            "config_file": str(self.config_file)
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Basic config validation"""
        required_fields = ["service", "version", "models"]
        for field in required_fields:
            if field not in config:
                logger.warning(f"Missing required config field: {field}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration when file loading fails"""
        return {
            "service": self.service_name,
            "version": "unknown",
            "models": {},
            "features": {},
            "performance": {
                "confidence_threshold": 0.3,
                "max_concurrent_requests": 5
            },
            "config_error": True
        }

# Environment variable overrides
def get_env_override(key: str, default: Any = None) -> Any:
    """Get configuration value from environment with fallback"""
    env_value = os.getenv(key)
    if env_value is None:
        return default
    
    # Try to parse as JSON for complex values
    try:
        return json.loads(env_value)
    except (json.JSONDecodeError, ValueError):
        # Return as string if not valid JSON
        return env_value

def merge_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge environment variable overrides into config"""
    # Model overrides
    for model_name in config.get("models", {}):
        enabled_key = f"{model_name.upper()}_ENABLED"
        if os.getenv(enabled_key):
            config["models"][model_name]["enabled"] = get_env_override(enabled_key, True)
    
    # Feature flag overrides  
    for feature_name in config.get("features", {}):
        enabled_key = f"FEATURE_{feature_name.upper()}_ENABLED"
        if os.getenv(enabled_key):
            config["features"][feature_name]["enabled"] = get_env_override(enabled_key, False)
    
    # Performance overrides
    perf_config = config.get("performance", {})
    perf_config["confidence_threshold"] = get_env_override("CONFIDENCE_THRESHOLD", perf_config.get("confidence_threshold", 0.3))
    perf_config["max_concurrent_requests"] = get_env_override("MAX_CONCURRENT_REQUESTS", perf_config.get("max_concurrent_requests", 5))
    
    return config