"""
Configuration management utilities for PyPOTS imputation models.

This module provides utilities for managing model configurations, including
loading, saving, and updating parameters with optimization results.
"""

import os
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


class ModelConfigManager:
    """Manages model configuration files and optimization results"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return self.config.get("params", {})
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Get parameter grid for optimization"""
        return self.config.get("params_grid", {})
    
    def is_optimized(self, partition_by: Optional[str] = None) -> bool:
        """Check if model has been optimized"""
        if partition_by:
            return self.config.get("optimized", {}).get(partition_by.lower(), False)
        return self.config.get("optimized", False)
    
    def is_model_trained(self, partition_by: str) -> bool:
        """Check if model has been trained for given partition"""
        return self.config.get("model-trained", {}).get(partition_by.lower(), False)
    
    def set_model_trained(self, partition_by: str, trained: bool = True):
        """Set model training status"""
        if "model-trained" not in self.config:
            self.config["model-trained"] = {}
        self.config["model-trained"][partition_by.lower()] = trained
    
    def update_with_optimization(self, best_params: Dict[str, Any], 
                               partition_by: Optional[str] = None,
                               optimization_info: Optional[Dict[str, Any]] = None):
        """
        Update configuration with optimization results
        
        Args:
            best_params: Best parameters found during optimization
            partition_by: Partition method used (if applicable)
            optimization_info: Additional optimization metadata
        """
        if not best_params:
            return
        
        # Update parameters
        for param, value in best_params.items():
            if param in self.config["params"]:
                old_value = self.config["params"][param]
                self.config["params"][param] = value
                print(f"Updated {param}: {old_value} -> {value}")
        
        # Mark as optimized
        if partition_by:
            if "optimized" not in self.config:
                self.config["optimized"] = {}
            self.config["optimized"][partition_by.lower()] = True
        else:
            self.config["optimized"] = True
        
        # Add optimization metadata
        if optimization_info:
            if "optimization_history" not in self.config:
                self.config["optimization_history"] = []
            
            opt_record = {
                "timestamp": datetime.now().isoformat(),
                "best_params": best_params.copy(),
                "partition_by": partition_by,
                **optimization_info
            }
            self.config["optimization_history"].append(opt_record)
        
        # Save updated configuration
        self.save_config()
        print(f"Updated configuration saved to: {self.config_path}")
    
    def get_optimization_history(self) -> list:
        """Get optimization history"""
        return self.config.get("optimization_history", [])
    
    def reset_optimization(self, partition_by: Optional[str] = None):
        """Reset optimization status"""
        if partition_by:
            if "optimized" in self.config and isinstance(self.config["optimized"], dict):
                self.config["optimized"][partition_by.lower()] = False
        else:
            self.config["optimized"] = False
        
        self.save_config()
    
    def backup_config(self, suffix: Optional[str] = None):
        """Create a backup of the current configuration"""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = f"{self.config_path}.backup_{suffix}"
        with open(backup_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Configuration backed up to: {backup_path}")
        return backup_path


def load_model_config(model_type: str, base_dir: str = "params") -> ModelConfigManager:
    """
    Load model configuration manager for given model type
    
    Args:
        model_type: Type of model (e.g., 'saits', 'brits')
        base_dir: Base directory containing configuration files
    
    Returns:
        ModelConfigManager instance
    """
    config_path = os.path.join(base_dir, f"{model_type}.yaml")
    return ModelConfigManager(config_path)


def create_default_config(model_type: str, base_dir: str = "params") -> str:
    """
    Create a default configuration file for a model type
    
    Args:
        model_type: Type of model
        base_dir: Base directory for configuration files
    
    Returns:
        Path to created configuration file
    """
    config_path = os.path.join(base_dir, f"{model_type}.yaml")
    
    # Default configuration template
    default_config = {
        "model-trained": {
            "feature": False,
            "station": False
        },
        "optimized": False,
        "params": {},
        "params_grid": {},
        "optimization_history": []
    }
    
    os.makedirs(base_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default configuration created: {config_path}")
    return config_path