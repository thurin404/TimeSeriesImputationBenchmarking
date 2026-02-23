import sys
import os
from typing import Dict, Any, Type
from pypots.imputation import SAITS

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class SAITSImputer(BasePypotsImputer):
    """SAITS imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("saits")
    
    def get_model_class(self) -> Type:
        """Return the SAITS model class"""
        return SAITS
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return SAITS-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "d_k": params["d_k"],
            "d_v": params["d_v"],
            "d_ffn": params["d_ffn"],
            "dropout": params["dropout"],
            "attn_dropout": params["attn_dropout"],
            "diagonal_attention_mask": params.get("diagonal_attention_mask", True),
            "ORT_weight": params.get("ORT_weight", 1),
            "MIT_weight": params.get("MIT_weight", 1),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params.get("patience", 20),
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define SAITS hyperparameter search space for Optuna optimization
        
        Note: SAITS has a constraint that d_model = n_heads * d_k
        We ensure this constraint is satisfied by sampling n_heads and d_k first,
        then calculating d_model accordingly.
        
        Search space is tuned to avoid GPU OOM on systems with limited memory.
        """
        # Check if low memory mode is enabled (use parameter or environment)
        low_memory = low_memory_mode or self._env_overrides.get("low_memory_mode", False)
        
        if low_memory:
            # Conservative parameters for low memory systems
            n_heads = trial.suggest_categorical("n_heads", [2, 4])
            d_k = trial.suggest_categorical("d_k", [32, 64])
            d_ffn_choices = [64, 128, 256]
            batch_size_choices = [16, 32]
            n_layers_max = 3
        else:
            # Standard parameters
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            d_k = trial.suggest_categorical("d_k", [32, 64, 96])
            d_ffn_choices = [64, 128, 256, 512]
            batch_size_choices = [32, 64, 96, 128, 256]
            n_layers_max = 4
        
        # Calculate d_model to satisfy the constraint d_model = n_heads * d_k
        d_model = n_heads * d_k
        
        # Cap d_model to avoid very large models
        if d_model > 384:
            # If calculated d_model is too large, reduce d_k
            d_k = min(d_k, 384 // n_heads)
            d_model = n_heads * d_k
        
        return {
            "n_layers": trial.suggest_int("n_layers", 2, n_layers_max),
            "d_model": d_model,  # Calculated to satisfy constraint
            "n_heads": n_heads,
            "d_k": d_k,
            "d_v": trial.suggest_categorical("d_v", [32, 64, 96]),
            "d_ffn": trial.suggest_categorical("d_ffn", d_ffn_choices),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3, step=0.1),
            "batch_size": trial.suggest_categorical("batch_size", batch_size_choices),
            "epochs": trial.suggest_int("epochs", 50, 200, step=50),
            "patience": trial.suggest_int("patience", 10, 30, step=5)
        }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use SAITSImputer class instead"""
    imputer = SAITSImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use SAITSImputer class instead"""
    imputer = SAITSImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/saits.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run SAITS imputer
    imputer = SAITSImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()