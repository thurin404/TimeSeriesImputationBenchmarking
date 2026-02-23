import sys
import os
from typing import Dict, Any, Type
from pypots.imputation import ImputeFormer


# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class ImputeFormerImputer(BasePypotsImputer):
    """ImputeFormer imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("imputeformer")
    
    def get_model_class(self) -> Type:
        """Return the ImputeFormer model class"""
        return ImputeFormer
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return ImputeFormer-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "d_input_embed": params["d_input_embed"],
            "d_learnable_embed": params["d_learnable_embed"],
            "d_proj": params["d_proj"],
            "d_ffn": params["d_ffn"],
            "n_temporal_heads": params["n_temporal_heads"],
            "dropout": params["dropout"],
            "input_dim": params.get("input_dim", 1),
            "output_dim": params.get("output_dim", 1),
            "ORT_weight": params.get("ORT_weight", 1.0),
            "MIT_weight": params.get("MIT_weight", 1.0),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define ImputeFormer hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 3),
                "d_input_embed": trial.suggest_categorical("d_input_embed", [32, 64]),
                "d_learnable_embed": trial.suggest_categorical("d_learnable_embed", [64, 128]),
                "d_proj": trial.suggest_categorical("d_proj", [32, 64]),
                "d_ffn": trial.suggest_categorical("d_ffn", [64, 128]),
                "n_temporal_heads": trial.suggest_categorical("n_temporal_heads", [2, 4]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "input_dim": 1,
                "output_dim": 1,
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 2.0),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 2.0),
                "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16]),
                "epochs": trial.suggest_int("epochs", 50, 100, step=25),
                "patience": trial.suggest_int("patience", 10, 20, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 8),
                "d_input_embed": trial.suggest_categorical("d_input_embed", [128, 256, 512]),
                "d_learnable_embed": trial.suggest_categorical("d_learnable_embed", [256, 512, 1024]),
                "d_proj": trial.suggest_categorical("d_proj", [64, 128, 256]),
                "d_ffn": trial.suggest_categorical("d_ffn", [128, 256, 512]),
                "n_temporal_heads": trial.suggest_categorical("n_temporal_heads", [2, 4, 6, 8]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
                "input_dim": 1,
                "output_dim": 1,
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 2.0),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 2.0),
                "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16]),
                "epochs": trial.suggest_int("epochs", 400, 500, step=100),
                "patience": trial.suggest_int("patience", 20, 30, step=10)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use ImputeFormerImputer class instead"""
    imputer = ImputeFormerImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use ImputeFormerImputer class instead"""
    imputer = ImputeFormerImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/imputeformer.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run ImputeFormer imputer
    imputer = ImputeFormerImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()