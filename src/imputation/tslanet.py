import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import TSLANet
except ImportError:
    print("TSLANet not available in current PyPOTS version")
    TSLANet = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class TSLANetImputer(BasePypotsImputer):
    """TSLANet (Time Series Lightweight Attention Network) imputation implementation"""
    
    def __init__(self):
        super().__init__("tslanet")
        # TSLANet is lightweight, doesn't need special memory handling
        self._env_overrides["low_memory_mode"] = False
    
    def get_model_class(self) -> Type:
        """Return the TSLANet model class"""
        if TSLANet is None:
            raise ImportError("TSLANet not available in current PyPOTS version")
        return TSLANet
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return TSLANet-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "patch_size": params["patch_size"],
            "d_embedding": params["d_embedding"],
            "mask_ratio": params.get("mask_ratio", 0.4),
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define TSLANet hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "patch_size": trial.suggest_categorical("patch_size", [8, 16]),
                "d_embedding": trial.suggest_categorical("d_embedding", [64, 128]),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.3, 0.5, step=0.1),
                "dropout": trial.suggest_float("dropout", 0.1, 0.3, step=0.1),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 25, 50, step=25),
                "patience": trial.suggest_int("patience", 5, 10, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 1, 5),
                "patch_size": trial.suggest_categorical("patch_size", [8, 16, 32, 64]),
                "d_embedding": trial.suggest_categorical("d_embedding", [28, 256, 512, 1024]),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.2, 0.7),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epochs": trial.suggest_int("epochs", 400, 500, step=100),
                "patience": trial.suggest_int("patience", 20, 30, step=10)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use TSLANetImputer class instead"""
    imputer = TSLANetImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use TSLANetImputer class instead"""
    imputer = TSLANetImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/tslanet.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run TSLANet imputer
    imputer = TSLANetImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()
