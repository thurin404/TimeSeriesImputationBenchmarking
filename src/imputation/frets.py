import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import FreTS
except ImportError:
    print("FreTS not available in current PyPOTS version")
    FreTS = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class FreTSImputer(BasePypotsImputer):
    """FreTS (Frequency-enhanced Time Series) imputation implementation"""
    
    def __init__(self):
        super().__init__("frets")
        # FreTS is moderately efficient
        self._env_overrides["low_memory_mode"] = False
    
    def get_model_class(self) -> Type:
        """Return the FreTS model class"""
        if FreTS is None:
            raise ImportError("FreTS not available in current PyPOTS version")
        return FreTS
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return FreTS-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "embed_size": params["embed_size"],
            "hidden_size": params["hidden_size"],
            "channel_independence": params.get("channel_independence", False),
            "ORT_weight": params.get("ORT_weight", 1.0),
            "MIT_weight": params.get("MIT_weight", 1.0),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define FreTS hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "embed_size": trial.suggest_categorical("embed_size", [64, 128]),
                "hidden_size": trial.suggest_categorical("hidden_size", [128, 256]),
                "channel_independence": trial.suggest_categorical("channel_independence", [False, True]),
                "ORT_weight": 1.0,
                "MIT_weight": 1.0,
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 25, 50, step=25),
                "patience": trial.suggest_int("patience", 5, 10, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "embed_size": trial.suggest_categorical("embed_size", [128, 256, 512]),
                "hidden_size": trial.suggest_categorical("hidden_size", [256, 512, 1024, 2048]),
                "channel_independence": trial.suggest_categorical("channel_independence", [False, True]),
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 2.0),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 2.0),
                "batch_size": trial.suggest_categorical("batch_size", [64, 96]),
                "epochs": trial.suggest_int("epochs", 400, 500, step=100),
                "patience": trial.suggest_int("patience", 20, 30, step=10)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use FreTSImputer class instead"""
    imputer = FreTSImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use FreTSImputer class instead"""
    imputer = FreTSImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/frets.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run FreTS imputer
    imputer = FreTSImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()
