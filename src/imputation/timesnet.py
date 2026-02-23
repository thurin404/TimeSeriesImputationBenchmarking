import sys
import os
from typing import Dict, Any, Type
from pypots.imputation import TimesNet

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class TimesNetImputer(BasePypotsImputer):
    """TimesNet imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("timesnet")
    
    def get_model_class(self) -> Type:
        """Return the TimesNet model class"""
        return TimesNet
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return TimesNet-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "d_model": params["d_model"],
            "d_ffn": params["d_ffn"],
            "dropout": params["dropout"],
            "top_k": params["top_k"],
            "n_kernels": params["n_kernels"],  # Fixed parameter name
            "apply_nonstationary_norm": params.get("apply_nonstationary_norm", False),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define TimesNet hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 3),
                "d_model": trial.suggest_categorical("d_model", [64, 128]),
                "d_ffn": trial.suggest_categorical("d_ffn", [128, 256]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "top_k": trial.suggest_int("top_k", 3, 5),
                "n_kernels": trial.suggest_int("n_kernels", 4, 6),
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 50, 150, step=50),
                "patience": trial.suggest_int("patience", 10, 20, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
                "d_ffn": trial.suggest_categorical("d_ffn", [128, 256, 512, 1024]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "top_k": trial.suggest_int("top_k", 3, 7),
                "n_kernels": trial.suggest_int("n_kernels", 4, 8),  # Fixed parameter name
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use TimesNetImputer class instead"""
    imputer = TimesNetImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use TimesNetImputer class instead"""
    imputer = TimesNetImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/timesnet.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run TimesNet imputer
    imputer = TimesNetImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()